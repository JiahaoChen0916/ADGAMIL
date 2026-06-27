# models/MHIM/network.py
import numpy as np
import torch
from math import ceil
from copy import deepcopy
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def _get_dropout_p(dropout):
    if isinstance(dropout, bool):
        return 0.25 if dropout else 0.0
    if dropout is None:
        return 0.0
    return float(dropout)


class AttentionGated(nn.Module):
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(AttentionGated, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention_a = [nn.Linear(self.L, self.D, bias=bias)]
        if act == 'gelu':
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]

        self.attention_b = [nn.Linear(self.L, self.D, bias=bias), nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

    def forward(self, x, no_norm=False, attn_mask=None):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # [B, 1, N]
        A_ori = A.clone()

        if attn_mask is not None:
            mask = attn_mask.bool().unsqueeze(1)
            A = A.masked_fill(~mask, float("-inf"))

        A = F.softmax(A, dim=-1)
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        else:
            return x, A


class DAttention(nn.Module):
    def __init__(self, input_dim=512, act='relu', gated=True, bias=False, dropout=False):
        super(DAttention, self).__init__()
        self.gated = gated
        self.attention = AttentionGated(input_dim, act, bias, dropout)

    def masking(self, x, ids_shuffle=None, len_keep=None):
        N, L, D = x.shape
        assert ids_shuffle is not None

        _, ids_restore = ids_shuffle.sort()
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, no_norm=False, mask_enable=False, attn_mask=None):
        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x, mask_ids, len_keep)
            attn_mask = None

        x, attn = self.attention(x, no_norm=no_norm, attn_mask=attn_mask)

        if return_attn:
            return x.squeeze(1), attn.squeeze(1)
        else:
            return x.squeeze(1)


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_landmarks=256,
        pinv_iterations=6,
        residual=True,
        residual_conv_kernel=33,
        eps=1e-8,
        dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        _, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps

        q_landmarks /= divisor
        k_landmarks /= divisor

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = torch.einsum(einops_eq, q, k_landmarks)
        attn2 = torch.einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = torch.einsum(einops_eq, q_landmarks, k)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        if self.residual:
            out += self.res_conv(v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn_ret = attn1[:, :, 0].unsqueeze(-2) @ attn2
            attn_ret = (attn_ret @ attn3)
            return out, attn_ret[:, :, 0, -n + 1:]

        return out


class PPEG(nn.Module):
    def __init__(self, dim=512, k=7, conv_1d=False, bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k, 1), 1, (k // 2, 0), groups=dim, bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5, 1), 1, (5 // 2, 0), groups=dim, bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim, bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3, 1), 1, (3 // 2, 0), groups=dim, bias=bias)

    def forward(self, x):
        B, N, C = x.shape

        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:, :add_length, :]], dim=1)

        if H < 7:
            H, W = 7, 7
            zero_pad = H * W - (N + add_length)
            x = torch.cat([x, torch.zeros((B, zero_pad, C), device=x.device)], dim=1)
            add_length += zero_pad

        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        if add_length > 0:
            x = x[:, :-add_length]
        return x


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, head=8):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=head,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1,
        )

    def forward(self, x, need_attn=False):
        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=need_attn)
            x = x + z
            return x, attn
        else:
            x = x + self.attn(self.norm(x))
            return x


class SAttention(nn.Module):
    def __init__(self, mlp_dim=512, pos_pos=0, pos='ppeg', peg_k=7, head=8):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))
        self.layer1 = TransLayer(dim=mlp_dim, head=head)
        self.layer2 = TransLayer(dim=mlp_dim, head=head)

        if pos == 'ppeg':
            self.pos_embedding = PPEG(dim=mlp_dim, k=peg_k)
        else:
            self.pos_embedding = nn.Identity()

        self.pos_pos = pos_pos

    def masking(self, x, ids_shuffle=None, len_keep=None):
        N, L, D = x.shape
        assert ids_shuffle is not None

        _, ids_restore = ids_shuffle.sort()
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, mask_enable=False, attn_mask=None):
        batch, num_patches, C = x.shape
        attn = []

        if self.pos_pos == -2:
            x = self.pos_embedding(x)

        if mask_enable and mask_ids is not None:
            x, _, _ = self.masking(x, mask_ids, len_keep)

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=batch)
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        if return_attn:
            x, _attn = self.layer1(x, True)
            attn.append(_attn.clone())
        else:
            x = self.layer1(x)

        if self.pos_pos == 0:
            x[:, 1:, :] = self.pos_embedding(x[:, 1:, :])

        if return_attn:
            x, _attn = self.layer2(x, True)
            attn.append(_attn.clone())
        else:
            x = self.layer2(x)

        x = self.norm(x)
        logits = x[:, 0, :]

        if return_attn:
            return logits, attn
        else:
            return logits


@torch.no_grad()
def ema_update(model, targ_model, mm=0.9999):
    assert 0.0 <= mm <= 1.0
    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)


class SoftTargetCrossEntropy_v2(nn.Module):
    def __init__(self, temp_t=1., temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool = True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t, dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        return loss.mean() if mean else loss


class MHIM(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        mlp_dim=512,
        mask_ratio=0,
        n_classes=2,
        temp_t=1.,
        temp_s=1.,
        dropout=0.25,
        act='relu',
        select_mask=True,
        select_inv=False,
        msa_fusion='vote',
        mask_ratio_h=0.,
        mrh_sche=None,
        mask_ratio_hr=0.,
        mask_ratio_l=0.,
        da_act='relu',
        baseline='selfattn',
        head=2,
        attn_layer=0
    ):
        super(MHIM, self).__init__()

        self.mask_ratio = mask_ratio
        self.mask_ratio_h = mask_ratio_h
        self.mask_ratio_hr = mask_ratio_hr
        self.mask_ratio_l = mask_ratio_l
        self.select_mask = select_mask
        self.select_inv = select_inv
        self.msa_fusion = msa_fusion
        self.mrh_sche = mrh_sche
        self.attn_layer = attn_layer

        drop_p = _get_dropout_p(dropout)

        patch_to_emb = [nn.Linear(input_dim, mlp_dim)]
        if act.lower() == 'relu':
            patch_to_emb += [nn.ReLU()]
        elif act.lower() == 'gelu':
            patch_to_emb += [nn.GELU()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")

        self.patch_to_emb = nn.Sequential(*patch_to_emb)
        self.dp = nn.Dropout(drop_p) if drop_p > 0. else nn.Identity()

        if baseline == 'selfattn':
            self.online_encoder = SAttention(mlp_dim=mlp_dim, head=head)
        elif baseline == 'attn':
            self.online_encoder = DAttention(mlp_dim, da_act)
        else:
            raise NotImplementedError(f"Unsupported baseline: {baseline}")

        self.predictor = nn.Linear(mlp_dim, n_classes)

        self.temp_t = temp_t
        self.temp_s = temp_s
        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t, self.temp_s)

        self.predictor_cl = nn.Identity()
        self.target_predictor = nn.Identity()

        self.apply(initialize_weights)

    def select_mask_fn(self, ps, attn, largest, mask_ratio, mask_ids_other=None, len_keep_other=None,
                       cls_attn_topk_idx_other=None, random_ratio=1., select_inv=False):
        ps_tmp = ps
        mask_ratio_ori = mask_ratio
        mask_ratio = mask_ratio / (random_ratio + 1e-8)
        if mask_ratio > 1:
            random_ratio = mask_ratio_ori
            mask_ratio = 1.

        if mask_ids_other is not None:
            if cls_attn_topk_idx_other is None:
                cls_attn_topk_idx_other = mask_ids_other[:, len_keep_other:].squeeze()
                ps_tmp = ps - cls_attn_topk_idx_other.size(0)

        if len(attn.size()) > 2:
            if self.msa_fusion == 'mean':
                _, cls_attn_topk_idx = torch.topk(attn, int(np.ceil((ps_tmp * mask_ratio)) // attn.size(1)), largest=largest)
                cls_attn_topk_idx = torch.unique(cls_attn_topk_idx.flatten(-3, -1))
            elif self.msa_fusion == 'vote':
                vote = attn.clone()
                vote[:] = 0
                _, idx = torch.topk(attn, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False, largest=largest)
                mask = vote.clone()
                mask = mask.scatter_(2, idx, 1) == 1
                vote[mask] = 1
                vote = vote.sum(dim=1)
                _, cls_attn_topk_idx = torch.topk(vote, k=int(np.ceil((ps_tmp * mask_ratio))), sorted=False)
                cls_attn_topk_idx = cls_attn_topk_idx[0]
        else:
            k = int(np.ceil((ps_tmp * mask_ratio)))
            _, cls_attn_topk_idx = torch.topk(attn, k, largest=largest)
            cls_attn_topk_idx = cls_attn_topk_idx.squeeze(0)

        if random_ratio < 1.:
            random_idx = torch.randperm(cls_attn_topk_idx.size(0), device=cls_attn_topk_idx.device)
            cls_attn_topk_idx = torch.gather(
                cls_attn_topk_idx,
                dim=0,
                index=random_idx[:int(np.ceil((cls_attn_topk_idx.size(0) * random_ratio)))]
            )

        if mask_ids_other is not None:
            cls_attn_topk_idx = torch.cat([cls_attn_topk_idx, cls_attn_topk_idx_other]).unique()

        len_keep = ps - cls_attn_topk_idx.size(0)
        a = set(cls_attn_topk_idx.tolist())
        b = set(list(range(ps)))
        mask_ids = torch.tensor(list(b.difference(a)), device=attn.device)

        if select_inv:
            mask_ids = torch.cat([cls_attn_topk_idx, mask_ids]).unsqueeze(0)
            len_keep = ps - len_keep
        else:
            mask_ids = torch.cat([mask_ids, cls_attn_topk_idx]).unsqueeze(0)

        return len_keep, mask_ids

    def get_mask(self, ps, i, attn, mrh=None):
        if attn is not None and isinstance(attn, (list, tuple)):
            attn = attn[1] if self.attn_layer == -1 else attn[self.attn_layer]

        if attn is not None and self.mask_ratio > 0.:
            len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio, select_inv=self.select_inv, random_ratio=0.001)
        else:
            len_keep, mask_ids = ps, None

        if attn is not None and self.mask_ratio_l > 0.:
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv)
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:, len_keep:].squeeze()
                len_keep, mask_ids = self.select_mask_fn(
                    ps, attn, False, self.mask_ratio_l, select_inv=self.select_inv,
                    mask_ids_other=mask_ids, len_keep_other=ps,
                    cls_attn_topk_idx_other=cls_attn_topk_idx_other
                )

        mask_ratio_h = self.mask_ratio_h
        if self.mrh_sche is not None:
            mask_ratio_h = self.mrh_sche[i]
        if mrh is not None:
            mask_ratio_h = mrh

        if mask_ratio_h > 0.:
            if mask_ids is None:
                len_keep, mask_ids = self.select_mask_fn(
                    ps, attn, largest=True, mask_ratio=mask_ratio_h,
                    len_keep_other=ps, random_ratio=self.mask_ratio_hr,
                    select_inv=self.select_inv
                )
            else:
                cls_attn_topk_idx_other = mask_ids[:, :len_keep].squeeze() if self.select_inv else mask_ids[:, len_keep:].squeeze()
                len_keep, mask_ids = self.select_mask_fn(
                    ps, attn, largest=True, mask_ratio=mask_ratio_h,
                    mask_ids_other=mask_ids, len_keep_other=ps,
                    cls_attn_topk_idx_other=cls_attn_topk_idx_other,
                    random_ratio=self.mask_ratio_hr, select_inv=self.select_inv
                )

        return len_keep, mask_ids

    @torch.no_grad()
    def forward_teacher(self, x, return_attn=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        if return_attn:
            x, attn = self.online_encoder(x, return_attn=True)
        else:
            x = self.online_encoder(x)
            attn = None

        return x, attn

    @torch.no_grad()
    def forward_test(self, x, return_attn=False, no_norm=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)

        if return_attn:
            feat, a = self.online_encoder(x, return_attn=True, no_norm=no_norm)
        else:
            feat = self.online_encoder(x)
            a = None

        logits = self.predictor(feat)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return logits, Y_prob, Y_hat, a, {"features": feat}

    def pure(self, x, return_attn=False):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if return_attn:
            feat, attn = self.online_encoder(x, return_attn=True)
        else:
            feat = self.online_encoder(x)
            attn = None

        logits = self.predictor(feat)

        if self.training:
            if return_attn:
                return logits, 0, ps, ps, attn
            else:
                return logits, 0, ps, ps
        else:
            if return_attn:
                return logits, attn
            else:
                return logits

    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            cls_loss = self.cl_loss(student_cls_feat, teacher_cls_feat.detach())
        else:
            cls_loss = 0.
        return cls_loss

    def forward(self, x, attn=None, teacher_cls_feat=None, i=None):
        x = self.patch_to_emb(x)
        x = self.dp(x)
        ps = x.size(1)

        if self.select_mask:
            len_keep, mask_ids = self.get_mask(ps, i, attn)
        else:
            len_keep, mask_ids = ps, None

        student_cls_feat = self.online_encoder(x, len_keep=len_keep, mask_ids=mask_ids, mask_enable=True)
        student_logit = self.predictor(student_cls_feat)
        cls_loss = self.forward_loss(student_cls_feat=student_cls_feat, teacher_cls_feat=teacher_cls_feat)

        return student_logit, cls_loss, ps, len_keep


class PURE_MHIM(nn.Module):
    def __init__(self, input_dim=1024, mlp_dim=512, n_classes=2, baseline='attn'):
        super(PURE_MHIM, self).__init__()
        self.model = MHIM(select_mask=False, input_dim=input_dim, mlp_dim=mlp_dim, n_classes=n_classes, baseline=baseline)

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        logits = self.model.pure(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        return logits, Y_prob, Y_hat, None, {"features": None}


class MHIM_MIL(nn.Module):
    """
    Subtype version of MHIM_MIL.

    Unified forward output in eval mode:
        logits, Y_prob, Y_hat, A_raw, results_dict

    Training mode:
        if called with return_loss=True -> returns unified output + aux loss in results_dict
        otherwise still returns unified output
    """
    def __init__(self, num_epoch, niter_per_ep, teacher_init=None, mm=0.9999, mm_sche=True, **kwargs):
        super(MHIM_MIL, self).__init__()
        self.mm = mm

        mrh_sche = cosine_scheduler(kwargs['mask_ratio_h'], 0., epochs=num_epoch, niter_per_ep=niter_per_ep)
        self.mm_sche = cosine_scheduler(self.mm, 1., epochs=num_epoch, niter_per_ep=niter_per_ep, start_warmup_value=1.) if mm_sche else None

        kwargs['mrh_sche'] = mrh_sche
        self.student = MHIM(**kwargs)
        self.teacher = deepcopy(self.student)

        if teacher_init is not None:
            print('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(teacher_init, map_location='cpu')
                info = self.teacher.load_state_dict(pre_dict, strict=False)
                print(info)
            except Exception:
                print('########## Init Error')

            print('######### Model Initializing.....')
            pre_dict = torch.load(teacher_init, map_location='cpu')
            new_state_dict = {}
            for _k, v in pre_dict.items():
                _k = _k.replace('patch_to_emb.', '') if 'patch_to_emb' in _k else _k
                new_state_dict[_k] = v
            info = self.student.patch_to_emb.load_state_dict(new_state_dict, strict=False)
            print(info)

    def update_teacher(self, index):
        mm = self.mm_sche[index] if self.mm_sche is not None else self.mm
        ema_update(self.student, self.teacher, mm)

    def train_forward(self, x, i):
        cls_tea, attn = self.teacher.forward_teacher(x, return_attn=True)
        train_logits, cls_loss, patch_num, keep_num = self.student(x, attn, cls_tea, i)

        Y_prob = F.softmax(train_logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
        results_dict = {
            "aux_loss": cls_loss,
            "patch_num": patch_num,
            "keep_num": keep_num,
            "features": None
        }
        return train_logits, Y_prob, Y_hat, None, results_dict

    def forward(self, x, label=None, instance_eval=False, attn_mask=None, i=None):
        if self.training:
            return self.train_forward(x, i)
        else:
            return self.student.forward_test(x)
