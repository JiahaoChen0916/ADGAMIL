import math
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from timm.models.layers import DropPath


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def _get_dropout_p(dropout):
    if isinstance(dropout, bool):
        return 0.25 if dropout else 0.0
    if dropout is None:
        return 0.0
    return float(dropout)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, input_dim=512, act="relu", bias=False, dropout=False):
        super().__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        layers = [nn.Linear(self.L, self.D, bias=bias)]

        if act == "gelu":
            layers += [nn.GELU()]
        elif act == "relu":
            layers += [nn.ReLU()]
        elif act == "tanh":
            layers += [nn.Tanh()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")

        if dropout:
            layers += [nn.Dropout(0.25)]

        layers += [nn.Linear(self.D, self.K, bias=bias)]
        self.attention = nn.Sequential(*layers)

    def forward(self, x, no_norm=False, attn_mask=None):
        A = self.attention(x)           # [B, N, 1]
        A = torch.transpose(A, -1, -2)  # [B, 1, N]
        A_ori = A.clone()

        if attn_mask is not None:
            mask = attn_mask.bool().unsqueeze(1)  # [B, 1, N]
            A = A.masked_fill(~mask, float("-inf"))

        A = F.softmax(A, dim=-1)
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        return x, A


class AttentionGated(nn.Module):
    def __init__(self, input_dim=512, act="relu", bias=False, dropout=False):
        super().__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        attention_a = [nn.Linear(self.L, self.D, bias=bias)]
        if act == "gelu":
            attention_a += [nn.GELU()]
        elif act == "relu":
            attention_a += [nn.ReLU()]
        elif act == "tanh":
            attention_a += [nn.Tanh()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")

        attention_b = [
            nn.Linear(self.L, self.D, bias=bias),
            nn.Sigmoid(),
        ]

        if dropout:
            attention_a += [nn.Dropout(0.25)]
            attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*attention_a)
        self.attention_b = nn.Sequential(*attention_b)
        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

    def forward(self, x, no_norm=False, attn_mask=None):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)         # [B, N, 1]
        A = torch.transpose(A, -1, -2)  # [B, 1, N]
        A_ori = A.clone()

        if attn_mask is not None:
            mask = attn_mask.bool().unsqueeze(1)
            A = A.masked_fill(~mask, float("-inf"))

        A = F.softmax(A, dim=-1)
        x = torch.matmul(A, x)

        if no_norm:
            return x, A_ori
        return x, A


class DAttention(nn.Module):
    def __init__(self, input_dim=512, act="relu", gated=False, bias=False, dropout=False):
        super().__init__()
        self.gated = gated
        if gated:
            self.attention = AttentionGated(input_dim, act, bias, dropout)
        else:
            self.attention = Attention(input_dim, act, bias, dropout)

    def forward(self, x, return_attn=False, no_norm=False, attn_mask=None, **kwargs):
        x, attn = self.attention(x, no_norm=no_norm, attn_mask=attn_mask)
        if return_attn:
            return x.squeeze(1), attn.squeeze(1)
        return x.squeeze(1)


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
        dropout=0.0,
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
            nn.Dropout(dropout),
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(
                heads,
                heads,
                (kernel_size, 1),
                padding=(padding, 0),
                groups=heads,
                bias=False,
            )

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = (
            *x.shape,
            self.heads,
            self.num_landmarks,
            self.pinv_iterations,
            self.eps,
        )
        orig_n = n

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)
            n = x.shape[1]

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        if exists(mask):
            mask = rearrange(mask, "b n -> b () n")
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        l = math.ceil(n / m)
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=l)
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=l)

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, "... (n l) -> ... n", "sum", l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        q_landmarks /= divisor
        k_landmarks /= divisor

        einops_eq = "... i d, ... j d -> ... i j"
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            attn1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            attn3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        if self.residual:
            out = out + self.res_conv(v)

        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -orig_n:]

        if return_attn:
            attn_ret = attn1[:, :, 0].unsqueeze(-2) @ attn2
            attn_ret = attn_ret @ attn3
            return out, attn_ret[:, :, 0, -orig_n + 1:]

        return out


def region_partition(x, region_size):
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions


def region_reverse(regions, region_size, H, W):
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class InnerAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        epeg=True,
        epeg_k=15,
        epeg_2d=False,
        epeg_bias=True,
        epeg_type="attn",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.epeg_2d = epeg_2d
        self.epeg_type = epeg_type
        if epeg:
            padding = epeg_k // 2
            if epeg_2d:
                if epeg_type == "attn":
                    self.pe = nn.Conv2d(
                        num_heads, num_heads, epeg_k,
                        padding=padding, groups=num_heads, bias=epeg_bias
                    )
                else:
                    self.pe = nn.Conv2d(
                        head_dim * num_heads, head_dim * num_heads, epeg_k,
                        padding=padding, groups=head_dim * num_heads, bias=epeg_bias
                    )
            else:
                if epeg_type == "attn":
                    self.pe = nn.Conv2d(
                        num_heads, num_heads, (epeg_k, 1),
                        padding=(padding, 0), groups=num_heads, bias=epeg_bias
                    )
                else:
                    self.pe = nn.Conv2d(
                        head_dim * num_heads, head_dim * num_heads, (epeg_k, 1),
                        padding=(padding, 0), groups=head_dim * num_heads, bias=epeg_bias
                    )
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if self.pe is not None and self.epeg_type == "attn":
            pe = self.pe(attn)
            attn = attn + pe

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        if self.pe is not None and self.epeg_type == "value_bf":
            side = int(np.ceil(np.sqrt(N)))
            pe = self.pe(
                v.permute(0, 3, 1, 2).reshape(B_, C, side, side)
            )
            v = v + pe.reshape(B_, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads * self.head_dim)

        if self.pe is not None and self.epeg_type == "value_af":
            side = int(np.ceil(np.sqrt(N)))
            pe = self.pe(
                v.permute(0, 3, 1, 2).reshape(B_, C, side, side)
            )
            x = x + pe.reshape(B_, self.num_heads * self.head_dim, N).transpose(-1, -2)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RegionAttntion(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        region_size=0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        region_num=8,
        epeg=False,
        min_region_num=0,
        min_region_ratio=0.0,
        region_attn="native",
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        if region_attn == "native":
            self.attn = InnerAttention(
                dim,
                head_dim=head_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                epeg=epeg,
                **kwargs,
            )
        elif region_attn == "ntrans":
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop,
            )
        else:
            raise NotImplementedError

    def padding(self, x):
        B, L, C = x.shape

        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        if add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
            region_num = 1

        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device, dtype=x.dtype)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)
        x_regions = region_partition(x, region_size)
        x_regions = x_regions.view(-1, region_size * region_size, C)

        attn_regions = self.attn(x_regions)
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)
        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class CrossRegionAttntion(nn.Module):
    def __init__(
        self,
        dim,
        head_dim=None,
        num_heads=8,
        region_size=0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        region_num=8,
        epeg=False,
        min_region_num=0,
        min_region_ratio=0.0,
        crmsa_k=3,
        crmsa_mlp=False,
        region_attn="native",
        **kwargs,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio

        self.attn = InnerAttention(
            dim,
            head_dim=head_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            epeg=epeg,
            **kwargs,
        )

        self.crmsa_mlp = crmsa_mlp
        if crmsa_mlp:
            self.phi = nn.Sequential(
                nn.Linear(self.dim, self.dim // 4, bias=False),
                nn.Tanh(),
                nn.Linear(self.dim // 4, crmsa_k, bias=False),
            )
        else:
            self.phi = nn.Parameter(torch.empty((self.dim, crmsa_k)))
            nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))

    def padding(self, x):
        B, L, C = x.shape

        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_size
            H, W = H + _n, W + _n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H + _n, W + _n
            region_size = int(H // self.region_num)
            region_num = self.region_num

        add_length = H * W - L

        if add_length > L / (self.min_region_ratio + 1e-8) or L < self.min_region_num:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H + _n, W + _n
            add_length = H * W - L
            region_size = H
            region_num = 1

        if add_length > 0:
            x = torch.cat([x, torch.zeros((B, add_length, C), device=x.device, dtype=x.dtype)], dim=1)

        return x, H, W, add_length, region_num, region_size

    def forward(self, x, return_attn=False):
        B, L, C = x.shape
        x, H, W, add_length, region_num, region_size = self.padding(x)

        x = x.view(B, H, W, C)
        x_regions = region_partition(x, region_size)
        x_regions = x_regions.view(-1, region_size * region_size, C)

        if self.crmsa_mlp:
            logits = self.phi(x_regions).transpose(1, 2)
        else:
            logits = torch.einsum("w p c, c n -> w p n", x_regions, self.phi).transpose(1, 2)

        combine_weights = logits.softmax(dim=-1)
        dispatch_weights = logits.softmax(dim=1)

        logits_min, _ = logits.min(dim=-1)
        logits_max, _ = logits.max(dim=-1)
        dispatch_weights_mm = (logits - logits_min.unsqueeze(-1)) / (
            logits_max.unsqueeze(-1) - logits_min.unsqueeze(-1) + 1e-8
        )

        attn_regions = torch.einsum("w p c, w n p -> w n p c", x_regions, combine_weights).sum(dim=-2).transpose(0, 1)
        attn_regions = self.attn(attn_regions).transpose(0, 1)

        attn_regions = torch.einsum("w n c, w n p -> w n p c", attn_regions, dispatch_weights_mm)
        attn_regions = torch.einsum("w n p c, w n p -> w n p c", attn_regions, dispatch_weights).sum(dim=1)

        attn_regions = attn_regions.view(-1, region_size, region_size, C)
        x = region_reverse(attn_regions, region_size, H, W)
        x = x.view(B, H * W, C)

        if add_length > 0:
            x = x[:, :-add_length]

        return x


class TransLayer(nn.Module):
    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        head=8,
        drop_out=0.1,
        drop_path=0.0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        trans_dim=64,
        attn="rmsa",
        n_region=8,
        epeg=False,
        region_size=0,
        min_region_num=0,
        min_region_ratio=0.0,
        qkv_bias=True,
        crmsa_k=3,
        epeg_k=15,
        **kwargs,
    ):
        super().__init__()

        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()

        if attn == "ntrans":
            self.attn = NystromAttention(
                dim=dim,
                dim_head=trans_dim,
                heads=head,
                num_landmarks=256,
                pinv_iterations=6,
                residual=True,
                dropout=drop_out,
            )
        elif attn == "rmsa":
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                epeg_k=epeg_k,
                **kwargs,
            )
        elif attn == "crmsa":
            self.attn = CrossRegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=dim // head,
                epeg=epeg,
                region_size=region_size,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                **kwargs,
            )
        else:
            raise NotImplementedError

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == "gelu" else nn.ReLU
        self.mlp = (
            Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_out)
            if ffn else nn.Identity()
        )

    def forward(self, x, need_attn=False):
        x, attn = self.forward_trans(x, need_attn=need_attn)
        if need_attn:
            return x, attn
        return x

    def forward_trans(self, x, need_attn=False):
        attn = None

        if need_attn:
            z, attn = self.attn(self.norm(x), return_attn=True)
        else:
            z = self.attn(self.norm(x))

        x = x + self.drop_path(z)

        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn


class RRTEncoder(nn.Module):
    def __init__(
        self,
        mlp_dim=512,
        pos_pos=0,
        pos="none",
        peg_k=7,
        attn="rmsa",
        region_num=8,
        drop_out=0.1,
        n_layers=2,
        n_heads=8,
        drop_path=0.0,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        trans_dim=64,
        epeg=True,
        epeg_k=15,
        region_size=0,
        min_region_num=0,
        min_region_ratio=0.0,
        qkv_bias=True,
        peg_bias=True,
        peg_1d=False,
        cr_msa=True,
        crmsa_k=3,
        all_shortcut=False,
        crmsa_mlp=False,
        crmsa_heads=8,
        need_init=False,
        **kwargs,
    ):
        super().__init__()

        self.final_dim = mlp_dim
        self.norm = nn.LayerNorm(self.final_dim)
        self.all_shortcut = all_shortcut

        layers = []
        for _ in range(n_layers - 1):
            layers += [
                TransLayer(
                    dim=mlp_dim,
                    head=n_heads,
                    drop_out=drop_out,
                    drop_path=drop_path,
                    ffn=ffn,
                    ffn_act=ffn_act,
                    mlp_ratio=mlp_ratio,
                    trans_dim=trans_dim,
                    attn=attn,
                    n_region=region_num,
                    epeg=epeg,
                    region_size=region_size,
                    min_region_num=min_region_num,
                    min_region_ratio=min_region_ratio,
                    qkv_bias=qkv_bias,
                    epeg_k=epeg_k,
                    **kwargs,
                )
            ]
        self.layers = nn.Sequential(*layers)

        self.cr_msa = (
            TransLayer(
                dim=mlp_dim,
                head=crmsa_heads,
                drop_out=drop_out,
                drop_path=drop_path,
                ffn=ffn,
                ffn_act=ffn_act,
                mlp_ratio=mlp_ratio,
                trans_dim=trans_dim,
                attn="crmsa",
                qkv_bias=qkv_bias,
                crmsa_k=crmsa_k,
                crmsa_mlp=crmsa_mlp,
                **kwargs,
            )
            if cr_msa else nn.Identity()
        )

        self.pos_embedding = nn.Identity()
        self.pos_pos = pos_pos

        if need_init:
            self.apply(initialize_weights)

    def forward(self, x):
        shape_len = 3

        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            shape_len = 2

        if len(x.shape) == 4:
            x = x.reshape(x.size(0), x.size(1), -1)
            x = x.transpose(1, 2)
            shape_len = 4

        batch, num_patches, C = x.shape
        x_shortcut = x

        # 官方逻辑是 -1，不是 -2
        if self.pos_pos == -1:
            x = self.pos_embedding(x)

        for i, layer in enumerate(self.layers.children()):
            if i == 1 and self.pos_pos == 0:
                x = self.pos_embedding(x)
            x = layer(x)

        x = self.cr_msa(x)

        if self.all_shortcut:
            x = x + x_shortcut

        x = self.norm(x)

        if shape_len == 2:
            x = x.squeeze(0)
        elif shape_len == 4:
            x = x.transpose(1, 2)
            x = x.reshape(batch, C, int(num_patches ** 0.5), int(num_patches ** 0.5))

        return x


class RRT(nn.Module):
    """
    Subtype version of RRTMIL.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(
        self,
        input_dim=1024,
        mlp_dim=512,
        act="relu",
        n_classes=2,
        dropout=0.25,
        pos_pos=0,
        pos="none",
        peg_k=7,
        attn="rmsa",
        pool="attn",
        region_num=8,
        n_layers=2,
        n_heads=8,
        drop_path=0.0,
        da_act="tanh",
        trans_dropout=0.1,
        ffn=False,
        ffn_act="gelu",
        mlp_ratio=4.0,
        da_gated=False,
        da_bias=False,
        da_dropout=False,
        trans_dim=64,
        epeg=True,
        min_region_num=0,
        qkv_bias=True,
        **kwargs,
    ):
        super().__init__()

        drop_p = _get_dropout_p(dropout)

        patch_to_emb = [nn.Linear(input_dim, mlp_dim)]
        if act.lower() == "relu":
            patch_to_emb += [nn.ReLU()]
        elif act.lower() == "gelu":
            patch_to_emb += [nn.GELU()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")

        self.patch_to_emb = nn.Sequential(*patch_to_emb)
        self.dp = nn.Dropout(drop_p) if drop_p > 0.0 else nn.Identity()

        self.online_encoder = RRTEncoder(
            mlp_dim=mlp_dim,
            pos_pos=pos_pos,
            pos=pos,
            peg_k=peg_k,
            attn=attn,
            region_num=region_num,
            n_layers=n_layers,
            n_heads=n_heads,
            drop_path=drop_path,
            drop_out=trans_dropout,
            ffn=ffn,
            ffn_act=ffn_act,
            mlp_ratio=mlp_ratio,
            trans_dim=trans_dim,
            epeg=epeg,
            min_region_num=min_region_num,
            qkv_bias=qkv_bias,
            **kwargs,
        )

        self.pool = pool
        self.pool_fn = (
            DAttention(
                self.online_encoder.final_dim,
                da_act,
                gated=da_gated,
                bias=da_bias,
                dropout=da_dropout,
            )
            if pool == "attn" else None
        )

        self.predictor = nn.Linear(self.online_encoder.final_dim, n_classes)
        self.apply(initialize_weights)

    def forward(
        self,
        x,
        label=None,
        instance_eval=False,
        attn_mask=None,
        return_attn=False,
        no_norm=False,
    ):
        """
        x: [B, N, C] or [N, C]
        attn_mask: [B, N] or [N]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)

        if attn_mask is not None and attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)

        B, N, _ = x.shape
        device = x.device

        if attn_mask is None:
            attn_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            attn_mask = attn_mask.bool().to(device)

        x = self.patch_to_emb(x)
        x = self.dp(x)

        # 按官方实现，不在 encoder 前做人为乘 mask
        x = self.online_encoder(x)

        if self.pool == "attn":
            if return_attn:
                x, a = self.pool_fn(
                    x,
                    return_attn=True,
                    no_norm=no_norm,
                    attn_mask=attn_mask,
                )
            else:
                x = self.pool_fn(x, attn_mask=attn_mask)
                a = None
        elif self.pool == "mean":
            denom = attn_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            x = (x * attn_mask.unsqueeze(-1).float()).sum(dim=1) / denom
            a = None
        else:
            x = x.mean(dim=1)
            a = None

        logits = self.predictor(x)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        A_raw = a if self.pool == "attn" else None
        results_dict = {"features": x}

        return logits, Y_prob, Y_hat, A_raw, results_dict
