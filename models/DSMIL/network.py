# models/DSMIL/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin import RRTEncoder


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
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


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1, dropout=0.25, act="relu", input_dim=1024):
        super(FCLayer, self).__init__()

        drop_p = _get_dropout_p(dropout)

        embed = [nn.Linear(input_dim, 512)]
        if act.lower() == "gelu":
            embed += [nn.GELU()]
        else:
            embed += [nn.ReLU()]
        if drop_p > 0:
            embed += [nn.Dropout(drop_p)]

        self.embed = nn.Sequential(*embed)
        self.fc = nn.Linear(in_size, out_size)

    def forward(self, feats):
        feats = self.embed(feats)
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)
        c = self.fc(feats.view(feats.shape[0], -1))
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=True):
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Tanh()
            )
        else:
            self.q = nn.Linear(input_size, 128)

        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):
        """
        feats: [N, K]
        c:     [N, C]
        """
        device = feats.device
        V = self.v(feats)                         # [N, K]
        Q = self.q(feats).view(feats.shape[0], -1)  # [N, 128]

        _, m_indices = torch.sort(c, 0, descending=True)     # [N, C]
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # [C, K]
        q_max = self.q(m_feats)                              # [C, 128]

        A = torch.mm(Q, q_max.transpose(0, 1))              # [N, C]
        A = F.softmax(
            A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
            dim=0
        )                                                   # [N, C]

        B = torch.mm(A.transpose(0, 1), V)                  # [C, K]
        B = B.view(1, B.shape[0], B.shape[1])               # [1, C, K]

        C = self.fcc(B)                                     # [1, C, 1]
        C = C.view(1, -1)                                   # [1, C]

        return C, A, B


class MILNet(nn.Module):
    """
    Subtype version of DSMIL.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(
        self,
        n_classes,
        dropout=0.25,
        act="relu",
        input_dim=1024,
        rrt=False,
        rrt_convk=15,
        rrt_moek=3,
        rrt_as=False,
        rrt_md=False,
        no_rrt_init=False,
        no_init=True
    ):
        super(MILNet, self).__init__()

        drop_p = _get_dropout_p(dropout)

        patch_to_emb = [nn.Linear(input_dim, 512)]
        if act.lower() == "relu":
            patch_to_emb += [nn.ReLU()]
        elif act.lower() == "gelu":
            patch_to_emb += [nn.GELU()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")

        self.patch_to_emb = nn.Sequential(*patch_to_emb)
        self.dp = nn.Dropout(drop_p) if drop_p > 0.0 else nn.Identity()

        self.rrt = (
            RRTEncoder(
                attn="rrt",
                pool="none",
                n_layers=2,
                epeg_k=rrt_convk,
                crmsa_k=rrt_moek,
                all_shortcut=rrt_as,
                moe_mask_diag=rrt_md,
                init=not no_rrt_init,
            )
            if rrt else nn.Identity()
        )

        self.i_classifier = nn.Linear(512, n_classes)
        self.b_classifier = BClassifier(512, n_classes)

        if not no_init:
            self.apply(initialize_weights)

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        """
        Args:
            x: [B, N, C] or [N, C]
            attn_mask: [B, N] or [N], True/1 for valid patches

        Returns:
            logits: [B, n_classes]  -> bag logits
            Y_prob: [B, n_classes]
            Y_hat: [B, 1]
            A_raw: [B, C, N_valid] or None
            results_dict: contains instance logits and attention
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, C]

        if x.size(0) != 1:
            raise ValueError(
                f"DSMIL currently expects batch_size=1, but got x.shape={tuple(x.shape)}"
            )

        device = x.device
        B, N, _ = x.shape

        if attn_mask is None:
            attn_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            if attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.bool().to(device)

        valid_x = x[0][attn_mask[0]]  # [N_valid, C]

        if valid_x.size(0) == 0:
            # 极端保护，避免全mask导致崩溃
            valid_x = x[0][:1]

        feats = self.patch_to_emb(valid_x)      # [N_valid, 512]
        feats = self.dp(feats)

        rrt_feats = self.rrt(feats)             # [N_valid, 512] or identity
        prediction_ins = self.i_classifier(rrt_feats)    # [N_valid, C]
        prediction_bag, A, B_feat = self.b_classifier(feats, prediction_ins)  # [1, C], [N_valid, C], [1, C, 512]

        logits = prediction_bag
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        top_instance_logits, top_instance_idx = prediction_ins.max(dim=0, keepdim=True)  # [1, C]
        A_raw = A.transpose(0, 1).unsqueeze(0)  # [1, C, N_valid]

        results_dict = {
            "instance_logits": prediction_ins,          # [N_valid, C]
            "bag_logits": prediction_bag,               # [1, C]
            "top_instance_logits": top_instance_logits, # [1, C]
            "top_instance_idx": top_instance_idx,       # [1, C]
            "attention_scores": A,                      # [N_valid, C]
            "bag_features": B_feat                      # [1, C, 512]
        }

        return logits, Y_prob, Y_hat, A_raw, results_dict
