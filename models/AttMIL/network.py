# models/AttMIL/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_dropout_p(dropout):
    if isinstance(dropout, bool):
        return 0.25 if dropout else 0.0
    if dropout is None:
        return 0.0
    return float(dropout)


class DAttention(nn.Module):
    """
    Subtype version of AB-MIL / AttMIL.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict

    Input:
        x: [B, N, C] or [N, C]
        attn_mask: [B, N] or [N], where True/1 means valid patch, False/0 means padded patch
    """

    def __init__(self, n_classes, dropout=0.25, act="relu", n_features=1024):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1

        drop_p = _get_dropout_p(dropout)

        feature_layers = [nn.Linear(n_features, self.L)]
        if act.lower() == "gelu":
            feature_layers += [nn.GELU()]
        else:
            feature_layers += [nn.ReLU()]

        if drop_p > 0:
            feature_layers += [nn.Dropout(drop_p)]

        self.feature = nn.Sequential(*feature_layers)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Linear(self.L * self.K, n_classes)

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        """
        Args:
            x: [B, N, C] or [N, C]
            label: unused, kept for unified interface
            instance_eval: unused, kept for unified interface
            attn_mask: [B, N] or [N], valid-token mask

        Returns:
            logits: [B, n_classes]
            Y_prob: [B, n_classes]
            Y_hat: [B, 1]
            A_raw: [B, K, N]
            results_dict: {}
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, C]

        if attn_mask is not None and attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)  # [1, N]

        feature = self.feature(x)  # [B, N, 512]

        A = self.attention(feature)            # [B, N, K]
        A = A.transpose(1, 2)                  # [B, K, N]
        A_raw = A.clone()

        if attn_mask is not None:
            mask = attn_mask.bool().unsqueeze(1)   # [B, 1, N]
            A = A.masked_fill(~mask, float("-inf"))

        A = F.softmax(A, dim=-1)               # [B, K, N]
        M = torch.bmm(A, feature)              # [B, K, 512]
        M = M.view(M.size(0), -1)              # [B, K*512]

        logits = self.classifier(M)            # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        results_dict = {}

        return logits, Y_prob, Y_hat, A_raw, results_dict
