# models/Mean_Max/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MeanMIL(nn.Module):
    """
    Mean-pooling MIL for subtype classification.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(self, n_features=1024, n_classes=2, dropout=0.25, act='relu'):
        super(MeanMIL, self).__init__()

        drop_p = _get_dropout_p(dropout)

        head = [nn.Linear(n_features, 512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")

        if drop_p > 0:
            head += [nn.Dropout(drop_p)]

        head += [nn.Linear(512, n_classes)]

        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        """
        Args:
            x: [B, N, C] or [N, C]
            attn_mask: [B, N] or [N], True/1 for valid patches

        Returns:
            logits: [B, n_classes]
            Y_prob: [B, n_classes]
            Y_hat: [B, 1]
            A_raw: None
            results_dict: {}
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, C]

        if attn_mask is not None and attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)

        patch_logits = self.head(x)  # [B, N, n_classes]

        if attn_mask is None:
            logits = patch_logits.mean(dim=1)
        else:
            mask = attn_mask.float().unsqueeze(-1)         # [B, N, 1]
            denom = mask.sum(dim=1).clamp(min=1.0)         # [B, 1]
            logits = (patch_logits * mask).sum(dim=1) / denom

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return logits, Y_prob, Y_hat, None, {}


class MaxMIL(nn.Module):
    """
    Max-pooling MIL for subtype classification.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(self, n_features=1024, n_classes=2, dropout=0.25, act='relu'):
        super(MaxMIL, self).__init__()

        drop_p = _get_dropout_p(dropout)

        head = [nn.Linear(n_features, 512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")

        if drop_p > 0:
            head += [nn.Dropout(drop_p)]

        head += [nn.Linear(512, n_classes)]

        self.head = nn.Sequential(*head)
        self.apply(initialize_weights)

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        """
        Args:
            x: [B, N, C] or [N, C]
            attn_mask: [B, N] or [N], True/1 for valid patches

        Returns:
            logits: [B, n_classes]
            Y_prob: [B, n_classes]
            Y_hat: [B, 1]
            A_raw: None
            results_dict: {}
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, C]

        if attn_mask is not None and attn_mask.dim() == 1:
            attn_mask = attn_mask.unsqueeze(0)

        patch_logits = self.head(x)  # [B, N, n_classes]

        if attn_mask is None:
            logits, _ = patch_logits.max(dim=1)
        else:
            mask = attn_mask.bool().unsqueeze(-1)  # [B, N, 1]
            masked_logits = patch_logits.masked_fill(~mask, float("-inf"))
            logits, _ = masked_logits.max(dim=1)

            # 防止某些极端情况下整行都被mask掉
            bad_rows = torch.isinf(logits).any(dim=1)
            if bad_rows.any():
                logits[bad_rows] = 0.0

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        return logits, Y_prob, Y_hat, None, {}
