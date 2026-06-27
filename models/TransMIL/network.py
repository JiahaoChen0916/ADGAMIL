# models/TransMIL/network.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import NystromAttention


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
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


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=False,
            dropout=0.1,
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm(x), mask=mask)
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    """
    Subtype version of TransMIL.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(self, n_classes, dropout=0.25, act='relu', n_features=1024):
        super(TransMIL, self).__init__()

        drop_p = _get_dropout_p(dropout)

        fc1 = [nn.Linear(n_features, 512)]
        if act.lower() == 'relu':
            fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            fc1 += [nn.GELU()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")
        if drop_p > 0:
            fc1 += [nn.Dropout(drop_p)]
        self._fc1 = nn.Sequential(*fc1)

        self.pos_layer = PPEG(dim=512)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)

        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        """
        Args:
            x: [B, N, C] or [N, C]
            attn_mask: [B, N] or [N], True/1 for valid patches
        Returns:
            logits, Y_prob, Y_hat, A_raw, results_dict
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, C]

        x = x.float()
        B, N, _ = x.shape
        device = x.device

        if attn_mask is None:
            attn_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            if attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.bool().to(device)

        h = self._fc1(x)  # [B, N, 512]
        h = h * attn_mask.unsqueeze(-1).float()

        # ----> pad to square number of tokens
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H

        if add_length > 0:
            h = torch.cat([h, h[:, :add_length, :]], dim=1)
            pad_mask = attn_mask[:, :add_length]
            attn_mask = torch.cat([attn_mask, pad_mask], dim=1)

        # ----> cls token
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        trans_mask = torch.cat((cls_mask, attn_mask), dim=1)

        # ----> Transformer layers
        h = self.layer1(h, mask=trans_mask)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h, mask=trans_mask)

        # ----> cls token
        h_cls = self.norm(h)[:, 0]  # [B, 512]

        logits = self.classifier(h_cls)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        results_dict = {
            "features": h_cls
        }

        return logits, Y_prob, Y_hat, None, results_dict


class TransMIL_NO_PPEG(nn.Module):
    """
    Subtype version of TransMIL without PPEG.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(self, n_classes, dropout=0.25, act='relu', n_features=1024):
        super(TransMIL_NO_PPEG, self).__init__()

        drop_p = _get_dropout_p(dropout)

        fc1 = [nn.Linear(n_features, 512)]
        if act.lower() == 'relu':
            fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            fc1 += [nn.GELU()]
        else:
            raise NotImplementedError(f"Unsupported act: {act}")
        if drop_p > 0:
            fc1 += [nn.Dropout(drop_p)]
        self._fc1 = nn.Sequential(*fc1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)

        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.classifier = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.float()
        B, N, _ = x.shape
        device = x.device

        if attn_mask is None:
            attn_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            if attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.bool().to(device)

        h = self._fc1(x)
        h = h * attn_mask.unsqueeze(-1).float()

        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
        trans_mask = torch.cat((cls_mask, attn_mask), dim=1)

        h = self.layer1(h, mask=trans_mask)
        h = self.layer2(h, mask=trans_mask)

        h_cls = self.norm(h)[:, 0]
        logits = self.classifier(h_cls)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        results_dict = {
            "features": h_cls
        }

        return logits, Y_prob, Y_hat, None, results_dict
