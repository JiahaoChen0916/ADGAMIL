# models/WiKG/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention


def _get_dropout_p(dropout):
    if isinstance(dropout, bool):
        return 0.25 if dropout else 0.0
    if dropout is None:
        return 0.0
    return float(dropout)


class WiKG(nn.Module):
    """
    Subtype version of WiKG.

    Unified forward output:
        logits, Y_prob, Y_hat, A_raw, results_dict

    Input:
        x: [B, N, C] or [N, C] or dict["feature"]
        attn_mask: [B, N] or [N], True/1 means valid patch, False/0 means padded patch
    """

    def __init__(
        self,
        dim_in=1024,
        dim_hidden=512,
        topk=6,
        n_classes=2,
        agg_type='bi-interaction',
        dropout=0.3,
        pool='attn'
    ):
        super().__init__()

        drop_p = _get_dropout_p(dropout)

        self._fc1 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.LeakyReLU()
        )

        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type
        self.pool = pool

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError(f"Unsupported agg_type: {agg_type}")

        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(drop_p) if drop_p > 0 else nn.Identity()

        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        if pool == "mean":
            self.readout = global_mean_pool
        elif pool == "max":
            self.readout = global_max_pool
        elif pool == "attn":
            att_net = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden // 2),
                nn.LeakyReLU(),
                nn.Linear(dim_hidden // 2, 1)
            )
            self.readout = GlobalAttention(att_net)
        else:
            raise NotImplementedError(f"Unsupported pool type: {pool}")

    def forward(self, x, label=None, instance_eval=False, attn_mask=None):
        """
        Args:
            x: [B, N, C] or [N, C] or {"feature": tensor}
            attn_mask: [B, N] or [N], valid-token mask

        Returns:
            logits: [B, n_classes]
            Y_prob: [B, n_classes]
            Y_hat: [B, 1]
            A_raw: top-k neighbor attention logits [B, N, K] or None
            results_dict: dict
        """
        if isinstance(x, dict):
            x = x["feature"]

        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, C]

        B, N, C = x.shape
        device = x.device

        if attn_mask is None:
            attn_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        else:
            if attn_mask.dim() == 1:
                attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask.bool().to(device)

        x = self._fc1(x)  # [B, N, H]
        x = x * attn_mask.unsqueeze(-1).float()

        # 用有效 patch 的均值做平滑
        denom = attn_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)  # [B,1,1]
        mean_feat = (x * attn_mask.unsqueeze(-1).float()).sum(dim=1, keepdim=True) / denom
        x = (x + mean_feat) * 0.5

        e_h = self.W_head(x)  # [B, N, H]
        e_t = self.W_tail(x)  # [B, N, H]

        # 构造 pairwise attention logits
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # [B, N, N]

        # mask 掉无效 patch，避免 topk 选到 padding
        key_mask = attn_mask.unsqueeze(1).expand(B, N, N)   # [B, N, N]
        attn_logit = attn_logit.masked_fill(~key_mask, float("-inf"))

        k = min(self.topk, N)
        topk_weight, topk_index = torch.topk(attn_logit, k=k, dim=-1)  # [B, N, K]

        batch_indices = torch.arange(B, device=device).view(B, 1, 1).expand_as(topk_index)
        Nb_h = e_t[batch_indices, topk_index, :]  # [B, N, K, H]

        topk_prob = F.softmax(topk_weight, dim=2)  # [B, N, K]

        # 原实现里的关系融合
        eh_r = topk_prob.unsqueeze(-1) * Nb_h + (1 - topk_prob).unsqueeze(-1) * e_h.unsqueeze(2)

        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, k, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('bnkh,bnkh->bnk', Nb_h, gate)
        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(2)  # [B, N, 1, K]
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(2)       # [B, N, H]

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            embedding = sum_embedding + bi_embedding
        else:
            raise NotImplementedError

        h = self.message_dropout(embedding)
        h = h * attn_mask.unsqueeze(-1).float()

        # readout
        pooled_list = []
        for b in range(B):
            valid_h = h[b][attn_mask[b]]  # [N_valid, H]
            if valid_h.size(0) == 0:
                valid_h = h[b][:1]

            if self.pool == "mean":
                batch_idx = torch.zeros(valid_h.size(0), dtype=torch.long, device=device)
                pooled = self.readout(valid_h, batch=batch_idx)  # [1, H]
            elif self.pool == "max":
                batch_idx = torch.zeros(valid_h.size(0), dtype=torch.long, device=device)
                pooled = self.readout(valid_h, batch=batch_idx)  # [1, H]
            elif self.pool == "attn":
                batch_idx = torch.zeros(valid_h.size(0), dtype=torch.long, device=device)
                pooled = self.readout(valid_h, batch=batch_idx)  # [1, H]
            else:
                raise NotImplementedError

            pooled_list.append(pooled)

        h = torch.cat(pooled_list, dim=0)  # [B, H]
        h = self.norm(h)
        logits = self.fc(h)                # [B, n_classes]

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        A_raw = topk_weight
        results_dict = {
            "features": h,
            "neighbor_index": topk_index,
            "neighbor_weight": topk_prob
        }

        return logits, Y_prob, Y_hat, A_raw, results_dict
