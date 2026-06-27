# models/dynamic_graph.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGraphEmbedding(nn.Module):
    """
    WiKG-inspired directional dynamic graph embedding used before CLAM attention pooling.

    Key design choices for efficiency:
    1. Only valid (unpadded) patches are processed.
    2. Top-k neighbour search is computed chunk-by-chunk, so peak memory is O(chunk_size * N)
       instead of explicitly materializing a full B x N x N similarity tensor.
    3. No dense adjacency matrix is stored.
    """

    def __init__(self, input_dim, hidden_dim, num_neighbors=5, chunk_size=512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.chunk_size = chunk_size

        self.input_proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)

        self.head_proj = nn.Linear(hidden_dim, hidden_dim)
        self.tail_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

        self.linear_sum = nn.Linear(hidden_dim, hidden_dim)
        self.linear_bi = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.norm = nn.LayerNorm(hidden_dim)

    def _update_single_bag(self, x):
        """
        Args:
            x: (n_valid, D)
        Returns:
            out: (n_valid, hidden_dim)
        """
        n_valid = x.size(0)
        base = self.input_proj(x)

        # Context stabilization used in WiKG: inject a light global context before edge construction.
        base = 0.5 * (base + base.mean(dim=0, keepdim=True))

        if n_valid <= 1 or self.num_neighbors <= 0:
            return self.norm(base)

        head = self.head_proj(base)
        tail = self.tail_proj(base)
        actual_k = min(self.num_neighbors, n_valid - 1)
        if actual_k <= 0:
            return self.norm(base)

        updated_chunks = []
        chunk_size = min(self.chunk_size, n_valid)

        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            q = end - start

            head_chunk = head[start:end]  # (q, C)
            logits = torch.matmul(head_chunk * self.scale, tail.transpose(0, 1))  # (q, n_valid)

            # Exclude self-loops for the current query chunk.
            row_ids = torch.arange(q, device=x.device)
            col_ids = torch.arange(start, end, device=x.device)
            logits[row_ids, col_ids] = torch.finfo(logits.dtype).min

            topk_weight, topk_index = torch.topk(logits, k=actual_k, dim=-1)  # (q, k)
            topk_prob = F.softmax(topk_weight, dim=-1)
            nb_tail = tail[topk_index]  # (q, k, C)

            relation = (
                topk_prob.unsqueeze(-1) * nb_tail
                + (1.0 - topk_prob).unsqueeze(-1) * head_chunk.unsqueeze(1)
            )

            gate = torch.tanh(head_chunk.unsqueeze(1) + relation)
            ka_weight = torch.sum(nb_tail * gate, dim=-1)  # (q, k)
            ka_prob = F.softmax(ka_weight, dim=-1)
            agg_neighbors = torch.sum(ka_prob.unsqueeze(-1) * nb_tail, dim=1)  # (q, C)

            sum_embedding = self.activation(self.linear_sum(head_chunk + agg_neighbors))
            bi_embedding = self.activation(self.linear_bi(head_chunk * agg_neighbors))
            updated = sum_embedding + bi_embedding

            # Residual connection keeps CLAM instance features stable.
            updated = self.norm(updated + base[start:end])
            updated_chunks.append(updated)

        return torch.cat(updated_chunks, dim=0)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, input_dim)
            mask: optional (B, N) boolean tensor, True for valid patches.
        Returns:
            graph_features: (B, N, hidden_dim)
        """
        B, N, _ = x.shape

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        else:
            mask = mask.bool()

        out = x.new_zeros(B, N, self.hidden_dim)

        for b in range(B):
            valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue

            x_valid = x[b, valid_idx]
            out_valid = self._update_single_bag(x_valid)

            if out_valid.dtype != out.dtype:
                out_valid = out_valid.to(dtype=out.dtype)
            out[b, valid_idx] = out_valid

        return out
