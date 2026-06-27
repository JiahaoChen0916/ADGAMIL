# models/SlideGraph/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import (
    GINConv,
    EdgeConv,
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


def _get_pool_fn(pooling: str):
    pooling = pooling.lower()
    if pooling == "max":
        return global_max_pool
    elif pooling == "mean":
        return global_mean_pool
    elif pooling == "add":
        return global_add_pool
    else:
        raise NotImplementedError(f"Unsupported pooling: {pooling}")


class SlideGraphGNN(nn.Module):
    """
    SlideGraph-style graph classifier adapted to the current subtype framework.

    Original public SlideGraph repo uses:
        pooling='max'
        dropout=0.0
        conv='GINConv'
        gembed=False
    and sums graph-level scores from each layer after pooling node-level scores.
    This implementation preserves that behavior while exposing the unified output:
        logits, Y_prob, Y_hat, A_raw, results_dict
    """

    def __init__(
        self,
        dim_features=1024,
        n_classes=2,
        layers=(256, 256),
        pooling="max",
        dropout=0.0,
        conv="GINConv",
        gembed=False,
        edgeconv_aggr="mean",
        **kwargs,
    ):
        super().__init__()

        if not isinstance(layers, (list, tuple)) or len(layers) == 0:
            raise ValueError("`layers` must be a non-empty list/tuple, e.g. (256, 256).")

        self.dropout = float(dropout)
        self.embeddings_dim = list(layers)
        self.no_layers = len(self.embeddings_dim)
        self.pooling = _get_pool_fn(pooling)
        self.gembed = bool(gembed)
        self.conv_name = conv

        self.first_h = None
        self.nns = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()

        for layer_idx, out_emb_dim in enumerate(self.embeddings_dim):
            if layer_idx == 0:
                self.first_h = Sequential(
                    Linear(dim_features, out_emb_dim),
                    BatchNorm1d(out_emb_dim),
                    ReLU(),
                )
                self.linears.append(Linear(out_emb_dim, n_classes))
            else:
                input_emb_dim = self.embeddings_dim[layer_idx - 1]
                self.linears.append(Linear(out_emb_dim, n_classes))

                if conv == "GINConv":
                    subnet = Sequential(
                        Linear(input_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                    )
                    self.nns.append(subnet)
                    self.convs.append(GINConv(self.nns[-1], **kwargs))

                elif conv == "EdgeConv":
                    subnet = Sequential(
                        Linear(2 * input_emb_dim, out_emb_dim),
                        BatchNorm1d(out_emb_dim),
                        ReLU(),
                    )
                    self.nns.append(subnet)
                    self.convs.append(EdgeConv(self.nns[-1], aggr=edgeconv_aggr, **kwargs))

                else:
                    raise NotImplementedError(f"Unsupported conv: {conv}")

    def forward(self, data=None, x_path=None, **kwargs):
        if data is None:
            data = x_path
        if data is None:
            raise ValueError("SlideGraphGNN requires a PyG Data/Batch input.")

        if not hasattr(data, "x") or data.x is None:
            raise ValueError("Input graph data must contain `x`.")
        if not hasattr(data, "edge_index") or data.edge_index is None:
            raise ValueError("Input graph data must contain `edge_index`.")

        x = data.x
        edge_index = data.edge_index

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_logits = None
        accumulated_node_logits = None
        node_embeddings = []

        for layer_idx in range(self.no_layers):
            if layer_idx == 0:
                x = self.first_h(x)
                z = self.linears[layer_idx](x)  # node-level logits
                pooled = self.pooling(z, batch)

            else:
                x = self.convs[layer_idx - 1](x, edge_index)

                if not self.gembed:
                    z = self.linears[layer_idx](x)  # node-level logits
                    pooled = self.pooling(z, batch)
                else:
                    z = None
                    pooled = self.linears[layer_idx](self.pooling(x, batch))

            pooled = F.dropout(pooled, p=self.dropout, training=self.training)

            if graph_logits is None:
                graph_logits = pooled
            else:
                graph_logits = graph_logits + pooled

            if z is not None:
                if accumulated_node_logits is None:
                    accumulated_node_logits = z
                else:
                    accumulated_node_logits = accumulated_node_logits + z

            node_embeddings.append(x)

        logits = graph_logits
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        results_dict = {
            "node_embeddings": node_embeddings[-1] if len(node_embeddings) > 0 else None,
            "all_node_embeddings": node_embeddings,
            "node_logits": accumulated_node_logits,
            "graph_embedding_mode": self.gembed,
            "conv_type": self.conv_name,
        }

        A_raw = None
        return logits, Y_prob, Y_hat, A_raw, results_dict
