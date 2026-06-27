# models/PatchGCN/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Sequential as Seq
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GENConv, DeepGCNLayer


def _get_dropout_p(dropout):
    if isinstance(dropout, bool):
        return 0.25 if dropout else 0.0
    if dropout is None:
        return 0.0
    return float(dropout)


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        drop_p = _get_dropout_p(dropout)

        attention_a = [nn.Linear(L, D), nn.Tanh()]
        attention_b = [nn.Linear(L, D), nn.Sigmoid()]

        if drop_p > 0:
            attention_a.append(nn.Dropout(drop_p))
            attention_b.append(nn.Dropout(drop_p))

        self.attention_a = nn.Sequential(*attention_a)
        self.attention_b = nn.Sequential(*attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class PatchGCN(nn.Module):
    """
    Subtype version of Patch-GCN.

    Unified output:
        logits, Y_prob, Y_hat, A_raw, results_dict

    Expected input:
        data: torch_geometric.data.Data or Batch
            data.x
            data.edge_index or data.edge_latent
            data.batch
    """

    def __init__(
        self,
        input_dim=1024,
        num_layers=4,
        edge_agg='spatial',
        resample=0,
        num_features=1024,
        hidden_dim=128,
        linear_dim=64,
        use_edges=False,
        pool=False,
        dropout=0.25,
        n_classes=2,
        multires=False,
        fusion=None,
    ):
        super().__init__()

        self.use_edges = use_edges
        self.fusion = fusion
        self.pool = pool
        self.edge_agg = edge_agg
        self.multires = multires
        self.num_layers = num_layers - 1
        self.resample = resample

        if self.resample > 0:
            self.fc = nn.Sequential(
                nn.Dropout(self.resample),
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
            first_dim = 256
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.25)
            )
            first_dim = hidden_dim

        self.layers = nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(
                hidden_dim,
                hidden_dim,
                aggr='softmax',
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm='layer'
            )
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(
                conv,
                norm,
                act,
                block='res',
                dropout=0.1,
                ckpt_grad=i % 3
            )
            self.layers.append(layer)

        concat_dim = first_dim + self.num_layers * hidden_dim
        self.path_phi = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.path_attention_head = Attn_Net_Gated(
            L=concat_dim,
            D=concat_dim,
            dropout=dropout,
            n_classes=1
        )
        self.path_rho = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(concat_dim, n_classes)

    def _pool_one_graph(self, h_graph):
        """
        h_graph: [num_nodes_graph, C]
        """
        A_path, h_path = self.path_attention_head(h_graph)  # [N,1], [N,C]
        A_path = torch.transpose(A_path, 1, 0)              # [1,N]
        A_soft = F.softmax(A_path, dim=1)
        h_path = torch.mm(A_soft, h_path)                   # [1,C]
        h = self.path_rho(h_path).squeeze(0)                # [C]
        logits = self.classifier(h).unsqueeze(0)            # [1,num_classes]
        return logits, A_path, h

    def forward(self, data=None, x_path=None, **kwargs):
        if data is None:
            data = x_path
        if data is None:
            raise ValueError("PatchGCN requires a PyG Data/Batch input.")

        if self.edge_agg == 'spatial':
            edge_index = data.edge_index
        elif self.edge_agg == 'latent':
            if hasattr(data, "edge_latent") and data.edge_latent is not None:
                edge_index = data.edge_latent
            else:
                edge_index = data.edge_index
        else:
            raise NotImplementedError(f"Unsupported edge_agg: {self.edge_agg}")

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        edge_attr = None

        x = self.fc(data.x)
        x_ = x

        if len(self.layers) > 0:
            x = self.layers[0].conv(x_, edge_index, edge_attr)
            x_ = torch.cat([x_, x], dim=1)

            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)
                x_ = torch.cat([x_, x], dim=1)

        h_path = self.path_phi(x_)

        unique_batches = torch.unique(batch)
        logits_list = []
        A_list = []
        graph_features = []

        for b in unique_batches:
            mask = (batch == b)
            h_graph = h_path[mask]
            logits_b, A_b, feat_b = self._pool_one_graph(h_graph)
            logits_list.append(logits_b)
            A_list.append(A_b)
            graph_features.append(feat_b.unsqueeze(0))

        logits = torch.cat(logits_list, dim=0)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        A_raw = A_list[0] if len(A_list) == 1 else A_list
        results_dict = {
            "node_embeddings": h_path,
            "graph_features": torch.cat(graph_features, dim=0) if len(graph_features) > 0 else None,
        }

        return logits, Y_prob, Y_hat, A_raw, results_dict
