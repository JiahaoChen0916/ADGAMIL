# Survival/models/ADGAMIL/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_graph import DynamicGraphEmbedding

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        mods = [nn.Linear(L, D), nn.Tanh()]
        if dropout: mods.append(nn.Dropout(0.25))
        mods.append(nn.Linear(D, n_classes))
        self.att = nn.Sequential(*mods)

    def forward(self, x):  # x: [B*N, L]
        return self.att(x), x

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(L, D), nn.Tanh(), *( [nn.Dropout(0.25)] if dropout else [] ))
        self.b = nn.Sequential(nn.Linear(L, D), nn.Sigmoid(), *( [nn.Dropout(0.25)] if dropout else [] ))
        self.c = nn.Linear(D, n_classes)

    def forward(self, x):  # x: [B*N, L]
        A = self.c(self.a(x) * self.b(x))
        return A, x

class ADGAMIL(nn.Module):
    """
    生存预测版 CLAM + 动态图：
    - 输入: [N, D] 或 [1, N, D]
    - 输出: hazards[B, K], S[B, K]，其中 K=离散时间bins（本仓库默认=4）
    """
    def __init__(self, n_classes=4, embed_dim=1024, size_arg="small",
                 gate=True, dropout=0.25, num_neighbors=5):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256],
                          "big":   [embed_dim, 512, 384]}
        in_dim, mid_dim, att_dim = self.size_dict[size_arg]

        # patch特征FC
        self.fc = nn.Sequential(nn.Linear(in_dim, mid_dim),
                                nn.ReLU(),
                                nn.Dropout(dropout))
        # 动态图嵌入（作用在 mid_dim 上）
        self.dynamic_graph = DynamicGraphEmbedding(input_dim=mid_dim,
                                                   hidden_dim=mid_dim,
                                                   num_neighbors=num_neighbors)
        # 注意力
        self.attn = Attn_Net_Gated(L=mid_dim, D=att_dim, dropout=dropout, n_classes=1) if gate \
                    else Attn_Net(L=mid_dim, D=att_dim, dropout=dropout, n_classes=1)
        # hazard 头
        self.hazard_head = nn.Linear(mid_dim, n_classes)

        initialize_weights(self)

    def forward(self, h):
        # 接受 [N, D] 或 [1, N, D]
        if h.dim() == 2:
            h = h.unsqueeze(0)  # -> [1, N, D]
        assert h.dim() == 3, "expect [B,N,D] or [N,D]"
        B, N, D = h.shape

        # FC到 mid_dim
        h = self.fc(h.view(B * N, D)).view(B, N, -1)  # [B,N,mid_dim]

        # 动态图(注意：此步为O(N^2)，N过大时请下采样或减小num_neighbors)
        h = self.dynamic_graph(h).view(B * N, -1)     # [B*N, mid_dim]

        # 注意力
        A, h = self.attn(h)                           # A: [B*N,1], h: [B*N,mid_dim]
        A = A.view(B, N, 1)
        A = F.softmax(A, dim=1)                       # over instances

        # Bag 特征 + 生存头
        h = h.view(B, N, -1)                          # [B,N,mid_dim]
        M = torch.bmm(A.transpose(1, 2), h).squeeze(1)  # [B,mid_dim]

        hazards = torch.sigmoid(self.hazard_head(M))  # [B,K]
        S = torch.cumprod(1 - hazards, dim=1)         # [B,K]
        return hazards, S
