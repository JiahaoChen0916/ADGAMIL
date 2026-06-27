# models/ADGAMIL/dynamic_graph.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGraphEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_neighbors=5):
        super(DynamicGraphEmbedding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors

        # 定义线性层，用于特征变换
        self.conv1 = nn.Linear(input_dim, hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征，形状 (B, num_patches, input_dim)
        Returns:
            graph_features: 图嵌入后的特征，形状 (B, num_patches, hidden_dim)
        """
        B, N, D = x.size()  # B: 批大小, N: 节点数（patch数）, D: 输入维度

        # 计算特征的余弦相似度
        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)  # 归一化，避免除以零
        S = torch.bmm(x_norm, x_norm.transpose(1, 2))  # 相似度矩阵，形状 (B, N, N)

        # 掩盖自环（节点与自身的相似度设为负无穷）
        mask = torch.eye(N, device=x.device).bool().unsqueeze(0).expand(B, -1, -1)
        S.masked_fill_(mask, float('-inf'))

        # 选择每个节点的 top-k 邻居
        values, indices = torch.topk(S, k=self.num_neighbors, dim=2)  # values: (B, N, num_neighbors), indices: (B, N, num_neighbors)

        # 计算邻居的加权系数（基于相似度）
        weights = F.softmax(values, dim=2)  # (B, N, num_neighbors)

        # 收集邻居特征
        neighbor_features = torch.stack([x[b, indices[b], :] for b in range(B)], dim=0)  # (B, N, num_neighbors, D)

        # 加权聚合邻居特征
        aggregate_neighbors = torch.sum(weights.unsqueeze(-1) * neighbor_features, dim=2)  # (B, N, D)

        # 结合自身特征与邻居特征
        h_combined = x + aggregate_neighbors  # (B, N, D)

        # 通过线性层处理
        h_combined = h_combined.view(B * N, D)
        h_combined = F.relu(self.conv1(h_combined))  # (B*N, hidden_dim)
        h_combined = F.relu(self.conv2(h_combined))  # (B*N, hidden_dim)
        graph_features = h_combined.view(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        return graph_features
