# utils/graph_builder.py
import math
import numpy as np
import torch

from torch_geometric.data import Data


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_tensor(x, dtype=None):
    if torch.is_tensor(x):
        return x if dtype is None else x.to(dtype=dtype)
    x = torch.from_numpy(np.asarray(x))
    return x if dtype is None else x.to(dtype=dtype)


def normalize_coords(coords):
    """
    coords: [N, 2]
    Min-max normalize to [0, 1].
    """
    coords = _to_numpy(coords).astype(np.float32)
    if coords.shape[0] == 0:
        return coords

    cmin = coords.min(axis=0, keepdims=True)
    cmax = coords.max(axis=0, keepdims=True)
    denom = np.maximum(cmax - cmin, 1e-8)
    coords = (coords - cmin) / denom
    return coords.astype(np.float32)


def pairwise_distance_knn(coords, k=8):
    """
    Build undirected KNN graph from coords.
    Returns edge_index [2, E]
    """
    coords = _to_numpy(coords).astype(np.float32)
    n = coords.shape[0]

    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    k = min(k, n - 1)

    # brute-force distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # [N, N]

    # ignore self
    np.fill_diagonal(dist, np.inf)

    nn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]  # [N, k]

    edges = []
    for i in range(n):
        for j in nn_idx[i]:
            edges.append((i, int(j)))
            edges.append((int(j), i))

    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edges = np.array(edges, dtype=np.int64)
    edges = np.unique(edges, axis=0)
    edge_index = edges.T  # [2, E]
    return edge_index


def radius_graph_from_coords(coords, radius=0.12, max_neighbors=None):
    """
    Build undirected radius graph from normalized coords.
    radius is expected in normalized coordinate space.
    """
    coords = _to_numpy(coords).astype(np.float32)
    n = coords.shape[0]

    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # [N, N]

    edges = []
    for i in range(n):
        neighbors = np.where((dist[i] <= radius) & (dist[i] > 0))[0]
        if max_neighbors is not None and len(neighbors) > max_neighbors:
            nn_order = np.argsort(dist[i, neighbors])[:max_neighbors]
            neighbors = neighbors[nn_order]

        for j in neighbors:
            edges.append((i, int(j)))
            edges.append((int(j), i))

    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int64)

    edges = np.array(edges, dtype=np.int64)
    edges = np.unique(edges, axis=0)
    edge_index = edges.T
    return edge_index


def add_self_loops(edge_index, num_nodes):
    edge_index = _to_numpy(edge_index).astype(np.int64)
    self_loops = np.stack([np.arange(num_nodes), np.arange(num_nodes)], axis=0)
    if edge_index.shape[1] == 0:
        merged = self_loops
    else:
        merged = np.concatenate([edge_index, self_loops], axis=1)
        merged = np.unique(merged.T, axis=0).T
    return merged.astype(np.int64)


def build_edge_attr_from_coords(coords, edge_index):
    """
    edge_attr = [dx, dy, euclidean_distance]
    coords should be normalized coords [N, 2]
    """
    coords = _to_numpy(coords).astype(np.float32)
    edge_index = _to_numpy(edge_index).astype(np.int64)

    if edge_index.shape[1] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    src = edge_index[0]
    dst = edge_index[1]
    delta = coords[dst] - coords[src]  # [E, 2]
    dist = np.sqrt((delta ** 2).sum(axis=1, keepdims=True))
    edge_attr = np.concatenate([delta, dist], axis=1).astype(np.float32)
    return edge_attr


def build_graph_from_coords_features(
    features,
    coords,
    label=None,
    slide_id=None,
    graph_mode="knn",
    k=8,
    radius=0.12,
    max_neighbors=None,
    add_self_loop=True,
    with_edge_attr=True,
):
    """
    features: [N, C]
    coords:   [N, 2]
    """
    features = _to_numpy(features).astype(np.float32)
    coords = _to_numpy(coords).astype(np.float32)

    if features.ndim != 2:
        raise ValueError(f"features must be [N, C], got shape={features.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must be [N, 2], got shape={coords.shape}")
    if features.shape[0] != coords.shape[0]:
        raise ValueError(
            f"features and coords must have same N, got {features.shape[0]} vs {coords.shape[0]}"
        )

    coords_norm = normalize_coords(coords)

    if graph_mode == "knn":
        edge_index = pairwise_distance_knn(coords_norm, k=k)
    elif graph_mode == "radius":
        edge_index = radius_graph_from_coords(coords_norm, radius=radius, max_neighbors=max_neighbors)
    else:
        raise NotImplementedError(f"Unsupported graph_mode: {graph_mode}")

    if add_self_loop:
        edge_index = add_self_loops(edge_index, num_nodes=features.shape[0])

    data = Data(
        x=_to_tensor(features, dtype=torch.float32),
        edge_index=_to_tensor(edge_index, dtype=torch.long),
    )

    data.pos = _to_tensor(coords_norm, dtype=torch.float32)
    data.coords = _to_tensor(coords, dtype=torch.float32)

    if with_edge_attr:
        edge_attr = build_edge_attr_from_coords(coords_norm, edge_index)
        data.edge_attr = _to_tensor(edge_attr, dtype=torch.float32)

    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)

    if slide_id is not None:
        data.slide_id = str(slide_id)

    return data
