# dataset_modules/dataset_graph.py
import os
import glob
import h5py
import copy
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.graph_builder import build_graph_from_coords_features


def _read_h5_first_existing_key(h5_path, candidate_keys):
    with h5py.File(h5_path, "r") as f:
        for k in candidate_keys:
            if k in f:
                return f[k][:]
        raise KeyError(f"None of keys {candidate_keys} found in {h5_path}")


def _load_pt_feature(pt_path):
    obj = torch.load(pt_path, map_location="cpu")

    if torch.is_tensor(obj):
        feats = obj
    elif isinstance(obj, dict):
        for key in ["features", "feature", "feat", "x"]:
            if key in obj:
                feats = obj[key]
                break
        else:
            raise KeyError(f"Cannot find feature tensor in pt file: {pt_path}")
    else:
        raise TypeError(f"Unsupported pt content type in {pt_path}: {type(obj)}")

    if not torch.is_tensor(feats):
        feats = torch.tensor(feats)
    return feats.float()


def _load_h5_feature(h5_path):
    arr = _read_h5_first_existing_key(
        h5_path,
        candidate_keys=["features", "feature", "feat", "x"],
    )
    return torch.tensor(arr, dtype=torch.float32)


def _load_coords_from_patch_h5(h5_path):
    """
    尽量兼容常见 patch h5 格式：
    - coords
    - coord
    - coordinates
    - x/y
    - x_patch/y_patch
    """
    with h5py.File(h5_path, "r") as f:
        if "coords" in f:
            coords = f["coords"][:]
        elif "coord" in f:
            coords = f["coord"][:]
        elif "coordinates" in f:
            coords = f["coordinates"][:]
        elif "x" in f and "y" in f:
            x = f["x"][:]
            y = f["y"][:]
            coords = np.stack([x, y], axis=1)
        elif "x_patch" in f and "y_patch" in f:
            x = f["x_patch"][:]
            y = f["y_patch"][:]
            coords = np.stack([x, y], axis=1)
        else:
            raise KeyError(
                f"Cannot find coordinates in patch h5 file: {h5_path}. "
                f"Available keys: {list(f.keys())}"
            )

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords in {h5_path} must be [N,2], got {coords.shape}")
    return torch.tensor(coords, dtype=torch.float32)


class Generic_Graph_Dataset(Dataset):
    """
    在线图构造 Dataset

    需要：
    1. csv_path: 含 slide_id 和 label 的 csv
    2. feature_dir: foundation model 特征目录（.pt 或 .h5）
    3. patch_h5_dir: patch 阶段生成的 h5 坐标目录

    兼容：
    - slide_id 与真实文件 stem 完全一致（PANDA / BRCA 常见）
    - slide_id 只是前缀（如 NSCLC patient-level CSV 只有前12位 case id）
    """

    def __init__(
        self,
        csv_path,
        feature_dir,
        patch_h5_dir,
        label_dict,
        label_col="label",
        feature_file_ext=".pt",
        patch_file_ext=".h5",
        graph_mode="knn",
        k=8,
        radius=0.12,
        max_neighbors=None,
        add_self_loop=True,
        with_edge_attr=True,
        shuffle=False,
        seed=1,
        print_info=False,
        ignore=None,
        cache_in_ram=False,
    ):
        super().__init__()

        self.feature_dir = feature_dir
        self.patch_h5_dir = patch_h5_dir
        self.label_dict = label_dict
        self.label_col = label_col
        self.feature_file_ext = feature_file_ext
        self.patch_file_ext = patch_file_ext

        self.graph_mode = graph_mode
        self.k = k
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.add_self_loop = add_self_loop
        self.with_edge_attr = with_edge_attr

        self.ignore = ignore if ignore is not None else []

        self.cache_in_ram = cache_in_ram
        self.graph_cache = {}

        self.slide_data = pd.read_csv(csv_path)
        self.slide_data = self.slide_data[
            ~self.slide_data[self.label_col].isin(self.ignore)
        ].reset_index(drop=True)
        self.slide_data["label"] = self.slide_data[self.label_col].map(self.label_dict)

        if self.slide_data["label"].isna().any():
            bad_rows = self.slide_data[self.slide_data["label"].isna()]
            raise ValueError(
                "Found labels not in label_dict. Bad rows example:\n"
                f"{bad_rows.head()}"
            )

        self.slide_data["label"] = self.slide_data["label"].astype(int)

        if shuffle:
            self.slide_data = self.slide_data.sample(frac=1, random_state=seed).reset_index(drop=True)

        if print_info:
            print(f"[Generic_Graph_Dataset] total slides = {len(self.slide_data)}")
            print(self.slide_data["label"].value_counts(dropna=False).sort_index())

    def __len__(self):
        return len(self.slide_data)

    def _feature_path(self, slide_id):
        return os.path.join(self.feature_dir, f"{slide_id}{self.feature_file_ext}")

    def _patch_h5_path(self, slide_id):
        return os.path.join(self.patch_h5_dir, f"{slide_id}{self.patch_file_ext}")

    def _resolve_one_file(self, base_dir, slide_id, file_ext, file_type="file"):
        """
        解析真实文件路径：
        1) 先精确匹配 slide_id.ext
        2) 再前缀匹配 slide_id*.ext

        适配：
        - BRCA/PANDA: slide_id 已是完整 stem
        - NSCLC patient-level: slide_id 只有前12位，如 TCGA-62-A46R
        """
        slide_id = str(slide_id).strip()

        # 1) exact match
        exact_path = os.path.join(base_dir, f"{slide_id}{file_ext}")
        if os.path.isfile(exact_path):
            return exact_path

        # 2) prefix match
        pattern = os.path.join(base_dir, f"{slide_id}*{file_ext}")
        matches = sorted(glob.glob(pattern))

        if len(matches) == 1:
            return matches[0]

        if len(matches) == 0:
            raise FileNotFoundError(
                f"{file_type} not found for slide_id={slide_id}\n"
                f"Exact tried: {exact_path}\n"
                f"Pattern tried: {pattern}"
            )

        # 若有多个候选，优先选最常见的 DX1 / 01Z，若仍不唯一则报错
        basename_matches = [os.path.basename(m) for m in matches]

        preferred = [
            m for m in matches
            if ("-01Z-" in os.path.basename(m)) or ("DX1" in os.path.basename(m))
        ]
        if len(preferred) == 1:
            return preferred[0]

        raise RuntimeError(
            f"Multiple {file_type}s matched slide_id={slide_id}\n"
            f"Pattern: {pattern}\n"
            f"Candidates:\n" + "\n".join(basename_matches)
        )

    def _resolve_feature_path(self, slide_id):
        return self._resolve_one_file(
            base_dir=self.feature_dir,
            slide_id=slide_id,
            file_ext=self.feature_file_ext,
            file_type="feature file",
        )

    def _resolve_patch_path(self, slide_id):
        return self._resolve_one_file(
            base_dir=self.patch_h5_dir,
            slide_id=slide_id,
            file_ext=self.patch_file_ext,
            file_type="patch h5 file",
        )

    def _load_features(self, slide_id):
        feature_path = self._resolve_feature_path(slide_id)

        if feature_path.endswith(".pt"):
            feats = _load_pt_feature(feature_path)
        elif feature_path.endswith(".h5"):
            feats = _load_h5_feature(feature_path)
        else:
            raise NotImplementedError(f"Unsupported feature file extension: {feature_path}")

        if feats.ndim != 2:
            raise ValueError(
                f"Feature tensor for slide_id={slide_id} must be [N,C], "
                f"got {tuple(feats.shape)} from {feature_path}"
            )
        return feats

    def _load_coords(self, slide_id):
        patch_h5_path = self._resolve_patch_path(slide_id)
        coords = _load_coords_from_patch_h5(patch_h5_path)
        return coords

    def _build_one_graph(self, idx):
        row = self.slide_data.iloc[idx]
        slide_id = str(row["slide_id"]).strip()
        label = int(row["label"])

        feats = self._load_features(slide_id)   # [N, C]
        coords = self._load_coords(slide_id)    # [N, 2]

        if feats.shape[0] != coords.shape[0]:
            raise ValueError(
                f"Slide {slide_id}: feature N != coord N, "
                f"got {feats.shape[0]} vs {coords.shape[0]}"
            )

        data = build_graph_from_coords_features(
            features=feats,
            coords=coords,
            label=label,
            slide_id=slide_id,
            graph_mode=self.graph_mode,
            k=self.k,
            radius=self.radius,
            max_neighbors=self.max_neighbors,
            add_self_loop=self.add_self_loop,
            with_edge_attr=self.with_edge_attr,
        )

        if not isinstance(data, Data):
            raise TypeError(
                "build_graph_from_coords_features must return "
                f"torch_geometric.data.Data, got {type(data)}"
            )

        return data

    def preload_graphs(self, verbose=True):
        total = len(self.slide_data)
        for idx in range(total):
            if idx not in self.graph_cache:
                self.graph_cache[idx] = self._build_one_graph(idx)

            if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == total):
                print(f"[Generic_Graph_Dataset] preload_graphs: {idx + 1}/{total}")

    def clear_cache(self):
        self.graph_cache.clear()

    def __getitem__(self, idx):
        if self.cache_in_ram:
            if idx not in self.graph_cache:
                self.graph_cache[idx] = self._build_one_graph(idx)

            data = self.graph_cache[idx]
            try:
                return data.clone()
            except Exception:
                return copy.deepcopy(data)

        return self._build_one_graph(idx)

    def return_splits(self, from_id=False, csv_path=None):
        if csv_path is None:
            raise ValueError("csv_path must be provided")

        split_df = pd.read_csv(csv_path)

        def make_subset(split_name):
            ids = split_df[split_name].dropna().astype(str).tolist()
            subset_df = self.slide_data[
                self.slide_data["slide_id"].astype(str).isin(ids)
            ].copy().reset_index(drop=True)

            subset = Generic_Graph_Dataset.__new__(Generic_Graph_Dataset)
            Dataset.__init__(subset)

            subset.feature_dir = self.feature_dir
            subset.patch_h5_dir = self.patch_h5_dir
            subset.label_dict = self.label_dict
            subset.label_col = self.label_col
            subset.feature_file_ext = self.feature_file_ext
            subset.patch_file_ext = self.patch_file_ext
            subset.graph_mode = self.graph_mode
            subset.k = self.k
            subset.radius = self.radius
            subset.max_neighbors = self.max_neighbors
            subset.add_self_loop = self.add_self_loop
            subset.with_edge_attr = self.with_edge_attr
            subset.ignore = self.ignore
            subset.cache_in_ram = self.cache_in_ram
            subset.graph_cache = {}
            subset.slide_data = subset_df
            return subset

        train_dataset = make_subset("train")
        val_dataset = make_subset("val")
        test_dataset = make_subset("test")
        return train_dataset, val_dataset, test_dataset
