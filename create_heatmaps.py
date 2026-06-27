# create_heatmaps_topk_3.py
from __future__ import print_function

import numpy as np
import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_clam import CLAM_MB, CLAM_SB
from models.builder import get_encoder
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from tqdm import tqdm
from PIL import Image
import gc  # 垃圾回收

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task_to_embed_dim = {
    "TCGA_BRCA_UNI_100": 1024,
    "TCGA_BRCA_CHIEF_100": 768,
    "TCGA_NSCLC_UNI_100": 1024,
    "TCGA_NSCLC_CHIEF_100": 768,
    "PANDA_UNI_100": 1024,
    "PANDA_CHIEF_100": 768,
}


def adjust_features(features, target_dim):
    current_dim = features.shape[-1]
    #print(f"Checking feature dim: current={current_dim}, expected={target_dim}")
    if current_dim != target_dim:
        raise ValueError(
            f"Feature dim mismatch: got {current_dim}, expected {target_dim}"
        )
    return features


def infer_single_slide(model, features, label, reverse_label_dict, k=1, embed_dim=1024):
    # print(f"Features shape before any adjustment: {features.shape}")
    if features.dim() == 2:
        features = features.unsqueeze(0)
        print(f"Added batch dimension. New features shape: {features.shape}")
    elif features.dim() != 3:
        raise ValueError(f"Unexpected feature dimensions: {features.dim()}")

    features = adjust_features(features, target_dim=embed_dim)
    features = features.to(device)
    #print(f"Features shape after adjustment and moving to device: {features.shape}")

    with torch.inference_mode():
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()
            if isinstance(model, CLAM_MB):
                A = A[Y_hat]
            A = A.view(-1, 1).cpu().numpy()
        else:
            raise NotImplementedError

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(
            reverse_label_dict[Y_hat], 
            label, 
            ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]
        ))
        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, A

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            try:
                val = dtype(val)
                if isinstance(val, str):
                    if len(val) > 0:
                        params[key] = val
                elif not np.isnan(val):
                    params[key] = val
                else:
                    pdb.set_trace()
            except:
                pdb.set_trace()
    return params

def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

# =========================
# Top-k 导出 & 分位数向量化
# =========================

def _safe_load_scores_coords(h5_path):
    """稳健读取 h5 中的 attention_scores 与 coords，统一形状，去除 NaN/Inf。"""
    with h5py.File(h5_path, "r") as f:
        scores = np.asarray(f["attention_scores"][()]).astype(np.float32)
        coords = np.asarray(f["coords"][()])
    scores = scores.reshape(-1)
    coords = coords.reshape(-1, 2).astype(np.int64)
    N = min(len(scores), coords.shape[0])
    if len(scores) != coords.shape[0]:
        print(f"[WARN] scores (N={len(scores)}) 与 coords (N={coords.shape[0]}) 长度不一致，截断为 N={N}")
    scores = scores[:N]
    coords = coords[:N]
    bad = ~np.isfinite(scores)
    if bad.any():
        print(f"[WARN] attention_scores 中存在 {bad.sum()} 个非有限值，已填充为 -inf")
        scores[bad] = -np.inf
    return scores, coords

def _topk_indices(scores, k):
    k = int(min(max(k, 0), len(scores)))
    if k == 0 or len(scores) == 0:
        return np.array([], dtype=np.int64)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]

def _topk_with_min_dist(scores, coords, k, min_dist):
    order = np.argsort(-scores)
    sel = []
    for i in order:
        if len(sel) >= k:
            break
        if not sel or min_dist <= 0:
            sel.append(i)
            continue
        d = np.sqrt(((coords[sel] - coords[i]) ** 2).sum(axis=1))
        if np.all(d >= min_dist):
            sel.append(i)
    return np.array(sel, dtype=np.int64)

# def _tissue_ratio(pil_img, sat_thresh=20, val_thresh=230):
#     """
#     简单判断 patch 中组织比例。
#     空白背景通常饱和度低、亮度高。
#     """
#     img = pil_img.convert("RGB")
#     arr = np.asarray(img).astype(np.uint8)

#     r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
#     maxc = arr.max(axis=2)
#     minc = arr.min(axis=2)

#     sat = maxc - minc
#     tissue = (sat > sat_thresh) & (maxc < val_thresh)

#     return tissue.mean()

# def export_topk_patches_clam_style(
#     wsi_object,
#     h5_path,
#     out_dir,
#     k,
#     patch_level,
#     patch_size,
#     convert_to_percentiles=True,
#     min_tissue_ratio=0.50,
#     max_candidates=20000
# ):
#     import shutil

#     if os.path.isdir(out_dir):
#         shutil.rmtree(out_dir)
#     os.makedirs(out_dir, exist_ok=True)

#     scores, coords = _safe_load_scores_coords(h5_path)

#     if len(scores) == 0:
#         print(f"[WARN] No valid attention_scores found in {h5_path}")
#         return

#     if len(scores) != len(coords):
#         raise ValueError(
#             f"scores and coords mismatch: scores={len(scores)}, coords={len(coords)}"
#         )

#     if convert_to_percentiles:
#         scores_for_rank = percentiles_from_ref(scores, scores)
#     else:
#         scores_for_rank = scores

#     order = np.argsort(-scores_for_rank)

#     meta_lines = [
#         "rank,score,x_level0,y_level0,patch_level,patch_size,tissue_ratio,h5"
#     ]

#     selected = []
#     rank = 1

#     for idx in order[:max_candidates]:
#         if len(selected) >= k:
#             break

#         score = float(scores_for_rank[idx])
#         x, y = coords[idx].astype(int)

#         patch = wsi_object.wsi.read_region(
#             (int(x), int(y)),
#             patch_level,
#             (patch_size, patch_size)
#         ).convert("RGB")

#         tissue_ratio = _tissue_ratio(
#             patch,
#             sat_thresh=10,
#             val_thresh=245
#         )

#         if tissue_ratio < min_tissue_ratio:
#             continue

#         selected.append(idx)

#         fname = (
#             f"rank_{rank:02d}_score_{score:.6f}_"
#             f"tissue_{tissue_ratio:.3f}_x_{int(x)}_y_{int(y)}.png"
#         )

#         patch.save(os.path.join(out_dir, fname))

#         meta_lines.append(
#             f"{rank},{score:.6f},{int(x)},{int(y)},"
#             f"{patch_level},{patch_size},{tissue_ratio:.4f},{h5_path}"
#         )

#         rank += 1

#     with open(os.path.join(out_dir, "topk.csv"), "w", encoding="utf-8") as f:
#         f.write("\n".join(meta_lines))

#     print(
#         f"[TopK] Saved {len(selected)} tissue-filtered patches to: {out_dir} "
#         f"(min_tissue_ratio={min_tissue_ratio}, max_candidates={max_candidates})"
#     )

#     if len(selected) < k:
#         print(
#             f"[WARN] Only found {len(selected)} patches after tissue filtering. "
#             f"Try lowering min_tissue_ratio."
#         )
def _patch_quality(pil_img):
    """
    返回 patch 质量指标：
    - tissue_ratio: 有染色组织比例
    - bright_ratio: 过亮背景比例
    - cellular_ratio: 深染/紫色细胞核比例
    - edge_score: 简单纹理/清晰度指标
    """
    img = pil_img.convert("RGB")
    arr = np.asarray(img).astype(np.float32)

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    sat = maxc - minc

    # 有染色组织：不能太白，且有一定颜色饱和度
    tissue = (sat > 25) & (maxc < 235)

    # 过亮背景/玻片空白
    bright = (maxc > 245) & (sat < 25)

    # 深染细胞/细胞核区域：偏紫、偏暗、有饱和度
    dark = (maxc < 180) & (sat > 25)
    purple = (b > g + 8) & (r > g + 5) & (maxc < 210) & (sat > 20)

    cellular = dark | purple

    # 简单清晰度/纹理分数
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    edge_x = np.abs(np.diff(gray, axis=1)).mean()
    edge_y = np.abs(np.diff(gray, axis=0)).mean()
    edge_score = float(edge_x + edge_y)

    return {
        "tissue_ratio": float(tissue.mean()),
        "bright_ratio": float(bright.mean()),
        "cellular_ratio": float(cellular.mean()),
        "edge_score": edge_score,
    }


def export_topk_patches_clam_style(
    wsi_object,
    h5_path,
    out_dir,
    k,
    patch_level,
    patch_size,
    convert_to_percentiles=True,
    min_tissue_ratio=0.25,
    max_bright_ratio=0.45,
    min_cellular_ratio=0.015,
    min_edge_score=2.0,
    max_candidates=50000
):
    import shutil

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    scores, coords = _safe_load_scores_coords(h5_path)

    if len(scores) == 0:
        print(f"[WARN] No valid attention_scores found in {h5_path}")
        return

    if len(scores) != len(coords):
        raise ValueError(
            f"scores and coords mismatch: scores={len(scores)}, coords={len(coords)}"
        )

    if convert_to_percentiles:
        scores_for_rank = percentiles_from_ref(scores, scores)
    else:
        scores_for_rank = scores

    order = np.argsort(-scores_for_rank)

    meta_lines = [
        "rank,score,x_level0,y_level0,patch_level,patch_size,"
        "tissue_ratio,bright_ratio,cellular_ratio,edge_score,h5"
    ]

    reject_lines = [
        "attention_rank,score,x_level0,y_level0,"
        "tissue_ratio,bright_ratio,cellular_ratio,edge_score,reason"
    ]

    selected = []
    rank = 1

    for attn_rank, idx in enumerate(order[:max_candidates], 1):
        if len(selected) >= k:
            break

        score = float(scores_for_rank[idx])
        x, y = coords[idx].astype(int)

        patch = wsi_object.wsi.read_region(
            (int(x), int(y)),
            patch_level,
            (patch_size, patch_size)
        ).convert("RGB")

        q = _patch_quality(patch)

        reason = None
        if q["tissue_ratio"] < min_tissue_ratio:
            reason = "low_tissue"
        elif q["bright_ratio"] > max_bright_ratio:
            reason = "too_bright"
        elif q["cellular_ratio"] < min_cellular_ratio:
            reason = "low_cellularity"
        elif q["edge_score"] < min_edge_score:
            reason = "too_blurry"

        if reason is not None:
            reject_lines.append(
                f"{attn_rank},{score:.6f},{int(x)},{int(y)},"
                f"{q['tissue_ratio']:.4f},{q['bright_ratio']:.4f},"
                f"{q['cellular_ratio']:.4f},{q['edge_score']:.4f},{reason}"
            )
            continue

        selected.append(idx)

        fname = (
            f"rank_{rank:02d}_score_{score:.6f}_"
            f"tissue_{q['tissue_ratio']:.3f}_"
            f"cell_{q['cellular_ratio']:.3f}_"
            f"edge_{q['edge_score']:.2f}_"
            f"x_{int(x)}_y_{int(y)}.png"
        )

        patch.save(os.path.join(out_dir, fname))

        meta_lines.append(
            f"{rank},{score:.6f},{int(x)},{int(y)},"
            f"{patch_level},{patch_size},"
            f"{q['tissue_ratio']:.4f},{q['bright_ratio']:.4f},"
            f"{q['cellular_ratio']:.4f},{q['edge_score']:.4f},{h5_path}"
        )

        rank += 1

    with open(os.path.join(out_dir, "topk.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    with open(os.path.join(out_dir, "rejected_candidates.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(reject_lines))

    print(
        f"[TopK] Saved {len(selected)} quality-filtered patches to: {out_dir} "
        f"(searched={min(max_candidates, len(order))}, "
        f"min_tissue={min_tissue_ratio}, max_bright={max_bright_ratio}, "
        f"min_cell={min_cellular_ratio}, min_edge={min_edge_score})"
    )

    if len(selected) < k:
        print(
            f"[WARN] Only found {len(selected)} patches after filtering. "
            f"Try lowering min_cellular_ratio or min_edge_score."
        )




def percentiles_from_ref(scores, ref_scores):
    """
    向量化分位数转换（近似 SciPy percentileofscore(kind='rank')）
    p = ((left + right) / 2) / N * 100
    """
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref_scores, dtype=np.float32).reshape(-1)
    if ref.size == 0:
        return np.zeros_like(scores, dtype=np.float32)
    ref_sorted = np.sort(ref)
    N = float(ref_sorted.size)
    left = np.searchsorted(ref_sorted, scores, side='left')
    right = np.searchsorted(ref_sorted, scores, side='right')
    pct = ((left + right) * 0.5) / N * 100.0
    return pct.astype(np.float32)

# =========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heatmap inference script')
    parser.add_argument('--save_exp_code', type=str, default=None, help='experiment code')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join('heatmaps/configs', args.config_file)
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))
    decision = input('Continue? Y/N ')
    if decision not in ['Y', 'y', 'Yes', 'yes']:
        exit()

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    # 修正：先在字典上添加 n_classes，再转 Namespace
    model_args_dict = dict(args['model_arguments'])
    model_args_dict['n_classes'] = args['exp_arguments']['n_classes']
    model_args = argparse.Namespace(**model_args_dict)
    encoder_args = argparse.Namespace(**args['encoder_arguments'])
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    if not hasattr(heatmap_args, 'convert_to_percentiles'):
        heatmap_args.convert_to_percentiles = True
    
    if not hasattr(heatmap_args, 'enable_fine_heatmap'):
        heatmap_args.enable_fine_heatmap = False

    patch_size = tuple([patch_args.patch_size for _ in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(
        patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))
    
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]
        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]
        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]
        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
    else:
        if os.path.isfile(data_args.process_list):
            process_list_path = data_args.process_list
        else:
            process_list_path = os.path.join('heatmaps/process_lists', data_args.process_list)

        df = pd.read_csv(process_list_path)
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)
        # df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
        # df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))
    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

    feature_extractor, img_transforms = get_encoder(encoder_args.model_name, target_img_size=encoder_args.target_img_size)
    _ = feature_extractor.eval()
    feature_extractor = feature_extractor.to(device)
    model = model.to(device)
    print('Done!')

    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    #for i in tqdm(range(len(process_stack))):
    for i in tqdm(range(len(process_stack)), ncols=100, leave=True):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)	

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')
        grouping = reverse_label_dict[label] if not isinstance(label, str) else label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i,'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')
        
        # 参数汇入
        seg_params = load_params(process_stack.loc[i], dict(def_seg_params))
        filter_params = load_params(process_stack.loc[i], dict(def_filter_params))
        vis_params = load_params(process_stack.loc[i], dict(def_vis_params))

        keep_ids = str(seg_params['keep_ids'])
        seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int) if len(keep_ids) > 0 and keep_ids != 'none' else []

        exclude_ids = str(seg_params['exclude_ids'])
        seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int) if len(exclude_ids) > 0 and exclude_ids != 'none' else []

        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.h5')
        mask_path = os.path.join(r_slide_save_dir, f'{slide_id}_mask.jpg')
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')

        # 1) 特征
        if not os.path.isfile(h5_path):
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)
            print(f"Processing with embed_dim: {embed_dim}")
            _, _, wsi_object = compute_from_patches(
                wsi_object=wsi_object, 
                model=model, 
                feature_extractor=feature_extractor, 
                img_transforms=img_transforms,
                batch_size=exp_args.batch_size, 
                embed_dim=embed_dim,
                top_left=top_left, 
                bot_right=bot_right, 
                patch_size=patch_size, 
                step_size=step_size,
                custom_downsample=patch_args.custom_downsample,
                patch_level=patch_args.patch_level,
                use_center_shift=heatmap_args.use_center_shift,
                attn_save_path=None, 
                feat_save_path=h5_path, 
                ref_scores=None
            )				

        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:].astype(np.float32))
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)
            print(f"Loaded features shape from h5: {features.shape}")
            features = adjust_features(features, target_dim=embed_dim)
            print(f"Features shape after adjustment: {features.shape}")
            features = features.unsqueeze(0)
            print(f"Features shape after adding batch dimension: {features.shape}")
            torch.save(features, features_path)
            file.close()

        # 2) 推理 -> blockmap.h5
        features = torch.load(features_path)
        print(f"Loaded features shape from pt: {features.shape}")
        process_stack.loc[i, 'bag_size'] = features.size(1)
        wsi_object.saveSegmentation(mask_file)

        task_name = exp_args.save_exp_code
        embed_dim = task_to_embed_dim.get(task_name, 1024)
        print(f"Inferencing single slide with embed_dim: {embed_dim}")
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, exp_args.n_classes, embed_dim)
        del features
        gc.collect()

        with h5py.File(h5_path, "r") as file:
            coords_nonoverlap = file["coords"][:].astype(np.int32)
        
        W0, H0 = wsi_object.wsi.level_dimensions[0]

        print(f"[DEBUG] WSI level-0 size: W={W0}, H={H0}")
        print(f"[DEBUG] coords shape: {coords_nonoverlap.shape}")
        print(f"[DEBUG] coords min: {coords_nonoverlap.min(axis=0)}")
        print(f"[DEBUG] coords max: {coords_nonoverlap.max(axis=0)}")
        print(f"[DEBUG] first 5 coords:\n{coords_nonoverlap[:5]}")

        if A.reshape(-1).shape[0] != coords_nonoverlap.shape[0]:
            raise ValueError(
                f"A and coords mismatch: A={A.reshape(-1).shape[0]}, "
                f"coords={coords_nonoverlap.shape[0]}"
            )

        asset_dict = {
            "attention_scores": A.astype(np.float32),
            "coords": coords_nonoverlap
        }

        block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode="w")

        
        for c in range(exp_args.n_classes):
            process_stack.loc[i, f'Pred_{c}'] = Y_hats_str[c]
            process_stack.loc[i, f'p_{c}'] = Y_probs[c]
        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_list_name = os.path.basename(data_args.process_list).replace(".csv", "")
            process_stack.to_csv(f'heatmaps/results/{process_list_name}.csv', index=False)
        else:
            process_stack.to_csv(f'heatmaps/results/{exp_args.save_exp_code}.csv', index=False)

        # 3) 立即导出 blockmap 的 top-k
        try:
            k_top = 15
            try:
                if hasattr(sample_args, 'samples') and isinstance(sample_args.samples, list) and len(sample_args.samples) > 0:
                    s0 = sample_args.samples[0]
                    if isinstance(s0, dict) and 'k' in s0:
                        k_top = int(s0['k'])
                    elif hasattr(s0, 'k'):
                        k_top = int(s0.k)
            except Exception:
                pass

            out_dir_block = os.path.join('heatmaps', 'patches', exp_args.save_exp_code, str(grouping), slide_id, 'topk_blockmap')
            # export_topk_patches_clam_style(
            #     wsi_object=wsi_object,
            #     h5_path=block_map_save_path,
            #     out_dir=out_dir_block,
            #     k=k_top,
            #     patch_level=patch_args.patch_level,
            #     patch_size=patch_args.patch_size,
            #     convert_to_percentiles=True,
            #     min_tissue_ratio=0.30,
            #     max_candidates=5000
            # )
            export_topk_patches_clam_style(
                wsi_object=wsi_object,
                h5_path=block_map_save_path,
                out_dir=out_dir_block,
                k=k_top,
                patch_level=patch_args.patch_level,
                patch_size=patch_args.patch_size,
                convert_to_percentiles=True,
                min_tissue_ratio=0.25,
                max_bright_ratio=0.45,
                min_cellular_ratio=0.015,
                min_edge_score=2.0,
                max_candidates=50000
            )
        except Exception as e:
            print(f"[WARN] 导出 blockmap top-k 失败: {e}")

        # # 4) 细粒度注意力 & 立即导出 fine top-k
        # save_path_fine = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}_roi_{heatmap_args.use_roi}.h5')
        # if heatmap_args.calc_heatmap:
        #     task_name = exp_args.save_exp_code
        #     embed_dim = task_to_embed_dim.get(task_name, 1024)
        #     compute_from_patches(
        #         wsi_object=wsi_object, 
        #         img_transforms=img_transforms,
        #         clam_pred=Y_hats[0], 
        #         model=model, 
        #         feature_extractor=feature_extractor, 
        #         batch_size=exp_args.batch_size, 
        #         embed_dim=embed_dim,
        #         attention_only=True,
        #         top_left=top_left, 
        #         bot_right=bot_right, 
        #         patch_size=patch_size, 
        #         step_size=step_size,
        #         custom_downsample=patch_args.custom_downsample,
        #         patch_level=patch_args.patch_level,
        #         use_center_shift=heatmap_args.use_center_shift,
        #         attn_save_path=save_path_fine,  
        #         ref_scores=None
        #     )

        # if os.path.isfile(save_path_fine):
        #     try:
        #         out_dir_fine = os.path.join('heatmaps', 'patches', exp_args.save_exp_code, str(grouping), slide_id, 'topk_fine')
        #         export_topk_patches_clam_style(
        #             wsi_object=wsi_object,
        #             h5_path=save_path_fine,
        #             out_dir=out_dir_fine,
        #             k=k_top,
        #             patch_level=patch_args.patch_level,
        #             patch_size=patch_args.patch_size,
        #             convert_to_percentiles=True
        #         )
        #     except Exception as e:
        #         print(f"[WARN] 导出 fine top-k 失败: {e}")
        # else:
        #     print("[INFO] 细粒度 h5 不存在，已导出 blockmap 的 top-k。")
        # 4) 细粒度注意力 & 立即导出 fine top-k
        save_path_fine = os.path.join(
            r_slide_save_dir,
            f'{slide_id}_{patch_args.overlap}_roi_{heatmap_args.use_roi}.h5'
        )

        if heatmap_args.enable_fine_heatmap and heatmap_args.calc_heatmap:
            task_name = exp_args.save_exp_code
            embed_dim = task_to_embed_dim.get(task_name, 1024)
            compute_from_patches(
                wsi_object=wsi_object,
                img_transforms=img_transforms,
                clam_pred=Y_hats[0],
                model=model,
                feature_extractor=feature_extractor,
                batch_size=exp_args.batch_size,
                embed_dim=embed_dim,
                attention_only=True,
                top_left=top_left,
                bot_right=bot_right,
                patch_size=patch_size,
                step_size=step_size,
                custom_downsample=patch_args.custom_downsample,
                patch_level=patch_args.patch_level,
                use_center_shift=heatmap_args.use_center_shift,
                attn_save_path=save_path_fine,
                ref_scores=None
            )

        if heatmap_args.enable_fine_heatmap and os.path.isfile(save_path_fine):
            try:
                out_dir_fine = os.path.join(
                    'heatmaps', 'patches', exp_args.save_exp_code,
                    str(grouping), slide_id, 'topk_fine'
                )
                # export_topk_patches_clam_style(
                #     wsi_object=wsi_object,
                #     h5_path=save_path_fine,
                #     out_dir=out_dir_fine,
                #     k=k_top,
                #     patch_level=patch_args.patch_level,
                #     patch_size=patch_args.patch_size,
                #     convert_to_percentiles=True,
                #     min_tissue_ratio=0.30,
                #     max_candidates=5000
                # )
                export_topk_patches_clam_style(
                    wsi_object=wsi_object,
                    h5_path=save_path_fine,
                    out_dir=out_dir_fine,
                    k=k_top,
                    patch_level=patch_args.patch_level,
                    patch_size=patch_args.patch_size,
                    convert_to_percentiles=True,
                    min_tissue_ratio=0.25,
                    max_bright_ratio=0.45,
                    min_cellular_ratio=0.015,
                    min_edge_score=2.0,
                    max_candidates=50000
                )


            except Exception as e:
                print(f"[WARN] 导出 fine top-k 失败: {e}")
        else:
            print("[INFO] fine heatmap disabled, only exported blockmap top-k.")


        # 5) 画热力图（向量化分位数）
        # 5.1 blockmap 可视化
        try:
            scores_block, coords_block = _safe_load_scores_coords(block_map_save_path)
            scores_block_vis = percentiles_from_ref(scores_block, scores_block) if heatmap_args.convert_to_percentiles else scores_block
            heatmap_block = drawHeatmap(
                scores=scores_block_vis.flatten(),
                coords=coords_block,
                slide_path=slide_path,
                wsi_object=wsi_object,
                cmap=heatmap_args.cmap,
                alpha=heatmap_args.alpha,
                use_holes=True,
                binarize=heatmap_args.binarize,
                vis_level=heatmap_args.vis_level,
                blank_canvas=heatmap_args.blank_canvas,
                thresh=heatmap_args.binary_thresh if heatmap_args.binarize else -1,
                patch_size=vis_patch_size,
                convert_to_percentiles=False
            )
            #heatmap_block.save(os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.png'))
            blockmap_path = os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.jpg')
            heatmap_block = heatmap_block.convert("RGB")
            heatmap_block.save(blockmap_path, quality=90, optimize=True, progressive=True)
            del heatmap_block
            gc.collect()
        except Exception as e:
            print(f"[WARN] blockmap 可视化失败: {e}")

        # # 5.2 细粒度可视化
        # try:
        #     if os.path.isfile(save_path_fine):
        #         scores_fine, coords_fine = _safe_load_scores_coords(save_path_fine)
        #         if heatmap_args.convert_to_percentiles:
        #             ref_for_fine = scores_block if heatmap_args.use_ref_scores else scores_fine
        #             scores_fine_vis = percentiles_from_ref(scores_fine, ref_for_fine)
        #         else:
        #             scores_fine_vis = scores_fine
        #         heatmap_fine = drawHeatmap(
        #             scores=scores_fine_vis.flatten(),
        #             coords=coords_fine,
        #             slide_path=slide_path,
        #             wsi_object=wsi_object,
        #             cmap=heatmap_args.cmap,
        #             alpha=heatmap_args.alpha,
        #             use_holes=True,
        #             binarize=heatmap_args.binarize,
        #             vis_level=heatmap_args.vis_level,
        #             blank_canvas=heatmap_args.blank_canvas,
        #             thresh=heatmap_args.binary_thresh if heatmap_args.binarize else -1,
        #             patch_size=vis_patch_size,
        #             convert_to_percentiles=False
        #         )

        #         heatmap_save_name = (
        #             f'{slide_id}_{float(patch_args.overlap)}_roi_{int(heatmap_args.use_roi)}_blur_{int(getattr(heatmap_args,"blur",False))}_'
        #             f'rs_{int(heatmap_args.use_ref_scores)}_bc_{int(heatmap_args.blank_canvas)}_a_{float(heatmap_args.alpha)}_'
        #             f'l_{int(heatmap_args.vis_level)}_bi_{int(heatmap_args.binarize)}_{float(heatmap_args.binary_thresh)}.{heatmap_args.save_ext}'
        #         )
        #         if heatmap_args.save_ext.lower() == 'jpg':
        #             #heatmap_fine.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
        #             heatmap_fine = heatmap_fine.convert("RGB")
        #             heatmap_fine.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=90, optimize=True, progressive=True)
        #         else:
        #             heatmap_fine.save(os.path.join(p_slide_save_dir, heatmap_save_name))
        #         del heatmap_fine
        #         gc.collect()
        # except Exception as e:
        #     print(f"[WARN] 细粒度可视化失败: {e}")
        # 5.2 细粒度可视化
        try:
            if heatmap_args.enable_fine_heatmap and os.path.isfile(save_path_fine):
                scores_fine, coords_fine = _safe_load_scores_coords(save_path_fine)

                if heatmap_args.convert_to_percentiles:
                    ref_for_fine = scores_block if heatmap_args.use_ref_scores else scores_fine
                    scores_fine_vis = percentiles_from_ref(scores_fine, ref_for_fine)
                else:
                    scores_fine_vis = scores_fine

                heatmap_fine = drawHeatmap(
                    scores=scores_fine_vis.flatten(),
                    coords=coords_fine,
                    slide_path=slide_path,
                    wsi_object=wsi_object,
                    cmap=heatmap_args.cmap,
                    alpha=heatmap_args.alpha,
                    use_holes=True,
                    binarize=heatmap_args.binarize,
                    vis_level=heatmap_args.vis_level,
                    blank_canvas=heatmap_args.blank_canvas,
                    thresh=heatmap_args.binary_thresh if heatmap_args.binarize else -1,
                    patch_size=vis_patch_size,
                    convert_to_percentiles=False
                )

                heatmap_save_name = (
                    f'{slide_id}_{float(patch_args.overlap)}_roi_{int(heatmap_args.use_roi)}_'
                    f'blur_{int(getattr(heatmap_args, "blur", False))}_'
                    f'rs_{int(heatmap_args.use_ref_scores)}_'
                    f'bc_{int(heatmap_args.blank_canvas)}_'
                    f'a_{float(heatmap_args.alpha)}_'
                    f'l_{int(heatmap_args.vis_level)}_'
                    f'bi_{int(heatmap_args.binarize)}_'
                    f'{float(heatmap_args.binary_thresh)}.{heatmap_args.save_ext}'
                )

                fine_heatmap_path = os.path.join(p_slide_save_dir, heatmap_save_name)

                if heatmap_args.save_ext.lower() == 'jpg':
                    heatmap_fine = heatmap_fine.convert("RGB")
                    heatmap_fine.save(
                        fine_heatmap_path,
                        quality=90,
                        optimize=True,
                        progressive=True
                    )
                else:
                    heatmap_fine.save(fine_heatmap_path)

                del heatmap_fine
                gc.collect()

            else:
                print("[INFO] fine heatmap visualization skipped.")

        except Exception as e:
            print(f"[WARN] 细粒度可视化失败: {e}")


        # 6) 保存原始视图
        if heatmap_args.save_orig:
            try:
                vis_level_final = heatmap_args.vis_level if heatmap_args.vis_level >= 0 else vis_params['vis_level']
                heatmap_save_name = f'{slide_id}_orig_{int(vis_level_final)}.{heatmap_args.save_ext}'
                if not os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                    heatmap_orig = wsi_object.visWSI(vis_level=vis_level_final, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                    if heatmap_args.save_ext.lower() == 'jpg':
                        heatmap_orig.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=90, optimize=True, progressive=True)
                    else:
                        heatmap_orig.save(os.path.join(p_slide_save_dir, heatmap_save_name))
                    del heatmap_orig
                    gc.collect()
            except Exception as e:
                print(f"[WARN] 保存原始视图失败: {e}")

    # 保存 config
    os.makedirs(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code), exist_ok=True)
    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
