# vis_utils/heatmap_utils.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from models.model_clam import CLAM_MB
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from dataset_modules.wsi_dataset import Wsi_Region
from utils.transform_utils import get_eval_transforms
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from utils.constants import MODEL2CONSTANTS
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    """
    Convert a score to its percentile based on reference scores.
    
    Args:
        score (float): The score to convert.
        ref (array-like): Reference scores to compute the percentile.
        
    Returns:
        float: Percentile of the score.
    """
    percentile = percentileofscore(ref, score)
    return percentile

# Cache for linear adjustment layers
linear_adjust_cache = {}

def get_linear_adjust_layer(current_dim, target_dim, device):
    """
    Retrieve a cached linear layer for adjusting feature dimensions.
    If not cached, create, initialize, cache, and return it.
    
    Args:
        current_dim (int): Current feature dimension.
        target_dim (int): Target feature dimension.
        device (torch.device): Device to place the linear layer on.
        
    Returns:
        nn.Linear: Linear layer adjusting from current_dim to target_dim.
    """
    if (current_dim, target_dim) in linear_adjust_cache:
        return linear_adjust_cache[(current_dim, target_dim)]
    
    print(f"Creating new linear layer for adjusting features from {current_dim} to {target_dim}")
    linear_adjust = nn.Linear(current_dim, target_dim).to(device)
    nn.init.xavier_uniform_(linear_adjust.weight)
    if linear_adjust.bias is not None:
        nn.init.zeros_(linear_adjust.bias)
    linear_adjust_cache[(current_dim, target_dim)] = linear_adjust
    return linear_adjust

def adjust_features(features, target_dim):
    """
    Adjust the features to match the target embedding dimension using a linear layer.
    Utilizes a cache to reuse existing layers for specific dimension adjustments.
    
    Args:
        features (torch.Tensor): Input features tensor.
        target_dim (int): Target embedding dimension.
        
    Returns:
        torch.Tensor: Adjusted features tensor.
    """
    current_dim = features.shape[-1]
    print(f"Adjusting features from {current_dim} to {target_dim}")
    if current_dim != target_dim:
        linear_adjust = get_linear_adjust_layer(current_dim, target_dim, features.device)
        features = linear_adjust(features)
        print(f"Adjusted features shape: {features.shape}")
    else:
        print("No adjustment needed for features.")
    return features

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, **kwargs):
    """
    Draw a heatmap based on the scores and coordinates.
    
    Args:
        scores (array-like): Attention scores for patches.
        coords (array-like): Coordinates of patches.
        slide_path (str, optional): Path to the slide file.
        wsi_object (WholeSlideImage, optional): WSI object.
        vis_level (int, optional): Visualization level.
        **kwargs: Additional arguments for visHeatmap.
        
    Returns:
        Image: Heatmap image.
    """
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    """
    Initialize a Whole Slide Image (WSI) object with segmentation.
    
    Args:
        wsi_path (str): Path to the WSI file.
        seg_mask_path (str, optional): Path to save segmentation mask.
        seg_params (dict, optional): Segmentation parameters.
        filter_params (dict, optional): Filtering parameters.
        
    Returns:
        WholeSlideImage: Initialized WSI object.
    """
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def compute_from_patches(
    wsi_object, 
    img_transforms, 
    feature_extractor=None, 
    clam_pred=None, 
    model=None, 
    batch_size=512,  
    attn_save_path=None, 
    ref_scores=None, 
    feat_save_path=None, 
    embed_dim=1024, 
    **wsi_kwargs
):    
    """
    Compute features and attention scores from patches of the WSI.
    Adjusts feature dimensions dynamically based on embed_dim.
    
    Args:
        wsi_object (WholeSlideImage): The WSI object.
        img_transforms (callable): Image transformations to apply.
        feature_extractor (torch.nn.Module, optional): Feature extractor model.
        clam_pred (int, optional): CLAM prediction.
        model (torch.nn.Module, optional): CLAM model.
        batch_size (int, optional): Batch size for processing.
        attn_save_path (str, optional): Path to save attention scores.
        ref_scores (array-like, optional): Reference scores for percentile computation.
        feat_save_path (str, optional): Path to save features.
        embed_dim (int, optional): Target embedding dimension.
        **wsi_kwargs: Additional keyword arguments for Wsi_Region.
        
    Returns:
        tuple: (attn_save_path, feat_save_path, wsi_object)
    """
    top_left = wsi_kwargs.get('top_left', None)
    bot_right = wsi_kwargs.get('bot_right', None)
    patch_size = wsi_kwargs.get('patch_size', None)
    attention_only = wsi_kwargs.get('attention_only', False)
    
        # --- 开始修改 ---
    # 将 'patch_level' 重命名为 'level'，因为 Wsi_Region 的 __init__ 很可能期望 'level' 参数
    if 'patch_level' in wsi_kwargs:
        print(f"DEBUG: 将 wsi_kwargs 中的 'patch_level' ({wsi_kwargs['patch_level']}) 重命名为 'level'。")
        wsi_kwargs['level'] = wsi_kwargs.pop('patch_level')
    # --- 结束修改 ---

    roi_dataset = Wsi_Region(wsi_object, t=img_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=0)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"

    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.inference_mode():
            features = feature_extractor(roi)
            print(f"Features shape before adjustment: {features.shape}")
            
            # Adjust features to target_dim
            features = adjust_features(features, target_dim=embed_dim)
            print(f"Features shape after adjustment: {features.shape}")

            # If attention_only is True, ensure features have three dimensions
            if attention_only:
                if features.dim() == 2:
                    features = features.unsqueeze(1)  # (B, 1, D)
                    print(f"Features reshaped for attention: {features.shape}")
                elif features.dim() == 3:
                    print("Features already have batch and instance dimensions.")
                else:
                    raise ValueError(f"Unexpected feature dimensions: {features.dim()}")

            if attn_save_path is not None:
                A = model(features, attention_only=True)

                if isinstance(model, CLAM_MB) and A.size(-1) > 1:  # CLAM multi-branch attention
                    A = A[:, :, clam_pred]

                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    # Vectorize the percentile computation
                    A_percentiles = np.array([score2percentile(a, ref_scores) for a in A[:,0]])
                    A = A_percentiles.reshape(-1, 1)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)

        if feat_save_path is not None:
            # Remove batch dimension before saving if present
            if features.dim() == 3 and features.size(0) == 1:
                features_to_save = features.squeeze(0).cpu().numpy()  # Shape (N, D)
            else:
                features_to_save = features.cpu().numpy()
            asset_dict = {'features': features_to_save, 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        print(f"Processed batch {idx+1}/{num_batches}")
        mode = "a"
    return attn_save_path, feat_save_path, wsi_object
