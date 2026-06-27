# # builder.py
# import os
# from functools import partial
# import timm
# from .timm_wrapper import TimmCNNEncoder
# import torch
# import torch.nn as nn
# from utils.constants import MODEL2CONSTANTS
# from utils.transform_utils import get_eval_transforms


# def has_CONCH():
#     HAS_CONCH = False
#     CONCH_CKPT_PATH = ''
#     try:
#         from conch.open_clip_custom import create_model_from_pretrained
#         if 'CONCH_CKPT_PATH' not in os.environ:
#             raise ValueError('CONCH_CKPT_PATH not set')
#         HAS_CONCH = True
#         CONCH_CKPT_PATH = os.environ['CONCH_CKPT_PATH']
#     except Exception as e:
#         print(e)
#         print('CONCH not installed or CONCH_CKPT_PATH not set')
#     return HAS_CONCH, CONCH_CKPT_PATH


# def has_UNI():
#     HAS_UNI = False
#     UNI_CKPT_PATH = ''
#     try:
#         if 'UNI_CKPT_PATH' not in os.environ:
#             raise ValueError('UNI_CKPT_PATH not set')
#         HAS_UNI = True
#         UNI_CKPT_PATH = os.environ['UNI_CKPT_PATH']
#     except Exception as e:
#         print(e)
#     return HAS_UNI, UNI_CKPT_PATH


# def has_CHIEF():
#     HAS_CHIEF = False
#     CHIEF_CKPT_PATH = ''
#     try:
#         if 'CHIEF_CKPT_PATH' not in os.environ:
#             raise ValueError('CHIEF_CKPT_PATH not set')
#         CHIEF_CKPT_PATH = os.environ['CHIEF_CKPT_PATH']
#         if not os.path.isfile(CHIEF_CKPT_PATH):
#             raise FileNotFoundError(f'CHIEF checkpoint not found: {CHIEF_CKPT_PATH}')
#         HAS_CHIEF = True
#     except Exception as e:
#         print(e)
#     return HAS_CHIEF, CHIEF_CKPT_PATH


# def _read_state_dict(ckpt_path):
#     ext = os.path.splitext(ckpt_path)[1].lower()

#     if ext == '.safetensors':
#         from safetensors.torch import load_file as safe_load_file
#         raw = safe_load_file(ckpt_path)
#     else:
#         raw = torch.load(ckpt_path, map_location='cpu')

#     if not isinstance(raw, dict):
#         raise RuntimeError(f'Unsupported checkpoint format: {type(raw)}')

#     if 'state_dict' in raw and isinstance(raw['state_dict'], dict):
#         state_dict = raw['state_dict']
#     elif 'model' in raw and isinstance(raw['model'], dict):
#         state_dict = raw['model']
#     else:
#         state_dict = raw

#     cleaned = {}
#     for k, v in state_dict.items():
#         nk = k
#         if nk.startswith('module.'):
#             nk = nk[len('module.'):]
#         if nk.startswith('model.'):
#             nk = nk[len('model.'):]
#         # 注意：这里不要去掉 backbone./encoder. 前缀
#         cleaned[nk] = v

#     return cleaned

# def _infer_swin_name_from_ckpt(cleaned_state_dict):
#     # 先用最稳的 norm 判断（ConvStem/普通patch_embed都适用）
#     if 'patch_embed.norm.weight' in cleaned_state_dict:
#         embed_dim = int(cleaned_state_dict['patch_embed.norm.weight'].shape[0])
#         src_key = 'patch_embed.norm.weight'
#     else:
#         # 兜底：普通Swin的 patch_embed.proj.weight
#         proj_key = None
#         for k in cleaned_state_dict.keys():
#             if k.endswith('patch_embed.proj.weight'):
#                 proj_key = k
#                 break
#         if proj_key is None:
#             sample = list(cleaned_state_dict.keys())[:50]
#             raise RuntimeError(
#                 'Cannot infer Swin embed_dim from checkpoint. '
#                 f'Sample keys: {sample}'
#             )
#         embed_dim = int(cleaned_state_dict[proj_key].shape[0])
#         src_key = proj_key

#     mapping = {
#         96: 'swin_small_patch4_window7_224',
#         128: 'swin_base_patch4_window7_224',
#         192: 'swin_large_patch4_window7_224',
#     }
#     if embed_dim not in mapping:
#         raise RuntimeError(f'Unknown Swin embed_dim={embed_dim}, key={src_key}')

#     return mapping[embed_dim], embed_dim

# def _load_chief_model_auto(ckpt_path):
#     cleaned = _read_state_dict(ckpt_path)
#     model_name, embed_dim = _infer_swin_name_from_ckpt(cleaned)

#     print(f'[CHIEF] inferred backbone: {model_name} (embed_dim={embed_dim})')
#     model = timm.create_model(model_name, pretrained=False, num_classes=0)

#     #missing, unexpected = model.load_state_dict(cleaned, strict=False)
#     model_sd = model.state_dict()
#     loadable = {}
#     for k, v in cleaned.items():
#         if k in model_sd and model_sd[k].shape == v.shape:
#             loadable[k] = v

#     missing, unexpected = model.load_state_dict(loadable, strict=False)
#     print(f'[CHIEF] loadable keys: {len(loadable)}/{len(cleaned)}')
#     print(f'[CHIEF] loaded from: {ckpt_path}')
#     print(f'[CHIEF] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
#     if unexpected:
#         print(f'[CHIEF] unexpected sample: {unexpected[:10]}')
#     if missing:
#         print(f'[CHIEF] missing sample: {missing[:10]}')

#     return model


# def get_encoder(model_name, target_img_size=224):
#     print('loading model checkpoint')

#     if model_name == 'resnet50_trunc':
#         model = TimmCNNEncoder()

#     elif model_name == 'uni_v1':
#         HAS_UNI, UNI_CKPT_PATH = has_UNI()
#         assert HAS_UNI, 'UNI is not available'
#         model = timm.create_model(
#             "vit_large_patch16_224",
#             init_values=1e-5,
#             num_classes=0,
#             dynamic_img_size=True
#         )
#         model.load_state_dict(torch.load(UNI_CKPT_PATH, map_location="cpu"), strict=True)

#     elif model_name == 'conch_v1':
#         HAS_CONCH, CONCH_CKPT_PATH = has_CONCH()
#         assert HAS_CONCH, 'CONCH is not available'
#         from conch.open_clip_custom import create_model_from_pretrained
#         model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
#         model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)

#     # elif model_name == 'chief':
#     #     HAS_CHIEF, CHIEF_CKPT_PATH = has_CHIEF()
#     #     assert HAS_CHIEF, 'CHIEF is not available'
#     #     model = _load_chief_model_auto(CHIEF_CKPT_PATH)
#     elif model_name == 'chief':
#         HAS_CHIEF, CHIEF_CKPT_PATH = has_CHIEF()
#         assert HAS_CHIEF, (
#             "CHIEF is not available. "
#             "Please set CHIEF_CKPT_PATH to CHIEF_CTransPath.pth."
#         )

#         # CHIEF official patch-level encoder
#         try:
#             from .ctran import ctranspath
#         except ImportError as e:
#             raise ImportError(
#                 "Cannot import models.ctran.ctranspath. "
#                 "Please ensure D:/ADGA-main/models/ctran.py exists "
#                 "and comes from the official CHIEF implementation."
#             ) from e

#         print("[CHIEF] Building official CTransPath patch encoder")
#         model = ctranspath()

#         # Remove the classification head to obtain 768-dimensional embeddings
#         model.head = nn.Identity()

#         print(f"[CHIEF] Loading patch encoder checkpoint: {CHIEF_CKPT_PATH}")

#         try:
#             checkpoint = torch.load(
#                 CHIEF_CKPT_PATH,
#                 map_location="cpu",
#                 weights_only=False,
#             )
#         except TypeError:
#             checkpoint = torch.load(
#                 CHIEF_CKPT_PATH,
#                 map_location="cpu",
#             )

#         if not isinstance(checkpoint, dict):
#             raise RuntimeError(
#                 f"Unsupported CHIEF checkpoint type: {type(checkpoint)}"
#             )

#         # Official CHIEF_CTransPath.pth stores weights under checkpoint['model']
#         if "model" in checkpoint and isinstance(checkpoint["model"], dict):
#             state_dict = checkpoint["model"]
#         elif "state_dict" in checkpoint and isinstance(
#             checkpoint["state_dict"], dict
#         ):
#             state_dict = checkpoint["state_dict"]
#         else:
#             state_dict = checkpoint

#         # Remove DataParallel prefix if present
#         cleaned_state_dict = {}
#         for key, value in state_dict.items():
#             new_key = key

#             if new_key.startswith("module."):
#                 new_key = new_key[len("module."):]

#             cleaned_state_dict[new_key] = value

#         try:
#             model.load_state_dict(cleaned_state_dict, strict=True)
#         except RuntimeError as e:
#             raise RuntimeError(
#                 "\nFailed to strictly load CHIEF_CTransPath.pth.\n"
#                 "This usually means that models/ctran.py or the installed timm "
#                 "version is inconsistent with the checkpoint.\n"
#                 f"Checkpoint: {CHIEF_CKPT_PATH}\n"
#                 f"Original error:\n{e}"
#             ) from e

#         print("[CHIEF] Official CTransPath weights loaded successfully")


#     else:
#         raise NotImplementedError(f'model {model_name} not implemented')

#     print(model)

#     if model_name not in MODEL2CONSTANTS:
#         raise KeyError(f'{model_name} not found in MODEL2CONSTANTS')

#     constants = MODEL2CONSTANTS[model_name]
#     img_transforms = get_eval_transforms(
#         mean=constants['mean'],
#         std=constants['std'],
#         target_img_size=target_img_size
#     )

#     return model, img_transforms


# -*- coding: utf-8 -*-
"""
builder.py compatible with CHIEF_CTransPath.pth and timm 1.x.

Place at:
    D:\ADGA-main\models\builder.py
"""

from __future__ import annotations

import os
import re
from functools import partial

import timm
import torch
import torch.nn as nn

from .timm_wrapper import TimmCNNEncoder
from utils.constants import MODEL2CONSTANTS
from utils.transform_utils import get_eval_transforms


def has_CONCH():
    has_conch = False
    conch_ckpt_path = ""
    try:
        from conch.open_clip_custom import create_model_from_pretrained  # noqa: F401
        if "CONCH_CKPT_PATH" not in os.environ:
            raise ValueError("CONCH_CKPT_PATH not set")
        has_conch = True
        conch_ckpt_path = os.environ["CONCH_CKPT_PATH"]
    except Exception as exc:
        print(exc)
        print("CONCH not installed or CONCH_CKPT_PATH not set")
    return has_conch, conch_ckpt_path


def has_UNI():
    has_uni = False
    uni_ckpt_path = ""
    try:
        if "UNI_CKPT_PATH" not in os.environ:
            raise ValueError("UNI_CKPT_PATH not set")
        has_uni = True
        uni_ckpt_path = os.environ["UNI_CKPT_PATH"]
    except Exception as exc:
        print(exc)
    return has_uni, uni_ckpt_path


def has_CHIEF():
    has_chief = False
    chief_ckpt_path = ""
    try:
        if "CHIEF_CKPT_PATH" not in os.environ:
            raise ValueError("CHIEF_CKPT_PATH not set")
        chief_ckpt_path = os.environ["CHIEF_CKPT_PATH"]
        if not os.path.isfile(chief_ckpt_path):
            raise FileNotFoundError(
                f"CHIEF checkpoint not found: {chief_ckpt_path}"
            )
        has_chief = True
    except Exception as exc:
        print(exc)
    return has_chief, chief_ckpt_path


def _load_torch_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(raw):
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {type(raw)}")

    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state_dict = raw["state_dict"]
    elif "model" in raw and isinstance(raw["model"], dict):
        state_dict = raw["model"]
    else:
        state_dict = raw

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value
    return cleaned


def _convert_chief_timm05_to_timm1(state_dict, model):
    model_state = model.state_dict()
    converted = {}
    remapped = []
    dropped_buffers = []
    incompatible = []

    for old_key, tensor in state_dict.items():
        key = old_key

        if key.endswith("relative_position_index") or key.endswith("attn_mask"):
            dropped_buffers.append(key)
            continue

        match = re.match(r"^layers\.(\d+)\.downsample\.(.+)$", key)
        if match:
            old_stage = int(match.group(1))
            suffix = match.group(2)
            candidate = f"layers.{old_stage + 1}.downsample.{suffix}"
            if candidate in model_state:
                remapped.append((key, candidate))
                key = candidate

        if key not in model_state:
            incompatible.append(
                (old_key, key, tuple(tensor.shape), "key_not_in_model")
            )
            continue

        expected_shape = tuple(model_state[key].shape)
        actual_shape = tuple(tensor.shape)
        if actual_shape != expected_shape:
            incompatible.append(
                (old_key, key, actual_shape, f"expected_{expected_shape}")
            )
            continue

        if key in converted:
            raise RuntimeError(
                f"Duplicate converted key: {key}; source key={old_key}"
            )

        converted[key] = tensor

    missing = sorted(set(model_state.keys()) - set(converted.keys()))

    print(
        f"[CHIEF] state conversion: loaded={len(converted)}, "
        f"downsample_remapped={len(remapped)}, "
        f"old_buffers_dropped={len(dropped_buffers)}, "
        f"incompatible={len(incompatible)}, missing={len(missing)}"
    )

    if remapped:
        print("[CHIEF] downsample remap examples:")
        for old_key, new_key in remapped[:6]:
            print(f"  {old_key} -> {new_key}")

    if dropped_buffers:
        print(
            "[CHIEF] dropped old generated buffers, examples: "
            f"{dropped_buffers[:6]}"
        )

    if incompatible or missing:
        details = []
        if incompatible:
            details.append(
                "Incompatible checkpoint entries:\n"
                + "\n".join(
                    f"  source={old!r}, converted={new!r}, "
                    f"shape={shape}, reason={reason}"
                    for old, new, shape, reason in incompatible[:30]
                )
            )
        if missing:
            details.append(
                "Missing model entries:\n"
                + "\n".join(f"  {key}" for key in missing[:30])
            )

        raise RuntimeError(
            "CHIEF checkpoint conversion was not complete; refusing to load "
            "partial weights.\n" + "\n".join(details)
        )

    return converted


def _load_chief_encoder(chief_ckpt_path):
    try:
        from .ctran import ctranspath
    except ImportError as exc:
        raise ImportError(
            "Cannot import models.ctran.ctranspath. "
            "Ensure D:/ADGA-main/models/ctran.py exists."
        ) from exc

    print("[CHIEF] Building CTransPath patch encoder")
    print(f"[CHIEF] timm version: {timm.__version__}")

    model = ctranspath()

    # Important for timm 1.x:
    # reset_classifier(0) removes the classifier but preserves global pooling,
    # so model(x) returns B x 768 instead of B x H x W x 768.
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(0)
    else:
        model.head = nn.Identity()

    print(f"[CHIEF] Loading checkpoint: {chief_ckpt_path}")
    raw = _load_torch_checkpoint(chief_ckpt_path)
    state_dict = _extract_state_dict(raw)

    try:
        timm_major = int(timm.__version__.split(".")[0])
    except Exception:
        timm_major = 0

    if timm_major >= 1:
        state_dict = _convert_chief_timm05_to_timm1(state_dict, model)

    model.load_state_dict(state_dict, strict=True)
    print("[CHIEF] CTransPath weights loaded strictly and successfully")
    return model


def get_encoder(model_name, target_img_size=224):
    print("loading model checkpoint")

    if model_name == "resnet50_trunc":
        model = TimmCNNEncoder()

    elif model_name == "uni_v1":
        has_uni, uni_ckpt_path = has_UNI()
        assert has_uni, "UNI is not available"
        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(
            torch.load(uni_ckpt_path, map_location="cpu"),
            strict=True,
        )

    elif model_name == "conch_v1":
        has_conch, conch_ckpt_path = has_CONCH()
        assert has_conch, "CONCH is not available"
        from conch.open_clip_custom import create_model_from_pretrained

        model, _ = create_model_from_pretrained(
            "conch_ViT-B-16",
            conch_ckpt_path,
        )
        model.forward = partial(
            model.encode_image,
            proj_contrast=False,
            normalize=False,
        )

    elif model_name == "chief":
        has_chief, chief_ckpt_path = has_CHIEF()
        assert has_chief, (
            "CHIEF is not available. Set CHIEF_CKPT_PATH to "
            "CHIEF_CTransPath.pth."
        )
        model = _load_chief_encoder(chief_ckpt_path)

    else:
        raise NotImplementedError(f"model {model_name} not implemented")

    print(model)

    if model_name not in MODEL2CONSTANTS:
        raise KeyError(f"{model_name} not found in MODEL2CONSTANTS")

    constants = MODEL2CONSTANTS[model_name]
    img_transforms = get_eval_transforms(
        mean=constants["mean"],
        std=constants["std"],
        target_img_size=target_img_size,
    )

    return model, img_transforms


