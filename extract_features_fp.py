import time
import os
import argparse
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import h5py
import openslide
from tqdm import tqdm
import numpy as np

from utils.file_utils import save_hdf5
from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder


torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def compute_w_loader(output_path, loader, model, verbose=0):
#     """
#     Args:
#         output_path: directory to save computed features (.h5 file)
#         loader: torch DataLoader
#         model: pytorch model
#         verbose: level of feedback
#     """
#     if verbose > 0:
#         print(f"processing a total of {len(loader)} batches")

#     mode = "w"

#     for count, data in enumerate(tqdm(loader)):
#         with torch.inference_mode():
#             batch = data["img"]
#             coords = data["coord"].numpy().astype(np.int32)

#             batch = batch.to(device, non_blocking=True)
#             features = model(batch)
#             features = features.cpu().numpy().astype(np.float32)

#             asset_dict = {
#                 "features": features,
#                 "coords": coords,
#             }

#             save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
#             mode = "a"

#     return output_path

def compute_w_loader(output_path, loader, model, verbose=0):
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches')

    all_features = []
    all_coords = []

    for count, data in enumerate(tqdm(loader, dynamic_ncols=True, mininterval=1.0)):
        with torch.inference_mode():
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                features = model(batch)

            features = features.cpu().numpy().astype(np.float32)

            all_features.append(features)
            all_coords.append(coords)

    all_features = np.concatenate(all_features, axis=0)
    all_coords = np.concatenate(all_coords, axis=0)

    asset_dict = {'features': all_features, 'coords': all_coords}
    save_hdf5(output_path, asset_dict, attr_dict=None, mode='w')

    return output_path, all_features, all_coords



parser = argparse.ArgumentParser(description="Feature Extraction")
parser.add_argument("--data_h5_dir", type=str, default=None)
parser.add_argument("--data_slide_dir", type=str, default=None)
parser.add_argument("--slide_ext", type=str, default=".svs")
parser.add_argument("--csv_path", type=str, default=None)
parser.add_argument("--feat_dir", type=str, default=None)

parser.add_argument(
    "--model_name",
    type=str,
    default="resnet50_trunc",
    choices=["resnet50_trunc", "uni_v1", "conch_v1", "prov_gigapath", "chief"],
)

parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--no_auto_skip", default=False, action="store_true")
parser.add_argument("--target_patch_size", type=int, default=224)

# 本地模型训练
parser.add_argument("--local_weight_path", type=str, default=None)

args = parser.parse_args()


if __name__ == "__main__":
    print("initializing dataset")

    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError("csv_path must be provided")

    bags_dataset = Dataset_All_Bags(csv_path)

    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "pt_files"), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, "h5_files"), exist_ok=True)

    dest_files = os.listdir(os.path.join(args.feat_dir, "pt_files"))

    if args.model_name == "prov_gigapath":
        print("Loading Prov-GigaPath from HuggingFace...")

        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath",
            pretrained=True,
        )

        img_transforms = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    elif args.model_name == "uni_v1":
        print("Loading UNI from local weights...")

        if args.local_weight_path is None:
            raise ValueError("For uni_v1, --local_weight_path must be provided")

        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )

        state_dict = torch.load(args.local_weight_path, map_location="cpu")

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v

        msg = model.load_state_dict(new_state_dict, strict=False)
        print("load_state_dict msg:", msg)

        img_transforms = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    elif args.model_name == "chief":
        print("Loading CHIEF (CTransPath) from HuggingFace/cache...")

        from timm.layers.helpers import to_2tuple

        class ConvStem(nn.Module):
            def __init__(
                self,
                img_size=224,
                patch_size=4,
                in_chans=3,
                embed_dim=768,
                norm_layer=None,
                **kwargs,
            ):
                super().__init__()

                assert patch_size == 4, "Patch size must be 4"
                assert embed_dim % 8 == 0, "Embedding dimension must be a multiple of 8"

                img_size = to_2tuple(img_size)
                patch_size = to_2tuple(patch_size)

                self.img_size = img_size
                self.patch_size = patch_size
                self.grid_size = (
                    img_size[0] // patch_size[0],
                    img_size[1] // patch_size[1],
                )
                self.num_patches = self.grid_size[0] * self.grid_size[1]

                stem = []
                input_dim, output_dim = 3, embed_dim // 8

                for _ in range(2):
                    stem.append(
                        nn.Conv2d(
                            input_dim,
                            output_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                            bias=False,
                        )
                    )
                    stem.append(nn.BatchNorm2d(output_dim))
                    stem.append(nn.ReLU(inplace=True))
                    input_dim = output_dim
                    output_dim *= 2

                stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))

                self.proj = nn.Sequential(*stem)
                self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

            def forward(self, x):
                B, C, H, W = x.shape
                assert H == self.img_size[0] and W == self.img_size[1], (
                    f"Input image size ({H}*{W}) doesn't match model "
                    f"({self.img_size[0]}*{self.img_size[1]})."
                )

                x = self.proj(x)
                x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
                x = self.norm(x)
                return x

        model = timm.create_model(
            model_name="hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
            embed_layer=ConvStem,
            pretrained=True,
        )
        model.reset_classifier(0)

        img_transforms = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    else:
        model, img_transforms = get_encoder(
            args.model_name,
            target_img_size=args.target_patch_size,
        )

    model = model.eval()
    model = model.to(device)

    total = len(bags_dataset)
    loader_kwargs = {"num_workers": 0, "pin_memory": True} if device.type == "cuda" else {}

    for bag_candidate_idx in tqdm(range(total)):
        slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
        bag_name = slide_id + ".h5"

        h5_file_path = os.path.join(args.data_h5_dir, "patches", bag_name)
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)

        if not os.path.isfile(h5_file_path):
            print(f"missing patch file: {h5_file_path}, skipped")
            continue

        print("\nprogress: {}/{}".format(bag_candidate_idx, total))
        print(slide_id)

        if not args.no_auto_skip and slide_id + ".pt" in dest_files:
            print("skipped {}".format(slide_id))
            continue

        output_path = os.path.join(args.feat_dir, "h5_files", bag_name)

        time_start = time.time()

        wsi = openslide.open_slide(slide_file_path)
        dataset = Whole_Slide_Bag_FP(
            file_path=h5_file_path,
            wsi=wsi,
            img_transforms=img_transforms,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            **loader_kwargs,
        )

        # output_file_path = compute_w_loader(
        #     output_path,
        #     loader=loader,
        #     model=model,
        #     verbose=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\ncomputing features for {} took {} s".format(output_file_path, time_elapsed))

        # with h5py.File(output_file_path, "r") as file:
        #     features = file["features"][:]
        #     print("features size: ", features.shape)
        #     print("coordinates size: ", file["coords"].shape)

        output_file_path, features_np, coords_np = compute_w_loader(
            output_path,
            loader=loader,
            model=model,
            verbose=1,
        )

        time_elapsed = time.time() - time_start
        print("\ncomputing features for {} took {} s".format(output_file_path, time_elapsed))
        print("features size: ", features_np.shape)
        print("coordinates size: ", coords_np.shape)

        features = torch.from_numpy(features_np)

		
        
        bag_base, _ = os.path.splitext(bag_name)
        torch.save(features, os.path.join(args.feat_dir, "pt_files", bag_base + ".pt"))

        del loader
        del dataset
        del wsi