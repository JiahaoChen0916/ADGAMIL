# utils/eval_utils.py
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# def initiate_model(args, ckpt_path, device='cuda'):
#     print('Init Model')    
#     model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "embed_dim": args.embed_dim}
    
#     if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
#         model_dict.update({"size_arg": args.model_size})
    
#     if args.model_type =='clam_sb':
#         model = CLAM_SB(**model_dict)
#     elif args.model_type =='clam_mb':
#         model = CLAM_MB(**model_dict)
#     else: # args.model_type == 'mil'
#         if args.n_classes > 2:
#             model = MIL_fc_mc(**model_dict)
#         else:
#             model = MIL_fc(**model_dict)

#     print_network(model)

#     ckpt = torch.load(ckpt_path)
#     ckpt_clean = {}
#     for key in ckpt.keys():
#         if 'instance_loss_fn' in key:
#             continue
#         ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
#     model.load_state_dict(ckpt_clean, strict=True)

#     _ = model.to(device)
#     _ = model.eval()
#     return model
def initiate_model(args, ckpt_path, device=None):
    print("Init Model")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    elif isinstance(device, str):
        device = torch.device(device)

    model_dict = {
        "gate": not getattr(args, "no_inst_cluster", False),
        "size_arg": getattr(args, "model_size", "small"),
        "dropout": float(args.drop_out),
        "k_sample": int(getattr(args, "B", 8)),
        "n_classes": int(args.n_classes),
        "instance_loss_fn": get_loss_function(
            getattr(args, "inst_loss", "svm")
        ),
        "subtyping": bool(getattr(args, "subtyping", True)),
        "embed_dim": int(args.embed_dim),
        "num_neighbors": int(
            getattr(args, "num_neighbors", 5)
        ),
    }

    if args.model_type == "clam_sb":
        model = CLAM_SB(**model_dict)
    elif args.model_type == "clam_mb":
        model = CLAM_MB(**model_dict)
    else:
        if args.n_classes > 2:
            model = MIL_fc_mc(
                dropout=args.drop_out,
                n_classes=args.n_classes,
                embed_dim=args.embed_dim,
            )
        else:
            model = MIL_fc(
                dropout=args.drop_out,
                n_classes=args.n_classes,
                embed_dim=args.embed_dim,
            )

    print_network(model)

    try:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
        )

    ckpt_clean = {}

    for key, value in ckpt.items():
        if "instance_loss_fn" in key:
            continue

        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        ckpt_clean[new_key] = value

    model.load_state_dict(
        ckpt_clean,
        strict=True,
    )

    model = model.to(device)
    model.eval()

    print(
        "[MODEL CONFIG] "
        f"dropout={model_dict['dropout']}, "
        f"num_neighbors={model_dict['num_neighbors']}, "
        f"k_sample={model_dict['k_sample']}, "
        f"subtyping={model_dict['subtyping']}"
    )

    return model


def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger
