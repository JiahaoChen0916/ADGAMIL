# main_uni_chief.py
from __future__ import print_function
import argparse
import pdb
import os
import math
import time

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.utils import collate_MIL_padded
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import logging
import sys

# Import CLAM model
from models.model_clam import CLAM_SB, CLAM_MB

# Import SummaryWriter for TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Import sklearn metrics for additional evaluation
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode='w')
    ]
)

logger = logging.getLogger('main')
logging.getLogger('dataset_modules.dataset_generic').setLevel(logging.ERROR)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {seconds}s"

def check_label_distribution(dataset, split_name):
    labels = dataset.slide_data['label'].astype(int).tolist()
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique, counts))
    logger.info(f"{split_name} Label Distribution: {label_dist}")


def check_split_overlap(train_dataset, val_dataset, test_dataset, fold, args):
    split_dir_name = os.path.basename(args.split_dir).upper()

    use_site_level = ('NSCLC' in args.task) and ('SITE' in split_dir_name)

    if use_site_level:
        train_groups = set(train_dataset.slide_data['site_id'].astype(str).tolist())
        val_groups   = set(val_dataset.slide_data['site_id'].astype(str).tolist())
        test_groups  = set(test_dataset.slide_data['site_id'].astype(str).tolist())
        group_name = 'site'
    else:
        train_groups = set(train_dataset.slide_data['case_id'].astype(str).tolist())
        val_groups   = set(val_dataset.slide_data['case_id'].astype(str).tolist())
        test_groups  = set(test_dataset.slide_data['case_id'].astype(str).tolist())
        group_name = 'patient'

    logger.info(
        f"Fold [{fold}] True {group_name}-level Overlap Check | "
        f"train∩val={len(train_groups & val_groups)}, "
        f"train∩test={len(train_groups & test_groups)}, "
        f"val∩test={len(val_groups & test_groups)}"
    )

    if len(train_groups & val_groups) > 0:
        raise RuntimeError(f"Fold [{fold}] Leakage: train∩val {group_name}s")
    if len(train_groups & test_groups) > 0:
        raise RuntimeError(f"Fold [{fold}] Leakage: train∩test {group_name}s")
    if len(val_groups & test_groups) > 0:
        raise RuntimeError(f"Fold [{fold}] Leakage: val∩test {group_name}s")





def main(args, dataset):
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
        
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    all_test_recall = []
    all_val_recall = []
    all_test_f1 = []
    all_val_f1 = []
    all_test_specificity = []
    all_val_specificity = []
    all_test_qwk = []
    all_val_qwk = []
    folds = np.arange(start, end)
    
    # Lists to store true labels and predicted probabilities for ROC/PR curves
    all_val_labels = []
    all_val_preds = []
    all_test_labels = []
    all_test_preds = []

    if args.log_data:
        tensorboard_root = os.path.join(args.results_dir, 'tensorboard')
        os.makedirs(tensorboard_root, exist_ok=True)
    else:
        tensorboard_root = None
    
    for fold in folds:
        if args.log_data:
            fold_tb_dir = os.path.join(tensorboard_root, f'fold_{fold}')
            os.makedirs(fold_tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=fold_tb_dir)
            logger.info(f"Initialized TensorBoard SummaryWriter at {fold_tb_dir}")
        else:
            writer = None

        logger.info(f"Starting Fold [{fold}]")
        seed_torch(args.seed + fold)
        split_path = os.path.join(args.split_dir, f"splits_{fold}.csv")
        if not os.path.isfile(split_path):
            logger.error(f"Split file not found: {split_path}")
            raise FileNotFoundError(f"Split file not found: {split_path}")

        train_dataset, val_dataset, test_dataset = dataset.return_splits(
            from_id=False, 
            csv_path=split_path
        )
                # --- 添加以下检查代码 ---
        if train_dataset is None or val_dataset is None or test_dataset is None:
            logger.error(f"Error: One of the datasets is None! Split path: {split_path}")
            # 打印一下 dataset 里的前几个 ID 和 CSV 里的前几个 ID 看看是否匹配
            logger.error(f"Dataset summary: {len(dataset)} samples")
            raise RuntimeError("Failed to create splits. Check if slide_ids in CSV match the dataset.")
        # -----------------------

        check_split_overlap(train_dataset, val_dataset, test_dataset, fold, args)


                # ========= 新增：每个fold单独初始化模型/优化器/scaler =========
        if args.model_type == 'clam_sb':
            clam_model = CLAM_SB(
                gate=not args.no_inst_cluster,
                size_arg=args.model_size,
                dropout=args.drop_out,
                k_sample=args.B,
                n_classes=args.n_classes,
                instance_loss_fn=get_loss_function(args.inst_loss),
                subtyping=args.subtyping,
                embed_dim=args.embed_dim,
                num_neighbors=args.num_neighbors
            )
        elif args.model_type == 'clam_mb':
            clam_model = CLAM_MB(
                gate=not args.no_inst_cluster,
                size_arg=args.model_size,
                dropout=args.drop_out,
                k_sample=args.B,
                n_classes=args.n_classes,
                instance_loss_fn=get_loss_function(args.inst_loss),
                subtyping=args.subtyping,
                embed_dim=args.embed_dim,
                num_neighbors=args.num_neighbors
            )
        else:
            raise NotImplementedError(f"Model type '{args.model_type}' is not implemented.")

        clam_model.to(device)
        optimizer = get_optim(clam_model, args)
        #scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        # ============================================================     

        if args.cache_in_ram:
            logger.info(f"Fold [{fold}] - Preloading train/val/test features into RAM ...")
            if hasattr(train_dataset, 'preload_features'):
                train_dataset.preload_features()
            if hasattr(val_dataset, 'preload_features'):
                val_dataset.preload_features()
            if hasattr(test_dataset, 'preload_features'):
                test_dataset.preload_features()
            logger.info(f"Fold [{fold}] - RAM cache preload finished")


        
        # 验证标签分布
        check_label_distribution(train_dataset, f"Fold [{fold}] Train")
        check_label_distribution(val_dataset, f"Fold [{fold}] Val")
        check_label_distribution(test_dataset, f"Fold [{fold}] Test")
        
        # # ===== sanity check: shuffle training labels =====
        # shuffled = np.random.permutation(train_dataset.slide_data['label'].values)
        # train_dataset.slide_data.loc[:, 'label'] = shuffled
        # logger.info(f"Fold [{fold}] Training labels have been shuffled for sanity check.")

        
        labels = train_dataset.slide_data['label'].astype(int).tolist()
        class_counts = np.bincount(labels, minlength=args.n_classes)


        class_weights = 1.0 / np.maximum(class_counts, 1)
        class_weights = class_weights / class_weights.sum() * args.n_classes
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        logger.info(f"Fold [{fold}] - Class counts: {class_counts.tolist()}")
        logger.info(f"Fold [{fold}] - Class weights: {class_weights.tolist()}")

        loader_kwargs = {
            'batch_size': 1,
            'collate_fn': collate_MIL_padded,
            'num_workers': args.num_workers,
            'pin_memory': args.pin_memory and (device.type == 'cuda'),
        }

        if args.num_workers > 0:
            loader_kwargs['persistent_workers'] = args.persistent_workers
            loader_kwargs['prefetch_factor'] = args.prefetch_factor


        # 修改2: 优化加权采样，反转权重以增加正类采样概率       
        if args.weighted_sample:
            weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float)
            sample_weights = weights[torch.tensor(labels, dtype=torch.long)]
            sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, len(sample_weights), replacement=True
            )

            train_loader = DataLoader(
                train_dataset,
                sampler=sampler,
                shuffle=False,
                **loader_kwargs
            )
            logger.info(f"Fold [{fold}] - Weighted sampling enabled with class counts: {dict(enumerate(class_counts))}")
        else:
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                **loader_kwargs
            )
            logger.info(f"Fold [{fold}] - Weighted sampling disabled, using shuffle")

        val_loader = DataLoader(val_dataset,shuffle=False,**loader_kwargs)
        test_loader = DataLoader(test_dataset,shuffle=False,**loader_kwargs)

        
        logger.info(f"Fold [{fold}] - Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
        if 'PANDA' in args.task:
            logger.info(f"Fold [{fold}] - Monitoring metric for model selection: QWK")
        else:
            logger.info(f"Fold [{fold}] - Monitoring metric for model selection: AUC")

        logger.info(f"Fold [{fold}] Sanity check before training...")
        pre_auc, pre_acc, pre_recall, pre_f1, pre_spec, pre_qwk, _, _ = evaluate(
            val_loader, clam_model, device, args.n_classes
        )
        logger.info(
            f"Fold [{fold}] BEFORE TRAIN | "
            f"val_auc={pre_auc:.4f} | val_acc={pre_acc:.4f} | "
            f"val_recall={pre_recall:.4f} | val_f1={pre_f1:.4f} | "
            f"val_spec={pre_spec:.4f} | val_qwk={pre_qwk:.4f}"
        )


        
        # patience = 5
        # min_delta = 0.001
        # counter = 0
        patience = args.patience
        min_delta = args.min_delta
        counter = 0

        best_score = -np.inf
        best_epoch = -1
        best_ckpt_path = os.path.join(args.results_dir, f's_{fold}_best_checkpoint.pt')
        best_metrics = {
            'val_auc': 0.0,
            'val_acc': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_specificity': 0.0,
            'val_qwk': 0.0
        }

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=5, min_lr=1e-6
        )

        epoch_times = []          # 每个fold单独计时
        printed_inst_keys = False # 每个fold只打印一次results_dict keys

        #previous_lr = optimizer.param_groups[0]['lr']
        
        for epoch in range(args.max_epochs):
            epoch_start_time = time.time()
            #logger.info(f"Epoch [{epoch+1}/{args.max_epochs}] Fold [{fold}]")
            clam_model.train()
            
            epoch_loss = 0.0
            for idx, batch in enumerate(train_loader):
                patch_features = batch['features'].to(device)
                slide_labels = batch['labels'].to(device)
                attn_mask = batch['mask'].to(device)

                optimizer.zero_grad(set_to_none=True)

                #with torch.cuda.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    if args.model_type in ['clam_sb', 'clam_mb']:
                        # CLAM 必须开启 instance_eval=True 以计算实例损失
                        outputs, Y_prob, Y_hat, A, results_dict = clam_model(
                            patch_features,
                            label= slide_labels,
                            instance_eval=True, 
                            attn_mask=attn_mask
                        )
                        loss = criterion(outputs, slide_labels)

                        if not printed_inst_keys:
                            if isinstance(results_dict, dict):
                                logger.info(f"[Fold {fold}] results_dict keys = {list(results_dict.keys())}")
                            else:
                                logger.info(f"[Fold {fold}] results_dict type = {type(results_dict)}")
                            printed_inst_keys = True
                        bag_loss = criterion(outputs, slide_labels)
                        # instance_loss = results_dict['instance_loss']
                        # # 联合优化：包级别损失 + 实例级别损失
                        # loss = args.bag_weight * bag_loss + (1 - args.bag_weight) * instance_loss

                        instance_loss = None
                        if isinstance(results_dict, dict):
                            if 'instance_loss' in results_dict:
                                instance_loss = results_dict['instance_loss']
                            elif 'inst_loss' in results_dict:   # 兼容你可能的自定义CLAM
                                instance_loss = results_dict['inst_loss']

                        if instance_loss is None:
                            # 没有实例损失就退化为bag loss，避免崩溃
                            if epoch == 0 and idx == 0:
                                logger.warning(f"[Fold {fold}] results_dict keys: {list(results_dict.keys()) if isinstance(results_dict, dict) else type(results_dict)}")
                            loss = bag_loss
                        else:
                            loss = args.bag_weight * bag_loss + (1 - args.bag_weight) * instance_loss


                    else:
                        outputs, Y_prob, Y_hat, A, _ = clam_model(
                            patch_features,
                            label=slide_labels,
                            instance_eval=True,
                            attn_mask=attn_mask
                        )
                        loss = criterion(outputs, slide_labels)

                scaler.scale(loss).backward()
                
                # 强烈建议添加梯度裁剪：防止在 AMP (FP16) 和高维特征（1024-d）下发生梯度爆炸/下溢
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clam_model.parameters(), max_norm=5.0)
                
                scaler.step(optimizer)
                scaler.update()


                
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            if writer:
                writer.add_scalar(f'Fold_{fold}/Train_Loss', avg_epoch_loss, epoch+1)
            val_auc, val_acc, val_recall, val_f1, val_specificity, val_qwk, val_labels, val_preds = evaluate(val_loader, clam_model, device, args.n_classes)
            
            #logger.info(f"Fold [{fold}] Epoch [{epoch+1}/{args.max_epochs}] Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}, Val Recall={val_recall:.4f}, Val F1={val_f1:.4f}, Val Specificity={val_specificity:.4f}")
            if writer:
                writer.add_scalar(f'Fold_{fold}/Val_AUC', val_auc, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_Acc', val_acc, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_Recall', val_recall, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_F1', val_f1, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_Specificity', val_specificity, epoch+1)
                writer.add_scalar(f'Fold_{fold}/Val_QWK', val_qwk, epoch+1)

            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)

            if len(epoch_times) >= 2:
                avg_epoch_time = sum(epoch_times) / len(epoch_times)
                remaining_epochs_in_fold = args.max_epochs - (epoch + 1)
                remaining_folds = end - (fold + 1)
                total_remaining_epochs = remaining_epochs_in_fold + remaining_folds * args.max_epochs
                estimated_remaining_time = avg_epoch_time * total_remaining_epochs
                eta_timestamp = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(time.time() + estimated_remaining_time)
                )
            else:
                eta_timestamp = "estimating..."

            logger.info(
                f"Fold [{fold}] Epoch [{epoch+1}/{args.max_epochs}] | "
                f"loss={avg_epoch_loss:.4f} | "
                f"val_auc={val_auc:.4f} | val_acc={val_acc:.4f} | "
                f"val_recall={val_recall:.4f} | val_f1={val_f1:.4f} | "
                f"val_spec={val_specificity:.4f} | val_qwk={val_qwk:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"eta={eta_timestamp}"
            )

            # improved = val_auc > best_score + min_delta
            # if improved:
            #     best_score = val_auc
            #     best_epoch = epoch + 1
            #     best_metrics = {
            #         'val_auc': val_auc,
            #         'val_acc': val_acc,
            #         'val_recall': val_recall,
            #         'val_f1': val_f1,
            #         'val_specificity': val_specificity,
            #         'val_qwk': val_qwk
            #     }
            #     counter = 0
            #     torch.save(clam_model.state_dict(), best_ckpt_path)
            #     logger.info(f"New best model saved for Fold [{fold}] at epoch {epoch+1} with Val AUC={val_auc:.4f}")
            # else:
            #     if args.early_stopping:
            #         counter += 1
            #         logger.info(f"No improvement in Val AUC for {counter} epochs (Current AUC={val_auc:.4f}, Best AUC={best_score:.4f})")

            # if args.early_stopping and counter >= patience:
            #     logger.info(f"Early stopping triggered for Fold [{fold}] at epoch {epoch+1}")
            #     break
            # PANDA 用 QWK 作为主评估指标；其他任务仍用 AUC
            if 'PANDA' in args.task:
                monitor_name = 'QWK'
                monitor_score = val_qwk
            else:
                monitor_name = 'AUC'
                monitor_score = val_auc

            improved = monitor_score > best_score + min_delta
            if improved:
                best_score = monitor_score
                best_epoch = epoch + 1
                best_metrics = {
                    'val_auc': val_auc,
                    'val_acc': val_acc,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'val_specificity': val_specificity,
                    'val_qwk': val_qwk
                }
                counter = 0
                torch.save(clam_model.state_dict(), best_ckpt_path)

                if 'PANDA' in args.task:
                    logger.info(
                        f"New best model saved for Fold [{fold}] at epoch {epoch+1} "
                        f"with Val QWK={val_qwk:.4f} | "
                        f"Acc={val_acc:.4f} | Macro-F1={val_f1:.4f} | AUC={val_auc:.4f}"
                    )
                else:
                    logger.info(
                        f"New best model saved for Fold [{fold}] at epoch {epoch+1} "
                        f"with Val AUC={val_auc:.4f} | "
                        f"Acc={val_acc:.4f} | Macro-F1={val_f1:.4f}"
                    )
            # else:
            #     if args.early_stopping:
            #         counter += 1
            #         if 'PANDA' in args.task:
            #             logger.info(
            #                 f"No improvement in Val QWK for {counter} epochs "
            #                 f"(Current QWK={val_qwk:.4f}, Best QWK={best_score:.4f})"
            #             )
            #         else:
            #             logger.info(
            #                 f"No improvement in Val AUC for {counter} epochs "
            #                 f"(Current AUC={val_auc:.4f}, Best AUC={best_score:.4f})"
            #             )

            # # if args.early_stopping and counter >= patience:
            # #     logger.info(
            # #         f"Early stopping triggered for Fold [{fold}] at epoch {epoch+1} "
            # #         f"based on Val {monitor_name}"
            # #     )
            #     break
            # if args.early_stopping and (epoch + 1) >= args.min_epochs and counter >= patience:
            #     logger.info(
            #         f"Early stopping triggered for Fold [{fold}] at epoch {epoch+1} "
            #         f"based on Val {monitor_name} | min_epochs={args.min_epochs}"
            #     )
            #     break
            else:
                if args.early_stopping:
                    counter += 1
                    if 'PANDA' in args.task:
                        logger.info(
                            f"No improvement in Val QWK for {counter} epochs "
                            f"(Current QWK={val_qwk:.4f}, Best QWK={best_score:.4f})"
                        )
                    else:
                        logger.info(
                            f"No improvement in Val AUC for {counter} epochs "
                            f"(Current AUC={val_auc:.4f}, Best AUC={best_score:.4f})"
                        )

            if args.early_stopping and (epoch + 1) >= args.min_epochs and counter >= patience:
                logger.info(
                    f"Early stopping triggered for Fold [{fold}] at epoch {epoch+1} "
                    f"based on Val {monitor_name} | min_epochs={args.min_epochs}"
                )
                break

            scheduler.step(monitor_score)  # 同时根据主监控指标调整学习率

            
            if writer:
                writer.flush()

        if os.path.isfile(best_ckpt_path):
            state_dict = torch.load(best_ckpt_path, map_location=device)
            clam_model.load_state_dict(state_dict)
            logger.info(f"Loaded best checkpoint for Fold [{fold}] from {best_ckpt_path} (best_epoch={best_epoch})")
        else:
            logger.warning(f"Best checkpoint not found for Fold [{fold}], using current model for testing.")

        # 用当前(最好是best ckpt)模型重新评估val，保证口径一致
        best_val_auc, best_val_acc, best_val_recall, best_val_f1, best_val_specificity, best_val_qwk, best_val_labels, best_val_preds = \
            evaluate(val_loader, clam_model, device, args.n_classes)

        logger.info(f"Evaluating Test Set for Fold [{fold}]")
        test_auc, test_acc, test_recall, test_f1, test_specificity, test_qwk, test_labels, test_preds = \
            evaluate(test_loader, clam_model, device, args.n_classes)
        
        all_test_auc.append(test_auc)
        all_val_auc.append(best_val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(best_val_acc)
        all_test_recall.append(test_recall)
        all_val_recall.append(best_val_recall)
        all_test_f1.append(test_f1)
        all_val_f1.append(best_val_f1)
        all_test_specificity.append(test_specificity)
        all_val_specificity.append(best_val_specificity)
        all_test_qwk.append(test_qwk)
        all_val_qwk.append(best_val_qwk)
        
        # Store true labels and predicted probabilities for this fold
        all_val_labels.append(best_val_labels)
        all_val_preds.append(best_val_preds)
        all_test_labels.append(test_labels)
        all_test_preds.append(test_preds)
        
        if writer:
            writer.add_scalar(f'Fold_{fold}/Test_AUC', test_auc, best_epoch)
            writer.add_scalar(f'Fold_{fold}/Test_Acc', test_acc, best_epoch)
            writer.add_scalar(f'Fold_{fold}/Test_Recall', test_recall, best_epoch)
            writer.add_scalar(f'Fold_{fold}/Test_F1', test_f1, best_epoch)
            writer.add_scalar(f'Fold_{fold}/Test_Specificity', test_specificity, best_epoch)
            writer.add_scalar(f'Fold_{fold}/Test_QWK', test_qwk, best_epoch)
            writer.flush()
        
        filename = os.path.join(args.results_dir, f'split_{fold}_results.pkl')

        save_pkl(filename, {
            'val_auc': best_val_auc,
            'test_auc': test_auc,
            'val_acc': best_val_acc,
            'test_acc': test_acc,
            'val_recall': best_val_recall,
            'test_recall': test_recall,
            'val_f1': best_val_f1,
            'test_f1': test_f1,
            'val_specificity': best_val_specificity,
            'test_specificity': test_specificity,
            'val_labels': best_val_labels,
            'val_preds': best_val_preds,
            'test_labels': test_labels,
            'test_preds': test_preds,
            'best_epoch': best_epoch,
            'val_qwk': best_val_qwk,
            'test_qwk': test_qwk
        })

        #logger.info(f"Fold [{fold}] Evaluation Results: Val AUC={val_auc:.4f}, Val Acc={val_acc:.4f}, Val Recall={val_recall:.4f}, Val F1={val_f1:.4f}, Val Specificity={val_specificity:.4f}, Test AUC={test_auc:.4f}, Test Acc={test_acc:.4f}, Test Recall={test_recall:.4f}, Test F1={test_f1:.4f}, Test Specificity={test_specificity:.4f}")
        logger.info(
            f"Fold [{fold}] Evaluation Results: "
            f"Best Val AUC={best_metrics['val_auc']:.4f}, "
            f"Best Val Acc={best_metrics['val_acc']:.4f}, "
            f"Best Val Recall={best_metrics['val_recall']:.4f}, "
            f"Best Val F1={best_metrics['val_f1']:.4f}, "
            f"Best Val Specificity={best_metrics['val_specificity']:.4f}, "
            f"Best Val QWK={best_metrics['val_qwk']:.4f}, "
            f"Test AUC={test_auc:.4f}, Test Acc={test_acc:.4f}, "
            f"Test Recall={test_recall:.4f}, Test F1={test_f1:.4f}, "
            f"Test Specificity={test_specificity:.4f}, Test QWK={test_qwk:.4f}"
        )


        # checkpoint_path = os.path.join(args.results_dir, f's_{fold}_checkpoint.pt')
        # torch.save(clam_model.state_dict(), checkpoint_path)
        # logger.info(f"Saved checkpoint for Fold [{fold}] at {checkpoint_path}")

        if writer:
            writer.close() 
        

    final_df = pd.DataFrame({
        'folds': folds,
        'test_auc': all_test_auc,
        'val_auc': all_val_auc,
        'test_acc': all_test_acc,
        'val_acc': all_val_acc,
        'test_recall': all_test_recall,
        'val_recall': all_val_recall,
        'test_f1': all_test_f1,
        'val_f1': all_val_f1,
        'test_specificity': all_test_specificity,
        'val_specificity': all_val_specificity,
        'test_qwk': all_test_qwk,
        'val_qwk': all_val_qwk
    })

    summary_stats_df = pd.DataFrame([
        {
            'metric': 'val_qwk',
            'mean': np.mean(all_val_qwk),
            'std': np.std(all_val_qwk)
        },
        {
            'metric': 'test_qwk',
            'mean': np.mean(all_test_qwk),
            'std': np.std(all_test_qwk)
        },
        {
            'metric': 'val_acc',
            'mean': np.mean(all_val_acc),
            'std': np.std(all_val_acc)
        },
        {
            'metric': 'test_acc',
            'mean': np.mean(all_test_acc),
            'std': np.std(all_test_acc)
        },
        {
            'metric': 'val_f1',
            'mean': np.mean(all_val_f1),
            'std': np.std(all_val_f1)
        },
        {
            'metric': 'test_f1',
            'mean': np.mean(all_test_f1),
            'std': np.std(all_test_f1)
        },
        {
            'metric': 'val_auc',
            'mean': np.mean(all_val_auc),
            'std': np.std(all_val_auc)
        },
        {
            'metric': 'test_auc',
            'mean': np.mean(all_test_auc),
            'std': np.std(all_test_auc)
        },
        {
            'metric': 'val_specificity',
            'mean': np.mean(all_val_specificity),
            'std': np.std(all_val_specificity)
        },
        {
            'metric': 'test_specificity',
            'mean': np.mean(all_test_specificity),
            'std': np.std(all_test_specificity)
        },
        {
            'metric': 'val_recall',
            'mean': np.mean(all_val_recall),
            'std': np.std(all_val_recall)
        },
        {
            'metric': 'test_recall',
            'mean': np.mean(all_test_recall),
            'std': np.std(all_test_recall)
        }
    ])

    summary_stats_path = os.path.join(args.results_dir, 'summary_stats.csv')
    summary_stats_df.to_csv(summary_stats_path, index=False)

    logger.info("################# Cross-validation Mean ± Std ###################")
    for _, row in summary_stats_df.iterrows():
        logger.info(f"{row['metric']}: {row['mean']:.4f} ± {row['std']:.4f}")


    if len(folds) != args.k:
        save_name = f'summary_partial_{start}_{end}.csv'
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)

    logger.info("################# Final Results ###################")
    logger.info(final_df)
    logger.info("Finished!")
    logger.info("End script")
        
    # Return true labels and predicted probabilities for all folds
    return {
        'val_labels': all_val_labels,  # List of y_true_fold_k for validation sets
        'val_preds': all_val_preds,    # List of y_pred_proba_fold_k for validation sets
        'test_labels': all_test_labels,  # List of y_true_fold_k for test sets
        'test_preds': all_test_preds     # List of y_pred_proba_fold_k for test sets
    }

# def evaluate(loader, model, device, n_classes):
#     model.eval()
#     all_probs = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in loader:
#             patch_features = batch['features'].to(device)
#             slide_labels = batch['labels'].to(device)
#             attn_mask = batch['mask'].to(device)

#             with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
#                 outputs, _, _, _, _ = model(
#                     patch_features,
#                     label= None,
#                     instance_eval=False,
#                     attn_mask=attn_mask
#                 )


#             probs = torch.softmax(outputs, dim=1).cpu().numpy()   # [B, C]
#             labels = slide_labels.view(-1).cpu().numpy()

#             all_probs.extend(probs)
#             all_labels.extend(labels)

#     all_probs = np.array(all_probs)
#     all_labels = np.array(all_labels)
#     pred_labels = np.argmax(all_probs, axis=1)
def evaluate(loader, model, device, n_classes):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            patch_features = batch['features'].to(device)
            slide_labels = batch['labels'].to(device)
            attn_mask = batch['mask'].to(device)

            # 评估阶段不要用混合精度，避免多分类 softmax 概率和不为 1
            with torch.cuda.amp.autocast(enabled=False):
                outputs, _, _, _, _ = model(
                    patch_features,
                    label=None,
                    instance_eval=False,
                    attn_mask=attn_mask
                )

            outputs = outputs.float()
            probs = torch.softmax(outputs, dim=1)
            probs = probs / probs.sum(dim=1, keepdim=True)  # 保证每行严格归一化

            probs = probs.cpu().numpy()
            labels = slide_labels.view(-1).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.asarray(all_probs, dtype=np.float64)
    all_labels = np.asarray(all_labels)
    pred_labels = np.argmax(all_probs, axis=1)


    acc = accuracy_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels, average='macro', zero_division=0)
    f1 = f1_score(all_labels, pred_labels, average='macro', zero_division=0)

    qwk = cohen_kappa_score(all_labels, pred_labels, weights='quadratic')

    if n_classes == 2:
        pos_probs = all_probs[:, 1]
        try:
            auc = roc_auc_score(all_labels, pos_probs)
        except ValueError:
            auc = 0.0

        cm = confusion_matrix(all_labels, pred_labels, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = 0.0

        stored_preds = pos_probs.tolist()

    else:
        # PANDA等多分类：AUC可选保留，但不作为主指标
        row_sums = all_probs.sum(axis=1)
        logger.info(f"AUC debug | min row sum={row_sums.min():.8f}, max row sum={row_sums.max():.8f}")
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except ValueError as e:
            logger.warning(f"AUC calculation failed: {e}. Setting AUC to 0.0 for this fold.")
            auc = 0.0

        cm = confusion_matrix(all_labels, pred_labels, labels=list(range(n_classes)))
        specificity_list = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_list.append(spec_i)
        specificity = float(np.mean(specificity_list))

        stored_preds = all_probs.tolist()

    return auc, acc, recall, f1, specificity, qwk, all_labels.tolist(), stored_preds


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--task', type=str, choices=[
    'TCGA_BRCA_UNI',
    'TCGA_BRCA_CHIEF',
    'TCGA_NSCLC_UNI',
    'TCGA_NSCLC_CHIEF',
    'PANDA_UNI',
    'PANDA_CHIEF'
], required=True, help='Task type')
parser.add_argument('--data_root_dir', type=str, required=True, help='Data directory (project root)')
parser.add_argument('--embed_dim', type=int, default=1024, help='Embedding dimension')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='Fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-2, help='Weight decay (default: 1e-2)')  # 用户自行调整为 1e-2
parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start', type=int, default=-1, help='Start fold (default: -1, first fold)')
parser.add_argument('--k_end', type=int, default=-1, help='End fold (default: -1, last fold)')
parser.add_argument('--results_dir', default='./results', help='Results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, help='Manually specify the set of splits to use')
parser.add_argument('--log_data', action='store_true', default=False, help='Log data using TensorBoard')
parser.add_argument('--testing', action='store_true', default=False, help='Debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='Enable early stopping')
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam', help='Optimizer (default: adam)')
parser.add_argument('--drop_out', type=float, default=0.25, help='Dropout rate (default: 0.25)')  # 用户要求保持不变
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce', help='Slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', help='Type of model (default: clam_sb)')
parser.add_argument('--exp_code', type=str, required=True, help='Experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='Enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='Size of model')
parser.add_argument('--no_inst_cluster', action='store_true', default=False, help='Disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', 'None'], default='svm', help='Instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, help='Subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7, help='CLAM: Weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='Number of positive/negative patches to sample for CLAM')
parser.add_argument('--num_neighbors', type=int, default=5, help='Number of neighbors in dynamic graph embedding')

parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers')
parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pin_memory in DataLoader')
parser.add_argument('--persistent_workers', action='store_true', default=False, help='Use persistent_workers when num_workers > 0')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor when num_workers > 0')
parser.add_argument('--cache_in_ram', action='store_true', default=False, help='Cache all pt features in RAM for each split')

parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')
parser.add_argument('--min_epochs', type=int, default=50, help='Minimum epochs before enabling early stopping')



device = None

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = parser.parse_args()

    if args.cache_in_ram and args.num_workers > 0:
        logger.info(f"cache_in_ram=True, forcing num_workers from {args.num_workers} to 0")
        args.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(args.seed)

    settings = {
        'num_splits': args.k,
        'k_start': args.k_start,
        'k_end': args.k_end,
        'task': args.task,
        'max_epochs': args.max_epochs,
        'results_dir': args.results_dir,
        'lr': args.lr,
        'experiment': args.exp_code,
        'reg': args.reg,
        'label_frac': args.label_frac,
        'bag_loss': args.bag_loss,
        'seed': args.seed,
        'model_type': args.model_type,
        'model_size': args.model_size,
        "use_drop_out": args.drop_out,
        'weighted_sample': args.weighted_sample,
        'opt': args.opt,
        'num_workers': args.num_workers,
        'pin_memory': args.pin_memory,
        'persistent_workers': args.persistent_workers,
        'prefetch_factor': args.prefetch_factor,
        'cache_in_ram': args.cache_in_ram,
    }

    if args.model_type in ['clam_sb', 'clam_mb']:
        settings.update({
            'bag_weight': args.bag_weight,
            'inst_loss': args.inst_loss,
            'B': args.B,
            'num_neighbors': args.num_neighbors
        })

    if args.task in ['TCGA_BRCA_UNI', 'TCGA_BRCA_CHIEF']:
        args.n_classes = 2
        label_dict = {'IDC': 0, 'ILC': 1}
        csv_path = os.path.join(args.data_root_dir, 'dataset_csv', 'tcga_brca_uni_chief_patient.csv')

        if args.task == 'TCGA_BRCA_UNI':
            data_dir = r"D:\ADGA_Datasets\BRCA\feats-UNI-reduced-graph\pt_files"
        else:
            data_dir = r"D:\ADGA_Datasets\BRCA\feats-CHIEF-reduced-graph\pt_files"

    elif args.task in ['TCGA_NSCLC_UNI', 'TCGA_NSCLC_CHIEF']:
        args.n_classes = 2
        label_dict = {'LUAD': 0, 'LUSC': 1}
        csv_path = os.path.join(args.data_root_dir, 'dataset_csv', 'tcga_nsclc_uni_chief_patient.csv')
        
        if args.task == 'TCGA_NSCLC_UNI':
            data_dir = r"D:\ADGA_Datasets\NSCLC\feats-UNI-reduced-graph\pt_files"
        else:
            data_dir = r"D:\ADGA_Datasets\NSCLC\feats-CHIEF-reduced-graph\pt_files"


    elif args.task in ['PANDA_UNI', 'PANDA_CHIEF']:
        args.n_classes = 6
        label_dict = {
            'grade_0': 0,
            'grade_1': 1,
            'grade_2': 2,
            'grade_3': 3,
            'grade_4': 4,
            'grade_5': 5
        }
        csv_path = os.path.join(args.data_root_dir, 'dataset_csv', 'panda_uni_chief.csv')

        if args.task == 'PANDA_UNI':
            data_dir = r"D:\ADGA_Datasets\PANDA\UNI_features\pt_files"
        else:
            data_dir = r"D:\ADGA_Datasets\PANDA\CHIEF_features\pt_files"

    else:
        raise NotImplementedError(f"Task '{args.task}' is not implemented.")

    dataset = Generic_MIL_Dataset(
        csv_path=csv_path,
        data_dir=data_dir,
        shuffle=False,
        seed=args.seed,
        print_info=False,
        label_dict=label_dict,
        label_col='label',
        ignore=[],
        cache_in_ram=args.cache_in_ram
    )

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    args.results_dir = os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}")
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.split_dir is None:
        args.split_dir = os.path.join('splits', f"{args.task}_{int(args.label_frac*100)}")
    else:
        args.split_dir = os.path.join('splits', args.split_dir)

    assert os.path.isdir(args.split_dir), f"Split directory does not exist: {args.split_dir}"

    settings.update({'split_dir': args.split_dir})

    experiment_settings_path = os.path.join(args.results_dir, f'experiment_{args.exp_code}.txt')
    with open(experiment_settings_path, 'w') as f:
        print(settings, file=f)

    logger.info(f"Task: {args.task}")
    logger.info(f"Split dir: {args.split_dir}")
    logger.info(f"Results dir: {args.results_dir}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(
        f"num_workers={args.num_workers}, "
        f"pin_memory={args.pin_memory}, "
        f"persistent_workers={args.persistent_workers}, "
        f"prefetch_factor={args.prefetch_factor}, "
        f"cache_in_ram={args.cache_in_ram}"
    )

    results = main(args, dataset)
    logger.info("Finished!")
    logger.info("End script")
