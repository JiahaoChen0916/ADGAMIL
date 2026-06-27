# create_splits_seq.py
import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')

    parser.add_argument('--label_frac', type=float, default=1.0,
                        help='Fraction of labels (default: 1)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of splits (default: 5)')
    parser.add_argument('--task', type=str, choices=[
        'camelyon16_plip',
        'camelyon16_R50',
        'TCGA_BC_PLIP',
        'TCGA_BC_R50',
        'TCGA_NSCLC_PLIP',
        'TCGA_NSCLC_R50',
        'TCGA_NSCLC_UNI',
        'TCGA_NSCLC_CHIEF'
    ], required=True, help='Task type')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of labels for validation (default: 0.1)')
    parser.add_argument('--test_frac', type=float, default=0.1,
                        help='Fraction of labels for test (default: 0.1)')

    args = parser.parse_args()

    # 配置每个任务的参数
    if args.task == 'camelyon16_plip':
        args.n_classes = 2
        csv_path = os.path.join('dataset_csv', 'camelyon16_plip.csv')
        label_dict = {'0': 0, '1': 1}  # 确保 CSV 中标签为字符串 '0' 和 '1'
        label_col = 'label'
    elif args.task == 'camelyon16_R50':
        args.n_classes = 2
        csv_path = os.path.join('dataset_csv', 'camelyon16_R50.csv')
        label_dict = {'0': 0, '1': 1}  # 确保 CSV 中标签为字符串 '0' 和 '1'
        label_col = 'label'
    elif args.task == 'TCGA_BC_PLIP':
        args.n_classes = 2
        csv_path = os.path.join('RRT_data', 'tcga-subtyping', 'TCGA-BRCA PLIP', 'label.csv')
        label_dict = {'IDC': 0, 'ILC': 1}
        label_col = 'label'
    elif args.task == 'TCGA_BC_R50':
        args.n_classes = 2
        csv_path = os.path.join('RRT_data', 'tcga-subtyping', 'TCGA-BRCA R50', 'label.csv')
        label_dict = {'IDC': 0, 'ILC': 1}
        label_col = 'label'
    elif args.task == 'TCGA_NSCLC_PLIP':
        args.n_classes = 2
        csv_path = os.path.join('RRT_data', 'tcga-subtyping', 'TCGA-NSCLC PLIP', 'label.csv')
        label_dict = {'LUAD': 0, 'LUSC': 1}  # 根据实际标签调整
        label_col = 'label'
    elif args.task == 'TCGA_NSCLC_R50':
        args.n_classes = 2
        csv_path = os.path.join('RRT_data', 'tcga-subtyping', 'TCGA-NSCLC R50', 'label.csv')
        label_dict = {'LUAD': 0, 'LUSC': 1}  # 根据实际标签调整
        label_col = 'label'
    
    elif args.task == 'TCGA_NSCLC_UNI':
        args.n_classes = 2
        csv_path = os.path.join('dataset_csv', 'tcga_nsclc_uni_chief_patient.csv')
        label_dict = {'LUAD': 0, 'LUSC': 1}
        label_col = 'label'

    elif args.task == 'TCGA_NSCLC_CHIEF':
        args.n_classes = 2
        csv_path = os.path.join('dataset_csv', 'tcga_nsclc_uni_chief_patient.csv')
        label_dict = {'LUAD': 0, 'LUSC': 1}
        label_col = 'label'
    
    else:
        raise NotImplementedError(f"Task '{args.task}' is not implemented.")

    # 打印任务配置以进行调试
    print(f"Task: {args.task}")
    print(f"CSV path: {csv_path}")
    print(f"Label dictionary: {label_dict}")
    print(f"Label column: {label_col}")

    # 创建数据集
    if args.task.startswith('camelyon16'):
        print("Initializing Generic_WSI_Classification_Dataset with patient_voting='max'")
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=csv_path,
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict=label_dict,
            patient_strat=True,
            ignore=[],
            label_col=label_col
        )
    else:  # TCGA 子任务
        print("Initializing Generic_WSI_Classification_Dataset with patient_voting='maj'")
        dataset = Generic_WSI_Classification_Dataset(
            csv_path=csv_path,
            shuffle=False,
            seed=args.seed,
            print_info=True,
            label_dict=label_dict,
            patient_strat=True,
            patient_voting='maj',
            ignore=[],
            label_col=label_col
        )

    # 计算验证集和测试集数量
    num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
    val_num = np.round(num_slides_cls * args.val_frac).astype(int)
    test_num = np.round(num_slides_cls * args.test_frac).astype(int)

    # 创建数据拆分
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = os.path.join('splits', f"{args.task}_{int(lf * 100)}")
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k=args.k, val_num=val_num, test_num=test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, f'splits_{i}.csv'))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, f'splits_{i}_bool.csv'), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, f'splits_{i}_descriptor.csv'))

if __name__ == '__main__':
    main()
