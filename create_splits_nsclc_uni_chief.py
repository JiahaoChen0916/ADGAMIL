# create_splits_nsclc_uni_chief.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def save_split_csv(train_ids, val_ids, test_ids, out_path):
    max_len = max(len(train_ids), len(val_ids), len(test_ids))
    rows = []
    for i in range(max_len):
        rows.append({
            "train": train_ids[i] if i < len(train_ids) else "",
            "val": val_ids[i] if i < len(val_ids) else "",
            "test": test_ids[i] if i < len(test_ids) else "",
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


def save_bool_csv(all_case_ids, train_set, val_set, test_set, out_path):
    rows = []
    for cid in all_case_ids:
        rows.append({
            "case_id": cid,
            "train": cid in train_set,
            "val": cid in val_set,
            "test": cid in test_set,
        })
    df = pd.DataFrame(rows).set_index("case_id")
    df.to_csv(out_path, encoding="utf-8-sig")


def save_descriptor_csv(df_all, train_set, val_set, test_set, out_path):
    labels = ["LUAD", "LUSC"]
    rows = []
    for lb in labels:
        rows.append({
            "label": lb,
            "train": int(df_all[(df_all["label"] == lb) & (df_all["case_id"].isin(train_set))].shape[0]),
            "val": int(df_all[(df_all["label"] == lb) & (df_all["case_id"].isin(val_set))].shape[0]),
            "test": int(df_all[(df_all["label"] == lb) & (df_all["case_id"].isin(test_set))].shape[0]),
        })
    pd.DataFrame(rows).set_index("label").to_csv(out_path, encoding="utf-8-sig")


def generate_splits(df, out_dir, k=5, val_frac=0.1, test_frac=0.1, seed=1):
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy().reset_index(drop=True)

    case_ids = df["case_id"].values
    labels = df["label"].values
    all_case_ids = df["case_id"].tolist()

    # 外层：切 test
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(case_ids, labels)):
        df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # 从 trainval 中再切 val
        # 目标总 val_frac，例如 0.1
        # trainval 占 (1 - test_frac)，所以 val_in_trainval = val_frac / (1 - test_frac)
        val_in_trainval = val_frac / (1.0 - test_frac)

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_in_trainval,
            random_state=seed + fold
        )

        tv_case_ids = df_trainval["case_id"].values
        tv_labels = df_trainval["label"].values

        inner_train_idx, val_idx = next(sss.split(tv_case_ids, tv_labels))

        df_train = df_trainval.iloc[inner_train_idx].reset_index(drop=True)
        df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

        train_ids = df_train["case_id"].tolist()
        val_ids = df_val["case_id"].tolist()
        test_ids = df_test["case_id"].tolist()

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        # overlap check
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

        save_split_csv(
            train_ids, val_ids, test_ids,
            os.path.join(out_dir, f"splits_{fold}.csv")
        )
        save_bool_csv(
            all_case_ids, train_set, val_set, test_set,
            os.path.join(out_dir, f"splits_{fold}_bool.csv")
        )
        save_descriptor_csv(
            df, train_set, val_set, test_set,
            os.path.join(out_dir, f"splits_{fold}_descriptor.csv")
        )

        print(f"[{out_dir}] fold {fold} done")
        print(f"  train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        # label summary
        for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
            counts = split_df["label"].value_counts().to_dict()
            print(f"    {split_name}: {counts}")


def main():
    parser = argparse.ArgumentParser(description="Create NSCLC UNI/CHIEF splits without torch")
    parser.add_argument("--csv_path", type=str,
                        default=r"D:\ADGA-main\dataset_csv\tcga_nsclc_uni_chief_patient.csv")
    parser.add_argument("--split_root", type=str,
                        default=r"D:\ADGA-main\splits")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    df.columns = [c.strip() for c in df.columns]

    required_cols = ["case_id", "slide_id", "label"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["slide_id"] = df["slide_id"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    if "match_id" in df.columns:
        df["match_id"] = df["match_id"].astype(str).str.strip()


    # 去重，按 case_id
    df = df.drop_duplicates(subset=["case_id"]).reset_index(drop=True)

    # 只允许 LUAD/LUSC
    valid_labels = {"LUAD", "LUSC"}
    bad = df[~df["label"].isin(valid_labels)]
    if len(bad) > 0:
        raise ValueError(f"发现非法标签:\n{bad.head()}")

    print("=" * 60)
    print("Input dataset stats")
    print(df["label"].value_counts())
    print(f"Total cases: {len(df)}")
    print("=" * 60)

    out_uni = os.path.join(args.split_root, "TCGA_NSCLC_UNI_100")
    out_chief = os.path.join(args.split_root, "TCGA_NSCLC_CHIEF_100")

    generate_splits(
        df=df,
        out_dir=out_uni,
        k=args.k,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )

    generate_splits(
        df=df,
        out_dir=out_chief,
        k=args.k,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )

    print("=" * 60)
    print("All splits created successfully.")
    print(f"UNI   -> {out_uni}")
    print(f"CHIEF -> {out_chief}")
    print("=" * 60)


if __name__ == "__main__":
    main()
