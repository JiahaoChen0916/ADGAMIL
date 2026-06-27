import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def infer_dataset_name(csv_path: str) -> str:
    name = os.path.basename(csv_path).lower()
    if "brca" in name:
        return "BRCA"
    if "nsclc" in name:
        return "NSCLC"
    raise ValueError(
        f"Cannot infer dataset type from csv path: {csv_path}. "
        f"Please pass a BRCA or NSCLC *_patient.csv file."
    )


def get_default_save_names(dataset_name: str):
    if dataset_name == "BRCA":
        return ["TCGA_BRCA_UNI_100", "TCGA_BRCA_CHIEF_100"]
    elif dataset_name == "NSCLC":
        return ["TCGA_NSCLC_UNI_100", "TCGA_NSCLC_CHIEF_100"]
    else:
        raise ValueError(dataset_name)


def validate_patient_csv(df: pd.DataFrame):
    required_cols = {"case_id", "slide_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["case_id"] = df["case_id"].astype(str)
    df["slide_id"] = df["slide_id"].astype(str)
    df["label"] = df["label"].astype(str)

    # Optional but useful
    if "site_id" not in df.columns:
        df["site_id"] = df["slide_id"].apply(lambda x: "-".join(str(x).split("-")[:2]))
    else:
        df["site_id"] = df["site_id"].astype(str)

    # Each patient must have exactly one label
    patient_label_nunique = df.groupby("case_id")["label"].nunique()
    bad_patients = patient_label_nunique[patient_label_nunique > 1]
    if len(bad_patients) > 0:
        print("[ERROR] Some patients have multiple labels:")
        print(bad_patients.head(20))
        raise RuntimeError("Patient-level labels are inconsistent.")

    return df


def build_patient_df(df: pd.DataFrame) -> pd.DataFrame:
    patient_df = (
        df[["case_id", "label"]]
        .drop_duplicates()
        .rename(columns={"case_id": "patient_id"})
        .sort_values(["label", "patient_id"])
        .reset_index(drop=True)
    )
    return patient_df


def make_one_fold(patient_df: pd.DataFrame, seed: int, val_frac: float, test_frac: float):
    """
    Repeated patient-level stratified split:
      1) split out test patients
      2) split remaining patients into train/val
    """
    y = patient_df["label"].values
    patient_ids = patient_df["patient_id"].values

    outer = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(outer.split(patient_ids, y))

    trainval_df = patient_df.iloc[trainval_idx].reset_index(drop=True)
    test_df = patient_df.iloc[test_idx].reset_index(drop=True)

    # Convert overall val fraction to relative val fraction inside trainval
    val_relative = val_frac / (1.0 - test_frac)
    if not (0 < val_relative < 1):
        raise ValueError(
            f"Invalid val_relative={val_relative:.4f}. "
            f"Check val_frac={val_frac}, test_frac={test_frac}."
        )

    inner = StratifiedShuffleSplit(n_splits=1, test_size=val_relative, random_state=seed + 1000)
    inner_train_idx, inner_val_idx = next(inner.split(trainval_df["patient_id"].values, trainval_df["label"].values))

    train_df = trainval_df.iloc[inner_train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[inner_val_idx].reset_index(drop=True)

    train_patients = set(train_df["patient_id"].tolist())
    val_patients = set(val_df["patient_id"].tolist())
    test_patients = set(test_df["patient_id"].tolist())

    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0

    return train_patients, val_patients, test_patients


def save_split_csv(split_dir, fold, train_slides, val_slides, test_slides):
    split_df = pd.DataFrame({
        "train": pd.Series(train_slides, dtype="object"),
        "val": pd.Series(val_slides, dtype="object"),
        "test": pd.Series(test_slides, dtype="object"),
    })
    split_df.to_csv(os.path.join(split_dir, f"splits_{fold}.csv"), index=False)


def save_bool_csv(split_dir, fold, train_mask, val_mask, test_mask):
    bool_df = pd.DataFrame({
        "train": train_mask.astype(bool),
        "val": val_mask.astype(bool),
        "test": test_mask.astype(bool),
    })
    bool_df.to_csv(os.path.join(split_dir, f"splits_{fold}_bool.csv"), index=False)


def save_descriptor_csv(split_dir, fold, df, train_mask, val_mask, test_mask):
    labels = sorted(df["label"].astype(str).unique().tolist())
    descriptor_df = pd.DataFrame(index=labels, columns=["train", "val", "test"]).fillna(0)

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        counts = df.loc[mask, "label"].astype(str).value_counts()
        for lab in labels:
            descriptor_df.loc[lab, split_name] = int(counts.get(lab, 0))

    descriptor_df.to_csv(os.path.join(split_dir, f"splits_{fold}_descriptor.csv"))


def print_fold_stats(df, patient_df, train_patients, val_patients, test_patients):
    train_mask = df["case_id"].isin(train_patients)
    val_mask = df["case_id"].isin(val_patients)
    test_mask = df["case_id"].isin(test_patients)

    print(f"  Train slides   : {int(train_mask.sum())} | patients: {len(train_patients)}")
    print(f"  Val slides     : {int(val_mask.sum())} | patients: {len(val_patients)}")
    print(f"  Test slides    : {int(test_mask.sum())} | patients: {len(test_patients)}")

    for name, pts in [("Train", train_patients), ("Val", val_patients), ("Test", test_patients)]:
        sub = patient_df[patient_df["patient_id"].isin(pts)]
        print(f"  {name} patient labels: {sub['label'].value_counts().to_dict()}")

    train_sites = set(df.loc[train_mask, "site_id"].unique().tolist())
    val_sites = set(df.loc[val_mask, "site_id"].unique().tolist())
    test_sites = set(df.loc[test_mask, "site_id"].unique().tolist())
    print(
        "  Site overlap (info only) | "
        f"train∩val={len(train_sites & val_sites)}, "
        f"train∩test={len(train_sites & test_sites)}, "
        f"val∩test={len(val_sites & test_sites)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Universal patient-level create_splits for TCGA BRCA / NSCLC"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to tcga_brca_uni_chief_patient.csv or tcga_nsclc_uni_chief_patient.csv"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of repeated folds")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Validation fraction at patient level")
    parser.add_argument("--test_frac", type=float, default=0.1, help="Test fraction at patient level")
    parser.add_argument(
        "--save_names",
        nargs="+",
        default=None,
        help="Output split directory names under ./splits/. "
             "If omitted, BRCA -> TCGA_BRCA_UNI_100 / TCGA_BRCA_CHIEF_100; "
             "NSCLC -> TCGA_NSCLC_UNI_100 / TCGA_NSCLC_CHIEF_100"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    dataset_name = infer_dataset_name(args.csv_path)
    save_names = args.save_names if args.save_names is not None else get_default_save_names(dataset_name)

    df = pd.read_csv(args.csv_path)
    df = validate_patient_csv(df)
    patient_df = build_patient_df(df)

    print("===== Dataset Summary =====")
    print(f"Dataset  : {dataset_name}")
    print(f"Slides   : {len(df)}")
    print(f"Patients : {df['case_id'].nunique()}")
    print(f"Sites    : {df['site_id'].nunique()}")
    print("Slide label counts:")
    print(df["label"].value_counts())
    print("Patient label counts:")
    print(patient_df["label"].value_counts())

    for save_name in save_names:
        split_dir = os.path.join("splits", save_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\n===== Writing splits to: {split_dir} =====")

        for fold in range(args.k):
            train_patients, val_patients, test_patients = make_one_fold(
                patient_df=patient_df,
                seed=args.seed + fold,
                val_frac=args.val_frac,
                test_frac=args.test_frac
            )

            train_mask = df["case_id"].isin(train_patients)
            val_mask = df["case_id"].isin(val_patients)
            test_mask = df["case_id"].isin(test_patients)

            train_slides = df.loc[train_mask, "slide_id"].tolist()
            val_slides = df.loc[val_mask, "slide_id"].tolist()
            test_slides = df.loc[test_mask, "slide_id"].tolist()

            save_split_csv(split_dir, fold, train_slides, val_slides, test_slides)
            save_bool_csv(split_dir, fold, train_mask, val_mask, test_mask)
            save_descriptor_csv(split_dir, fold, df, train_mask, val_mask, test_mask)

            print(f"\nGenerated Fold [{fold}]")
            print_fold_stats(df, patient_df, train_patients, val_patients, test_patients)

    print("\nDone.")


if __name__ == "__main__":
    main()
