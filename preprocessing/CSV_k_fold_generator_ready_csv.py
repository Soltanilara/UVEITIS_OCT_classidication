import argparse
import os
import random

import numpy as np
import pandas as pd


def normalize_relative_path(path_val: str) -> str:
    path = str(path_val).replace("\\", "/").strip()
    patient_idx = path.lower().find("patient")
    return path[patient_idx:] if patient_idx >= 0 else path


def save_class_weights(prefix_path: str, df_subset: pd.DataFrame) -> None:
    label_counts = df_subset["Label"].value_counts()
    total = label_counts.sum()
    if total == 0:
        np.save(os.path.join(prefix_path, "classWeights.npy"), np.array([1.0, 1.0]))
        return

    neg_count = label_counts.get("negative", 0)
    weight_negative = 2 * neg_count / total
    weight_positive = 2 - weight_negative
    np.save(os.path.join(prefix_path, "classWeights.npy"), np.array([weight_positive, weight_negative]))


def save_graded_weights(prefix_path: str, df_subset: pd.DataFrame) -> None:
    label_categories = ["negative", "mild", "moderate", "severe"]
    label_counts = df_subset["Label"].value_counts()
    total_samples = len(df_subset)

    if total_samples == 0:
        np.save(os.path.join(prefix_path, "gradedWeights.npy"), np.array([0.25, 0.25, 0.25, 0.25]))
        return

    weights = np.zeros(len(label_categories), dtype=np.float32)
    for idx, label in enumerate(label_categories):
        if label in label_counts:
            weights[idx] = total_samples / label_counts[label]

    if weights.sum() > 0:
        weights = weights / weights.sum()

    np.save(os.path.join(prefix_path, "gradedWeights.npy"), weights)


def build_label_from_zones(df: pd.DataFrame, zone_cols: list[str]) -> pd.Series:
    zone_numeric = df[zone_cols].apply(pd.to_numeric, errors="coerce")
    max_zone = zone_numeric.max(axis=1).fillna(0).round().astype(int)
    label_map = {0: "negative", 1: "mild", 2: "moderate", 3: "severe"}
    return max_zone.map(label_map).fillna("negative")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level k-fold CSVs from a ready row-level CSV.")
    parser.add_argument("--csv_path", required=True, help="Input ready dataset CSV.")
    parser.add_argument("--output_root", required=True, help="Output root containing fold_*/ split CSVs.")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_val", type=int, default=10, help="Number of validation patients per fold.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_column", default="UWFFP", help="Column to use for fundus Image File paths.")
    parser.add_argument("--group_column", default="Patient_ID", help="Patient-level grouping column.")
    parser.add_argument("--drop_missing_zone_rows", default="all", choices=["none", "any", "all"])
    parser.add_argument("--dataset_root", default="", help="Optional dataset root used to verify Image File paths.")
    parser.add_argument("--drop_missing_images", action="store_true", help="Drop rows whose Image File is missing under --dataset_root.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    df = pd.read_csv(args.csv_path)
    zone_cols = [f"Zone{i}_label" for i in range(1, 11)]
    missing_zone_cols = [col for col in zone_cols if col not in df.columns]
    if missing_zone_cols:
        raise ValueError(f"Missing zone columns: {missing_zone_cols}")
    if args.image_column not in df.columns:
        raise ValueError(f"Image column {args.image_column!r} not found in {args.csv_path}")
    if args.group_column not in df.columns:
        raise ValueError(f"Group column {args.group_column!r} not found in {args.csv_path}")

    zone_numeric = df[zone_cols].apply(pd.to_numeric, errors="coerce")
    if args.drop_missing_zone_rows == "any":
        keep_mask = ~zone_numeric.isna().any(axis=1)
        df = df.loc[keep_mask].copy()
        zone_numeric = zone_numeric.loc[keep_mask].copy()
    elif args.drop_missing_zone_rows == "all":
        keep_mask = ~zone_numeric.isna().all(axis=1)
        df = df.loc[keep_mask].copy()
        zone_numeric = zone_numeric.loc[keep_mask].copy()
    else:
        df = df.copy()

    df["Image File"] = df[args.image_column].map(normalize_relative_path)
    if args.drop_missing_images:
        if not args.dataset_root:
            raise ValueError("--drop_missing_images requires --dataset_root.")
        dataset_root = os.path.abspath(args.dataset_root)
        image_exists = df["Image File"].map(lambda path: os.path.exists(os.path.join(dataset_root, path)))
        missing_count = int((~image_exists).sum())
        if missing_count:
            print(f"Dropping {missing_count} rows with missing images under {dataset_root}")
        df = df.loc[image_exists].copy()
        zone_numeric = zone_numeric.loc[image_exists].copy()

    df["AllZoneLabelsMissing"] = zone_numeric.isna().all(axis=1).astype(int)
    df["Label"] = build_label_from_zones(df, zone_cols)
    df["SplitGroup"] = df[args.group_column].astype(str)

    unique_groups = np.sort(df["SplitGroup"].unique())
    if len(unique_groups) < args.n_val + args.n_folds:
        raise ValueError("Not enough groups for requested n_folds and n_val.")

    shuffled_groups = np.random.permutation(unique_groups)
    folds = np.array_split(shuffled_groups, args.n_folds)
    os.makedirs(args.output_root, exist_ok=True)

    summary_rows = []
    for fold_idx in range(args.n_folds):
        test_groups = np.sort(folds[fold_idx])
        remaining_groups = np.concatenate([folds[j] for j in range(args.n_folds) if j != fold_idx])
        val_groups = np.sort(np.random.choice(remaining_groups, size=args.n_val, replace=False))
        train_groups = np.sort(list(set(remaining_groups) - set(val_groups)))

        test_df = df[df["SplitGroup"].isin(test_groups)].copy()
        val_df = df[df["SplitGroup"].isin(val_groups)].copy()
        train_df = df[df["SplitGroup"].isin(train_groups)].copy()
        train_final_df = pd.concat([train_df, val_df], ignore_index=True)

        for split_df in (test_df, val_df, train_df, train_final_df):
            split_df.sort_values(by="Image File", inplace=True)
            split_df.drop(columns=["SplitGroup"], inplace=True)

        fold_dir = os.path.join(args.output_root, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        split_payloads = {
            "test": test_df,
            "val": val_df,
            "train": train_df,
            "train_final": train_final_df,
        }
        for split_name, split_df in split_payloads.items():
            split_df.to_csv(os.path.join(fold_dir, f"{split_name}.csv"), index=False)
            summary_rows.append(
                {
                    "fold": fold_idx,
                    "split": split_name,
                    "rows": len(split_df),
                    "patients": split_df[args.group_column].nunique(),
                    "positive_rows": int((split_df["Label"] != "negative").sum()),
                    "negative_rows": int((split_df["Label"] == "negative").sum()),
                }
            )

        save_class_weights(fold_dir, train_df)
        save_graded_weights(fold_dir, train_df)
        os.rename(os.path.join(fold_dir, "classWeights.npy"), os.path.join(fold_dir, "classWeights_train.npy"))
        os.rename(os.path.join(fold_dir, "gradedWeights.npy"), os.path.join(fold_dir, "gradedWeights_train.npy"))

        save_class_weights(fold_dir, train_final_df)
        save_graded_weights(fold_dir, train_final_df)
        os.rename(os.path.join(fold_dir, "classWeights.npy"), os.path.join(fold_dir, "classWeights_final.npy"))
        os.rename(os.path.join(fold_dir, "gradedWeights.npy"), os.path.join(fold_dir, "gradedWeights_final.npy"))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_root, "fold_summary.csv"), index=False)
    print(f"Wrote {args.n_folds} folds to {args.output_root}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
