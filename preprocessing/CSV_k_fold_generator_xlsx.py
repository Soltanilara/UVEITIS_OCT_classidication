import argparse
import os
import random

import numpy as np
import pandas as pd


def normalize_relative_path(path_val: str) -> str:
    """Convert Windows-style absolute path to patient-relative path."""
    p = str(path_val).replace("\\", "/")
    idx = p.lower().find("patient")
    return p[idx:] if idx >= 0 else p


def save_class_weights(prefix_path: str, df_subset: pd.DataFrame) -> None:
    label_counts = df_subset["Label"].value_counts()
    total = label_counts.sum()
    if total == 0:
        np.save(os.path.join(prefix_path, "classWeights.npy"), np.array([1.0, 1.0]))
        return

    neg_count = label_counts.get("negative", 0)
    weight_negative = 2 * neg_count / total
    weight_positive = 2 - weight_negative
    class_weights = np.array([weight_positive, weight_negative])
    np.save(os.path.join(prefix_path, "classWeights.npy"), class_weights)


def save_graded_weights(prefix_path: str, df_subset: pd.DataFrame) -> None:
    label_categories = ["negative", "mild", "moderate", "severe"]
    label_counts = df_subset["Label"].value_counts()
    total_samples = len(df_subset)

    if total_samples == 0:
        np.save(os.path.join(prefix_path, "gradedWeights.npy"), np.array([0.25, 0.25, 0.25, 0.25]))
        return

    weights = np.zeros(len(label_categories), dtype=np.float32)
    for i, label in enumerate(label_categories):
        if label in label_counts:
            weights[i] = total_samples / label_counts[label]
        else:
            weights[i] = 0.0

    if weights.sum() > 0:
        weights = weights / weights.sum()

    np.save(os.path.join(prefix_path, "gradedWeights.npy"), weights)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-level k-fold CSVs from XLSX while preserving zone columns.")
    parser.add_argument("--xlsx_path", type=str, required=True, help="Path to annotation XLSX file.")
    parser.add_argument("--sheet_name", type=str, default="Data", help="Excel sheet name.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--n_val", type=int, default=10, help="Number of validation patients per fold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out_prefix", type=str, default="fold", help="Output directory prefix.")
    parser.add_argument(
        "--image_column",
        type=str,
        default="auto",
        help="Image-path column to use. Set to 'auto', 'UWFFP', or 'Image_File(FA)'.",
    )
    parser.add_argument(
        "--group_column",
        type=str,
        default="auto",
        help="Grouping column for patient-level separation. Set to 'auto' to prefer Patient_ID.",
    )
    parser.add_argument(
        "--drop_missing_zone_rows",
        type=str,
        default="none",
        choices=["none", "any", "all"],
        help="Drop rows where zone labels are missing: none, any, or all.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    df = pd.read_excel(args.xlsx_path, sheet_name=args.sheet_name)
    zone_cols = [c for c in df.columns if c.startswith("Zone") and c.endswith("_label")]
    if not zone_cols:
        raise ValueError("No zone label columns found (expected Zone*_label).")

    if args.image_column != "auto":
        if args.image_column not in df.columns:
            raise ValueError(f"Requested image column '{args.image_column}' not found in XLSX.")
        path_col = args.image_column
    elif "UWFFP" in df.columns:
        path_col = "UWFFP"
    elif "Image_File(FA)" in df.columns:
        path_col = "Image_File(FA)"
    else:
        raise ValueError("Expected either 'UWFFP' or 'Image_File(FA)' column.")

    df["Image File"] = df[path_col].apply(normalize_relative_path)
    df["Prefix"] = df["Image File"].apply(lambda x: str(x).split("/")[0])
    if args.group_column != "auto":
        if args.group_column not in df.columns:
            raise ValueError(f"Requested group column '{args.group_column}' not found in XLSX.")
        df["SplitGroup"] = df[args.group_column].astype(str)
    elif "Patient_ID" in df.columns:
        df["SplitGroup"] = df["Patient_ID"].astype(str)
    else:
        df["SplitGroup"] = df["Prefix"].astype(str)

    # Optionally remove rows with missing zone labels before split creation.
    zone_numeric = df[zone_cols].apply(pd.to_numeric, errors="coerce")
    if args.drop_missing_zone_rows == "any":
        keep_mask = ~zone_numeric.isna().any(axis=1)
        df = df[keep_mask].copy()
        zone_numeric = zone_numeric[keep_mask].copy()
    elif args.drop_missing_zone_rows == "all":
        keep_mask = ~zone_numeric.isna().all(axis=1)
        df = df[keep_mask].copy()
        zone_numeric = zone_numeric[keep_mask].copy()

    df["AllZoneLabelsMissing"] = zone_numeric.isna().all(axis=1).astype(int)
    max_zone = zone_numeric.max(axis=1).fillna(0).round().astype(int)
    label_map = {0: "negative", 1: "mild", 2: "moderate", 3: "severe"}
    df["Label"] = max_zone.map(label_map).fillna("negative")

    unique_groups = np.sort(df["SplitGroup"].unique())
    num_groups = len(unique_groups)
    if num_groups < args.n_val + args.n_folds:
        raise ValueError("Not enough patients for requested n_folds and n_val.")

    shuffled_groups = np.random.permutation(unique_groups)
    folds = np.array_split(shuffled_groups, args.n_folds)

    for i in range(args.n_folds):
        test_groups = np.sort(folds[i])
        remaining_groups = np.concatenate([folds[j] for j in range(args.n_folds) if j != i])
        val_groups = np.sort(np.random.choice(remaining_groups, size=args.n_val, replace=False))
        train_groups = np.sort(list(set(remaining_groups) - set(val_groups)))

        test_df = df[df["SplitGroup"].isin(test_groups)].copy()
        val_df = df[df["SplitGroup"].isin(val_groups)].copy()
        train_df = df[df["SplitGroup"].isin(train_groups)].copy()
        train_final_df = pd.concat([train_df, val_df], ignore_index=True)

        for split_df in (test_df, val_df, train_df, train_final_df):
            split_df.sort_values(by="Image File", inplace=True)

        # Keep all original columns + Image File/Label/AllZoneLabelsMissing; drop internal Prefix.
        for split_df in (test_df, val_df, train_df, train_final_df):
            split_df.drop(columns=["Prefix", "SplitGroup"], inplace=True)

        fold_dir = f"{args.out_prefix}_{i}"
        os.makedirs(fold_dir, exist_ok=True)

        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        train_final_df.to_csv(os.path.join(fold_dir, "train_final.csv"), index=False)

        save_class_weights(fold_dir, train_df)
        save_graded_weights(fold_dir, train_df)
        os.rename(os.path.join(fold_dir, "classWeights.npy"), os.path.join(fold_dir, "classWeights_train.npy"))
        os.rename(os.path.join(fold_dir, "gradedWeights.npy"), os.path.join(fold_dir, "gradedWeights_train.npy"))

        save_class_weights(fold_dir, train_final_df)
        save_graded_weights(fold_dir, train_final_df)
        os.rename(os.path.join(fold_dir, "classWeights.npy"), os.path.join(fold_dir, "classWeights_final.npy"))
        os.rename(os.path.join(fold_dir, "gradedWeights.npy"), os.path.join(fold_dir, "gradedWeights_final.npy"))

    print(
        f"{args.n_folds}-fold patient-level splits created successfully from XLSX: {args.xlsx_path} "
        f"(image_column={path_col}, group_column={'auto->Patient_ID' if args.group_column == 'auto' and 'Patient_ID' in df.columns else args.group_column})"
    )


if __name__ == "__main__":
    main()
