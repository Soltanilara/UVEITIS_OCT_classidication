import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

FALLBACK_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def normalize_relative_path(path_val: str) -> str:
    path = str(path_val).replace("\\", "/").strip()
    patient_idx = path.lower().find("patient")
    return path[patient_idx:] if patient_idx >= 0 else path


def _strip_sequence_token(stem: str) -> str:
    parts = stem.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def resolve_existing_image(dataset_root: str, rel_path: str) -> tuple[str | None, str]:
    rel_path = normalize_relative_path(rel_path)
    candidate = Path(dataset_root) / rel_path
    if candidate.exists():
        return rel_path, "exact_path"

    root = candidate.with_suffix("")
    for ext in FALLBACK_EXTS:
        for ext_variant in (ext, ext.upper()):
            alt = root.with_suffix(ext_variant)
            if alt.exists():
                return str(alt.relative_to(dataset_root)).replace("\\", "/"), "extension_fallback"

    rel_parts = Path(rel_path).parts
    if not rel_parts:
        return None, "empty_path"

    patient_dir = Path(dataset_root) / rel_parts[0]
    if not patient_dir.is_dir():
        return None, "missing_patient_dir"

    requested_name = Path(rel_path).name.lower()
    requested_stem = Path(rel_path).stem.lower()
    relaxed_stem = _strip_sequence_token(requested_stem)
    requested_tokens = [token for token in relaxed_stem.split("_") if token]

    search_roots = []
    if len(rel_parts) > 1:
        date_dir = patient_dir / rel_parts[1]
        if date_dir.is_dir():
            search_roots.append(date_dir)
    search_roots.append(patient_dir)

    matches = {"exact_name": [], "same_stem": [], "relaxed_stem": [], "tokens": []}
    seen_roots = set()
    for search_root in search_roots:
        if search_root in seen_roots:
            continue
        seen_roots.add(search_root)
        for walk_root, _, filenames in os.walk(search_root):
            for fname in filenames:
                f_path = Path(walk_root) / fname
                if f_path.suffix.lower() not in FALLBACK_EXTS:
                    continue
                fname_lower = fname.lower()
                f_stem_lower = f_path.stem.lower()
                f_relaxed_stem = _strip_sequence_token(f_stem_lower)
                if fname_lower == requested_name:
                    matches["exact_name"].append(f_path)
                elif f_stem_lower == requested_stem:
                    matches["same_stem"].append(f_path)
                elif f_relaxed_stem == relaxed_stem:
                    matches["relaxed_stem"].append(f_path)
                elif all(token in fname_lower for token in requested_tokens):
                    matches["tokens"].append(f_path)

    for match_type in ("exact_name", "same_stem", "relaxed_stem", "tokens"):
        candidates = sorted(set(matches[match_type]))
        if candidates:
            if len(candidates) > 1:
                print(f"Warning: multiple {match_type} matches for {rel_path}; using {candidates[0]}")
            return str(candidates[0].relative_to(dataset_root)).replace("\\", "/"), match_type

    return None, "unresolved"


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
    parser.add_argument("--drop_missing_images", action="store_true", help="Drop rows whose Image File cannot be resolved under --dataset_root.")
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
        original_paths = df["Image File"].copy()
        resolved_records = df["Image File"].map(lambda path: resolve_existing_image(dataset_root, path))
        resolved_paths = resolved_records.map(lambda record: record[0])
        resolve_methods = resolved_records.map(lambda record: record[1])
        corrected_count = int(
            ((resolved_paths.notna()) & (resolved_paths != original_paths)).sum()
        )
        if corrected_count:
            print(f"Corrected {corrected_count} image paths using filesystem matches under {dataset_root}")
        image_exists = resolved_paths.notna()
        missing_count = int((~image_exists).sum())
        if missing_count:
            print(f"Dropping {missing_count} rows with unresolved images under {dataset_root}")
        audit_df = df.copy()
        audit_df["Original Image File"] = original_paths
        audit_df["Resolved Image File"] = resolved_paths
        audit_df["Image Resolve Method"] = resolve_methods
        audit_df["Image Path Changed"] = (resolved_paths.notna()) & (resolved_paths != original_paths)
        os.makedirs(args.output_root, exist_ok=True)
        audit_df.loc[audit_df["Image Path Changed"]].to_csv(
            os.path.join(args.output_root, "image_path_corrections.csv"),
            index=False,
        )
        audit_df.loc[~image_exists].to_csv(
            os.path.join(args.output_root, "unresolved_image_paths.csv"),
            index=False,
        )
        df = df.loc[image_exists].copy()
        zone_numeric = zone_numeric.loc[image_exists].copy()
        df["Image File"] = resolved_paths.loc[image_exists].values

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
