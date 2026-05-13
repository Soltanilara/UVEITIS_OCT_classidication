#!/usr/bin/env python3
"""
Attach FA-derived mask columns from the ready dataset CSV onto existing fold CSVs.

This keeps the original fundus `Image File` column intact, and adds:
- `FA_Mask_Path`
- `FA_Mask_Abs_Path`
- `FA_Image_Abs_Path`
- `FA_Mask_Exists`

Rows without a matched ready mask can optionally be dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


JOIN_KEY_CANDIDATES = ("UWFFA", "Image_File(FA)")
ATTACH_COLUMNS = [
    "FA_Mask_Path",
    "FA_Mask_Abs_Path",
    "FA_Image_Abs_Path",
    "FA_Mask_Exists",
    "FA_Final_Status",
    "FA_Final_Recovery_Method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-csv", required=True, help="Ready dataset CSV with FA mask columns.")
    parser.add_argument("--input-root", required=True, help="Directory containing fold_*/ train/val/test CSVs.")
    parser.add_argument("--output-root", required=True, help="Directory where enriched fold_* CSVs will be written.")
    parser.add_argument("--drop-missing-mask", action="store_true", help="Drop split rows that do not have a matched mask.")
    parser.add_argument("--dataset-root", default="", help="Optional dataset root used to verify that the image file exists.")
    parser.add_argument(
        "--mask-suffix",
        default="",
        help="Optional replacement suffix for FA mask files, e.g. _masks_v2.npy.",
    )
    parser.add_argument("--image-column", default="Image File", help="Relative image-path column to validate under --dataset-root.")
    parser.add_argument("--drop-missing-image", action="store_true", help="Drop split rows whose image file does not exist under --dataset-root.")
    return parser.parse_args()


def normalize_join_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
    )


def pick_join_key(ready_df: pd.DataFrame, split_df: pd.DataFrame | None = None) -> str:
    candidates = [key for key in JOIN_KEY_CANDIDATES if key in ready_df.columns]
    if split_df is not None:
        candidates = [key for key in candidates if key in split_df.columns]

    if not candidates:
        ready_cols = ", ".join(ready_df.columns)
        split_cols = ", ".join(split_df.columns) if split_df is not None else ""
        raise ValueError(
            "Could not find a compatible join key. "
            f"Expected one of {JOIN_KEY_CANDIDATES}. "
            f"Ready CSV columns: [{ready_cols}]"
            + (f" Split CSV columns: [{split_cols}]" if split_cols else "")
        )

    return candidates[0]


def build_lookup(ready_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    missing = [col for col in [join_key, *ATTACH_COLUMNS] if col not in ready_df.columns]
    if missing:
        raise ValueError(f"Ready CSV is missing required columns: {missing}")

    lookup = (
        ready_df[[join_key, *ATTACH_COLUMNS]]
        .drop_duplicates(subset=[join_key], keep="first")
        .copy()
    )
    lookup[join_key] = normalize_join_series(lookup[join_key])
    return lookup


def replace_mask_suffix(path_value: str, mask_suffix: str) -> str:
    path = Path(str(path_value))
    stem = path.stem
    if "_masks" in stem:
        stem = stem[:stem.rfind("_masks")]
    return str(path.with_name(stem + mask_suffix))


def apply_mask_suffix_override(lookup: pd.DataFrame, mask_suffix: str) -> pd.DataFrame:
    if not mask_suffix:
        return lookup

    updated = lookup.copy()
    updated["FA_Mask_Path"] = updated["FA_Mask_Path"].map(lambda value: replace_mask_suffix(value, mask_suffix))
    updated["FA_Mask_Abs_Path"] = updated["FA_Mask_Abs_Path"].map(lambda value: replace_mask_suffix(value, mask_suffix))
    updated["FA_Mask_Exists"] = updated["FA_Mask_Abs_Path"].map(lambda value: Path(str(value)).exists())
    return updated


def enrich_split(
    split_csv: Path,
    lookup: pd.DataFrame,
    join_key: str,
    drop_missing_mask: bool,
    dataset_root: Path | None,
    image_column: str,
    drop_missing_image: bool,
) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    if join_key not in split_df.columns:
        raise ValueError(f"{split_csv} is missing join key column {join_key!r}")

    split_df = split_df.copy()
    overlapping_attach_cols = [col for col in ATTACH_COLUMNS if col in split_df.columns]
    if overlapping_attach_cols:
        split_df = split_df.drop(columns=overlapping_attach_cols)
    split_df[join_key] = normalize_join_series(split_df[join_key])
    merged = split_df.merge(lookup, on=join_key, how="left", validate="m:1")
    matched = merged["FA_Mask_Abs_Path"].notna()

    if drop_missing_mask:
        merged = merged.loc[matched].reset_index(drop=True)

    if drop_missing_image:
        if dataset_root is None:
            raise ValueError("--drop-missing-image requires --dataset-root.")
        if image_column not in merged.columns:
            raise ValueError(f"{split_csv} is missing image column {image_column!r}")
        exists = merged[image_column].map(lambda rel: (dataset_root / str(rel)).exists())
        merged = merged.loc[exists].reset_index(drop=True)

    return merged


def main() -> None:
    args = parse_args()
    ready_df = pd.read_csv(args.ready_csv)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root) if args.dataset_root else None

    split_names = ("train.csv", "val.csv", "test.csv", "train_final.csv")
    lookup = None
    join_key = None

    for fold_dir in sorted(input_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue

        out_fold_dir = output_root / fold_dir.name
        out_fold_dir.mkdir(parents=True, exist_ok=True)

        for split_name in split_names:
            split_csv = fold_dir / split_name
            if not split_csv.exists():
                continue
            if lookup is None:
                split_df = pd.read_csv(split_csv, nrows=1)
                join_key = pick_join_key(ready_df, split_df)
                lookup = build_lookup(ready_df, join_key)
                lookup = apply_mask_suffix_override(lookup, args.mask_suffix)
                print(f"Using join key: {join_key}")
            enriched = enrich_split(
                split_csv,
                lookup,
                join_key,
                drop_missing_mask=args.drop_missing_mask,
                dataset_root=dataset_root,
                image_column=args.image_column,
                drop_missing_image=args.drop_missing_image,
            )
            enriched.to_csv(out_fold_dir / split_name, index=False)
            print(f"Wrote {out_fold_dir / split_name} ({len(enriched)} rows)")


if __name__ == "__main__":
    main()
