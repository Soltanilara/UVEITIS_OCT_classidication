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


JOIN_KEY = "UWFFA"
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
    return parser.parse_args()


def build_lookup(ready_df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in [JOIN_KEY, *ATTACH_COLUMNS] if col not in ready_df.columns]
    if missing:
        raise ValueError(f"Ready CSV is missing required columns: {missing}")

    lookup = (
        ready_df[[JOIN_KEY, *ATTACH_COLUMNS]]
        .drop_duplicates(subset=[JOIN_KEY], keep="first")
        .copy()
    )
    return lookup


def enrich_split(split_csv: Path, lookup: pd.DataFrame, drop_missing_mask: bool) -> pd.DataFrame:
    split_df = pd.read_csv(split_csv)
    if JOIN_KEY not in split_df.columns:
        raise ValueError(f"{split_csv} is missing join key column {JOIN_KEY!r}")

    merged = split_df.merge(lookup, on=JOIN_KEY, how="left", validate="m:1")
    matched = merged["FA_Mask_Abs_Path"].notna()

    if drop_missing_mask:
        merged = merged.loc[matched].reset_index(drop=True)

    return merged


def main() -> None:
    args = parse_args()
    ready_df = pd.read_csv(args.ready_csv)
    lookup = build_lookup(ready_df)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_names = ("train.csv", "val.csv", "test.csv", "train_final.csv")

    for fold_dir in sorted(input_root.glob("fold_*")):
        if not fold_dir.is_dir():
            continue

        out_fold_dir = output_root / fold_dir.name
        out_fold_dir.mkdir(parents=True, exist_ok=True)

        for split_name in split_names:
            split_csv = fold_dir / split_name
            if not split_csv.exists():
                continue
            enriched = enrich_split(split_csv, lookup, drop_missing_mask=args.drop_missing_mask)
            enriched.to_csv(out_fold_dir / split_name, index=False)
            print(f"Wrote {out_fold_dir / split_name} ({len(enriched)} rows)")


if __name__ == "__main__":
    main()
