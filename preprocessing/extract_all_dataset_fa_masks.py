#!/usr/bin/env python3
"""
Extract FA zone masks for every unique FA image listed in the dataset spreadsheet.

Outputs:
- per-image `_masks.npy` files written next to the source FA image
- JSONL log with one record per attempted image
- JSON summary with aggregate counts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mask import run_mask_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx_path", required=True, help="Path to the dataset spreadsheet")
    parser.add_argument("--image_dir", required=True, help="Dataset root that contains Patient*/... image paths")
    parser.add_argument("--sheet_name", default=0, help="Excel sheet name or index")
    parser.add_argument("--path_column", default="Image_File(FA)", help="Spreadsheet column containing FA image paths")
    parser.add_argument("--log_dir", required=True, help="Directory where logs and summaries will be written")
    parser.add_argument(
        "--require_token",
        default="_FA_",
        help="Only process image paths containing this token. Set empty string to disable filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.xlsx_path, sheet_name=args.sheet_name)
    if args.path_column not in df.columns:
        raise SystemExit(f"Missing expected column: {args.path_column}")

    paths = (
        df[args.path_column]
        .dropna()
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .drop_duplicates()
        .tolist()
    )
    if args.require_token:
        original_count = len(paths)
        paths = [path for path in paths if args.require_token in Path(path).name]
        print(f"[filter] Kept {len(paths)} / {original_count} paths containing token {args.require_token!r}")

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fa_mask_extraction.jsonl"
    summary_path = log_dir / "fa_mask_extraction_summary.json"

    print(f"[count] Processing {len(paths)} unique FA images from {args.xlsx_path}")
    summary = run_mask_extraction(
        image_dir=args.image_dir,
        paths=paths,
        log_path=log_path,
        summary_path=summary_path,
    )
    print(f"[log] {log_path}")
    print(f"[summary] {summary_path}")
    print(summary)


if __name__ == "__main__":
    main()
