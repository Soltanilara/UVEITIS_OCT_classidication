#!/usr/bin/env python3
"""
Build final FA dataset CSVs for training from the corrected path mapping and
all remediation passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing Patient*/... image folders")
    parser.add_argument("--corrected_csv", required=True, help="Corrected row-level FA CSV")
    parser.add_argument("--mapping_csv", required=True, help="Unique corrected FA path mapping CSV")
    parser.add_argument("--remediation_csv", required=True, help="Final remaining-failure remediation CSV")
    parser.add_argument("--unresolved_csv", required=True, help="Unique unresolved FA path CSV")
    parser.add_argument("--output_dir", required=True, help="Directory for final training CSV outputs")
    return parser.parse_args()


def mask_rel_path_for_image(rel_path: str) -> str:
    path = Path(rel_path)
    return str(path.with_name(path.stem + "_masks.npy")).replace("\\", "/")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    corrected_df = pd.read_csv(args.corrected_csv)
    mapping_df = pd.read_csv(args.mapping_csv)
    remediation_df = pd.read_csv(args.remediation_csv)
    unresolved_df = pd.read_csv(args.unresolved_csv)

    remediation_lookup = remediation_df.rename(
        columns={
            "input_path": "Image_File(FA)",
            "strategy": "FA_Remediation_Strategy",
            "detail": "FA_Remediation_Detail",
            "status": "FA_Remediation_Status",
        }
    )[
        ["Image_File(FA)", "FA_Remediation_Status", "FA_Remediation_Strategy", "FA_Remediation_Detail"]
    ]

    unique_df = mapping_df.copy()
    unique_df = unique_df.rename(
        columns={
            "original_path": "Original_Image_File(FA)",
            "corrected_path": "Image_File(FA)",
            "resolution_status": "FA_Path_Resolution_Status",
            "correction_type": "FA_Path_Correction_Type",
            "note": "FA_Path_Correction_Note",
        }
    )
    unique_df = unique_df.merge(
        remediation_lookup,
        on="Image_File(FA)",
        how="left",
    )

    unique_df["FA_Mask_Path"] = unique_df["Image_File(FA)"].map(mask_rel_path_for_image)
    unique_df["FA_Mask_Abs_Path"] = unique_df["FA_Mask_Path"].map(lambda p: str((dataset_root / p).resolve()))
    unique_df["FA_Image_Abs_Path"] = unique_df["Image_File(FA)"].map(lambda p: str((dataset_root / p).resolve()))
    unique_df["FA_Mask_Exists"] = unique_df["FA_Mask_Path"].map(lambda p: (dataset_root / p).exists())

    def final_method(row: pd.Series) -> str:
        if isinstance(row.get("FA_Remediation_Strategy"), str) and row["FA_Remediation_Strategy"]:
            return row["FA_Remediation_Strategy"]
        if row["FA_Path_Resolution_Status"] == "unresolved":
            return "missing_source_image"
        if row["FA_Path_Correction_Type"] == "exact_exists":
            return "direct_extraction"
        return f"path_corrected:{row['FA_Path_Correction_Type']}"

    unique_df["FA_Final_Recovery_Method"] = unique_df.apply(final_method, axis=1)
    unique_df["FA_Final_Status"] = unique_df["FA_Mask_Exists"].map(lambda ok: "ready" if ok else "missing_mask")

    row_df = corrected_df.copy()
    row_df = row_df.merge(
        unique_df[
            [
                "Original_Image_File(FA)",
                "Image_File(FA)",
                "FA_Mask_Path",
                "FA_Mask_Abs_Path",
                "FA_Image_Abs_Path",
                "FA_Mask_Exists",
                "FA_Final_Recovery_Method",
                "FA_Final_Status",
                "FA_Remediation_Status",
                "FA_Remediation_Strategy",
                "FA_Remediation_Detail",
            ]
        ],
        on=["Original_Image_File(FA)", "Image_File(FA)"],
        how="left",
    )

    unique_output = output_dir / "fa_final_master_unique_paths.csv"
    unique_ready_output = output_dir / "fa_final_master_unique_paths_ready.csv"
    row_output = output_dir / "fa_final_training_dataset.csv"
    row_ready_output = output_dir / "fa_final_training_dataset_ready.csv"
    unresolved_output = output_dir / "fa_final_missing_source_rows.csv"

    unique_df.to_csv(unique_output, index=False)
    unique_df[unique_df["FA_Final_Status"] == "ready"].to_csv(unique_ready_output, index=False)
    row_df.to_csv(row_output, index=False)
    row_df[row_df["FA_Final_Status"] == "ready"].to_csv(row_ready_output, index=False)
    row_df[row_df["FA_Final_Status"] != "ready"].to_csv(unresolved_output, index=False)

    summary = {
        "unique_fa_paths": int(len(unique_df)),
        "unique_ready_masks": int(unique_df["FA_Mask_Exists"].sum()),
        "unique_missing_masks": int((~unique_df["FA_Mask_Exists"]).sum()),
        "row_count": int(len(row_df)),
        "row_ready_masks": int((row_df["FA_Final_Status"] == "ready").sum()),
        "row_missing_masks": int((row_df["FA_Final_Status"] != "ready").sum()),
        "recovery_method_counts": row_df["FA_Final_Recovery_Method"].value_counts(dropna=False).to_dict(),
    }
    with open(output_dir / "fa_final_training_dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "# Final FA Training Dataset",
        "",
        f"- Unique FA paths: `{summary['unique_fa_paths']}`",
        f"- Unique masks ready: `{summary['unique_ready_masks']}`",
        f"- Unique masks missing: `{summary['unique_missing_masks']}`",
        f"- Training rows: `{summary['row_count']}`",
        f"- Training rows ready: `{summary['row_ready_masks']}`",
        f"- Training rows missing mask: `{summary['row_missing_masks']}`",
        "",
        "## Recovery Methods",
        "",
    ]
    for method, count in summary["recovery_method_counts"].items():
        report_lines.append(f"- `{method}`: `{count}`")
    (output_dir / "fa_final_training_dataset_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[unique_csv] {unique_output}")
    print(f"[unique_ready_csv] {unique_ready_output}")
    print(f"[training_csv] {row_output}")
    print(f"[training_ready_csv] {row_ready_output}")
    print(f"[missing_csv] {unresolved_output}")
    print(summary)


if __name__ == "__main__":
    main()
