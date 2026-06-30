#!/usr/bin/env python3
"""
Check that every FA image listed in the annotation workbook exists under the
canonical dataset root.

Run from the repository root, for example:

python scripts/check_annotated_fa_files.py \
  --annotations "UWFAFP_Annotations_Mo_4.5.2026 (Uveitis).xlsx" \
  --dataset-root "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import pandas as pd


DEFAULT_ANNOTATIONS = "UWFAFP_Annotations_Mo_4.5.2026 (Uveitis).xlsx"
DEFAULT_DATASET_ROOT = "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical"
DEFAULT_OUTPUT_DIR = "outputs/annotation_file_check"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default=DEFAULT_ANNOTATIONS, help="Path to annotation .xlsx file")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Canonical dataset root")
    parser.add_argument("--sheet", default="Data", help="Workbook sheet name")
    parser.add_argument("--path-column", default="Image_File(FA)", help="Column containing relative FA image paths")
    parser.add_argument("--patient-column", default="Patient_ID", help="Column containing patient IDs")
    parser.add_argument("--eye-column", default="Eye", help="Column containing OD/OS")
    parser.add_argument("--date-column", default="Visit_Date", help="Column containing visit dates")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/JSON reports")
    parser.add_argument(
        "--check-masks",
        action="store_true",
        help="Also check for a sibling *_masks.npy or *_masks_v2.npy file for each image",
    )
    return parser.parse_args()


def normalize_rel_path(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().strip('"').strip("'")
    if not text:
        return ""

    # Workbook paths are Windows-style; absolute Windows paths should keep only
    # the canonical Patient*/date/image suffix.
    windows_parts = PureWindowsPath(text).parts
    for idx, part in enumerate(windows_parts):
        if re.fullmatch(r"Patient\d+", part, flags=re.IGNORECASE):
            return str(Path(*windows_parts[idx:])).replace("\\", "/")

    return text.replace("\\", "/").lstrip("/")


def patient_dir_from_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"\d+(\.0)?", text):
        return f"Patient{int(float(text)):03d}"
    match = re.search(r"(\d+)", text)
    if match:
        return f"Patient{int(match.group(1)):03d}"
    return text


def date_dir_from_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return parsed.strftime("%Y%m%d")
    digits = re.sub(r"\D", "", text)
    return digits if len(digits) == 8 else text


def expected_from_columns(row: pd.Series, patient_col: str, date_col: str, eye_col: str) -> tuple[str, str, str]:
    patient = patient_dir_from_value(row.get(patient_col, ""))
    date = date_dir_from_value(row.get(date_col, ""))
    eye = "" if pd.isna(row.get(eye_col, "")) else str(row.get(eye_col, "")).strip().upper()
    return patient, date, eye


def mask_candidates_for(image_path: Path) -> list[Path]:
    return [
        image_path.with_name(f"{image_path.stem}_masks.npy"),
        image_path.with_name(f"{image_path.stem}_masks_v2.npy"),
    ]


def main() -> int:
    args = parse_args()
    annotations = Path(args.annotations).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not annotations.exists():
        print(f"[error] Annotation workbook not found: {annotations}", file=sys.stderr)
        return 2
    if not dataset_root.exists():
        print(f"[error] Dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    df = pd.read_excel(annotations, sheet_name=args.sheet, engine="openpyxl")
    required = [args.path_column, args.patient_column, args.eye_column, args.date_column]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        print(f"[error] Missing expected workbook columns: {missing_columns}", file=sys.stderr)
        print(f"[info] Available columns: {list(df.columns)}", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    for excel_row, row in df.iterrows():
        rel_path = normalize_rel_path(row[args.path_column])
        rel_parts = Path(rel_path).parts
        image_path = dataset_root / rel_path if rel_path else dataset_root / "__missing_path__"
        patient_expected, date_expected, eye_expected = expected_from_columns(
            row, args.patient_column, args.date_column, args.eye_column
        )

        patient_in_path = rel_parts[0] if len(rel_parts) >= 1 else ""
        date_in_path = rel_parts[1] if len(rel_parts) >= 2 else ""
        filename = rel_parts[-1] if rel_parts else ""

        filename_expected_prefix = "_".join(part for part in [patient_expected, date_expected, eye_expected, "FA"] if part)
        filename_matches_columns = bool(filename) and filename.startswith(filename_expected_prefix)
        location_matches_columns = patient_in_path == patient_expected and date_in_path == date_expected
        file_exists = image_path.is_file()

        mask_candidates = mask_candidates_for(image_path)
        existing_masks = [str(path) for path in mask_candidates if path.is_file()]

        status = "ok"
        issues: list[str] = []
        if not rel_path:
            issues.append("blank_annotation_path")
        if not location_matches_columns:
            issues.append("patient_or_date_location_mismatch")
        if not filename_matches_columns:
            issues.append("filename_mismatch_with_columns")
        if not file_exists:
            issues.append("missing_image_file")
        if args.check_masks and not existing_masks:
            issues.append("missing_mask_file")
        if issues:
            status = "problem"

        records.append(
            {
                "excel_row": int(excel_row) + 2,
                "status": status,
                "issues": ";".join(issues),
                "Patient_ID": row[args.patient_column],
                "Eye": row[args.eye_column],
                "Visit_Date": row[args.date_column],
                "annotation_path": row[args.path_column],
                "relative_path": rel_path,
                "absolute_path": str(image_path),
                "file_exists": file_exists,
                "expected_patient_dir": patient_expected,
                "actual_patient_dir": patient_in_path,
                "expected_date_dir": date_expected,
                "actual_date_dir": date_in_path,
                "filename": filename,
                "filename_matches_columns": filename_matches_columns,
                "mask_exists": bool(existing_masks) if args.check_masks else "",
                "existing_masks": ";".join(existing_masks),
            }
        )

    result_df = pd.DataFrame(records)
    problem_df = result_df[result_df["status"] != "ok"].copy()
    missing_df = result_df[result_df["issues"].str.contains("missing_image_file", na=False)].copy()

    all_csv = output_dir / "annotated_fa_file_check_all_rows.csv"
    problems_csv = output_dir / "annotated_fa_file_check_problems.csv"
    missing_csv = output_dir / "annotated_fa_file_check_missing_images.csv"
    summary_json = output_dir / "annotated_fa_file_check_summary.json"

    result_df.to_csv(all_csv, index=False)
    problem_df.to_csv(problems_csv, index=False)
    missing_df.to_csv(missing_csv, index=False)

    issue_counts: Counter[str] = Counter()
    for issue_text in result_df["issues"].dropna():
        for issue in str(issue_text).split(";"):
            if issue:
                issue_counts[issue] += 1

    summary = {
        "annotations": str(annotations),
        "dataset_root": str(dataset_root),
        "sheet": args.sheet,
        "rows_checked": int(len(result_df)),
        "ok_rows": int((result_df["status"] == "ok").sum()),
        "problem_rows": int(len(problem_df)),
        "missing_image_rows": int(len(missing_df)),
        "unique_missing_image_paths": int(missing_df["relative_path"].nunique()),
        "issue_counts": dict(sorted(issue_counts.items())),
        "reports": {
            "all_rows_csv": str(all_csv),
            "problems_csv": str(problems_csv),
            "missing_images_csv": str(missing_csv),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if len(problem_df):
        print(f"\n[problem_rows] See {problems_csv}")
        return 1
    print("\n[ok] Every annotated FA image exists in the expected canonical location.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
