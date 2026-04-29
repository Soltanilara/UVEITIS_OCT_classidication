#!/usr/bin/env python3
"""
Create corrected FA path mappings from the dataset spreadsheet and rerun mask extraction
for recoverable path-mismatch failures.
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

from mask import run_mask_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx_path", required=True, help="Path to the source spreadsheet")
    parser.add_argument("--image_dir", required=True, help="Dataset root containing Patient*/... folders")
    parser.add_argument(
        "--failure_csv",
        required=True,
        help="Failure CSV from the initial FA-only preprocessing pass",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where corrected CSVs, mapping files, reports, and rerun logs will be written",
    )
    parser.add_argument("--sheet_name", default=0, help="Excel sheet name or index")
    parser.add_argument("--path_column", default="Image_File(FA)", help="Spreadsheet FA path column")
    return parser.parse_args()


def normalize_rel_path(path_value: str) -> str:
    return str(path_value).replace("\\", "/")


def classify_path(rel_path: str, image_dir: Path) -> dict[str, object]:
    rel = Path(rel_path)
    abs_path = image_dir / rel
    result = {
        "original_path": rel.as_posix(),
        "corrected_path": rel.as_posix(),
        "resolution_status": "resolved" if abs_path.exists() else "unresolved",
        "correction_type": "exact_exists" if abs_path.exists() else "unresolved",
        "resolved_exists": abs_path.exists(),
        "candidate_count": 0,
        "note": "",
    }
    if abs_path.exists():
        return result

    directory = image_dir / rel.parent
    if not directory.exists():
        result["note"] = "parent directory missing"
        return result

    files = sorted(path.name for path in directory.iterdir() if path.is_file())
    fa_files = [name for name in files if "_FA_" in name]
    same_eye_fa = [
        name for name in fa_files
        if ("_OD_" in rel.name and "_OD_" in name) or ("_OS_" in rel.name and "_OS_" in name)
    ]
    result["candidate_count"] = len(same_eye_fa)

    candidates: list[tuple[str, str]] = []
    png_name = rel.with_suffix(".png").name
    if png_name in files:
        candidates.append(("extension_to_png", png_name))

    swapped_0001 = rel.name.replace("_FA_0000", "_FA_0001")
    if swapped_0001 in files:
        candidates.append(("swap_0000_to_0001", swapped_0001))

    swapped_0000 = rel.name.replace("_FA_0001", "_FA_0000")
    if swapped_0000 in files:
        candidates.append(("swap_0001_to_0000", swapped_0000))

    truncated = rel.name.replace("_FA_000.png", "_FA_0001.png")
    if truncated in files:
        candidates.append(("truncated_to_0001", truncated))

    if len(same_eye_fa) == 1:
        candidates.append(("single_same_eye_fa", same_eye_fa[0]))

    seen = set()
    unique_candidates: list[tuple[str, str]] = []
    for correction_type, name in candidates:
        if name in seen:
            continue
        seen.add(name)
        unique_candidates.append((correction_type, name))

    if len(unique_candidates) == 1:
        correction_type, corrected_name = unique_candidates[0]
        corrected_rel = (rel.parent / corrected_name).as_posix()
        result.update(
            {
                "corrected_path": corrected_rel,
                "resolution_status": "resolved",
                "correction_type": correction_type,
                "resolved_exists": True,
            }
        )
        return result

    if len(unique_candidates) > 1:
        result["note"] = "multiple plausible candidates"
    elif fa_files:
        result["note"] = f"available_fa_files={fa_files}"
    else:
        result["note"] = "no FA files in directory"
    return result


def build_mapping_df(xlsx_path: Path, image_dir: Path, sheet_name: str | int, path_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if path_column not in df.columns:
        raise SystemExit(f"Missing expected FA path column: {path_column}")

    df[path_column] = df[path_column].astype(str).map(normalize_rel_path)
    df = df[df[path_column].str.contains("_FA_", regex=False, na=False)].copy()

    unique_paths = df[path_column].dropna().drop_duplicates().tolist()
    mapping_rows = [classify_path(path, image_dir) for path in unique_paths]
    mapping_df = pd.DataFrame(mapping_rows)

    corrected_df = df.copy()
    corrected_df["Original_Image_File(FA)"] = corrected_df[path_column]
    corrected_df = corrected_df.merge(
        mapping_df[["original_path", "corrected_path", "resolution_status", "correction_type", "note"]],
        left_on=path_column,
        right_on="original_path",
        how="left",
    )
    corrected_df[path_column] = corrected_df["corrected_path"].fillna(corrected_df[path_column])
    corrected_df.rename(
        columns={
            "resolution_status": "FA_Path_Resolution_Status",
            "correction_type": "FA_Path_Correction_Type",
            "note": "FA_Path_Correction_Note",
        },
        inplace=True,
    )
    corrected_df.drop(columns=["original_path", "corrected_path"], inplace=True)
    return mapping_df, corrected_df


def create_report(
    mapping_df: pd.DataFrame,
    rerun_summary: dict[str, object],
    unresolved_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    correction_counts = mapping_df["correction_type"].value_counts().to_dict()
    lines = [
        "# FA Path Auto-Fix Report",
        "",
        f"- Total unique FA paths reviewed: `{len(mapping_df)}`",
        f"- Resolved paths: `{int((mapping_df['resolution_status'] == 'resolved').sum())}`",
        f"- Unresolved paths: `{int((mapping_df['resolution_status'] == 'unresolved').sum())}`",
        "",
        "## Correction Counts",
        "",
    ]
    for correction_type, count in correction_counts.items():
        lines.append(f"- `{correction_type}`: `{count}`")

    lines.extend(
        [
            "",
            "## Rerun Summary",
            "",
            f"- Total rerun paths: `{rerun_summary['total']}`",
            f"- Recovered ok: `{rerun_summary['ok']}`",
            f"- Remaining errors: `{rerun_summary['errors']}`",
            "",
            "## Unresolved Paths",
            "",
        ]
    )
    if unresolved_rows.empty:
        lines.append("- None")
    else:
        for _, row in unresolved_rows.iterrows():
            note = row["note"] if isinstance(row["note"], str) and row["note"] else "no matching file"
            lines.append(f"- `{row['original_path']}`: {note}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    xlsx_path = Path(args.xlsx_path).resolve()
    image_dir = Path(args.image_dir).resolve()
    failure_csv = Path(args.failure_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_df, corrected_df = build_mapping_df(
        xlsx_path=xlsx_path,
        image_dir=image_dir,
        sheet_name=args.sheet_name,
        path_column=args.path_column,
    )

    mapping_csv = output_dir / "fa_path_mapping_unique.csv"
    corrected_csv = output_dir / "annotations_fa_paths_corrected.csv"
    unresolved_csv = output_dir / "fa_path_mapping_unresolved.csv"
    mapping_df.to_csv(mapping_csv, index=False)
    corrected_df.to_csv(corrected_csv, index=False)
    mapping_df[mapping_df["resolution_status"] == "unresolved"].to_csv(unresolved_csv, index=False)

    failure_df = pd.read_csv(failure_csv)
    missing_df = failure_df[failure_df["error_category"] == "missing_or_unreadable_file"].copy()
    recoverable_df = missing_df.merge(
        mapping_df[["original_path", "corrected_path", "resolution_status", "correction_type"]],
        left_on="input_path",
        right_on="original_path",
        how="left",
    )
    rerun_paths = (
        recoverable_df[
            (recoverable_df["resolution_status"] == "resolved")
            & (recoverable_df["corrected_path"] != recoverable_df["original_path"])
        ]["corrected_path"]
        .drop_duplicates()
        .tolist()
    )

    rerun_log = output_dir / "rerun_fa_mask_extraction.jsonl"
    rerun_summary_path = output_dir / "rerun_fa_mask_extraction_summary.json"
    rerun_summary = run_mask_extraction(
        image_dir=image_dir,
        paths=rerun_paths,
        log_path=rerun_log,
        summary_path=rerun_summary_path,
    )

    recoverable_df.to_csv(output_dir / "recoverable_missing_path_rows.csv", index=False)
    report_path = output_dir / "auto_fix_report.md"
    create_report(
        mapping_df=mapping_df,
        rerun_summary=rerun_summary,
        unresolved_rows=mapping_df[mapping_df["resolution_status"] == "unresolved"],
        output_path=report_path,
    )

    summary_payload = {
        "mapping_csv": str(mapping_csv),
        "corrected_csv": str(corrected_csv),
        "unresolved_csv": str(unresolved_csv),
        "rerun_log": str(rerun_log),
        "rerun_summary": rerun_summary,
        "unresolved_count": int((mapping_df["resolution_status"] == "unresolved").sum()),
    }
    with open(output_dir / "auto_fix_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[mapping_csv] {mapping_csv}")
    print(f"[corrected_csv] {corrected_csv}")
    print(f"[report] {report_path}")
    print(f"[rerun_log] {rerun_log}")
    print(f"[rerun_summary] {rerun_summary_path}")
    print(summary_payload)


if __name__ == "__main__":
    main()
