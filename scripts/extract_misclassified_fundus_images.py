#!/usr/bin/env python3
"""
Copy misclassified FA/FUNDUS pairs listed in misclassified.csv into a review
folder.

By default this resolves each FA path in misclassified.csv plus the matching
FUNDUS/FP image under the canonical dataset root, then saves all files in one
output folder:
- the resolved FA images
- the resolved FUNDUS images
- one metadata JSON per copied pair with wrong_zones, preds, labels, and source paths
- manifest.csv summarizing all copied/missing rows
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical")
DEFAULT_SERVER_REPO_ROOT = Path("/home/shashank/UVEITIS_OCT_classidication")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        default="misclassified.csv",
        help="Path to the misclassified CSV. Defaults to ./misclassified.csv.",
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DEFAULT_DATASET_ROOT),
        help=f"Canonical dataset root. Defaults to {DEFAULT_DATASET_ROOT}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder for extracted cases. Defaults to <repo>/misclassified_fundus_images, "
            "using /home/shashank/UVEITIS_OCT_classidication if present."
        ),
    )
    parser.add_argument(
        "--image-kind",
        choices=("both", "fundus", "fa"),
        default="both",
        help="Copy both FA and matching FUNDUS/FP images by default.",
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="CSV column containing the source FA image path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files in the output folder.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    if DEFAULT_SERVER_REPO_ROOT.exists():
        return DEFAULT_SERVER_REPO_ROOT
    return Path(__file__).resolve().parents[1]


def parse_list(value: str) -> list[Any]:
    if value is None or value == "":
        return []
    parsed = ast.literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def parse_list_or_raw(value: str) -> list[Any]:
    try:
        return parse_list(value)
    except (SyntaxError, ValueError):
        return [value]


def source_to_rel_path(source_path: str) -> Path:
    normalized = str(source_path).strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    for idx, part in enumerate(parts):
        if re.fullmatch(r"Patient\d+", part):
            return Path(*parts[idx:])
    raise ValueError(f"Could not find Patient*/... relative path in {source_path!r}")


def candidate_names_for_fundus(fa_name: str) -> list[str]:
    candidates: list[str] = []
    if "_FA_" in fa_name:
        fp_name = fa_name.replace("_FA_", "_FP_")
        candidates.append(fp_name)
        candidates.append(fp_name.replace("_FP_0001.", "_FP_0000."))
        candidates.append(fp_name.replace("_FP_0000.", "_FP_0001."))

    stem = Path(fa_name).stem
    suffix_candidates = []
    for name in candidates:
        name_path = Path(name)
        suffix_candidates.append(name)
        for ext in IMAGE_EXTS:
            suffix_candidates.append(f"{name_path.stem}{ext}")
            suffix_candidates.append(f"{name_path.stem}{ext.upper()}")

    seen = set()
    unique = []
    for name in suffix_candidates:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    if not unique and "_FA_" in stem:
        unique.append(stem.replace("_FA_", "_FP_") + ".png")
    return unique


def resolve_fa_path(dataset_root: Path, fa_rel_path: Path) -> Path:
    candidate = dataset_root / fa_rel_path
    if candidate.exists():
        return candidate

    folder = dataset_root / fa_rel_path.parent
    if not folder.exists():
        raise FileNotFoundError(f"Missing patient/date folder: {folder}")

    fa_name = fa_rel_path.name
    for ext in IMAGE_EXTS:
        ext_candidate = folder / f"{fa_rel_path.stem}{ext}"
        if ext_candidate.exists():
            return ext_candidate

    eye_token = "_OD_" if "_OD_" in fa_name else "_OS_" if "_OS_" in fa_name else ""
    fa_candidates = sorted(path for path in folder.iterdir() if path.is_file() and "_FA_" in path.name)
    same_eye = [path for path in fa_candidates if not eye_token or eye_token in path.name]
    if len(same_eye) == 1:
        return same_eye[0]

    raise FileNotFoundError(f"Could not resolve FA image for {fa_rel_path}")


def resolve_fundus_path(dataset_root: Path, fa_rel_path: Path) -> Path:
    folder = dataset_root / fa_rel_path.parent
    if not folder.exists():
        raise FileNotFoundError(f"Missing patient/date folder: {folder}")

    tried: list[Path] = []
    for name in candidate_names_for_fundus(fa_rel_path.name):
        candidate = folder / name
        tried.append(candidate)
        if candidate.exists():
            return candidate

    eye_token = "_OD_" if "_OD_" in fa_rel_path.name else "_OS_" if "_OS_" in fa_rel_path.name else ""
    fp_candidates = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and "FP" in path.name.upper()
        and (not eye_token or eye_token in path.name)
        and path.suffix.lower() in IMAGE_EXTS
    )
    if len(fp_candidates) == 1:
        return fp_candidates[0]
    for candidate in fp_candidates:
        if "_FP_" in candidate.name:
            return candidate

    all_fp_candidates = sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and "_FP_" in path.name and path.suffix.lower() in IMAGE_EXTS
    )
    if len(all_fp_candidates) == 1:
        return all_fp_candidates[0]

    tried_preview = ", ".join(str(path) for path in tried[:8])
    raise FileNotFoundError(f"Could not resolve FUNDUS image for {fa_rel_path}. Tried: {tried_preview}")


def safe_folder_name(row_number: int, image_name: str) -> str:
    stem = Path(image_name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{row_number:05d}_{safe_stem}"


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def copy_image(source: Path, destination_dir: Path, row_prefix: str, image_prefix: str, overwrite: bool) -> Path:
    destination = destination_dir / f"{row_prefix}_{image_prefix}_{source.name}"
    if overwrite or not destination.exists():
        shutil.copy2(source, destination)
    return destination


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else repo_root() / "misclassified_fundus_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    copied = 0
    missing = 0

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if args.image_column not in (reader.fieldnames or []):
            raise SystemExit(f"Missing required image column {args.image_column!r} in {csv_path}")

        for row_index, row in enumerate(reader, start=1):
            source_image = row[args.image_column]
            wrong_zones = parse_list_or_raw(row.get("wrong_zones", ""))
            preds = parse_list_or_raw(row.get("preds", ""))
            labels = parse_list_or_raw(row.get("labels", ""))
            row_prefix = safe_folder_name(row_index, source_image)

            status = "copied"
            error = ""
            resolved_fa_image = None
            resolved_fundus_image = None
            copied_fa_image = None
            copied_fundus_image = None
            fa_rel_path = None

            try:
                fa_rel_path = source_to_rel_path(source_image)
                if args.image_kind in {"both", "fa"}:
                    resolved_fa_image = resolve_fa_path(dataset_root, fa_rel_path)
                    copied_fa_image = copy_image(resolved_fa_image, output_dir, row_prefix, "FA", args.overwrite)
                if args.image_kind in {"both", "fundus"}:
                    resolved_fundus_image = resolve_fundus_path(dataset_root, fa_rel_path)
                    copied_fundus_image = copy_image(
                        resolved_fundus_image,
                        output_dir,
                        row_prefix,
                        "FUNDUS",
                        args.overwrite,
                    )
                copied += 1
            except Exception as exc:  # Keep processing all rows and report misses in the manifest.
                status = "missing"
                error = str(exc)
                missing += 1

            metadata = {
                "row_index": row_index,
                "status": status,
                "image_kind": args.image_kind,
                "source_csv_image": source_image,
                "fa_relative_path": fa_rel_path.as_posix() if fa_rel_path else "",
                "resolved_fa_image": str(resolved_fa_image) if resolved_fa_image else "",
                "resolved_fundus_image": str(resolved_fundus_image) if resolved_fundus_image else "",
                "copied_fa_image": str(copied_fa_image) if copied_fa_image else "",
                "copied_fundus_image": str(copied_fundus_image) if copied_fundus_image else "",
                "wrong_zones": wrong_zones,
                "preds": preds,
                "labels": labels,
                "error": error,
            }
            write_json(output_dir / f"{row_prefix}_metadata.json", metadata, overwrite=args.overwrite)

            manifest_rows.append(
                {
                    **metadata,
                    "wrong_zones": json.dumps(wrong_zones),
                    "preds": json.dumps(preds),
                    "labels": json.dumps(labels),
                }
            )

    manifest_path = output_dir / "manifest.csv"
    fieldnames = [
        "row_index",
        "status",
        "image_kind",
        "source_csv_image",
        "fa_relative_path",
        "resolved_fa_image",
        "resolved_fundus_image",
        "copied_fa_image",
        "copied_fundus_image",
        "wrong_zones",
        "preds",
        "labels",
        "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "csv_path": str(csv_path),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "image_kind": args.image_kind,
        "rows": len(manifest_rows),
        "copied": copied,
        "missing": missing,
        "manifest": str(manifest_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
