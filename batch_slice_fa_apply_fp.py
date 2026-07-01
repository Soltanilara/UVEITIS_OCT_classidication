#!/usr/bin/env python3
"""
Batch FA zone mask extraction and FP masking.

This wraps "Slice Zone Mohammad 5.1.2026.py" for the full Uveitis FA/FP CSV:
  - detects FA fovea/crosshair geometry from each FA image
  - writes binary zone masks
  - applies the same masks to the paired FP image
  - writes a manifest plus a failures CSV for review/resume

Example:
  python batch_slice_fa_apply_fp.py \
    --csv uveitis_fa_annotations_cleaned.csv \
    --dataset-root "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical" \
    --output-root fa_zone_masks_fp_masked
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_PATH = Path(__file__).with_name("Slice Zone Mohammad 5.1.2026.py")
DEFAULT_DATASET_ROOT = Path(
    "/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical"
)
ZONE_NAMES = [
    "Zone_01_inner_upper_nasal",
    "Zone_02_inner_upper_temporal",
    "Zone_03_inner_lower_temporal",
    "Zone_04_inner_lower_nasal",
    "Zone_05_ring_upper_nasal",
    "Zone_06_ring_upper_temporal",
    "Zone_07_ring_lower_temporal",
    "Zone_08_ring_lower_nasal",
    "Zone_09_optic_disc",
    "Zone_10_far_periphery",
]


def load_slicer_module(script_path: Path = SCRIPT_PATH) -> Any:
    spec = importlib.util.spec_from_file_location("mohammad_zone_slicer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load slicer script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clean_relative_path(value: Any) -> Path:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        raise ValueError("empty relative path")

    text = text.replace("\\", "/")
    if ":" in text or text.startswith("/"):
        parts = PureWindowsPath(str(value)).parts
        patient_idx = next(
            (idx for idx, part in enumerate(parts) if part.lower().startswith("patient")),
            None,
        )
        if patient_idx is None:
            raise ValueError(f"could not derive dataset-relative path from {value!r}")
        text = "/".join(parts[patient_idx:])

    return Path(text)


def paired_fp_relative_path(row: pd.Series, fa_rel: Path) -> Path:
    if "Image File" in row and not pd.isna(row["Image File"]):
        return clean_relative_path(row["Image File"])
    if "UWFFP" in row and not pd.isna(row["UWFFP"]):
        return clean_relative_path(row["UWFFP"])

    fp_name = fa_rel.name.replace("_FA_", "_FP_")
    if fp_name == fa_rel.name:
        raise ValueError(f"could not derive FP filename from {fa_rel}")
    return fa_rel.with_name(fp_name)


def resolve_existing_path(dataset_root: Path, rel_path: Path) -> Path:
    full_path = dataset_root / rel_path
    if full_path.exists():
        return full_path

    preferred_exts = [
        rel_path.suffix,
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    ]
    for ext in dict.fromkeys(ext for ext in preferred_exts if ext):
        candidate = full_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    return full_path


def crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return arr
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return arr[rmin : rmax + 1, cmin : cmax + 1]


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def resize_masks_if_needed(masks: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = size_hw
    if masks.shape[1:] == (target_h, target_w):
        return masks

    resized = []
    for mask in masks:
        img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        img = img.resize((target_w, target_h), resample=Image.Resampling.NEAREST)
        resized.append(np.array(img) > 0)
    return np.stack(resized, axis=0)


def process_one(
    row: pd.Series,
    row_index: int,
    dataset_root: Path,
    output_root: Path,
    slicer: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fa_rel = clean_relative_path(row[args.fa_column])
    fp_rel = paired_fp_relative_path(row, fa_rel)
    fa_path = resolve_existing_path(dataset_root, fa_rel)
    fp_path = resolve_existing_path(dataset_root, fp_rel)

    if not fa_path.exists():
        raise FileNotFoundError(f"FA image not found: {fa_path}")
    if not fp_path.exists():
        raise FileNotFoundError(f"FP image not found: {fp_path}")

    if args.list_paths_only:
        return {
            "row_index": row_index,
            "status": "paths_ok",
            "patient_id": row.get("Patient_ID", ""),
            "eye": row.get("Eye", ""),
            "visit_date": row.get("Visit_Date", ""),
            "fa_rel": fa_rel.as_posix(),
            "fp_rel": fp_rel.as_posix(),
            "fa_path": fa_path.as_posix(),
            "fp_path": fp_path.as_posix(),
        }

    fa_img = Image.open(fa_path).convert("RGBA")
    fa_arr = np.array(fa_img)
    fa_h, fa_w = fa_arr.shape[:2]

    debug_dir = None
    if args.save_debug:
        debug_dir = output_root / "debug" / fa_rel.with_suffix("")
        debug_dir.mkdir(parents=True, exist_ok=True)

    cx, cy, angle_deg, yellow_count = slicer.detect_crosshair_from_yellow(
        fa_arr,
        output_dir=str(debug_dir) if debug_dir is not None else None,
        save_debug=args.save_debug,
    )
    r_inner = args.inner_r_mm * args.px_per_mm
    r_outer = args.outer_r_mm * args.px_per_mm

    mask_dict = slicer.make_masks(
        fa_w,
        fa_h,
        cx,
        cy,
        r_inner,
        r_outer,
        angle_deg,
        args.onh_offset_x,
        args.onh_offset_y,
        args.onh_rx,
        args.onh_ry,
    )
    masks = np.stack([mask_dict[name] for name in ZONE_NAMES], axis=0).astype(bool)

    stem_rel = fa_rel.with_suffix("")
    mask_npy_rel = stem_rel.with_name(stem_rel.name + "_zone_masks.npy")
    label_png_rel = stem_rel.with_name(stem_rel.name + "_zone_labels.png")
    mask_npy_path = output_root / "masks_npy" / mask_npy_rel
    label_png_path = output_root / "mask_labels_png" / label_png_rel
    mask_png_dir = output_root / "mask_png" / stem_rel
    fp_zone_dir = output_root / "fp_masked_zones" / stem_rel

    if args.resume and mask_npy_path.exists() and fp_zone_dir.exists():
        return {
            "row_index": row_index,
            "status": "skipped_existing",
            "fa_rel": fa_rel.as_posix(),
            "fp_rel": fp_rel.as_posix(),
            "mask_npy": mask_npy_path.as_posix(),
            "fp_zone_dir": fp_zone_dir.as_posix(),
        }

    if not args.dry_run:
        mask_npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask_npy_path, masks.astype(np.uint8))

        labels = np.zeros((fa_h, fa_w), dtype=np.uint8)
        for zone_num, mask in enumerate(masks, start=1):
            labels[mask] = zone_num
        write_png(label_png_path, labels)

        if args.save_individual_masks:
            for zone_num, mask in enumerate(masks, start=1):
                mask_png_path = mask_png_dir / f"{zone_num:02d}_{ZONE_NAMES[zone_num - 1]}.png"
                write_png(mask_png_path, (mask.astype(np.uint8) * 255))

        fp_img = Image.open(fp_path).convert("RGBA")
        fp_arr = np.array(fp_img)
        fp_h, fp_w = fp_arr.shape[:2]
        fp_masks = resize_masks_if_needed(masks, (fp_h, fp_w))
        fp_zone_dir.mkdir(parents=True, exist_ok=True)
        for zone_num, mask in enumerate(fp_masks, start=1):
            zone_arr = fp_arr.copy()
            zone_arr[~mask, 3] = 0
            if args.crop:
                zone_arr = crop_to_mask(zone_arr, mask)
            out_path = fp_zone_dir / f"{zone_num:02d}_{ZONE_NAMES[zone_num - 1]}.png"
            write_png(out_path, zone_arr)

    return {
        "row_index": row_index,
        "status": "ok" if not args.dry_run else "dry_run_ok",
        "patient_id": row.get("Patient_ID", ""),
        "eye": row.get("Eye", ""),
        "visit_date": row.get("Visit_Date", ""),
        "fa_rel": fa_rel.as_posix(),
        "fp_rel": fp_rel.as_posix(),
        "fa_path": fa_path.as_posix(),
        "fp_path": fp_path.as_posix(),
        "fa_width": fa_w,
        "fa_height": fa_h,
        "fovea_x": cx,
        "fovea_y": cy,
        "angle_deg": angle_deg,
        "yellow_pixels": yellow_count,
        "mask_npy": mask_npy_path.as_posix(),
        "label_png": label_png_path.as_posix(),
        "fp_zone_dir": fp_zone_dir.as_posix(),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_limit(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create FA binary zone masks and apply them to paired FP images."
    )
    parser.add_argument("--csv", default="uveitis_fa_annotations_cleaned.csv")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("fa_zone_masks_fp_masked"))
    parser.add_argument("--fa-column", default="Image_File(FA)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows; <=0 means all.")
    parser.add_argument("--start-row", type=int, default=0, help="Zero-based CSV row offset.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--list-paths-only",
        action="store_true",
        help="Only verify FA/FP paths exist; do not open images or detect masks.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows whose outputs already exist.")
    parser.add_argument("--crop", action="store_true", help="Crop FP zone PNGs to each mask bounding box.")
    parser.add_argument("--save-debug", action="store_true", help="Save detected fovea debug PNGs.")
    parser.add_argument("--save-individual-masks", action="store_true")
    parser.add_argument("--px_per_mm", type=float, default=53.0)
    parser.add_argument("--inner_r_mm", type=float, default=3.0)
    parser.add_argument("--outer_r_mm", type=float, default=16.0)
    parser.add_argument("--onh_offset_x", type=int, default=270)
    parser.add_argument("--onh_offset_y", type=int, default=0)
    parser.add_argument("--onh_rx", type=int, default=80)
    parser.add_argument("--onh_ry", type=int, default=95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Missing slicer script: {SCRIPT_PATH}")

    slicer = load_slicer_module(SCRIPT_PATH)
    df = pd.read_csv(args.csv)
    if args.fa_column not in df.columns:
        raise KeyError(f"CSV is missing FA column {args.fa_column!r}")

    start_row = max(0, args.start_row)
    limit = finite_limit(args.limit)
    selected = df.iloc[start_row:]
    if limit is not None:
        selected = selected.head(limit)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    total = len(selected)
    for ordinal, (row_index, row) in enumerate(selected.iterrows(), start=1):
        label = f"[{ordinal}/{total} row={row_index}]"
        try:
            if args.dry_run:
                result = process_one(row, row_index, args.dataset_root, args.output_root, slicer, args)
            else:
                with open(args.output_root / "batch_stdout.log", "a", encoding="utf-8") as log:
                    with contextlib.redirect_stdout(log):
                        result = process_one(
                            row, row_index, args.dataset_root, args.output_root, slicer, args
                        )
            manifest_rows.append(result)
            print(f"{label} {result['status']}: {result['fa_rel']}")
        except Exception as exc:
            fa_value = row.get(args.fa_column, "")
            failure = {
                "row_index": row_index,
                "status": "error",
                "fa_value": fa_value,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failure_rows.append(failure)
            print(f"{label} error: {fa_value} :: {type(exc).__name__}: {exc}")

    manifest_path = args.output_root / "manifest.csv"
    failures_path = args.output_root / "failures.csv"
    summary_path = args.output_root / "summary.json"

    if manifest_rows:
        write_rows(manifest_path, manifest_rows)
    if failure_rows:
        write_rows(failures_path, failure_rows)

    summary = {
        "csv": str(args.csv),
        "dataset_root": args.dataset_root.as_posix(),
        "output_root": args.output_root.as_posix(),
        "start_row": start_row,
        "limit": limit,
        "selected_rows": total,
        "succeeded": sum(
            row["status"] in {"ok", "dry_run_ok", "skipped_existing", "paths_ok"}
            for row in manifest_rows
        ),
        "failed": len(failure_rows),
        "dry_run": args.dry_run,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if failure_rows:
        print(f"Done with {len(failure_rows)} failures. See {failures_path}")
        return 1
    print(f"Done. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
