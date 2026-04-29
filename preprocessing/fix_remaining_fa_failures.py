#!/usr/bin/env python3
"""
Repair the remaining FA mask failures using three strategies:
1. Re-run extraction with the more permissive contour fallback.
2. Copy same-eye FA_0001 masks onto unannotated FA_0000 companions.
3. Mirror the opposite-eye mask when no same-eye annotated companion exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mask import create_zone_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True, help="Dataset root containing Patient*/... image folders")
    parser.add_argument("--output_dir", required=True, help="Directory for remediation reports")
    return parser.parse_args()


def build_remaining_failures() -> list[str]:
    orig_fail = pd.read_csv("logs/fa_mask_pass_20260427_fa_only/fa_mask_extraction_failures.csv")
    orig_true = orig_fail[orig_fail["error_category"] != "missing_or_unreadable_file"]["input_path"].tolist()
    rerun_fail = pd.read_csv("logs/fa_path_autofix_20260427/rerun_combined_failures.csv")["input_path"].tolist()
    merged = []
    seen = set()
    for path in orig_true + rerun_fail:
        if path not in seen:
            merged.append(path)
            seen.add(path)
    return merged


def yellow_pixel_count(image_path: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    yellow = cv2.dilate(yellow, np.ones((3, 3), np.uint8), iterations=1)
    return int((yellow > 0).sum())


def mask_path_for_image(image_path: Path) -> Path:
    return image_path.with_name(image_path.stem + "_masks.npy")


def ensure_mask_exists(image_path: Path) -> str:
    result = create_zone_masks(str(image_path))
    return result["method"]


def try_same_eye_transfer(image_path: Path) -> tuple[bool, str]:
    if "_FA_0000" not in image_path.stem:
        return False, "not_applicable"
    source_path = image_path.with_name(image_path.stem.replace("_FA_0000", "_FA_0001") + image_path.suffix)
    if not source_path.exists():
        return False, "missing_same_eye_pair"
    source_mask = mask_path_for_image(source_path)
    if not source_mask.exists():
        ensure_mask_exists(source_path)
    shutil.copy2(source_mask, mask_path_for_image(image_path))
    return True, f"same_eye_transfer:{source_path.name}"


def try_other_eye_mirror(image_path: Path) -> tuple[bool, str]:
    name = image_path.name
    if "_OD_" in name:
        other_name = name.replace("_OD_", "_OS_")
    elif "_OS_" in name:
        other_name = name.replace("_OS_", "_OD_")
    else:
        return False, "missing_eye_token"
    other_path = image_path.with_name(other_name)
    if not other_path.exists():
        return False, "missing_other_eye_pair"
    other_mask_path = mask_path_for_image(other_path)
    if not other_mask_path.exists():
        transferred, _ = try_same_eye_transfer(other_path)
        if not transferred:
            ensure_mask_exists(other_path)
    label_map = np.load(other_mask_path)
    mirrored = np.fliplr(label_map)
    np.save(mask_path_for_image(image_path), mirrored)
    return True, f"mirrored_other_eye:{other_path.name}"


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    for rel_path in build_remaining_failures():
        image_path = dataset_root / rel_path
        record = {
            "input_path": rel_path,
            "resolved_path": str(image_path),
            "status": "error",
            "strategy": "",
            "detail": "",
        }
        if not image_path.exists():
            record["detail"] = "source image missing from dataset"
            records.append(record)
            continue

        try:
            yellow_pixels = yellow_pixel_count(image_path)
            if yellow_pixels == 0:
                transferred, detail = try_same_eye_transfer(image_path)
                if transferred:
                    record["status"] = "ok"
                    record["strategy"] = "same_eye_transfer"
                    record["detail"] = detail
                else:
                    mirrored, mirror_detail = try_other_eye_mirror(image_path)
                    if mirrored:
                        record["status"] = "ok"
                        record["strategy"] = "mirrored_other_eye"
                        record["detail"] = mirror_detail
                    else:
                        record["detail"] = f"{detail}; {mirror_detail}"
            else:
                method = ensure_mask_exists(image_path)
                record["status"] = "ok"
                record["strategy"] = "reextract"
                record["detail"] = method
        except Exception as exc:
            record["detail"] = str(exc)
        records.append(record)

    remediation_df = pd.DataFrame(records)
    remediation_csv = output_dir / "remaining_failure_remediation.csv"
    remediation_df.to_csv(remediation_csv, index=False)

    summary = {
        "total_remaining_inputs": int(len(remediation_df)),
        "recovered_ok": int((remediation_df["status"] == "ok").sum()),
        "still_failed": int((remediation_df["status"] != "ok").sum()),
        "strategy_counts": remediation_df["strategy"].value_counts(dropna=False).to_dict(),
    }
    with open(output_dir / "remaining_failure_remediation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[remediation_csv] {remediation_csv}")
    print(summary)


if __name__ == "__main__":
    main()
