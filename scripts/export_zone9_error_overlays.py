#!/usr/bin/env python3
"""Export original FA images and ground-truth zone-mask overlays for errors.

The input CSV is normally the file produced by the Experiment D extractor:
``analysis/experiment_D_group_mlps/zone9_misclassified_images.csv``.  For each
row this script writes two files: the untouched image and an RGBA overlay with
the anatomical masks, zone labels, and Zone-9 prediction details.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = [
    (255, 89, 94), (255, 146, 76), (255, 202, 58), (138, 201, 38),
    (82, 183, 136), (25, 130, 196), (66, 103, 172), (106, 76, 147),
    (247, 37, 133), (173, 181, 189),
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--errors-csv", type=Path, required=True)
    p.add_argument("--annotations-csv", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--mask-root", type=Path, default=None,
                   help="Root containing masks_npy/. Defaults to the FA dataset sibling.")
    p.add_argument("--alpha", type=int, default=85, help="Mask opacity, 0-255.")
    p.add_argument("--limit", type=int, default=0, help="0 exports every row.")
    return p.parse_args()


def norm_path(value: str) -> str:
    return re.sub(r"/+", "/", str(value).replace("\\", "/")).lower().rstrip("/")


def image_key(value: str) -> str:
    text = norm_path(value)
    match = re.search(r"(patient[^/]+/.+)$", text)
    return match.group(1) if match else text


def mask_path(image_path: Path, mask_root: Path) -> Path:
    # Masks use the image's Patient/date directory and append _zone_masks.npy.
    return mask_root / "masks_npy" / image_path.parent.parent.name / image_path.parent.name / (
        image_path.stem + "_zone_masks.npy"
    )


def load_stack(path: Path) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim == 3 and mask.shape[0] >= 10:
        return mask[:10] > 0
    if mask.ndim == 3 and mask.shape[-1] >= 10:
        return np.moveaxis(mask[..., :10] > 0, -1, 0)
    if mask.ndim == 2:
        return np.stack([mask == zone for zone in range(1, 11)])
    raise ValueError(f"Unsupported mask shape {mask.shape}: {path}")


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    ns = args()
    ns.output_dir.mkdir(parents=True, exist_ok=True)
    with ns.annotations_csv.open(newline="", encoding="utf-8-sig") as f:
        annotations = {image_key(r.get("FA_Image_Abs_Path", r.get("Image_File(FA)", ""))): r
                       for r in csv.DictReader(f)}
    with ns.errors_csv.open(newline="", encoding="utf-8-sig") as f:
        errors = list(csv.DictReader(f))
    if ns.limit > 0:
        errors = errors[:ns.limit]

    ok = 0
    for index, row in enumerate(errors, 1):
        source = Path(row["image_path"])
        if not source.is_file():
            print(f"SKIP missing image: {source}")
            continue
        root = ns.mask_root or Path(str(source).split("/Sample 2.5.2026_canonical/", 1)[0]) / (
            "Sample 2.5.2026_canonical_fa_zone_masks"
        )
        mpath = mask_path(source, root)
        if not mpath.is_file():
            print(f"SKIP missing mask: {mpath}")
            continue
        image = Image.open(source).convert("RGB")
        original_name = f"{index:03d}_fold{row.get('fold','?')}_{row['error_type']}_{source.stem}.png"
        image.save(ns.output_dir / ("original_" + original_name))

        stack = load_stack(mpath)
        base = np.asarray(image)
        if stack.shape[1:] != base.shape[:2]:
            stack = np.stack([np.asarray(Image.fromarray(x.astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)) > 0
                              for x in stack])
        rgba = np.zeros((*base.shape[:2], 4), dtype=np.uint8)
        for zone, zone_mask in enumerate(stack, 1):
            rgba[zone_mask] = (*COLORS[zone - 1], ns.alpha)
        overlay = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(rgba, "RGBA"))
        draw = ImageDraw.Draw(overlay, "RGBA")
        fnt = font()
        ann = annotations.get(image_key(str(source)), {})
        lines = ["Ground-truth anatomical labels"]
        for zone in (5, 8, 9):
            value = ann.get(f"Zone{zone}_label", "?")
            lines.append(f"Zone {zone} ground truth: {value}")
        box_h = 34 * len(lines) + 16
        draw.rectangle((8, 8, 760, box_h), fill=(0, 0, 0, 185))
        for line_no, line in enumerate(lines):
            draw.text((18, 16 + line_no * 34), line, fill=(255, 255, 255, 255), font=fnt)
        overlay.convert("RGB").save(ns.output_dir / ("overlay_" + original_name))
        ok += 1
    print(f"Exported {ok}/{len(errors)} image pairs to {ns.output_dir}")


if __name__ == "__main__":
    main()
