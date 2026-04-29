#!/usr/bin/env python3
"""
Extract the 10 annotated FA zones from a single image with yellow overlays.

Outputs:
- zone_01.png ... zone_10.png
- label_map.png
- qc_overlay.png
- retina_mask.png
- yellow_overlay_mask.png
- geometry.json
- zones_contact_sheet.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.extract_fa_zone_masks import process_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", required=True, help="Path to a single annotated FA image")
    parser.add_argument("--output-dir", required=True, help="Directory to store extracted masks")
    parser.add_argument(
        "--input-root",
        default=None,
        help="Optional input root used to preserve relative structure. Defaults to the image parent directory.",
    )
    parser.add_argument("--min-retina-threshold", type=int, default=8)
    parser.add_argument("--yellow-s-threshold", type=int, default=30)
    parser.add_argument("--yellow-v-threshold", type=int, default=80)
    return parser.parse_args()


def build_contact_sheet(image_output_dir: Path) -> Path:
    item_names = ["qc_overlay.png"] + [f"zone_{zone:02d}.png" for zone in range(1, 11)]
    tiles: list[Image.Image] = []

    for name in item_names:
        image = Image.open(image_output_dir / name).convert("L")
        image = ImageOps.autocontrast(image)
        thumb = image.resize((320, 320))
        tile = Image.new("L", (320, 350), 0)
        tile.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 325), name, fill=255)
        tiles.append(tile)

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("L", (cols * 320, rows * 350), 0)
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * 320
        y = (idx // cols) * 350
        sheet.paste(tile, (x, y))

    output_path = image_output_dir / "zones_contact_sheet.png"
    sheet.save(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    image_path = Path(args.image_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    input_root = Path(args.input_root).resolve() if args.input_root else image_path.parent

    image_output_dir = process_image(
        image_path,
        output_dir=output_dir,
        input_root=input_root,
        retina_threshold=args.min_retina_threshold,
        yellow_s_threshold=args.yellow_s_threshold,
        yellow_v_threshold=args.yellow_v_threshold,
    )
    contact_sheet_path = build_contact_sheet(image_output_dir)

    print(f"image_output_dir={image_output_dir}")
    print(f"contact_sheet={contact_sheet_path}")


if __name__ == "__main__":
    main()
