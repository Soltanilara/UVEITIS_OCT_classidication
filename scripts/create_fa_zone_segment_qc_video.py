#!/usr/bin/env python3
"""
Create a QC video for FA anatomical zone segments.

Each video frame shows one FA image as an 11-tile montage:
full image first, followed by Zone 1 through Zone 10. Zone tiles are labeled
with their corresponding ZoneN_label annotation from the CSV.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import unicodedata
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont


DEFAULT_DATASET_ROOT = Path("/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical")
DEFAULT_MASK_ROOT = Path("/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical_fa_zone_masks")
ZONE_COLORS = {
    1: "#ff595e",
    2: "#ff924c",
    3: "#ffca3a",
    4: "#8ac926",
    5: "#52b788",
    6: "#1982c4",
    7: "#4267ac",
    8: "#6a4c93",
    9: "#f72585",
    10: "#adb5bd",
}
ZONE_NAMES = {
    1: "inner upper nasal",
    2: "inner upper temporal",
    3: "inner lower temporal",
    4: "inner lower nasal",
    5: "ring upper nasal",
    6: "ring upper temporal",
    7: "ring lower temporal",
    8: "ring lower nasal",
    9: "optic disc",
    10: "far periphery",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="uveitis_fa_annotations_cleaned.csv",
        help="Annotation CSV with Image_File(FA) and Zone1_label...Zone10_label columns.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=DEFAULT_MASK_ROOT,
        help="Mask output root containing masks_npy/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fa_zone_segment_qc"),
        help="Directory for rendered frames and the video.",
    )
    parser.add_argument("--output-name", default="fa_zone_segment_qc_video.mp4")
    parser.add_argument("--tile-size", type=int, default=260)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--seconds-per-image", type=float, default=1.5)
    parser.add_argument("--max-images", type=int, default=0, help="0 means include every resolvable row.")
    parser.add_argument("--sampling", choices=("head", "even"), default="even")
    parser.add_argument("--alpha", type=int, default=96, help="Zone overlay opacity, 0..255.")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def safe_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text if ascii_text else "[blank]"


def clean_relative_path(value: Any) -> Path:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        raise ValueError("empty relative path")
    text = text.replace("\\", "/")
    if ":" in text or text.startswith("/"):
        parts = PureWindowsPath(str(value)).parts
        patient_idx = next((idx for idx, part in enumerate(parts) if part.lower().startswith("patient")), None)
        if patient_idx is None:
            raise ValueError(f"Could not derive dataset-relative path from {value!r}")
        text = "/".join(parts[patient_idx:])
    return Path(text)


def resolve_existing_image(dataset_root: Path, rel_path: Path) -> Path:
    candidate = dataset_root / rel_path
    if candidate.exists():
        return candidate
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        alt = candidate.with_suffix(ext)
        if alt.exists():
            return alt
        alt_upper = candidate.with_suffix(ext.upper())
        if alt_upper.exists():
            return alt_upper
    raise FileNotFoundError(candidate)


def mask_path_for_fa(mask_root: Path, fa_rel: Path) -> Path:
    return mask_root / "masks_npy" / fa_rel.with_name(f"{fa_rel.stem}_zone_masks.npy")


def zone_stack_from_mask(mask: np.ndarray, mask_path: Path) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] >= 10:
        return (mask[:10] > 0).astype(np.uint8)
    if mask.ndim == 3 and mask.shape[-1] >= 10:
        return np.moveaxis((mask[..., :10] > 0).astype(np.uint8), -1, 0)
    if mask.ndim == 2:
        return np.stack([(mask == zone) for zone in range(1, 11)], axis=0).astype(np.uint8)
    raise ValueError(f"Unsupported mask shape in {mask_path}: {tuple(mask.shape)}")


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST), dtype=np.uint8)


def fit_and_pad(image: Image.Image, tile_size: int, fill: tuple[int, int, int] = (8, 10, 12)) -> Image.Image:
    image = image.convert("RGB")
    scale = min(tile_size / image.width, tile_size / image.height)
    resized = image.resize(
        (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    tile = Image.new("RGB", (tile_size, tile_size), fill)
    tile.paste(resized, ((tile_size - resized.width) // 2, (tile_size - resized.height) // 2))
    return tile


def resize_image_and_masks_to_tile(
    image: Image.Image,
    zone_stack: np.ndarray,
    tile_size: int,
    fill: tuple[int, int, int] = (8, 10, 12),
) -> tuple[Image.Image, np.ndarray]:
    """Resize once per frame and pad image/masks into tile coordinates."""
    image = image.convert("RGB")
    scale = min(tile_size / image.width, tile_size / image.height)
    resized_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resized_image = image.resize(resized_size, Image.Resampling.BILINEAR)
    image_tile = Image.new("RGB", (tile_size, tile_size), fill)
    left = (tile_size - resized_image.width) // 2
    top = (tile_size - resized_image.height) // 2
    image_tile.paste(resized_image, (left, top))

    mask_tiles = np.zeros((zone_stack.shape[0], tile_size, tile_size), dtype=np.uint8)
    for zone_idx in range(zone_stack.shape[0]):
        resized_mask = resize_mask(zone_stack[zone_idx], resized_size)
        mask_tiles[zone_idx, top : top + resized_size[1], left : left + resized_size[0]] = resized_mask
    return image_tile, mask_tiles


def make_zone_tile(image_tile: Image.Image, zone_mask: np.ndarray, zone: int, alpha: int) -> Image.Image:
    image_arr = np.array(image_tile, dtype=np.uint8)
    zone_bool = zone_mask.astype(bool)

    masked = np.zeros_like(image_arr)
    masked[zone_bool] = image_arr[zone_bool]
    tile = Image.fromarray(masked)

    color = ImageColor.getrgb(ZONE_COLORS[zone])
    overlay = np.zeros((*zone_bool.shape, 4), dtype=np.uint8)
    overlay[zone_bool] = (*color, alpha)
    composited = Image.alpha_composite(tile.convert("RGBA"), Image.fromarray(overlay))
    return composited.convert("RGB")


def draw_label_bar(tile: Image.Image, title: str, subtitle: str, font: ImageFont.ImageFont) -> Image.Image:
    tile = tile.convert("RGB")
    draw = ImageDraw.Draw(tile, mode="RGBA")
    bar_h = 42
    draw.rectangle((0, tile.height - bar_h, tile.width, tile.height), fill=(0, 0, 0, 185))
    draw.text((8, tile.height - bar_h + 5), safe_text(title), fill="white", font=font)
    draw.text((8, tile.height - bar_h + 22), safe_text(subtitle), fill=(210, 225, 255), font=font)
    return tile


def annotation_text(row: pd.Series, zone: int) -> str:
    value = row.get(f"Zone{zone}_label", "")
    if pd.isna(value):
        return "annotation: missing"
    try:
        numeric = float(value)
        if numeric.is_integer():
            value = int(numeric)
    except (TypeError, ValueError):
        pass
    return f"annotation: {value}"


def build_frame(row: pd.Series, dataset_root: Path, mask_root: Path, tile_size: int, columns: int, alpha: int) -> Image.Image:
    fa_rel = clean_relative_path(row["Image_File(FA)"])
    fa_path = resolve_existing_image(dataset_root, fa_rel)
    mask_path = mask_path_for_fa(mask_root, fa_rel)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    image = Image.open(fa_path).convert("RGB")
    zone_stack = zone_stack_from_mask(np.load(mask_path), mask_path)
    if zone_stack.shape[1:] != (image.height, image.width):
        zone_stack = np.stack([resize_mask(zone_stack[z], image.size) for z in range(10)], axis=0)

    image_tile, zone_stack_tile = resize_image_and_masks_to_tile(image, zone_stack, tile_size)

    font = ImageFont.load_default()
    tiles: list[Image.Image] = []
    full_subtitle = f"{row.get('Patient_ID', '')} {row.get('Eye', '')} {row.get('Visit_Date', '')}".strip()
    tiles.append(draw_label_bar(image_tile.copy(), "Full FA image", full_subtitle, font))

    for zone in range(1, 11):
        zone_tile = make_zone_tile(image_tile, zone_stack_tile[zone - 1], zone, alpha)
        title = f"Zone {zone}: {ZONE_NAMES[zone]}"
        tiles.append(draw_label_bar(zone_tile, title, annotation_text(row, zone), font))

    rows = math.ceil(len(tiles) / columns)
    padding = 14
    header_h = 56
    page_w = padding + columns * (tile_size + padding)
    page_h = header_h + padding + rows * (tile_size + padding)
    frame = Image.new("RGB", (page_w, page_h), color=(18, 22, 26))
    draw = ImageDraw.Draw(frame)
    draw.text((padding, 12), safe_text(fa_rel.as_posix()), fill="white", font=font)
    draw.text((padding, 30), safe_text(f"mask: {mask_path.relative_to(mask_root).as_posix()}"), fill=(190, 210, 230), font=font)

    for idx, tile in enumerate(tiles):
        col = idx % columns
        row_idx = idx // columns
        left = padding + col * (tile_size + padding)
        top = header_h + padding + row_idx * (tile_size + padding)
        frame.paste(tile, (left, top))
        draw.rectangle((left, top, left + tile_size, top + tile_size), outline=(215, 220, 225), width=1)
    return frame


def pick_rows(df: pd.DataFrame, max_images: int, sampling: str) -> pd.DataFrame:
    if max_images <= 0 or len(df) <= max_images:
        return df.copy()
    if sampling == "head":
        return df.head(max_images).copy()
    indices = np.linspace(0, len(df) - 1, num=max_images, dtype=int)
    return df.iloc[indices].copy()


def filter_resolvable_rows(df: pd.DataFrame, dataset_root: Path, mask_root: Path) -> tuple[pd.DataFrame, list[str]]:
    keep: list[int] = []
    skipped: list[str] = []
    for idx, row in df.iterrows():
        try:
            fa_rel = clean_relative_path(row["Image_File(FA)"])
            resolve_existing_image(dataset_root, fa_rel)
            if not mask_path_for_fa(mask_root, fa_rel).exists():
                raise FileNotFoundError(mask_path_for_fa(mask_root, fa_rel))
            keep.append(idx)
        except Exception as exc:
            skipped.append(f"{row.get('Image_File(FA)', idx)} :: {exc}")
    return df.loc[keep].reset_index(drop=True), skipped


def write_video_from_frames(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe is None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("Install imageio-ffmpeg or make ffmpeg available on PATH to encode the QC video.") from exc
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    pattern = str(output_path.parent / "frame_%06d.png")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return

    fallback = [
        ffmpeg_exe,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        pattern,
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(fallback, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    mask_root = args.mask_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = {"Image_File(FA)", *{f"Zone{i}_label" for i in range(1, 11)}}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    resolved_df, skipped = filter_resolvable_rows(df, dataset_root, mask_root)
    selected_df = pick_rows(resolved_df, args.max_images, args.sampling)
    if selected_df.empty:
        raise SystemExit("No rows with resolvable FA images and zone masks were found.")

    fps = 1.0 / max(args.seconds_per_image, 0.1) if args.seconds_per_image > 0 else args.fps
    frame_paths: list[Path] = []
    for frame_idx, (_, row) in enumerate(selected_df.iterrows(), start=1):
        frame = build_frame(row, dataset_root, mask_root, args.tile_size, args.columns, args.alpha)
        frame_path = output_dir / f"frame_{frame_idx:06d}.png"
        frame.save(frame_path, quality=95)
        frame_paths.append(frame_path)
        if frame_idx == 1 or frame_idx % 25 == 0 or frame_idx == len(selected_df):
            print(f"[rendered] {frame_idx}/{len(selected_df)}", flush=True)

    video_path = output_dir / args.output_name
    write_video_from_frames(frame_paths, video_path, fps=fps)

    skipped_path = output_dir / "skipped_rows.txt"
    skipped_path.write_text("\n".join(skipped) + ("\n" if skipped else ""))
    first_frame_path = output_dir / "first_frame.png"
    Image.open(frame_paths[0]).save(first_frame_path, quality=95)

    if not args.keep_frames:
        for frame_path in frame_paths:
            frame_path.unlink()

    print(f"[video] {video_path}")
    print(f"[first_frame] {first_frame_path}")
    print(f"[selected_rows] {len(selected_df)}")
    print(f"[resolvable_rows] {len(resolved_df)}")
    print(f"[skipped_rows] {len(skipped)} -> {skipped_path}")


if __name__ == "__main__":
    main()
