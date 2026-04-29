#!/usr/bin/env python3
"""
Create a montage of fundus images with extracted FA zone masks overlaid.

The FA and fundus images are assumed to be registered, so the FA-derived mask
can be drawn directly on the corresponding fundus image.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import imageio_ffmpeg
from PIL import Image, ImageColor, ImageDraw, ImageFont
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Ready-only dataset CSV with FA_Mask_Path and UWFFP columns")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing Patient*/... directories")
    parser.add_argument("--output-dir", required=True, help="Directory where montage pages will be written")
    parser.add_argument("--max-images", type=int, default=24, help="Maximum number of images to include")
    parser.add_argument("--tile-size", type=int, default=320, help="Square tile size for each fundus panel")
    parser.add_argument("--columns", type=int, default=4, help="Number of columns in the montage")
    parser.add_argument("--alpha", type=int, default=108, help="Overlay opacity in the range 0..255")
    parser.add_argument(
        "--sampling",
        choices=("head", "even"),
        default="even",
        help="How to pick images when max-images is smaller than the dataset",
    )
    parser.add_argument("--video", action="store_true", help="Create a QC video for all resolvable images")
    parser.add_argument("--fps", type=float, default=3.0, help="Frames per second for the QC video")
    parser.add_argument(
        "--video-mode",
        choices=("pages", "single"),
        default="pages",
        help="Use one montage page per video step, or one image per frame",
    )
    parser.add_argument("--seconds-per-page", type=float, default=1.5, help="Display time per montage page in page-video mode")
    return parser.parse_args()


def resolve_fundus_path(dataset_root: Path, fa_rel_path: str, uwffp_value: str) -> Path:
    fa_parent = Path(str(fa_rel_path).replace("\\", "/")).parent
    fa_name = Path(str(fa_rel_path).replace("\\", "/")).name
    folder = dataset_root / fa_parent
    fundus_name = Path(str(uwffp_value).replace("\\", "/")).name
    candidate = folder / fundus_name
    if candidate.exists():
        return candidate

    expected_name = fa_name.replace("_FA_", "_FP_").replace("_0001.", "_0000.").replace("_0000.", "_0000.")
    expected_stem = Path(expected_name).stem
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        candidate = folder / f"{expected_stem}{ext}"
        if candidate.exists():
            return candidate

    eye_token = "_OD_" if "_OD_" in fa_name else "_OS_" if "_OS_" in fa_name else ""
    eye_candidates = sorted(folder.glob(f"*{eye_token}*FP*")) if eye_token else []
    if len(eye_candidates) == 1:
        return eye_candidates[0]
    if eye_candidates:
        for eye_candidate in eye_candidates:
            if "_FP_" in eye_candidate.name:
                return eye_candidate

    all_fp_candidates = sorted(folder.glob("*_FP_*"))
    if len(all_fp_candidates) == 1:
        return all_fp_candidates[0]
    raise FileNotFoundError(f"Could not resolve fundus image for {fa_rel_path} using {fundus_name}")


def safe_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text if ascii_text else "[unprintable]"


def pick_rows(df: pd.DataFrame, max_images: int, sampling: str) -> pd.DataFrame:
    if len(df) <= max_images:
        return df.copy()
    if sampling == "head":
        return df.head(max_images).copy()
    indices = np.linspace(0, len(df) - 1, num=max_images, dtype=int)
    return df.iloc[indices].copy()


def filter_resolvable_rows(df: pd.DataFrame, dataset_root: Path) -> tuple[pd.DataFrame, list[str]]:
    keep_rows: list[int] = []
    skipped: list[str] = []
    for idx, row in df.iterrows():
        fa_rel_path = str(row["Image_File(FA)"])
        try:
            resolve_fundus_path(dataset_root, fa_rel_path, str(row["UWFFP"]))
            keep_rows.append(idx)
        except FileNotFoundError:
            skipped.append(fa_rel_path)
    return df.loc[keep_rows].copy(), skipped


def fit_and_crop(image: Image.Image, tile_size: int) -> Image.Image:
    width, height = image.size
    scale = tile_size / min(width, height)
    resized = image.resize((int(round(width * scale)), int(round(height * scale))), Image.Resampling.LANCZOS)
    left = max(0, (resized.width - tile_size) // 2)
    top = max(0, (resized.height - tile_size) // 2)
    return resized.crop((left, top, left + tile_size, top + tile_size))


def fit_and_crop_pair(image: Image.Image, mask: np.ndarray, tile_size: int) -> tuple[Image.Image, np.ndarray]:
    width, height = image.size
    scale = tile_size / min(width, height)
    resized_size = (int(round(width * scale)), int(round(height * scale)))
    resized_image = image.resize(resized_size, Image.Resampling.LANCZOS)
    resized_mask = Image.fromarray(mask.astype(np.uint8), mode="L").resize(resized_size, Image.Resampling.NEAREST)
    left = max(0, (resized_image.width - tile_size) // 2)
    top = max(0, (resized_image.height - tile_size) // 2)
    crop_box = (left, top, left + tile_size, top + tile_size)
    cropped_image = resized_image.crop(crop_box)
    cropped_mask = np.array(resized_mask.crop(crop_box), dtype=np.uint8)
    return cropped_image, cropped_mask


def build_overlay_tile(fundus_path: Path, mask_path: Path, tile_size: int, alpha: int) -> Image.Image:
    base = Image.open(fundus_path).convert("RGB")
    mask = np.load(mask_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D label map in {mask_path}, got shape {mask.shape}")

    base, mask = fit_and_crop_pair(base, mask, tile_size)

    base_arr = np.array(base, dtype=np.uint8)
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    border = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)

    for zone, color_hex in ZONE_COLORS.items():
        zone_mask = mask == zone
        if not np.any(zone_mask):
            continue
        r, g, b = ImageColor.getrgb(color_hex)
        overlay[zone_mask] = (r, g, b, alpha)

        zone_u8 = zone_mask.astype(np.uint8)
        interior = zone_u8.copy()
        interior &= np.pad(zone_u8[1:, :], ((0, 1), (0, 0)), constant_values=0)
        interior &= np.pad(zone_u8[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        interior &= np.pad(zone_u8[:, 1:], ((0, 0), (0, 1)), constant_values=0)
        interior &= np.pad(zone_u8[:, :-1], ((0, 0), (1, 0)), constant_values=0)
        border |= zone_mask & (~interior.astype(bool))

    overlay_img = Image.fromarray(overlay, mode="RGBA")
    composited = Image.alpha_composite(Image.fromarray(base_arr).convert("RGBA"), overlay_img)

    comp_arr = np.array(composited, dtype=np.uint8)
    comp_arr[border] = (255, 255, 255, 255)
    comp_arr[mask == 9] = np.clip(comp_arr[mask == 9].astype(np.int16) + np.array([20, -20, 20, 0]), 0, 255).astype(np.uint8)
    return Image.fromarray(comp_arr, mode="RGBA").convert("RGB")


def draw_caption(draw: ImageDraw.ImageDraw, tile_box: tuple[int, int, int, int], line1: str, line2: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = tile_box
    text_bg_top = bottom - 38
    draw.rectangle((left, text_bg_top, right, bottom), fill=(0, 0, 0, 170))
    draw.text((left + 8, text_bg_top + 4), safe_text(line1), fill="white", font=font)
    draw.text((left + 8, text_bg_top + 20), safe_text(line2), fill="white", font=font)


def make_single_frame(
    row: pd.Series,
    dataset_root: Path,
    tile_size: int,
    alpha: int,
    frame_index: int,
    total_frames: int,
) -> Image.Image:
    font = ImageFont.load_default()
    padding = 18
    header_h = 34
    footer_h = 78
    frame_w = tile_size + padding * 2
    frame_h = header_h + tile_size + footer_h + padding
    frame = Image.new("RGB", (frame_w, frame_h), color=(20, 24, 28))
    draw = ImageDraw.Draw(frame, mode="RGBA")

    fa_rel_path = str(row["Image_File(FA)"])
    fundus_path = resolve_fundus_path(dataset_root, fa_rel_path, str(row["UWFFP"]))
    mask_path = dataset_root / str(row["FA_Mask_Path"])
    tile = build_overlay_tile(fundus_path, mask_path, tile_size, alpha)

    title = f"Fundus + FA-derived zone masks   {frame_index}/{total_frames}"
    draw.text((padding, 10), safe_text(title), fill="white", font=font)

    image_top = header_h
    frame.paste(tile, (padding, image_top))
    draw.rectangle((padding, image_top, padding + tile_size, image_top + tile_size), outline=(220, 220, 220), width=1)

    footer_top = image_top + tile_size + 10
    draw.text((padding, footer_top), safe_text(Path(fa_rel_path).parent.as_posix()), fill="white", font=font)
    draw.text((padding, footer_top + 16), safe_text(Path(fa_rel_path).name), fill="white", font=font)
    draw.text((padding, footer_top + 32), safe_text(Path(fundus_path).name), fill=(190, 220, 255), font=font)

    return frame


def create_page(rows: pd.DataFrame, dataset_root: Path, tile_size: int, columns: int, alpha: int) -> Image.Image:
    font = ImageFont.load_default()
    padding = 18
    caption_height = 0
    rows_count = math.ceil(len(rows) / columns)
    page_width = padding + columns * (tile_size + padding)
    page_height = padding + rows_count * (tile_size + padding + caption_height)
    page = Image.new("RGB", (page_width, page_height), color=(20, 24, 28))
    draw = ImageDraw.Draw(page, mode="RGBA")

    for idx, (_, row) in enumerate(rows.iterrows()):
        col = idx % columns
        row_idx = idx // columns
        left = padding + col * (tile_size + padding)
        top = padding + row_idx * (tile_size + padding + caption_height)

        fa_rel_path = str(row["Image_File(FA)"])
        fundus_path = resolve_fundus_path(dataset_root, fa_rel_path, str(row["UWFFP"]))
        mask_path = dataset_root / str(row["FA_Mask_Path"])
        tile = build_overlay_tile(fundus_path, mask_path, tile_size, alpha)
        page.paste(tile, (left, top))

        tile_box = (left, top, left + tile_size, top + tile_size)
        draw_caption(
            draw,
            tile_box,
            Path(fa_rel_path).parent.as_posix(),
            Path(fa_rel_path).name.replace("_FA_0001", ""),
            font,
        )
    return page


def write_video(
    rows: pd.DataFrame,
    dataset_root: Path,
    output_path: Path,
    tile_size: int,
    alpha: int,
    fps: float,
) -> None:
    if output_path.exists():
        output_path.unlink()
    sample_frame = make_single_frame(rows.iloc[0], dataset_root, tile_size, alpha, 1, len(rows))
    width, height = sample_frame.size
    chosen_codec = "MJPG"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*chosen_codec), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("Could not open a Motion-JPEG video writer.")

    try:
        for idx, (_, row) in enumerate(rows.iterrows(), start=1):
            frame = make_single_frame(row, dataset_root, tile_size, alpha, idx, len(rows))
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            if idx == 1 or idx % 25 == 0 or idx == len(rows):
                print(f"[video_progress] {idx}/{len(rows)}", flush=True)
    finally:
        writer.release()

    print(f"[video_codec] {chosen_codec}")


def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start:start + chunk_size].copy() for start in range(0, len(df), chunk_size)]


def write_page_video(
    rows: pd.DataFrame,
    dataset_root: Path,
    output_dir: Path,
    output_path: Path,
    tile_size: int,
    columns: int,
    page_size: int,
    alpha: int,
    fps: float,
    seconds_per_page: float,
) -> None:
    if output_path.exists():
        output_path.unlink()
    for stale_page in sorted(output_dir.glob("fundus_zone_mask_page_*.png")):
        stale_page.unlink()
    pages = chunk_dataframe(rows.reset_index(drop=True), page_size)
    if not pages:
        raise RuntimeError("No rows available for page video.")
    rendered_pages: list[Path] = []
    for idx, page_rows in enumerate(pages, start=1):
        page = create_page(page_rows, dataset_root, tile_size, columns, alpha)
        page_path = output_dir / f"fundus_zone_mask_page_{idx:03d}.png"
        page.save(page_path, quality=95)
        rendered_pages.append(page_path)
        print(f"[page_rendered] {idx}/{len(pages)}", flush=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    page_rate = 1.0 / max(seconds_per_page, 0.1)
    input_pattern = str(output_dir / "fundus_zone_mask_page_%03d.png")
    ffmpeg_cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate",
        f"{page_rate:.6f}",
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        fallback_cmd = [
            ffmpeg_exe,
            "-y",
            "-framerate",
            f"{page_rate:.6f}",
            "-i",
            input_pattern,
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(fallback_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg mp4 encoding failed:\n{result.stderr[-2000:]}")
        print("[video_codec] mpeg4", flush=True)
    else:
        print("[video_codec] libx264", flush=True)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required_cols = {"Image_File(FA)", "UWFFP", "FA_Mask_Path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    resolved_df, skipped = filter_resolvable_rows(df, dataset_root)
    if resolved_df.empty:
        raise SystemExit("No fundus/FA pairs could be resolved for montage generation.")

    df = pick_rows(resolved_df, args.max_images, args.sampling)
    page = create_page(df, dataset_root, args.tile_size, args.columns, args.alpha)
    page_path = output_dir / "fundus_zone_mask_montage_page_01.png"
    page.save(page_path, quality=95)

    legend_width = 360
    legend_height = 28 + 24 * len(ZONE_COLORS)
    legend = Image.new("RGB", (legend_width, legend_height), color=(20, 24, 28))
    draw = ImageDraw.Draw(legend)
    font = ImageFont.load_default()
    draw.text((16, 10), "Zone legend", fill="white", font=font)
    y = 32
    for zone, color_hex in ZONE_COLORS.items():
        draw.rectangle((16, y, 36, y + 14), fill=color_hex)
        draw.text((46, y - 1), f"Zone {zone}", fill="white", font=font)
        y += 22
    legend_path = output_dir / "fundus_zone_mask_montage_legend.png"
    legend.save(legend_path, quality=95)

    if args.video:
        video_rows = resolved_df.reset_index(drop=True)
        video_name = "fundus_zone_mask_qc_video_single.avi" if args.video_mode == "single" else "fundus_zone_mask_qc_video_pages.mp4"
        video_path = output_dir / video_name
        if args.video_mode == "single":
            write_video(video_rows, dataset_root, video_path, args.tile_size, args.alpha, args.fps)
        else:
            page_size = max(1, args.max_images)
            write_page_video(
                video_rows,
                dataset_root,
                output_dir,
                video_path,
                args.tile_size,
                args.columns,
                page_size,
                args.alpha,
                args.fps,
                args.seconds_per_page,
            )
        print(f"[video] {video_path}")

    print(f"[montage] {page_path}")
    print(f"[legend] {legend_path}")
    print(f"[images] {len(df)}")
    print(f"[resolvable_pairs] {len(resolved_df)}")
    print(f"[skipped_missing_fundus] {len(skipped)}")


if __name__ == "__main__":
    main()
