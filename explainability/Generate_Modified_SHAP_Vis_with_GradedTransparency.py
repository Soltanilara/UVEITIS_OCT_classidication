#!/usr/bin/env python
"""
Generate SHAP visualizations that mirror the Grad‑CAM style but **drop negative evidence *per channel before aggregation***.

Changes from v1
----------------
* Apply **ReLU first, then sum** (channel‑wise) so no negative SHAP value can cancel out positives from another channel — exactly what you asked for.
* Docstring and comments updated accordingly.

CLI usage
~~~~~~~~~
python shap_positive_visualizations.py \
    --scores_root  /path/to/output_fold_0/... \
    --csv_file     /path/to/test.csv \
    --dataset_dir  "Dataset 01032025"
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# --------------------------- helper functions --------------------------- #

def get_image_path(idx: int, df: pd.DataFrame, dataset_dir: Path) -> Path:
    """Return the absolutized path to the *idx*-th image in *df*."""
    rel = df.iloc[idx]["Image File"]
    return dataset_dir / rel


def build_overlay(original_bgr: np.ndarray, pos_map: np.ndarray, min_alpha: float = 0.1) -> np.ndarray:
    """
    Resize *pos_map* to *original*, apply JET colormap, and alpha-blend according to SHAP score.
    Transparency is proportional to SHAP score, with a minimum alpha.
    """
    h, w = original_bgr.shape[:2]
    pos_map_resized = cv2.resize(pos_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize and clip SHAP scores for transparency
    alpha_map = np.clip(pos_map_resized, 0, 1)  # Ensure in [0, 1]
    alpha_map[alpha_map<min_alpha] = 0
    # alpha_map = min_alpha + (1 - min_alpha) * alpha_map  # Map [0,1] to [min_alpha,1]

    # Apply JET colormap
    pos_map_8u = np.uint8(511 * pos_map_resized)
    heatmap = cv2.applyColorMap(pos_map_8u, cv2.COLORMAP_JET)

    # Convert to float32 for blending
    original_bgr_f = original_bgr.astype(np.float32) / 255
    heatmap_f = heatmap.astype(np.float32) / 255
    alpha_map_f = alpha_map[..., np.newaxis]  # Expand to (H, W, 1)

    # Alpha blend
    blended = (1 - alpha_map_f) * original_bgr_f + alpha_map_f * heatmap_f
    blended_uint8 = np.uint8(blended * 255)

    return blended_uint8


def process_one(idx: int, score_path: Path, out_dir: Path, df: pd.DataFrame, dataset_dir: Path):
    """Create the Grad‑CAM‑style visualization for a single index."""
    # 1) load SHAP tensor  (shape ≈ (3, H, W)  or  (H, W))
    score = np.load(score_path)

    # 2) keep only positive evidence **per channel** (ReLU)
    if score.ndim == 3:
        score = np.maximum(score, 0)       # ReLU across each RGB channel
        score = score.sum(axis=0)          # then aggregate
    else:
        score = np.maximum(score, 0)

    # normalise
    if score.max() != 0:
        score = score / score.max()

    # 3) original image (RGB → BGR for OpenCV)
    img_path = get_image_path(idx, df, dataset_dir)
    original_rgb = np.array(Image.open(img_path).convert("RGB"))
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    # 4) overlay
    overlay_bgr = build_overlay(original_bgr, score)

    # 5) concatenate & save
    concatenated = cv2.hconcat([original_bgr, overlay_bgr])
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{idx}.png"), concatenated)


# ------------------------------ main CLI ------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Create positive‑evidence SHAP visualizations (ReLU before sum).")
    parser.add_argument("--scores_root", type=Path,
                        default = "output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0",
                        help="Root dir that has SHAP_Positive_Scores etc.")
    parser.add_argument("--csv_file", type=Path,
                        default = "ext_test/test.csv",
                        help="CSV with 'Image File' column used in training.")
    parser.add_argument("--dataset_dir", type=Path, default='Dataset 01032025',
                        help="Directory containing the raw images.")
    parser.add_argument("--classes", nargs="*", default=["Positive", "Negative"],
                        help="Which SHAP class folders to process.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    for cls in args.classes:
        in_dir = args.scores_root / f"SHAP_{cls}_Scores"
        out_dir = args.scores_root / f"SHAP_{cls}_Modified_Graded_Transparency"
        if not in_dir.exists():
            print(f"[WARN] {in_dir} not found; skipping {cls} class.")
            continue
        npy_files = sorted(in_dir.glob("*.npy"))
        print(f"Processing {len(npy_files)} files in {in_dir.name} → {out_dir.name}")
        for npy_path in npy_files:
            try:
                idx = int(npy_path.stem)
            except ValueError:
                print(f"  skipping {npy_path.name} (stem is not an int)")
                continue
            process_one(idx, npy_path, out_dir, df, args.dataset_dir)

    print("✓ All done – modified SHAP visualizations saved.")


if __name__ == "__main__":
    main()
