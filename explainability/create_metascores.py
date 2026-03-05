#!/usr/bin/env python3

"""
Script to generate "meta‐scores" by combining IG & SHAP attributions for each test image
into a single (H,W) map, plus 3‐panel visualizations.

Usage Example:
  python create_meta_scores.py \
      --csvpath your_csv_dir \
      --dataset_path "Dataset 01032025" \
      --output_path "output_fold_0/final_finetune_resnet50_pretraining_swav_CE_loss_..." \
      --gpu 0  # (Optional, if GPU usage is relevant)
"""

import os
import numpy as np
import pandas as pd
import cv2
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------
# 1. We import or define your existing "CustomImageDataset" and transforms
#    EXACTLY as in your original code. If you already have them in a separate
#    file, you can do `from your_module import CustomImageDataset, val_transform, ...`
#    For clarity, I’ll show a minimal approach:

from torchvision import transforms

# ------ COPY OR IMPORT your CustomImageDataset here ------
# from your_code import CustomImageDataset, load_data  # or something similar


def load_data(csv_file, csvpath, folder):
    df = pd.read_csv(os.path.join(csvpath,csv_file))
    df['Label'] = df['Label'].map(lambda x: 0 if x == 'negative' else 1)
    paths = df['Image File'].apply(lambda x: os.path.join(folder, x))
    labels = df['Label'].values
    return paths, torch.tensor(labels, dtype=torch.long)

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, csvpath, folder, transform=None):
        self.paths, self.labels = load_data(csv_file, csvpath, folder)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_val_transform():
    """
    Example of a transform for your evaluation/test set,
    consistent with your original code's 'val_transform'.
    Modify if needed.
    """
    # For example:
    return transforms.Compose([
        # transforms.Resize((224,224)),  # Or the size your model expects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
# --------------------------------------------------------------------

def minmax_norm_2d(array_2d):
    """Given a 2D array, returns a [0..1]-scaled version."""
    mi, ma = array_2d.min(), array_2d.max()
    denom = (ma - mi) if (ma != mi) else 1e-8
    return (array_2d - mi) / denom

def combine_ig_shap(ig_3d, shap_3d, normalize=True):
    """
    Combine IG & SHAP arrays of shape (C, H, W) into a single (H, W) map.
    Steps:
      1) Take absolute value => (C, H, W)
      2) Sum across channels => (H, W)
      3) Minmax normalize each => (H, W) in [0,1]
      4) Average them => final (H, W)
    """
    ig_abs   = np.abs(ig_3d)
    shap_abs = np.abs(shap_3d)

    ig_2d   = ig_abs.sum(axis=0)   # shape (H, W)
    shap_2d = shap_abs.sum(axis=0) # shape (H, W)

    ig_2d_norm   = minmax_norm_2d(ig_2d)
    shap_2d_norm = minmax_norm_2d(shap_2d)

    meta_2d = ig_2d_norm * shap_2d_norm
    if normalize:
        mi, ma = meta_2d.min(), meta_2d.max()
        if ma > mi:  # Avoid division by zero
            meta_2d = (meta_2d - mi) / (ma - mi)
        else:
            meta_2d = np.zeros_like(meta_2d)  # Edge case where everything is same value
    return meta_2d  # shape (H, W)

def create_3panel_visualization(pil_img_path, meta_2d):
    """
    Creates a 3‐panel visualization:
      1) Original (in BGR)
      2) meta_2d heatmap (color-coded)
      3) overlay of original * meta_2d
    Returns a BGR image (NumPy array).
    """
    # Load the original image *untransformed*, so we see it in normal pixel space
    original_pil = Image.open(pil_img_path).convert('RGB')
    original_rgb = np.array(original_pil)
    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

    # meta_2d is [0..1], shape (H, W)
    heatmap_8u = (meta_2d * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_VIRIDIS)

    # Overlay
    mask_3c = np.stack([meta_2d]*3, axis=-1)  # shape (H, W, 3)
    overlay_bgr = (original_bgr.astype(float) * mask_3c).astype(np.uint8)

    panel = cv2.hconcat([original_bgr, heatmap_bgr, overlay_bgr])
    return panel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvpath', type=str,default='ext_test',
                        help='Path containing test.csv')
    parser.add_argument('--dataset_path', type=str, default='Dataset 01032025',
                        help='Path to image folder (Dataset) used in test.csv')
    parser.add_argument('--output_path', type=str, default='output_ext_test/final_finetune_resnet50_pretraining_swav_CE_loss_unweighted_batch_64_lr_0.001_decay_1e-05_10_10_1_1e-05_epochs_100_hflip_seed_0',
                        help='Path where IG_* and SHAP_* folders are, also where we save meta-scores')
    parser.add_argument('--test_csv', type=str, default='test.csv',
                        help='Name of your test CSV file (default: test.csv)')
    args = parser.parse_args()

    # 1) Prepare test dataset
    val_transform = get_val_transform()
    test_dataset = CustomImageDataset(
        csv_file=args.test_csv,
        csvpath=args.csvpath,
        folder=args.dataset_path,
        transform=val_transform
    )
    # We won't actually feed them into a DataLoader for this script, we just need the indexing.

    # 2) Paths to existing folders
    IG_NEG_DIR   = os.path.join(args.output_path, "IG_Negative_Scores")
    IG_POS_DIR   = os.path.join(args.output_path, "IG_Positive_Scores")
    SHAP_NEG_DIR = os.path.join(args.output_path, "SHAP_Negative_Scores")
    SHAP_POS_DIR = os.path.join(args.output_path, "SHAP_Positive_Scores")

    # 3) Create new output folders for meta-scores
    META_NEG_SCORES_DIR = os.path.join(args.output_path, "MetaScores_Negative_Scores")
    META_NEG_VIS_DIR    = os.path.join(args.output_path, "MetaScores_Negative")
    META_POS_SCORES_DIR = os.path.join(args.output_path, "MetaScores_Positive_Scores")
    META_POS_VIS_DIR    = os.path.join(args.output_path, "MetaScores_Positive")

    os.makedirs(META_NEG_SCORES_DIR, exist_ok=True)
    os.makedirs(META_NEG_VIS_DIR,    exist_ok=True)
    os.makedirs(META_POS_SCORES_DIR, exist_ok=True)
    os.makedirs(META_POS_VIS_DIR,    exist_ok=True)

    # 4) Loop over each item in the test dataset.
    #    For each idx, we look for IG_Negative_Scores/{idx}.npy, SHAP_Negative_Scores/{idx}.npy, etc.
    #    The assumption is that your previously-run code used exactly this index-based naming
    #    to store the .npy files.

    print("Combining Negative attributions (IG + SHAP)...")
    for idx in tqdm(range(len(test_dataset))):
        ig_neg_path   = os.path.join(IG_NEG_DIR,   f"{idx}.npy")
        shap_neg_path = os.path.join(SHAP_NEG_DIR, f"{idx}.npy")
        if not (os.path.exists(ig_neg_path) and os.path.exists(shap_neg_path)):
            # skip if missing
            continue

        ig_neg   = np.load(ig_neg_path)   # shape (C,H,W)
        shap_neg = np.load(shap_neg_path) # shape (C,H,W)

        meta_neg_2d = combine_ig_shap(ig_neg, shap_neg)  # shape (H,W)

        # Save raw .npy
        out_scores_path = os.path.join(META_NEG_SCORES_DIR, f"{idx}.npy")
        np.save(out_scores_path, meta_neg_2d)

        # Create visualization
        # We want the *untransformed* path to the original image:
        original_img_path = test_dataset.paths[idx]  # e.g. ".../some_image.jpg"
        panel_bgr = create_3panel_visualization(original_img_path, meta_neg_2d)

        out_vis_path = os.path.join(META_NEG_VIS_DIR, f"{idx}.png")
        cv2.imwrite(out_vis_path, panel_bgr)

    print("Combining Positive attributions (IG + SHAP)...")
    for idx in tqdm(range(len(test_dataset))):
        ig_pos_path   = os.path.join(IG_POS_DIR,   f"{idx}.npy")
        shap_pos_path = os.path.join(SHAP_POS_DIR, f"{idx}.npy")
        if not (os.path.exists(ig_pos_path) and os.path.exists(shap_pos_path)):
            continue

        ig_pos   = np.load(ig_pos_path)
        shap_pos = np.load(shap_pos_path)

        meta_pos_2d = combine_ig_shap(ig_pos, shap_pos)

        out_scores_path = os.path.join(META_POS_SCORES_DIR, f"{idx}.npy")
        np.save(out_scores_path, meta_pos_2d)

        original_img_path = test_dataset.paths[idx]
        panel_bgr = create_3panel_visualization(original_img_path, meta_pos_2d)

        out_vis_path = os.path.join(META_POS_VIS_DIR, f"{idx}.png")
        cv2.imwrite(out_vis_path, panel_bgr)

    print("\nDone! Created meta‐scores (npy) and 3‐panel visuals (png) in:")
    print(f"  {META_NEG_SCORES_DIR}/ and {META_NEG_VIS_DIR}/  (Negative class)")
    print(f"  {META_POS_SCORES_DIR}/ and {META_POS_VIS_DIR}/  (Positive class)\n")


if __name__ == "__main__":
    main()
