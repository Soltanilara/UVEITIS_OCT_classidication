import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_and_prepare_images(im1_path, im2_path, im3_path):
    # Load grayscale image
    im1 = Image.open(im1_path).convert('L')  # Ensure grayscale
    im1_np = np.array(im1)

    # Load binary masks
    im2 = Image.open(im2_path).convert('1')  # Binary
    im3 = Image.open(im3_path).convert('1')

    # Resize masks to match im1
    im2_resized = im2.resize(im1.size, resample=Image.NEAREST)
    im3_resized = im3.resize(im1.size, resample=Image.NEAREST)

    # Convert resized masks to numpy
    im2_np = np.array(im2_resized)
    im3_np = np.array(im3_resized)

    return im1_np, im2_np, im3_np

def overlay_mask(base_img, mask, color, alpha=0.5):
    """
    base_img: grayscale image (H x W)
    mask: binary mask (H x W), with values {0, 1}
    color: tuple of (R, G, B)
    alpha: transparency of overlay
    """
    h, w = base_img.shape
    base_rgb = np.stack([base_img]*3, axis=-1)  # Convert to RGB
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[mask == 1] = color
    out = (1 - alpha) * base_rgb + alpha * overlay
    return out.astype(np.uint8)

def save_image(img_np, save_path):
    Image.fromarray(img_np).save(save_path)
    print(f"Saved: {save_path}")

def generate_and_save_all(im1_path, im2_path, im3_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    im1, im2, im3 = load_and_prepare_images(im1_path, im2_path, im3_path)

    # 1. im2 overlaid on im1 with black and blue
    over1 = overlay_mask(im1, im2, color=(0, 0, 255), alpha=0.2)
    save_image(over1, os.path.join(output_dir, 'FA_Mohammad_overlay.png'))

    # 2. im3 overlaid on im1 with black and red
    over2 = overlay_mask(im1, im3, color=(255, 0, 0), alpha=0.2)
    save_image(over2, os.path.join(output_dir, 'FA_Exp_overlay.png'))

    # 3. im2 and im3 overlaid on im1 (blue and red)
    combined = overlay_mask(im1, im2, color=(0, 0, 255), alpha=0.2)
    combined = overlay_mask(combined.mean(axis=-1).astype(np.uint8), im3, color=(255, 0, 0), alpha=0.2)
    save_image(combined, os.path.join(output_dir, 'FA_Mohammad_Exp_overlay.png'))

    # 4. im3 overlaid on im2 (binary to RGB)
    im2_rgb = np.stack([im2*255]*3, axis=-1)
    over4 = overlay_mask(im2*255, im3, color=(255, 0, 0), alpha=0.5)
    save_image(over4, os.path.join(output_dir, 'Mohammad_Exp_overlay.png'))

# Example usage
im1_path = 'Exp_90_92/Region Crops/1_Patient090_20250107_OS_FO_0001___Patient090_20250107_OS_FA_0000.png'
im2_path = 'Exp_90_92/SHAP/masks/Patient090_20250107_OS_FO_gt_bin_raw.png'
im3_path = 'Exp_90_92/SHAP/masks/Patient090_20250107_OS_FO_best_pred.png'
output_dir = 'Exp_90_92/SHAP/Patient090 True Exp Annotation Overlays'

# im1_path = 'Exp_90_92/Region Crops/50_Patient092_20250402_OS_FO_0001___Patient092_20250402_OS_FA_0000.png'
# im2_path = 'Exp_90_92/SHAP/masks/Patient092_20250402_OS_FO_gt_bin_raw.png'
# im3_path = 'Exp_90_92/SHAP/masks/Patient092_20250402_OS_FO_best_pred.png'
# output_dir = 'Exp_90_92/SHAP/Patient092 True Exp Annotation Overlays'

generate_and_save_all(im1_path, im2_path, im3_path, output_dir)
