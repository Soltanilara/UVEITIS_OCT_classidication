import numpy as np
import json
from skimage.transform import resize
import matplotlib.pyplot as plt
import os

# Binarize but DO NOT resize
def binarize_only(data, threshold=127):
    processed = []
    for lst in data:
        binary = np.where(np.array(lst) > threshold, 1, 0).astype(np.uint8)
        processed.append(binary)
    return np.flipud(np.array(processed, dtype=np.uint8))

def binarize_and_resize(data, target_length=512, threshold=127):
    processed = []

    for lst in data:
        # Convert to binary (0 or 255)
        binary = np.where(np.array(lst) > threshold, 1, 0).astype(np.uint8)

        # Resize to target_length (with interpolation)
        # Normalize to [0, 1] before resizing
        resized = resize(binary, (target_length,), order=0, mode='edge', anti_aliasing=False)

        # Threshold again after resizing to get binary
        # binary_resized = (resized > 0.5).astype(np.uint8) * 255

        processed.append(resized)

    return np.flipud(np.array(processed, dtype=np.uint8))

def compute_iou(gt, pred_bin):
    intersection = np.logical_and(gt, pred_bin).sum()
    union = np.logical_or(gt, pred_bin).sum()
    return intersection / union if union != 0 else 0

def compute_dice(gt, pred_bin):
    intersection = np.logical_and(gt, pred_bin).sum()
    return 2 * intersection / (gt.sum() + pred_bin.sum()) if (gt.sum() + pred_bin.sum()) != 0 else 0

def find_best_threshold(gt, pred, criterion = 'iou'):
    best = -1
    best_threshold = -1
    best_pred_bin = None

    for t in range(256):
        pred_bin = (pred >= t).astype(np.uint8)
        if criterion=='iou':
            cur = compute_iou(gt, pred_bin)
        elif criterion=='dice':
            cur = compute_dice(gt, pred_bin)
        if cur > best:
            best = cur
            best_threshold = t
            best_pred_bin = pred_bin

    best_dice = compute_dice(gt, best_pred_bin)
    best_iou = compute_iou(gt, best_pred_bin)
    return best_threshold, best_iou, best_dice, best_pred_bin

def repeat_rows(img, times=10):
    return np.repeat(img, repeats=times, axis=0)

def visualize(gt, pred_bin, threshold, save_path=None):
    gt_vis = repeat_rows(gt * 255)
    pred_vis = repeat_rows(pred_bin * 255)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(gt_vis, cmap='gray')
    axs[0].set_title('Ground Truth')
    axs[1].imshow(pred_vis, cmap='gray')
    axs[1].set_title(f'Prediction (thr={threshold})')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()

    # Save the plot if path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()

def save_metrics(threshold, iou, dice, filepath="metrics.txt"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(f"Best Threshold: {threshold}\n")
        f.write(f"IoU: {iou:.4f}\n")
        f.write(f"Dice Score: {dice:.4f}\n")
    print(f"Metrics saved to {filepath}")

gt_path = 'Exp_90_92/test_90_92.json'

# pred_path = 'Exp_90_92/SHAP/Patient092_20250402_OS_FO.npy'
pred_path = 'Exp_90_92/SHAP/Patient090_20250107_OS_FO.npy'

# pred_path = 'Exp_90_92/GradCAM/Patient092_20250402_OS_FO.npy'
# pred_path = 'Exp_90_92/GradCAM/Patient090_20250107_OS_FO.npy'

# pred_path = 'Exp_90_92/IG/Patient092_20250402_OS_FO.npy'
# pred_path = 'Exp_90_92/IG/Patient090_20250107_OS_FO.npy'
plot_path = pred_path[:-4]+'.png'
metric_path = pred_path[:-4]+'_metrics.txt'
with open(gt_path, 'r') as f:
    gt = json.load(f)
# Save binarized ground truth (after binarization, before resizing)
with open(gt_path, 'r') as f:
    raw_gt = json.load(f)

gt_bin_only = binarize_only(raw_gt[:49])  # apply same slice as before

gt = binarize_and_resize(gt[:49])
# gt = binarize_and_resize(gt[:49])
# gt = binarize_and_resize(gt[49:],target_length=16)
# gt = binarize_and_resize(gt[:49],target_length=16)
pred = np.load(pred_path)

best_threshold, best_iou, best_dice, best_pred_bin = find_best_threshold(gt, pred, criterion = 'iou')
# best_threshold, best_iou, best_dice, best_pred_bin = find_best_threshold(gt, pred, criterion = 'dice')
visualize(gt, best_pred_bin, best_threshold, save_path=plot_path)
save_metrics(best_threshold, best_iou, best_dice, filepath=metric_path)

print(f"Best threshold: {best_threshold}")
print(f"IoU: {best_iou:.4f}")
print(f"Dice: {best_dice:.4f}")

from PIL import Image

# Create output directory for masks
mask_output_dir = os.path.join(os.path.dirname(pred_path), 'masks')
os.makedirs(mask_output_dir, exist_ok=True)


gt_img = Image.fromarray(gt_bin_only * 255)  # scale to [0, 255] for visualization
gt_img.save(os.path.join(mask_output_dir, os.path.basename(pred_path)[:-4] + '_gt_bin_raw.png'))

# Save best prediction binary mask
pred_img = Image.fromarray(best_pred_bin * 255)
pred_img.save(os.path.join(mask_output_dir, os.path.basename(pred_path)[:-4] + '_best_pred.png'))

print("Binarized ground truth and prediction saved.")