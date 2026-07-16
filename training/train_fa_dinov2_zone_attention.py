#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm


NUM_TOTAL_ZONES = 10
NUM_TARGET_ZONES = 9
PATCH_SIZE = 14
ZONE_COLUMNS = [f"Zone{i}_label" for i in range(1, NUM_TOTAL_ZONES + 1)]
TARGET_ZONE_COLUMNS = ZONE_COLUMNS[:NUM_TARGET_ZONES]
FALLBACK_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
DINO_ARCHES = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
}
wandb_run = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone FA zone classifier: full FA image -> DINOv2 patch tokens -> "
            "soft zone-mask attention pooling -> shared binary MLP for Zones 1-9."
        )
    )
    parser.add_argument("--csvpath", type=str, default="fold_zone_masks_ready_patient_split/fold_0")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical",
        help="Fallback root for relative FA image paths.",
    )
    parser.add_argument(
        "--mask_dataset_path",
        type=str,
        default="/mnt/NAS/Shashank/datasets/UveitisFundus/Sample 2.5.2026_canonical_fa_zone_masks",
        help="Fallback root for relative FA zone-mask paths.",
    )
    parser.add_argument("--output_path", type=str, default="output_fa_dinov2_zone_attention")
    parser.add_argument("--image_absolute_column", type=str, default="FA_Image_Abs_Path")
    parser.add_argument("--mask_absolute_column", type=str, default="FA_Mask_Abs_Path")
    parser.add_argument("--image_column", type=str, default="Image_File(FA)")
    parser.add_argument("--mask_column", type=str, default="FA_Mask_Path")
    parser.add_argument("--drop_missing_zone_rows", choices=["none", "any", "all"], default="all")
    parser.add_argument("--dinov2_arch", choices=sorted(DINO_ARCHES), default="dinov2_vitb14")
    parser.add_argument("--image_size", type=int, default=392, help="392 gives a 28x28 patch grid for DINOv2 patch-14.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unweighted", action="store_true", help="Disable per-zone BCE positive weights.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fallback threshold used for training metrics.")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum backbone LR; the head keeps the same LR ratio.")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Request deterministic PyTorch/CUDA execution and use a separately seeded training DataLoader.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true", help="Initialize DINOv2 architecture without pretrained weights.")
    tracking = parser.add_argument_group("Weights & Biases")
    tracking.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    tracking.add_argument("--wandb_project", type=str, default="uveitis-fa-zone-attention")
    tracking.add_argument("--wandb_entity", type=str, default="")
    tracking.add_argument("--wandb_name", type=str, default="")
    tracking.add_argument("--wandb_group", type=str, default="")
    tracking.add_argument("--wandb_tags", type=str, default="", help="Comma-separated W&B tags.")
    tracking.add_argument("--wandb_mode", choices=["", "online", "offline", "disabled"], default="")
    augmentation = parser.add_argument_group("training augmentation")
    augmentation.add_argument("--rotation", action="store_true", help="Enable random rotation augmentation.")
    augmentation.add_argument("--rotation_prob", type=float, default=0.7)
    augmentation.add_argument("--rotation_degrees", type=float, default=10.0, help="Maximum absolute rotation in degrees.")
    augmentation.add_argument("--translation", action="store_true", help="Enable random translation augmentation.")
    augmentation.add_argument("--translation_prob", type=float, default=0.5)
    augmentation.add_argument(
        "--translation_fraction",
        type=float,
        default=0.05,
        help="Maximum absolute translation as a fraction of image width/height.",
    )
    augmentation.add_argument("--scale", action="store_true", help="Enable random scale augmentation.")
    augmentation.add_argument("--scale_prob", type=float, default=0.5)
    augmentation.add_argument("--scale_min", type=float, default=0.9)
    augmentation.add_argument("--scale_max", type=float, default=1.1)
    augmentation.add_argument("--brightness", action="store_true", help="Enable random brightness augmentation.")
    augmentation.add_argument("--brightness_prob", type=float, default=0.5)
    augmentation.add_argument(
        "--brightness_fraction",
        type=float,
        default=0.10,
        help="Maximum fractional brightness change (0.10 means +/-10%%).",
    )
    augmentation.add_argument("--contrast", action="store_true", help="Enable random contrast augmentation.")
    augmentation.add_argument("--contrast_prob", type=float, default=0.5)
    augmentation.add_argument("--contrast_min", type=float, default=0.9)
    augmentation.add_argument("--contrast_max", type=float, default=1.2)
    augmentation.add_argument("--gamma", action="store_true", help="Enable random gamma augmentation.")
    augmentation.add_argument("--gamma_prob", type=float, default=0.5)
    augmentation.add_argument("--gamma_min", type=float, default=0.8)
    augmentation.add_argument("--gamma_max", type=float, default=1.2)
    augmentation.add_argument("--gaussian_noise", action="store_true", help="Enable Gaussian noise augmentation.")
    augmentation.add_argument("--gaussian_noise_prob", type=float, default=0.3)
    augmentation.add_argument("--gaussian_noise_sigma_min", type=float, default=2.0, help="Noise sigma in 8-bit intensity units.")
    augmentation.add_argument("--gaussian_noise_sigma_max", type=float, default=5.0, help="Noise sigma in 8-bit intensity units.")
    augmentation.add_argument("--gaussian_blur", action="store_true", help="Enable Gaussian blur augmentation.")
    augmentation.add_argument("--gaussian_blur_prob", type=float, default=0.2)
    augmentation.add_argument("--gaussian_blur_sigma_min", type=float, default=0.1)
    augmentation.add_argument("--gaussian_blur_sigma_max", type=float, default=0.5)
    augmentation.add_argument("--clahe", action="store_true", help="Enable stochastic CLAHE augmentation.")
    augmentation.add_argument("--clahe_prob", type=float, default=0.3)
    augmentation.add_argument("--clahe_clip_limit", type=float, default=2.0)
    augmentation.add_argument("--clahe_grid_size", type=int, default=8, help="Number of CLAHE tiles along each image axis.")
    augmentation.add_argument("--random_erasing", action="store_true", help="Enable random erasing augmentation.")
    augmentation.add_argument("--random_erasing_prob", type=float, default=0.1)
    augmentation.add_argument(
        "--random_erasing_max_area",
        type=float,
        default=0.02,
        help="Maximum erased fraction of image area.",
    )
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def init_wandb(args: argparse.Namespace) -> None:
    global wandb_run
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Weights & Biases logging was requested with --wandb, but wandb is not installed.") from exc

    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
    fold_name = os.path.basename(os.path.normpath(args.csvpath))
    run_name = args.wandb_name or os.path.basename(os.path.normpath(args.output_path))
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    enabled_augmentations = [
        name
        for name in (
            "rotation",
            "translation",
            "scale",
            "brightness",
            "contrast",
            "gamma",
            "gaussian_noise",
            "gaussian_blur",
            "clahe",
            "random_erasing",
        )
        if getattr(args, name)
    ]
    tags.extend(["dinov2-zone-attention", *enabled_augmentations, fold_name, f"seed:{args.seed}"])
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        group=args.wandb_group or None,
        tags=tags,
        dir=args.output_path,
        config={key: value for key, value in vars(args).items() if key != "wandb"},
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")
    wandb.define_metric("val_zone/*", step_metric="epoch")
    wandb.define_metric("early_stopping/*", step_metric="epoch")
    wandb.define_metric("learning_rate/*", step_metric="epoch")


def wandb_log(metrics: dict[str, Any], step: int | None = None) -> None:
    if wandb_run is None:
        return
    if step is not None:
        metrics = {"epoch": step, **metrics}
    wandb_run.log(metrics)


def finish_wandb() -> None:
    if wandb_run is not None:
        wandb_run.finish()


def resolve_existing_path(base_folder: str, path_value: Any) -> str:
    if pd.isna(path_value):
        raise FileNotFoundError("Path value is missing.")
    path_str = str(path_value).strip().replace("\\", "/")
    candidates = []

    if os.path.isabs(path_str):
        candidates.append(path_str)
    elif base_folder:
        candidates.append(os.path.join(base_folder, path_str))
    candidates.append(path_str)

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
        root, _ = os.path.splitext(candidate)
        for ext in FALLBACK_EXTS:
            for ext_candidate in (root + ext, root + ext.upper()):
                if os.path.exists(ext_candidate):
                    return ext_candidate

    raise FileNotFoundError(f"Could not resolve path {path_value!r}; tried {candidates[:4]}")


def zone_stack_from_mask_array(mask: np.ndarray, mask_path: str) -> np.ndarray:
    if mask.ndim == 2:
        return np.stack([(mask == zone_id) for zone_id in range(1, NUM_TOTAL_ZONES + 1)], axis=0).astype(np.uint8)

    if mask.ndim == 3:
        if mask.shape[0] >= NUM_TOTAL_ZONES:
            return (mask[:NUM_TOTAL_ZONES] > 0).astype(np.uint8)
        if mask.shape[-1] >= NUM_TOTAL_ZONES:
            return np.moveaxis((mask[..., :NUM_TOTAL_ZONES] > 0).astype(np.uint8), -1, 0)

    raise ValueError(f"Unsupported zone mask shape in {mask_path}: {tuple(mask.shape)}")


def resize_zone_stack(zone_stack: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    resized = []
    for zone_idx in range(zone_stack.shape[0]):
        zone_image = Image.fromarray(zone_stack[zone_idx].astype(np.uint8), mode="L")
        resized.append(np.asarray(zone_image.resize(size, resample), dtype=np.float32))
    return np.stack(resized, axis=0)


@dataclass
class SplitData:
    image_paths: list[str]
    mask_paths: list[str]
    labels: torch.Tensor
    observed_mask: torch.Tensor
    zone_nonempty: torch.Tensor
    empty_mask_records: list[dict[str, Any]]
    metadata: dict[str, list[str]]


def read_split(csv_file: str, args: argparse.Namespace) -> SplitData:
    df = pd.read_csv(csv_file)
    missing = [col for col in TARGET_ZONE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_file} is missing required zone columns: {missing}")

    zone_df = df[TARGET_ZONE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if args.drop_missing_zone_rows == "any":
        keep_mask = ~zone_df.isna().any(axis=1).to_numpy()
    elif args.drop_missing_zone_rows == "all":
        keep_mask = ~zone_df.isna().all(axis=1).to_numpy()
    else:
        keep_mask = np.ones(len(df), dtype=bool)

    df = df.loc[keep_mask].reset_index(drop=True)
    zone_df = zone_df.loc[keep_mask].reset_index(drop=True)
    observed_mask = torch.tensor(zone_df.notna().to_numpy(dtype=bool), dtype=torch.bool)
    labels_df = zone_df.apply(lambda col: col.map(lambda value: -1 if pd.isna(value) else int(int(round(float(value))) != 0)))
    labels = torch.tensor(labels_df.to_numpy(dtype=np.int64), dtype=torch.long)

    image_paths = []
    mask_paths = []
    for _, row in df.iterrows():
        if args.image_absolute_column in df.columns and not pd.isna(row[args.image_absolute_column]):
            image_paths.append(resolve_existing_path(args.dataset_path, row[args.image_absolute_column]))
        elif args.image_column in df.columns:
            image_paths.append(resolve_existing_path(args.dataset_path, row[args.image_column]))
        else:
            raise ValueError(
                f"{csv_file} needs either {args.image_absolute_column!r} or {args.image_column!r}."
            )

        if args.mask_absolute_column in df.columns and not pd.isna(row[args.mask_absolute_column]):
            mask_paths.append(resolve_existing_path(args.mask_dataset_path, row[args.mask_absolute_column]))
        elif args.mask_column in df.columns:
            mask_paths.append(resolve_existing_path(args.mask_dataset_path, row[args.mask_column]))
        else:
            raise ValueError(f"{csv_file} needs either {args.mask_absolute_column!r} or {args.mask_column!r}.")

    zone_nonempty_rows = []
    empty_mask_records = []
    for image_path, mask_path in zip(image_paths, mask_paths, strict=True):
        raw_mask = np.load(mask_path)
        zone_stack = zone_stack_from_mask_array(raw_mask, mask_path)
        if zone_stack.shape[0] < NUM_TOTAL_ZONES:
            raise ValueError(f"Zone 10 mask is required to remove Zone 10: {mask_path}")
        zone_nonempty = zone_stack[:NUM_TARGET_ZONES].reshape(NUM_TARGET_ZONES, -1).sum(axis=1) > 0
        zone_nonempty_rows.append(zone_nonempty)
        for zone_idx in np.flatnonzero(~zone_nonempty):
            empty_mask_records.append(
                {
                    "image_path": image_path,
                    "zone": int(zone_idx + 1),
                    "mask_path": mask_path,
                    "empty_mask_count": 1,
                }
            )

    zone_nonempty_tensor = torch.tensor(np.stack(zone_nonempty_rows), dtype=torch.bool)
    observed_mask &= zone_nonempty_tensor

    image_id_col = args.image_column if args.image_column in df.columns else df.columns[0]
    metadata = {
        "image_file": df[image_id_col].astype(str).tolist(),
        "patient_id": df["Patient_ID"].astype(str).tolist() if "Patient_ID" in df.columns else [""] * len(df),
        "eye": df["Eye"].astype(str).tolist() if "Eye" in df.columns else [""] * len(df),
        "visit_date": df["Visit_Date"].astype(str).tolist() if "Visit_Date" in df.columns else [""] * len(df),
    }
    return SplitData(
        image_paths=image_paths,
        mask_paths=mask_paths,
        labels=labels,
        observed_mask=observed_mask,
        zone_nonempty=zone_nonempty_tensor,
        empty_mask_records=empty_mask_records,
        metadata=metadata,
    )


def validate_augmentation_args(args: argparse.Namespace) -> None:
    probability_names = [
        "rotation_prob",
        "translation_prob",
        "scale_prob",
        "brightness_prob",
        "contrast_prob",
        "gamma_prob",
        "gaussian_noise_prob",
        "gaussian_blur_prob",
        "clahe_prob",
        "random_erasing_prob",
    ]
    for name in probability_names:
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be between 0 and 1; got {value}.")

    ordered_positive_ranges = [
        ("scale_min", "scale_max"),
        ("contrast_min", "contrast_max"),
        ("gamma_min", "gamma_max"),
        ("gaussian_noise_sigma_min", "gaussian_noise_sigma_max"),
        ("gaussian_blur_sigma_min", "gaussian_blur_sigma_max"),
    ]
    for minimum_name, maximum_name in ordered_positive_ranges:
        minimum = getattr(args, minimum_name)
        maximum = getattr(args, maximum_name)
        if minimum < 0.0 or maximum < minimum:
            raise ValueError(f"Require 0 <= --{minimum_name} <= --{maximum_name}.")
    if args.scale_min <= 0.0 or args.gamma_min <= 0.0:
        raise ValueError("Scale and gamma values must be greater than zero.")
    if args.rotation_degrees < 0.0 or not 0.0 <= args.translation_fraction <= 1.0:
        raise ValueError("Require --rotation_degrees >= 0 and 0 <= --translation_fraction <= 1.")
    if not 0.0 <= args.brightness_fraction <= 1.0:
        raise ValueError("--brightness_fraction must be between 0 and 1.")
    if args.clahe_clip_limit <= 0.0 or args.clahe_grid_size <= 0:
        raise ValueError("CLAHE clip limit and grid size must be greater than zero.")
    if not 0.0 <= args.random_erasing_max_area <= 1.0:
        raise ValueError("--random_erasing_max_area must be between 0 and 1.")


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def apply_mask_aware_affine(
    image: Image.Image,
    zone_stack: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Image.Image, np.ndarray]:
    """Sample one affine transform and apply it identically to image and masks."""
    angle = (
        random.uniform(-args.rotation_degrees, args.rotation_degrees)
        if args.rotation and random.random() < args.rotation_prob
        else 0.0
    )
    if args.translation and random.random() < args.translation_prob:
        max_dx = args.translation_fraction * image.width
        max_dy = args.translation_fraction * image.height
        translate = [int(round(random.uniform(-max_dx, max_dx))), int(round(random.uniform(-max_dy, max_dy)))]
    else:
        translate = [0, 0]
    scale = (
        random.uniform(args.scale_min, args.scale_max)
        if args.scale and random.random() < args.scale_prob
        else 1.0
    )

    image = TF.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    transformed_masks = []
    for zone_mask in zone_stack:
        mask_image = Image.fromarray((zone_mask > 0).astype(np.uint8) * 255, mode="L")
        mask_image = TF.affine(
            mask_image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        transformed_masks.append((np.asarray(mask_image, dtype=np.uint8) > 0).astype(np.float32))
    return image, np.stack(transformed_masks, axis=0)


def apply_clahe(image: Image.Image, clip_limit: float, grid_size: int) -> Image.Image:
    """Apply CLAHE to luminance while preserving the image's chroma channels."""
    try:
        from skimage import exposure
    except ImportError as exc:
        raise ImportError("CLAHE augmentation requires scikit-image (`pip install scikit-image`).") from exc

    ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.uint8).copy()
    height, width = ycbcr.shape[:2]
    kernel_size = (max(1, math.ceil(height / grid_size)), max(1, math.ceil(width / grid_size)))
    # skimage uses a normalized clip limit; 0.02 corresponds to the commonly
    # used OpenCV-style clip limit of approximately 2.
    normalized_clip_limit = min(1.0, clip_limit / 100.0)
    luminance = exposure.equalize_adapthist(
        ycbcr[..., 0].astype(np.float32) / 255.0,
        kernel_size=kernel_size,
        clip_limit=normalized_clip_limit,
    )
    ycbcr[..., 0] = np.clip(np.rint(luminance * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(ycbcr, mode="YCbCr").convert("RGB")


def apply_random_erasing(image: Image.Image, max_area_fraction: float) -> Image.Image:
    """Erase at most the requested image fraction using the image mean."""
    if max_area_fraction <= 0.0:
        return image
    image_array = np.asarray(image, dtype=np.uint8).copy()
    height, width = image_array.shape[:2]
    image_area = height * width
    max_area_pixels = max(1, int(math.floor(max_area_fraction * image_area)))
    target_area = random.randint(1, max_area_pixels)
    aspect_ratio = random.uniform(0.5, 2.0)
    erase_height = min(height, max(1, int(round(math.sqrt(target_area / aspect_ratio)))))
    erase_width = min(width, max(1, int(round(math.sqrt(target_area * aspect_ratio)))))
    while erase_height * erase_width > max_area_pixels:
        if erase_height >= erase_width and erase_height > 1:
            erase_height -= 1
        elif erase_width > 1:
            erase_width -= 1
        else:
            break
    top = random.randint(0, height - erase_height)
    left = random.randint(0, width - erase_width)
    fill = np.rint(image_array.reshape(-1, 3).mean(axis=0)).astype(np.uint8)
    image_array[top : top + erase_height, left : left + erase_width] = fill
    return Image.fromarray(image_array, mode="RGB")


def apply_intensity_augmentations(image: Image.Image, args: argparse.Namespace) -> Image.Image:
    if args.clahe and random.random() < args.clahe_prob:
        image = apply_clahe(image, args.clahe_clip_limit, args.clahe_grid_size)
    if args.brightness and random.random() < args.brightness_prob:
        factor = random.uniform(1.0 - args.brightness_fraction, 1.0 + args.brightness_fraction)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if args.contrast and random.random() < args.contrast_prob:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(args.contrast_min, args.contrast_max))
    if args.gamma and random.random() < args.gamma_prob:
        image = TF.adjust_gamma(image, gamma=random.uniform(args.gamma_min, args.gamma_max), gain=1.0)
    if args.gaussian_blur and random.random() < args.gaussian_blur_prob:
        sigma = random.uniform(args.gaussian_blur_sigma_min, args.gaussian_blur_sigma_max)
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if args.gaussian_noise and random.random() < args.gaussian_noise_prob:
        sigma = random.uniform(args.gaussian_noise_sigma_min, args.gaussian_noise_sigma_max)
        image_array = np.asarray(image, dtype=np.float32)
        # One noise field shared by all channels keeps monochromatic FA images
        # monochromatic instead of introducing artificial color speckle.
        noise = np.random.normal(0.0, sigma, size=(*image_array.shape[:2], 1))
        image = Image.fromarray(np.clip(np.rint(image_array + noise), 0, 255).astype(np.uint8), mode="RGB")
    if args.random_erasing and random.random() < args.random_erasing_prob:
        image = apply_random_erasing(image, args.random_erasing_max_area)
    return image


def letterbox_image_and_masks(
    image: Image.Image,
    zone_stack: np.ndarray,
    target_size: int,
) -> tuple[Image.Image, np.ndarray]:
    """Resize image and masks with one scale, then apply identical centered padding."""
    width, height = image.size
    scale = target_size / max(height, width)
    resized_width = min(target_size, max(1, int(round(width * scale))))
    resized_height = min(target_size, max(1, int(round(height * scale))))
    resized_size = (resized_width, resized_height)
    left = (target_size - resized_width) // 2
    top = (target_size - resized_height) // 2

    resized_image = image.resize(resized_size, Image.Resampling.BILINEAR)
    padded_image = Image.new("RGB", (target_size, target_size), color=(0, 0, 0))
    padded_image.paste(resized_image, (left, top))

    resized_masks = resize_zone_stack(zone_stack, resized_size, Image.Resampling.NEAREST)
    padded_masks = np.zeros((zone_stack.shape[0], target_size, target_size), dtype=np.float32)
    padded_masks[:, top : top + resized_height, left : left + resized_width] = resized_masks
    return padded_image, padded_masks


class FAZoneDataset(Dataset):
    def __init__(self, split: SplitData, args: argparse.Namespace, train: bool):
        self.split = split
        self.args = args
        self.train = train
        self.image_transform = build_transform()

    def __len__(self) -> int:
        return len(self.split.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image = Image.open(self.split.image_paths[idx]).convert("RGB")
        raw_mask = np.load(self.split.mask_paths[idx])
        zone_stack = zone_stack_from_mask_array(raw_mask, self.split.mask_paths[idx])
        if zone_stack.shape[0] < NUM_TOTAL_ZONES:
            raise ValueError(f"Zone 10 mask is required to remove Zone 10: {self.split.mask_paths[idx]}")

        if zone_stack.shape[1:] != (image.height, image.width):
            zone_stack = resize_zone_stack(zone_stack, image.size, Image.Resampling.NEAREST)

        image_arr = np.asarray(image, dtype=np.uint8)
        keep_mask = (zone_stack[9] == 0).astype(np.uint8)
        image = Image.fromarray(image_arr * keep_mask[..., None], mode="RGB")
        image, zone_stack = letterbox_image_and_masks(image, zone_stack, self.args.image_size)
        if self.train:
            if self.args.rotation or self.args.translation or self.args.scale:
                image, zone_stack = apply_mask_aware_affine(image, zone_stack, self.args)
            image = apply_intensity_augmentations(image, self.args)
        full_image = self.image_transform(image)

        zone_masks = np.clip(zone_stack[:NUM_TARGET_ZONES], 0.0, 1.0).astype(np.float32)
        zone_nonempty = zone_masks.reshape(NUM_TARGET_ZONES, -1).sum(axis=1) > 0
        observed_mask = self.split.observed_mask[idx] & torch.from_numpy(zone_nonempty)

        return {
            "full_image": full_image,
            "zone_masks": torch.from_numpy(zone_masks),
            "labels": self.split.labels[idx],
            "observed_mask": observed_mask,
            "zone_nonempty": torch.from_numpy(zone_nonempty),
            "image_file": self.split.metadata["image_file"][idx],
            "patient_id": self.split.metadata["patient_id"][idx],
            "eye": self.split.metadata["eye"][idx],
            "visit_date": self.split.metadata["visit_date"][idx],
            "image_path": self.split.image_paths[idx],
            "mask_path": self.split.mask_paths[idx],
        }


class DinoTokenBackbone(nn.Module):
    def __init__(self, arch: str, image_size: int, pretrained: bool):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("This script requires timm for DINOv2 backbones.") from exc

        self.backbone = timm.create_model(
            DINO_ARCHES[arch],
            pretrained=pretrained,
            img_size=image_size,
            num_classes=0,
        )
        self.feature_dim = int(getattr(self.backbone, "num_features", getattr(self.backbone, "embed_dim", 0)))
        if self.feature_dim <= 0:
            raise ValueError("Could not infer DINOv2 feature dimension.")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            cls_token = features.get("x_norm_clstoken")
            patch_tokens = features.get("x_norm_patchtokens")
            if cls_token is not None and patch_tokens is not None:
                return cls_token, patch_tokens
            token_tensor = features.get("x_norm")
            if token_tensor is None:
                token_tensor = features.get("x")
            if token_tensor is None:
                token_tensor = features.get("tokens")
            if token_tensor is None:
                raise ValueError(f"Unsupported DINOv2 feature keys: {sorted(features.keys())}")
            features = token_tensor

        if not torch.is_tensor(features) or features.ndim != 3 or features.shape[1] < 2:
            raise ValueError(f"Expected DINOv2 token tensor [B,N,D], got {type(features)} {getattr(features, 'shape', None)}")
        return features[:, 0], features[:, 1:]


class SoftZoneAttentionPooling(nn.Module):
    def __init__(self, feature_dim: int, num_zones: int = NUM_TARGET_ZONES, eps: float = 1e-6):
        super().__init__()
        self.zone_queries = nn.Parameter(torch.randn(num_zones, feature_dim) * 0.02)
        self.norm = nn.LayerNorm(feature_dim)
        self.eps = eps
        self.scale = feature_dim ** -0.5

    def forward(self, patch_tokens: torch.Tensor, zone_masks: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        batch_size, num_tokens, feature_dim = patch_tokens.shape
        hpatch, wpatch = grid_size
        if num_tokens != hpatch * wpatch:
            raise ValueError(f"Patch token count {num_tokens} does not match grid {grid_size}.")
        if zone_masks.shape[1] != self.zone_queries.shape[0]:
            raise ValueError(f"Expected {self.zone_queries.shape[0]} zone masks, got {zone_masks.shape[1]}.")

        soft_masks = F.interpolate(zone_masks.float(), size=grid_size, mode="area")
        soft_masks = soft_masks.reshape(batch_size, zone_masks.shape[1], hpatch * wpatch).clamp(0.0, 1.0)
        nonempty = soft_masks.sum(dim=-1, keepdim=True) > self.eps

        norm_tokens = self.norm(patch_tokens)
        norm_queries = F.normalize(self.zone_queries, dim=-1)
        scores = torch.einsum("bnd,zd->bzn", norm_tokens, norm_queries) * self.scale
        scores = scores + soft_masks.clamp_min(self.eps).log()
        attention = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("bzn,bnd->bzd", attention, patch_tokens)
        return pooled * nonempty.to(pooled.dtype)


class FAZoneDinoClassifier(nn.Module):
    def __init__(self, arch: str, image_size: int, pretrained: bool, dropout: float):
        super().__init__()
        if image_size % PATCH_SIZE != 0:
            raise ValueError(f"--image_size must be divisible by {PATCH_SIZE}; got {image_size}.")
        self.backbone = DinoTokenBackbone(arch=arch, image_size=image_size, pretrained=pretrained)
        self.pool = SoftZoneAttentionPooling(feature_dim=self.backbone.feature_dim)
        hidden_dim = self.backbone.feature_dim
        self.head = nn.Sequential(
            nn.Linear(2 * self.backbone.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.grid_size = (image_size // PATCH_SIZE, image_size // PATCH_SIZE)

    def forward(self, full_image: torch.Tensor, zone_masks: torch.Tensor) -> torch.Tensor:
        cls_token, patch_tokens = self.backbone(full_image)
        zone_embeddings = self.pool(patch_tokens=patch_tokens, zone_masks=zone_masks, grid_size=self.grid_size)
        cls_per_zone = cls_token.unsqueeze(1).expand(-1, NUM_TARGET_ZONES, -1)
        fused = torch.cat([zone_embeddings, cls_per_zone], dim=-1)
        return self.head(fused).squeeze(-1)


def compute_pos_weights(labels: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    pos_weights = torch.ones(NUM_TARGET_ZONES, dtype=torch.float32)
    for zone_idx in range(NUM_TARGET_ZONES):
        mask = observed_mask[:, zone_idx]
        if not mask.any():
            continue
        zone_labels = labels[mask, zone_idx]
        positives = float((zone_labels == 1).sum().item())
        negatives = float((zone_labels == 0).sum().item())
        if positives > 0.0 and negatives > 0.0:
            pos_weights[zone_idx] = negatives / positives
    return pos_weights


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    observed_mask: torch.Tensor,
    pos_weights: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    labels = labels.float()
    total_loss = logits.sum() * 0.0
    total_observed = 0
    for zone_idx in range(logits.shape[1]):
        mask = observed_mask[:, zone_idx]
        count = int(mask.sum().item())
        if count == 0:
            continue
        kwargs = {}
        if pos_weights is not None:
            kwargs["pos_weight"] = pos_weights[zone_idx].view(1)
        total_loss = total_loss + F.binary_cross_entropy_with_logits(
            logits[mask, zone_idx],
            labels[mask, zone_idx],
            reduction="sum",
            **kwargs,
        )
        total_observed += count
    if total_observed == 0:
        return total_loss, 0
    return total_loss / total_observed, total_observed


def normalize_thresholds(threshold: float | list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(threshold, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(NUM_TARGET_ZONES, float(values))
    if values.shape != (NUM_TARGET_ZONES,):
        raise ValueError(f"Expected one threshold per zone, got shape {values.shape}.")
    return values


def tune_zone_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    observed_mask: np.ndarray,
    fallback: float = 0.5,
) -> np.ndarray:
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    thresholds = np.full(NUM_TARGET_ZONES, fallback, dtype=np.float64)
    for zone_idx in range(NUM_TARGET_ZONES):
        mask = observed_mask[:, zone_idx].astype(bool)
        if not mask.any():
            continue
        scores = [
            f1_score(y_true[mask, zone_idx], y_prob[mask, zone_idx] >= candidate, zero_division=0)
            for candidate in candidates
        ]
        # Resolve ties toward 0.5 to avoid unnecessarily extreme thresholds.
        best_score = max(scores)
        tied = candidates[np.isclose(scores, best_score)]
        thresholds[zone_idx] = tied[np.argmin(np.abs(tied - 0.5))]
    return thresholds


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    observed_mask: np.ndarray,
    threshold: float | list[float] | np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    thresholds = normalize_thresholds(threshold)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int64)
    rows = []
    flat_true = []
    flat_pred = []
    for zone_idx in range(NUM_TARGET_ZONES):
        mask = observed_mask[:, zone_idx].astype(bool)
        if not mask.any():
            rows.append({"Zone": zone_idx + 1, "ObservedCount": 0})
            continue
        z_true = y_true[mask, zone_idx]
        z_pred = y_pred[mask, zone_idx]
        z_prob = y_prob[mask, zone_idx]
        flat_true.append(z_true)
        flat_pred.append(z_pred)
        cm = confusion_matrix(z_true, z_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        roc_auc = None
        average_precision = None
        if len(np.unique(z_true)) == 2:
            roc_auc = float(roc_auc_score(z_true, z_prob))
        if np.any(z_true == 1):
            average_precision = float(average_precision_score(z_true, z_prob))
        rows.append(
            {
                "Zone": zone_idx + 1,
                "Threshold": float(thresholds[zone_idx]),
                "ObservedCount": int(mask.sum()),
                "PositiveRate": float(np.mean(z_true == 1)),
                "Accuracy": float(np.mean(z_true == z_pred)),
                "BinaryF1": float(f1_score(z_true, z_pred, zero_division=0)),
                "Precision": float(precision_score(z_true, z_pred, zero_division=0)),
                "Recall": float(recall_score(z_true, z_pred, zero_division=0)),
                "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
                "AveragePrecision": average_precision,
                "RocAuc": roc_auc,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    zone_df = pd.DataFrame(rows)
    if flat_true:
        flat_true_arr = np.concatenate(flat_true)
        flat_pred_arr = np.concatenate(flat_pred)
        micro_f1 = float(f1_score(flat_true_arr, flat_pred_arr, average="micro", zero_division=0))
        macro_f1 = float(f1_score(flat_true_arr, flat_pred_arr, average="macro", zero_division=0))
    else:
        micro_f1 = 0.0
        macro_f1 = 0.0

    ap_by_zone = zone_df.set_index("Zone")["AveragePrecision"] if "AveragePrecision" in zone_df else pd.Series(dtype=float)
    auc_by_zone = zone_df.set_index("Zone")["RocAuc"] if "RocAuc" in zone_df else pd.Series(dtype=float)
    f1_by_zone = zone_df.set_index("Zone")["BinaryF1"] if "BinaryF1" in zone_df else pd.Series(dtype=float)
    mean_ap_1_8 = float(ap_by_zone.reindex(range(1, 9)).dropna().mean()) if not ap_by_zone.empty else 0.0
    zone9_ap = float(ap_by_zone.get(9)) if pd.notna(ap_by_zone.get(9, np.nan)) else None
    zone9_f1 = float(f1_by_zone.get(9)) if pd.notna(f1_by_zone.get(9, np.nan)) else None
    mean_ap = float(ap_by_zone.dropna().mean()) if not ap_by_zone.dropna().empty else 0.0
    mean_roc_auc = float(auc_by_zone.dropna().mean()) if not auc_by_zone.dropna().empty else 0.0
    checkpoint_score = 0.7 * mean_ap_1_8 + 0.3 * zone9_ap if zone9_ap is not None else mean_ap

    summary = {
        "mean_binary_f1": float(zone_df["BinaryF1"].dropna().mean()) if "BinaryF1" in zone_df else 0.0,
        "mean_average_precision": mean_ap,
        "mean_roc_auc": mean_roc_auc,
        "mean_ap_zones_1_8": mean_ap_1_8,
        "zone9_average_precision": zone9_ap,
        "zone9_f1": zone9_f1,
        "checkpoint_score": float(checkpoint_score),
        "mean_accuracy": float(zone_df["Accuracy"].dropna().mean()) if "Accuracy" in zone_df else 0.0,
        "micro_f1_flat": micro_f1,
        "macro_f1_flat": macro_f1,
        "observed_zone_labels": int(observed_mask.sum()),
        "zone_thresholds": thresholds.tolist(),
    }
    return zone_df, summary


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weights: torch.Tensor | None,
    threshold: float | list[float] | np.ndarray,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_observed = 0
    all_labels = []
    all_probs = []
    all_masks = []
    all_zone_nonempty = []
    all_meta = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            batch = move_batch(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
                logits = model(batch["full_image"], batch["zone_masks"])
                loss, observed = masked_bce_loss(logits, batch["labels"], batch["observed_mask"], pos_weights)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += float(loss.item()) * max(observed, 1)
            total_observed += observed
            all_labels.append(batch["labels"].detach().cpu().numpy())
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_masks.append(batch["observed_mask"].detach().cpu().numpy())
            all_zone_nonempty.append(batch["zone_nonempty"].detach().cpu().numpy())
            all_meta.extend(
                zip(
                    batch["image_file"],
                    batch["patient_id"],
                    batch["eye"],
                    batch["visit_date"],
                    batch["image_path"],
                    batch["mask_path"],
                    strict=False,
                )
            )

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    observed_mask = np.concatenate(all_masks, axis=0).astype(bool)
    zone_nonempty = np.concatenate(all_zone_nonempty, axis=0).astype(bool)
    zone_df, summary = compute_metrics(y_true, y_prob, observed_mask, threshold=threshold)
    summary["loss"] = total_loss / max(total_observed, 1)
    summary["empty_mask_zone_labels"] = int((~zone_nonempty).sum())
    return {
        "summary": summary,
        "zone_metrics": zone_df,
        "y_true": y_true,
        "y_prob": y_prob,
        "observed_mask": observed_mask,
        "zone_nonempty": zone_nonempty,
        "metadata": all_meta,
    }


def save_predictions(result: dict[str, Any], path: str, threshold: float | list[float] | np.ndarray) -> None:
    y_true = result["y_true"]
    y_prob = result["y_prob"]
    thresholds = normalize_thresholds(threshold)
    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(np.int64)
    observed_mask = result["observed_mask"]
    zone_nonempty = result["zone_nonempty"]
    rows = []
    for idx, meta in enumerate(result["metadata"]):
        image_file, patient_id, eye, visit_date, image_path, mask_path = meta
        row = {
            "image_file": image_file,
            "patient_id": patient_id,
            "eye": eye,
            "visit_date": visit_date,
            "image_path": image_path,
            "mask_path": mask_path,
        }
        for zone_idx in range(NUM_TARGET_ZONES):
            zone = zone_idx + 1
            row[f"Zone{zone}_observed"] = bool(observed_mask[idx, zone_idx])
            row[f"Zone{zone}_valid"] = bool(observed_mask[idx, zone_idx])
            row[f"Zone{zone}_empty_mask"] = bool(not zone_nonempty[idx, zone_idx])
            row[f"Zone{zone}_true"] = int(y_true[idx, zone_idx]) if observed_mask[idx, zone_idx] else np.nan
            row[f"Zone{zone}_prob"] = float(y_prob[idx, zone_idx])
            row[f"Zone{zone}_pred"] = int(y_pred[idx, zone_idx]) if observed_mask[idx, zone_idx] else np.nan
            row[f"Zone{zone}_threshold"] = float(thresholds[zone_idx])
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def split_csv_path(csvpath: str, split_name: str) -> str:
    path = os.path.join(csvpath, f"{split_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split CSV: {path}")
    return path


def build_optimizer(model: FAZoneDinoClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
    head_params = list(model.pool.parameters()) + list(model.head.parameters())
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    groups = [
        {"params": head_params, "lr": args.head_lr},
        {"params": backbone_params, "lr": args.lr},
    ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)


def build_step_scheduler(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = min(args.warmup_epochs * steps_per_epoch, total_steps)
    if args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("Require 0 <= --min_lr <= --lr and --lr > 0.")
    min_scale = args.min_lr / args.lr

    def lr_scale(step_idx: int) -> float:
        # LambdaLR evaluates step 0 at construction, before the first optimizer update.
        completed_step = step_idx + 1
        if warmup_steps > 0 and completed_step <= warmup_steps:
            return completed_step / warmup_steps
        decay_steps = max(total_steps - warmup_steps, 1)
        progress = min(max((completed_step - warmup_steps) / decay_steps, 0.0), 1.0)
        return min_scale + 0.5 * (1.0 - min_scale) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)


def main() -> None:
    args = parse_args()
    validate_augmentation_args(args)
    set_seed(args.seed, deterministic=args.deterministic)
    os.makedirs(args.output_path, exist_ok=True)
    init_wandb(args)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.image_size % PATCH_SIZE != 0:
        raise ValueError(f"--image_size must be divisible by {PATCH_SIZE}; got {args.image_size}.")

    train_split = read_split(split_csv_path(args.csvpath, "train"), args)
    val_split = read_split(split_csv_path(args.csvpath, "val"), args)
    test_split = read_split(split_csv_path(args.csvpath, "test"), args)
    for split_name, split in (("train", train_split), ("val", val_split), ("test", test_split)):
        pd.DataFrame(
            split.empty_mask_records,
            columns=["image_path", "zone", "mask_path", "empty_mask_count"],
        ).to_csv(os.path.join(args.output_path, f"{split_name}_empty_masks.csv"), index=False)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        FAZoneDataset(train_split, args, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator,
    )
    val_loader = DataLoader(
        FAZoneDataset(val_split, args, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        FAZoneDataset(test_split, args, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FAZoneDinoClassifier(
        arch=args.dinov2_arch,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    pos_weights = None if args.unweighted else compute_pos_weights(train_split.labels, train_split.observed_mask).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_step_scheduler(optimizer, len(train_loader), args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scaler = scaler if scaler.is_enabled() else None

    metadata = {
        "args": vars(args),
        "num_train": len(train_split.image_paths),
        "num_val": len(val_split.image_paths),
        "num_test": len(test_split.image_paths),
        "target_zones": list(range(1, NUM_TARGET_ZONES + 1)),
        "zone10_removed_from_full_fa": True,
        "aspect_ratio_preserved_with_padding": True,
        "patch_grid": [args.image_size // PATCH_SIZE, args.image_size // PATCH_SIZE],
        "pos_weights": None if pos_weights is None else pos_weights.detach().cpu().tolist(),
        "empty_mask_counts": {
            "train": len(train_split.empty_mask_records),
            "val": len(val_split.empty_mask_records),
            "test": len(test_split.empty_mask_records),
        },
    }
    with open(os.path.join(args.output_path, "train_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "num_train": metadata["num_train"],
                "num_val": metadata["num_val"],
                "num_test": metadata["num_test"],
                "pos_weights": metadata["pos_weights"],
            },
            allow_val_change=True,
        )

    best_val_score = -math.inf
    best_val_f1 = -math.inf
    best_epoch = 0
    bad_epochs = 0
    history = []
    checkpoint_path = os.path.join(args.output_path, "checkpoint.pt")

    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            pos_weights=pos_weights,
            threshold=args.threshold,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
        )
        val_result = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            pos_weights=pos_weights,
            threshold=args.threshold,
        )
        train_summary = train_result["summary"]
        zone_thresholds = tune_zone_thresholds(
            val_result["y_true"],
            val_result["y_prob"],
            val_result["observed_mask"],
            fallback=args.threshold,
        )
        val_zone_metrics, val_summary = compute_metrics(
            val_result["y_true"], val_result["y_prob"], val_result["observed_mask"], zone_thresholds
        )
        val_summary["loss"] = val_result["summary"]["loss"]
        val_summary["empty_mask_zone_labels"] = val_result["summary"]["empty_mask_zone_labels"]
        val_result["zone_metrics"] = val_zone_metrics
        val_result["summary"] = val_summary
        row = {
            "epoch": epoch,
            "train_loss": train_summary["loss"],
            "train_mean_f1": train_summary["mean_binary_f1"],
            "val_loss": val_summary["loss"],
            "val_mean_f1": val_summary["mean_binary_f1"],
            "val_mean_average_precision": val_summary["mean_average_precision"],
            "val_mean_roc_auc": val_summary["mean_roc_auc"],
            "val_mean_ap_zones_1_8": val_summary["mean_ap_zones_1_8"],
            "val_zone9_average_precision": val_summary["zone9_average_precision"],
            "val_zone9_f1": val_summary["zone9_f1"],
            "val_checkpoint_score": val_summary["checkpoint_score"],
            "lr_backbone": optimizer.param_groups[1]["lr"],
            "lr_head": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_mean_f1']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_f1={row['val_mean_f1']:.4f} val_mAP={row['val_mean_average_precision']:.4f} "
            f"score={row['val_checkpoint_score']:.4f}"
        )

        if val_summary["checkpoint_score"] > best_val_score:
            best_val_score = val_summary["checkpoint_score"]
            best_val_f1 = val_summary["mean_binary_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_mean_f1": best_val_f1,
                    "best_val_checkpoint_score": best_val_score,
                    "checkpoint_objective": "0.7 * mean_ap_zones_1_8 + 0.3 * zone9_average_precision",
                    "zone_thresholds": zone_thresholds.tolist(),
                },
                checkpoint_path,
            )
            val_result["zone_metrics"].to_csv(os.path.join(args.output_path, "val_zone_metrics.csv"), index=False)
            save_predictions(val_result, os.path.join(args.output_path, "val_predictions.csv"), threshold=zone_thresholds)
            with open(os.path.join(args.output_path, "val_summary.json"), "w") as f:
                json.dump(val_summary, f, indent=2)
        else:
            bad_epochs += 1

        epoch_metrics = {
            "train/loss": train_summary["loss"],
            "train/mean_binary_f1": train_summary["mean_binary_f1"],
            "train/mean_average_precision": train_summary["mean_average_precision"],
            "train/mean_roc_auc": train_summary["mean_roc_auc"],
            "val/loss": val_summary["loss"],
            "val/mean_binary_f1": val_summary["mean_binary_f1"],
            "val/mean_average_precision": val_summary["mean_average_precision"],
            "val/mean_roc_auc": val_summary["mean_roc_auc"],
            "val/mean_ap_zones_1_8": val_summary["mean_ap_zones_1_8"],
            "val/zone9_average_precision": val_summary["zone9_average_precision"],
            "val/zone9_f1": val_summary["zone9_f1"],
            "val/checkpoint_score": val_summary["checkpoint_score"],
            "val/best_checkpoint_score": best_val_score,
            "val/best_epoch": best_epoch,
            "early_stopping/bad_epochs": bad_epochs,
            "learning_rate/backbone": optimizer.param_groups[1]["lr"],
            "learning_rate/head": optimizer.param_groups[0]["lr"],
        }
        for zone_idx, zone_row in val_zone_metrics.iterrows():
            zone = int(zone_row["Zone"])
            epoch_metrics[f"val_zone/{zone}/f1"] = float(zone_row["BinaryF1"])
            epoch_metrics[f"val_zone/{zone}/average_precision"] = float(zone_row["AveragePrecision"])
            epoch_metrics[f"val_zone/{zone}/roc_auc"] = float(zone_row["RocAuc"])
            epoch_metrics[f"val_zone/{zone}/threshold"] = float(zone_thresholds[zone_idx])
        wandb_log(epoch_metrics, step=epoch)

        pd.DataFrame(history).to_csv(os.path.join(args.output_path, "history.csv"), index=False)
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without validation checkpoint-score improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    zone_thresholds = np.asarray(checkpoint["zone_thresholds"], dtype=np.float64)
    test_result = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        pos_weights=pos_weights,
        threshold=zone_thresholds,
    )
    test_result["zone_metrics"].to_csv(os.path.join(args.output_path, "test_zone_metrics.csv"), index=False)
    save_predictions(test_result, os.path.join(args.output_path, "test_predictions.csv"), threshold=zone_thresholds)
    test_summary = {
        **test_result["summary"],
        "best_epoch": best_epoch,
        "best_val_mean_f1": best_val_f1,
        "best_val_checkpoint_score": best_val_score,
    }
    with open(os.path.join(args.output_path, "test_summary.json"), "w") as f:
        json.dump(test_summary, f, indent=2)
    final_metrics = {
        "test/loss": test_summary["loss"],
        "test/mean_binary_f1": test_summary["mean_binary_f1"],
        "test/mean_average_precision": test_summary["mean_average_precision"],
        "test/mean_roc_auc": test_summary["mean_roc_auc"],
        "test/mean_accuracy": test_summary["mean_accuracy"],
        "test/macro_f1_flat": test_summary["macro_f1_flat"],
        "test/zone9_average_precision": test_summary["zone9_average_precision"],
        "test/zone9_f1": test_summary["zone9_f1"],
        "test/checkpoint_score": test_summary["checkpoint_score"],
        "test/best_epoch": best_epoch,
        "test/best_val_checkpoint_score": best_val_score,
    }
    for _, zone_row in test_result["zone_metrics"].iterrows():
        zone = int(zone_row["Zone"])
        final_metrics[f"test_zone/{zone}/f1"] = float(zone_row["BinaryF1"])
        final_metrics[f"test_zone/{zone}/average_precision"] = float(zone_row["AveragePrecision"])
        final_metrics[f"test_zone/{zone}/roc_auc"] = float(zone_row["RocAuc"])
    wandb_log(final_metrics)
    if wandb_run is not None:
        wandb_run.summary.update(final_metrics)
    print(
        f"Done. best_epoch={best_epoch} best_val_checkpoint_score={best_val_score:.4f} "
        f"best_val_mean_f1={best_val_f1:.4f}"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        finish_wandb()
