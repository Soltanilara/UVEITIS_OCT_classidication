import argparse
import json
import os
import random
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

try:
    from pytorch_pretrained_vit import ViT
except ImportError:
    ViT = None

matplotlib.rc("font", family="serif", size=14)
matplotlib.rc("axes", titlesize=14)
matplotlib.rc("axes", labelsize=14)
matplotlib.rc("legend", fontsize=12)
matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)


NUM_ZONES = 10
NUM_CLASSES = 2
ORIGINAL_NUM_CLASSES = 3
ZONE_COLUMNS = [f"Zone{i}_label" for i in range(1, NUM_ZONES + 1)]
FALLBACK_EXTS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train")
parser.add_argument("--final", action="store_true", help="Final training on train+val")
parser.add_argument("--checkpoint", type=str, default="checkpoint.pt")
parser.add_argument("--ckpt_interval", type=int, default=100)
parser.add_argument("--mixup", action="store_true", help="Enable input mixup")
parser.add_argument("--mixup_alpha", type=float, default=1.0)
parser.add_argument("--mixup_beta", type=float, default=1.0)
parser.add_argument("--weightedSampling", action="store_true", help="Enable WeightedRandomSampler")
parser.add_argument("--hflip", action="store_true", help="Apply random horizontal flip augmentation")
parser.add_argument("--elastic", action="store_true", help="Apply affine augmentation")
parser.add_argument("--brightness", action="store_true")
parser.add_argument("--contrast", action="store_true")
parser.add_argument("--gnoise", action="store_true")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--image_size", type=int, default=512, help="Resize for full-image inputs.")
parser.add_argument("--zone_crop_size", type=int, default=224, help="Resize for zone crops.")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--decay", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--warmup_epochs", type=int, default=10)
parser.add_argument("--T_0", type=int, default=10)
parser.add_argument("--T_multi", type=int, default=1)
parser.add_argument("--eta_min", type=float, default=1e-5)
parser.add_argument("--earlystop", action="store_true")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--loss", type=str, default="CE", choices=["CE", "focal"])
parser.add_argument("--unweighted", action="store_true")
parser.add_argument("--gamma", type=float, default=2.0)
parser.add_argument("--label_smoothing", type=float, default=0.0)
parser.add_argument("--protocol", type=str, default="finetune", choices=["finetune", "lin_eval", "scratch", "complement"])
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--pretraining", type=str, default="swav", choices=["swav", "barlowtwins", "supervised", "vit", "dino", "oct"])
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--csvpath", type=str, default="fold_0")
parser.add_argument("--metadata_path", type=str, default="metadata_0")
parser.add_argument("--dataset_path", type=str, default="Dataset 01032025")
parser.add_argument("--output_path", type=str, default="output_fold_0")
parser.add_argument("--image_column", type=str, default="Image File")
parser.add_argument("--image_absolute_column", type=str, default="")
parser.add_argument("--mask_column", type=str, default="")
parser.add_argument("--mask_absolute_column", type=str, default="")
parser.add_argument(
    "--image_resolver",
    type=str,
    default="direct",
    choices=["direct", "fundus_from_fa_pair"],
    help="How to resolve rows in --image_column to a local file.",
)
parser.add_argument("--apply_mask", action="store_true", help="Apply a binary version of each mask to its image before transforms.")
parser.add_argument("--drop_missing_zone_rows", type=str, default="all", choices=["none", "any", "all"])
parser.add_argument(
    "--input_mode",
    type=str,
    default="full_image_zone_head",
    choices=["full_image_zone_head", "zone_crops_shared", "hybrid_global_plus_zone"],
)
parser.add_argument("--zone_template_json", type=str, default="configs/fundus_10zone_template.json")
parser.add_argument("--thresholds_json", type=str, default="")
parser.add_argument("--fundus_pretrained_ckpt", type=str, default="")
parser.add_argument("--hybrid_freeze_crop_epochs", type=int, default=5)
parser.add_argument("--swa", action="store_true")
parser.add_argument("--swa_start_epoch", type=int, default=70)
parser.add_argument("--visit_bootstrap_samples", type=int, default=1000)
parser.add_argument("--gradcam", action="store_true")
parser.add_argument("--IG", action="store_true")
parser.add_argument("--shap", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()


if args.gradcam or args.IG or args.shap:
    print("Warning: --gradcam/--IG/--shap are disabled in this binary multi-zone script.")

if args.hflip or args.elastic:
    print(
        "Warning: geometry-changing augmentation is enabled. This is usually not appropriate for canonical zone-"
        "supervised fundus experiments unless zone labels are remapped."
    )

if args.protocol == "complement" and args.input_mode != "full_image_zone_head":
    raise ValueError("protocol='complement' only supports --input_mode full_image_zone_head.")

if args.mixup and args.input_mode != "full_image_zone_head":
    print("Warning: mixup is enabled for a zone-aware architecture. The recommended plan is to keep mixup off.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(args.seed)


def build_folder_name() -> str:
    bits = [
        "final" if args.final else None,
        args.input_mode,
        args.protocol,
        args.model,
        f"pretraining_{args.pretraining}" if args.protocol != "scratch" else None,
        args.loss.lower(),
        f"gamma_{args.gamma:g}" if args.loss == "focal" else None,
        "unweighted" if args.unweighted else "class_weighted",
        "weightedSampling" if args.weightedSampling else None,
        f"mixup_{args.mixup_alpha:g}_{args.mixup_beta:g}" if args.mixup else None,
        f"mask_{args.drop_missing_zone_rows}",
        f"img_{args.image_size}",
        f"crop_{args.zone_crop_size}" if args.input_mode != "full_image_zone_head" else None,
        f"label_smoothing_{args.label_smoothing:.2f}" if args.label_smoothing > 0 else None,
        "swa" if args.swa else None,
        f"batch_{args.batch_size}",
        f"lr_{args.lr}",
        f"decay_{args.decay}",
        f"warmup_{args.warmup_epochs}",
        f"T0_{args.T_0}",
        f"Tmult_{args.T_multi}",
        f"eta_{args.eta_min}",
        f"epochs_{args.num_epochs}",
        "hflip" if args.hflip else None,
        "elastic" if args.elastic else None,
        "brightness" if args.brightness else None,
        "contrast" if args.contrast else None,
        "gnoise" if args.gnoise else None,
        f"beta_{args.beta}" if args.protocol == "complement" else None,
        f"zones{NUM_ZONES}x{NUM_CLASSES}",
        f"seed_{args.seed}",
    ]
    return "_".join([bit for bit in bits if bit])


folder_name = build_folder_name()
full_output_path = os.path.join(args.output_path, folder_name)
os.makedirs(full_output_path, exist_ok=True)

if args.final:
    full_metadata_path = os.path.join(args.output_path, args.metadata_path)
    print(f"Metadata loaded from {full_metadata_path}.")


def _validate_zone_values(zone_df: pd.DataFrame, csv_file: str) -> pd.DataFrame:
    non_missing = zone_df.notna()
    rounded = zone_df.round()

    if non_missing.any().any():
        src_values = zone_df.where(non_missing).to_numpy(dtype=float)
        rounded_values = rounded.where(non_missing).to_numpy(dtype=float)
        if not np.allclose(src_values[~np.isnan(src_values)], rounded_values[~np.isnan(rounded_values)]):
            raise ValueError(f"Non-integer zone class value found in {csv_file}; expected integer classes in {{0,1,2}}.")

    as_int = rounded.astype("Int64")
    bad_mask = ((as_int < 0) | (as_int >= ORIGINAL_NUM_CLASSES)).fillna(False)
    if bad_mask.values.any():
        r, c = np.argwhere(bad_mask.to_numpy())[0]
        raise ValueError(
            f"Invalid zone class in {csv_file} at row index {r}, column {ZONE_COLUMNS[c]}: "
            f"value={as_int.iat[r, c]} (expected 0..{ORIGINAL_NUM_CLASSES - 1})"
        )

    return as_int


def resolve_image_path(base_folder: str, rel_path: str) -> str:
    candidate = os.path.join(base_folder, rel_path)
    if os.path.exists(candidate):
        return candidate

    root, _ = os.path.splitext(candidate)
    tried = [candidate]

    for fallback_ext in FALLBACK_EXTS:
        alt = root + fallback_ext
        tried.append(alt)
        if os.path.exists(alt):
            return alt

        alt_upper = root + fallback_ext.upper()
        tried.append(alt_upper)
        if os.path.exists(alt_upper):
            return alt_upper

    parent_dir = os.path.dirname(candidate)
    stem = os.path.splitext(os.path.basename(candidate))[0].lower()
    if os.path.isdir(parent_dir):
        for fname in os.listdir(parent_dir):
            f_stem, f_ext = os.path.splitext(fname)
            if f_stem.lower() == stem and f_ext.lower() in FALLBACK_EXTS:
                return os.path.join(parent_dir, fname)

    raise FileNotFoundError(
        "Image not found. Tried path and extension fallbacks.\n"
        f"Requested: {candidate}\n"
        f"Tried: {tried[:8]}{' ...' if len(tried) > 8 else ''}"
    )


def resolve_optional_path(base_folder: str, path_value: Any) -> str | None:
    if pd.isna(path_value):
        return None
    path_str = str(path_value).strip()
    if not path_str:
        return None
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    return resolve_image_path(base_folder, path_str)


def resolve_fundus_path_from_fa_pair(base_folder: str, fa_rel_path: str, uwffp_value: str) -> str:
    fa_parent = os.path.dirname(str(fa_rel_path).replace("\\", "/"))
    fa_name = os.path.basename(str(fa_rel_path).replace("\\", "/"))
    folder = os.path.join(base_folder, fa_parent)
    fundus_name = os.path.basename(str(uwffp_value).replace("\\", "/"))

    candidate = os.path.join(folder, fundus_name)
    if os.path.exists(candidate):
        return candidate

    expected_name = fa_name.replace("_FA_", "_FP_").replace("_0001.", "_0000.")
    stem, _ = os.path.splitext(expected_name)
    tried = [candidate]
    for fallback_ext in FALLBACK_EXTS:
        for ext_candidate in (stem + fallback_ext, stem + fallback_ext.upper()):
            candidate = os.path.join(folder, ext_candidate)
            tried.append(candidate)
            if os.path.exists(candidate):
                return candidate

    eye_token = "_OD_" if "_OD_" in fa_name else "_OS_" if "_OS_" in fa_name else ""
    if os.path.isdir(folder):
        eye_candidates = sorted(
            fname for fname in os.listdir(folder) if (not eye_token or eye_token in fname) and "FP" in fname.upper()
        )
        if len(eye_candidates) == 1:
            return os.path.join(folder, eye_candidates[0])
        for eye_candidate in eye_candidates:
            if "_FP_" in eye_candidate:
                return os.path.join(folder, eye_candidate)

    raise FileNotFoundError(
        "Fundus image not found from FA/fundus pair resolution.\n"
        f"FA relative path: {fa_rel_path}\n"
        f"UWFFP value: {uwffp_value}\n"
        f"Tried: {tried[:8]}{' ...' if len(tried) > 8 else ''}"
    )


def _drop_rows_by_missing_policy(df: pd.DataFrame, zone_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.drop_missing_zone_rows == "none":
        keep_mask = np.ones(len(df), dtype=bool)
    elif args.drop_missing_zone_rows == "any":
        keep_mask = ~zone_df.isna().any(axis=1).to_numpy()
    elif args.drop_missing_zone_rows == "all":
        keep_mask = ~zone_df.isna().all(axis=1).to_numpy()
    else:
        raise ValueError(f"Unsupported drop policy: {args.drop_missing_zone_rows}")

    kept_df = df.loc[keep_mask].reset_index(drop=True)
    kept_zone_df = zone_df.loc[keep_mask].reset_index(drop=True)
    return kept_df, kept_zone_df


def build_visit_ids(df: pd.DataFrame) -> list[str]:
    image_id_series = df[args.image_column].astype(str) if args.image_column in df.columns else df.iloc[:, 0].astype(str)
    if {"Patient_ID", "Eye", "Visit_Date"}.issubset(df.columns):
        visit_series = (
            df["Patient_ID"].astype(str)
            + "|"
            + df["Eye"].astype(str)
            + "|"
            + df["Visit_Date"].astype(str)
            + "|"
            + image_id_series
        )
        return visit_series.tolist()
    return image_id_series.tolist()


def load_data(csv_file: str, csvpath: str, folder: str):
    csv_full_path = os.path.join(csvpath, csv_file)
    df = pd.read_csv(csv_full_path)

    required_cols = [args.image_column, *ZONE_COLUMNS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_full_path} is missing required columns: {missing}")

    zone_df = df[ZONE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    zone_df = _validate_zone_values(zone_df, csv_file)
    df, zone_df = _drop_rows_by_missing_policy(df, zone_df)

    observed_mask_df = zone_df.notna()
    binary_df = zone_df.copy()
    for col in ZONE_COLUMNS:
        binary_df[col] = binary_df[col].map(lambda x: pd.NA if pd.isna(x) else int(int(x) != 0))

    labels_df = binary_df.fillna(-1).astype(int)
    paths = []
    mask_paths = []
    for _, row in df.iterrows():
        if args.image_absolute_column and args.image_absolute_column in df.columns and not pd.isna(row[args.image_absolute_column]):
            image_path = resolve_optional_path(folder, row[args.image_absolute_column])
        elif args.image_resolver == "fundus_from_fa_pair":
            if "Image_File(FA)" not in df.columns or "UWFFP" not in df.columns:
                raise ValueError("--image_resolver fundus_from_fa_pair requires Image_File(FA) and UWFFP columns.")
            image_path = resolve_fundus_path_from_fa_pair(folder, row["Image_File(FA)"], row["UWFFP"])
        else:
            image_path = resolve_image_path(folder, str(row[args.image_column]))
        paths.append(image_path)

        mask_path = None
        if args.mask_absolute_column and args.mask_absolute_column in df.columns:
            mask_path = resolve_optional_path(folder, row[args.mask_absolute_column])
        elif args.mask_column and args.mask_column in df.columns:
            mask_path = resolve_optional_path(folder, row[args.mask_column])
        mask_paths.append(mask_path)

    metadata = {
        "image_files": df[args.image_column].astype(str).tolist(),
        "visit_ids": build_visit_ids(df),
        "patient_ids": df["Patient_ID"].astype(str).tolist() if "Patient_ID" in df.columns else df[args.image_column].astype(str).tolist(),
        "eyes": df["Eye"].astype(str).tolist() if "Eye" in df.columns else [""] * len(df),
        "visit_dates": df["Visit_Date"].astype(str).tolist() if "Visit_Date" in df.columns else [""] * len(df),
        "mask_files": mask_paths,
    }

    labels = torch.tensor(labels_df.to_numpy(dtype=np.int64), dtype=torch.long)
    observed_mask = torch.tensor(observed_mask_df.to_numpy(dtype=bool), dtype=torch.bool)

    drop_counts = {
        "all_missing_rows": int(zone_df.isna().all(axis=1).sum()),
        "partially_missing_rows": int(zone_df.isna().any(axis=1).sum() - zone_df.isna().all(axis=1).sum()),
    }
    return paths, labels, observed_mask, metadata, drop_counts


def model_uses_half_stats() -> bool:
    return args.model in {"B_16_imagenet1k", "L_16_imagenet1k", "vitb14_dino"} or args.protocol == "scratch"


def full_image_resize() -> tuple[int, int]:
    if args.model in {"B_16_imagenet1k", "L_16_imagenet1k"}:
        return (384, 384)
    if args.model == "vitb14_dino":
        return (518, 518)
    return (args.image_size, args.image_size)


def crop_resize() -> tuple[int, int]:
    return (args.zone_crop_size, args.zone_crop_size)


def build_transform(train: bool, target_size: tuple[int, int]):
    mean = [0.5, 0.5, 0.5] if model_uses_half_stats() else [0.485, 0.456, 0.406]
    std = [0.5, 0.5, 0.5] if model_uses_half_stats() else [0.229, 0.224, 0.225]

    transform_list = [transforms.Resize(target_size)]

    if train and args.hflip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if train and args.elastic:
        transform_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10))

    if train and (args.brightness or args.contrast):
        brightness = 0.2 if args.brightness else 0.0
        contrast = 0.2 if args.contrast else 0.0
        transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    transform_list.append(transforms.ToTensor())

    if train and args.gnoise:
        transform_list.append(transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)))

    transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)


def build_transforms():
    train_full = build_transform(train=True, target_size=full_image_resize())
    val_full = build_transform(train=False, target_size=full_image_resize())
    train_crop = build_transform(train=True, target_size=crop_resize())
    val_crop = build_transform(train=False, target_size=crop_resize())
    return train_full, val_full, train_crop, val_crop


def _parse_box_entry(entry: dict[str, Any]) -> tuple[float, float, float, float]:
    if "bbox" in entry:
        bbox = entry["bbox"]
        if len(bbox) != 4:
            raise ValueError("Zone template bbox entries must have four numbers.")
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    if {"x0", "y0", "x1", "y1"}.issubset(entry):
        return float(entry["x0"]), float(entry["y0"]), float(entry["x1"]), float(entry["y1"])

    if {"x", "y", "width", "height"}.issubset(entry):
        x0 = float(entry["x"])
        y0 = float(entry["y"])
        x1 = x0 + float(entry["width"])
        y1 = y0 + float(entry["height"])
        return x0, y0, x1, y1

    raise ValueError("Each zone template entry needs bbox, x0/y0/x1/y1, or x/y/width/height.")


def load_zone_template(path: str) -> list[tuple[float, float, float, float]]:
    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        zones = payload.get("zones", [])
    else:
        zones = payload

    if len(zones) != NUM_ZONES:
        raise ValueError(f"Expected {NUM_ZONES} zones in template {path}, found {len(zones)}.")

    boxes = []
    for idx, zone_entry in enumerate(zones):
        x0, y0, x1, y1 = _parse_box_entry(zone_entry)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"Zone template entry {idx + 1} is outside normalized [0, 1] coordinates.")
        boxes.append((x0, y0, x1, y1))
    return boxes


def extract_zone_crops(image: Image.Image, zone_template: list[tuple[float, float, float, float]]) -> list[Image.Image]:
    width, height = image.size
    crops = []
    for x0, y0, x1, y1 in zone_template:
        left = int(round(x0 * width))
        top = int(round(y0 * height))
        right = int(round(x1 * width))
        bottom = int(round(y1 * height))
        crops.append(image.crop((left, top, right, bottom)))
    return crops


class CustomImageDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        csvpath: str,
        folder: str,
        input_mode: str,
        full_image_transform=None,
        crop_transform=None,
        zone_template: list[tuple[float, float, float, float]] | None = None,
    ):
        self.paths, self.labels, self.observed_mask, self.metadata, self.drop_counts = load_data(csv_file, csvpath, folder)
        self.input_mode = input_mode
        self.full_image_transform = full_image_transform
        self.crop_transform = crop_transform
        self.zone_template = zone_template or []

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        mask_path = self.metadata["mask_files"][idx]
        if args.apply_mask:
            if mask_path is None:
                raise ValueError(f"--apply_mask is enabled but no mask is available for index {idx}: {self.paths[idx]}")
            mask = np.load(mask_path)
            if mask.ndim != 2:
                raise ValueError(f"Expected 2D label map in {mask_path}, got shape {mask.shape}")
            if mask.shape != (image.height, image.width):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8), mode="L").resize(image.size, Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            image_arr = np.array(image, dtype=np.uint8)
            image = Image.fromarray(image_arr * (mask > 0)[..., None].astype(np.uint8), mode="RGB")
        sample = {
            "labels": self.labels[idx],
            "observed_mask": self.observed_mask[idx],
            "image_file": self.metadata["image_files"][idx],
            "mask_file": mask_path or "",
            "visit_id": self.metadata["visit_ids"][idx],
            "patient_id": self.metadata["patient_ids"][idx],
            "eye": self.metadata["eyes"][idx],
            "visit_date": self.metadata["visit_dates"][idx],
        }

        if self.input_mode in {"full_image_zone_head", "hybrid_global_plus_zone"}:
            if self.full_image_transform is None:
                raise ValueError("full_image_transform is required for full-image or hybrid input modes.")
            sample["full_image"] = self.full_image_transform(image.copy())

        if self.input_mode in {"zone_crops_shared", "hybrid_global_plus_zone"}:
            if not self.zone_template:
                raise ValueError("zone_template is required for crop-based input modes.")
            if self.crop_transform is None:
                raise ValueError("crop_transform is required for crop-based input modes.")
            crops = extract_zone_crops(image, self.zone_template)
            sample["zone_crops"] = torch.stack([self.crop_transform(crop) for crop in crops], dim=0)

        return sample


def compute_zone_class_weights(labels: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
    weights = torch.ones((NUM_ZONES, NUM_CLASSES), dtype=torch.float32)
    for z in range(NUM_ZONES):
        zone_mask = observed_mask[:, z]
        if not zone_mask.any():
            continue
        zone_labels = labels[zone_mask, z]
        counts = torch.bincount(zone_labels, minlength=NUM_CLASSES).float()
        valid = counts > 0
        if valid.any():
            inv = torch.zeros_like(counts)
            inv[valid] = counts.sum() / counts[valid]
            inv_mean = inv[valid].mean().item()
            if inv_mean > 0:
                inv = inv / inv_mean
            weights[z] = inv
    return weights


def compute_sample_weights(labels: torch.Tensor, observed_mask: torch.Tensor, zone_weights: torch.Tensor) -> list[float]:
    weights = []
    for i in range(labels.shape[0]):
        zone_indices = torch.where(observed_mask[i])[0]
        if len(zone_indices) == 0:
            weights.append(1.0)
            continue
        per_zone = [zone_weights[z, labels[i, z]].item() for z in zone_indices]
        weights.append(float(np.mean(per_zone)))
    return weights


class BackboneEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim > 2:
            feats = torch.flatten(feats, start_dim=1)
        return feats


class FullImageZoneClassifier(nn.Module):
    def __init__(self, encoder: BackboneEncoder, feature_dim: int):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleList([nn.Linear(feature_dim, NUM_CLASSES) for _ in range(NUM_ZONES)])

    def forward(self, full_image: torch.Tensor, zone_crops: torch.Tensor | None = None) -> torch.Tensor:
        global_feats = self.encoder(full_image)
        logits = [head(global_feats).unsqueeze(1) for head in self.heads]
        return torch.cat(logits, dim=1)


class ZoneCropSharedClassifier(nn.Module):
    def __init__(self, crop_encoder: BackboneEncoder, feature_dim: int):
        super().__init__()
        self.crop_encoder = crop_encoder
        self.heads = nn.ModuleList([nn.Linear(feature_dim, NUM_CLASSES) for _ in range(NUM_ZONES)])

    def forward(self, full_image: torch.Tensor | None = None, zone_crops: torch.Tensor | None = None) -> torch.Tensor:
        if zone_crops is None:
            raise ValueError("zone_crops_shared mode requires zone_crops input.")
        batch_size, num_zones, channels, height, width = zone_crops.shape
        flat_crops = zone_crops.reshape(batch_size * num_zones, channels, height, width)
        crop_feats = self.crop_encoder(flat_crops).reshape(batch_size, num_zones, -1)
        logits = [self.heads[z](crop_feats[:, z, :]).unsqueeze(1) for z in range(NUM_ZONES)]
        return torch.cat(logits, dim=1)


class HybridGlobalZoneClassifier(nn.Module):
    def __init__(self, global_encoder: BackboneEncoder, global_dim: int, crop_encoder: BackboneEncoder, crop_dim: int):
        super().__init__()
        self.global_encoder = global_encoder
        self.crop_encoder = crop_encoder
        self.heads = nn.ModuleList([nn.Linear(global_dim + crop_dim, NUM_CLASSES) for _ in range(NUM_ZONES)])
        self._crop_branch_frozen = False

    def set_crop_branch_frozen(self, frozen: bool) -> None:
        self._crop_branch_frozen = frozen
        for param in self.crop_encoder.parameters():
            param.requires_grad = not frozen

    def forward(self, full_image: torch.Tensor | None = None, zone_crops: torch.Tensor | None = None) -> torch.Tensor:
        if full_image is None or zone_crops is None:
            raise ValueError("hybrid_global_plus_zone mode requires both full_image and zone_crops inputs.")

        global_feats = self.global_encoder(full_image)
        batch_size, num_zones, channels, height, width = zone_crops.shape
        flat_crops = zone_crops.reshape(batch_size * num_zones, channels, height, width)
        crop_feats = self.crop_encoder(flat_crops).reshape(batch_size, num_zones, -1)

        logits = []
        for z in range(NUM_ZONES):
            fused = torch.cat([global_feats, crop_feats[:, z, :]], dim=1)
            logits.append(self.heads[z](fused).unsqueeze(1))
        return torch.cat(logits, dim=1)


class R50Complement(nn.Module):
    def __init__(self):
        super().__init__()
        if args.pretraining == "swav":
            self.frozen50 = torch.hub.load("facebookresearch/swav:main", "resnet50")
        elif args.pretraining == "barlowtwins":
            self.frozen50 = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
        else:
            raise ValueError("protocol='complement' currently supports pretraining in {'swav', 'barlowtwins'}")

        self.frozen50.fc = nn.Identity()
        for param in self.frozen50.parameters():
            param.requires_grad = False

        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(4096, NUM_ZONES * NUM_CLASSES)

    def forward(self, x):
        out1 = self.frozen50(x)
        out2 = self.encoder(x)
        combined = torch.cat((out1, out2), dim=1)
        out = self.classifier(combined)
        return out, out1, out2


def load_supervised_backbone(model_name: str, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        return models.resnet18(pretrained=pretrained)
    if model_name == "resnet50":
        return models.resnet50(pretrained=pretrained)
    if model_name == "vgg19_bn":
        return models.vgg19_bn(pretrained=pretrained)
    if model_name == "densenet121":
        return models.densenet121(pretrained=pretrained)
    if model_name == "densenet201":
        return models.densenet201(pretrained=pretrained)
    raise ValueError(f"Unsupported supervised backbone: {model_name}")


def strip_model_head(model: nn.Module) -> tuple[nn.Module, int]:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        return model, in_features

    if hasattr(model, "linear_head") and isinstance(model.linear_head, nn.Module):
        in_features = model.linear_head.in_features
        model.linear_head = nn.Identity()
        return model, in_features

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            in_features = classifier.in_features
            model.classifier = nn.Identity()
            return model, in_features
        if isinstance(classifier, nn.Sequential):
            layers = list(classifier)
            for idx in range(len(layers) - 1, -1, -1):
                if isinstance(layers[idx], nn.Linear):
                    in_features = layers[idx].in_features
                    layers[idx] = nn.Identity()
                    model.classifier = nn.Sequential(*layers)
                    return model, in_features

    raise ValueError(f"Could not strip output head for model type: {type(model)}")


def unwrap_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]
    if isinstance(checkpoint_obj, dict):
        return checkpoint_obj
    raise ValueError("Checkpoint does not contain a supported state_dict payload.")


def flexible_load_state_dict(module: nn.Module, ckpt_path: str, strip_prefixes: list[str] | None = None) -> int:
    checkpoint_obj = torch.load(ckpt_path, map_location="cpu")
    raw_state = unwrap_state_dict(checkpoint_obj)
    module_state = module.state_dict()
    strip_prefixes = strip_prefixes or []

    candidate_states = [raw_state]
    for prefix in strip_prefixes:
        candidate_states.append(
            {
                key[len(prefix):]: value
                for key, value in raw_state.items()
                if key.startswith(prefix)
            }
        )

    best_candidate = {}
    best_count = -1
    for candidate in candidate_states:
        matched = {
            key: value
            for key, value in candidate.items()
            if key in module_state and module_state[key].shape == value.shape
        }
        if len(matched) > best_count:
            best_candidate = matched
            best_count = len(matched)

    if best_count <= 0:
        print(f"Warning: no matching keys found while loading {ckpt_path} into {module.__class__.__name__}.")
        return 0

    module.load_state_dict(best_candidate, strict=False)
    print(f"Loaded {best_count} tensors from {ckpt_path} into {module.__class__.__name__}.")
    return best_count


def build_raw_backbone() -> nn.Module:
    if args.protocol == "scratch":
        return load_supervised_backbone(args.model, pretrained=False)

    if args.pretraining == "swav":
        if args.model in {"resnet50", "resnet50w5"}:
            return torch.hub.load("facebookresearch/swav:main", args.model)
        raise ValueError(f"Unsupported model {args.model} for swav pretraining")

    if args.pretraining == "oct":
        if args.model == "resnet50":
            model = torch.hub.load("facebookresearch/swav:main", "resnet50")
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 4)
            oct_model_path = (
                "pretraining_OCT2017_output/finetune_resnet50_pretraining_swav_CE_loss_unweighted_pickby"
                "_f1batch_64_lr_0.001_10_10_1_1e-05_epochs_100_hflip_seed_0/checkpoint.pt"
            )
            model.load_state_dict(torch.load(oct_model_path, map_location="cpu"))
            return model
        if args.model == "L_16_imagenet1k":
            if ViT is None:
                raise ImportError("pytorch_pretrained_vit is required for model L_16_imagenet1k")
            model = ViT(args.model, pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 4)
            oct_model_path = (
                "pretraining_OCT2017_output/finetune_L_16_imagenet1k_pretraining_vit_CE_loss_unweighted"
                "_pickby_f1batch_24_lr_1e-05_10_10_1_1e-07_epochs_100_hflip_seed_0/checkpoint.pt"
            )
            model.load_state_dict(torch.load(oct_model_path, map_location="cpu"))
            return model
        if args.model == "vitb14_dino":
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc")
            num_ftrs = model.linear_head.in_features
            model.linear_head = nn.Linear(num_ftrs, 4)
            oct_model_path = (
                "pretraining_OCT2017_output/finetune_vitb14_dino_pretraining_dino_CE_loss_unweighted"
                "_pickby_f1batch_16_lr_1e-05_10_10_1_1e-07_epochs_100_hflip_seed_0/checkpoint.pt"
            )
            model.load_state_dict(torch.load(oct_model_path, map_location="cpu"))
            return model
        raise ValueError(f"Unsupported model {args.model} for oct pretraining")

    if args.pretraining == "barlowtwins":
        if args.model != "resnet50":
            raise ValueError(f"Unsupported model {args.model} for barlowtwins pretraining")
        return torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")

    if args.pretraining == "dino":
        if args.model == "vitb14_dino":
            return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc")
        raise ValueError(f"Unsupported model {args.model} for dino pretraining")

    if args.pretraining == "supervised":
        return load_supervised_backbone(args.model, pretrained=True)

    if args.pretraining == "vit":
        if ViT is None:
            raise ImportError("pytorch_pretrained_vit is required for --pretraining vit")
        return ViT(args.model, pretrained=True)

    raise ValueError(f"Unsupported pretraining option: {args.pretraining}")


def maybe_freeze_encoder(encoder: nn.Module) -> None:
    if args.protocol == "lin_eval":
        for param in encoder.parameters():
            param.requires_grad = False


def build_encoder() -> tuple[BackboneEncoder, int]:
    backbone = build_raw_backbone()
    backbone, feature_dim = strip_model_head(backbone)
    encoder = BackboneEncoder(backbone, feature_dim)
    maybe_freeze_encoder(encoder)
    return encoder, feature_dim


def build_model() -> nn.Module:
    if args.protocol == "complement":
        return R50Complement()

    if args.input_mode == "full_image_zone_head":
        encoder, feature_dim = build_encoder()
        model = FullImageZoneClassifier(encoder, feature_dim)
        if args.fundus_pretrained_ckpt:
            flexible_load_state_dict(model.encoder, args.fundus_pretrained_ckpt, strip_prefixes=["encoder."])
        return model

    if args.input_mode == "zone_crops_shared":
        crop_encoder, crop_dim = build_encoder()
        model = ZoneCropSharedClassifier(crop_encoder, crop_dim)
        if args.fundus_pretrained_ckpt:
            flexible_load_state_dict(model.crop_encoder, args.fundus_pretrained_ckpt, strip_prefixes=["encoder.", "crop_encoder."])
        return model

    if args.input_mode == "hybrid_global_plus_zone":
        global_encoder, global_dim = build_encoder()
        crop_encoder, crop_dim = build_encoder()
        model = HybridGlobalZoneClassifier(global_encoder, global_dim, crop_encoder, crop_dim)
        if args.fundus_pretrained_ckpt:
            flexible_load_state_dict(model.global_encoder, args.fundus_pretrained_ckpt, strip_prefixes=["encoder.", "global_encoder."])
            flexible_load_state_dict(model.crop_encoder, args.fundus_pretrained_ckpt, strip_prefixes=["encoder.", "crop_encoder."])
        model.set_crop_branch_frozen(args.hybrid_freeze_crop_epochs > 0 and args.protocol != "lin_eval")
        return model

    raise ValueError(f"Unsupported input mode: {args.input_mode}")


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).view(-1)
        pt = log_pt.exp()

        focal_weight = (1 - pt) ** self.gamma
        if self.class_weights is not None:
            class_weights = self.class_weights.gather(0, targets.view(-1))
            focal_weight = focal_weight * class_weights

        return -focal_weight * log_pt


class MultiZoneCriterion:
    def __init__(self, loss_name: str, zone_class_weights: torch.Tensor | None, gamma: float, label_smoothing: float = 0.0):
        self.loss_name = loss_name
        self.zone_class_weights = zone_class_weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def _zone_loss_vector(self, zone_logits: torch.Tensor, zone_targets: torch.Tensor, zone_idx: int) -> torch.Tensor:
        zone_weight = None
        if self.zone_class_weights is not None:
            zone_weight = self.zone_class_weights[zone_idx]

        if self.loss_name == "CE":
            return F.cross_entropy(
                zone_logits,
                zone_targets,
                weight=zone_weight,
                reduction="none",
                label_smoothing=self.label_smoothing,
            )

        if self.loss_name == "focal":
            focal = FocalLoss(class_weights=zone_weight, gamma=self.gamma)
            return focal(zone_logits, zone_targets)

        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, observed_mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        total_loss = logits.sum() * 0.0
        total_observed = 0
        for z in range(NUM_ZONES):
            zone_mask = observed_mask[:, z]
            zone_count = int(zone_mask.sum().item())
            if zone_count == 0:
                continue
            zone_logits = logits[zone_mask, z, :]
            zone_targets = targets[zone_mask, z]
            zone_loss = self._zone_loss_vector(zone_logits, zone_targets, z)
            total_loss = total_loss + zone_loss.sum()
            total_observed += zone_count

        if total_observed == 0:
            return total_loss, 0

        return total_loss / total_observed, total_observed


def warmup_lr_lambda(current_step, warmup_steps=args.warmup_epochs):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return 1.0


class EarlyStopping:
    def __init__(self, patience=args.patience, verbose=False, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_metric = -1.0
        self.early_stop = False

    def __call__(self, metric_value: float, model: nn.Module, epoch: int):
        score = metric_value
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric_value, model, epoch)
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric_value, model, epoch)
            self.counter = 0

    def save_checkpoint(self, metric_value: float, model: nn.Module, epoch: int):
        if self.verbose:
            print(f"Validation mean binary F1 improved ({self.best_metric:.6f} -> {metric_value:.6f}). Saving model ...")
        torch.save(model.state_dict(), os.path.join(full_output_path, "checkpoint.pt"))
        self.best_metric = metric_value
        self.best_epoch = epoch


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def mixup_batch(batch: dict[str, Any], lam: float, index: torch.Tensor) -> dict[str, Any]:
    mixed = {}
    for key, value in batch.items():
        if key in {"labels", "observed_mask", "image_file", "visit_id", "patient_id", "eye", "visit_date"}:
            mixed[key] = value
        elif torch.is_tensor(value):
            mixed[key] = lam * value + (1 - lam) * value[index]
        else:
            mixed[key] = value
    return mixed


def mixup_data(batch: dict[str, Any], mixup_alpha=1.0, mixup_beta=1.0):
    lam = np.random.beta(mixup_alpha, mixup_beta)
    batch_size = batch["labels"].size(0)
    index = torch.randperm(batch_size, device=batch["labels"].device)
    mixed_batch = mixup_batch(batch, lam=lam, index=index)
    y_a, y_b = batch["labels"], batch["labels"][index]
    m_a, m_b = batch["observed_mask"], batch["observed_mask"][index]
    return mixed_batch, y_a, m_a, y_b, m_b, lam


def mixup_criterion(criterion_fn, pred, y_a, m_a, y_b, m_b, lam):
    loss_a, obs_a = criterion_fn(pred, y_a, m_a)
    loss_b, obs_b = criterion_fn(pred, y_b, m_b)
    total_obs = max(obs_a + obs_b, 1)
    mixed_loss = lam * loss_a + (1 - lam) * loss_b
    return mixed_loss, total_obs


def load_zone_thresholds(path: str) -> np.ndarray:
    if not path:
        return np.full(NUM_ZONES, 0.5, dtype=np.float32)

    with open(path, "r") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "thresholds" in payload:
        payload = payload["thresholds"]

    if isinstance(payload, dict):
        thresholds = [payload.get(f"Zone{i}", payload.get(f"Zone{i}_label", 0.5)) for i in range(1, NUM_ZONES + 1)]
    elif isinstance(payload, list):
        thresholds = payload
    else:
        raise ValueError(f"Unsupported threshold payload in {path}")

    if len(thresholds) != NUM_ZONES:
        raise ValueError(f"Expected {NUM_ZONES} thresholds in {path}, found {len(thresholds)}")

    thresholds = np.asarray(thresholds, dtype=np.float32)
    if np.any((thresholds <= 0.0) | (thresholds >= 1.0)):
        raise ValueError("Zone thresholds must be in the open interval (0, 1).")
    return thresholds


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def nanmean_or_zero(values: pd.Series) -> float:
    clean = values.dropna()
    return float(clean.mean()) if len(clean) > 0 else 0.0


def compute_zone_metrics(y_true: np.ndarray, y_pred: np.ndarray, observed_mask: np.ndarray):
    rows = []
    flat_true = []
    flat_pred = []

    for z in range(NUM_ZONES):
        zone_mask = observed_mask[:, z].astype(bool)
        observed_count = int(zone_mask.sum())
        if observed_count == 0:
            rows.append(
                {
                    "Zone": z + 1,
                    "ObservedCount": 0,
                    "PositiveRate": np.nan,
                    "Accuracy": np.nan,
                    "BinaryF1": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "Specificity": np.nan,
                    "TN": 0,
                    "FP": 0,
                    "FN": 0,
                    "TP": 0,
                }
            )
            continue

        z_true = y_true[zone_mask, z]
        z_pred = y_pred[zone_mask, z]
        flat_true.append(z_true)
        flat_pred.append(z_pred)

        acc = float(np.mean(z_true == z_pred))
        binary_f1 = float(f1_score(z_true, z_pred, average="binary", pos_label=1, zero_division=0))
        cm = confusion_matrix(z_true, z_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)

        rows.append(
            {
                "Zone": z + 1,
                "ObservedCount": observed_count,
                "PositiveRate": float(np.mean(z_true == 1)),
                "Accuracy": acc,
                "BinaryF1": binary_f1,
                "Precision": precision,
                "Recall": recall,
                "Specificity": specificity,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    df = pd.DataFrame(rows)
    if flat_true:
        flat_true_arr = np.concatenate(flat_true, axis=0)
        flat_pred_arr = np.concatenate(flat_pred, axis=0)
        micro_f1_flat = float(f1_score(flat_true_arr, flat_pred_arr, average="micro", labels=[0, 1], zero_division=0))
        macro_f1_flat = float(f1_score(flat_true_arr, flat_pred_arr, average="macro", labels=[0, 1], zero_division=0))
    else:
        micro_f1_flat = 0.0
        macro_f1_flat = 0.0

    summary = {
        "mean_accuracy": nanmean_or_zero(df["Accuracy"]),
        "mean_binary_f1": nanmean_or_zero(df["BinaryF1"]),
        "mean_precision": nanmean_or_zero(df["Precision"]),
        "mean_recall": nanmean_or_zero(df["Recall"]),
        "mean_specificity": nanmean_or_zero(df["Specificity"]),
        "macro_f1_across_zones": nanmean_or_zero(df["BinaryF1"]),
        "micro_f1_flat": micro_f1_flat,
        "macro_f1_flat": macro_f1_flat,
        "observed_zone_labels": int(observed_mask.sum()),
        "rows_with_observed_zone_labels": int(observed_mask.any(axis=1).sum()),
    }
    return df, summary


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def compute_any_positive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_pos: np.ndarray,
    observed_mask: np.ndarray,
) -> dict[str, Any]:
    row_mask = observed_mask.any(axis=1)
    if not row_mask.any():
        empty = np.array([], dtype=np.float32)
        return {
            "valid_row_mask": row_mask,
            "y_true": empty,
            "y_pred": empty,
            "y_prob": empty,
            "metrics": {
                "n_rows": 0,
                "f1": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "roc_auc": None,
                "pr_auc": None,
            },
        }

    observed_true = np.where(observed_mask, y_true, 0)
    observed_pred = np.where(observed_mask, y_pred, 0)
    observed_prob = np.where(observed_mask, y_prob_pos, 0.0)

    any_positive_true = (observed_true == 1).any(axis=1).astype(int)[row_mask]
    any_positive_pred = (observed_pred == 1).any(axis=1).astype(int)[row_mask]
    any_positive_prob = (1.0 - np.prod(np.where(observed_mask, 1.0 - observed_prob, 1.0), axis=1))[row_mask]

    cm = confusion_matrix(any_positive_true, any_positive_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    precision_val = precision_score(any_positive_true, any_positive_pred, zero_division=0)
    sensitivity_val = recall_score(any_positive_true, any_positive_pred, zero_division=0)
    specificity_val = safe_div(tn, tn + fp)

    metrics = {
        "n_rows": int(len(any_positive_true)),
        "f1": float(f1_score(any_positive_true, any_positive_pred, zero_division=0)),
        "accuracy": float(np.mean(any_positive_true == any_positive_pred)),
        "precision": float(precision_val),
        "sensitivity": float(sensitivity_val),
        "specificity": float(specificity_val),
        "roc_auc": safe_roc_auc(any_positive_true, any_positive_prob),
        "pr_auc": safe_pr_auc(any_positive_true, any_positive_prob),
    }
    return {
        "valid_row_mask": row_mask,
        "y_true": any_positive_true,
        "y_pred": any_positive_pred,
        "y_prob": any_positive_prob,
        "metrics": metrics,
    }


def bootstrap_any_positive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {}

    rng = np.random.default_rng(seed)
    samples = {
        "f1": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    indices = np.arange(len(y_true))
    for _ in range(n_boot):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        y_t = y_true[boot_idx]
        y_p = y_pred[boot_idx]
        y_s = y_prob[boot_idx]

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        samples["f1"].append(float(f1_score(y_t, y_p, zero_division=0)))
        samples["accuracy"].append(float(np.mean(y_t == y_p)))
        samples["sensitivity"].append(float(recall_score(y_t, y_p, zero_division=0)))
        samples["specificity"].append(float(safe_div(tn, tn + fp)))

        roc_auc = safe_roc_auc(y_t, y_s)
        pr_auc = safe_pr_auc(y_t, y_s)
        if roc_auc is not None:
            samples["roc_auc"].append(roc_auc)
        if pr_auc is not None:
            samples["pr_auc"].append(pr_auc)

    def summarize(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float32)
        return {
            "mean": float(arr.mean()),
            "ci_low": float(np.percentile(arr, 2.5)),
            "ci_high": float(np.percentile(arr, 97.5)),
        }

    return {metric_name: summarize(metric_values) for metric_name, metric_values in samples.items()}


def forward_logits(model: nn.Module, batch: dict[str, Any]):
    reg_term = None

    if args.protocol == "complement":
        logits_raw, out1, out2 = model(batch["full_image"])
        out1 = out1 - out1.mean(dim=0)
        out1 = out1 / (torch.norm(out1, dim=0) + 1e-8)
        out2 = out2 - out2.mean(dim=0)
        out2 = out2 / (torch.norm(out2, dim=0) + 1e-8)
        reg_term = args.beta * torch.sum(torch.pow(torch.matmul(out1.T, out2), 2))
        logits = logits_raw.reshape(-1, NUM_ZONES, NUM_CLASSES)
        return logits, reg_term

    outputs = model(
        full_image=batch.get("full_image"),
        zone_crops=batch.get("zone_crops"),
    )
    if outputs.ndim != 3 or outputs.shape[1:] != (NUM_ZONES, NUM_CLASSES):
        raise RuntimeError(f"Expected model output shape [B, {NUM_ZONES}, {NUM_CLASSES}] but got {tuple(outputs.shape)}")
    return outputs, reg_term


def compute_predictions_from_logits(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = F.softmax(logits, dim=2)
    positive_probs = probs[:, :, 1]
    preds = (positive_probs >= zone_thresholds_tensor).long()
    return preds, probs


def run_epoch(model, loader, criterion_fn, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_observed = 0
    all_preds = []
    all_labels = []
    all_probs = []
    all_masks = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            batch = move_batch_to_device(batch, device)
            labels = batch["labels"]
            observed_mask = batch["observed_mask"]

            if is_train:
                optimizer.zero_grad()

            if is_train and args.mixup:
                mixed_batch, targets_a, mask_a, targets_b, mask_b, lam = mixup_data(
                    batch,
                    mixup_alpha=args.mixup_alpha,
                    mixup_beta=args.mixup_beta,
                )
                logits, reg_term = forward_logits(model, mixed_batch)
                loss, batch_observed = mixup_criterion(criterion_fn, logits, targets_a, mask_a, targets_b, mask_b, lam)
            else:
                logits, reg_term = forward_logits(model, batch)
                loss, batch_observed = criterion_fn(logits, labels, observed_mask)

            if reg_term is not None:
                loss = loss + reg_term

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * max(batch_observed, 1)
            total_observed += batch_observed

            preds, probs = compute_predictions_from_logits(logits)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
            all_masks.append(observed_mask.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    observed_mask = np.concatenate(all_masks, axis=0).astype(bool)

    zone_df, summary = compute_zone_metrics(y_true, y_pred, observed_mask)
    any_positive = compute_any_positive_metrics(y_true, y_pred, y_prob[:, :, 1], observed_mask)
    avg_loss = total_loss / max(total_observed, 1)

    return {
        "loss": avg_loss,
        "mean_acc": summary["mean_accuracy"],
        "mean_f1": summary["mean_binary_f1"],
        "summary": summary,
        "zone_metrics": zone_df,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "observed_mask": observed_mask,
        "any_positive": any_positive,
    }


def update_batchnorm_stats(model: nn.Module, loader: DataLoader):
    bn_layers = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return

    was_training = model.training
    model.train()
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Updating SWA batch norm"):
            batch = move_batch_to_device(batch, device)
            forward_logits(model, batch)
    model.train(was_training)


def maybe_update_hybrid_freeze_state(model: nn.Module, epoch: int) -> None:
    if not isinstance(model, HybridGlobalZoneClassifier):
        return
    if args.protocol == "lin_eval":
        model.set_crop_branch_frozen(True)
        return
    model.set_crop_branch_frozen(epoch < args.hybrid_freeze_crop_epochs)


def train_model(model, criterion_fn, optimizer, num_epochs=args.num_epochs, ckpt_interval=args.ckpt_interval):
    early_stopping = EarlyStopping(verbose=True) if not args.final else None
    swa_model = AveragedModel(model).to(device) if args.swa else None
    swa_updates = 0

    stats = {
        "train_loss": [],
        "train_mean_acc": [],
        "train_mean_f1": [],
        "val_loss": [],
        "val_mean_acc": [],
        "val_mean_f1": [],
        "val_any_positive_f1": [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        maybe_update_hybrid_freeze_state(model, epoch)

        train_out = run_epoch(model, train_loader, criterion_fn, optimizer=optimizer)
        stats["train_loss"].append(train_out["loss"])
        stats["train_mean_acc"].append(train_out["mean_acc"])
        stats["train_mean_f1"].append(train_out["mean_f1"])

        print(
            f"Train - Loss: {train_out['loss']:.4f}, MeanAcc: {train_out['mean_acc']:.4f}, "
            f"MeanBinaryF1: {train_out['mean_f1']:.4f}, AnyPositiveF1: {train_out['any_positive']['metrics']['f1']:.4f}"
        )

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(epoch - args.warmup_epochs)

        if args.swa and epoch >= args.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_updates += 1

        if not args.final:
            val_out = run_epoch(model, val_loader, criterion_fn, optimizer=None)
            stats["val_loss"].append(val_out["loss"])
            stats["val_mean_acc"].append(val_out["mean_acc"])
            stats["val_mean_f1"].append(val_out["mean_f1"])
            stats["val_any_positive_f1"].append(val_out["any_positive"]["metrics"]["f1"])

            print(
                f"Val   - Loss: {val_out['loss']:.4f}, MeanAcc: {val_out['mean_acc']:.4f}, "
                f"MeanBinaryF1: {val_out['mean_f1']:.4f}, AnyPositiveF1: {val_out['any_positive']['metrics']['f1']:.4f}"
            )

            early_stopping(val_out["mean_f1"], model, epoch)
            if args.earlystop and early_stopping.early_stop:
                print("Early stopping")
                break

        if (epoch + 1) % ckpt_interval == 0:
            torch.save(model.state_dict(), os.path.join(full_output_path, f"checkpoint_epoch_{epoch + 1}.pt"))

        if args.final and (epoch + 1) == num_epochs:
            torch.save(model.state_dict(), os.path.join(full_output_path, "checkpoint.pt"))

    if args.swa and swa_updates > 0:
        update_batchnorm_stats(swa_model, train_loader)
        torch.save(swa_model.state_dict(), os.path.join(full_output_path, "checkpoint_swa.pt"))
        print(f"Saved SWA checkpoint with {swa_updates} updates.")

    if not args.final:
        model.load_state_dict(torch.load(os.path.join(full_output_path, "checkpoint.pt"), map_location=device))
        print(
            f"Best model saved at epoch {early_stopping.best_epoch} with val mean binary F1 = "
            f"{early_stopping.best_metric:.4f}"
        )

        if args.swa and swa_updates > 0:
            swa_val_out = run_epoch(swa_model, val_loader, criterion_fn, optimizer=None)
            print(
                f"SWA   - Val MeanAcc: {swa_val_out['mean_acc']:.4f}, MeanBinaryF1: {swa_val_out['mean_f1']:.4f}, "
                f"AnyPositiveF1: {swa_val_out['any_positive']['metrics']['f1']:.4f}"
            )
            if swa_val_out["mean_f1"] > early_stopping.best_metric:
                torch.save(swa_model.state_dict(), os.path.join(full_output_path, "checkpoint.pt"))
                model.load_state_dict(torch.load(os.path.join(full_output_path, "checkpoint.pt"), map_location=device))
                print("SWA checkpoint outperformed the best validation checkpoint and was promoted to checkpoint.pt.")

        return model, stats, early_stopping.best_epoch, early_stopping.best_metric

    if args.final and args.swa and swa_updates > 0:
        model.load_state_dict(torch.load(os.path.join(full_output_path, "checkpoint_swa.pt"), map_location=device))

    return model, stats


def save_history_plot(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_mean_f1"], label="Train Mean Binary F1")
    if history["val_mean_f1"]:
        plt.plot(epochs, history["val_mean_f1"], label="Val Mean Binary F1")
    plt.title("Mean Binary F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if history["val_loss"]:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 3)
    if history["val_any_positive_f1"]:
        plt.plot(range(1, len(history["val_any_positive_f1"]) + 1), history["val_any_positive_f1"], label="Val Any-Positive F1")
        plt.legend()
    plt.title("Derived Visit F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")

    plt.tight_layout()
    plt.savefig(os.path.join(full_output_path, "history.png"))
    plt.close()


def build_prediction_dataframe(eval_out: dict[str, Any], loader: DataLoader) -> pd.DataFrame:
    pred_df = pd.DataFrame(
        {
            "Image File": list(loader.dataset.metadata["image_files"]),
            "VisitID": list(loader.dataset.metadata["visit_ids"]),
            "Patient_ID": list(loader.dataset.metadata["patient_ids"]),
            "Eye": list(loader.dataset.metadata["eyes"]),
            "Visit_Date": list(loader.dataset.metadata["visit_dates"]),
        }
    )

    y_true = eval_out["y_true"]
    y_pred = eval_out["y_pred"]
    y_prob = eval_out["y_prob"]
    observed_mask = eval_out["observed_mask"]
    any_positive = eval_out["any_positive"]
    row_mask = any_positive["valid_row_mask"]

    any_positive_true_full = np.full(len(pred_df), np.nan)
    any_positive_pred_full = np.full(len(pred_df), np.nan)
    any_positive_prob_full = np.full(len(pred_df), np.nan)
    any_positive_true_full[row_mask] = any_positive["y_true"]
    any_positive_pred_full[row_mask] = any_positive["y_pred"]
    any_positive_prob_full[row_mask] = any_positive["y_prob"]

    pred_df["AnyPositive_True"] = any_positive_true_full
    pred_df["AnyPositive_Pred"] = any_positive_pred_full
    pred_df["AnyPositive_Prob"] = any_positive_prob_full
    pred_df["AnyPositive_Observed"] = row_mask.astype(int)

    for z in range(NUM_ZONES):
        pred_df[f"Zone{z + 1}_Observed"] = observed_mask[:, z].astype(int)
        pred_df[f"Zone{z + 1}_True"] = np.where(observed_mask[:, z], y_true[:, z], np.nan)
        pred_df[f"Zone{z + 1}_Pred"] = np.where(observed_mask[:, z], y_pred[:, z], np.nan)
        pred_df[f"Zone{z + 1}_Prob_0"] = y_prob[:, z, 0]
        pred_df[f"Zone{z + 1}_Prob_1"] = y_prob[:, z, 1]
        pred_df[f"Zone{z + 1}_Threshold"] = zone_thresholds[z]

    return pred_df


def evaluate_on_split(model, loader, criterion_fn, save_path: str, split_name: str):
    eval_out = run_epoch(model, loader, criterion_fn, optimizer=None)
    avg_loss = eval_out["loss"]
    zone_df = eval_out["zone_metrics"]
    summary_core = eval_out["summary"]
    any_positive = eval_out["any_positive"]
    bootstrap_ci = bootstrap_any_positive_metrics(
        any_positive["y_true"],
        any_positive["y_pred"],
        any_positive["y_prob"],
        n_boot=args.visit_bootstrap_samples,
        seed=args.seed,
    )

    zone_metrics_path = os.path.join(save_path, f"{split_name}_zone_metrics.csv")
    zone_df.to_csv(zone_metrics_path, index=False)

    summary = {
        "split": split_name,
        "loss": avg_loss,
        "mean_accuracy": summary_core["mean_accuracy"],
        "mean_binary_f1": summary_core["mean_binary_f1"],
        "mean_precision": summary_core["mean_precision"],
        "mean_recall": summary_core["mean_recall"],
        "mean_specificity": summary_core["mean_specificity"],
        "micro_f1_flat": summary_core["micro_f1_flat"],
        "macro_f1_flat": summary_core["macro_f1_flat"],
        "macro_f1_across_zones": summary_core["macro_f1_across_zones"],
        "observed_zone_labels": summary_core["observed_zone_labels"],
        "rows_with_observed_zone_labels": summary_core["rows_with_observed_zone_labels"],
        "zone_thresholds": zone_thresholds.tolist(),
        "any_positive_visit_f1": any_positive["metrics"]["f1"],
        "any_positive_visit_accuracy": any_positive["metrics"]["accuracy"],
        "any_positive_visit_precision": any_positive["metrics"]["precision"],
        "any_positive_visit_sensitivity": any_positive["metrics"]["sensitivity"],
        "any_positive_visit_specificity": any_positive["metrics"]["specificity"],
        "any_positive_visit_roc_auc": any_positive["metrics"]["roc_auc"],
        "any_positive_visit_pr_auc": any_positive["metrics"]["pr_auc"],
        "any_positive_visit": any_positive["metrics"],
        "any_positive_visit_bootstrap_ci": bootstrap_ci,
    }

    with open(os.path.join(save_path, f"{split_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    y_true = eval_out["y_true"]
    y_pred = eval_out["y_pred"]
    observed_mask = eval_out["observed_mask"]

    cm_rows = []
    report_rows = []
    for z in range(NUM_ZONES):
        zone_mask = observed_mask[:, z].astype(bool)
        z_true = y_true[zone_mask, z]
        z_pred = y_pred[zone_mask, z]

        if len(z_true) == 0:
            cm_rows.append({"Zone": z + 1, "ObservedCount": 0, "cm_00": 0, "cm_01": 0, "cm_10": 0, "cm_11": 0})
            continue

        cm = confusion_matrix(z_true, z_pred, labels=[0, 1])
        cm_rows.append(
            {
                "Zone": z + 1,
                "ObservedCount": int(len(z_true)),
                "cm_00": int(cm[0, 0]),
                "cm_01": int(cm[0, 1]),
                "cm_10": int(cm[1, 0]),
                "cm_11": int(cm[1, 1]),
            }
        )

        report = classification_report(z_true, z_pred, labels=[0, 1], output_dict=True, zero_division=0)
        for cls_name, values in report.items():
            if isinstance(values, dict):
                report_rows.append(
                    {
                        "Zone": z + 1,
                        "ObservedCount": int(len(z_true)),
                        "Class": cls_name,
                        "Precision": values.get("precision", 0.0),
                        "Recall": values.get("recall", 0.0),
                        "F1": values.get("f1-score", 0.0),
                        "Support": values.get("support", 0),
                    }
                )

    pd.DataFrame(cm_rows).to_csv(os.path.join(save_path, f"{split_name}_confusion_matrices.csv"), index=False)
    pd.DataFrame(report_rows).to_csv(os.path.join(save_path, f"{split_name}_classification_report.csv"), index=False)

    pred_df = build_prediction_dataframe(eval_out, loader)
    pred_df.to_csv(os.path.join(save_path, f"{split_name}_predictions.csv"), index=False)

    print(
        f"[{split_name}] Loss={avg_loss:.4f}, MeanAcc={summary['mean_accuracy']:.4f}, "
        f"MeanBinaryF1={summary['mean_binary_f1']:.4f}, AnyPositiveF1={summary['any_positive_visit_f1']:.4f}. "
        f"Saved metrics to {save_path}"
    )


train_full_transform, val_full_transform, train_crop_transform, val_crop_transform = build_transforms()
zone_template = None
if args.input_mode in {"zone_crops_shared", "hybrid_global_plus_zone"}:
    zone_template = load_zone_template(args.zone_template_json)

train_dataset_kwargs = {
    "input_mode": args.input_mode,
    "full_image_transform": train_full_transform,
    "crop_transform": train_crop_transform,
    "zone_template": zone_template,
}
val_dataset_kwargs = {
    "input_mode": args.input_mode,
    "full_image_transform": val_full_transform,
    "crop_transform": val_crop_transform,
    "zone_template": zone_template,
}

if args.final:
    train_dataset = CustomImageDataset("train_final.csv", args.csvpath, args.dataset_path, **train_dataset_kwargs)
else:
    train_dataset = CustomImageDataset("train.csv", args.csvpath, args.dataset_path, **train_dataset_kwargs)

zone_class_weights_cpu = compute_zone_class_weights(train_dataset.labels, train_dataset.observed_mask)

if args.weightedSampling:
    sample_weights = compute_sample_weights(train_dataset.labels, train_dataset.observed_mask, zone_class_weights_cpu)
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = CustomImageDataset("test.csv", args.csvpath, args.dataset_path, **val_dataset_kwargs)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if not args.final:
    val_dataset = CustomImageDataset("val.csv", args.csvpath, args.dataset_path, **val_dataset_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
zone_thresholds = load_zone_thresholds(args.thresholds_json)
zone_thresholds_tensor = torch.tensor(zone_thresholds, dtype=torch.float32, device=device).view(1, NUM_ZONES)
model = build_model().to(device)

zone_class_weights = None if args.unweighted else zone_class_weights_cpu.to(device)
criterion_fn = MultiZoneCriterion(
    loss_name=args.loss,
    zone_class_weights=zone_class_weights,
    gamma=args.gamma,
    label_smoothing=args.label_smoothing,
)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps=args.warmup_epochs))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=args.T_0,
    T_mult=args.T_multi,
    eta_min=args.eta_min,
)


def save_train_metadata(best_epoch: int | None, best_val_mean_f1: float | None):
    metadata = {
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_val_mean_binary_f1": float(best_val_mean_f1) if best_val_mean_f1 is not None else None,
        "num_zones": NUM_ZONES,
        "num_classes": NUM_CLASSES,
        "input_mode": args.input_mode,
        "drop_missing_zone_rows": args.drop_missing_zone_rows,
        "zone_thresholds": zone_thresholds.tolist(),
        "label_smoothing": float(args.label_smoothing),
        "loss": args.loss,
        "gamma": float(args.gamma),
        "image_column": args.image_column,
        "image_absolute_column": args.image_absolute_column,
        "mask_column": args.mask_column,
        "mask_absolute_column": args.mask_absolute_column,
        "image_resolver": args.image_resolver,
        "apply_mask": bool(args.apply_mask),
        "fundus_pretrained_ckpt": args.fundus_pretrained_ckpt,
        "zone_template_json": args.zone_template_json if zone_template is not None else "",
        "swa_enabled": bool(args.swa),
        "hybrid_freeze_crop_epochs": int(args.hybrid_freeze_crop_epochs),
        "train_rows": len(train_dataset),
        "train_rows_with_any_missing_zone": int(train_dataset.observed_mask.any(dim=1).sum().item() - train_dataset.observed_mask.all(dim=1).sum().item()),
        "train_rows_with_all_zones_missing_after_drop": int((~train_dataset.observed_mask.any(dim=1)).sum().item()),
    }
    metadata_file = os.path.join(full_output_path, "train_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")


if args.mode == "train":
    if args.final:
        best_epoch = args.num_epochs - 1
        metadata_file = os.path.join(full_metadata_path, "train_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            if metadata.get("best_epoch") is not None:
                best_epoch = int(metadata["best_epoch"])
                print(f"Loaded best_epoch={best_epoch} from {metadata_file}")

        model, history = train_model(model, criterion_fn, optimizer, num_epochs=best_epoch + 1)
        save_train_metadata(best_epoch=best_epoch, best_val_mean_f1=None)
    else:
        model, history, best_epoch, best_val_mean_f1 = train_model(model, criterion_fn, optimizer)
        save_train_metadata(best_epoch=best_epoch, best_val_mean_f1=best_val_mean_f1)

    save_history_plot(history)

    if not args.final:
        evaluate_on_split(model, val_loader, criterion_fn, full_output_path, split_name="val")
    evaluate_on_split(model, test_loader, criterion_fn, full_output_path, split_name="test")

elif args.mode == "eval":
    model.load_state_dict(torch.load(os.path.join(full_output_path, args.checkpoint), map_location=device))
    if not args.final:
        evaluate_on_split(model, val_loader, criterion_fn, full_output_path, split_name="val")
    evaluate_on_split(model, test_loader, criterion_fn, full_output_path, split_name="test")
