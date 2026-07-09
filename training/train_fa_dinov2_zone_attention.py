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
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone FA zone classifier: full FA image -> DINOv2 patch tokens -> "
            "soft zone-mask attention pooling -> shared binary MLP for Zones 1-9."
        )
    )
    parser.add_argument("--csvpath", type=str, default="fold_masked_server_clean/fold_0")
    parser.add_argument("--dataset_path", type=str, default="", help="Fallback root for relative image/mask paths.")
    parser.add_argument("--output_path", type=str, default="output_fa_dinov2_zone_attention")
    parser.add_argument("--image_absolute_column", type=str, default="FA_Image_Abs_Path")
    parser.add_argument("--mask_absolute_column", type=str, default="FA_Mask_Abs_Path")
    parser.add_argument("--image_column", type=str, default="Image_File(FA)")
    parser.add_argument("--mask_column", type=str, default="FA_Mask_Path")
    parser.add_argument("--drop_missing_zone_rows", choices=["none", "any", "all"], default="all")
    parser.add_argument("--dinov2_arch", choices=sorted(DINO_ARCHES), default="dinov2_vitb14")
    parser.add_argument("--image_size", type=int, default=196, help="196 gives a 14x14 patch grid for DINOv2 patch-14.")
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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true", help="Initialize DINOv2 architecture without pretrained weights.")
    parser.add_argument("--brightness", action="store_true")
    parser.add_argument("--contrast", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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
            mask_paths.append(resolve_existing_path(args.dataset_path, row[args.mask_absolute_column]))
        elif args.mask_column in df.columns:
            mask_paths.append(resolve_existing_path(args.dataset_path, row[args.mask_column]))
        else:
            raise ValueError(f"{csv_file} needs either {args.mask_absolute_column!r} or {args.mask_column!r}.")

    image_id_col = args.image_column if args.image_column in df.columns else df.columns[0]
    metadata = {
        "image_file": df[image_id_col].astype(str).tolist(),
        "patient_id": df["Patient_ID"].astype(str).tolist() if "Patient_ID" in df.columns else [""] * len(df),
        "eye": df["Eye"].astype(str).tolist() if "Eye" in df.columns else [""] * len(df),
        "visit_date": df["Visit_Date"].astype(str).tolist() if "Visit_Date" in df.columns else [""] * len(df),
    }
    return SplitData(image_paths=image_paths, mask_paths=mask_paths, labels=labels, observed_mask=observed_mask, metadata=metadata)


def build_transform(train: bool, args: argparse.Namespace) -> transforms.Compose:
    ops = [transforms.Resize((args.image_size, args.image_size))]
    if train and (args.brightness or args.contrast):
        ops.append(
            transforms.ColorJitter(
                brightness=0.2 if args.brightness else 0.0,
                contrast=0.2 if args.contrast else 0.0,
            )
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


class FAZoneDataset(Dataset):
    def __init__(self, split: SplitData, args: argparse.Namespace, train: bool):
        self.split = split
        self.args = args
        self.image_transform = build_transform(train=train, args=args)

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
        full_image = self.image_transform(image)

        mask_size = (self.args.image_size, self.args.image_size)
        zone_masks = resize_zone_stack(zone_stack[:NUM_TARGET_ZONES], mask_size, Image.Resampling.NEAREST)
        zone_masks = np.clip(zone_masks, 0.0, 1.0).astype(np.float32)

        return {
            "full_image": full_image,
            "zone_masks": torch.from_numpy(zone_masks),
            "labels": self.split.labels[idx],
            "observed_mask": self.split.observed_mask[idx],
            "image_file": self.split.metadata["image_file"][idx],
            "patient_id": self.split.metadata["patient_id"][idx],
            "eye": self.split.metadata["eye"][idx],
            "visit_date": self.split.metadata["visit_date"][idx],
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
        empty = soft_masks.sum(dim=-1, keepdim=True) <= self.eps
        soft_masks = torch.where(empty, torch.ones_like(soft_masks), soft_masks)

        norm_tokens = self.norm(patch_tokens)
        norm_queries = F.normalize(self.zone_queries, dim=-1)
        scores = torch.einsum("bnd,zd->bzn", norm_tokens, norm_queries) * self.scale
        scores = scores + soft_masks.clamp_min(self.eps).log()
        attention = torch.softmax(scores, dim=-1)
        return torch.einsum("bzn,bnd->bzd", attention, patch_tokens)


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


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, observed_mask: np.ndarray, threshold: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    y_pred = (y_prob >= threshold).astype(np.int64)
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
        if len(np.unique(z_true)) == 2:
            roc_auc = float(roc_auc_score(z_true, z_prob))
        rows.append(
            {
                "Zone": zone_idx + 1,
                "ObservedCount": int(mask.sum()),
                "PositiveRate": float(np.mean(z_true == 1)),
                "Accuracy": float(np.mean(z_true == z_pred)),
                "BinaryF1": float(f1_score(z_true, z_pred, zero_division=0)),
                "Precision": float(precision_score(z_true, z_pred, zero_division=0)),
                "Recall": float(recall_score(z_true, z_pred, zero_division=0)),
                "Specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
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

    summary = {
        "mean_binary_f1": float(zone_df["BinaryF1"].dropna().mean()) if "BinaryF1" in zone_df else 0.0,
        "mean_accuracy": float(zone_df["Accuracy"].dropna().mean()) if "Accuracy" in zone_df else 0.0,
        "micro_f1_flat": micro_f1,
        "macro_f1_flat": macro_f1,
        "observed_zone_labels": int(observed_mask.sum()),
        "threshold": threshold,
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
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_observed = 0
    all_labels = []
    all_probs = []
    all_masks = []
    all_meta = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            batch = move_batch(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
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

            total_loss += float(loss.item()) * max(observed, 1)
            total_observed += observed
            all_labels.append(batch["labels"].detach().cpu().numpy())
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_masks.append(batch["observed_mask"].detach().cpu().numpy())
            all_meta.extend(
                zip(
                    batch["image_file"],
                    batch["patient_id"],
                    batch["eye"],
                    batch["visit_date"],
                    strict=False,
                )
            )

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    observed_mask = np.concatenate(all_masks, axis=0).astype(bool)
    zone_df, summary = compute_metrics(y_true, y_prob, observed_mask, threshold=threshold)
    summary["loss"] = total_loss / max(total_observed, 1)
    return {
        "summary": summary,
        "zone_metrics": zone_df,
        "y_true": y_true,
        "y_prob": y_prob,
        "observed_mask": observed_mask,
        "metadata": all_meta,
    }


def save_predictions(result: dict[str, Any], path: str, threshold: float) -> None:
    y_true = result["y_true"]
    y_prob = result["y_prob"]
    y_pred = (y_prob >= threshold).astype(np.int64)
    observed_mask = result["observed_mask"]
    rows = []
    for idx, meta in enumerate(result["metadata"]):
        image_file, patient_id, eye, visit_date = meta
        row = {
            "image_file": image_file,
            "patient_id": patient_id,
            "eye": eye,
            "visit_date": visit_date,
        }
        for zone_idx in range(NUM_TARGET_ZONES):
            zone = zone_idx + 1
            row[f"Zone{zone}_observed"] = bool(observed_mask[idx, zone_idx])
            row[f"Zone{zone}_true"] = int(y_true[idx, zone_idx])
            row[f"Zone{zone}_prob"] = float(y_prob[idx, zone_idx])
            row[f"Zone{zone}_pred"] = int(y_pred[idx, zone_idx])
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_path, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.image_size % PATCH_SIZE != 0:
        raise ValueError(f"--image_size must be divisible by {PATCH_SIZE}; got {args.image_size}.")

    train_split = read_split(split_csv_path(args.csvpath, "train"), args)
    val_split = read_split(split_csv_path(args.csvpath, "val"), args)
    test_split = read_split(split_csv_path(args.csvpath, "test"), args)

    train_loader = DataLoader(
        FAZoneDataset(train_split, args, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=1e-6,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    scaler = scaler if scaler.is_enabled() else None

    metadata = {
        "args": vars(args),
        "num_train": len(train_split.image_paths),
        "num_val": len(val_split.image_paths),
        "num_test": len(test_split.image_paths),
        "target_zones": list(range(1, NUM_TARGET_ZONES + 1)),
        "zone10_removed_from_full_fa": True,
        "patch_grid": [args.image_size // PATCH_SIZE, args.image_size // PATCH_SIZE],
        "pos_weights": None if pos_weights is None else pos_weights.detach().cpu().tolist(),
    }
    with open(os.path.join(args.output_path, "train_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

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
        )
        val_result = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            pos_weights=pos_weights,
            threshold=args.threshold,
        )
        if epoch > args.warmup_epochs:
            scheduler.step()

        train_summary = train_result["summary"]
        val_summary = val_result["summary"]
        row = {
            "epoch": epoch,
            "train_loss": train_summary["loss"],
            "train_mean_f1": train_summary["mean_binary_f1"],
            "val_loss": val_summary["loss"],
            "val_mean_f1": val_summary["mean_binary_f1"],
            "lr_backbone": optimizer.param_groups[1]["lr"],
            "lr_head": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"train_f1={row['train_mean_f1']:.4f} val_loss={row['val_loss']:.4f} "
            f"val_f1={row['val_mean_f1']:.4f}"
        )

        if val_summary["mean_binary_f1"] > best_val_f1:
            best_val_f1 = val_summary["mean_binary_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_mean_f1": best_val_f1,
                },
                checkpoint_path,
            )
            val_result["zone_metrics"].to_csv(os.path.join(args.output_path, "val_zone_metrics.csv"), index=False)
            save_predictions(val_result, os.path.join(args.output_path, "val_predictions.csv"), threshold=args.threshold)
            with open(os.path.join(args.output_path, "val_summary.json"), "w") as f:
                json.dump(val_summary, f, indent=2)
        else:
            bad_epochs += 1

        pd.DataFrame(history).to_csv(os.path.join(args.output_path, "history.csv"), index=False)
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} epochs without validation F1 improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_result = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        pos_weights=pos_weights,
        threshold=args.threshold,
    )
    test_result["zone_metrics"].to_csv(os.path.join(args.output_path, "test_zone_metrics.csv"), index=False)
    save_predictions(test_result, os.path.join(args.output_path, "test_predictions.csv"), threshold=args.threshold)
    test_summary = {
        **test_result["summary"],
        "best_epoch": best_epoch,
        "best_val_mean_f1": best_val_f1,
    }
    with open(os.path.join(args.output_path, "test_summary.json"), "w") as f:
        json.dump(test_summary, f, indent=2)
    print(f"Done. best_epoch={best_epoch} best_val_mean_f1={best_val_f1:.4f}")


if __name__ == "__main__":
    main()
