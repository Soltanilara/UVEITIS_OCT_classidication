import argparse
import json
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.optim.lr_scheduler import LambdaLR
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
NUM_CLASSES = 3
ZONE_COLUMNS = [f"Zone{i}_label" for i in range(1, NUM_ZONES + 1)]


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
parser.add_argument("--image_size", type=int, default=512, help="Resize for non-ViT models (square size).")
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
parser.add_argument("--protocol", type=str, default="finetune", choices=["finetune", "lin_eval", "scratch", "complement"])
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--pretraining", type=str, default="swav", choices=["swav", "barlowtwins", "supervised", "vit", "dino", "oct"])
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--csvpath", type=str, default="fold_0")
parser.add_argument("--metadata_path", type=str, default="metadata_0")
parser.add_argument("--dataset_path", type=str, default="Dataset 01032025")
parser.add_argument("--output_path", type=str, default="output_fold_0")
parser.add_argument("--gradcam", action="store_true")
parser.add_argument("--IG", action="store_true")
parser.add_argument("--shap", action="store_true")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()


if args.gradcam or args.IG or args.shap:
    print("Warning: --gradcam/--IG/--shap are disabled in this multi-zone script.")


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


folder_name = (
    f"{f'final_' if args.final else ''}"
    f"{args.protocol}_{args.model}_{f'pretraining_{args.pretraining}_' if args.protocol != 'scratch' else ''}"
    f"{args.loss}_loss_{str(args.gamma) + '_' if args.loss == 'focal' else ''}"
    f"{f'unweighted_' if args.unweighted else ''}"
    f"{f'weightedSampling_' if args.weightedSampling else ''}"
    f"{f'mixup_{args.mixup_alpha}_{args.mixup_beta}_' if args.mixup else ''}"
    f"batch_{args.batch_size}_lr_{args.lr}_decay_{args.decay}_{args.warmup_epochs}_{args.T_0}_{args.T_multi}_{args.eta_min}"
    f"_epochs_{args.num_epochs}{f'_hflip' if args.hflip else ''}{f'_elastic' if args.elastic else ''}"
    f"{f'_brightness' if args.brightness else ''}{f'_contrast' if args.contrast else ''}"
    f"{f'_gnoise' if args.gnoise else ''}{f'_beta_{args.beta}' if args.protocol == 'complement' else ''}"
    f"_zones{NUM_ZONES}x{NUM_CLASSES}_seed_{args.seed}"
)
full_output_path = os.path.join(args.output_path, folder_name)
os.makedirs(full_output_path, exist_ok=True)

if args.final:
    full_metadata_path = os.path.join(args.output_path, args.metadata_path)
    print(f"Metadata loaded from {full_metadata_path}.")


def _validate_zone_values(zone_df: pd.DataFrame, csv_file: str) -> None:
    if zone_df.isna().any().any():
        missing_counts = zone_df.isna().sum()
        raise ValueError(
            f"Missing values found in zone columns for {csv_file}:\n{missing_counts[missing_counts > 0].to_string()}"
        )

    as_int = zone_df.astype(int)
    bad_mask = (as_int.values < 0) | (as_int.values >= NUM_CLASSES)
    if bad_mask.any():
        bad_locs = np.argwhere(bad_mask)
        r, c = bad_locs[0]
        raise ValueError(
            f"Invalid zone class in {csv_file} at row index {r}, column {ZONE_COLUMNS[c]}: "
            f"value={as_int.iat[r, c]} (expected 0..{NUM_CLASSES - 1})"
        )


def load_data(csv_file: str, csvpath: str, folder: str):
    df = pd.read_csv(os.path.join(csvpath, csv_file))

    required_cols = ["Image File", *ZONE_COLUMNS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {os.path.join(csvpath, csv_file)} is missing required columns: {missing}"
        )

    zone_df = df[ZONE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    _validate_zone_values(zone_df, csv_file)

    paths = df["Image File"].astype(str).apply(lambda x: os.path.join(folder, x)).tolist()
    labels = torch.tensor(zone_df.astype(int).values, dtype=torch.long)

    return paths, labels


class CustomImageDataset(Dataset):
    def __init__(self, csv_file: str, csvpath: str, folder: str, transform=None):
        self.paths, self.labels = load_data(csv_file, csvpath, folder)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        labels = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, labels


def build_transforms():
    default_resize = (args.image_size, args.image_size)

    if args.model in {"B_16_imagenet1k", "L_16_imagenet1k"}:
        val_transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    elif args.model in {"vitb14_dino"}:
        val_transform = transforms.Compose(
            [
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    elif args.protocol == "scratch":
        val_transform = transforms.Compose(
            [
                transforms.Resize(default_resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
    else:
        val_transform = transforms.Compose(
            [
                transforms.Resize(default_resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    transform_list = []
    if args.model in {"B_16_imagenet1k", "L_16_imagenet1k"}:
        transform_list.append(transforms.Resize((384, 384)))
    elif args.model in {"vitb14_dino"}:
        transform_list.append(transforms.Resize((518, 518)))
    else:
        transform_list.append(transforms.Resize(default_resize))

    transform_list.append(transforms.ToTensor())

    if args.hflip:
        transform_list.append(transforms.RandomHorizontalFlip())

    if args.elastic:
        transform_list.append(
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)
        )

    if args.brightness or args.contrast:
        brightness = 0.2 if args.brightness else 0.0
        contrast = 0.2 if args.contrast else 0.0
        transform_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast))

    if args.gnoise:
        transform_list.append(transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)))

    if args.model in {"B_16_imagenet1k", "L_16_imagenet1k", "vitb14_dino"} or args.protocol == "scratch":
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    else:
        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose(transform_list)
    return train_transform, val_transform


def compute_zone_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    labels: [N, NUM_ZONES], values in 0..NUM_CLASSES-1
    returns: [NUM_ZONES, NUM_CLASSES]
    """
    weights = torch.ones((NUM_ZONES, NUM_CLASSES), dtype=torch.float32)

    for z in range(NUM_ZONES):
        counts = torch.bincount(labels[:, z], minlength=NUM_CLASSES).float()
        valid = counts > 0
        if valid.any():
            inv = torch.zeros_like(counts)
            inv[valid] = counts.sum() / counts[valid]
            inv_mean = inv[valid].mean().item()
            if inv_mean > 0:
                inv = inv / inv_mean
            weights[z] = inv

    return weights


def compute_sample_weights(labels: torch.Tensor, zone_weights: torch.Tensor) -> list[float]:
    """
    Approximate per-sample weight as mean of zone-level class weights.
    """
    w = []
    for i in range(labels.shape[0]):
        per_zone = [zone_weights[z, labels[i, z]].item() for z in range(NUM_ZONES)]
        w.append(float(np.mean(per_zone)))
    return w


def set_model_head(model: nn.Module, out_dim: int) -> nn.Module:
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_dim)
        return model

    if hasattr(model, "linear_head") and isinstance(model.linear_head, nn.Module):
        in_features = model.linear_head.in_features
        model.linear_head = nn.Linear(in_features, out_dim)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, out_dim)
            return model
        if isinstance(classifier, nn.Sequential):
            layers = list(classifier)
            for idx in range(len(layers) - 1, -1, -1):
                if isinstance(layers[idx], nn.Linear):
                    layers[idx] = nn.Linear(layers[idx].in_features, out_dim)
                    model.classifier = nn.Sequential(*layers)
                    return model

    raise ValueError(f"Could not set output head for model type: {type(model)}")


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


def build_model() -> nn.Module:
    if args.protocol == "complement":
        return R50Complement()

    model = None

    if args.protocol == "scratch":
        if args.model == "resnet50":
            model = models.resnet50(pretrained=False)
        elif args.model == "resnet18":
            model = models.resnet18(pretrained=False)
        elif args.model == "densenet121":
            model = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=False)
        elif args.model == "densenet201":
            model = torch.hub.load("pytorch/vision:v0.10.0", "densenet201", pretrained=False)
        else:
            raise ValueError(f"Unsupported model for scratch protocol: {args.model}")

    else:
        if args.pretraining == "swav":
            if args.model == "resnet50":
                model = torch.hub.load("facebookresearch/swav:main", "resnet50")
            elif args.model == "resnet50w5":
                backbone = torch.hub.load("facebookresearch/swav:main", "resnet50w5")
                if args.protocol == "lin_eval":
                    for p in backbone.parameters():
                        p.requires_grad = False
                model = nn.Sequential(backbone, nn.Linear(10240, NUM_ZONES * NUM_CLASSES))
            else:
                raise ValueError(f"Unsupported model {args.model} for swav pretraining")

        elif args.pretraining == "oct":
            if args.model == "resnet50":
                model = torch.hub.load("facebookresearch/swav:main", "resnet50")
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 4)
                oct_model_path = (
                    "pretraining_OCT2017_output/finetune_resnet50_pretraining_swav_CE_loss_unweighted_pickby"
                    "_f1batch_64_lr_0.001_10_10_1_1e-05_epochs_100_hflip_seed_0/checkpoint.pt"
                )
                model.load_state_dict(torch.load(oct_model_path, map_location="cpu"))
            elif args.model == "L_16_imagenet1k":
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
            elif args.model == "vitb14_dino":
                model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc")
                num_ftrs = model.linear_head.in_features
                model.linear_head = nn.Linear(num_ftrs, 4)
                oct_model_path = (
                    "pretraining_OCT2017_output/finetune_vitb14_dino_pretraining_dino_CE_loss_unweighted"
                    "_pickby_f1batch_16_lr_1e-05_10_10_1_1e-07_epochs_100_hflip_seed_0/checkpoint.pt"
                )
                model.load_state_dict(torch.load(oct_model_path, map_location="cpu"))
            else:
                raise ValueError(f"Unsupported model {args.model} for oct pretraining")

        elif args.pretraining == "barlowtwins":
            model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")

        elif args.pretraining == "dino":
            if args.model == "vitb14_dino":
                model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_lc")
            else:
                raise ValueError(f"Unsupported model {args.model} for dino pretraining")

        elif args.pretraining == "supervised":
            if args.model == "resnet18":
                model = models.resnet18(pretrained=True)
            elif args.model == "resnet50":
                model = models.resnet50(pretrained=True)
            elif args.model == "vgg19_bn":
                model = models.vgg19_bn(pretrained=True)
            elif args.model == "densenet121":
                model = torch.hub.load("pytorch/vision:v0.10.0", "densenet121", pretrained=True)
            elif args.model == "densenet201":
                model = torch.hub.load("pytorch/vision:v0.10.0", "densenet201", pretrained=True)
            else:
                raise ValueError(f"Unsupported model {args.model} for supervised pretraining")

        elif args.pretraining == "vit":
            if ViT is None:
                raise ImportError("pytorch_pretrained_vit is required for --pretraining vit")
            model = ViT(args.model, pretrained=True)

        else:
            raise ValueError(f"Unsupported pretraining option: {args.pretraining}")

    if model is None:
        raise RuntimeError("Model build failed")

    if args.protocol == "lin_eval":
        for p in model.parameters():
            p.requires_grad = False

    if args.model != "resnet50w5":
        model = set_model_head(model, NUM_ZONES * NUM_CLASSES)

    return model


class FocalLoss(nn.Module):
    def __init__(self, class_weights=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        targets = targets.view(-1, 1)
        log_pt = log_probs.gather(1, targets).view(-1)
        pt = log_pt.exp()

        focal_weight = (1 - pt) ** self.gamma

        if self.class_weights is not None:
            cw = self.class_weights.gather(0, targets.view(-1))
            focal_weight = focal_weight * cw

        loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiZoneCriterion:
    def __init__(self, loss_name: str, zone_class_weights: torch.Tensor | None, gamma: float):
        self.loss_name = loss_name
        self.zone_class_weights = zone_class_weights
        self.gamma = gamma

    def _zone_loss(self, zone_logits: torch.Tensor, zone_targets: torch.Tensor, zone_idx: int):
        zone_weight = None
        if self.zone_class_weights is not None:
            zone_weight = self.zone_class_weights[zone_idx]

        if self.loss_name == "CE":
            return F.cross_entropy(zone_logits, zone_targets, weight=zone_weight)

        if self.loss_name == "focal":
            focal = FocalLoss(class_weights=zone_weight, gamma=self.gamma)
            return focal(zone_logits, zone_targets)

        raise ValueError(f"Unsupported loss: {self.loss_name}")

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor):
        losses = []
        for z in range(NUM_ZONES):
            losses.append(self._zone_loss(logits[:, z, :], targets[:, z], z))
        return torch.stack(losses).mean()


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
            print(
                f"Validation mean macro F1 improved ({self.best_metric:.6f} -> {metric_value:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), os.path.join(full_output_path, "checkpoint.pt"))
        self.best_metric = metric_value
        self.best_epoch = epoch


def mixup_data(x, y, mixup_alpha=1.0, mixup_beta=1.0):
    lam = np.random.beta(mixup_alpha, mixup_beta)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion_fn, pred, y_a, y_b, lam):
    return lam * criterion_fn(pred, y_a) + (1 - lam) * criterion_fn(pred, y_b)


def forward_logits(model: nn.Module, inputs: torch.Tensor):
    reg_term = None
    outputs = model(inputs)

    if args.protocol == "complement":
        logits_raw, out1, out2 = outputs
        out1 = out1 - out1.mean(dim=0)
        out1 = out1 / (torch.norm(out1, dim=0) + 1e-8)
        out2 = out2 - out2.mean(dim=0)
        out2 = out2 / (torch.norm(out2, dim=0) + 1e-8)
        reg_term = args.beta * torch.sum(torch.pow(torch.matmul(out1.T, out2), 2))
    else:
        logits_raw = outputs[0] if isinstance(outputs, tuple) else outputs

    if logits_raw.ndim != 2 or logits_raw.shape[1] != NUM_ZONES * NUM_CLASSES:
        raise RuntimeError(
            f"Expected model output shape [B, {NUM_ZONES * NUM_CLASSES}] but got {tuple(logits_raw.shape)}"
        )

    logits = logits_raw.view(-1, NUM_ZONES, NUM_CLASSES)
    return logits, reg_term


def compute_zone_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rows = []
    for z in range(NUM_ZONES):
        z_true = y_true[:, z]
        z_pred = y_pred[:, z]

        acc = float(np.mean(z_true == z_pred))
        macro_f1 = float(
            f1_score(
                z_true,
                z_pred,
                average="macro",
                labels=list(range(NUM_CLASSES)),
                zero_division=0,
            )
        )

        rows.append(
            {
                "Zone": z + 1,
                "Accuracy": acc,
                "MacroF1": macro_f1,
            }
        )

    df = pd.DataFrame(rows)
    mean_acc = float(df["Accuracy"].mean())
    mean_f1 = float(df["MacroF1"].mean())
    return df, mean_acc, mean_f1


def run_epoch(model, loader, criterion_fn, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, labels in tqdm(loader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            if is_train and args.mixup:
                mixed_inputs, targets_a, targets_b, lam = mixup_data(
                    inputs,
                    labels,
                    mixup_alpha=args.mixup_alpha,
                    mixup_beta=args.mixup_beta,
                )
                logits, reg_term = forward_logits(model, mixed_inputs)
                loss = mixup_criterion(criterion_fn, logits, targets_a, targets_b, lam)
            else:
                logits, reg_term = forward_logits(model, inputs)
                loss = criterion_fn(logits, labels)

            if reg_term is not None:
                loss = loss + reg_term

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(logits, dim=2)

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    zone_df, mean_acc, mean_f1 = compute_zone_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)

    return {
        "loss": avg_loss,
        "mean_acc": mean_acc,
        "mean_f1": mean_f1,
        "zone_metrics": zone_df,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_model(model, criterion_fn, optimizer, num_epochs=args.num_epochs, ckpt_interval=args.ckpt_interval):
    early_stopping = EarlyStopping(verbose=True) if not args.final else None

    stats = {
        "train_loss": [],
        "train_mean_acc": [],
        "train_mean_f1": [],
        "val_loss": [],
        "val_mean_acc": [],
        "val_mean_f1": [],
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        train_out = run_epoch(model, train_loader, criterion_fn, optimizer=optimizer)
        stats["train_loss"].append(train_out["loss"])
        stats["train_mean_acc"].append(train_out["mean_acc"])
        stats["train_mean_f1"].append(train_out["mean_f1"])

        print(
            f"Train - Loss: {train_out['loss']:.4f}, MeanAcc: {train_out['mean_acc']:.4f}, "
            f"MeanMacroF1: {train_out['mean_f1']:.4f}"
        )

        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(epoch - args.warmup_epochs)

        if not args.final:
            val_out = run_epoch(model, val_loader, criterion_fn, optimizer=None)
            stats["val_loss"].append(val_out["loss"])
            stats["val_mean_acc"].append(val_out["mean_acc"])
            stats["val_mean_f1"].append(val_out["mean_f1"])

            print(
                f"Val   - Loss: {val_out['loss']:.4f}, MeanAcc: {val_out['mean_acc']:.4f}, "
                f"MeanMacroF1: {val_out['mean_f1']:.4f}"
            )

            early_stopping(val_out["mean_f1"], model, epoch)
            if args.earlystop and early_stopping.early_stop:
                print("Early stopping")
                break

        if (epoch + 1) % ckpt_interval == 0:
            torch.save(model.state_dict(), os.path.join(full_output_path, f"checkpoint_epoch_{epoch + 1}.pt"))

        if args.final and (epoch + 1) == num_epochs:
            torch.save(model.state_dict(), os.path.join(full_output_path, "checkpoint.pt"))

    if not args.final:
        model.load_state_dict(torch.load(os.path.join(full_output_path, "checkpoint.pt"), map_location=device))
        print(
            f"Best model saved at epoch {early_stopping.best_epoch} with val mean macro F1 = "
            f"{early_stopping.best_metric:.4f}"
        )
        return model, stats, early_stopping.best_epoch, early_stopping.best_metric

    return model, stats


def save_history_plot(history: dict):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_mean_f1"], label="Train Mean Macro F1")
    if history["val_mean_f1"]:
        plt.plot(epochs, history["val_mean_f1"], label="Val Mean Macro F1")
    plt.title("Mean Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    if history["val_loss"]:
        plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(full_output_path, "history.png"))
    plt.close()


def evaluate_on_split(model, loader, criterion_fn, save_path: str, split_name: str):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits, _ = forward_logits(model, inputs)
            loss = criterion_fn(logits, labels)
            total_loss += loss.item() * inputs.size(0)

            probs = F.softmax(logits, dim=2)
            preds = torch.argmax(probs, dim=2)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    avg_loss = total_loss / len(loader.dataset)
    zone_df, mean_acc, mean_f1 = compute_zone_metrics(y_true, y_pred)

    zone_metrics_path = os.path.join(save_path, f"{split_name}_zone_metrics.csv")
    zone_df.to_csv(zone_metrics_path, index=False)

    summary = {
        "split": split_name,
        "loss": avg_loss,
        "mean_accuracy": mean_acc,
        "mean_macro_f1": mean_f1,
    }
    with open(os.path.join(save_path, f"{split_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    cm_rows = []
    report_rows = []
    for z in range(NUM_ZONES):
        z_true = y_true[:, z]
        z_pred = y_pred[:, z]

        cm = confusion_matrix(z_true, z_pred, labels=list(range(NUM_CLASSES)))
        cm_rows.append(
            {
                "Zone": z + 1,
                "cm_00": int(cm[0, 0]),
                "cm_01": int(cm[0, 1]),
                "cm_02": int(cm[0, 2]),
                "cm_10": int(cm[1, 0]),
                "cm_11": int(cm[1, 1]),
                "cm_12": int(cm[1, 2]),
                "cm_20": int(cm[2, 0]),
                "cm_21": int(cm[2, 1]),
                "cm_22": int(cm[2, 2]),
            }
        )

        report = classification_report(
            z_true,
            z_pred,
            labels=list(range(NUM_CLASSES)),
            output_dict=True,
            zero_division=0,
        )

        for cls_name, values in report.items():
            if isinstance(values, dict):
                report_rows.append(
                    {
                        "Zone": z + 1,
                        "Class": cls_name,
                        "Precision": values.get("precision", 0.0),
                        "Recall": values.get("recall", 0.0),
                        "F1": values.get("f1-score", 0.0),
                        "Support": values.get("support", 0),
                    }
                )

    pd.DataFrame(cm_rows).to_csv(os.path.join(save_path, f"{split_name}_confusion_matrices.csv"), index=False)
    pd.DataFrame(report_rows).to_csv(os.path.join(save_path, f"{split_name}_classification_report.csv"), index=False)

    pred_df = pd.DataFrame()
    for z in range(NUM_ZONES):
        pred_df[f"Zone{z + 1}_True"] = y_true[:, z]
        pred_df[f"Zone{z + 1}_Pred"] = y_pred[:, z]
        for c in range(NUM_CLASSES):
            pred_df[f"Zone{z + 1}_Prob_{c}"] = y_prob[:, z, c]

    pred_df.to_csv(os.path.join(save_path, f"{split_name}_predictions.csv"), index=False)

    print(
        f"[{split_name}] Loss={avg_loss:.4f}, MeanAcc={mean_acc:.4f}, MeanMacroF1={mean_f1:.4f}. "
        f"Saved metrics to {save_path}"
    )


train_transform, val_transform = build_transforms()

if args.final:
    train_dataset = CustomImageDataset("train_final.csv", args.csvpath, args.dataset_path, transform=train_transform)
else:
    train_dataset = CustomImageDataset("train.csv", args.csvpath, args.dataset_path, transform=train_transform)

zone_class_weights_cpu = compute_zone_class_weights(train_dataset.labels)

if args.weightedSampling:
    sample_weights = compute_sample_weights(train_dataset.labels, zone_class_weights_cpu)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)


test_dataset = CustomImageDataset("test.csv", args.csvpath, args.dataset_path, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

if not args.final:
    val_dataset = CustomImageDataset("val.csv", args.csvpath, args.dataset_path, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)


device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

model = build_model().to(device)

zone_class_weights = None if args.unweighted else zone_class_weights_cpu.to(device)
criterion_fn = MultiZoneCriterion(loss_name=args.loss, zone_class_weights=zone_class_weights, gamma=args.gamma)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warmup_lr_lambda(step, warmup_steps=args.warmup_epochs))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=args.T_0,
    T_mult=args.T_multi,
    eta_min=args.eta_min,
)


if args.mode == "train":
    if args.final:
        best_epoch = args.num_epochs - 1
        metadata_file = os.path.join(full_metadata_path, "train_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            best_epoch = int(metadata.get("best_epoch", best_epoch))
            print(f"Loaded best_epoch={best_epoch} from {metadata_file}")

        model, history = train_model(model, criterion_fn, optimizer, num_epochs=best_epoch + 1)

    else:
        model, history, best_epoch, best_val_mean_f1 = train_model(model, criterion_fn, optimizer)

        metadata = {
            "best_epoch": int(best_epoch),
            "best_val_mean_macro_f1": float(best_val_mean_f1),
            "num_zones": NUM_ZONES,
            "num_classes": NUM_CLASSES,
        }
        metadata_file = os.path.join(full_output_path, "train_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_file}")

    save_history_plot(history)

    if not args.final:
        evaluate_on_split(model, val_loader, criterion_fn, full_output_path, split_name="val")
    evaluate_on_split(model, test_loader, criterion_fn, full_output_path, split_name="test")

elif args.mode == "eval":
    model.load_state_dict(torch.load(os.path.join(full_output_path, args.checkpoint), map_location=device))

    if not args.final:
        evaluate_on_split(model, val_loader, criterion_fn, full_output_path, split_name="val")
    evaluate_on_split(model, test_loader, criterion_fn, full_output_path, split_name="test")
