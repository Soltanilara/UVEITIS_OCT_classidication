from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .zone_embeddings import ZoneEmbedding
from .zone_pooling import ZoneMeanPooling


class RetFoundDinoTokenBackbone(nn.Module):
    """Wrap a RETFound-DINO/timm ViT backbone and expose CLS plus patch tokens."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(getattr(backbone, "num_features", getattr(backbone, "embed_dim", 0)))
        if self.feature_dim <= 0:
            raise ValueError("Could not infer RETFound-DINO feature dimension from the backbone.")

    @property
    def patch_size(self) -> tuple[int, int]:
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", 14)
        if isinstance(patch_size, tuple):
            return int(patch_size[0]), int(patch_size[1])
        return int(patch_size), int(patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone.forward_features(x) if hasattr(self.backbone, "forward_features") else self.backbone(x)

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
                raise ValueError(f"Unsupported RETFound-DINO feature dictionary keys: {sorted(features.keys())}")
            features = token_tensor

        if not torch.is_tensor(features):
            raise ValueError(f"Unsupported RETFound-DINO feature type: {type(features)}")

        if features.ndim == 3:
            if features.shape[1] < 2:
                raise ValueError(f"Expected CLS plus patch tokens, got token shape {tuple(features.shape)}")
            return features[:, 0], features[:, 1:]

        if features.ndim == 2:
            raise ValueError("Backbone returned pooled features only; patch tokens are required for zone-aware pooling.")

        raise ValueError(f"Unsupported RETFound-DINO feature shape: {tuple(features.shape)}")


class RetFoundDinoZoneModel(nn.Module):
    """Zone-Aware RETFound-DINO classifier.

    The backbone runs once on a complete Zone-10-masked image. Patch tokens are
    pooled inside anatomical Zones 1-9, fused with the CLS token and zone
    identity embedding, then scored by a shared binary MLP.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_zones: int = 9,
        zone_embedding_dim: int = 64,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_zones = num_zones
        self.token_backbone = RetFoundDinoTokenBackbone(backbone)
        self.zone_pooling = ZoneMeanPooling()
        self.zone_embedding = ZoneEmbedding(num_zones=num_zones, embedding_dim=zone_embedding_dim)

        feature_dim = self.token_backbone.feature_dim
        hidden_dim = hidden_dim or feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(2 * feature_dim + zone_embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def feature_dim(self) -> int:
        return self.token_backbone.feature_dim

    def _patch_grid_shape(self, image: torch.Tensor, num_patch_tokens: int) -> tuple[int, int]:
        patch_h, patch_w = self.token_backbone.patch_size
        hpatch = image.shape[-2] // patch_h
        wpatch = image.shape[-1] // patch_w
        if hpatch * wpatch == num_patch_tokens:
            return hpatch, wpatch

        square = int(math.sqrt(num_patch_tokens))
        if square * square == num_patch_tokens:
            return square, square

        raise ValueError(
            "Could not infer patch grid from image and patch tokens: "
            f"image={tuple(image.shape)}, patch_size={(patch_h, patch_w)}, tokens={num_patch_tokens}"
        )

    def _resize_zone_masks(self, zone_masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if zone_masks.ndim != 4:
            raise ValueError(f"zone_masks must have shape [B,9,H,W], got {tuple(zone_masks.shape)}")
        if zone_masks.shape[1] != self.num_zones:
            raise ValueError(f"Expected {self.num_zones} zone masks, got {zone_masks.shape[1]}")
        resized = F.interpolate(zone_masks.float(), size=size, mode="nearest")
        return resized > 0.5

    def forward(self, full_image: torch.Tensor, zone_masks: torch.Tensor) -> torch.Tensor:
        """Return independent binary logits with shape [B, 9]."""
        cls_token, patch_tokens = self.token_backbone(full_image)
        hpatch, wpatch = self._patch_grid_shape(full_image, patch_tokens.shape[1])
        patch_grid = patch_tokens.reshape(patch_tokens.shape[0], hpatch, wpatch, patch_tokens.shape[-1])

        patch_zone_masks = self._resize_zone_masks(zone_masks, size=(hpatch, wpatch))
        zone_local = self.zone_pooling(patch_grid, patch_zone_masks)

        cls_per_zone = cls_token.unsqueeze(1).expand(-1, self.num_zones, -1)
        zone_identity = self.zone_embedding(batch_size=full_image.shape[0], device=full_image.device)
        fused = torch.cat([zone_local, cls_per_zone, zone_identity], dim=-1)
        return self.classifier(fused).squeeze(-1)
