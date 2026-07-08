from __future__ import annotations

import torch
import torch.nn as nn


class ZoneMeanPooling(nn.Module):
    """Mean-pool ViT patch tokens inside each anatomical zone.

    Args:
        patch_grid: Tensor with shape [B, Hpatch, Wpatch, D].
        zone_masks: Boolean or binary tensor with shape [B, Z, Hpatch, Wpatch].

    Returns:
        Tensor with shape [B, Z, D].
    """

    def forward(self, patch_grid: torch.Tensor, zone_masks: torch.Tensor) -> torch.Tensor:
        if patch_grid.ndim != 4:
            raise ValueError(f"patch_grid must have shape [B,H,W,D], got {tuple(patch_grid.shape)}")
        if zone_masks.ndim != 4:
            raise ValueError(f"zone_masks must have shape [B,Z,H,W], got {tuple(zone_masks.shape)}")
        if patch_grid.shape[0] != zone_masks.shape[0] or patch_grid.shape[1:3] != zone_masks.shape[2:4]:
            raise ValueError(
                "Patch grid and zone masks have incompatible shapes: "
                f"{tuple(patch_grid.shape)} vs {tuple(zone_masks.shape)}"
            )

        masks = zone_masks.to(dtype=patch_grid.dtype)
        weighted_tokens = patch_grid.unsqueeze(1) * masks.unsqueeze(-1)
        summed = weighted_tokens.sum(dim=(2, 3))
        counts = masks.sum(dim=(2, 3)).clamp_min(1.0).unsqueeze(-1)
        return summed / counts

