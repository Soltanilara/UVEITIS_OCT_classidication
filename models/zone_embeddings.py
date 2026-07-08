from __future__ import annotations

import torch
import torch.nn as nn


class ZoneEmbedding(nn.Module):
    """Learnable anatomical zone identity embeddings."""

    def __init__(self, num_zones: int = 9, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_zones, embedding_dim=embedding_dim)

    def forward(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        zone_ids = torch.arange(self.embedding.num_embeddings, device=device)
        zone_embeddings = self.embedding(zone_ids)
        return zone_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

