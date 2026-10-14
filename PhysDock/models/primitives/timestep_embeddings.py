"""Timestep embedding: 256-d sinusoidal features [cos, sin] followed by a two-layer SiLU MLP."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, dim_in: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(dim_in, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


class TimestepEmbeddings(nn.Module):
    """Sinusoidal embedding of a noise-conditioning scalar, then MLP to embedding_dim."""

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.timestep_embedder = _MLP(256, embedding_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] -> [B, embedding_dim]"""
        freqs = torch.exp(-math.log(10000) * torch.arange(0,
                          128, dtype=torch.float32, device=t.device) / 128)
        angles = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return self.timestep_embedder(emb.to(dtype=t.dtype))
