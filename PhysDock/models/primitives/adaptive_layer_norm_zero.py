# Modified From https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/normalization.py

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .layer_norm import LayerNorm
from .linear import Linear


class AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, eps: float):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = Linear(256, 3 * embedding_dim, bias=True)
        self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = self.linear(self.silu(t[..., None, :])).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate
