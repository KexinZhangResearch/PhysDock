import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .linear import Linear


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: Optional[int] = None,
            multiple_of: int = 128,
            ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)

        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
