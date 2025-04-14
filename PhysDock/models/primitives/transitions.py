import torch
import torch.nn as nn

from .feed_forward import FeedForward
from .rms_norm import RMSNorm
from .adaptive_layer_norm_zero import AdaLayerNormZero


class Transition(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.feed_forward = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.ffn_norm(x)
        x = self.feed_forward(x_norm).float()
        return x


class DiTTransition(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.ffn_norm = AdaLayerNormZero(dim, eps=eps)
        self.feed_forward = FeedForward(dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_norm, gate = self.ffn_norm(x, t)
        x = self.feed_forward(x_norm).float() * gate.float()
        return x
