"""Basic layers: Linear, RMSNorm, AdaLayerNormZero; the other building blocks are re-exported from the submodules."""
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

_SKIP_INIT = False


@contextlib.contextmanager
def skip_weight_init():
    """Skip the slow truncated-normal init of Linear (use only when all weights are loaded afterwards)."""
    global _SKIP_INIT
    _SKIP_INIT = True
    try:
        yield
    finally:
        _SKIP_INIT = False


def lecun_normal_init_(weight: torch.Tensor) -> None:
    """In-place truncated normal (2 std) with variance 1 / fan_in (AlphaFold init)."""
    fan_in = weight.shape[1]
    std = math.sqrt(1.0 / max(1, fan_in)) / truncnorm.std(a=-2, b=2, loc=0, scale=1)
    samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=math.prod(weight.shape))
    with torch.no_grad():
        weight.copy_(torch.tensor(np.reshape(samples, weight.shape), device=weight.device))


class Linear(nn.Linear):
    """nn.Linear with AlphaFold init ("default" LeCun, "final" zeros, "gating" zeros + bias 1); bf16 stays bf16."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, init: str = "default") -> None:
        super().__init__(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            if bias:
                self.bias.fill_(0)
            if init == "default":
                if not _SKIP_INIT:
                    lecun_normal_init_(self.weight)
            elif init in ("final", "gating"):
                self.weight.fill_(0.0)
                if init == "gating" and bias:
                    self.bias.fill_(1.0)
            else:
                raise ValueError(f"unknown init {init!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., in_dim] -> [..., out_dim]"""
        if x.dtype is torch.bfloat16:
            with torch.autocast(device_type=x.device.type, enabled=False):
                bias = self.bias.to(torch.bfloat16) if self.bias is not None else None
                return F.linear(x, self.weight.to(torch.bfloat16), bias)
        return F.linear(x, self.weight, self.bias)


class RMSNorm(nn.Module):
    """RMS norm over the last dim, computed in float32, with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> [..., dim]"""
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class AdaLayerNormZero(nn.Module):
    """DiT adaptive LayerNorm: LN(x) * (1 + scale) + shift, plus a gate for the following residual branch."""

    def __init__(self, dim: int, eps: float, cond_dim: int = 256) -> None:
        super().__init__()
        self.linear = Linear(cond_dim, 3 * dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Args:
            x: [B, N, dim]
            t: [B, cond_dim] timestep embedding

        Returns:
            x: [B, N, dim] modulated input, gate: [B, 1, dim]
        """
        shift, scale, gate = self.linear(F.silu(t[..., None, :])).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


# the submodules use the layers above, so they are imported last
from .transitions import FeedForward, Transition, DiTTransition  # noqa: E402
from .opm import OuterProductMean  # noqa: E402
from .attentions import (AttentionWithPairBias, MSARowAttentionWithPairBias, MSAColumnAttention,  # noqa: E402
                         TriangleUpdate, TriangleAttention, DiTAttention)
from .timestep_embeddings import TimestepEmbeddings  # noqa: E402
from .transformers import AtomTransformer, Triangleformer, Evoformer, Pairformer, DiT  # noqa: E402
