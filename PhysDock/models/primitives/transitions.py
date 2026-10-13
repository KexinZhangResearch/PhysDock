"""Position-wise feed-forward blocks: SwiGLU :class:`FeedForward`, :class:`Transition`, :class:`DiTTransition`."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import AdaLayerNormZero, Linear, RMSNorm

__all__ = ["FeedForward", "Transition", "DiTTransition"]


class FeedForward(nn.Module):
    """SwiGLU feed-forward network ``w2(silu(w1(x)) * w3(x))`` (LLaMA style, no biases).

    The hidden width is ``2/3`` of ``4 * dim`` (or of ``hidden_dim``), optionally scaled by
    ``ffn_dim_multiplier`` and rounded up to a multiple of ``multiple_of``.

    Args:
        dim: Input and output channel size.
        hidden_dim: Base hidden width before the 2/3 rule; defaults to ``4 * dim``.
        multiple_of: The final hidden width is rounded up to a multiple of this value.
        ffn_dim_multiplier: Optional extra factor applied after the 2/3 rule.
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: Optional[int] = None,
            multiple_of: int = 128,
            ffn_dim_multiplier: Optional[float] = None,
    ) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = Linear(dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, dim, bias=False)
        self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the network.

        Args:
            x: Tensor of shape ``[..., dim]``.

        Returns:
            Tensor of shape ``[..., dim]``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Transition(nn.Module):
    """Pre-norm feed-forward residual branch: ``FeedForward(RMSNorm(x))``.

    Note that :meth:`forward` returns only the *update*; the caller adds it to the residual stream.

    Args:
        dim: Channel size.
        eps: Epsilon of the RMSNorm.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.feed_forward = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the residual update.

        Args:
            x: Tensor of shape ``[..., dim]``.

        Returns:
            Float32 tensor of shape ``[..., dim]``.
        """
        x_norm = self.ffn_norm(x)
        return self.feed_forward(x_norm).float()


class DiTTransition(nn.Module):
    """Feed-forward residual branch modulated by the timestep embedding (AdaLN-Zero).

    Returns only the gated update ``gate * FeedForward(AdaLN(x, t))``.

    Args:
        dim: Channel size.
        eps: Epsilon of the LayerNorm inside :class:`AdaLayerNormZero`.
    """

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.ffn_norm = AdaLayerNormZero(dim, eps=eps)
        self.feed_forward = FeedForward(dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the gated residual update.

        Args:
            x: Activations, shape ``[B, N, dim]``.
            t: Timestep embedding, shape ``[B, 256]``.

        Returns:
            Float32 tensor of shape ``[B, N, dim]``.
        """
        x_norm, gate = self.ffn_norm(x, t)
        return self.feed_forward(x_norm).float() * gate.float()
