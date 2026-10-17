"""Feed-forward blocks: SwiGLU FeedForward, Transition (RMSNorm + FFN) and DiTTransition (AdaLN + FFN)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import AdaLayerNormZero, Linear, RMSNorm


class FeedForward(nn.Module):
    """SwiGLU: w2(silu(w1(x)) * w3(x)); hidden size 2/3 * 4 * dim rounded up to a multiple of 128."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = int(2 * (4 * dim) / 3)
        hidden = 128 * ((hidden + 127) // 128)
        self.w13 = Linear(dim, 2 * hidden, bias=False)              # rows w1 | w3 (one matmul)
        self.w2 = Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> [..., dim]"""
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)

    def residual(self, x: torch.Tensor, fused: bool = False) -> torch.Tensor:
        """x + self(x). fused: the Triton kernel of PhysDock.models.kernels.pair_ffn when it applies (16 / 32 channels, hidden 128:
        the atom-pair features), which does not keep the [..., hidden] activations of the three matmuls for the backward (7 GiB
        for 3072 atoms), same matmul precision as the torch expression"""
        from PhysDock.models.kernels import pair_ffn
        if fused and pair_ffn.can_fuse(x, self.w2.in_features):
            return pair_ffn.pair_ffn(x, self.w13.weight, self.w2.weight)
        return x + self(x)


class Transition(nn.Module):
    """Residual update FeedForward(RMSNorm(x))."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.feed_forward = FeedForward(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> update [..., dim]"""
        return self.feed_forward(self.ffn_norm(x))


class DiTTransition(nn.Module):
    """Gated residual update gate * FeedForward(AdaLN(x, t))."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.ffn_norm = AdaLayerNormZero(dim, eps=eps)
        self.feed_forward = FeedForward(dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
            t: [B, 256] timestep embedding

        Returns:
            [B, N, dim] update
        """
        x, gate = self.ffn_norm(x, t)
        return self.feed_forward(x) * gate
