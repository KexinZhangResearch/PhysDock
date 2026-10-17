"""Feed-forward blocks: SwiGLU FeedForward, Transition (RMSNorm + FFN) and DiTTransition (AdaLN + FFN)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import AdaLayerNormZero, Linear, RMSNorm


def set_ffn_backend(model: nn.Module, backend: str) -> None:
    """Implementation of the SwiGLU feed-forwards of model: "torch" or "triton" (the fused kernels of PhysDock.models.kernels:
    swiglu_ffn for the atom-pair features, rmsnorm_swiglu_ffn for the Transitions, adaln_swiglu_ffn for the DiT transitions; each is
    used only where it is faster than torch, with the matmul precision of the torch expression)."""
    if backend not in ("torch", "triton"):
        raise ValueError(f"ffn backend must be torch or triton, got {backend!r}")
    for module in model.modules():
        if isinstance(module, (Transition, DiTTransition)) or type(module).__name__ == "AtomEmbedder":
            module.ffn_backend = backend


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
        """x + self(x). fused: the Triton kernel of PhysDock.models.kernels.swiglu_ffn when it applies (16 / 32 channels, hidden 128:
        the atom-pair features), which does not keep the [..., hidden] activations of the three matmuls for the backward (7 GiB
        for 3072 atoms), same matmul precision as the torch expression"""
        from PhysDock.models.kernels import swiglu_ffn
        if fused and swiglu_ffn.can_fuse(x, self.w2.in_features):
            return swiglu_ffn.swiglu_ffn(x, self.w13.weight, self.w2.weight)
        return x + self(x)


class Transition(nn.Module):
    """Residual update FeedForward(RMSNorm(x))."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.ffn_norm = RMSNorm(dim, eps=eps)
        self.feed_forward = FeedForward(dim)
        self.ffn_backend = "torch"                      # see set_ffn_backend

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> update [..., dim]"""
        return self.feed_forward(self.ffn_norm(x))

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        """x + self(x). With the Triton backend and a shape / precision it pays off for (PhysDock.models.kernels.rmsnorm_swiglu_ffn:
        the 128-channel pair transition) one fused kernel pair that keeps no [..., 2 * hidden] activation for the backward (361 MiB
        per block at 384 tokens), same matmul precision as the torch expression"""
        if self.ffn_backend == "triton":
            from PhysDock.models.kernels import rmsnorm_swiglu_ffn as K
            ff = self.feed_forward
            if K.can_fuse(x, ff.w2.in_features):
                return K.rmsnorm_swiglu_ffn(x, self.ffn_norm.weight, ff.w13.weight, ff.w2.weight, self.ffn_norm.eps)
        return x + self(x)


class DiTTransition(nn.Module):
    """Gated residual update gate * FeedForward(AdaLN(x, t))."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.ffn_norm = AdaLayerNormZero(dim, eps=eps)
        self.feed_forward = FeedForward(dim)
        self.ffn_backend = "torch"                      # see set_ffn_backend

    def residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x + self(x, t); with the Triton backend the fused kernels of PhysDock.models.kernels.adaln_swiglu_ffn where they are enabled
        (they are not by default: slower than torch in the decoder's TF32 matmuls)"""
        if self.ffn_backend == "triton":
            from PhysDock.models.kernels import adaln_swiglu_ffn as K
            ff = self.feed_forward
            if K.can_fuse(x, ff.w2.in_features):
                shift, scale, gate = self.ffn_norm.linear(F.silu(t[..., None, :])).chunk(3, dim=-1)
                return K.adaln_swiglu_ffn(x, shift[:, 0], scale[:, 0], gate[:, 0], ff.w13.weight, ff.w2.weight, self.ffn_norm.norm.eps)
        return x + self(x, t)

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
