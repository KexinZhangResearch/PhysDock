"""Attention layers, built from the kernel-level modules of PhysDock.models.primitives. Every head has 32 channels; masks become
additive -inf biases. Pair masks are symmetric. Every layer returns the residual-added stream, x + update."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import (AdaLayerNormZero, Linear, LinearGateResidual, RMSNorm, RMSNormLinear, RMSNormPairBias, SharedBiasAttention)

HEAD_DIM = 32


def _heads(x: torch.Tensor) -> torch.Tensor:
    """[..., N, H * 32] -> [..., H, N, 32]"""
    return x.reshape(*x.shape[:-1], -1, HEAD_DIM).transpose(-2, -3)


def _merge(x: torch.Tensor) -> torch.Tensor:
    """[..., H, N, 32] -> [..., N, H * 32]"""
    x = x.transpose(-2, -3)
    return x.reshape(*x.shape[:-2], -1)


class AttentionWithBias(nn.Module):
    """Gated self-attention over N positions (the single representation of the Pairformer, the atoms of the AtomTransformer) with a
    per-head logit bias from the pair representation."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.qkvg = RMSNormLinear(c_s, 4 * c_s, eps, bias_dim=c_s)
        self.pair_bias = RMSNormPairBias(c_z, c_s // HEAD_DIM, inf, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [N, c_s]
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, c_s] s + update
        """
        q, k, v, g = self.qkvg(s).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        return self.out(s, _merge(self.attend(q, k, v, self.pair_bias(z, z_mask, q))), g)


class MSARowAttentionWithBias(nn.Module):
    """Gated attention along the tokens of every MSA row, with a logit bias from the pair representation."""

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.qkvg = RMSNormLinear(c_m, 4 * c_m, eps, bias_dim=c_m)
        self.pair_bias = RMSNormPairBias(c_z, c_m // HEAD_DIM, inf, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: [N_msa, N_token, c_m]
            z: [N_token, N_token, c_z]
            z_mask: [N_token, N_token]

        Returns:
            [N_msa, N_token, c_m] m + update
        """
        q, k, v, g = self.qkvg(m).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        return self.out(m, _merge(self.attend(q, k, v, self.pair_bias(z, z_mask, q))), g)


class MSAColumnAttention(nn.Module):
    """Gated attention along the MSA rows of every token (no mask, no bias)."""

    def __init__(self, c_m: int, eps: float) -> None:
        super().__init__()
        self.qkvg = RMSNormLinear(c_m, 4 * c_m, eps, bias_dim=c_m)
        self.out = LinearGateResidual(c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """m: [N_msa, N_token, c_m] -> m + update"""
        m = m.transpose(-2, -3)
        q, k, v, g = self.qkvg(m).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        return self.out(m, _merge(F.scaled_dot_product_attention(q, k, v)), g).transpose(-2, -3)


class TriangleAttention(nn.Module):
    """Gated attention along the rows of z with a bias from z (starting node); starting_node=False attends along the columns (ending
    node). The bias is the pair bias of z (its own RMSNorm; checkpoints from before it have the bias from the norm of the q, k, v, g
    projection, see PhysDock.utils.import_old_weights)."""

    def __init__(self, c_z: int, inf: float, eps: float, starting_node: bool = True) -> None:
        super().__init__()
        self.starting_node = starting_node
        self.qkvg = RMSNormLinear(c_z, 4 * c_z, eps, bias_dim=c_z)
        self.pair_bias = RMSNormPairBias(c_z, c_z // HEAD_DIM, inf, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_z)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] z + update
        """
        if not self.starting_node:
            z = z.transpose(-2, -3)
        q, k, v, g = self.qkvg(z).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        z = self.out(z, _merge(self.attend(q, k, v, self.pair_bias(z, z_mask, q))), g)
        return z if self.starting_node else z.transpose(-2, -3)


class DiTAttention(nn.Module):
    """DiT self-attention: AdaLN-Zero input, RMS-normalised q/k, gated output with residual. The logit bias [H, N, N] (shared by all
    samples) is an argument: it comes from the pair_bias layer of the DiT (LayerNormPairBias)."""

    def __init__(self, c_s: int, eps: float) -> None:
        super().__init__()
        self.norm_s = AdaLayerNormZero(c_s, eps)
        self.linear_qkv = Linear(c_s, 3 * c_s, bias=False)              # rows q, k, v
        self.norm_q = RMSNorm(HEAD_DIM, eps)
        self.norm_k = RMSNorm(HEAD_DIM, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_s)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, c_s]
            bias: [H, N, N] logit bias (pair bias + mask)
            mod: [B, 1, 3 * c_s] AdaLN modulation (TimeModulation)

        Returns:
            [B, N, c_s] x + update
        """
        q, k, v, gate = self.project(x, mod)
        return self.output(x, self.attend(q, k, v, bias), gate)

    def project(self, x: torch.Tensor, mod: torch.Tensor):
        """Attention inputs: q, k, v [B, H, N, 32] and the output gate [B, 1, c_s]"""
        h, gate = self.norm_s(x, mod)
        q, k, v = self.linear_qkv(h).split(h.shape[-1], dim=-1)
        return self.norm_q(_heads(q)), self.norm_k(_heads(k)), _heads(v), gate

    def output(self, x: torch.Tensor, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Attention output o [B, H, N, 32] -> x + gated update [B, N, c_s]"""
        return self.out(x, _merge(o), gate)
