"""Attention layers, built from the kernel-level modules of PhysDock.models.primitives. Every head has 32 channels; masks become
additive -inf biases. Pair masks are symmetric. Every layer returns the update that its block adds to the stream."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import (AdaLayerNormZero, Linear, RMSNorm, RMSNormLinear, RMSNormPairBias, SharedBiasAttention)
from PhysDock.utils.tensor_utils import gen_attn_mask

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
        self.pair_bias = RMSNormPairBias(c_z, c_s // HEAD_DIM, inf, eps)
        self.qkvg = RMSNormLinear(c_s, 4 * c_s, eps, bias_dim=c_s)
        self.attend = SharedBiasAttention()
        self.linear_o = Linear(c_s, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [N, c_s]
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, c_s] update
        """
        q, k, v, g = self.qkvg(s).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        o = self.attend(q, k, v, self.pair_bias(z, z_mask, q))
        return self.linear_o(_merge(o)) * g


class MSARowAttentionWithBias(nn.Module):
    """Gated attention along the tokens of every MSA row, with a logit bias from the pair representation."""

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.pair_bias = RMSNormPairBias(c_z, c_m // HEAD_DIM, inf, eps)
        self.qkvg = RMSNormLinear(c_m, 4 * c_m, eps, bias_dim=c_m)
        self.attend = SharedBiasAttention()
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: [N_msa, N_token, c_m]
            z: [N_token, N_token, c_z]
            z_mask: [N_token, N_token]

        Returns:
            [N_msa, N_token, c_m] update
        """
        q, k, v, g = self.qkvg(m).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        o = self.attend(q, k, v, self.pair_bias(z, z_mask, q))
        return self.linear_o(_merge(o)) * g


class MSAColumnAttention(nn.Module):
    """Gated attention along the MSA rows of every token (no mask, no bias)."""

    def __init__(self, c_m: int, eps: float) -> None:
        super().__init__()
        self.qkvg = RMSNormLinear(c_m, 4 * c_m, eps, bias_dim=c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """m: [N_msa, N_token, c_m] -> update [N_msa, N_token, c_m]"""
        q, k, v, g = self.qkvg(m.transpose(-2, -3)).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        o = F.scaled_dot_product_attention(q, k, v)
        return (self.linear_o(_merge(o)) * g).transpose(-2, -3)


class TriangleAttention(nn.Module):
    """Gated attention along the rows of z with a bias from z (starting node); starting_node=False attends along the columns (ending
    node). The bias comes from the same normalised z as q, k, v: the norm and the projection (qkvg) are shared."""

    def __init__(self, c_z: int, inf: float, eps: float, starting_node: bool = True) -> None:
        super().__init__()
        self.inf = inf
        self.starting_node = starting_node
        self.qkvg = RMSNormLinear(c_z, 4 * c_z, eps, bias_dim=c_z)
        self.linear_z = Linear(c_z, c_z // HEAD_DIM, bias=False)
        self.attend = SharedBiasAttention()
        self.linear_o = Linear(c_z, c_z)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] update
        """
        if not self.starting_node:
            z = z.transpose(-2, -3)
        qkvg, z = self.qkvg(z, return_normed=True)
        q, k, v, g = qkvg.chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        bias = self.linear_z(z).permute(2, 0, 1) + gen_attn_mask(z_mask.type_as(q), -self.inf)[None]       # [H, N, N]
        o = self.linear_o(_merge(self.attend(q, k, v, bias))) * g
        if not self.starting_node:
            o = o.transpose(-2, -3)
        return o


class DiTAttention(nn.Module):
    """DiT self-attention: AdaLN-Zero input, RMS-normalised q/k, pair bias shared by all samples, gated output.

    The pair bias linear_z(norm_z(z)) + mask is computed by DiT.pair_bias (once per structure for all blocks, from the norm_z and
    linear_z of every block)."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm_s = AdaLayerNormZero(c_s, eps)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_qkv = Linear(c_s, 3 * c_s, bias=False)              # rows q, k, v
        self.linear_z = Linear(c_z, c_s // HEAD_DIM, bias=False)
        self.norm_q = RMSNorm(HEAD_DIM, eps)
        self.norm_k = RMSNorm(HEAD_DIM, eps)
        self.attend = SharedBiasAttention()
        self.linear_o = Linear(c_s, c_s)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, c_s]
            bias: [H, N, N] logit bias (pair bias + mask, see DiT.pair_bias)
            t: [B, 256] timestep embedding

        Returns:
            [B, N, c_s] update
        """
        q, k, v, gate = self.project(x, t)
        return self.output(self.attend(q, k, v, bias), gate)

    def project(self, x: torch.Tensor, t: torch.Tensor):
        """Attention inputs: q, k, v [B, H, N, 32] and the output gate [B, 1, c_s]"""
        x, gate = self.norm_s(x, t)
        q, k, v = self.linear_qkv(x).split(x.shape[-1], dim=-1)
        return self.norm_q(_heads(q)), self.norm_k(_heads(k)), _heads(v), gate

    def output(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Attention output o [B, H, N, 32] -> gated update [B, N, c_s]"""
        return self.linear_o(_merge(o)) * gate
