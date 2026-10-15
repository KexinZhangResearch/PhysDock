"""Attention layers. Every head has 32 channels; masks become additive -inf biases. Pair masks are symmetric."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import AdaLayerNormZero, Linear, RMSNorm
from PhysDock.utils.tensor_utils import gen_attn_mask


def _heads(x: torch.Tensor) -> torch.Tensor:
    """[..., N, H * 32] -> [..., H, N, 32]"""
    return x.reshape(*x.shape[:-1], -1, 32).transpose(-2, -3)


def _merge(x: torch.Tensor) -> torch.Tensor:
    """[..., H, N, 32] -> [..., N, H * 32]"""
    x = x.transpose(-2, -3)
    return x.reshape(*x.shape[:-2], -1)


class AttentionWithPairBias(nn.Module):
    """Gated self-attention over N positions with a per-head logit bias from the pair representation."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm_s = RMSNorm(c_s, eps)
        self.norm_z = RMSNorm(c_z, eps)
        self.linear_z = Linear(c_z, c_s // 32, bias=False)
        self.linear_q = Linear(c_s, c_s, bias=False)
        self.linear_k = Linear(c_s, c_s, bias=False)
        self.linear_v = Linear(c_s, c_s, bias=False)
        self.linear_g = Linear(c_s, c_s)
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
        s = self.norm_s(s)
        q, k, v = _heads(self.linear_q(s)), _heads(self.linear_k(s)), _heads(self.linear_v(s))
        bias = self.linear_z(self.norm_z(z)).permute(2, 0, 1) + gen_attn_mask(z_mask.type_as(q), -self.inf)[None]
        o = F.scaled_dot_product_attention(q, k, v, bias)
        return self.linear_o(_merge(o)) * self.linear_g(s)


class MSARowAttentionWithPairBias(nn.Module):
    """Gated attention along the tokens of every MSA row, with a logit bias from the pair representation."""

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm_m = RMSNorm(c_m, eps)
        self.norm_z = RMSNorm(c_z, eps)
        self.linear_z = Linear(c_z, c_m // 32, bias=False)
        self.linear_q = Linear(c_m, c_m, bias=False)
        self.linear_k = Linear(c_m, c_m, bias=False)
        self.linear_v = Linear(c_m, c_m, bias=False)
        self.linear_g = Linear(c_m, c_m)
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
        m = self.norm_m(m)
        q, k, v = _heads(self.linear_q(m)), _heads(self.linear_k(m)), _heads(self.linear_v(m))
        bias = self.linear_z(self.norm_z(z)).permute(2, 0, 1)[None] + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None]
        o = F.scaled_dot_product_attention(q, k, v, bias)
        return self.linear_o(_merge(o)) * self.linear_g(m)


class MSAColumnAttention(nn.Module):
    """Gated attention along the MSA rows of every token (no mask, no bias)."""

    def __init__(self, c_m: int, inf: float, eps: float) -> None:
        super().__init__()
        self.norm_m = RMSNorm(c_m, eps)
        self.linear_q = Linear(c_m, c_m, bias=False)
        self.linear_k = Linear(c_m, c_m, bias=False)
        self.linear_v = Linear(c_m, c_m, bias=False)
        self.linear_g = Linear(c_m, c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """m: [N_msa, N_token, c_m] -> update [N_msa, N_token, c_m]"""
        m = self.norm_m(m.transpose(-2, -3))
        q, k, v = _heads(self.linear_q(m)), _heads(self.linear_k(m)), _heads(self.linear_v(m))
        o = F.scaled_dot_product_attention(q, k, v)
        return (self.linear_o(_merge(o)) * self.linear_g(m)).transpose(-2, -3)


class TriangleUpdate(nn.Module):
    """Gated triangular multiplicative update out[i, k] = sum_j q[i, j] * k[k, j]; transpose=True for incoming edges."""

    def __init__(self, c_z: int, eps: float, transpose: bool = False) -> None:
        super().__init__()
        self.transpose = transpose
        self.norm_in = RMSNorm(c_z, eps)
        self.norm_out = RMSNorm(32, eps)
        self.linear_q = Linear(c_z, 32)
        self.linear_qx = Linear(c_z, 32)
        self.linear_k = Linear(c_z, 32)
        self.linear_kx = Linear(c_z, 32)
        self.linear_g = Linear(c_z, c_z, init="gating")
        self.linear_z = Linear(32, c_z, init="final")

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] update
        """
        if self.transpose:
            z = z.transpose(-2, -3)
        z = self.norm_in(z)
        q = self.linear_qx(z) * torch.sigmoid(self.linear_q(z)) * z_mask[..., None]
        k = self.linear_kx(z) * torch.sigmoid(self.linear_k(z)) * z_mask[..., None]
        o = self.linear_z(self.norm_out(torch.einsum("...ijc,...Ijc->...iIc", q, k))) * torch.sigmoid(self.linear_g(z))
        if self.transpose:
            o = o.transpose(-2, -3)
        return o


class TriangleAttention(nn.Module):
    """Gated attention along the rows of z with a bias from z; transpose=True attends along the columns."""

    def __init__(self, c_z: int, inf: float, eps: float, transpose: bool = False) -> None:
        super().__init__()
        self.inf = inf
        self.transpose = transpose
        self.norm = RMSNorm(c_z, eps=eps)
        self.linear_q = Linear(c_z, c_z, bias=False)
        self.linear_k = Linear(c_z, c_z, bias=False)
        self.linear_v = Linear(c_z, c_z, bias=False)
        self.linear_z = Linear(c_z, c_z // 32, bias=False)
        self.linear_g = Linear(c_z, c_z)
        self.linear_o = Linear(c_z, c_z)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] update
        """
        if self.transpose:
            z = z.transpose(-2, -3)
        z = self.norm(z)
        q, k, v = _heads(self.linear_q(z)), _heads(self.linear_k(z)), _heads(self.linear_v(z))
        bias = self.linear_z(z).permute(2, 0, 1)[None] + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None]
        o = self.linear_o(_merge(F.scaled_dot_product_attention(q, k, v, bias))) * self.linear_g(z)
        if self.transpose:
            o = o.transpose(-2, -3)
        return o


class DiTAttention(nn.Module):
    """DiT self-attention: AdaLN-Zero input, RMS-normalised q/k, pair bias shared by all samples, gated output.

    The pair bias linear_z(norm_z(z)) + mask is computed by DiT.pair_bias (once per structure for all blocks)."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm_s = AdaLayerNormZero(c_s, eps)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_q = Linear(c_s, c_s, bias=False)
        self.linear_k = Linear(c_s, c_s, bias=False)
        self.linear_v = Linear(c_s, c_s, bias=False)
        self.linear_z = Linear(c_z, c_s // 32, bias=False)
        self.norm_q = RMSNorm(32, eps)
        self.norm_k = RMSNorm(32, eps)
        self.linear_o = Linear(c_s, c_s)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, c_s]
            bias: [H, N, N] contiguous logit bias (pair bias + mask, see DiT.pair_bias)
            t: [B, 256] timestep embedding

        Returns:
            [B, N, c_s] update
        """
        x, gate = self.norm_s(x, t)
        q = self.norm_q(_heads(self.linear_q(x)))
        k = self.norm_k(_heads(self.linear_k(x)))
        v = _heads(self.linear_v(x))
        o = F.scaled_dot_product_attention(q, k, v, bias[None])
        return self.linear_o(_merge(o)) * gate
