"""Attention variants used by PhysDock: pair-biased, MSA row/column, triangle (update and attention), DiT.

All attention layers use a fixed per-head width of 32 channels, so the number of heads is ``c / 32``. Masks are
turned into additive biases of ``-inf`` (a large negative number) before ``scaled_dot_product_attention``.
Pair masks are assumed to be symmetric (``z_mask[i, j] == z_mask[j, i]``), which is true for the masks built
from per-token masks; the transposed ("column"/"incoming") variants therefore reuse the mask untransposed.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import AdaLayerNormZero, LayerNorm, Linear, RMSNorm
from PhysDock.utils.tensor_utils import gen_attn_mask

__all__ = [
    "AttentionWithPairBias", "MSARowAttentionWithPairBias", "MSAColumnAttention",
    "TriangleUpdate", "TriangleAttention", "DiTAttention",
]

HEAD_DIM = 32  # channels per attention head


def _num_heads(c: int, name: str) -> int:
    """Number of heads for ``c`` channels; ``c`` must be a multiple of :data:`HEAD_DIM`."""
    if c % HEAD_DIM != 0:
        raise ValueError(f"{name}={c} must be a multiple of {HEAD_DIM}")
    return c // HEAD_DIM


def _split_heads(x: torch.Tensor, no_heads: int) -> torch.Tensor:
    """``[..., N, H * D] -> [..., H, N, D]``."""
    return x.reshape(*x.shape[:-1], no_heads, HEAD_DIM).transpose(-2, -3)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``[..., H, N, D] -> [..., N, H * D]``."""
    x = x.transpose(-2, -3)
    return x.reshape(*x.shape[:-2], -1)


class AttentionWithPairBias(nn.Module):
    """Gated multi-head self-attention whose logits receive a bias from a pair representation.

    Used for the single representation in the Pairformer (``c_s``/``c_z``) and for atoms in the atom
    transformer (``c_a``/``c_ap``) with dense ``[N, N]`` pair features.

    Args:
        c_s: Channel size of the sequence-like input (multiple of 32).
        c_z: Channel size of the pair representation.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the RMSNorms.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.c_s = c_s
        self.c_hidden = HEAD_DIM
        self.no_heads = _num_heads(c_s, "c_s")
        self.inf = inf
        self.eps = eps

        self.norm_s = RMSNorm(c_s, eps)
        self.norm_z = RMSNorm(c_z, eps)

        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.linear_q = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_g = Linear(c_s, c_s)
        self.linear_o = Linear(c_s, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """Attend over the ``N`` positions of ``s``.

        Args:
            s: Input, shape ``[N, c_s]``.
            z: Pair representation providing the attention bias, shape ``[N, N, c_z]``.
            z_mask: Pair mask, shape ``[N, N]`` (1 = attend, 0 = masked).

        Returns:
            Float32 residual update of shape ``[N, c_s]``.
        """
        s_norm = self.norm_s(s)
        z_norm = self.norm_z(z)

        q = _split_heads(self.linear_q(s_norm), self.no_heads)
        k = _split_heads(self.linear_k(s_norm), self.no_heads)
        v = _split_heads(self.linear_v(s_norm), self.no_heads)
        g = self.linear_g(s_norm)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None]

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0, scale=None)

        o = self.linear_o(_merge_heads(attn_out)) * g
        return o.float()


class MSARowAttentionWithPairBias(nn.Module):
    """Gated attention along the token axis of each MSA row, with a pair-representation bias.

    Args:
        c_m: Channel size of the MSA representation (multiple of 32).
        c_z: Channel size of the pair representation.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the RMSNorms.
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_hidden = HEAD_DIM
        self.no_heads = _num_heads(c_m, "c_m")
        self.inf = inf
        self.eps = eps

        self.norm_m = RMSNorm(c_m, eps)
        self.norm_z = RMSNorm(c_z, eps)

        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.linear_q = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_g = Linear(c_m, c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """Attend over tokens independently for every MSA row.

        Args:
            m: MSA representation, shape ``[N_msa, N_token, c_m]``.
            z: Pair representation, shape ``[N_token, N_token, c_z]``.
            z_mask: Pair mask, shape ``[N_token, N_token]``.

        Returns:
            Float32 residual update of shape ``[N_msa, N_token, c_m]``.
        """
        m_norm = self.norm_m(m)
        z_norm = self.norm_z(z)

        q = _split_heads(self.linear_q(m_norm), self.no_heads)
        k = _split_heads(self.linear_k(m_norm), self.no_heads)
        v = _split_heads(self.linear_v(m_norm), self.no_heads)
        g = self.linear_g(m_norm)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])[None]
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None]

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0, scale=None)

        o = self.linear_o(_merge_heads(attn_out)) * g
        return o.float()


class MSAColumnAttention(nn.Module):
    """Gated attention along the MSA-row axis, independently for every token (no mask, no bias).

    Args:
        c_m: Channel size of the MSA representation (multiple of 32).
        inf: Unused, kept for a uniform constructor signature.
        eps: Epsilon of the RMSNorm.
    """

    def __init__(self, c_m: int, inf: float, eps: float) -> None:
        super().__init__()
        self.c_hidden = HEAD_DIM
        self.c_m = c_m
        self.no_heads = _num_heads(c_m, "c_m")
        self.inf = inf
        self.eps = eps

        self.norm_m = RMSNorm(c_m, eps)
        self.linear_q = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_g = Linear(c_m, c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Attend across MSA rows for every token.

        Args:
            m: MSA representation, shape ``[N_msa, N_token, c_m]``.

        Returns:
            Float32 residual update of shape ``[N_msa, N_token, c_m]``.
        """
        m = m.transpose(-2, -3)                       # [N_token, N_msa, c_m]
        m_norm = self.norm_m(m)

        q = _split_heads(self.linear_q(m_norm), self.no_heads)
        k = _split_heads(self.linear_k(m_norm), self.no_heads)
        v = _split_heads(self.linear_v(m_norm), self.no_heads)
        g = self.linear_g(m_norm)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0, scale=None)

        o = self.linear_o(_merge_heads(attn_out)) * g
        return o.transpose(-2, -3).float()


class TriangleUpdate(nn.Module):
    """Gated triangular multiplicative update of the pair representation (outgoing or incoming edges).

    ``out[i, I] = sum_j q[i, j] * k[I, j]`` over the shared index ``j`` (per channel), followed by a norm, an
    output projection and a gate. With ``transpose=True`` the roles of rows and columns are swapped, which gives
    the "incoming edges" variant.

    Args:
        c_z: Channel size of the pair representation.
        eps: Epsilon of the RMSNorms.
        transpose: Operate on the transposed pair representation.
    """

    def __init__(self, c_z: int, eps: float, transpose: bool = False) -> None:
        super().__init__()
        self.c_hidden = HEAD_DIM
        self.c_z = c_z
        self.eps = eps
        self.transpose = transpose

        self.norm_in = RMSNorm(c_z, eps)
        self.norm_out = RMSNorm(self.c_hidden, eps)

        self.linear_q = Linear(c_z, self.c_hidden)
        self.linear_qx = Linear(c_z, self.c_hidden)
        self.linear_k = Linear(c_z, self.c_hidden)
        self.linear_kx = Linear(c_z, self.c_hidden)
        self.linear_g = Linear(c_z, c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, c_z, init="final")

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """Compute the pair update.

        Args:
            z: Pair representation, shape ``[N, N, c_z]``.
            z_mask: Symmetric pair mask, shape ``[N, N]``.

        Returns:
            Float32 residual update of shape ``[N, N, c_z]``.
        """
        if self.transpose:
            z = z.transpose(-2, -3)
        z = self.norm_in(z)
        q = self.linear_qx(z) * torch.sigmoid(self.linear_q(z)) * z_mask[..., None]
        k = self.linear_kx(z) * torch.sigmoid(self.linear_k(z)) * z_mask[..., None]
        g = torch.sigmoid(self.linear_g(z))
        attn_score = torch.einsum("...ijc,...Ijc->...iIc", q, k)
        attn_score = self.norm_out(attn_score)
        attn_score = self.linear_z(attn_score)
        o = attn_score * g
        if self.transpose:
            o = o.transpose(-2, -3)
        return o.float()


class TriangleAttention(nn.Module):
    """Triangular self-attention on the pair representation (starting or ending node).

    For every row ``i`` the positions ``j`` attend to each other with a logit bias taken from ``z[j, k]``. With
    ``transpose=True`` the same is done along columns (ending node).

    Args:
        c_z: Channel size of the pair representation (multiple of 32).
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the RMSNorm.
        transpose: Operate on the transposed pair representation.
    """

    def __init__(self, c_z: int, inf: float, eps: float, transpose: bool = False) -> None:
        super().__init__()
        self.c_hidden = HEAD_DIM
        self.c_z = c_z
        self.no_heads = _num_heads(c_z, "c_z")
        self.inf = inf
        self.eps = eps
        self.transpose = transpose

        self.norm = RMSNorm(c_z, eps=eps)

        self.linear_q = Linear(c_z, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_z, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_z, self.no_heads * self.c_hidden, bias=False)
        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.linear_g = Linear(c_z, c_z)
        self.linear_o = Linear(c_z, c_z)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """Compute the pair update.

        Args:
            z: Pair representation, shape ``[N, N, c_z]``.
            z_mask: Symmetric pair mask, shape ``[N, N]``.

        Returns:
            Float32 residual update of shape ``[N, N, c_z]``.
        """
        if self.transpose:
            z = z.transpose(-2, -3)

        z_norm = self.norm(z)

        q = _split_heads(self.linear_q(z_norm), self.no_heads)
        k = _split_heads(self.linear_k(z_norm), self.no_heads)
        v = _split_heads(self.linear_v(z_norm), self.no_heads)
        g = self.linear_g(z_norm)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])[None]
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None, :, :]
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0, scale=None)

        o = self.linear_o(_merge_heads(attn_out)) * g
        if self.transpose:
            o = o.transpose(-2, -3)
        return o.float()


class DiTAttention(nn.Module):
    """Self-attention block of the diffusion transformer: AdaLN-Zero, QK-norm, pair bias, output gate.

    One bias per head is computed from the pair representation and shared by all ``B`` noisy samples. An
    optional per-sample bias ``beta`` can be added on top.

    Args:
        c_s: Channel size of the activations (multiple of 32).
        c_z: Channel size of the pair representation.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.c_hidden = HEAD_DIM
        self.c_s = c_s
        self.c_z = c_z
        self.no_heads = _num_heads(c_s, "c_s")
        self.inf = inf
        self.eps = eps

        self.norm_s = AdaLayerNormZero(c_s, eps)
        self.norm_z = LayerNorm(c_z)
        self.linear_q = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.norm_q = RMSNorm(self.c_hidden, eps)
        self.norm_k = RMSNorm(self.c_hidden, eps)
        self.linear_o = Linear(c_s, c_s)

    def forward(
            self,
            bs: torch.Tensor,
            z: torch.Tensor,
            t: torch.Tensor,
            z_mask: torch.Tensor,
            beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend over the ``N`` positions for every sample in the batch.

        Args:
            bs: Activations, shape ``[B, N, c_s]``.
            z: Pair representation, shape ``[N, N, c_z]``.
            t: Timestep embedding, shape ``[B, 256]``.
            z_mask: Pair mask, shape ``[N, N]``.
            beta: Optional extra per-sample logit bias, shape ``[B, N, N]``.

        Returns:
            Float32 gated residual update of shape ``[B, N, c_s]``.
        """
        bs_norm, gate = self.norm_s(bs, t)
        z_norm = self.norm_z(z)

        q = self.norm_q(_split_heads(self.linear_q(bs_norm), self.no_heads))
        k = self.norm_k(_split_heads(self.linear_k(bs_norm), self.no_heads))
        v = _split_heads(self.linear_v(bs_norm), self.no_heads)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])[None, :, :, :]
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None]
        if beta is not None:
            attn_bias = attn_bias + beta.type_as(q)[:, None, :, :]

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0, scale=None)

        o = self.linear_o(_merge_heads(attn_out)).float() * gate.float()
        return o.float()
