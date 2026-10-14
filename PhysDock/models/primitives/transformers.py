"""Transformer stacks (atom, triangle, Evoformer, Pairformer, DiT). Blocks use activation checkpointing when grads are on."""
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from PhysDock.models.primitives import (AttentionWithPairBias, DiTAttention, DiTTransition, MSAColumnAttention,
                                        MSARowAttentionWithPairBias, OuterProductMean, Transition, TriangleAttention,
                                        TriangleUpdate)


def _run(block: nn.Module, *args):
    if torch.is_grad_enabled():
        return checkpoint(block, *args, use_reentrant=False)
    return block(*args)


class AtomBlock(nn.Module):
    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = AttentionWithPairBias(c_a, c_ap, inf, eps)
        self.transition = Transition(c_a, eps)

    def forward(self, a, ap, ap_mask):
        a = a + self.attention(a, ap, ap_mask)
        return a + self.transition(a)


class AtomTransformer(nn.Module):
    """Atom transformer: pair-biased attention + transition.

    a: [N_atom, c_a], ap: [N_atom, N_atom, c_ap], ap_mask: [N_atom, N_atom] -> a: [N_atom, c_a]
    """

    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([AtomBlock(c_a, c_ap, inf, eps) for _ in range(no_blocks)])

    def forward(self, a: torch.Tensor, ap: torch.Tensor, ap_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            a = _run(block, a, ap, ap_mask)
        return a


class TriangleBlock(nn.Module):
    def __init__(self, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)

    def forward(self, z, z_mask):
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        return z + self.pair_transition(z)


class Triangleformer(nn.Module):
    """Pair-only stack (template embedder).

    z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> z
    """

    def __init__(self, c_z: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TriangleBlock(c_z, inf, eps) for _ in range(no_blocks)])

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            z = _run(block, z, z_mask)
        return z


class EvoformerBlock(nn.Module):
    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.msa_row_attention = MSARowAttentionWithPairBias(c_m, c_z, inf, eps)
        self.msa_col_attention = MSAColumnAttention(c_m, inf, eps)
        self.msa_transition = Transition(c_m, eps)
        self.opm = OuterProductMean(c_m, c_z, eps)
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)

    def forward(self, m, z, z_mask):
        m = m + self.msa_row_attention(m, z, z_mask)
        m = m + self.msa_col_attention(m)
        m = m + self.msa_transition(m)
        z = z + self.opm(m)
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        return m, z + self.pair_transition(z)


class Evoformer(nn.Module):
    """MSA stack.

    m: [N_msa, N_token, c_m], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (m, z)
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float, no_blocks: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([EvoformerBlock(c_m, c_z, inf, eps) for _ in range(no_blocks)])

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor):
        for block in self.blocks:
            m, z = _run(block, m, z, z_mask)
        return m, z


class PairFormerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)
        self.attention = AttentionWithPairBias(c_s, c_z, inf, eps)
        self.transition = Transition(c_s, eps)

    def forward(self, s, z, z_mask):
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        z = z + self.pair_transition(z)
        s = s + self.attention(s, z, z_mask)
        return s + self.transition(s), z


class Pairformer(nn.Module):
    """Pair + single stack.

    s: [N_token, c_s], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (s, z)
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 24) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PairFormerBlock(c_s, c_z, inf, eps) for _ in range(no_blocks)])

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor):
        for block in self.blocks:
            s, z = _run(block, s, z, z_mask)
        return s, z


class DiTBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = DiTAttention(c_s, c_z, inf, eps)
        self.transition = DiTTransition(c_s, eps)

    def forward(self, x, z, t, z_mask, beta):
        x = x + self.attention(x, z, t, z_mask, beta)
        return x + self.transition(x, t)


class DiT(nn.Module):
    """Diffusion transformer conditioned on a timestep embedding.

    x: [B, N, c_s], z: [N, N, c_z], t: [B, 256], z_mask: [N, N], beta: optional [B, N, N] -> x: [B, N, c_s]
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 12) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DiTBlock(c_s, c_z, inf, eps) for _ in range(no_blocks)])

    def forward(self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor, z_mask: torch.Tensor,
                beta: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            x = _run(block, x, z, t, z_mask, beta)
        return x
