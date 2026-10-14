"""Transformer blocks and their stacks (Stack: atom, triangle, Evoformer, Pairformer, DiT)."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from PhysDock.models.primitives import (AttentionWithPairBias, DiTAttention, DiTTransition, MSAColumnAttention,
                                        MSARowAttentionWithPairBias, OuterProductMean, Transition, TriangleAttention,
                                        TriangleUpdate)


class Stack(nn.Module):
    """no_blocks copies of a block, applied in sequence (activation checkpointing while grads are on).

    The first n_state arguments are the state that every block updates and returns (one tensor or a tuple of
    n_state tensors); the remaining arguments (pair features, masks, timestep, ...) go unchanged to every block.

    Args:
        block: block class
        no_blocks: number of blocks
        *args, **kwargs: block constructor arguments
        n_state: number of state tensors
    """

    def __init__(self, block, no_blocks: int, *args, n_state: int = 1, **kwargs) -> None:
        super().__init__()
        self.n_state = n_state
        self.blocks = nn.ModuleList([block(*args, **kwargs) for _ in range(no_blocks)])

    def forward(self, *args):
        state, rest = args[:self.n_state], args[self.n_state:]
        for block in self.blocks:
            if torch.is_grad_enabled():
                out = checkpoint(block, *state, *rest, use_reentrant=False)
            else:
                out = block(*state, *rest)
            state = out if self.n_state > 1 else (out,)
        return state if self.n_state > 1 else state[0]


class AtomBlock(nn.Module):
    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = AttentionWithPairBias(c_a, c_ap, inf, eps)
        self.transition = Transition(c_a, eps)

    def forward(self, a, ap, ap_mask):
        a = a + self.attention(a, ap, ap_mask)
        return a + self.transition(a)


class AtomTransformer(Stack):
    """Atom transformer: pair-biased attention + transition.

    a: [N_atom, c_a], ap: [N_atom, N_atom, c_ap], ap_mask: [N_atom, N_atom] -> a: [N_atom, c_a]
    """

    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(AtomBlock, no_blocks, c_a, c_ap, inf, eps)


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


class Triangleformer(Stack):
    """Pair-only stack (template embedder).

    z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> z
    """

    def __init__(self, c_z: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(TriangleBlock, no_blocks, c_z, inf, eps)


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


class Evoformer(Stack):
    """MSA stack.

    m: [N_msa, N_token, c_m], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (m, z)
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float, no_blocks: int = 4) -> None:
        super().__init__(EvoformerBlock, no_blocks, c_m, c_z, inf, eps, n_state=2)


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


class Pairformer(Stack):
    """Pair + single stack.

    s: [N_token, c_s], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (s, z)
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 24) -> None:
        super().__init__(PairFormerBlock, no_blocks, c_s, c_z, inf, eps, n_state=2)


class DiTBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = DiTAttention(c_s, c_z, inf, eps)
        self.transition = DiTTransition(c_s, eps)

    def forward(self, x, z, t, z_mask, beta=None):
        x = x + self.attention(x, z, t, z_mask, beta)
        return x + self.transition(x, t)


class DiT(Stack):
    """Diffusion transformer conditioned on a timestep embedding.

    x: [B, N, c_s], z: [N, N, c_z], t: [B, 256], z_mask: [N, N], beta: optional [B, N, N] -> x: [B, N, c_s]
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 12) -> None:
        super().__init__(DiTBlock, no_blocks, c_s, c_z, inf, eps)
