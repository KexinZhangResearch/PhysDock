"""Transformer stacks of PhysDock (atom, triangle, Evoformer, Pairformer and DiT stacks).

Every stack runs its blocks through activation checkpointing while gradients are being tracked
(``torch.is_grad_enabled()``); under ``torch.no_grad()`` the blocks are simply called.

Residual convention: all sub-layers (attention, transition, triangle updates, outer product mean) return the
*update*, and the block adds it to the running representation.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from PhysDock.models.primitives import (AttentionWithPairBias, DiTAttention, DiTTransition, MSAColumnAttention,
                                        MSARowAttentionWithPairBias, OuterProductMean, Transition, TriangleAttention,
                                        TriangleUpdate)

__all__ = ["AtomTransformer", "Triangleformer", "Evoformer", "Pairformer", "DiT"]


# ======================================================================================
# helpers
# ======================================================================================
def _run(fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn(*args)``, with activation checkpointing when autograd is recording."""
    if torch.is_grad_enabled():
        return checkpoint(fn, *args, use_reentrant=False)
    return fn(*args)


class _PairStack(nn.Module):
    """Base class for blocks that contain the five-module pair update (two triangle updates, two triangle
    attentions and a transition). The modules are registered by :meth:`_build_pair_stack` so every block keeps
    the parameter names ``triangle_row_update`` ... ``pair_transition``."""

    def _build_pair_stack(self, c_z: int, inf: float, eps: float) -> None:
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)

    def _pair_update(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        z = z + self.pair_transition(z)
        return z


class _BlockStack(nn.Module):
    """A ``ModuleList`` of identical blocks (``blocks``) and their count (``no_blocks``)."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.no_blocks = len(blocks)
        self.blocks = nn.ModuleList(blocks)


# ======================================================================================
# atom transformer
# ======================================================================================
class AtomBlock(nn.Module):
    """Pair-biased attention followed by a transition, both residual.

    Args:
        c_a: Atom channel size (multiple of 32).
        c_ap: Atom-pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
    """

    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = AttentionWithPairBias(c_a, c_ap, inf, eps)
        self.transition = Transition(c_a, eps)

    def forward(self, a: torch.Tensor, ap: torch.Tensor, ap_mask: torch.Tensor) -> torch.Tensor:
        """Update the atom representation.

        Args:
            a: Atom representation, shape ``[N_atom, c_a]``.
            ap: Atom-pair representation, shape ``[N_atom, N_atom, c_ap]``.
            ap_mask: Atom-pair mask, shape ``[N_atom, N_atom]``.

        Returns:
            Updated atom representation, shape ``[N_atom, c_a]``.
        """
        a = a + self.attention(a, ap, ap_mask)
        a = a + self.transition(a)
        return a


class AtomTransformer(_BlockStack):
    """Stack of :class:`AtomBlock` (atom-level transformer of the conditioning network).

    Args:
        c_a: Atom channel size.
        c_ap: Atom-pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks: Number of blocks.
    """

    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__([AtomBlock(c_a, c_ap, inf, eps) for _ in range(no_blocks)])

    def forward(self, a: torch.Tensor, ap: torch.Tensor, ap_mask: torch.Tensor) -> torch.Tensor:
        """Run all blocks.

        Args:
            a: Atom representation, shape ``[N_atom, c_a]``.
            ap: Atom-pair representation, shape ``[N_atom, N_atom, c_ap]``.
            ap_mask: Atom-pair mask, shape ``[N_atom, N_atom]``.

        Returns:
            Atom representation, shape ``[N_atom, c_a]``.
        """
        for block in self.blocks:
            a = _run(block, a, ap, ap_mask)
        return a


# ======================================================================================
# pair-only stack (template embedding)
# ======================================================================================
class TriangleBlock(_PairStack):
    """Triangle updates, triangle attentions and pair transition (no MSA, no single representation).

    Args:
        c_z: Pair channel size (multiple of 32).
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
    """

    def __init__(self, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self._build_pair_stack(c_z, inf, eps)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """Update the pair representation.

        Args:
            z: Pair representation, shape ``[N_token, N_token, c_z]``.
            z_mask: Pair mask, shape ``[N_token, N_token]``.

        Returns:
            Updated pair representation, same shape as ``z``.
        """
        return self._pair_update(z, z_mask)


class Triangleformer(_BlockStack):
    """Stack of :class:`TriangleBlock` (used by the template pair embedder).

    Args:
        c_z: Pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks: Number of blocks.
    """

    def __init__(self, c_z: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__([TriangleBlock(c_z, inf, eps) for _ in range(no_blocks)])

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """Run all blocks.

        Args:
            z: Pair representation, shape ``[N_token, N_token, c_z]``.
            z_mask: Pair mask, shape ``[N_token, N_token]``.

        Returns:
            Updated pair representation, same shape as ``z``.
        """
        for block in self.blocks:
            z = _run(block, z, z_mask)
        return z


# ======================================================================================
# Evoformer
# ======================================================================================
class EvoformerBlock(_PairStack):
    """MSA row/column attention and transition, outer-product-mean into the pair stack, then the pair update.

    Args:
        c_m: MSA channel size (multiple of 32).
        c_z: Pair channel size (multiple of 32).
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.msa_row_attention = MSARowAttentionWithPairBias(c_m, c_z, inf, eps)
        self.msa_col_attention = MSAColumnAttention(c_m, inf, eps)
        self.msa_transition = Transition(c_m, eps)
        self.opm = OuterProductMean(c_m, c_z, eps)
        self._build_pair_stack(c_z, inf, eps)

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the MSA and pair representations.

        Args:
            m: MSA representation, shape ``[N_msa, N_token, c_m]``.
            z: Pair representation, shape ``[N_token, N_token, c_z]``.
            z_mask: Pair mask, shape ``[N_token, N_token]``.

        Returns:
            Tuple ``(m, z)`` with the shapes of the inputs.
        """
        m = m + self.msa_row_attention(m, z, z_mask)
        m = m + self.msa_col_attention(m)
        m = m + self.msa_transition(m)
        z = z + self.opm(m)
        z = self._pair_update(z, z_mask)
        return m, z


class Evoformer(_BlockStack):
    """Stack of :class:`EvoformerBlock`.

    Args:
        c_m: MSA channel size.
        c_z: Pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks: Number of blocks.
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float, no_blocks: int = 4) -> None:
        super().__init__([EvoformerBlock(c_m, c_z, inf, eps) for _ in range(no_blocks)])

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all blocks.

        Args:
            m: MSA representation, shape ``[N_msa, N_token, c_m]``.
            z: Pair representation, shape ``[N_token, N_token, c_z]``.
            z_mask: Pair mask, shape ``[N_token, N_token]``.

        Returns:
            Tuple ``(m, z)`` with the shapes of the inputs.
        """
        for block in self.blocks:
            m, z = _run(block, m, z, z_mask)
        return m, z


# ======================================================================================
# Pairformer
# ======================================================================================
class PairFormerBlock(_PairStack):
    """Pair update followed by pair-biased single attention and a single transition.

    Args:
        c_s: Single (token) channel size (multiple of 32).
        c_z: Pair channel size (multiple of 32).
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self._build_pair_stack(c_z, inf, eps)
        self.attention = AttentionWithPairBias(c_s, c_z, inf, eps)
        self.transition = Transition(c_s, eps)

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the single and pair representations.

        Args:
            s: Single representation, shape ``[N_token, c_s]``.
            z: Pair representation, shape ``[N_token, N_token, c_z]``.
            z_mask: Pair mask, shape ``[N_token, N_token]``.

        Returns:
            Tuple ``(s, z)`` with the shapes of the inputs.
        """
        z = self._pair_update(z, z_mask)
        s = s + self.attention(s, z, z_mask)
        s = s + self.transition(s)
        return s, z


class Pairformer(_BlockStack):
    """Stack of :class:`PairFormerBlock`.

    Args:
        c_s: Single channel size.
        c_z: Pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks: Number of blocks.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 24) -> None:
        super().__init__([PairFormerBlock(c_s, c_z, inf, eps) for _ in range(no_blocks)])

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run all blocks.

        Args:
            s: Single representation, shape ``[N_token, c_s]``.
            z: Pair representation, shape ``[N_token, N_token, c_z]``.
            z_mask: Pair mask, shape ``[N_token, N_token]``.

        Returns:
            Tuple ``(s, z)`` with the shapes of the inputs.
        """
        for block in self.blocks:
            s, z = _run(block, s, z, z_mask)
        return s, z


# ======================================================================================
# diffusion transformer
# ======================================================================================
class DiTBlock(nn.Module):
    """Timestep-modulated attention and transition (both AdaLN-Zero gated, residual).

    Args:
        c_s: Channel size of the activations (multiple of 32).
        c_z: Pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = DiTAttention(c_s, c_z, inf, eps)
        self.transition = DiTTransition(c_s, eps)

    def forward(
            self,
            bs: torch.Tensor,
            z: torch.Tensor,
            t: torch.Tensor,
            z_mask: torch.Tensor,
            beta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Update the activations of every noisy sample.

        Args:
            bs: Activations, shape ``[B, N, c_s]``.
            z: Pair representation, shape ``[N, N, c_z]``.
            t: Timestep embedding, shape ``[B, 256]``.
            z_mask: Pair mask, shape ``[N, N]``.
            beta: Optional extra per-sample logit bias, shape ``[B, N, N]``.

        Returns:
            Updated activations, shape ``[B, N, c_s]``.
        """
        bs = bs + self.attention(bs, z, t, z_mask, beta)
        bs = bs + self.transition(bs, t)
        return bs


class DiT(_BlockStack):
    """Stack of :class:`DiTBlock`.

    Args:
        c_s: Channel size of the activations.
        c_z: Pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks: Number of blocks.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 12) -> None:
        super().__init__([DiTBlock(c_s, c_z, inf, eps) for _ in range(no_blocks)])

    def forward(
            self,
            bs: torch.Tensor,
            z: torch.Tensor,
            t: torch.Tensor,
            z_mask: torch.Tensor,
            beta: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run all blocks.

        Args:
            bs: Activations, shape ``[B, N, c_s]``.
            z: Pair representation, shape ``[N, N, c_z]``.
            t: Timestep embedding, shape ``[B, 256]``.
            z_mask: Pair mask, shape ``[N, N]``.
            beta: Optional extra per-sample logit bias, shape ``[B, N, N]``.

        Returns:
            Activations, shape ``[B, N, c_s]``.
        """
        for block in self.blocks:
            bs = _run(block, bs, z, t, z_mask, beta)
        return bs
