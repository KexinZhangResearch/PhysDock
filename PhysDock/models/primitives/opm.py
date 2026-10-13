"""Outer-product update of the pair representation from the MSA representation."""
from __future__ import annotations

import torch
import torch.nn as nn

from PhysDock.models.primitives import Linear, RMSNorm

__all__ = ["OuterProductMean"]


class OuterProductMean(nn.Module):
    """Update the pair representation with the outer product of two MSA projections.

    Every MSA row is projected to two ``c_hidden`` vectors ``q`` and ``k``; for every token pair ``(i, j)`` the
    outer product ``q_i (x) k_j`` is **summed over the MSA rows** (it is not divided by the number of rows),
    flattened, projected to ``c_z`` and normalised. The output projection starts at zero.

    Args:
        c_m: Channel size of the MSA representation.
        c_z: Channel size of the pair representation.
        eps: Epsilon of the RMSNorms.
    """

    def __init__(self, c_m: int, c_z: int, eps: float) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_hidden = 32
        self.eps = eps

        self.norm_in = RMSNorm(c_m, eps)

        self.linear_q = Linear(c_m, self.c_hidden)
        self.linear_k = Linear(c_m, self.c_hidden)
        self.linear_o = Linear(self.c_hidden ** 2, c_z, init="final")
        self.norm_out = RMSNorm(c_z, eps)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Compute the pair update.

        Args:
            m: MSA representation, shape ``[N_msa, N_token, c_m]``.

        Returns:
            Float32 pair update of shape ``[N_token, N_token, c_z]`` (to be added to ``z``).
        """
        _, n_token, _ = m.shape
        m_norm = self.norm_in(m)
        q = self.linear_q(m_norm)
        k = self.linear_k(m_norm)
        outer = torch.einsum("...bic,...bjd->...ijcd", q, k).reshape([n_token, n_token, -1])
        outer = self.linear_o(outer)
        outer = self.norm_out(outer)
        return outer.float()
