"""Outer product (summed over MSA rows) as pair update."""
import torch
import torch.nn as nn

from PhysDock.models.primitives import Linear, RMSNorm


class OuterProductMean(nn.Module):
    """Pair update from the outer product q_i x k_j of two MSA projections, summed (not averaged) over rows."""

    def __init__(self, c_m: int, c_z: int, eps: float) -> None:
        super().__init__()
        self.norm_in = RMSNorm(c_m, eps)
        self.linear_q = Linear(c_m, 32)
        self.linear_k = Linear(c_m, 32)
        self.linear_o = Linear(32 * 32, c_z, init="final")
        self.norm_out = RMSNorm(c_z, eps)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """m: [N_msa, N_token, c_m] -> update [N_token, N_token, c_z]"""
        n = m.shape[1]
        m = self.norm_in(m)
        outer = torch.einsum("bic,bjd->ijcd", self.linear_q(m), self.linear_k(m)).reshape(n, n, -1)
        return self.norm_out(self.linear_o(outer))
