import torch
import torch.nn as nn

from .rms_norm import RMSNorm
from .linear import Linear


class OuterProductMean(nn.Module):
    def __init__(self, c_m: int, c_z: int, eps: float):
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
        B, S, _ = m.shape
        m_norm = self.norm_in(m)
        q = self.linear_q(m_norm)
        k = self.linear_k(m_norm)
        outer = torch.einsum("...bic,...bjd->...ijcd", q, k).reshape([S, S, -1])
        outer = self.linear_o(outer)
        outer = self.norm_out(outer)
        return outer.float()
