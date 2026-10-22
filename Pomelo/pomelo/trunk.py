"""One block of the trunk: the pair block, then the single block (the order of PhysDock's current structure: z learns from s, then s from z)."""
from typing import Tuple

import torch
import torch.nn as nn

from pomelo.blocks import SingleBlock
from pomelo.irreps import Feats, Mult
from pomelo.pair import PairBlock


class TrunkBlock(nn.Module):
    def __init__(self, mult_s: Mult, mult_z: Mult, heads_s: int, heads_z: int, channels: int = 16, product_channels: int = 32, length_scale: float = 8.0,
                 zero_init: bool = False) -> None:
        super().__init__()
        self.pair = PairBlock(mult_s, mult_z, heads_z, channels, length_scale, zero_init=zero_init)
        self.single = SingleBlock(mult_s, mult_z, heads_s, product_channels, zero_init=zero_init)

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor) -> Tuple[Feats, Feats]:
        z = self.pair(s, z, centers, mask)
        return self.single(s, z, centers, mask), z
