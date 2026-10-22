"""Model size and settings of Pomelo (the counterpart of the model configs of PhysDock)."""
from dataclasses import dataclass

from pomelo.models.primitives.irreps import Mult


@dataclass
class Config:
    mult_a: Mult = None          # atom stream
    mult_s: Mult = None          # unit stream (the stored features of the trunk)
    mult_ap: Mult = None         # atom pair features read by the atom attention (0e, 0o)
    mult_z: Mult = None          # unit pair features (0e, 0o)
    heads: int = 4
    n_atom_blocks: int = 1       # atom blocks before and after the unit blocks
    n_unit_blocks: int = 2
    t_dim: int = 64
    atom_static: int = 32
    pair_static: int = 16
    product_channels: int = 8
    relpos: bool = False         # residue relative positions in the unit pair features (proteins)
    max_unit_size: int = 8       # largest unit (atoms) with its own size embedding

    def __post_init__(self):
        self.mult_a = self.mult_a or {"0e": 64, "0o": 16, "1o": 16, "1e": 8}
        self.mult_s = self.mult_s or {"0e": 128, "0o": 32, "1o": 32, "1e": 16}
        self.mult_ap = self.mult_ap or {"0e": 16, "0o": 8}
        self.mult_z = self.mult_z or {"0e": 32, "0o": 8}
