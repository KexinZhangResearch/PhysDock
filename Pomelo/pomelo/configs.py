"""The size and settings of Pomelo's network. Everything outside the network (sigma_data, the diffusion, the number of noisy samples, the loss, the data) is PhysDock's own
config (configs/model/mini_p2.yaml and configs/train.yaml), used unchanged."""
import dataclasses
from dataclasses import dataclass

from pomelo.models.primitives.irreps import Mult


@dataclass
class Config:
    mult_a: Mult = None             # atom stream (PhysDock c_a 128)
    mult_s: Mult = None             # token stream (PhysDock c_s 512)
    heads_atom: int = 4             # PhysDock: channels / 32
    heads_token: int = 16
    n_atom_blocks: int = 3          # the local atom DiT before and after the token trunk (PhysDock mini_p2: 3 + 3)
    n_token_blocks: int = 12        # the global token trunk (PhysDock mini_p2: Pairformer 8 + token DiT 6; here one trunk, no pair representation)
    t_dim: int = 256                # PhysDock's TimestepEmbeddings
    atom_product_channels: int = 16 # channels of every product of the GeometricProduct of an atom / token block
    token_product_channels: int = 32
    rope_res: int = 12              # RoPE frequencies of the residue index / of the chain index on the 0e channels of a head (q, k)
    rope_chain: int = 4
    centroid_length: float = 16.0   # the input vector x - centroid squashed at this length (units of 4 A)
    direction_length: float = 8.0   # the direction messages squashed at this length (units of 4 A)
    zero_gates: bool = True         # the AdaLN gates start at about 0 (AdaLN-Zero, as PhysDock)

    def __post_init__(self):
        self.mult_a = dict(self.mult_a or {"0e": 128, "0o": 32, "1o": 32, "1e": 16})
        self.mult_s = dict(self.mult_s or {"0e": 512, "0o": 64, "1o": 64, "1e": 32})


PRESETS = {
    "mini": Config(),                                                                                     # the scale of PhysDock mini_p2
    "tiny": Config(mult_a={"0e": 64, "0o": 16, "1o": 16, "1e": 8}, mult_s={"0e": 128, "0o": 32, "1o": 32, "1e": 16}, heads_atom=4, heads_token=4,
                   n_atom_blocks=1, n_token_blocks=2, t_dim=128, atom_product_channels=8, token_product_channels=8, rope_res=6, rope_chain=2),
}


def preset(name_or_dict) -> Config:
    """a preset by name, or a Config from a dict (the overrides of a preset: {"preset": "mini", "n_token_blocks": 8, ...})"""
    if isinstance(name_or_dict, Config):
        return name_or_dict
    if isinstance(name_or_dict, str):
        return dataclasses.replace(PRESETS[name_or_dict])
    d = dict(name_or_dict)
    return dataclasses.replace(PRESETS[d.pop("preset", "mini")], **d)
