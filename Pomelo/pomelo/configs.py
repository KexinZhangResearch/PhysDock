"""Model size and settings of Pomelo (the counterpart of the model configs of PhysDock)."""
from dataclasses import dataclass

from pomelo.models.primitives.irreps import Mult


@dataclass
class Config:
    mult_a: Mult = None          # atom stream
    mult_s: Mult = None          # token stream
    mult_ap: Mult = None         # atom pair features read by the atom attention (0e, 0o)
    mult_z: Mult = None          # token pair features (0e, 0o)
    heads_atom: int = 4          # attention heads of the atom and of the token streams (PhysDock: channels / 32)
    heads_token: int = 4
    n_atom_blocks: int = 1       # decoder: atom blocks before and after the token blocks
    n_token_blocks: int = 2      # decoder: token blocks
    n_encoder_atom_blocks: int = 1
    n_template_blocks: int = 1   # encoder: template stack
    n_encoder_blocks: int = 2    # encoder: Pairformer (TrunkBlock) blocks on the tokens
    t_dim: int = 64
    product_channels: int = 8
    distance_logits: bool = False   # decoder: the attention logits get - w_h(t) |x_i - x_j|^2 of the noisy sample (SharedBiasAttention)
    centroid_length: float = 4.0    # decoder: the input vector x - centroid squashed at this length (units of 4 A)
    direction_length: float = 2.0   # decoder: the direction messages squashed at this length (units of 4 A)
    vector_cond_gate: bool = False  # decoder: the encoder vectors moved into the noisy sample are gated per channel by the noise level
    zero_gates: bool = False        # decoder: the AdaLN gates of the DiT blocks start at about 0 (AdaLN-Zero, as PhysDock), not at 1

    def __post_init__(self):
        self.mult_a = self.mult_a or {"0e": 64, "0o": 16, "1o": 16, "1e": 8}
        self.mult_s = self.mult_s or {"0e": 128, "0o": 32, "1o": 32, "1e": 16}
        self.mult_ap = self.mult_ap or {"0e": 16, "0o": 8}
        self.mult_z = self.mult_z or {"0e": 32, "0o": 8}


def mini_p2() -> Config:
    """the scale of PhysDock's configs/model/mini_p2.yaml (the network of the smoke tests of scripts/p2align): atom blocks 3 (encoder) and 3 + 3 (decoder), template 2, Pairformer 8,
    token DiT 6; the even channels of PhysDock (c_a 128, c_ap 16, c_s 512 -> 384 here, c_z 128) and the odd and vector ones beside them, heads of 32 channels"""
    return Config(mult_a={"0e": 128, "0o": 32, "1o": 32, "1e": 16}, mult_ap={"0e": 16, "0o": 8}, mult_s={"0e": 384, "0o": 64, "1o": 64, "1e": 32},
                  mult_z={"0e": 128, "0o": 32}, heads_atom=4, heads_token=16, n_atom_blocks=3, n_token_blocks=6, n_encoder_atom_blocks=3, n_template_blocks=2,
                  n_encoder_blocks=8, t_dim=256)


def tiny() -> Config:
    """a small model for quick tests on the real data: half the widths of mini_p2 and few blocks (encoder: atom 1, template 1, Pairformer 2; decoder: atom 1 + 1, token 2)"""
    return Config(mult_a={"0e": 64, "0o": 16, "1o": 16, "1e": 8}, mult_ap={"0e": 16, "0o": 8}, mult_s={"0e": 128, "0o": 32, "1o": 32, "1e": 16},
                  mult_z={"0e": 64, "0o": 16}, heads_atom=4, heads_token=4, n_atom_blocks=1, n_token_blocks=2, n_encoder_atom_blocks=1, n_template_blocks=1,
                  n_encoder_blocks=2, t_dim=128)


def tiny_dist() -> Config:
    """tiny with the distance term in the attention logits of the decoder, the wider squash lengths (centroid vector 16 = 64 A, direction messages 8 = 32 A), the noise gate of
    the moved encoder vectors and the AdaLN gates of the DiT starting at about 0"""
    import dataclasses
    return dataclasses.replace(tiny(), distance_logits=True, centroid_length=16.0, direction_length=8.0, vector_cond_gate=True, zero_gates=True)
