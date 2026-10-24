"""The size and settings of the single-track network. It reads the inputs of Pomelo (PhysDock.pomelo.embedders.ConditionEmbedder: mult_a / mult_s set the scalar
conditions there, only their 0e and 0o are used), so the config is Pomelo's plus the widths of the one track of the atoms and of the tokens."""
import dataclasses
from dataclasses import dataclass

from PhysDock.pomelo.configs import Config as PomeloConfig


@dataclass
class Config(PomeloConfig):
    c_a: int = 256                  # the atom track (Pomelo mini: 128 0e + 32 0o + 32 1o + 16 1e = 304 numbers per atom)
    c_s: int = 576                  # the token track (Pomelo mini: 512 + 64 + 3 * 64 + 3 * 32 = 864)
    heads_atom: int = 8
    heads_token: int = 16
    distance_bias: bool = True      # - w_h(t) |x_i - x_j|^2 in the logits, as Pomelo
    use_template: bool = False      # SingleTrackFold: the self-template xyz added to the tokens (on in the fold presets; SimpleFold has none)


PRESETS = {
    "mini": Config(),               # ~ the parameters of Pomelo mini
    "tiny": Config(mult_a={"0e": 64, "0o": 16, "1o": 16, "1e": 8}, mult_s={"0e": 128, "0o": 32, "1o": 32, "1e": 16}, c_a=64, c_s=128, heads_atom=4, heads_token=4,
                   n_atom_blocks=1, n_token_blocks=2, t_dim=128, rope_res=6, rope_chain=2),
}
# SingleTrackSimple (PhysDock.singletrack.simple): SimpleFold's plain attention, no distance term (mult_a / mult_s are not used)
PRESETS["simple_mini"] = dataclasses.replace(PRESETS["mini"], distance_bias=False, ligand_graph=False, chain_codes=0)    # (not in the comparison: as before)
PRESETS["simple_tiny"] = dataclasses.replace(PRESETS["tiny"], distance_bias=False, ligand_graph=False, chain_codes=0)
# SingleTrackFold (PhysDock.singletrack.fold): SimpleFold's inputs, plain attention with axial RoPE over every channel
PRESETS["fold_mini"] = dataclasses.replace(PRESETS["mini"], distance_bias=False, use_template=True, t_dim=192)    # t_dim 192: ~70M (256: 73.1M)
PRESETS["fold_tiny"] = dataclasses.replace(PRESETS["tiny"], distance_bias=False, use_template=True)


def preset(name_or_dict) -> Config:
    """a preset by name, or a Config from a dict (the overrides of a preset: {"preset": "mini", "n_token_blocks": 8, ...})"""
    if isinstance(name_or_dict, Config):
        return name_or_dict
    if isinstance(name_or_dict, str):
        return dataclasses.replace(PRESETS[name_or_dict])
    d = dict(name_or_dict)
    return dataclasses.replace(PRESETS[d.pop("preset", "mini")], **d)
