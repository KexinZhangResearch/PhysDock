"""The models of Pomelo: an encoder (the six stored features of the units) + the multi-scale decoder (pomelo.models.decoders), wrapped in the EDM preconditioning.

Denoiser         Embedder (static features: graph, stereo labels as input) + PomeloDiT
TrunkDenoiser    EncoderTrunk (reference conformers in, no label read) + PomeloDiT

D(mol, x_t [B, Na, 3], sigma [B]) -> [B, Na, 3] in coordinates normalised by sigma_data (16 Angstrom). The noisy structure is centred, the network sees  c_in * (x_t - mean),
the denoised structure is  mean + c_skip * x_c + c_out * F(...). The decoder is vmapped over the B noisy samples of a structure (the static features are shared).
"""
import torch
import torch.nn as nn

from pomelo.configs import Config
from pomelo.models.decoders import PomeloDiT
from pomelo.models.encoders import Embedder, EncoderTrunk


def edm_denoise(dit: PomeloDiT, st, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    xbar = x_t.mean(1, keepdim=True)
    xc = x_t - xbar
    s = sigma[:, None, None]
    c_in, c_skip, c_out = (s ** 2 + 1) ** -0.5, 1 / (1 + s ** 2), s * (s ** 2 + 1) ** -0.5
    raw = torch.vmap(lambda x, c: dit(x, c, st))(c_in * xc, torch.log(sigma) / 4)
    return xbar + c_skip * xc + c_out * raw


class Denoiser(nn.Module):
    """EDM denoiser D(x_t, sigma) of a molecule: embedder + PomeloDiT, vmapped over the noisy samples; the stereo labels of the molecule are an input"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embedder = Embedder(cfg.mult_s, cfg.mult_z, cfg.atom_static, cfg.pair_static, cfg.relpos, cfg.max_unit_size)
        self.dit = PomeloDiT(cfg)

    def forward(self, mol, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x_t [B, Na, 3] (normalised, with noise of standard deviation sigma [B]) -> the denoised coordinates [B, Na, 3]"""
        return edm_denoise(self.dit, self.embedder(mol), x_t, sigma)


class TrunkDenoiser(nn.Module):
    """EncoderTrunk (from the reference conformers) + PomeloDiT: the stereo labels and the static unit features are replaced by what the encoder trunk makes of the reference positions.
    The structure is in one frame, the references are in a frame per conformer."""

    def __init__(self, cfg: Config, n_atom_blocks: int = 1, n_blocks: int = 2) -> None:
        super().__init__()
        self.encoder = EncoderTrunk(cfg, n_atom_blocks, n_blocks)
        self.dit = PomeloDiT(cfg)

    def forward(self, mol, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return edm_denoise(self.dit, self.encoder.condition(self.encoder(mol)), x_t, sigma)
