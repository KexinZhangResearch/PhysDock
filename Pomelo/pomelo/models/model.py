"""The model of Pomelo, as PhysDock's: the encoder (pomelo.models.encoders.EncoderTrunk: the conformer features in, the Pairformer on the tokens) + the multi-scale decoder
(pomelo.models.decoders.PomeloDiT, conditioned on the 12 outputs of the encoder), wrapped in the EDM preconditioning.

D(mol, x_t [B, Na, 3], sigma [B]) -> [B, Na, 3] in coordinates normalised by sigma_data (16 Angstrom). The noisy structure is centred, the network sees  c_in * (x_t - mean),
the denoised structure is  mean + c_skip * x_c + c_out * F(...). The encoder runs once per structure, the decoder on the B noisy samples at once (PhysDock's layout).
"""
import torch
import torch.nn as nn

from pomelo.configs import Config
from pomelo.models.decoders import PomeloDiT
from pomelo.models.encoders import EncoderTrunk


def edm_denoise(dit: PomeloDiT, enc, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    xbar = x_t.mean(1, keepdim=True)
    xc = x_t - xbar
    s = sigma[:, None, None]
    c_in, c_skip, c_out = (s ** 2 + 1) ** -0.5, 1 / (1 + s ** 2), s * (s ** 2 + 1) ** -0.5
    raw = dit(c_in * xc, torch.log(sigma) / 4, enc)                     # the decoder takes the batch of noisy samples (shared biases)
    return xbar + c_skip * xc + c_out * raw


class Denoiser(nn.Module):
    """EDM denoiser D(x_t, sigma) of a system: EncoderTrunk + PomeloDiT. No stereo label is read: the hand of every molecule is the hand of its reference conformer."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.encoder = EncoderTrunk(cfg)
        self.dit = PomeloDiT(cfg)

    def forward(self, mol, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x_t [B, Na, 3] (normalised, with noise of standard deviation sigma [B]) -> the denoised coordinates [B, Na, 3]"""
        return edm_denoise(self.dit, self.encoder(mol), x_t, sigma)
