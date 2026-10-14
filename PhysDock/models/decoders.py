"""Decoder (the paper's DiffusionModule): atom DiT -> pool to tokens -> token DiT -> broadcast to atoms -> atom DiT."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import DiT, Linear, TimestepEmbeddings
from PhysDock.utils.tensor_utils import pair_mask, segment_mean_pool


class AllAtomDecoder(nn.Module):
    """Multi-scale all-atom network. Pre-/post-conditioning (c_in, c_skip, c_out) is done by the diffuser."""

    def __init__(self, c_a: int, c_ap: int, c_s: int, c_z: int, inf: float, eps: float, no_blocks_atom: int,
                 no_blocks_dit: int) -> None:
        super().__init__()
        self.linear_x = Linear(3, c_a)
        self.linear_downscale = Linear(c_a, c_s)
        self.linear_upscale = Linear(c_s, c_a)
        self.time_embedder = TimestepEmbeddings()
        self.atom_dit_encoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)
        self.token_dit = DiT(c_s, c_z, inf, eps, no_blocks_dit)
        self.atom_dit_decoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)
        self.norm_r = nn.LayerNorm(c_a, eps)
        self.linear_r = Linear(c_a, 3, bias=False)

    def forward(self, batch, x_in: torch.Tensor, t_in: torch.Tensor, conds) -> torch.Tensor:
        """
        Args:
            batch: atom_mask [N_atom], token_mask [N_token], split_sizes [N_token], atom_to_token [N_atom]
            x_in: [B, N_atom, 3] preconditioned noisy coordinates
            t_in: [B] noise-conditioning input
            conds: encoder output (a, ap, s, z)

        Returns:
            [B, N_atom, 3] raw network output
        """
        a, ap, s, z = conds
        ap_mask = pair_mask(batch["atom_mask"])
        t = self.time_embedder(t_in)

        x = self.linear_x(x_in) + a[None]
        x = self.atom_dit_encoder(x, ap, t, ap_mask)
        xs = segment_mean_pool(F.silu(self.linear_downscale(x)), batch["split_sizes"]) + s[None]   # atoms -> tokens
        xs = self.token_dit(xs, z, t, pair_mask(batch["token_mask"]))
        x = x + self.linear_upscale(xs)[:, batch["atom_to_token"]].float()                         # tokens -> atoms
        x = self.atom_dit_decoder(x, ap, t, ap_mask)
        return self.linear_r(self.norm_r(x))
