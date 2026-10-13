"""Decoder of PhysDock: the diffusion transformer that denoises atom coordinates."""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.data import TensorDict
from PhysDock.models.primitives import DiT, LayerNorm, Linear, TimestepEmbeddings
from PhysDock.utils.tensor_utils import segment_mean_pool, segment_unpool_1d

__all__ = ["AF3DiT"]


class AF3DiT(nn.Module):
    """AlphaFold-3-style diffusion transformer that denoises atom coordinates (EDM parameterisation).

    The noisy coordinates ``x_hat`` are embedded per atom and refined by an atom-level DiT encoder; atoms are
    average-pooled into tokens, processed by a token-level DiT conditioned on the single (``s``) and pair
    (``z``) representation, broadcast back to atoms, refined by an atom-level decoder, and finally mapped to a
    coordinate update that is combined with the input through the EDM skip/out scalings.

    Args:
        c_a: Atom channel size.
        c_ap: Atom-pair channel size.
        c_s: Token channel size.
        c_z: Token-pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks_atom: Blocks in each of the atom encoder and decoder.
        no_blocks_dit: Blocks in the token-level DiT.
        sigma_data: Data standard deviation of the EDM preconditioning (Angstrom).
        unpool_method: How token activations are broadcast to atoms: ``"index"`` (gather with
            ``atom_id_to_token_id``, default) or ``"segment"`` (from ``token_id_to_chunk_sizes``), see
            :func:`~PhysDock.utils.tensor_utils.segment_unpool_1d`.
    """

    def __init__(
            self,
            c_a: int,
            c_ap: int,
            c_s: int,
            c_z: int,
            inf: float,
            eps: float,
            no_blocks_atom: int,
            no_blocks_dit: int,
            sigma_data: float,
            unpool_method: Literal["index", "segment"] = "index",
    ) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.unpool_method = unpool_method
        self.linear_x = Linear(3, c_a)
        self.linear_downscale = Linear(c_a, c_s)
        self.linear_upscale = Linear(c_s, c_a)
        self.time_embedder = TimestepEmbeddings()

        self.atom_dit_encoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)
        self.token_dit = DiT(c_s, c_z, inf, eps, no_blocks_dit)
        self.atom_dit_decoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)

        self.norm_r = LayerNorm(c_a, eps)
        self.linear_r = Linear(c_a, 3, bias=False)

    def downscale(self, ba: torch.Tensor, s: torch.Tensor, token_id_to_chunk_sizes: torch.Tensor) -> torch.Tensor:
        """Average-pool atom activations into tokens and add the single representation.

        Args:
            ba: Atom activations, shape ``[B, N_atom, c_a]`` (atoms of one token are contiguous).
            s: Single representation, shape ``[N_token, c_s]``.
            token_id_to_chunk_sizes: Number of atoms of every token, shape ``[N_token]``.

        Returns:
            Token activations, shape ``[B, N_token, c_s]``.
        """
        x = segment_mean_pool(F.silu(self.linear_downscale(ba)), token_id_to_chunk_sizes)
        return x + s[None]

    def upscale(
            self,
            ba: torch.Tensor,
            bs: torch.Tensor,
            atom_id_to_token_id: torch.Tensor,
            token_id_to_chunk_sizes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Broadcast token activations back to their atoms and add them.

        Args:
            ba: Atom activations, shape ``[B, N_atom, c_a]``.
            bs: Token activations, shape ``[B, N_token, c_s]``.
            atom_id_to_token_id: Token index of every atom, shape ``[N_atom]``.
            token_id_to_chunk_sizes: Number of atoms of every token, shape ``[N_token]``; only needed when
                ``unpool_method="segment"``.

        Returns:
            Atom activations, shape ``[B, N_atom, c_a]``.
        """
        up = segment_unpool_1d(self.linear_upscale(bs), token_id_to_chunk_sizes, num_rows=ba.shape[-2], dim=-2,
                               index=atom_id_to_token_id, method=self.unpool_method)
        return ba + up.float()

    def precond(
            self,
            x_hat: torch.Tensor,
            t_hat: torch.Tensor,
            a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """EDM input preconditioning and conditioning embeddings.

        Args:
            x_hat: Noisy coordinates, shape ``[B, N_atom, 3]``.
            t_hat: Noise levels, shape ``[B]``.
            a: Atom conditioning, shape ``[N_atom, c_a]``.

        Returns:
            Tuple ``(ba, t, beta)``: initial atom activations ``[B, N_atom, c_a]``, the timestep embedding
            ``[B, 256]`` (computed from ``t_hat * c_noise``, with ``c_noise = log(t_hat / sigma_data) / 4``),
            and an optional logit bias (always ``None`` here).
        """
        c_in = 1 / (torch.sqrt(t_hat[:, None, None] ** 2 + self.sigma_data ** 2))
        c_noise = torch.log(t_hat / self.sigma_data) / 4.0

        ba = self.linear_x(x_hat * c_in) + a[None]
        t = self.time_embedder(t_hat * c_noise)
        return ba, t, None

    def denoise(self, x_hat: torch.Tensor, t_hat: torch.Tensor, ba: torch.Tensor) -> torch.Tensor:
        """EDM output scaling: ``c_skip * x_hat + c_out * F(ba)``.

        Args:
            x_hat: Noisy coordinates, shape ``[B, N_atom, 3]``.
            t_hat: Noise levels, shape ``[B]``.
            ba: Final atom activations, shape ``[B, N_atom, c_a]``.

        Returns:
            Denoised coordinates, shape ``[B, N_atom, 3]``.
        """
        c_skip = (self.sigma_data ** 2 / (self.sigma_data ** 2 + t_hat ** 2))[:, None, None]
        c_out = (self.sigma_data * t_hat / torch.sqrt(self.sigma_data ** 2 + t_hat ** 2))[:, None, None]
        return c_skip * x_hat + c_out * self.linear_r(self.norm_r(ba))

    def forward(
            self,
            batch: TensorDict,
            x_hat: torch.Tensor,
            t_hat: torch.Tensor,
            a: torch.Tensor,
            ap: torch.Tensor,
            s: torch.Tensor,
            z: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise a batch of noisy structures.

        Args:
            batch: Feature dict; uses ``ap_mask`` ``[N_atom, N_atom]``, ``z_mask`` ``[N_token, N_token]``,
                ``token_id_to_chunk_sizes`` ``[N_token]`` and ``atom_id_to_token_id`` ``[N_atom]``.
            x_hat: Noisy coordinates, shape ``[B, N_atom, 3]``.
            t_hat: Noise levels, shape ``[B]``.
            a: Atom conditioning, shape ``[N_atom, c_a]``.
            ap: Atom-pair conditioning, shape ``[N_atom, N_atom, c_ap]``.
            s: Single conditioning, shape ``[N_token, c_s]``.
            z: Pair conditioning, shape ``[N_token, N_token, c_z]``.

        Returns:
            Denoised coordinates, shape ``[B, N_atom, 3]``.
        """
        ap_mask = batch["ap_mask"]
        z_mask = batch["z_mask"]
        token_id_to_chunk_sizes = batch["token_id_to_chunk_sizes"]
        atom_id_to_token_id = batch["atom_id_to_token_id"]

        ba, t, beta = self.precond(x_hat, t_hat, a)

        ba = self.atom_dit_encoder(ba, ap, t, ap_mask, beta)
        bs = self.downscale(ba, s, token_id_to_chunk_sizes)

        bs = self.token_dit(bs, z, t, z_mask, None)

        ba = self.upscale(ba, bs, atom_id_to_token_id, token_id_to_chunk_sizes)

        ba = self.atom_dit_decoder(ba, ap, t, ap_mask, beta)

        return self.denoise(x_hat, t_hat, ba)
