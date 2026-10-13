"""Decoder of PhysDock: the multi-scale all-atom network of the paper's diffusion module."""
from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.data import TensorDict
from PhysDock.models.encoders import Conditions
from PhysDock.models.primitives import DiT, LayerNorm, Linear, TimestepEmbeddings
from PhysDock.utils.tensor_utils import segment_mean_pool, segment_unpool_1d

__all__ = ["AllAtomDecoder"]


class AllAtomDecoder(nn.Module):
    """Multi-scale all-atom network: this is the *DiffusionModule* of the paper (AlphaFold 3 Algorithm 20).

    The noisy atom coordinates are embedded per atom and refined by an atom-level DiT encoder; the atoms of every
    token are average-pooled into one token vector, which a token-level DiT processes conditioned on the single
    (``s``) and pair (``z``) representation; the result is broadcast back to the atoms and refined by an atom-level
    DiT decoder, which finally predicts a 3-D vector per atom. The three scales (atom, token, atom) are why the
    module is called multi-scale.

    Compared with the paper's DiffusionModule, the pre- and post-processing around the network is **not** part of
    this class: scaling the noisy coordinates by ``c_in``, computing the noise-conditioning input and converting
    the network output into the denoised structure (``c_skip * x + c_out * output``) are done by the
    :class:`~PhysDock.models.diffusion.Diffuser` (PhysDock2 style). This module is the bare network, so its output
    is the raw prediction ``F`` of the EDM parameterisation.

    Args:
        c_a: Atom channel size.
        c_ap: Atom-pair channel size.
        c_s: Token channel size.
        c_z: Token-pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks_atom: Blocks in each of the atom-level DiT encoder and decoder.
        no_blocks_dit: Blocks in the token-level DiT.
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
            unpool_method: Literal["index", "segment"] = "index",
    ) -> None:
        super().__init__()
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

    def forward(
            self,
            batch: TensorDict,
            x_in: torch.Tensor,
            t_in: torch.Tensor,
            conds: Conditions,
    ) -> torch.Tensor:
        """Predict the raw network output for preconditioned noisy coordinates.

        Args:
            batch: Feature dict; uses ``ap_mask`` ``[N_atom, N_atom]``, ``z_mask`` ``[N_token, N_token]``,
                ``token_id_to_chunk_sizes`` ``[N_token]`` and ``atom_id_to_token_id`` ``[N_atom]``.
            x_in: Preconditioned noisy coordinates, shape ``[B, N_atom, 3]``.
            t_in: Noise-conditioning input of the timestep embedder, shape ``[B]``.
            conds: Output of the encoder: atom ``a`` ``[N_atom, c_a]``, atom-pair ``ap`` ``[N_atom, N_atom, c_ap]``,
                single ``s`` ``[N_token, c_s]`` and pair ``z`` ``[N_token, N_token, c_z]`` conditioning.

        Returns:
            Raw network output ``F``, shape ``[B, N_atom, 3]``.
        """
        ap_mask = batch["ap_mask"]
        z_mask = batch["z_mask"]
        token_id_to_chunk_sizes = batch["token_id_to_chunk_sizes"]
        atom_id_to_token_id = batch["atom_id_to_token_id"]
        a, ap, s, z = conds

        ba = self.linear_x(x_in) + a[None]
        t = self.time_embedder(t_in)

        ba = self.atom_dit_encoder(ba, ap, t, ap_mask, None)
        bs = self.downscale(ba, s, token_id_to_chunk_sizes)

        bs = self.token_dit(bs, z, t, z_mask, None)

        ba = self.upscale(ba, bs, atom_id_to_token_id, token_id_to_chunk_sizes)

        ba = self.atom_dit_decoder(ba, ap, t, ap_mask, None)

        return self.linear_r(self.norm_r(ba))
