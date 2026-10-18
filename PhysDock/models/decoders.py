"""Decoder (the paper's DiffusionModule): atom DiT -> pool to tokens -> token DiT -> broadcast to atoms -> atom DiT."""
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.kernels.shared_bias_attention import radius_for_noise
from PhysDock.models.primitives import DiT, Linear, TimestepEmbeddings
from PhysDock.utils.tensor_utils import pair_mask, segment_mean_pool


class DecoderConditions(NamedTuple):
    """Decoder inputs that depend on the structure only: a [N_atom, c_a], s [N_token, c_s] and the attention logit
    biases (pair bias + mask) of every DiT block, atom_bias_* [no_blocks_atom, H_atom, N_atom, N_atom] and
    token_bias [no_blocks_dit, H_token, N_token, N_token]; atom_valid and token_valid: the lengths of the real part of the padded atoms and
    tokens (valid_length, only with the Triton backend, which skips the padding in the attentions; else None)."""
    a: torch.Tensor
    s: torch.Tensor
    atom_bias_encoder: torch.Tensor
    token_bias: torch.Tensor
    atom_bias_decoder: torch.Tensor
    atom_valid: Optional[torch.Tensor] = None
    token_valid: Optional[torch.Tensor] = None


class AllAtomDecoder(nn.Module):
    """Multi-scale all-atom network. Pre-/post-conditioning (c_in, c_skip, c_out) is done by the diffuser.

    Always runs in float32 (TF32 matmuls per torch.set_float32_matmul_precision), also under bf16 autocast: the
    token stream is large (rms ~3e3, from the encoder's s) and is added back to the atom stream, so bf16 rounding
    there shows up directly in the coordinates (ligand bond errors x2.5 in sampling).

    atom_decay (default None: plain attention) multiplies the attention of distant atoms in the two atom DiTs by a switching factor, as a force
    field cuts off an interaction (PhysDock.models.primitives.SharedBiasAttention): with {"radius": 12.0, "strength": 10.0} the weight of a pair
    is multiplied by 1.0 inside 12 A, 0.94 at 13 A, 0.75 at 14 A and 0.54 at 15 A (a flat-bottom restraint: nothing changes inside the radius).
    The distances are those of the decoder's own input
    x_in = (x + t e) / sqrt(1 + t^2), whose scale is 1 at every noise level, and the radius follows the noise with
    radius_for_noise, so the factor keeps its meaning throughout the denoising and switches itself off at high noise (every pair then has the
    same distance and the softmax ignores a constant). "learnable" makes the strength a parameter per head (12 scalars); otherwise it is fixed.
    The token DiT is unchanged and the positions receive no gradient. This changes the function: train with it from the start.
    """

    def __init__(self, c_a: int, c_ap: int, c_s: int, c_z: int, inf: float, eps: float, no_blocks_atom: int,
                 no_blocks_dit: int, sigma_data: float = 16.0, atom_decay: Optional[dict] = None) -> None:
        super().__init__()
        self.sigma_data, self.atom_decay = sigma_data, atom_decay
        self.linear_x = Linear(3, c_a)
        self.linear_downscale = Linear(c_a, c_s)
        self.linear_upscale = Linear(c_s, c_a)
        self.time_embedder = TimestepEmbeddings()
        self.atom_dit_encoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)
        self.token_dit = DiT(c_s, c_z, inf, eps, no_blocks_dit)
        self.atom_dit_decoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)
        self.norm_r = nn.LayerNorm(c_a, eps)
        self.linear_r = Linear(c_a, 3, bias=False)
        if atom_decay is not None:
            unknown = set(atom_decay) - {"radius", "strength", "learnable"}
            if unknown:
                raise ValueError(f"atom_decay takes radius, strength and learnable, got also {sorted(unknown)}")
            for stack in (self.atom_dit_encoder, self.atom_dit_decoder):
                stack.set_distance_decay(float(atom_decay.get("strength", 10.0)), bool(atom_decay.get("learnable", False)))

    def condition(self, batch, conds) -> DecoderConditions:
        """Per-structure decoder inputs from the encoder output (computed once, reused by every noisy sample and
        denoising step), in float32.

        Args:
            batch: atom_mask [N_atom], token_mask [N_token]
            conds: encoder output (a, ap, s, z)
        """
        with torch.autocast(batch["atom_mask"].device.type, enabled=False):
            a, ap, s, z = (c.float() for c in conds)
            ap_mask, z_mask = pair_mask(batch["atom_mask"].float()), pair_mask(batch["token_mask"].float())
            attend = self.token_dit.blocks[0].attention.attend
            return DecoderConditions(a, s, self.atom_dit_encoder.pair_bias(ap, ap_mask),
                                     self.token_dit.pair_bias(z, z_mask), self.atom_dit_decoder.pair_bias(ap, ap_mask),
                                     attend.valid_length(ap_mask), attend.valid_length(z_mask))

    def forward(self, batch, x_in: torch.Tensor, t_in: torch.Tensor, cond: DecoderConditions) -> torch.Tensor:
        """
        Args:
            batch: split_sizes [N_token], atom_to_token [N_atom]
            x_in: [B, N_atom, 3] preconditioned noisy coordinates
            t_in: [B] noise-conditioning input
            cond: condition(batch, encoder output)

        Returns:
            [B, N_atom, 3] raw network output
        """
        with torch.autocast(x_in.device.type, enabled=False):
            pos = radius = None
            if self.atom_decay is not None:                      # the decay of the atom attentions: its own input, radius following the noise
                pos = x_in.float()
                radius = radius_for_noise(t_in.float(), float(self.atom_decay.get("radius", 12.0)), self.sigma_data)
            t = self.time_embedder(t_in.float())
            x = self.linear_x(x_in.float()) + cond.a[None]
            x = self.atom_dit_encoder(x, cond.atom_bias_encoder, t, cond.atom_valid, pos, radius)
            xs = segment_mean_pool(F.silu(self.linear_downscale(x)), batch["split_sizes"]) + cond.s[None]   # to tokens
            xs = self.token_dit(xs, cond.token_bias, t, cond.token_valid)
            x = x + self.linear_upscale(xs)[:, batch["atom_to_token"]]                                     # to atoms
            x = self.atom_dit_decoder(x, cond.atom_bias_decoder, t, cond.atom_valid, pos, radius)
            return self.linear_r(self.norm_r(x))
