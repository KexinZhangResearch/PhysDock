"""PhysDock model: encoder (MSATemplEncoder), decoder (AllAtomDecoder), distogram head and diffuser.

Coordinates are normalised here and only here: the diffused structure is x_gt / sigma_data, diffusion, sampling and
the coordinate losses work on normalised coordinates, and sampled structures are scaled back by sigma_data. The input
features (ref_pos, distances in the features) stay in Angstrom: the encoder was trained on them.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from PhysDock.models.decoders import AllAtomDecoder
from PhysDock.models.diffusion import build_diffuser, replace
from PhysDock.models.encoders import MSATemplEncoder
from PhysDock.models.guidance import LigandGuidance
from PhysDock.models.primitives import Linear, set_attention_backend, set_ffn_backend, set_triangle_update_backend
from PhysDock.models.primitives.checkpoint_keys import upgrade_legacy_keys


def _upgrade_legacy_state_dict(state_dict, prefix, *args) -> None:
    upgrade_legacy_keys(state_dict, prefix)


class PhysDock(nn.Module):
    """Built from the model config (configs/model/*.yaml; c_z, sigma_data, distogram_bins and the channels from
    configs/model/global.yaml). Encoder and decoder keep the checkpoint names
    ``diffusion_conditioning`` and ``dit``.

    Args:
        c_z: pair channels (input of the distogram head)
        num_augmentation_sample: noisy samples per structure in training
        encoder, decoder: MSATemplEncoder / AllAtomDecoder arguments
        diffusion: diffusion settings (see models/diffusion.py), in normalised units
        sigma_data: data scale (Angstrom); x_normalised = x_Angstrom / sigma_data
        distogram_bins: distogram head bins
        attention_backend: "sdpa" (torch) or "triton" (PhysDock.models.kernels) for the triangle and DiT attention
        triangle_update_backend: "torch" or "triton" (PhysDock.models.kernels.triangle_update) for the triangle updates
        ffn_backend: "torch" or "triton" (PhysDock.models.kernels.*swiglu_ffn) for the SwiGLU feed-forwards (transitions)
    """

    def __init__(self, c_z: int, num_augmentation_sample: int, encoder, decoder, diffusion, sigma_data: float,
                 distogram_bins: int, attention_backend: str, triangle_update_backend: str, ffn_backend: str = "torch") -> None:
        super().__init__()
        self.num_augmentation_sample = num_augmentation_sample
        self.sigma_data = sigma_data
        self.diffuser = build_diffuser(diffusion)
        self.diffusion_conditioning = MSATemplEncoder(**encoder)
        self.dit = AllAtomDecoder(**decoder)
        self.linear_distogram = Linear(c_z, distogram_bins, init="final")
        set_attention_backend(self, attention_backend)
        set_triangle_update_backend(self, triangle_update_backend)
        set_ffn_backend(self, ffn_backend)
        self._register_load_state_dict_pre_hook(_upgrade_legacy_state_dict)     # every load accepts the old layout

    @classmethod
    def from_config(cls, cfg) -> "PhysDock":
        """Model from a model config (DictConfig of configs/model/*.yaml or a plain dict)."""
        from PhysDock.configs import to_dict
        return cls(**to_dict(cfg))

    def forward(self, batch):
        """Training step on one (padded) structure.

        Returns (coordinates and noise levels normalised by sigma_data):
            x_gt [N_atom, 3] target, x_denoised [S, N_atom, 3], x_hat [S, N_atom, 3] noisy input, t_hat [S] noise level,
            p_distogram [N_token, N_token, 39]; S = num_augmentation_sample
        """
        x_gt = batch["x_gt"] / self.sigma_data
        conds = self.diffusion_conditioning(batch)
        z, t = self.diffuser.noising(x_gt, batch["atom_mask"], self.num_augmentation_sample)
        x_denoised = self.diffuser.denoise(self.dit, batch, z, t, self.dit.condition(batch, conds))
        return {"x_gt": x_gt, "x_denoised": x_denoised, "x_hat": z, "t_hat": self.diffuser.noise_level(t),
                "p_distogram": self.distogram_logits(conds.z)}

    def distogram_logits(self, z: torch.Tensor) -> torch.Tensor:
        """z: [N_token, N_token, c_z] -> symmetric logits [N_token, N_token, bins], in float32.

        The released head adds a shift of ~1e3 shared by all bins (bin differences ~30); the softmax ignores it, but
        it cancels catastrophically in low precision (TF32 / bf16 log-prob errors 0.15 / 2). The shift is removed
        from weight and bias first: the same distribution and gradients, ~15x smaller rounding errors.
        """
        w, b = self.linear_distogram.weight, self.linear_distogram.bias
        with torch.autocast(z.device.type, enabled=False):
            logits = F.linear(z.float(), w - w.mean(dim=0), b - b.mean())
        return logits + logits.transpose(-2, -3)

    @torch.no_grad()
    def sample_diffusion(self, batch, num_sample: int = 5, steps: int = 200, gamma_0: float = None,
                         gamma_min: float = None, noise_scale_lambda: float = None, step_scale_eta: float = None,
                         ode_step_scale_eta: float = None, ref_mol: Chem.Mol = None, ref_mol_poses: torch.Tensor = None,
                         use_ref_mol_poses: bool = False, mmff_gamma_0_factor: float = 1.0, mmff_iters: int = 5,
                         align_ref_pos: bool = True, karras_noise_schedule_power: float = None) -> torch.Tensor:
        """Sample structures with optional ligand guidance (see LigandGuidance).

        Args:
            batch: features of one structure
            num_sample, steps: number of structures, denoising steps
            gamma_0, gamma_min, noise_scale_lambda, step_scale_eta, ode_step_scale_eta: sampler overrides (None = config;
                gamma_min in units of sigma_data)
            ref_mol, ref_mol_poses ([C, N_lig, 3] Angstrom), use_ref_mol_poses, mmff_gamma_0_factor, mmff_iters,
                align_ref_pos: guidance
            karras_noise_schedule_power: override rho of the noise schedule

        Returns:
            [num_sample, N_atom, 3] Angstrom
        """
        conds = self.dit.condition(batch, self.diffusion_conditioning(batch))     # once for all steps
        sampler = replace(self.diffuser.sampler, gamma_0=gamma_0, gamma_min=gamma_min, noise_scale=noise_scale_lambda,
                          step_scale=step_scale_eta, ode_step_scale=ode_step_scale_eta)
        schedule = replace(self.diffuser.noise_schedule, rho=karras_noise_schedule_power)
        guidance = None
        if ref_mol is not None or ref_mol_poses is not None or use_ref_mol_poses or align_ref_pos:
            guidance = LigandGuidance(batch, num_sample, sampler.gamma_min, self.sigma_data, ref_mol=ref_mol,
                                      ref_mol_poses=ref_mol_poses, use_ref_mol_poses=use_ref_mol_poses,
                                      align_ref_pos=align_ref_pos, mmff_gamma_0_factor=mmff_gamma_0_factor,
                                      mmff_iters=mmff_iters)
        x = self.diffuser.sample(self.dit, batch, conds, batch["atom_mask"], num_sample, steps, sampler=sampler,
                                 noise_schedule=schedule, guidance=guidance)
        return x * self.sigma_data
