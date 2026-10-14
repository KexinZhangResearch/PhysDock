"""PhysDock model: encoder (MSATemplEncoder), decoder (AllAtomDecoder), distogram head and diffuser."""
import torch
import torch.nn as nn
from rdkit import Chem

from PhysDock.models.decoders import AllAtomDecoder
from PhysDock.models.diffusion import build_diffuser, replace
from PhysDock.models.encoders import MSATemplEncoder
from PhysDock.models.guidance import LigandGuidance
from PhysDock.models.primitives import Linear


class PhysDock(nn.Module):
    """Encoder and decoder keep the checkpoint names ``diffusion_conditioning`` and ``dit``."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_augmentation_sample = config.model.num_augmentation_sample
        self.diffuser = build_diffuser(config.diffusion)
        self.diffusion_conditioning = MSATemplEncoder(**config.model.encoder)
        self.dit = AllAtomDecoder(**config.model.decoder)
        self.linear_distogram = Linear(config.model.c_z, 39, init="final")

    def forward(self, batch):
        """Training step on one (padded) structure.

        Returns:
            x_denoised [S, N_atom, 3], x_hat [S, N_atom, 3] noisy input, t_hat [S] noise level (Angstrom),
            p_distogram [N_token, N_token, 39]; S = num_augmentation_sample
        """
        conds = self.diffusion_conditioning(batch)
        z, t = self.diffuser.noising(batch["x_gt"], batch["atom_mask"], self.num_augmentation_sample)
        x_denoised = self.diffuser.denoise(self.dit, batch, z, t, conds)
        p_distogram = self.linear_distogram(conds.z)
        p_distogram = p_distogram + p_distogram.transpose(-2, -3)
        return {"x_denoised": x_denoised, "x_hat": z, "t_hat": self.diffuser.noise_level(t), "p_distogram": p_distogram}

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
            gamma_0, gamma_min, noise_scale_lambda, step_scale_eta, ode_step_scale_eta: sampler overrides (None = config)
            ref_mol, ref_mol_poses, use_ref_mol_poses, mmff_gamma_0_factor, mmff_iters, align_ref_pos: guidance
            karras_noise_schedule_power: override rho of the noise schedule

        Returns:
            [num_sample, N_atom, 3]
        """
        conds = self.diffusion_conditioning(batch)
        sampler = replace(self.diffuser.sampler, gamma_0=gamma_0, gamma_min=gamma_min, noise_scale=noise_scale_lambda,
                          step_scale=step_scale_eta, ode_step_scale=ode_step_scale_eta)
        schedule = replace(self.diffuser.noise_schedule, rho=karras_noise_schedule_power)
        guidance = None
        if ref_mol is not None or ref_mol_poses is not None or use_ref_mol_poses or align_ref_pos:
            guidance = LigandGuidance(batch, num_sample, sampler.gamma_min, ref_mol=ref_mol, ref_mol_poses=ref_mol_poses,
                                      use_ref_mol_poses=use_ref_mol_poses, align_ref_pos=align_ref_pos,
                                      mmff_gamma_0_factor=mmff_gamma_0_factor, mmff_iters=mmff_iters)
        return self.diffuser.sample(self.dit, batch, conds, batch["atom_mask"], num_sample, steps, sampler=sampler,
                                    noise_schedule=schedule, guidance=guidance)
