"""PhysDock: an all-atom diffusion model for protein-ligand complexes.

The model is split along the structure of the paper's diffusion model:

* :class:`~PhysDock.models.encoders.MSATemplEncoder` (the paper's *DiffusionConditioning*): trunk that turns MSA,
  templates and features into the conditioning; evaluated once per structure.
* :class:`~PhysDock.models.decoders.AllAtomDecoder` (the paper's *DiffusionModule*): the multi-scale all-atom network.
* :class:`~PhysDock.models.diffusion.Diffuser` (EDM-VE or linear flow matching, ``config.diffusion.type``): everything
  around the network, i.e. noising, preconditioning, conversion of the network output into a structure and the sampler.
* :class:`~PhysDock.models.guidance.LigandGuidance`: optional physics guidance of the ligand while sampling.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.rdBase import DisableLog

from PhysDock.data import TensorDict
from PhysDock.models.decoders import AllAtomDecoder
from PhysDock.models.diffusion import build_diffuser
from PhysDock.models.encoders import MSATemplEncoder
from PhysDock.models.guidance import LigandGuidance
from PhysDock.models.primitives import Linear

torch.set_float32_matmul_precision("high")
DisableLog('rdApp.*')


class PhysDock(nn.Module):
    """Encoder, decoder, distogram head and diffuser.

    The encoder and the decoder are stored as ``diffusion_conditioning`` and ``dit`` so that the parameter names of
    released checkpoints stay valid; they are also reachable as :attr:`encoder` and :attr:`decoder`.

    Args:
        config: Configuration from :func:`~PhysDock.configs.PhysDockConfig` with ``model.encoder``,
            ``model.decoder``, ``model.c_z``, ``model.num_augmentation_sample`` and the ``diffusion`` section.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.num_augmentation_sample = config.model.num_augmentation_sample
        self.diffuser = build_diffuser(config.diffusion)
        self.diffusion_conditioning = MSATemplEncoder(**config.model.encoder)
        self.dit = AllAtomDecoder(**config.model.decoder)
        self.linear_distogram = Linear(config.model.c_z, 39, init="final")

    @property
    def encoder(self) -> MSATemplEncoder:
        """The :class:`MSATemplEncoder` (stored as ``diffusion_conditioning``)."""
        return self.diffusion_conditioning

    @property
    def decoder(self) -> AllAtomDecoder:
        """The :class:`AllAtomDecoder` network (stored as ``dit``)."""
        return self.dit

    def forward(self, batch: TensorDict) -> Dict[str, torch.Tensor]:
        """Training forward pass: noise the ground truth at random noise levels and denoise it.

        Args:
            batch: Feature dict of one structure (padded to the crop sizes).

        Returns:
            Dict with ``x_denoised`` (predicted clean structures ``[S, N_atom, 3]``), ``x_hat`` (their noisy inputs),
            ``t_hat`` (noise levels in Angstrom ``[S]``, the ``sigma_data * sqrt(NSR)`` that the loss weighting needs,
            for every diffusion type) and ``p_distogram`` (logits ``[N_token, N_token, 39]``),
            ``S = num_augmentation_sample``.
        """
        conds = self.diffusion_conditioning(batch)
        z_t, t = self.diffuser.noising(batch["x_gt"], batch["x_exists"], self.num_augmentation_sample)
        x_denoised = self.diffuser.denoise(self.dit, batch, z_t, t, conds)

        p_distogram = self.linear_distogram(conds.z)
        p_distogram = p_distogram + p_distogram.transpose(-2, -3)

        return {
            "x_denoised": x_denoised,
            "x_hat": z_t,
            "t_hat": self.diffuser.noise_level(t),
            "p_distogram": p_distogram,
        }

    @torch.no_grad()
    def sample_diffusion(
            self,
            batch: TensorDict,
            num_sample: int = 5,
            steps: int = 200,
            gamma_0: Optional[float] = None,
            gamma_min: Optional[float] = None,
            noise_scale_lambda: Optional[float] = None,
            step_scale_eta: Optional[float] = None,
            ode_step_scale_eta: Optional[float] = None,
            ref_mol: Optional[Chem.Mol] = None,
            ref_mol_poses: Optional[torch.Tensor] = None,
            use_ref_mol_poses: bool = False,
            mmff_gamma_0_factor: float = 1.0,
            mmff_iters: int = 5,
            align_ref_pos: bool = True,
            karras_noise_schedule_power: Optional[float] = None,
    ) -> torch.Tensor:
        """Sample structures, optionally with ligand physics guidance (:class:`LigandGuidance`).

        Arguments that are ``None`` take the value of ``config.diffusion`` (for the released EDM-VE configuration:
        ``gamma_0=0.8``, ``gamma_min=1``, ``noise_scale_lambda=1.003``, ``step_scale_eta=1.5``,
        ``ode_step_scale_eta=1``, ``karras_noise_schedule_power=7``).

        Args:
            batch: Feature dict of one structure.
            num_sample: Number of structures to sample.
            steps: Number of denoising steps (at least 2).
            gamma_0: Churn: the noise level is raised by the factor ``1 + gamma_0`` while it exceeds ``gamma_min``;
                0 switches the churn off.
            gamma_min: Noise level (Angstrom) below which the churn is off.
            noise_scale_lambda: Factor on the noise added by the churn.
            step_scale_eta: Euler step scale while the noise level exceeds ``gamma_min``.
            ode_step_scale_eta: Euler step scale once the noise level is at or below ``gamma_min``.
            ref_mol: RDKit molecule of the ligand; enables the low-noise MMFF guidance.
            ref_mol_poses: Template conformers ``[C, N_lig, 3]`` for the high-noise guidance.
            use_ref_mol_poses: Without templates, embed RDKit conformers of ``ref_mol`` and use them.
            mmff_gamma_0_factor: The two guidance regimes switch at the noise level ``gamma_min * mmff_gamma_0_factor``.
            mmff_iters: MMFF iterations per step.
            align_ref_pos: Enable the high-noise guidance (alignment to a reference conformer).
            karras_noise_schedule_power: Exponent ``rho`` of the Karras noise schedule.

        Returns:
            Coordinates in Angstrom, shape ``[num_sample, N_atom, 3]``.
        """
        conds = self.diffusion_conditioning(batch)
        overrides = {"gamma_0": gamma_0, "gamma_min": gamma_min, "noise_scale": noise_scale_lambda,
                     "step_scale": step_scale_eta, "ode_step_scale": ode_step_scale_eta}
        sampler = dataclasses.replace(self.diffuser.sampler, **{k: v for k, v in overrides.items() if v is not None})
        schedule = (self.diffuser.noise_schedule if karras_noise_schedule_power is None
                    else self.diffuser.with_noise_schedule(rho=karras_noise_schedule_power))
        guidance = None
        if ref_mol is not None or ref_mol_poses is not None or use_ref_mol_poses or align_ref_pos:
            guidance = LigandGuidance(
                batch, num_sample, sampler.gamma_min, ref_mol=ref_mol, ref_mol_poses=ref_mol_poses,
                use_ref_mol_poses=use_ref_mol_poses, align_ref_pos=align_ref_pos,
                mmff_gamma_0_factor=mmff_gamma_0_factor, mmff_iters=mmff_iters)
        return self.diffuser.sample(
            self.dit, batch, conds, batch["a_mask"], num_sample, steps, sampler=sampler,
            noise_schedule=schedule, guidance=guidance)
