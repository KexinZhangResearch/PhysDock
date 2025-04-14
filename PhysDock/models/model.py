import copy

import torch

torch.set_float32_matmul_precision("high")
import torch.nn as nn
from typing import Optional, Tuple, Callable
from functools import partial
import ml_collections as mlc

from PhysDock.data import TensorDict
from PhysDock.models.primitives import Linear
from PhysDock.models.layers.diffusion_conditioning import DiffusionConditioning
from PhysDock.models.layers.transformers import AF3DiT
# from PhysDock.models.layers.confidence_module import ConfidenceModule
from PhysDock.utils.tensor_utils import centre_random_augmentation
from PhysDock.utils.tensor_utils import weighted_rigid_align

from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit.rdBase import DisableLog

DisableLog('rdApp.*')


def get_next_step_pos(
        ref_mol,
        current_step_pos: torch.Tensor,
        mmff_iters=5,
) -> torch.Tensor:
    conf = ref_mol.GetConformer()
    device = current_step_pos.device
    dtype = current_step_pos.dtype

    num_samples = len(current_step_pos)
    current_step_pos_cpu = current_step_pos.cpu().tolist()
    next_step_poses = []
    for sample_id in range(num_samples):

        for i in range(conf.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(*current_step_pos_cpu[sample_id][i]))

        AllChem.MMFFOptimizeMolecule(ref_mol, mmffVariant="MMFF94", maxIters=mmff_iters,
                                     ignoreInterfragInteractions=True)
        conf = ref_mol.GetConformer()
        next_step_pos = []
        for i, atom in enumerate(ref_mol.GetAtoms()):
            next_step_pos.append([conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z])
        next_step_poses.append(torch.tensor(next_step_pos, device=device, dtype=dtype))


    return torch.stack(next_step_poses, dim=0)


class PhysDock(nn.Module):
    def __init__(self, config: mlc.ConfigDict):
        super().__init__()
        self.config = config
        self.num_augmentation_sample = self.config.model.num_augmentation_sample
        self.sigma_data = self.config.sigma_data
        self.diffusion_conditioning = DiffusionConditioning(
            **self.config.model.diffusion_conditioning
        )
        self.dit = AF3DiT(
            **self.config.model.dit
        )
        self.linear_distogram = Linear(self.config.model.c_z, 39, init="final")
        # self.confidence_module = ConfidenceModule(**self.config.model.confidence_module)

    def diffuse(
            self,
            x_cur: torch.Tensor,
            t_hat: torch.Tensor,
            t_cur: Optional[int] = None,
            noise_scale_lambda: Optional[float] = None,
    ) -> torch.Tensor:
        noise = torch.normal(
            0, 1, size=x_cur.shape, dtype=x_cur.dtype, device=x_cur.device, requires_grad=False)
        noise_scale_lambda = 1.0 if noise_scale_lambda is None else noise_scale_lambda
        if t_cur is not None:
            ksi = noise_scale_lambda * noise * torch.sqrt(t_hat ** 2 - t_cur ** 2)[..., None, None]
        else:
            ksi = noise * t_hat[..., None, None]
        x_hat = x_cur + ksi  # + Force Field Term
        return x_hat

    def augmentation_diffuse(self, batch: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            x_gt = batch["x_gt"]
            x_exists = batch["x_exists"]
            # EDM initialization of sigma weights
            t_hat = (torch.exp(torch.normal(
                0, 1, (self.num_augmentation_sample,), device=x_gt.device, dtype=x_gt.dtype) * 1.5 - 1.2)
                     * self.sigma_data)
            x_gt_samples = x_gt[None].repeat([self.num_augmentation_sample, 1, 1])
            x_hat = centre_random_augmentation(self.diffuse(x_gt_samples, t_hat), x_exists)
        return x_hat, t_hat

    def forward(self, batch: TensorDict) -> TensorDict:

        a, ap, s, z = self.diffusion_conditioning(batch)

        x_hat, t_hat = self.augmentation_diffuse(batch)

        x_denoised = self.dit(batch, x_hat, t_hat, a, ap, s, z)

        p_distogram = self.linear_distogram(z)
        p_distogram = p_distogram + p_distogram.transpose(-2, -3)

        return {
            "x_denoised": x_denoised,
            "x_hat": x_hat,
            "t_hat": t_hat,
            "p_distogram": p_distogram,
        }

    def karras_noise_schedule(
            self,
            num_steps: int = 200,
            sigma_data: float = 16,  # sigma_scale
            s_max: float = 160,  # sigma_max: maximun noise level TODO: reduce s_mask to 80-100
            s_min: float = 4 * 10e-4,  # sigma_min: minimum noise level
            p: float = 7,  # sampling schedule power controls  EDM: pho
    ) -> torch.Tensor:
        step_indices = torch.arange(num_steps, dtype=torch.float32)
        t_steps = sigma_data * (s_max ** (1 / p) + step_indices / (num_steps - 1) * (
                s_min ** (1 / p) - s_max ** (1 / p))) ** p
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        return t_steps

    def prepare_solver(
            self,
            batch: TensorDict,
            num_sample: int,
            steps: int,
            noise_scale_lambda: float,
            karras_noise_schedule_power: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Callable, Callable]:
        device = batch["ref_pos"].device
        dtype = batch["ref_pos"].dtype
        shape = (*batch["ref_pos"].shape[:-2], num_sample, *batch["x_gt"].shape[-2:])

        # Diffusion Conditioning
        a, ap, s, z = self.diffusion_conditioning(batch)

        # Sigams and Init Pos
        sigmas = self.karras_noise_schedule(num_steps=steps, p=karras_noise_schedule_power).to(device).to(dtype)
        x_next = sigmas[0] * torch.normal(
            0, 1, size=shape, dtype=dtype, device=device, requires_grad=False)

        # Diffuser and Denoiser
        diffuser = partial(self.diffuse, noise_scale_lambda=noise_scale_lambda)
        denoiser = partial(self.dit, batch=batch, a=a, ap=ap, s=s, z=z)

        return x_next, sigmas, diffuser, denoiser

    def sample_diffusion(
            self,
            batch: TensorDict,
            num_sample: int = 5,
            steps: int = 200,
            gamma_0: float = 0.8,
            gamma_min: float = 1.0,
            noise_scale_lambda: float = 1.003,
            step_scale_eta: float = 1.5,
            ode_step_scale_eta=1.0,
            ref_mol=None,
            ref_mol_poses=None,
            use_ref_mol_poses=False,
            mmff_gamma_0_factor=1.0,
            mmff_iters=5,
            align_ref_pos=True,
            karras_noise_schedule_power=7,
    ) -> torch.Tensor:
        with torch.no_grad():
            x_exists = batch["a_mask"]
            device = batch["x_gt"].device
            dtype = batch["x_gt"].dtype
            x_gt = batch["x_gt"]

            is_ligand_atom = batch["is_ligand"][batch["atom_id_to_token_id"]].bool()

            batch_ref_pos = batch["ref_pos"][None].repeat([num_sample, 1, 1])
            if ref_mol_poses is not None:
                ref_mol_poses = ref_mol_poses.to(device)
                ref_mol_poses_dist = torch.norm(ref_mol_poses[:, :, None] - ref_mol_poses[:, None], dim=-1)
            elif use_ref_mol_poses:
                def _get_ref_mol_poses(ref_mol, num_confs=512):
                    mol = copy.deepcopy(ref_mol)
                    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, enforceChirality=True)
                    num_atoms = mol.GetNumAtoms()
                    coordinates = torch.zeros(num_confs, num_atoms, 3)
                    for i, cid in enumerate(cids):
                        conf = mol.GetConformer(cid)
                        for j in range(num_atoms):
                            pos = conf.GetAtomPosition(j)
                            coordinates[i, j, 0] = pos.x
                            coordinates[i, j, 1] = pos.y
                            coordinates[i, j, 2] = pos.z
                    return coordinates

                ref_mol_poses = _get_ref_mol_poses(ref_mol, num_confs=512).to(device)[:,
                                :len(x_gt[is_ligand_atom])]
                ref_mol_poses_dist = torch.norm(ref_mol_poses[:, :, None] - ref_mol_poses[:, None], dim=-1)
            else:
                ref_mol_poses = None
                ref_mol_poses_dist = None

            x_next, sigmas, diffuser, denoiser = self.prepare_solver(
                batch, num_sample, steps, noise_scale_lambda, karras_noise_schedule_power)
            for i, (t_cur, t_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                x_cur = centre_random_augmentation(x_next, x_exists)
                if t_cur > gamma_min:
                    t_hat = torch.full(
                        [num_sample], fill_value=t_cur * (gamma_0 + 1), device=device, dtype=dtype)
                    x_hat = diffuser(x_cur, t_hat, t_cur)  # EDM: x_noisy
                else:
                    t_hat = torch.full(
                        [num_sample], fill_value=t_cur, device=device, dtype=dtype)
                    x_hat = x_cur
                x_denoised = denoiser(x_hat=x_hat, t_hat=t_hat)

                if align_ref_pos and t_cur > gamma_min * mmff_gamma_0_factor:  # 6
                    weights = x_exists * batch["is_ligand"][batch["atom_id_to_token_id"]]

                    # if ref_mol_poses is None:
                    #     ref_pos = batch["ref_pos"]
                    # TODO: DEBUG atoms not equal for ligand atoms and ref_atoms
                    try:
                        if ref_mol_poses is not None:
                            ligand_pos = x_denoised[:, is_ligand_atom]

                            ligand_dist = torch.norm(ligand_pos[:, :, None] - ligand_pos[:, None], dim=-1)

                            delta = (ligand_dist[:, None] - ref_mol_poses_dist[None]).abs()
                            epsilon = 0.25 * (
                                    torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(
                                -2 + delta) + torch.sigmoid(-4 + delta))
                            epsilon = epsilon.mean(dim=[-1, -2])
                            used_inds = torch.argmin(epsilon, dim=-1)
                            batch_ref_pos[:, is_ligand_atom] = ref_mol_poses[used_inds]
                    except:
                        pass

                    ligand_denoised = weighted_rigid_align(x_denoised * x_exists[..., None], batch_ref_pos, weights)

                    d_ligand = (x_hat - ligand_denoised) / t_hat[..., None, None] * weights[None, :, None]

                    d_cur = ((x_hat - x_denoised) / t_hat[..., None, None]) * (
                            1 - weights[None, :, None]) + d_ligand

                elif ref_mol is not None and t_cur <= gamma_min * mmff_gamma_0_factor:
                    weights = x_exists * batch["is_ligand"][batch["atom_id_to_token_id"]]
                    x_ref = copy.deepcopy(x_denoised)
                    x_ref[:, is_ligand_atom] = get_next_step_pos(ref_mol, x_denoised[:, is_ligand_atom],
                                                                 mmff_iters)
                    ligand_denoised = weighted_rigid_align(x_denoised * x_exists[..., None], x_ref, weights)
                    d_ligand = (x_hat - ligand_denoised) / t_hat[..., None, None] * weights[None, :, None]

                    d_cur = ((x_hat - x_denoised) / t_hat[..., None, None]) * (
                            1 - weights[None, :, None]) + d_ligand
                else:
                    d_cur = (x_hat - x_denoised) / t_hat[..., None, None]
                dt = (t_next - t_hat)[..., None, None]

                # if ref_mol is not None:
                # if False:
                #     mmff_step_factor = 1
                #     x_ref = copy.deepcopy(x_denoised)
                #     x_ref[:, is_ligand_atom] = get_next_step_pos(ref_mol, x_denoised[:, is_ligand_atom])
                #     ffstep = x_ref - x_hat
                #
                #     if t_cur > gamma_min:
                #         x_next = x_hat + step_scale_eta * dt * d_cur + ffstep * mmff_step_factor
                #     else:
                #         x_next = x_hat + dt * d_cur + ffstep * mmff_step_factor
                # else:
                if t_cur > gamma_min:
                    x_next = x_hat + step_scale_eta * dt * d_cur  # replace 2nd order correction
                else:
                    x_next = x_hat + ode_step_scale_eta * dt * d_cur
        return x_next
