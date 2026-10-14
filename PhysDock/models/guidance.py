"""Ligand physics guidance: replace the denoised ligand by an aligned reference conformer (high noise) or by an
MMFF-relaxed copy (low noise). Protein atoms are untouched.

The sampler works on normalised coordinates (Angstrom / sigma_data): the Angstrom inputs (ref_pos, template
conformers) are normalised once, MMFF and the template distance score run in Angstrom (their natural units)."""
import copy
import logging

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from PhysDock.utils.tensor_utils import weighted_rigid_align

logger = logging.getLogger(__name__)


def mmff_relax(ref_mol: Chem.Mol, positions: torch.Tensor, max_iters: int = 5) -> torch.Tensor:
    """A few MMFF94 steps per pose (ref_mol's conformer is overwritten).

    positions: [S, N_lig, 3] -> [S, N_lig, 3]
    """
    conf = ref_mol.GetConformer()
    relaxed = []
    for pose in positions.cpu().tolist():
        for i in range(conf.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(*pose[i]))
        AllChem.MMFFOptimizeMolecule(ref_mol, mmffVariant="MMFF94", maxIters=max_iters, ignoreInterfragInteractions=True)
        conf = ref_mol.GetConformer()
        relaxed.append(conf.GetPositions().tolist())
    return torch.tensor(relaxed, device=positions.device, dtype=positions.dtype)


def embed_conformers(ref_mol: Chem.Mol, num_confs: int, num_atoms: int = None) -> torch.Tensor:
    """RDKit conformers with enforced chirality -> [C, N, 3], C <= num_confs (failed embeddings are left out)."""
    mol = copy.deepcopy(ref_mol)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, enforceChirality=True))
    coords = torch.zeros(len(cids), mol.GetNumAtoms(), 3)
    for i, cid in enumerate(cids):
        coords[i] = torch.tensor(mol.GetConformer(cid).GetPositions())
    return coords if num_atoms is None else coords[:, :num_atoms]


class LigandGuidance:
    """Guidance hook of Diffuser.sample; one instance per sampling run.

    Args:
        batch: features; uses atom_mask, is_ligand, atom_to_token, ref_pos (Angstrom)
        num_samples: number of parallel samples S
        gamma_min: sampler churn threshold (units of sigma_data); the regimes switch at gamma_min * mmff_gamma_0_factor
        sigma_data: data scale (Angstrom) of the normalised coordinates
        ref_mol: RDKit ligand (atom order of the ligand atoms); enables MMFF at low noise
        ref_mol_poses: template conformers [C, N_lig, 3] (Angstrom) for the alignment at high noise
        use_ref_mol_poses: without templates, embed 512 RDKit conformers of ref_mol
        align_ref_pos: enable the alignment at high noise
        mmff_gamma_0_factor, mmff_iters: threshold factor and MMFF iterations per step
    """

    def __init__(self, batch, num_samples: int, gamma_min: float, sigma_data: float, ref_mol: Chem.Mol = None,
                 ref_mol_poses: torch.Tensor = None, use_ref_mol_poses: bool = False, align_ref_pos: bool = True,
                 mmff_gamma_0_factor: float = 1.0, mmff_iters: int = 5) -> None:
        self.atom_mask = batch["atom_mask"]
        self.weights = self.atom_mask * batch["is_ligand"][batch["atom_to_token"]]   # real ligand atoms
        self.is_ligand_atom = self.weights.bool()
        self.sigma_data = sigma_data
        self.ref_pos = batch["ref_pos"][None].repeat([num_samples, 1, 1]) / sigma_data
        self.ref_mol = copy.deepcopy(ref_mol) if ref_mol is not None else None
        self.align_ref_pos = align_ref_pos
        self.mmff_iters = mmff_iters
        self.threshold = gamma_min * mmff_gamma_0_factor

        n_lig = int(self.is_ligand_atom.sum())
        templates = None
        if ref_mol_poses is not None:
            templates = ref_mol_poses.to(self.atom_mask.device)
        elif use_ref_mol_poses:
            if ref_mol is None:
                raise ValueError("use_ref_mol_poses=True needs ref_mol (or ref_mol_poses)")
            templates = embed_conformers(ref_mol, 512, n_lig).to(self.atom_mask.device)
        if templates is not None and (len(templates) == 0 or templates.shape[1] != n_lig):
            logger.warning("no usable template conformers (%s for %d ligand atoms); aligning to ref_pos instead",
                           tuple(templates.shape), n_lig)
            templates = None
        self.templates = None if templates is None else templates / sigma_data
        self.templates_dist = None if templates is None else torch.norm(templates[:, :, None] - templates[:, None], dim=-1)

    def _nearest_templates(self, x: torch.Tensor) -> torch.Tensor:
        """x: [S, N_atom, 3] -> per sample the template with the closest internal distances (smooth-lDDT score in
        Angstrom) [S, N_lig, 3]"""
        lig = x[:, self.is_ligand_atom] * self.sigma_data
        delta = (torch.norm(lig[:, :, None] - lig[:, None], dim=-1)[:, None] - self.templates_dist[None]).abs()
        eps = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(-2 + delta)
                      + torch.sigmoid(-4 + delta))
        return self.templates[torch.argmin(eps.mean(dim=[-1, -2]), dim=-1)]

    def __call__(self, x_hat: torch.Tensor, x_denoised: torch.Tensor, level: torch.Tensor):
        """x_hat, x_denoised: [S, N_atom, 3] normalised, level: noise level -> guided x_denoised or None"""
        if not self.is_ligand_atom.any():           # nothing to guide (protein-only structure)
            return None
        if self.align_ref_pos and level > self.threshold:
            if self.templates is not None:
                self.ref_pos[:, self.is_ligand_atom] = self._nearest_templates(x_denoised)
            x_ref = self.ref_pos
        elif self.ref_mol is not None and level <= self.threshold:
            x_ref = x_denoised.clone()
            relaxed = mmff_relax(self.ref_mol, x_denoised[:, self.is_ligand_atom] * self.sigma_data, self.mmff_iters)
            x_ref[:, self.is_ligand_atom] = relaxed / self.sigma_data
        else:
            return None
        aligned = weighted_rigid_align(x_denoised * self.atom_mask[..., None], x_ref, self.weights)
        w = self.weights[None, :, None]
        return x_denoised * (1 - w) + aligned * w
