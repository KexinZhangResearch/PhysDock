"""Ligand physics guidance for the sampler (the physics-guided part of PhysDock).

During reverse diffusion the predicted clean structure of the ligand atoms can be replaced by a physically
sensible conformer; protein atoms are not touched. Because the guidance works on the clean-structure estimate and
the noise level (Angstrom), it does not depend on the diffusion process. Two regimes, split at the noise level
``gamma_min * mmff_gamma_0_factor``:

* **High noise** (``align_ref_pos=True``): the denoised ligand is rigidly aligned onto a reference conformer, the
  closest of a set of templates (RDKit conformers and/or earlier samples) per sample, or the conformer given by
  the input features if there are no templates.
* **Low noise** (a reference molecule is given): the denoised ligand is relaxed by a few MMFF94 steps and the
  relaxed ligand is aligned back.

In both cases the estimate of the ligand atoms becomes the aligned ligand while all other atoms keep ``x_denoised``;
the sampler then derives the update direction ``(x_hat - estimate) / t_hat`` from it.
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from PhysDock.data import TensorDict
from PhysDock.utils.tensor_utils import weighted_rigid_align

__all__ = ["mmff_relax", "embed_conformers", "LigandGuidance"]

logger = logging.getLogger(__name__)


def mmff_relax(ref_mol: Chem.Mol, positions: torch.Tensor, max_iters: int = 5) -> torch.Tensor:
    """Relax ligand poses with a few MMFF94 steps (inter-fragment interactions ignored).

    The conformer of ``ref_mol`` is overwritten for every pose, so ``ref_mol`` is modified in place.

    Args:
        ref_mol: RDKit molecule with a conformer whose atoms are in the order of ``positions``.
        positions: Ligand coordinates, shape ``[S, N_lig, 3]``.
        max_iters: Maximum number of MMFF iterations per pose.

    Returns:
        Relaxed coordinates, shape ``[S, N_lig, 3]``, same dtype and device as ``positions``.
    """
    conf = ref_mol.GetConformer()
    coords = positions.cpu().tolist()
    relaxed = []
    for sample in coords:
        for i in range(conf.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(*sample[i]))
        AllChem.MMFFOptimizeMolecule(ref_mol, mmffVariant="MMFF94", maxIters=max_iters, ignoreInterfragInteractions=True)
        conf = ref_mol.GetConformer()
        relaxed.append(conf.GetPositions().tolist())
    return torch.tensor(relaxed, device=positions.device, dtype=positions.dtype)


def embed_conformers(ref_mol: Chem.Mol, num_confs: int, num_atoms: Optional[int] = None) -> torch.Tensor:
    """Embed random conformers of a molecule with RDKit (chirality enforced).

    Args:
        ref_mol: RDKit molecule.
        num_confs: Number of conformers to request; embedding may return fewer.
        num_atoms: Keep only the first ``num_atoms`` atoms.

    Returns:
        Float32 CPU tensor of shape ``[C, N, 3]``.
    """
    mol = copy.deepcopy(ref_mol)
    cids = AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, enforceChirality=True)
    n = mol.GetNumAtoms()
    coords = torch.zeros(num_confs, n, 3)
    for i, cid in enumerate(cids):
        coords[i] = torch.tensor(mol.GetConformer(cid).GetPositions())
    return coords if num_atoms is None else coords[:, :num_atoms]


class LigandGuidance:
    """Callable guidance for :meth:`~PhysDock.models.diffusion.Diffuser.sample`, see the module docstring.

    One instance is used for one sampling run: it keeps the reference conformer of every sample between steps.

    Args:
        batch: Input features; uses ``ref_pos``, ``a_mask``, ``is_ligand`` and ``atom_id_to_token_id``.
        num_samples: Number of structures sampled in parallel.
        gamma_min: Churn threshold of the sampler (Angstrom); the guidance regimes switch at the noise level
            ``gamma_min * mmff_gamma_0_factor``.
        ref_mol: RDKit molecule of the ligand (atoms in the order of the ligand atoms of ``batch``); enables the
            low-noise MMFF regime. It is copied, the caller's molecule is not modified.
        ref_mol_poses: Template conformers, shape ``[C, N_lig, 3]``, for the high-noise regime.
        use_ref_mol_poses: If no templates are given, embed 512 RDKit conformers of ``ref_mol`` and use them.
        align_ref_pos: Enable the high-noise regime.
        mmff_gamma_0_factor: Scales the noise-level threshold between the two regimes.
        mmff_iters: MMFF iterations per step.

    Raises:
        ValueError: If ``use_ref_mol_poses`` is requested without templates and without ``ref_mol``.
    """

    def __init__(
            self,
            batch: TensorDict,
            num_samples: int,
            gamma_min: float,
            ref_mol: Optional[Chem.Mol] = None,
            ref_mol_poses: Optional[torch.Tensor] = None,
            use_ref_mol_poses: bool = False,
            align_ref_pos: bool = True,
            mmff_gamma_0_factor: float = 1.0,
            mmff_iters: int = 5,
    ) -> None:
        self.x_exists = batch["a_mask"]
        self.is_ligand_atom = batch["is_ligand"][batch["atom_id_to_token_id"]].bool()
        self.weights = self.x_exists * batch["is_ligand"][batch["atom_id_to_token_id"]]
        self.ref_pos = batch["ref_pos"][None].repeat([num_samples, 1, 1])
        self.ref_mol = copy.deepcopy(ref_mol) if ref_mol is not None else None
        self.align_ref_pos = align_ref_pos
        self.mmff_iters = mmff_iters
        self.threshold = gamma_min * mmff_gamma_0_factor
        num_lig_atoms = int(self.is_ligand_atom.sum())
        device = self.x_exists.device

        if ref_mol_poses is not None:
            templates = ref_mol_poses.to(device)
        elif use_ref_mol_poses:
            if ref_mol is None:
                raise ValueError("use_ref_mol_poses=True needs `ref_mol` (or explicit `ref_mol_poses`)")
            templates = embed_conformers(ref_mol, 512, num_lig_atoms).to(device)
        else:
            templates = None
        if templates is not None and templates.shape[1] != num_lig_atoms:
            logger.warning("template conformers have %d atoms but the ligand has %d; ignoring them and aligning to "
                           "the reference conformer of the input features", templates.shape[1], num_lig_atoms)
            templates = None
        self.templates = templates
        self.templates_dist = None if templates is None else torch.norm(templates[:, :, None] - templates[:, None], dim=-1)

    def _nearest_templates(self, x_denoised: torch.Tensor) -> torch.Tensor:
        """Per sample, the template whose interatomic distances are closest (smooth-lDDT style) to the denoised ligand.

        Args:
            x_denoised: Denoised coordinates, shape ``[S, N_atom, 3]``.

        Returns:
            Template conformers, shape ``[S, N_lig, 3]``.
        """
        ligand_pos = x_denoised[:, self.is_ligand_atom]
        ligand_dist = torch.norm(ligand_pos[:, :, None] - ligand_pos[:, None], dim=-1)
        delta = (ligand_dist[:, None] - self.templates_dist[None]).abs()
        epsilon = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(
            -2 + delta) + torch.sigmoid(-4 + delta))
        epsilon = epsilon.mean(dim=[-1, -2])
        return self.templates[torch.argmin(epsilon, dim=-1)]

    def __call__(
            self,
            x_hat: torch.Tensor,
            x_denoised: torch.Tensor,
            sigma_cur: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return the guided clean-structure estimate, or ``None`` to keep ``x_denoised``.

        Args:
            x_hat: Noisy coordinates, shape ``[S, N_atom, 3]``.
            x_denoised: Estimated clean coordinates, same shape.
            sigma_cur: Noise level in Angstrom of the step before churn, 0-d tensor.

        Returns:
            Coordinates of shape ``[S, N_atom, 3]`` or ``None``.
        """
        if self.align_ref_pos and sigma_cur > self.threshold:
            if self.templates is not None:
                self.ref_pos[:, self.is_ligand_atom] = self._nearest_templates(x_denoised)
            x_ref = self.ref_pos
        elif self.ref_mol is not None and sigma_cur <= self.threshold:
            x_ref = x_denoised.clone()
            x_ref[:, self.is_ligand_atom] = mmff_relax(self.ref_mol, x_denoised[:, self.is_ligand_atom], self.mmff_iters)
        else:
            return None
        ligand_denoised = weighted_rigid_align(x_denoised * self.x_exists[..., None], x_ref, self.weights)
        w = self.weights[None, :, None]
        return x_denoised * (1 - w) + ligand_denoised * w
