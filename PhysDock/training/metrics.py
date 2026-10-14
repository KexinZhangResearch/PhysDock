"""Structure metrics of sampled structures (validation, after PhysDock2's evaluation).

All error-type metrics: lower is better, except success / best_success (higher is better).
"""
import torch
from rdkit import Chem

from PhysDock.models.loss import aligned_mse, smooth_lddt
from PhysDock.utils.tensor_utils import masked_mean, weighted_rigid_align

_pt = Chem.GetPeriodicTable()
# vdW radius by element index of ref_feat (index 0 = H); unknown elements 2.0 A
VDW_RADIUS = torch.tensor([_pt.GetRvdw(z) if z <= 118 else 2.0 for z in range(1, 129)], dtype=torch.float32)
ELEMENT_SLICE = slice(4, 132)       # element one-hot in ref_feat (after ref_pos 3 and charge 1)


@torch.no_grad()
def sample_metrics(x: torch.Tensor, batch) -> dict:
    """
    Args:
        x: [S, N_atom, 3] sampled structures
        batch: unbatched features (x_gt, atom_mask, atom_to_token, atom_to_conformer, is_ligand, asym_id,
            token_bonds, ref_feat)

    Returns:
        {name: scalar}; ligand metrics only if the structure has ligand atoms, clash only with >= 2 chains:
        lrmsd (ligand RMSD after superposing on all atoms), success (share of samples < 2 A), best_success
        (any sample < 2 A), ligand_mse (ligand superposed on the ligand), slddt / within_slddt (smooth lDDT error
        of all pairs / pairs inside each residue or ligand), ligand_bond_error (mean |d - d_gt| of ligand bonds and
        angles, Angstrom), inter_clash (share of inter-chain atom pairs closer than r_i + r_j - 0.4 A)
    """
    x_gt, atom_mask, a2t = batch["x_gt"], batch["atom_mask"], batch["atom_to_token"]
    out = {}
    out["slddt"], out["within_slddt"] = smooth_lddt(x, x_gt, atom_mask, batch["atom_to_conformer"])

    is_lig = batch["is_ligand"][a2t] * atom_mask
    if is_lig.sum() > 0:
        gt_aligned = weighted_rigid_align(x * atom_mask[..., None], x_gt, atom_mask)
        lrmsd = torch.sqrt(masked_mean(is_lig[None], torch.sum((x - gt_aligned) ** 2, dim=-1), dim=-1))
        out["lrmsd"] = lrmsd.mean()
        out["success"] = (lrmsd < 2).float().mean()
        out["best_success"] = (lrmsd < 2).any().float()
        out["ligand_mse"] = aligned_mse(x, x_gt, atom_mask / 1000 + is_lig * 1000, is_lig).mean()

        bonds = batch["token_bonds"][a2t][:, a2t] * is_lig[:, None] * is_lig[None, :]   # ligand atoms are tokens
        if bonds.sum() > 0:
            angles = ((bonds @ bonds) > 0).float() * (1 - bonds) * (1 - torch.eye(len(bonds), device=bonds.device))
            mask = (bonds + angles).clamp(max=1)
            d = torch.norm(x[:, :, None] - x[:, None, :], dim=-1)
            d_gt = torch.norm(x_gt[:, None] - x_gt[None, :], dim=-1)
            out["ligand_bond_error"] = masked_mean(mask[None], (d - d_gt[None]).abs(), dim=(-1, -2)).mean()

    asym = batch["asym_id"][a2t]
    inter = (asym[:, None] != asym[None, :]).float() * atom_mask[:, None] * atom_mask[None, :]
    if inter.sum() > 0:
        radius = VDW_RADIUS.to(x.device)[batch["ref_feat"][:, ELEMENT_SLICE].argmax(-1)]
        d = torch.norm(x[:, :, None] - x[:, None, :], dim=-1)
        clash = (d < (radius[:, None] + radius[None, :] - 0.4)[None]).float()
        out["inter_clash"] = masked_mean(inter[None], clash, dim=(-1, -2)).mean()
    return out
