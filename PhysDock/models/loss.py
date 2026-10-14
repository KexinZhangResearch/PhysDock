"""Training loss (PhysDock2 style): aligned MSE, ligand-frame MSE, smooth lDDT (all pairs and within each conformer)
and distogram. S = number of noisy samples; weights and settings come from configs/loss/*.yaml.

Fixes w.r.t. the released PhysDock loss:
* distogram: padding tokens were counted (their pseudo-beta index 0 points at a real atom); now masked by token_mask
* smooth lDDT: atom self-pairs (always delta 0) were included and diluted the loss; now excluded (AF3)
* the MSE was clamped with torch.clamp(max=1e4), which silently zeroes the gradient of large losses; removed
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.utils.tensor_utils import masked_mean, weighted_rigid_align


def edm_weight(t_hat: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """EDM / AF3 loss weight (t^2 + sigma^2) / (t sigma)^2 / 3; t_hat: [S] noise level (Angstrom) -> [S]"""
    return (t_hat ** 2 + sigma_data ** 2) / (t_hat * sigma_data) ** 2 / 3


def aligned_mse(x_denoised, x_gt, align_weights, mse_mask) -> torch.Tensor:
    """Squared error after superposing x_gt onto every sample (no gradient through the superposition).

    Args:
        x_denoised: [S, N_atom, 3]
        x_gt: [N_atom, 3]
        align_weights: [N_atom] weights of the superposition
        mse_mask: [N_atom] atoms averaged in the error

    Returns:
        [S] mean squared error per sample
    """
    with torch.no_grad():
        x_gt_aligned = weighted_rigid_align(x_denoised * (align_weights > 0)[..., None], x_gt, align_weights)
    sq = torch.norm(x_denoised - x_gt_aligned, dim=-1) ** 2
    return masked_mean(mse_mask[None], sq, dim=-1)


def smooth_lddt(x_denoised, x_gt, atom_mask, atom_to_conformer, max_clamp_distance: float = 15.):
    """1 - smooth lDDT (AF3 eq. 9) over atom pairs i != j closer than max_clamp_distance in the ground truth, and
    over the pairs inside each conformer (residue / ligand).

    Args:
        x_denoised: [S, N_atom, 3]
        x_gt: [N_atom, 3]
        atom_mask: [N_atom]
        atom_to_conformer: [N_atom]

    Returns:
        (all-pair loss, within-conformer loss), scalars
    """
    d_pred = torch.norm(x_denoised[..., None, :] - x_denoised[..., None, :, :], dim=-1)
    d_gt = torch.norm(x_gt[..., None, :] - x_gt[..., None, :, :], dim=-1)
    n = x_gt.shape[-2]
    pairs = atom_mask[:, None] * atom_mask[None, :] * (1 - torch.eye(n, device=x_gt.device, dtype=x_gt.dtype))
    near = (pairs * (d_gt < max_clamp_distance))[None]
    same_conf = (pairs * (atom_to_conformer[:, None] == atom_to_conformer[None, :]))[None]
    delta = (d_pred - d_gt[None]).abs()
    eps = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(-2 + delta)
                  + torch.sigmoid(-4 + delta))
    return masked_mean(near, eps, dim=(-1, -2)).mean(), masked_mean(same_conf, eps, dim=(-1, -2)).mean()


def distogram_loss(p_distogram, x_gt, token_mask, token_to_pseudo_beta_atom, min_bin: float = 3.25,
                   max_bin: float = 50.75, no_bins: int = 39) -> torch.Tensor:
    """Cross-entropy of the binned pseudo-beta distances over real token pairs.

    Args:
        p_distogram: [N_token, N_token, no_bins] logits
        x_gt: [N_atom, 3]
        token_mask: [N_token]
        token_to_pseudo_beta_atom: [N_token]

    Returns:
        scalar
    """
    x = x_gt[token_to_pseudo_beta_atom]
    mask = token_mask[:, None] * token_mask[None, :]
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=p_distogram.device) ** 2
    d2 = torch.sum((x[:, None] - x[None, :]) ** 2, dim=-1, keepdims=True)
    labels = F.one_hot(torch.sum(d2 > boundaries, dim=-1), no_bins)
    errors = -torch.sum(labels * F.log_softmax(p_distogram, dim=-1), dim=-1)
    return masked_mean(mask, errors, dim=(-1, -2))


class PhysDockLoss(nn.Module):
    """Weighted sum of the loss terms; a non-finite term is logged and replaced by 0.

    Args:
        sigma_data: data scale of the EDM weight (Angstrom)
        weights: {x_mse, ligand_mse, smooth_lddt, within_smooth_lddt, distogram} loss weights
        alpha_ligand: x_mse superposition weight of ligand atoms is 1 + alpha_ligand
        ligand_align_weight: ligand_mse superposes on the ligand (ligand atoms this weight, other atoms 1 / it)
        max_clamp_distance: smooth lDDT pair cutoff (Angstrom)
        distogram: {min_bin, max_bin, no_bins}
    """

    def __init__(self, sigma_data: float = 16., weights=None, alpha_ligand: float = 10.,
                 ligand_align_weight: float = 1000., max_clamp_distance: float = 15., distogram=None) -> None:
        super().__init__()
        self.sigma_data = sigma_data
        self.weights = dict(weights or {"x_mse": 4., "ligand_mse": 0.5, "smooth_lddt": 4., "within_smooth_lddt": 0.5,
                                        "distogram": 0.03})
        self.alpha_ligand = alpha_ligand
        self.ligand_align_weight = ligand_align_weight
        self.max_clamp_distance = max_clamp_distance
        self.distogram = dict(distogram or {"min_bin": 3.25, "max_bin": 50.75, "no_bins": 39})

    def forward(self, outputs, batch):
        """
        Args:
            outputs: x_denoised [S, N_atom, 3], t_hat [S], p_distogram [N_token, N_token, no_bins]
            batch: x_gt, atom_mask, atom_to_token, atom_to_conformer, is_ligand, token_mask, token_to_pseudo_beta_atom

        Returns:
            (loss, {name: detached value})
        """
        x, x_gt, atom_mask = outputs["x_denoised"], batch["x_gt"], batch["atom_mask"]
        is_lig = batch["is_ligand"][batch["atom_to_token"]] * atom_mask          # real ligand atoms
        w_t = edm_weight(outputs["t_hat"], self.sigma_data)
        terms = {}
        if self.weights.get("x_mse"):
            terms["x_mse"] = (aligned_mse(x, x_gt, atom_mask * (1 + self.alpha_ligand * is_lig), atom_mask) * w_t).mean()
        if self.weights.get("ligand_mse"):
            align = atom_mask / self.ligand_align_weight + is_lig * self.ligand_align_weight
            terms["ligand_mse"] = (aligned_mse(x, x_gt, align, is_lig) * w_t).mean()
        if self.weights.get("smooth_lddt") or self.weights.get("within_smooth_lddt"):
            terms["smooth_lddt"], terms["within_smooth_lddt"] = smooth_lddt(
                x, x_gt, atom_mask, batch["atom_to_conformer"], self.max_clamp_distance)
        if self.weights.get("distogram"):
            terms["distogram"] = distogram_loss(outputs["p_distogram"], x_gt, batch["token_mask"],
                                                batch["token_to_pseudo_beta_atom"], **self.distogram)
        total = 0.
        logs = {}
        for name, loss in terms.items():
            if not torch.isfinite(loss):
                logging.warning(f"{name} is not finite, skipped")
                loss = torch.zeros_like(loss, requires_grad=True)
            total = total + self.weights.get(name, 0.) * loss
            logs[name] = loss.detach().clone()
        logs["loss"] = total.detach().clone()
        return total, logs
