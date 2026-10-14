"""Training loss: weighted MSE + smooth lDDT + distogram (AF3 eq. 6/9/24). S = number of noisy samples."""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.utils.tensor_utils import masked_mean, weighted_rigid_align


def weighted_mse_loss(x_denoised, x_gt, t_hat, atom_mask, is_ligand, atom_to_token, sigma_data: float,
                      alpha_ligand: float, max_loss: float = 10000.) -> torch.Tensor:
    """MSE after aligning the ground truth onto each sample, ligand atoms weighted 1 + alpha_ligand, EDM weighting.

    Args:
        x_denoised: [S, N_atom, 3]
        x_gt: [N_atom, 3]
        t_hat: [S] noise level (Angstrom)
        atom_mask: [N_atom]
        is_ligand: [N_token]
        atom_to_token: [N_atom]

    Returns:
        scalar
    """
    weights = (1 + is_ligand * alpha_ligand)[atom_to_token] * atom_mask
    with torch.no_grad():
        x_gt_aligned = weighted_rigid_align(x_denoised * atom_mask[..., None], x_gt, weights)
    sq = torch.norm(x_denoised - x_gt_aligned, dim=-1) ** 2
    loss = (t_hat ** 2 + sigma_data ** 2) / (t_hat * sigma_data) ** 2 * masked_mean(weights[None], sq, dim=(-1, -2)) / 3
    return torch.clamp(loss.mean(), max=max_loss)


def smooth_lddt_loss(x_denoised, x_gt, atom_mask, max_clamp_distance: float = 15.) -> torch.Tensor:
    """1 - smooth lDDT over atom pairs closer than max_clamp_distance in the ground truth.

    Args:
        x_denoised: [S, N_atom, 3]
        x_gt: [N_atom, 3]
        atom_mask: [N_atom]

    Returns:
        scalar
    """
    d_pred = torch.norm(x_denoised[..., None, :] - x_denoised[..., None, :, :], dim=-1)
    d_gt = torch.norm(x_gt[..., None, :] - x_gt[..., None, :, :], dim=-1)
    mask = ((d_gt < max_clamp_distance) * atom_mask[..., None] * atom_mask[..., None, :])[None]
    delta = (d_pred - d_gt[None]).abs()
    eps = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(-2 + delta)
                  + torch.sigmoid(-4 + delta))
    return masked_mean(mask, eps, dim=(-1, -2)).mean()


def distogram_loss(p_distogram, x_gt, atom_mask, token_to_pseudo_beta_atom, min_bin: float = 3.25,
                   max_bin: float = 50.75, no_bins: int = 39) -> torch.Tensor:
    """Cross-entropy of the binned pseudo-beta distances.

    Args:
        p_distogram: [N_token, N_token, no_bins] logits
        x_gt: [N_atom, 3]
        atom_mask: [N_atom]
        token_to_pseudo_beta_atom: [N_token]

    Returns:
        scalar
    """
    x = x_gt[token_to_pseudo_beta_atom]
    exists = atom_mask[token_to_pseudo_beta_atom]
    mask = exists[:, None] * exists[None, :]
    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=p_distogram.device) ** 2
    d2 = torch.sum((x[:, None] - x[None, :]) ** 2, dim=-1, keepdims=True)
    labels = F.one_hot(torch.sum(d2 > boundaries, dim=-1), no_bins) * mask[..., None]
    errors = -torch.sum(labels * F.log_softmax(p_distogram * mask[..., None], dim=-1), dim=-1)
    return masked_mean(mask, errors, dim=(-1, -2))


class PhysDockLoss(nn.Module):
    """Weighted sum of the three losses (weights in config.loss); a non-finite term is replaced by 0."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config.loss

    def forward(self, outputs, feats):
        """
        Args:
            outputs: x_denoised [S, N_atom, 3], t_hat [S], p_distogram [N_token, N_token, 39]
            feats: x_gt, atom_mask, is_ligand, atom_to_token, token_to_pseudo_beta_atom

        Returns:
            (loss, {name: detached value})
        """
        cfg = self.config
        losses = {
            "weighted_mse_loss": weighted_mse_loss(
                outputs["x_denoised"], feats["x_gt"], outputs["t_hat"], feats["atom_mask"], feats["is_ligand"],
                feats["atom_to_token"], cfg.weighted_mse_loss.sigma_data, cfg.weighted_mse_loss.alpha_ligand),
            "smooth_lddt_loss": smooth_lddt_loss(
                outputs["x_denoised"], feats["x_gt"], feats["atom_mask"], cfg.smooth_lddt_loss.max_clamp_distance),
            "distogram_loss": distogram_loss(
                outputs["p_distogram"], feats["x_gt"], feats["atom_mask"], feats["token_to_pseudo_beta_atom"],
                cfg.distogram_loss.min_bin, cfg.distogram_loss.max_bin, cfg.distogram_loss.no_bins),
        }
        total = 0.
        logs = {}
        for name, loss in losses.items():
            if not torch.isfinite(loss):
                logging.warning(f"{name} is not finite, skipped")
                loss = torch.zeros_like(loss, requires_grad=True)
            total = total + cfg[name].weight * loss
            logs[name] = loss.detach().clone()
        logs["loss"] = total.detach().clone()
        return total, logs
