"""Training loss of PhysDock: weighted MSE, smooth lDDT and distogram.

Notation: ``S`` is the number of noisy samples per structure (``num_augmentation_sample``), ``N_atom`` /
``N_token`` the number of (possibly padded) atoms / tokens. Padding is handled through ``atom_mask``.
"""
from __future__ import annotations

import logging
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.utils.tensor_utils import masked_mean, weighted_rigid_align

__all__ = ["weighted_mse_loss", "smooth_lddt_loss", "distogram_loss", "PhysDockLoss"]


def weighted_mse_loss(
        x_denoised: torch.Tensor,
        x_gt: torch.Tensor,
        t_hat: torch.Tensor,
        atom_mask: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        atom_to_token: torch.Tensor,
        sigma_data: float,
        alpha_dna: float,
        alpha_rna: float,
        alpha_ligand: float,
        max_loss: float = 10000.,
) -> torch.Tensor:
    """Noise-level weighted MSE after rigid alignment (AlphaFold 3 eq. 6 / 7), ligands up-weighted.

    The ground truth is superposed onto every denoised sample with a weighted Kabsch alignment (no gradient
    through the alignment). The squared error is weighted per atom (``1 + alpha_* * is_*`` of its token) and per
    sample with the EDM weight ``(t_hat^2 + sigma_data^2) / (t_hat * sigma_data)^2``.

    Args:
        x_denoised: Denoised coordinates, shape ``[S, N_atom, 3]``.
        x_gt: Ground-truth coordinates, shape ``[N_atom, 3]``.
        t_hat: Noise level of every sample, shape ``[S]``.
        atom_mask: 1 for atoms with a ground-truth position, shape ``[N_atom]``.
        is_dna: Token mask, shape ``[N_token]``.
        is_rna: Token mask, shape ``[N_token]``.
        is_ligand: Token mask, shape ``[N_token]``.
        atom_to_token: Token index of every atom, shape ``[N_atom]``.
        sigma_data: Data standard deviation of the diffusion parameterisation.
        alpha_dna: Extra weight of DNA atoms.
        alpha_rna: Extra weight of RNA atoms.
        alpha_ligand: Extra weight of ligand atoms.
        max_loss: The mean loss is clamped to this value for stability.

    Returns:
        Scalar loss.
    """
    weights = (1 + is_dna * alpha_dna + is_rna * alpha_rna + is_ligand * alpha_ligand)[
                  ..., atom_to_token] * atom_mask
    with torch.no_grad():
        x_gt_aligned = weighted_rigid_align(x_denoised * atom_mask[..., None], x_gt, weights)

    squared_diff = torch.norm(x_denoised - x_gt_aligned, dim=-1) ** 2                       # [S, N_atom]
    loss = (t_hat ** 2 + sigma_data ** 2) / (t_hat * sigma_data) ** 2 \
           * masked_mean(weights[..., None, :], squared_diff, dim=(-1, -2)) / 3
    return torch.clamp(loss.mean(), max=max_loss)


def smooth_lddt_loss(
        x_denoised: torch.Tensor,
        x_gt: torch.Tensor,
        atom_mask: torch.Tensor,
        max_clamp_distance: float = 15.,
) -> torch.Tensor:
    """Smooth lDDT loss (AlphaFold 3 eq. 9): ``1 - lDDT`` with sigmoid-smoothed distance-error thresholds.

    Pairs are all atom pairs whose ground-truth distance is below ``max_clamp_distance`` (the diagonal is
    included) and for which both atoms exist. The score of a pair is the mean of ``sigmoid(delta - c)`` for
    thresholds ``c`` of 0.5, 1, 2 and 4 A, where ``delta`` is the absolute distance error.

    Args:
        x_denoised: Denoised coordinates, shape ``[S, N_atom, 3]``.
        x_gt: Ground-truth coordinates, shape ``[N_atom, 3]``.
        atom_mask: 1 for atoms with a ground-truth position, shape ``[N_atom]``.
        max_clamp_distance: Only pairs closer than this in the ground truth (A) are scored.

    Returns:
        Scalar loss.
    """
    diff_denoised = torch.norm(x_denoised[..., None, :] - x_denoised[..., None, :, :], dim=-1)   # [S, N, N]
    diff_gt = torch.norm(x_gt[..., None, :] - x_gt[..., None, :, :], dim=-1)                     # [N, N]

    diff_mask = ((diff_gt < max_clamp_distance) * atom_mask[..., None] * atom_mask[..., None, :])[..., None, :, :]

    delta = (diff_denoised - diff_gt[..., None, :, :]).abs()
    epsilon = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(
        -2 + delta) + torch.sigmoid(-4 + delta))
    loss = masked_mean(diff_mask, epsilon, dim=(-1, -2))
    return loss.mean()


def distogram_loss(
        p_distogram: torch.Tensor,
        x_gt: torch.Tensor,
        atom_mask: torch.Tensor,
        token_to_pseudo_beta_atom: torch.Tensor,
        min_bin: float = 3.25,
        max_bin: float = 50.75,
        no_bins: int = 39,
) -> torch.Tensor:
    """Cross-entropy of the predicted token-pair distance histogram (AlphaFold 3 eq. 24).

    Distances are measured between the pseudo-beta atoms (C-beta, C-alpha for glycine, the atom itself for
    ligands) and binned into ``no_bins`` bins between ``min_bin`` and ``max_bin`` Angstrom.

    Args:
        p_distogram: Predicted logits, shape ``[N_token, N_token, no_bins]``.
        x_gt: Ground-truth coordinates, shape ``[N_atom, 3]``.
        atom_mask: 1 for atoms with a ground-truth position, shape ``[N_atom]``.
        token_to_pseudo_beta_atom: Pseudo-beta atom index of every token, shape ``[N_token]``.
        min_bin: Lower edge of the first bin (A).
        max_bin: Upper edge of the last bin (A).
        no_bins: Number of bins.

    Returns:
        Scalar loss.
    """
    x_pb = x_gt[..., token_to_pseudo_beta_atom, :]
    exists = atom_mask[token_to_pseudo_beta_atom]
    mask = exists[..., None] * exists[..., None, :]                                             # [N_token, N_token]

    boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=p_distogram.device) ** 2
    dists = torch.sum((x_pb[..., None, :] - x_pb[..., None, :, :]) ** 2, dim=-1, keepdims=True)
    true_bins = torch.sum(dists > boundaries, dim=-1)                                           # [N_token, N_token]

    logits = p_distogram * mask[..., :, :, None]
    labels = F.one_hot(true_bins, no_bins) * mask[..., None]
    errors = -torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return masked_mean(mask, errors, dim=(-1, -2))


class PhysDockLoss(nn.Module):
    """Weighted sum of :func:`weighted_mse_loss`, :func:`smooth_lddt_loss` and :func:`distogram_loss`.

    Args:
        config: Model configuration (see ``PhysDockConfig``); ``config.loss`` holds one section per loss with
            its ``weight`` and keyword arguments.

    A non-finite loss term is replaced by zero (with a warning) so that a single bad sample does not poison
    the gradients.
    """

    def __init__(self, config: Mapping) -> None:
        super().__init__()
        self.config = config.loss

    def forward(
            self,
            outputs: Mapping[str, torch.Tensor],
            feats: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the training loss.

        Args:
            outputs: Model outputs with ``x_denoised`` ``[S, N_atom, 3]``, ``t_hat`` ``[S]`` and
                ``p_distogram`` ``[N_token, N_token, no_bins]``.
            feats: Input features (:data:`PhysDock.data.FEATURES`) with ``x_gt``, ``atom_mask``, ``is_ligand``,
                ``atom_to_token`` and ``token_to_pseudo_beta_atom``. PhysDock has no DNA / RNA token type, so the
                DNA and RNA weights of the MSE loss are never applied.

        Returns:
            Tuple ``(loss, logs)``: the differentiable total loss and a dict with the detached value of every
            term (``weighted_mse_loss``, ``smooth_lddt_loss``, ``distogram_loss``) and of the total (``loss``).
        """
        cfg = self.config
        no_nucleic = torch.zeros_like(feats["is_ligand"])
        terms = {
            "weighted_mse_loss": lambda: weighted_mse_loss(
                outputs["x_denoised"], feats["x_gt"], outputs["t_hat"], feats["atom_mask"], no_nucleic,
                no_nucleic, feats["is_ligand"], feats["atom_to_token"],
                sigma_data=cfg.weighted_mse_loss.sigma_data, alpha_dna=cfg.weighted_mse_loss.alpha_dna,
                alpha_rna=cfg.weighted_mse_loss.alpha_rna, alpha_ligand=cfg.weighted_mse_loss.alpha_ligand),
            "smooth_lddt_loss": lambda: smooth_lddt_loss(
                outputs["x_denoised"], feats["x_gt"], feats["atom_mask"],
                max_clamp_distance=cfg.smooth_lddt_loss.max_clamp_distance),
            "distogram_loss": lambda: distogram_loss(
                outputs["p_distogram"], feats["x_gt"], feats["atom_mask"], feats["token_to_pseudo_beta_atom"],
                min_bin=cfg.distogram_loss.min_bin, max_bin=cfg.distogram_loss.max_bin,
                no_bins=cfg.distogram_loss.no_bins),
        }

        total = 0.
        logs: Dict[str, torch.Tensor] = {}
        for name, compute in terms.items():
            loss = compute()
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{name} is not finite. Skipping...")
                loss = torch.zeros_like(loss, requires_grad=True)
            total = total + cfg[name].weight * loss
            logs[name] = loss.detach().clone()
        logs["loss"] = total.detach().clone()
        return total, logs
