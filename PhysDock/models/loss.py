import torch
import torch.nn as nn
import logging

from PhysDock.utils.tensor_utils import masked_mean


def cross_entropy_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        eps: float = 1e-9,
) -> torch.Tensor:
    """
    Args:
        logits: logits, torch.Tensor, [..., num_classes]
        labels: labels, torch.Tensor, [...]
    """
    return -torch.mean(torch.sum(labels * torch.log(logits + eps), dim=-1))


def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def weighted_rigid_align(
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        weights: torch.Tensor,
) -> torch.Tensor:
    """
    Implements Algorithm 28. Weighted Rigid Align
    Args:
        x_pred: predicted atom positions, torch.Tensor, [..., num_samples, num_atoms, 3]
        x_gt: ground truth atom positions, torch.Tensor, [..., num_atoms, 3]
        weights: weights for each atom, torch.Tensor, [..., num_atoms]
    """
    # Mean-centre positions
    # [*, num_samples, 3]

    mu_pred = torch.sum(x_pred * weights[..., None, :, None], dim=-2) / torch.sum(weights[..., None, :], dim=-1,
                                                                                  keepdim=True)
    mu_gt = torch.sum(x_gt[..., None, :, :] * weights[..., None, :, None], dim=-2) / torch.sum(weights[..., None, :],
                                                                                               dim=-1, keepdim=True)

    # [*, num_samples, num_atoms, 3]
    x_pred_hat = x_pred - mu_pred[..., None, :]
    x_gt_hat = x_gt[..., None, :, :] - mu_gt[..., None, :]

    # Find optimal rotation from singular value decomposition
    outer_product = torch.einsum("...ij,...ik->...ijk", x_gt_hat, x_pred_hat)
    H = torch.sum(outer_product * weights[..., None, :, None, None], dim=-3)
    U, _, Vh = torch.linalg.svd(H)

    F = torch.eye(3, device=H.device)
    F[-1, -1] = -1

    R = torch.matmul(U, Vh)
    R_reflection = torch.matmul(U, F).matmul(Vh)

    reflection_mask = torch.det(R) < 0
    R = torch.where(reflection_mask[..., None, None], R_reflection, R)
    R = torch.transpose(R, -1, -2)

    # Apply rotation
    # [*, num_samples, num_atoms, 3]
    x_pred_align_to_gt = torch.einsum("...ij,...kj->...ki", R, x_gt_hat) + mu_pred[..., None, :]

    # Return aligned positions with stopping gradients
    # TODO: Check if detach is necessary
    # [*, num_samples, num_atoms, 3]
    return x_pred_align_to_gt


def distogram_loss(
        p_distogram: torch.Tensor,
        x_gt: torch.Tensor,
        x_exists,
        token_id_to_pseudo_beta_atom_id: torch.Tensor,
        min_bin=3.25,
        max_bin=50.75,
        no_bins=39,
        eps=1e-9,
        **kwargs,
):
    # x_gt_distogram_atom = torch.gather(x_gt, -2, token_id_to_pseudo_beta_atom_id[..., :, None].repeat(1, 1, 3))
    x_gt_distogram_atom = x_gt[..., token_id_to_pseudo_beta_atom_id, :]
    mask = x_exists[token_id_to_pseudo_beta_atom_id][..., None] * x_exists[token_id_to_pseudo_beta_atom_id][..., None,
                                                                  :]
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=p_distogram.device,
    )
    boundaries = boundaries ** 2

    dists = torch.sum(
        (x_gt_distogram_atom[..., None, :] - x_gt_distogram_atom[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)
    errors = softmax_cross_entropy(
        p_distogram * mask[..., :, :, None],
        torch.nn.functional.one_hot(true_bins, no_bins) * mask[..., None],
    )

    mean = masked_mean(mask, errors, dim=(-1, -2))

    return mean


def weighted_mse_loss(
        x_denoised: torch.Tensor,  # [48, num_atom, 3]
        x_gt: torch.Tensor,  # [120, num_atom, 3]
        t_hat: torch.Tensor,
        sigma_data: float,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        alpha_dna: float,
        alpha_rna: float,
        alpha_ligand: float,
        atom_id_to_token_id: torch.Tensor,
        x_exists,
        **kwargs,  # ensembled x_gt [num_ensemble, num_token, num_token] -> [num_ensemble]
) -> torch.Tensor:
    """
    Args:
        x_denoised: predicted atom positions, torch.Tensor, [..., num_samples, num_atoms, 3]
        x_gt: ground truth atom positions, torch.Tensor, [..., num_atoms, 3]
    """
    weights = (1 + is_dna * alpha_dna + is_rna * alpha_rna + is_ligand * alpha_ligand)[
                  ..., atom_id_to_token_id] * x_exists
    # [*, num_samples, num_atoms, 3]
    with torch.no_grad():
        x_gt_aligned = weighted_rigid_align(x_denoised * x_exists[..., None], x_gt, weights)

    # [*, num_samples, num_atoms]
    squared_diff = torch.norm(x_denoised - x_gt_aligned, dim=-1) ** 2

    # [*, num_samples]
    # loss = ((t_hat ** 2 + sigma_data ** 2) / (t_hat + sigma_data) ** 2 / 3 *
    #         masked_mean(weights[..., None, :], squared_diff, dim=(-1, -2)))

    # EDM loss weight: 2*t_hat**2 - 2*t_hat + 1
    # loss = ((1 / t_hat ** 2) *
    #         masked_mean(weights[..., None, :], squared_diff, dim=(-1, -2)))
    # loss = ((2 * t_hat ** 2 - 2 * t_hat + 1) *
    #         masked_mean(weights[..., None, :], squared_diff, dim=(-1, -2))) / 3 * 16 ** 2
    sigma_data = 16
    loss = (t_hat ** 2 + sigma_data ** 2) / (t_hat * sigma_data) ** 2 \
           * masked_mean(weights[..., None, :], squared_diff, dim=(-1, -2)) / 3
    return torch.clamp(loss.mean(), max=10000.)


def smooth_lddt_loss(
        x_denoised,
        x_gt,
        x_exists,
        t_hat,
        atom_id_to_token_id,
        max_clamp_distance=16,  # 16
        **kwargs
):
    diff_denoised = torch.norm(x_denoised[..., None, :] - x_denoised[..., None, :, :], dim=-1)
    diff_gt = torch.norm(x_gt[..., None, :] - x_gt[..., None, :, :], dim=-1)

    diff_mask = ((diff_gt < max_clamp_distance) * x_exists[..., None] * x_exists[..., None, :])[..., None, :, :]

    delta = (diff_denoised - diff_gt[..., None, :, :]).abs()
    epsilon = 0.25 * (torch.sigmoid(-0.5 + delta) + torch.sigmoid(-1 + delta) + torch.sigmoid(
        -2 + delta) + torch.sigmoid(-4 + delta))
    loss = masked_mean(
        diff_mask, epsilon, dim=(-1, -2))
    return loss.mean()


def express_coordinates_in_frame(x, x_frames):
    # [*, num_token, 3] | [*, num_token, 3, 3]
    # [*, num_atom, 3] | [*, num_frame, abc, 3]

    with torch.no_grad():
        a, b, c = x_frames[..., 0, :], x_frames[..., 1, :], x_frames[..., 2, :]
        w1 = (a - b) / (torch.norm(a - b + 1e-6, dim=-1, keepdim=True))
        w2 = (c - b) / (torch.norm(c - b + 1e-6, dim=-1, keepdim=True))

        cos_theta = torch.sum(w1 * w2, dim=-1)

        valid_mask = cos_theta < 0.906308  # cos(25)

        e1 = (w1 + w2) / (torch.norm(w1 + w2 + 1e-6, dim=-1, keepdim=True))
        e2 = (w2 - w1) / (torch.norm(w2 - w1 + 1e-6, dim=-1, keepdim=True))
        e3 = torch.cross(e1, e2, dim=-1)
        # [*,num_frame,3,3]
        R = torch.stack([e1, e2, e3], dim=-1).transpose(-1, -2)

    # [*,num_frame,num_atom,3]
    d = x[..., None, :, :] - b[..., None, :]

    x = torch.einsum("...FIJ,...FAJ->...FAI", R, d)
    return x, valid_mask


def fape_loss(
        x_denoised,
        x_gt,
        x_exists,
        token_id_to_centre_atom_id,
        token_id_to_frame_atom_id_0,
        token_id_to_frame_atom_id_1,
        token_id_to_frame_atom_id_2,
        **kwargs,
):
    token_id_to_frame_atom_ids = torch.stack([
        token_id_to_frame_atom_id_0, token_id_to_frame_atom_id_1, token_id_to_frame_atom_id_2
    ], dim=-1)

    with torch.no_grad():
        x_gt_token_exists = x_exists[token_id_to_centre_atom_id]
        frames_gt = x_gt[token_id_to_frame_atom_ids]
        x_gt_expressed, x_gt_valid_mask = express_coordinates_in_frame(x_gt, frames_gt)

    frames_denoised = x_denoised[:, token_id_to_frame_atom_ids]

    x_denoised_expressed, x_denoised_valid_mask = express_coordinates_in_frame(x_denoised, frames_denoised)

    dist = torch.norm(x_gt[token_id_to_centre_atom_id][:, None] - x_gt[None], dim=-1)
    clamp_mask = (dist < 1.).float()
    # clamp_mask.requires_grad = False
    # num_sample, num_frame, num_atom
    error = torch.mean((x_denoised_expressed - x_gt_expressed[None]) ** 2, dim=-1)
    mask = (x_exists[None] * x_gt_token_exists[:, None] * clamp_mask * x_gt_valid_mask[..., None])[None] * \
           x_denoised_valid_mask[..., None]
    mask.requires_grad = False
    return masked_mean(mask, error, dim=[-1, -2, -3])


# TODO: add x_exists
def bond_loss(
        x_denoised: torch.Tensor,
        x_gt: torch.Tensor,
        x_exists,
        token_bonds: torch.Tensor,
        t_hat: torch.Tensor,
        sigma_data: float,
        atom_id_to_token_id: torch.Tensor,
        token_id_to_centre_atom_id: torch.Tensor,
        eps: float = 1e-9,

        **kwargs,
) -> torch.Tensor:
    """
    Args:
        x_denoised: predicted atom positions, torch.Tensor, [..., num_samples, num_atoms, 3]
        x_gt: ground truth atom positions, torch.Tensor, [..., num_atoms, 3]
        token_bonds: bond mask for tokens, torch.Tensor, [..., num_tokens, num_tokens]
    """

    #
    x_denoised_centre = x_denoised[:, token_id_to_centre_atom_id, :]
    x_gt_centre = x_gt[token_id_to_centre_atom_id, :]

    bond_lengths_pred = torch.norm(x_denoised_centre[..., None, :, :] - x_denoised_centre[..., :, None, :], dim=-1)
    bond_lengths_gt = torch.norm(x_gt_centre[..., None, :, :] - x_gt_centre[..., :, None, :], dim=-1)[..., None, :, :]

    #
    bond_diff = bond_lengths_pred - bond_lengths_gt[None]

    # # [*, num_samples, num_atoms, num_atoms]
    # atom_bonds = token_bonds[..., atom_id_to_token_id, :][..., :, atom_id_to_token_id][..., None, :, :]

    # [*, num_samples]
    loss = (t_hat ** 2 + sigma_data ** 2) / (t_hat * sigma_data) ** 2 * \
           torch.mean(torch.sum(token_bonds[None] * bond_diff ** 2, dim=(-1, -2))
                      / (torch.sum(token_bonds, dim=(-1, -2)) + eps))

    ###################################### OLD ############################################################
    # bond_lengths_pred = torch.norm(x_denoised[..., None, :, :] - x_denoised[..., :, None, :], dim=-1)
    # bond_lengths_gt = torch.norm(x_gt[..., None, :, :] - x_gt[..., :, None, :], dim=-1)[..., None, :, :]
    #
    # #
    # bond_diff = bond_lengths_pred - bond_lengths_gt
    #
    # # [*, num_samples, num_atoms, num_atoms]
    # atom_bonds = token_bonds[..., atom_id_to_token_id, :][..., :, atom_id_to_token_id][..., None, :, :]
    #
    # # [*, num_samples]
    # loss = (t_hat ** 2 + sigma_data ** 2) / (t_hat + sigma_data) ** 2 * \
    #        torch.mean(torch.sum(atom_bonds * bond_diff ** 2, dim=(-1, -2))
    #                   / (torch.sum(atom_bonds, dim=(-1, -2)) + eps)
    #                   )

    ###################################### efficient old ####################################################
    # has_bond = torch.any(token_bonds > 0, dim=-1).bool()
    # atom_has_bond = has_bond[atom_id_to_token_id]
    #
    # atom_bonds = token_bonds[atom_id_to_token_id, :][atom_has_bond, :][:, atom_id_to_token_id][
    #              :, atom_has_bond][..., None, :, :]
    # bond_lengths_pred = torch.norm(
    #     x_denoised[..., atom_has_bond, :][..., None, :, :] - x_denoised[..., atom_has_bond, :][..., :, None, :], dim=-1)
    # bond_lengths_gt = torch.norm(
    #     x_gt[..., atom_has_bond, :][..., None, :, :] - x_gt[..., atom_has_bond, :][..., :, None, :], dim=-1)[..., None,
    #                   :, :]
    #
    # bond_diff = bond_lengths_pred - bond_lengths_gt[None]
    # loss = (t_hat ** 2 + sigma_data ** 2) / (t_hat + sigma_data) ** 2 * \
    #        torch.mean(torch.sum(atom_bonds * bond_diff ** 2, dim=(-1, -2))
    #                   / (torch.sum(atom_bonds, dim=(-1, -2)) + eps)
    #                   )

    return loss.mean()


def cal_lddt(
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_polymer: torch.Tensor,
        token_id_to_centre_atom_id: torch.Tensor,
):
    """
    Args:
        x_pred: predicted atom positions, torch.Tensor, [..., num_atoms, 3]
        x_gt: ground truth atom positions, torch.Tensor, [..., num_atoms, 3]
        is_dna: mask for DNA atoms, torch.Tensor, [..., num_tokens]
        is_rna: mask for RNA atoms, torch.Tensor, [..., num_tokens]
        is_polymer: mask for polymer atoms, torch.Tensor, [..., num_tokens]
        token_id_to_centre_atom_id: index of the centre atom for each token, torch.Tensor, [..., num_tokens]
    """
    # The set of atoms m ∈ R is defined as:
    #   • Atoms such that the distance in the ground truth between atom l and atom m is less than 15 Å if m is a protein
    #     atom or less than 30 Å if m is a nucleic acid atom.
    #   • Only atoms in polymer chains.
    #   • One atom per token - Cα for standard protein residues and C1′ for standard nucleic acid residues.

    # [*, num_tokens, 3]
    # x_pred_token_centre = torch.gather(x_pred, -2, token_id_to_centre_atom_id[..., :, None].repeat(1, 1, 3))
    # x_gt_token_centre = torch.gather(x_gt, -2, token_id_to_centre_atom_id[..., :, None].repeat(1, 1, 3))
    x_pred_token_centre = x_pred[..., token_id_to_centre_atom_id, :]
    x_gt_token_centre = x_gt[..., token_id_to_centre_atom_id, :]

    # [*, 1, num_tokens]
    is_nucleotide = (is_dna + is_rna)[..., None, :]

    # [*, num_atoms. num_tokens]
    d_pred = torch.norm(x_pred[..., :, None, :] - x_pred_token_centre[..., None, :, :], dim=-1)
    d_gt = torch.norm(x_gt[..., :, None, :] - x_gt_token_centre[..., None, :, :], dim=-1)

    # [*, num_atoms, num_tokens]
    d_lm = torch.abs(d_pred - d_gt)
    score = 0.25 * (
            (d_lm < 0.5).type(d_lm.dtype) +
            (d_lm < 1.0).type(d_lm.dtype) +
            (d_lm < 2.0).type(d_lm.dtype) +
            (d_lm < 4.0).type(d_lm.dtype)
    )
    # [*, num_atoms, num_tokens]
    mask_R = (d_gt < 30) * is_nucleotide + (d_gt < 15) * (1 - is_nucleotide)
    mask_R = mask_R * is_polymer[..., None, :]
    # mask_R = mask_R * x_exists[..., None]

    # [*, num_atoms]
    lddt = torch.sum(mask_R * score, dim=-1) / torch.sum(mask_R, dim=-1)
    # lddt = masked_mean(mask_R, score, dim=-1)
    return lddt


def plddt_loss(
        p_plddt: torch.Tensor,
        no_bins: int,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        x_exists,
        is_dna: torch.Tensor,
        is_rna: torch.Tensor,
        is_ligand: torch.Tensor,
        token_id_to_centre_atom_id: torch.Tensor,
        **kwargs,
) -> torch.Tensor:
    """Supplementary Information section 4.3.1, equation 9, pLDDT loss

    Args:
        p_plddt (torch.Tensor): probabilities of lddt bins retured by plddt head, [..., num_atoms, no_bins]
        x_pred (torch.Tensor): predicted atom positions, [..., num_atoms, 3]
        x_gt (torch.Tensor): ground truth atom positions, [..., num_atoms, 3]
        is_dna (torch.Tensor): mask for DNA atoms, [..., num_atoms]
        is_rna (torch.Tensor): mask for RNA atoms, [..., num_atoms]
        is_polymer (torch.Tensor): mask for polymer atoms, [..., num_atoms]
        token_id_to_centre_atom_id (torch.Tensor): index of the centre atom for each token, [..., num_tokens]

    Returns:
        torch.Tensor: plddt loss
    """
    is_polymer = ~is_ligand

    # [*, num_atoms]
    # TODO: additional dimention
    with torch.no_grad():
        lddt = cal_lddt(x_pred, x_gt, is_dna, is_rna, is_polymer, token_id_to_centre_atom_id)[0]
    # lddt = lddt.detach()
    bin_index = torch.clamp((lddt * no_bins).long(), 0, no_bins - 1)
    # [*, num_atoms, no_bins]
    lddt_bined = nn.functional.one_hot(bin_index, num_classes=no_bins).type(lddt.dtype)

    # x_exists_token = x_exists[token_id_to_centre_atom_id]

    # print(lddt_bined.shape, x_exists_token.shape, p_plddt.shape)
    # [*, ]
    # l_plddt = cross_entropy_loss(p_plddt * x_exists[..., None], lddt_bined * x_exists[..., None])
    # print(lddt_bined.shape,x_pred.shape,p_plddt.shape)
    l_plddt = softmax_cross_entropy(p_plddt * x_exists[..., None], lddt_bined * x_exists[..., None])
    mean = masked_mean(x_exists, l_plddt, dim=(-1))
    # mean = masked_mean(mask[None], l_plddt, dim=-1)
    return mean


def pae_loss(
        p_pae: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        x_exists,
        token_id_to_centre_atom_id: torch.Tensor,  # [num_token]
        # token_id_to_frame_atom_ids: torch.Tensor,  # [num_token,3]
        token_id_to_frame_atom_id_0: torch.Tensor,  # [num_token,3]
        token_id_to_frame_atom_id_1: torch.Tensor,  # [num_token,3]
        token_id_to_frame_atom_id_2: torch.Tensor,  # [num_token,3]
        min_bin=0,
        max_bin=32,
        no_bins=64,
        **kwargs,
) -> torch.Tensor:
    with torch.no_grad():
        token_id_to_frame_atom_ids = torch.stack([
            token_id_to_frame_atom_id_0, token_id_to_frame_atom_id_1, token_id_to_frame_atom_id_2
        ], dim=-1)
        x_gt_token_exists = x_exists[token_id_to_centre_atom_id]
        x_gt_token = x_gt[token_id_to_centre_atom_id]
        frames_gt = x_gt[token_id_to_frame_atom_ids]
        # logging.warning(f"x_gt_token {x_gt_token.shape}")
        # logging.warning(f"frames_gt {frames_gt.shape}")

        x_gt_expressed, x_gt_valid_mask = express_coordinates_in_frame(x_gt_token, frames_gt)
        # logging.warning(f"x_gt_expressed {x_gt_expressed.shape}")
        x_pred_token = x_pred[0, token_id_to_centre_atom_id]
        frames_pred = x_pred[0, token_id_to_frame_atom_ids]
        # logging.warning(f"x_pred_token {x_pred_token.shape}")
        # logging.warning(f"frames_pred {frames_pred.shape}")
        x_pred_expressed, x_pred_valid_mask = express_coordinates_in_frame(x_pred_token, frames_pred)
        # logging.warning(f"x_pred_expressed {x_pred_expressed.shape}")
        error = torch.norm(x_pred_expressed - x_gt_expressed, dim=-1) * x_gt_valid_mask[..., None] * x_pred_valid_mask[
            ..., None]
        # error_mean = error.sum(dim=-1)/

    bin_index = torch.clamp(((error - min_bin) / (max_bin - min_bin) * no_bins).long(), 0, no_bins - 1)
    error_bined = nn.functional.one_hot(bin_index, num_classes=no_bins).type(error.dtype)

    mask = x_gt_token_exists[..., None] * x_gt_token_exists[..., None, :]
    # l_pae = cross_entropy_loss(p_pae[None] * mask[..., None, :, :, None], error_bined * mask[..., None, :, :, None])
    l_pae = softmax_cross_entropy(p_pae * mask[..., None], error_bined * mask[..., None])
    mean = masked_mean(mask, l_pae, dim=(-1, -2))
    # mean = masked_mean(mask[..., None, :], l_pae, dim=(-1, -2))
    return mean


def pde_loss(
        p_pde: torch.Tensor,
        x_pred: torch.Tensor,
        x_gt: torch.Tensor,
        x_exists,
        token_id_to_centre_atom_id: torch.Tensor,
        min_bin: float = 0.0,
        max_bin: float = 32.0,
        no_bins: int = 64,
        **kwargs,
) -> torch.Tensor:
    """Supplementary Information section 4.3.2, equation 13, pde loss

    Args:
        p_pde (torch.Tensor): probabilities of distance bins returned by pde head, [..., num_atoms, num_atoms, no_bins]
        x_pred (torch.Tensor): predicted atom positions, [..., num_atoms, 3]
        x_gt (torch.Tensor): ground truth atom positions, [..., num_atoms, 3]
        token_id_to_centre_atom_id (torch.Tensor): index of the centre atom for each token, [..., num_tokens]
        min_bin (float, optional): minimum distance. Defaults to 0.0.
        max_bin (float, optional): maximum distance. Defaults to 32.0.
        no_bins (int, optional): number of bins. Defaults to 64.

    Returns:
        torch.Tensor:
    """
    # x_pred_token_centre = torch.gather(x_pred, -2, token_id_to_centre_atom_id[..., :, None].repeat(1, 1, 3))
    # x_gt_token_centre = torch.gather(x_gt, -2, token_id_to_centre_atom_id[..., :, None].repeat(1, 1, 3))
    x_pred_token_centre = x_pred[..., token_id_to_centre_atom_id, :]
    x_gt_token_centre = x_gt[..., token_id_to_centre_atom_id, :]

    # [*, num_tokens, num_tokens]
    d_pred = torch.norm(x_pred_token_centre[..., :, None, :] - x_pred_token_centre[..., None, :, :], dim=-1)
    d_gt = torch.norm(x_gt_token_centre[..., :, None, :] - x_gt_token_centre[..., None, :, :], dim=-1)

    # [*, num_tokens, num_tokens]
    d_diff = torch.abs(d_pred - d_gt)[0]

    # [*, num_tokens, num_tokens, no_bins]
    bin_index = torch.clamp(((d_diff - min_bin) / (max_bin - min_bin) * no_bins).long(), 0, no_bins - 1)
    d_diff_bined = nn.functional.one_hot(bin_index, num_classes=no_bins).type(d_diff.dtype)

    # [*, ]
    mask = x_exists[token_id_to_centre_atom_id][..., None] * x_exists[token_id_to_centre_atom_id][..., None, :]

    # l_pde = cross_entropy_loss(p_pde[None] * mask[..., None, :, :, None], d_diff_bined * mask[..., None, :, :, None])

    errors = softmax_cross_entropy(
        p_pde * mask[..., None],
        d_diff_bined * mask[..., None]
    )

    mean = masked_mean(mask, errors, dim=(-1, -2))

    # errors = softmax_cross_entropy(
    #     p_distogram * mask[..., :, :, None],
    #     torch.nn.functional.one_hot(true_bins, no_bins) * mask[..., None],
    # )
    #
    # mean = masked_mean(mask, errors, dim=(-1, -2))

    return mean


def key_res_loss(
        x_denoised: torch.Tensor,
        x_gt: torch.Tensor,  # [120, num_atom, 3]
        t_hat: torch.Tensor,
        is_ligand: torch.Tensor,
        is_key_res,
        sigma_data,
        token_id_to_centre_atom_id,
        eps=1e-9,
        **kwargs,
):
    x_denoised_centre = x_denoised[:, token_id_to_centre_atom_id, :]
    x_gt_centre = x_gt[token_id_to_centre_atom_id, :]

    bond_lengths_pred = torch.norm(x_denoised_centre[..., None, :, :] - x_denoised_centre[..., :, None, :], dim=-1)
    bond_lengths_gt = torch.norm(x_gt_centre[..., None, :, :] - x_gt_centre[..., :, None, :], dim=-1)[..., None, :, :]
    bond_diff = (bond_lengths_pred - bond_lengths_gt[None]).abs()
    bond_diff = 0.25 * (torch.sigmoid(-0.5 + bond_diff) + torch.sigmoid(-1 + bond_diff) + torch.sigmoid(
        -2 + bond_diff) + torch.sigmoid(-4 + bond_diff))
    mask = (is_key_res[:, None] * is_ligand[None])[None]
    w2 = (t_hat ** 2 + sigma_data ** 2) / (t_hat * sigma_data) ** 2
    loss = torch.mean(torch.sum(mask[None] * bond_diff ** 2, dim=(-1, -2))
                      / (torch.sum(mask, dim=(-1, -2)) + eps)) * w2

    return loss.mean()


def experimentally_resolved_loss(
        p_resolved: torch.Tensor,
        is_resolved: torch.Tensor,
        **kwargs,
):
    """
    Args:
        p_resolved: probabilities of resolution bins, torch.Tensor, [..., no_bins]
        is_resolved: mask for resolution, torch.Tensor, [...]
    """
    is_resolved_bined = nn.functional.one_hot(is_resolved.long(), num_classes=2).type(p_resolved.dtype)
    return cross_entropy_loss(p_resolved, is_resolved_bined)


class PhysDockLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config.loss

    def forward(self, outputs, feats):
        loss_fns = {
            "weighted_mse_loss": lambda: weighted_mse_loss(
                **outputs, **feats, **self.config.weighted_mse_loss
            ),
            "smooth_lddt_loss": lambda: smooth_lddt_loss(
                **outputs, **feats, **self.config.smooth_lddt_loss,
            ),
            "bond_loss": lambda: bond_loss(
                **outputs, **feats, **self.config.bond_loss
            ),
            "key_res_loss": lambda: key_res_loss(
                **outputs, **feats, **self.config.key_res_loss
            ),
            "distogram_loss": lambda: distogram_loss(
                **outputs, **feats, **self.config.distogram_loss
            ),
        }
        # if self.use_mini_rollout:
        #     loss_fns.update({
        #         "plddt_loss": lambda: plddt_loss(
        #             **outputs, **feats, **self.config.plddt_loss
        #         ),
        #         "pae_loss": lambda: pae_loss(
        #             **outputs, **feats, **self.config.pae_loss
        #         ),
        #         "pde_loss": lambda: pde_loss(
        #             **outputs, **feats, **self.config.pde_loss
        #         ),
        #     })

        cum_loss = 0.
        losses = {}
        for loss_name, loss_fn in loss_fns.items():
            weight = self.config[loss_name].weight
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = torch.zeros_like(loss, requires_grad=True)
            cum_loss = cum_loss + weight * loss
            losses[loss_name] = loss.detach().clone()

        losses["loss"] = cum_loss.detach().clone()
        return cum_loss, losses
