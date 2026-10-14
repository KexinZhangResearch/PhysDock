# Copyright 2021 AlQuraishi Laboratory, 2021 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Simplified for PhysDock.
"""Tensor helpers: masks, segment pooling, random rigid augmentation, distograms, weighted rigid alignment."""
from functools import partial

import numpy as np
import torch


def masked_mean(mask, value, dim, eps=1e-9):
    """Mean of value over dim where mask is 1 (mask broadcast to value)."""
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """mask: [..., N] -> [..., N, N]"""
    return mask[..., None, :] * mask[..., :, None]


def segment_mean_pool(x: torch.Tensor, segment_sizes: torch.Tensor, eps: float = 1e-3,
                      method: str = "segment_reduce") -> torch.Tensor:
    """Mean of consecutive rows per segment: sum(rows) / (size + eps); rows after the last segment are ignored.

    The eps is part of the trained model. method "segment_reduce" sums every segment on its own (exact, default);
    "cumsum" is the original running-sum version (cancellation error grows with N), kept to reproduce old results.

    Args:
        x: [..., N, C]
        segment_sizes: [S] rows per segment (zeros allowed, sum <= N)

    Returns:
        [..., S, C]
    """
    if x.dim() < 2 or segment_sizes.dim() != 1:
        raise ValueError(f"need x [..., N, C] and segment_sizes [S], got {tuple(x.shape)} and {tuple(segment_sizes.shape)}")
    sizes = segment_sizes.to(device=x.device, dtype=torch.long)
    denom = (sizes[:, None] + eps).to(x.dtype if x.dtype.is_floating_point else torch.float32)
    if method == "cumsum":
        value = torch.cumsum(x, dim=-2)[..., torch.cumsum(sizes, dim=-1) - 1, :]
        return torch.cat([value[..., 0:1, :], torch.diff(value, dim=-2)], dim=-2) / denom
    if method != "segment_reduce":
        raise ValueError(f"unknown method {method!r}, expected 'segment_reduce' or 'cumsum'")
    n_rows, n_seg = x.shape[-2], sizes.shape[0]
    # segment_reduce needs lengths that add up to N: the padding rows form a last segment that is dropped
    lengths = torch.cat([sizes, (n_rows - sizes.sum()).reshape(1)])
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    flat = x.reshape(-1, n_rows, x.shape[-1]).to(compute_dtype)
    try:
        out = torch.segment_reduce(flat, "sum", lengths=lengths[None].expand(flat.shape[0], -1), axis=1)[:, :n_seg]
    except RuntimeError as e:
        if "negative" in str(e):
            raise ValueError(f"segment sizes add up to more than the number of rows ({n_rows})") from e
        raise
    return out.reshape(*x.shape[:-2], n_seg, x.shape[-1]).to(x.dtype) / denom


def dict_map(fn, dic, leaf_type):
    return {k: dict_map(fn, v, leaf_type) if type(v) is dict else tree_map(fn, v, leaf_type) for k, v in dic.items()}


def tree_map(fn, tree, leaf_type):
    """Apply fn to every leaf of type leaf_type in nested dicts / lists / tuples."""
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    if isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, x, leaf_type) for x in tree)
    if isinstance(tree, leaf_type):
        return fn(tree)
    return None                                     # other leaves are dropped


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


# ======================================================================================
# random rotations and augmentation (AF3 Alg. 19)
# ======================================================================================
def _uniform_sphere_point_np(size):
    phi = np.random.random(*size) * 2 * np.pi
    theta = np.arccos(np.random.random(*size) * 2 - 1)
    return np.stack([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)], axis=-1)


def uniform_random_rotation_np(size):
    """Random rotation matrices [*size, 3, 3] (np.random)."""
    e0 = _uniform_sphere_point_np(size)
    e1 = _uniform_sphere_point_np(size)
    e1 = e1 - e0 * np.sum(e1 * e0, axis=-1, keepdims=True)
    e1 = e1 / np.linalg.norm(e1, axis=-1, keepdims=True)
    return np.stack([e0, e1, np.cross(e0, e1)], axis=-2)


def centre_random_augmentation_np_apply(ref_pos: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Rotate and shift (N(0, 1)) every group of points independently.

    ref_pos: [N, 3], group: [N] group index (0 .. G-1) -> [N, 3]
    """
    num_groups = int(group.max()) + 1
    R = uniform_random_rotation_np([num_groups])[group]
    T = np.random.normal(size=[num_groups, 3])[group]
    return np.einsum("xij,xi->xj", R, ref_pos) + T


def _uniform_sphere_point(size, device):
    phi = torch.rand([*size], device=device, dtype=torch.float32) * 2 * torch.pi
    theta = torch.acos(torch.rand([*size], device=device, dtype=torch.float32) * 2 - 1)
    return torch.stack([torch.cos(phi) * torch.sin(theta), torch.sin(phi) * torch.sin(theta), torch.cos(theta)], dim=-1)


def uniform_random_rotation(size, device):
    """Random rotation matrices [*size, 3, 3] (torch RNG)."""
    e0 = _uniform_sphere_point(size, device)
    e1 = _uniform_sphere_point(size, device)
    e1 = e1 - e0 * (e1 * e0).sum(dim=-1, keepdim=True)
    e1 = e1 / torch.norm(e1, dim=-1, keepdim=True)
    return torch.stack([e0, e1, torch.cross(e0, e1, dim=-1)], dim=-2)


def centre_random_augmentation(x: torch.Tensor, atom_mask: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Centre on the masked mean, random rotation, random translation N(0, s^2) per sample.

    x: [S, N, 3], atom_mask: [N] -> [S, N, 3]
    """
    x = x - torch.sum(x * atom_mask[None, :, None], dim=-2, keepdim=True) / torch.sum(atom_mask)
    R = uniform_random_rotation(x.shape[:-2], device=x.device)
    x = torch.einsum("...ij,...kj->...ki", R, x)
    return x + s * torch.normal(mean=0, std=1, size=x.shape[:-2] + (3,), device=x.device, dtype=x.dtype)[..., None, :]


# ======================================================================================
# masks, distograms, alignment
# ======================================================================================
def gen_attn_mask(mask: torch.Tensor, neg_inf: float) -> torch.Tensor:
    """0 where mask is 1, neg_inf where mask is 0 (additive attention bias)."""
    assert neg_inf < -1e4
    attn_mask = torch.zeros_like(mask)
    attn_mask[mask == 0] = neg_inf
    return attn_mask


def dgram_from_positions(pos: torch.Tensor, min_bin: float = 3.25, max_bin: float = 50.75, no_bins: int = 39,
                         inf: float = 1e8) -> torch.Tensor:
    """pos: [..., N, 3] -> one-hot distance bins [..., N, N, no_bins]"""
    dgram = torch.sum((pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True)
    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    return ((dgram > lower) * (dgram < upper)).type(dgram.dtype)


def weighted_rigid_align(x_pred: torch.Tensor, x_gt: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Superpose x_gt onto every x_pred with weighted Kabsch (AF3 Alg. 28), in float32.

    Args:
        x_pred: [..., S, N, 3]
        x_gt: [..., N, 3] or [..., S, N, 3]
        weights: [..., N]

    Returns:
        x_gt aligned to x_pred, [..., S, N, 3]
    """
    with torch.amp.autocast("cuda", enabled=False):
        x_pred, x_gt, weights = x_pred.float(), x_gt.float(), weights.float()
        if len(x_gt.shape) == 2:
            x_gt = x_gt[..., None, :, :]
        w_sum = torch.sum(weights[..., None, :], dim=-1, keepdim=True)
        mu_pred = torch.sum(x_pred * weights[..., None, :, None], dim=-2) / w_sum
        mu_gt = torch.sum(x_gt * weights[..., None, :, None], dim=-2) / w_sum
        x_pred_hat = x_pred - mu_pred[..., None, :]
        x_gt_hat = x_gt - mu_gt[..., None, :]

        H = torch.sum(torch.einsum("...ij,...ik->...ijk", x_gt_hat, x_pred_hat) * weights[..., None, :, None, None], dim=-3)
        U, _, Vh = torch.linalg.svd(H)
        F = torch.eye(3, device=H.device)
        F[-1, -1] = -1
        R = torch.matmul(U, Vh)
        R_reflection = torch.matmul(U, F).matmul(Vh)
        R = torch.where((torch.det(R) < 0)[..., None, None], R_reflection, R)
        R = torch.transpose(R, -1, -2)
        return torch.einsum("...ij,...kj->...ki", R, x_gt_hat) + mu_pred[..., None, :]
