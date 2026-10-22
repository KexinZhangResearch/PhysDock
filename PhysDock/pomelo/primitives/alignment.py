"""Proper Kabsch rotations: the frame of the reference positions of every group (a conformer: its reference conformer; the template) moved into the frame of a noisy sample.

    R_g = argmin over proper rotations of  sum_a |R (y_a - c_y) - (x_a - c_x)|^2   (the real atoms a of group g)

computed by an SVD of H = sum_a (y_a - c_y)(x_a - c_x)^T = U S V^T:  R = V diag(1, 1, det(V U^T)) U^T. Under the symmetry of the problem (the noisy sample by Q, the reference
of a group by its own P with det P = det Q) R -> Q R P^T, so R (y - c_y) transforms as a polar vector of the noisy sample: the reference vectors enter the vector channels
of the network as features of the frame of the sample. R is orthogonal: every distance, angle and signed volume inside a group is exact, for any noisy sample (at large noise
only the orientation relative to the noisy vectors is random). R is proper: the hand of the reference is kept, a noisy sample of the other hand cannot be matched.
R depends on the inputs only (the noisy coordinates and the reference positions): it is computed without gradient. A group with fewer than three real atoms (or collinear
ones) has no unique R; the vectors R (y - c_y) are unique anyway (along the axis, or zero).
"""
import torch


@torch.no_grad()
def kabsch(x: torch.Tensor, y: torch.Tensor, group: torch.Tensor, mask: torch.Tensor, n_groups: int):
    """x [B, N, 3] (the noisy sample), y [N, 3] (the references, every group in a frame of its own), group [N], mask [N] -> R [B, G, 3, 3], the centred references
    y - c_y [N, 3] (0 at the padding)"""
    m = mask.to(x.dtype)
    cnt = torch.zeros(n_groups, dtype=x.dtype, device=x.device).index_add(0, group, m).clamp(min=1)
    cy = torch.zeros(n_groups, 3, dtype=x.dtype, device=x.device).index_add(0, group, y * m[:, None]) / cnt[:, None]
    cx = torch.zeros(x.shape[0], n_groups, 3, dtype=x.dtype, device=x.device).index_add(1, group, x * m[None, :, None]) / cnt[None, :, None]
    yc, xc = (y - cy[group]) * m[:, None], (x - cx[:, group]) * m[None, :, None]
    h = torch.zeros(x.shape[0], n_groups, 3, 3, dtype=x.dtype, device=x.device).index_add(1, group, yc[None, :, :, None] * xc[:, :, None, :])
    u, _, vh = torch.linalg.svd(h)
    d = torch.sign(torch.linalg.det(vh.transpose(-1, -2) @ u.transpose(-1, -2)))
    d = torch.where(d == 0, torch.ones_like(d), d)
    flip = torch.ones(*d.shape, 3, dtype=x.dtype, device=x.device)
    flip[..., 2] = d
    return vh.transpose(-1, -2) @ (flip[..., None] * u.transpose(-1, -2)), yc


def rotate(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """r [B, N, 3, 3] (the rotation of the group of every row), v [N, C, 3] or [N, 3] -> [B, N, C, 3] / [B, N, 3]"""
    return torch.einsum("bnij,ncj->bnci", r, v) if v.dim() == 3 else torch.einsum("bnij,nj->bni", r, v)
