"""Moving the vectors of a conformer from the frame of its reference positions (the encoder) into the frame of the noisy structure (the decoder).

Every conformer k has its reference positions R (centred on the conformer, in a frame of their own) and its atoms in the noisy structure X (centred the same way). The transport is the
linear least-squares map between them, regularised,

    A_k = (sum_a X_a R_a^T) (sum_a R_a R_a^T + lambda I)^-1,

normalised by its size, plus  cof(A_k) N_k  with N_k = lambda (sum R R^T + lambda I)^-1 (the projector onto the directions the reference does not span: the normal of a planar
conformer, which the least-squares map sends to 0; the cofactor matrix carries it as the cross product of two in-plane directions).

Symmetry (the group of the problem: the noisy structure transformed by Q, the reference of every conformer by its own P_k, det P_k = det Q):  A_k -> Q A_k P_k^T, so a vector v of the
encoder, polar (v -> P_k v) or axial (v -> det(P_k) P_k v), arrives as  Q (A_k v), respectively det(Q) Q (A_k v): of the right kind in the decoder. No SVD, no sign choice: smooth in X,
and the inverse is a constant of the molecule. At high noise X carries no information and the transported vectors are noise, of bounded size.
"""
import torch


def reference_moments(ref_rel: torch.Tensor, conformer: torch.Tensor, n: int, rel_lambda: float = 1e-3):
    """the constants of the transport of every conformer: inv_k = (sum R R^T + lambda I)^-1 and null_k = lambda inv_k, [n, 3, 3]; lambda = rel_lambda * trace(sum R R^T) / 3 + 1e-6 (a
    scalar of the conformer, so the constants transform as P inv P^T)"""
    c = torch.zeros(n, 3, 3, dtype=ref_rel.dtype, device=ref_rel.device).index_add_(0, conformer, ref_rel[:, :, None] * ref_rel[:, None, :])
    lam = rel_lambda * torch.diagonal(c, dim1=-2, dim2=-1).sum(-1) / 3 + 1e-6
    inv = torch.linalg.inv(c + lam[:, None, None] * torch.eye(3, dtype=c.dtype, device=c.device))
    return inv, lam[:, None, None] * inv


def cofactor(a: torch.Tensor) -> torch.Tensor:
    """the cofactor matrix of [..., 3, 3] (rows: the cross products of the rows of a): cof(a) = det(a) a^-T, also for a singular a;  cof(a) (u x w) = (a u) x (a w)"""
    r = a.unbind(-2)
    return torch.stack([torch.linalg.cross(r[1], r[2], dim=-1), torch.linalg.cross(r[2], r[0], dim=-1), torch.linalg.cross(r[0], r[1], dim=-1)], dim=-2)


def transport(x_rel: torch.Tensor, ref_rel: torch.Tensor, conformer: torch.Tensor, inv: torch.Tensor, null: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """[n, 3, 3] the map of every conformer from its reference frame into the frame of x: x_rel, ref_rel [Na, 3] (each centred on its conformer), conformer [Na], inv / null of
    reference_moments. A rigid copy (x = Q R) gives Q, up to the regularisation."""
    m = torch.zeros_like(inv).index_add(0, conformer, x_rel[:, :, None] * ref_rel[:, None, :])
    a = m @ inv
    a = a / torch.sqrt(a.square().sum((-1, -2), keepdim=True) / 3 + eps)
    return a + cofactor(a) @ null


def conformer_centred(x: torch.Tensor, conformer: torch.Tensor, n: int) -> torch.Tensor:
    """x [Na, 3] minus the mean of its conformer"""
    count = torch.bincount(conformer, minlength=n).clamp(min=1).to(x.dtype)
    mean = torch.zeros(n, 3, dtype=x.dtype, device=x.device).index_add(0, conformer, x) / count[:, None]          # out of place: it runs under vmap
    return x - mean[conformer]


def move(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """v [N, C, 3] times the matrix t [N, 3, 3] of its row"""
    return torch.einsum("nij,ncj->nci", t, v)
