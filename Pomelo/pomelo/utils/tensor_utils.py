"""Tensor utilities (the counterpart of PhysDock's utils/tensor_utils): rotations, superposition, radial basis functions."""
import torch

N_RBF = 8


def random_rotations(b: int, generator=None, device=None, dtype=torch.float32) -> torch.Tensor:
    """b uniformly distributed proper rotations [b, 3, 3] (the augmentation of a model that is not equivariant: no reflections, they would change the hand)"""
    q, r = torch.linalg.qr(torch.randn(b, 3, 3, generator=generator, device=device, dtype=dtype))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None, :]
    return q * torch.sign(torch.linalg.det(q))[:, None, None]                                  # a 3x3 matrix times -1 has the opposite determinant


def kabsch(p: torch.Tensor, q: torch.Tensor, proper: bool = True) -> torch.Tensor:
    """the structure p [N, 3] superposed on q [N, 3] (both centred here): by a proper rotation (proper=True), or by the best orthogonal transformation, a reflection included"""
    p0, q0 = p - p.mean(0), q - q.mean(0)
    u, _, vt = torch.linalg.svd(p0.T @ q0)
    if proper:
        d = torch.sign(torch.det(u @ vt))
        u = u * torch.tensor([1.0, 1.0, float(d)], device=p.device, dtype=p.dtype)
    return p0 @ (u @ vt) + q.mean(0)


def rbf(d: torch.Tensor, n: int = N_RBF, hi: float = 8.0) -> torch.Tensor:
    centers = torch.linspace(0, hi, n, device=d.device, dtype=d.dtype)
    return torch.exp(-((d[..., None] - centers) / (hi / n)) ** 2)
