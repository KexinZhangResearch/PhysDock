"""The training loss of Pomelo (as PhysDock2 / AF3, without the terms the experiments do not use): x_mse, the EDM-weighted squared error of the centred coordinates (no superposition of
the target on the prediction: the network is equivariant, its output is in the frame of its input), and the smooth lDDT. Coordinates are normalised by sigma_data = 16 Angstrom."""
import torch

from pomelo.models.diffusion import SIGMA_DATA


def x_mse(pred: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """EDM-weighted squared error of the centred coordinates (normalised units): mean over the samples of w(sigma) * mean over atoms |D - x0|^2, w = (sigma^2 + 1) / sigma^2 / 3"""
    d = (pred - pred.mean(-2, keepdim=True)) - (x0 - x0.mean(-2, keepdim=True))
    w = (sigma ** 2 + 1) / sigma ** 2 / 3
    return (w * d.square().sum(-1).mean(-1)).mean()


def smooth_lddt(pred: torch.Tensor, x0: torch.Tensor, cutoff: float = 15.0) -> torch.Tensor:
    """AF3's smooth lDDT as a loss (PhysDock's form, 0.25 * sum of sigmoid(delta - threshold) over the thresholds 0.5, 1, 2, 4 A, averaged over the pairs closer than `cutoff` A in the
    target, self pairs excluded): the loss of a perfect structure is 0.196, not 0. pred [B, N, 3], x0 [N, 3] or [B, N, 3] (normalised units)"""
    n = x0.shape[-2]
    dp = torch.cdist(pred, pred) * SIGMA_DATA
    dt = torch.cdist(x0, x0) * SIGMA_DATA
    delta = (dp - dt).abs()
    eps = 0.25 * sum(torch.sigmoid(delta - t) for t in (0.5, 1.0, 2.0, 4.0))
    pairs = ((dt < cutoff) & ~torch.eye(n, dtype=torch.bool, device=x0.device)).to(eps.dtype)
    return ((eps * pairs).sum((-1, -2)) / pairs.sum((-1, -2))).mean()
