"""Metrics of sampled structures (the counterpart of PhysDock's training/metrics): the RMSD after superposition and the share of the stereocentres with the labelled hand."""
import torch

from pomelo.data.molecule import signed_volume
from pomelo.utils.tensor_utils import kabsch


def rmsd(p: torch.Tensor, q: torch.Tensor, proper: bool = True) -> float:
    return float((kabsch(p, q, proper) - q).square().sum(-1).mean().sqrt())


def chirality_agreement(x: torch.Tensor, center, nbrs, sign) -> float:
    """the share of the stereocentres whose signed volume in the structure x [N, 3] has the sign of the label"""
    if not center.numel():
        return 1.0
    return float((torch.sign(signed_volume(x, center, nbrs)) == sign).float().mean())
