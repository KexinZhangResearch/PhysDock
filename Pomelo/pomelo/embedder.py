"""Embedder: a molecule (pomelo.data) -> the six stored features of the trunk, at the level of the units, and the static atom features of the decoder.

    s_0e [Nu, C]    unit size, the mean of the atom features
    s_0o [Nu, C]    the stereo labels of the unit: sum over its stereo atoms of  label * W(atom features)   (a pseudoscalar times an even feature is a pseudoscalar)
    s_1o, s_1e      zero: vectors need a structure, which the decoder brings in from the noisy coordinates (with no structure an equivariant map has no vector to give)
    z_0e [Nu, Nu, C]  graph distance between the units (the smallest over their atoms), same unit
    z_0o [Nu, Nu, C]  label(unit i) * f(z_0e_ij) + label(unit j) * g(z_0e_ij), the label of a unit being the sum of the labels of its stereo atoms

and the static atom part: a_0e (element, charge, aromaticity, hybridisation, degree, hydrogens, ring), ap_0e (graph distance and bond type of every atom pair), the stereo table as a
scatter matrix, the pooling matrix atoms -> units.
"""
from typing import Dict

import torch
import torch.nn as nn

from pomelo import geometry as G
from pomelo.data import N_ELEMENTS
from pomelo.irreps import Mult

FAR = 9                      # graph distances 0 .. 8 and "far"


def graph_distance(n: int, bonds: torch.Tensor) -> torch.Tensor:
    """[n, n] shortest path length in bonds, FAR (9) beyond 8 and between disconnected atoms"""
    adj = torch.zeros(n, n)
    if len(bonds):
        adj[bonds[:, 0], bonds[:, 1]] = 1
        adj[bonds[:, 1], bonds[:, 0]] = 1
    spd = torch.full((n, n), FAR, dtype=torch.long)
    reach = torch.eye(n)
    spd[reach > 0] = 0
    for k in range(1, 9):
        reach = (reach @ adj).clamp(max=1)
        spd[(reach > 0) & (spd == FAR)] = k
    return spd


class Embedder(nn.Module):
    def __init__(self, mult_s: Mult, mult_z: Mult, atom_channels: int = 32, pair_channels: int = 16) -> None:
        super().__init__()
        self.mult_s, self.mult_z, self.atom_channels, self.pair_channels = dict(mult_s), dict(mult_z), atom_channels, pair_channels
        c = atom_channels
        self.emb = nn.ModuleDict({"z": nn.Embedding(N_ELEMENTS, c), "charge": nn.Embedding(5, c), "aromatic": nn.Embedding(2, c), "hyb": nn.Embedding(4, c),
                                  "degree": nn.Embedding(6, c), "n_h": nn.Embedding(5, c), "in_ring": nn.Embedding(2, c)})
        self.spd_emb, self.bond_emb = nn.Embedding(FAR + 1, pair_channels), nn.Embedding(5, pair_channels)
        self.size_emb = nn.Embedding(9, mult_s["0e"])
        self.s_0e = nn.Linear(c, mult_s["0e"])
        self.s_0o = nn.Linear(c, mult_s["0o"], bias=False)                      # 0e -> 0o: times the label, no bias
        self.z_spd, self.z_same = nn.Embedding(FAR + 1, mult_z["0e"]), nn.Embedding(2, mult_z["0e"])
        self.z_0o_i, self.z_0o_j = nn.Linear(mult_z["0e"], mult_z["0o"], bias=False), nn.Linear(mult_z["0e"], mult_z["0o"], bias=False)

    def forward(self, mol: Dict[str, torch.Tensor]) -> Dict[str, object]:
        dev = self.s_0e.weight.device
        mol = {k: v.to(dev) for k, v in mol.items()}
        n, nu = mol["z"].shape[0], int(mol["unit"].max()) + 1
        a0e = (self.emb["z"](mol["z"]) + self.emb["charge"]((mol["charge"] + 2).clamp(0, 4)) + self.emb["aromatic"](mol["aromatic"]) + self.emb["hyb"](mol["hyb"])
               + self.emb["degree"](mol["degree"].clamp(max=5)) + self.emb["n_h"](mol["n_h"].clamp(max=4)) + self.emb["in_ring"](mol["in_ring"]))
        spd = graph_distance(n, mol["bonds"].cpu()).to(dev)
        bond = torch.zeros(n, n, dtype=torch.long, device=dev)
        if len(mol["bonds"]):
            bond[mol["bonds"][:, 0], mol["bonds"][:, 1]] = mol["bonds"][:, 2]
            bond[mol["bonds"][:, 1], mol["bonds"][:, 0]] = mol["bonds"][:, 2]
        ap0e = self.spd_emb(spd) + self.bond_emb(bond)
        unit_onehot = torch.nn.functional.one_hot(mol["unit"], nu).T.to(a0e.dtype)             # [Nu, Na]
        size = unit_onehot.sum(1)
        pool = unit_onehot / size[:, None]
        ns = mol["stereo_sign"].shape[0]
        scatter = torch.zeros(n, ns, dtype=a0e.dtype, device=dev)
        if ns:
            scatter[mol["stereo_center"], torch.arange(ns, device=dev)] = 1
        tag = scatter @ mol["stereo_sign"].to(a0e.dtype)                                          # [Na]  the label at the atom of the stereocentre (0 elsewhere)
        tag_u = unit_onehot @ tag                                                                # [Nu]  the sum of the labels of the unit
        s_0e = self.s_0e(pool @ a0e) + self.size_emb(size.long().clamp(max=8))
        s_0o = unit_onehot @ (tag[:, None] * self.s_0o(a0e))                                      # 0o * 0e = 0o
        umin = torch.where(unit_onehot[:, None, :, None].bool() & unit_onehot[None, :, None, :].bool(), spd[None, None], torch.full_like(spd, 99)[None, None]).amin((-1, -2))
        umin = umin.clamp(max=FAR)
        z_0e = self.z_spd(umin) + self.z_same(torch.eye(nu, dtype=torch.long, device=dev))
        z_0o = (G.scale(self.z_0o_i(z_0e), "0e", tag_u[:, None, None], "0o")[0] + G.scale(self.z_0o_j(z_0e), "0e", tag_u[None, :, None], "0o")[0])
        zeros = lambda c: torch.zeros(nu, c, 3, dtype=a0e.dtype, device=dev)
        return {"s": {"0e": s_0e, "0o": s_0o, "1o": zeros(self.mult_s["1o"]), "1e": zeros(self.mult_s["1e"])}, "z": {"0e": z_0e, "0o": z_0o},
                "a0e": a0e, "ap0e": ap0e, "scatter": scatter, "tag": tag, "stereo_center": mol["stereo_center"], "stereo_nbrs": mol["stereo_nbrs"],
                "pool": pool, "atom_to_unit": mol["unit"], "n_atoms": n, "n_units": nu}
