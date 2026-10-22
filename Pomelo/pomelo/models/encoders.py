"""The encoders of Pomelo: the Embedder (a molecule -> the six stored features of the units and the static atom features) and the EncoderTrunk (the counterpart of PhysDock's atom
embedder + Pairformer: the reference conformers in, the parity-equivariant Pairformer on the units).

Embedder: a molecule (pomelo.data) -> the six stored features of the trunk, at the level of the units, and the static atom features of the decoder.

    s_0e [Nu, C]    unit size, the mean of the atom features
    s_0o [Nu, C]    the stereo labels of the unit: sum over its stereo atoms of  label * W(atom features)   (a pseudoscalar times an even feature is a pseudoscalar)
    s_1o, s_1e      zero: vectors need a structure, which the decoder brings in from the noisy coordinates (with no structure an equivariant map has no vector to give)
    z_0e [Nu, Nu, C]  graph distance between the units (the smallest over their atoms), same unit, residue offset
    z_0o [Nu, Nu, C]  label(unit i) * f(z_0e_ij) + label(unit j) * g(z_0e_ij), the label of a unit being the sum of the labels of its stereo atoms

and the static atom part: a_0e (element, charge, aromaticity, hybridisation, degree, hydrogens, ring), ap_0e (graph distance and bond type of every atom pair), the stereo table as a
scatter matrix, the pooling matrix atoms -> units.

EncoderTrunk: parity-equivariant, with the reference positions as input.

Input: a molecule / system of pomelo.data with the reference conformers: ref_pos [Na, 3] (every conformer in a frame of its own: nothing relates the frames of two conformers) and
conformer [Na]. No stereo label is read: the chirality is the one of the reference conformers (the signed volumes of the ordered neighbours of the stereocentres, pseudoscalars).

  1. atom stage, per conformer (atoms attend only inside their conformer):  a_0e (static atom features), a_0o (signed volume), a_1o (positions relative to the unit and the conformer),
     a_1e (their cross product), pair bias ap from the reference distances; SingleBlocks on the atoms; typed pooling atoms -> units.
  2. unit stage: n_blocks TrunkBlocks (pair block: single -> pair, triangle multiplication out / in, triangle attention start / end, transition; then single block). The vector terms of a
     block act between units of the same conformer only (`same`): two conformers have no common frame. A residue is a conformer of its own (no vector terms between residues, the
     pair representation is scalars), a ligand is one conformer (all its fragments share a frame).

Symmetry: invariant to an independent rotation and translation of every conformer; parity-equivariant under a reflection of all of them (0e unchanged, 0o negated, the vectors of a
unit transform with the matrix of its conformer). The vectors (s_1o, s_1e) are in the frame of their conformer: only s_0e, s_0o, z_0e, z_0o can be handed to a decoder that works in
the frame of the noisy structure; `condition` returns those.
"""
from typing import Dict

import torch
import torch.nn as nn

from pomelo.configs import Config
from pomelo.data.molecule import FAR, N_ELEMENTS, add_graph_features, signed_volume
from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.irreps import Mult, TypedLinear, TypedPool
from pomelo.models.primitives.transformers import SingleBlock, TrunkBlock
from pomelo.utils.tensor_utils import N_RBF, rbf

R_MAX = 32


class Embedder(nn.Module):
    def __init__(self, mult_s: Mult, mult_z: Mult, atom_channels: int = 32, pair_channels: int = 16, relpos: bool = False, max_unit_size: int = 8,
                 use_labels: bool = True) -> None:
        """use_labels: the odd features (s_0o, z_0o, the tag of the atoms) come from the stereo labels of the table (the labels are the input). False: the embedder gives the static
        even features only, and no label is read: the chirality has to come from the reference positions (pomelo.encoder_trunk.EncoderTrunk)"""
        super().__init__()
        self.relpos, self.max_unit_size, self.use_labels = relpos, max_unit_size, use_labels
        self.mult_s, self.mult_z, self.atom_channels, self.pair_channels = dict(mult_s), dict(mult_z), atom_channels, pair_channels
        c = atom_channels
        self.emb = nn.ModuleDict({"z": nn.Embedding(N_ELEMENTS, c), "charge": nn.Embedding(5, c), "aromatic": nn.Embedding(2, c), "hyb": nn.Embedding(4, c),
                                  "degree": nn.Embedding(6, c), "n_h": nn.Embedding(5, c), "in_ring": nn.Embedding(2, c)})
        self.spd_emb, self.bond_emb = nn.Embedding(FAR + 1, pair_channels), nn.Embedding(5, pair_channels)
        self.size_emb = nn.Embedding(max_unit_size + 1, mult_s["0e"])
        self.s_0e = nn.Linear(c, mult_s["0e"])
        if use_labels:
            self.s_0o = nn.Linear(c, mult_s["0o"], bias=False)                  # 0e -> 0o: times the label, no bias
        self.z_spd, self.z_same = nn.Embedding(FAR + 1, mult_z["0e"]), nn.Embedding(2, mult_z["0e"])
        if relpos:                      # AF3's relative position of the residues of a chain (pomelo.protein: `residue_index`, `chain`): 65 offsets, "other chain"
            self.z_rel = nn.Embedding(2 * R_MAX + 2, mult_z["0e"])
        if use_labels:
            self.z_0o_i, self.z_0o_j = nn.Linear(mult_z["0e"], mult_z["0o"], bias=False), nn.Linear(mult_z["0e"], mult_z["0o"], bias=False)

    def forward(self, mol: Dict[str, torch.Tensor]) -> Dict[str, object]:
        dev = self.s_0e.weight.device
        mol = {k: v.to(dev) for k, v in mol.items()}
        n, nu = mol["z"].shape[0], int(mol["unit"].max()) + 1
        a0e = (self.emb["z"](mol["z"]) + self.emb["charge"]((mol["charge"] + 2).clamp(0, 4)) + self.emb["aromatic"](mol["aromatic"]) + self.emb["hyb"](mol["hyb"])
               + self.emb["degree"](mol["degree"].clamp(max=5)) + self.emb["n_h"](mol["n_h"].clamp(max=4)) + self.emb["in_ring"](mol["in_ring"]))
        if "spd" not in mol:                                                                       # the static graph features are computed once with the data
            mol = {**mol, **{k: v.to(dev) for k, v in add_graph_features({k: v.cpu() for k, v in mol.items()}).items() if k in ("spd", "bond", "unit_spd")}}
        spd, bond = mol["spd"], mol["bond"]
        ap0e = self.spd_emb(spd) + self.bond_emb(bond)
        unit_onehot = torch.nn.functional.one_hot(mol["unit"], nu).T.to(a0e.dtype)             # [Nu, Na]
        size = unit_onehot.sum(1)
        pool = unit_onehot / size[:, None]
        ns = mol["stereo_sign"].shape[0]
        scatter = torch.zeros(n, ns, dtype=a0e.dtype, device=dev)
        if ns:
            scatter[mol["stereo_center"], torch.arange(ns, device=dev)] = 1
        tag = scatter @ mol["stereo_sign"].to(a0e.dtype) if self.use_labels else torch.zeros(n, dtype=a0e.dtype, device=dev)       # [Na] the label at the atom of the stereocentre
        tag_u = unit_onehot @ tag                                                                # [Nu]  the sum of the labels of the unit
        s_0e = self.s_0e(pool @ a0e) + self.size_emb(size.long().clamp(max=self.max_unit_size))
        s_0o = unit_onehot @ (tag[:, None] * self.s_0o(a0e)) if self.use_labels else torch.zeros(nu, self.mult_s["0o"], dtype=a0e.dtype, device=dev)    # 0o * 0e = 0o
        z_0e = self.z_spd(mol["unit_spd"]) + self.z_same(torch.eye(nu, dtype=torch.long, device=dev))
        if self.relpos:
            res, chain = mol["residue_index"], mol["chain"]
            off = (res[:, None] - res[None, :]).clamp(-R_MAX, R_MAX) + R_MAX
            z_0e = z_0e + self.z_rel(torch.where(chain[:, None] == chain[None, :], off, torch.full_like(off, 2 * R_MAX + 1)))
        z_0o = ((G.scale(self.z_0o_i(z_0e), "0e", tag_u[:, None, None], "0o")[0] + G.scale(self.z_0o_j(z_0e), "0e", tag_u[None, :, None], "0o")[0]) if self.use_labels
                else torch.zeros(nu, nu, self.mult_z["0o"], dtype=a0e.dtype, device=dev))
        zeros = lambda c: torch.zeros(nu, c, 3, dtype=a0e.dtype, device=dev)
        return {"s": {"0e": s_0e, "0o": s_0o, "1o": zeros(self.mult_s["1o"]), "1e": zeros(self.mult_s["1e"])}, "z": {"0e": z_0e, "0o": z_0o},
                "a0e": a0e, "ap0e": ap0e, "scatter": scatter, "tag": tag, "stereo_center": mol["stereo_center"], "stereo_nbrs": mol["stereo_nbrs"],
                "pool": pool, "atom_to_unit": mol["unit"], "n_atoms": n, "n_units": nu}


REF_SCALE = 0.25                       # reference positions in Angstrom: distances / 4 A for the RBF (the decoder works in units of 4 A of the normalised structure)


class EncoderTrunk(nn.Module):
    def __init__(self, cfg: Config, n_atom_blocks: int = 1, n_blocks: int = 2, heads_z: int = 4, channels: int = 8) -> None:
        super().__init__()
        self.cfg = cfg
        ma, mp, ms, mz = cfg.mult_a, cfg.mult_ap, cfg.mult_s, cfg.mult_z
        self.embedder = Embedder(ms, mz, cfg.atom_static, cfg.pair_static, cfg.relpos, cfg.max_unit_size, use_labels=False)
        self.a_in_0e = nn.Linear(cfg.atom_static, ma["0e"])
        self.a_in_0o = TypedLinear({"0o": 1}, {"0o": ma["0o"]}, bias=False)
        self.a_in_1o = TypedLinear({"1o": 2}, {"1o": ma["1o"]}, bias=False)
        self.a_in_1e = TypedLinear({"1e": 1}, {"1e": ma["1e"]}, bias=False)
        self.ap_in = nn.Linear(cfg.pair_static + N_RBF, mp["0e"])
        self.ap_vol_i, self.ap_vol_j = nn.Linear(mp["0e"], mp["0o"], bias=False), nn.Linear(mp["0e"], mp["0o"], bias=False)
        self.atom_blocks = nn.ModuleList([SingleBlock(ma, mp, cfg.heads, cfg.product_channels) for _ in range(n_atom_blocks)])
        self.pool = TypedPool(ma, ms)
        self.blocks = nn.ModuleList([TrunkBlock(ms, mz, cfg.heads, heads_z, channels, cfg.product_channels) for _ in range(n_blocks)])

    def forward(self, mol: Dict[str, torch.Tensor]) -> Dict[str, object]:
        dev = self.a_in_0e.weight.device
        mol = {k: v.to(dev) for k, v in mol.items()}
        st = self.embedder(mol)
        n, nu = st["n_atoms"], st["n_units"]
        ref, conf, unit_conf = mol["ref_pos"], mol["conformer"], mol["unit_conformer"]
        dtype = ref.dtype
        same_atom = conf[:, None] == conf[None, :]
        same_unit = unit_conf[:, None] == unit_conf[None, :]
        pool, a2u, scatter = st["pool"].to(dtype), st["atom_to_unit"], st["scatter"].to(dtype)
        # the chirality of the reference conformers: the signed volume at every stereocentre (0 at the other atoms); a pseudoscalar of the atom
        vol = scatter @ signed_volume(ref, mol["stereo_center"], mol["stereo_nbrs"]) if mol["stereo_center"].numel() else torch.zeros(n, dtype=dtype, device=dev)
        centre_u = pool @ ref                                                                                  # the unit centre, in the frame of its conformer
        conf_mean = torch.zeros(int(conf.max()) + 1, 3, dtype=dtype, device=dev).index_add_(0, conf, ref) / torch.bincount(conf).clamp(min=1)[:, None].to(dtype)
        vec = torch.stack([ref - centre_u[a2u], ref - conf_mean[conf]], dim=-2) * REF_SCALE                         # [Na, 2, 3] polar, relative: free of the translation
        a = {"0e": self.a_in_0e(st["a0e"]), **self.a_in_0o({"0o": vol[:, None]}), **self.a_in_1o({"1o": vec}),
             **self.a_in_1e({"1e": torch.linalg.cross(vec[:, 0], vec[:, 1], dim=-1)[:, None]})}
        d = torch.cdist(ref, ref) * REF_SCALE
        ap0e = self.ap_in(torch.cat([st["ap0e"], rbf(d)], -1))
        ap = {"0e": ap0e, "0o": G.scale(self.ap_vol_i(ap0e), "0e", vol[:, None, None], "0o")[0] + G.scale(self.ap_vol_j(ap0e), "0e", vol[None, :, None], "0o")[0]}
        mask_a, mask_u = torch.ones(n, dtype=dtype, device=dev), torch.ones(nu, dtype=dtype, device=dev)
        for blk in self.atom_blocks:
            a = blk(a, ap, ref, mask_a, same_atom, same_atom)                                                  # the atoms of a conformer attend to each other only
        s = self.pool(a, pool)
        s = {**s, "0e": s["0e"] + st["s"]["0e"]}                                                              # + the static unit features (pooled atoms, size): no label in them
        z = {"0e": st["z"]["0e"], "0o": torch.zeros(nu, nu, self.cfg.mult_z["0o"], dtype=dtype, device=dev)}
        for blk in self.blocks:
            s, z = blk(s, z, centre_u, mask_u, same_unit)
        return {**st, "s": s, "z": z, "tag": vol, "ref_unit_centres": centre_u}

    def condition(self, st: Dict[str, object]) -> Dict[str, object]:
        """what a decoder in another frame can use: the scalars of s and z (the vectors are in the frames of their conformers)"""
        return {**st, "s": {k: st["s"][k] for k in ("0e", "0o")}}
