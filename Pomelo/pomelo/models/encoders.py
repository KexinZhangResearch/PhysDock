"""The encoder of Pomelo, as PhysDock's: the conformer features in (InputEmbedder), an atom transformer inside every conformer, the atoms pooled to the tokens, a Pairformer on the tokens.
The difference: parity-equivariant layers, so the reference conformers enter as vectors and the hand of a molecule is the hand of its reference conformer (no stereo label is read).

Tokens as in PhysDock: a standard residue is one token (and one conformer), every atom of a ligand is one token (the ligand is one conformer).

InputEmbedder: the static (scalar, even) features of pomelo.data.molecule, shared with the baselines
    a0e [Na, atom_static]    element, charge, aromatic, hybridisation, degree, hydrogens, ring (in the reference molecule)
    ap0e [Na, Na, pair_static]   graph distance and bond type inside a conformer (FAR / none between two conformers: "another conformer")
    s0e [Nt, C]              the mean of the atom features of the token + restype
    z0e [Nt, Nt, C]          graph distance of the tokens inside a conformer, AF3's relative position (residue offset in a chain, token offset in a residue, other chain)

EncoderTrunk: parity-equivariant. Every conformer has its reference positions in a frame of its own (nothing relates two of them), so:
  1. atoms:  a_0e = Linear(a0e);  a_1o = Linear[ref - conformer centre, ref - mean of the bonded neighbours] (relative: free of the translation);  a_1e = Linear(their cross product);
             a_0o = Linear(the mean of ap_0o over the conformer, below);  ap_0e = Linear[ap0e, RBF(|ref_i - ref_j|)] with the RBF inside a conformer only (a
             distance between two frames has no meaning);  ap_0o = vol_ij * Linear(ap_0e), vol_ij the signed volume of the bond directions of i and j (ref - mean of the bonded
             neighbours) and the direction from i to j, inside a conformer: the hand of the reference, a pseudoscalar of the pair.
             SingleBlocks, the atoms of a conformer attending to each other only; then ap += PairFromSingle(a) (the atom pairs of the encoder are a condition of the decoder).
  2. tokens: s = TypedPool(a) + (s0e on 0e);  z = (z0e, 0);  TrunkBlocks (pair block: single -> pair, triangle multiplication out / in, triangle attention start / end, transition; single
             block). The vector terms act between tokens of the same conformer only: a residue has no vector terms with another residue (their pair representation is scalars).

Symmetry: invariant to an independent rotation and translation of every conformer, parity-equivariant under a reflection of all of them: 0e unchanged, 0o negated, the vectors of an atom
or a token transform with the matrix of its conformer. The output is the 12 conditions of the decoder (atoms: a 0e 0o 1o 1e, ap 0e 0o; tokens: s 0e 0o 1o 1e, z 0e 0o) and the constants
that move the vectors from the frame of their conformer into the frame of the noisy structure (pomelo.models.primitives.frames).
"""
from typing import Dict

import torch
import torch.nn as nn

from pomelo.configs import Config
from pomelo.data.molecule import FAR, N_ELEMENTS, N_RESTYPE, add_features
from pomelo.models.primitives import frames as Fr
from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.irreps import TypedLinear, TypedPool, add
from pomelo.models.primitives.transformers import SingleBlock, TrunkBlock
from pomelo.models.primitives.triangle import PairFromSingle
from pomelo.utils.tensor_utils import N_RBF, rbf

R_MAX = 32
REF_SCALE = 0.25                       # reference positions in Angstrom, in units of 4 A (the decoder works in units of 4 A of the normalised structure)


def neighbour_mean(bond: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[Na, Na] the matrix of the mean over the bonded neighbours (a row of zeros for an atom with none), in the dtype of the coordinates (1/3 in float32 times coordinates in a frame
    20 A away is an error of 1e-6)"""
    adj = (bond > 0).to(dtype)
    return adj / adj.sum(1, keepdim=True).clamp(min=1)


class InputEmbedder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        c, ms, mz = cfg.atom_static, cfg.mult_s, cfg.mult_z
        self.emb = nn.ModuleDict({"ref_element": nn.Embedding(N_ELEMENTS, c), "ref_charge": nn.Embedding(5, c), "ref_is_aromatic": nn.Embedding(2, c),
                                  "ref_hybridization": nn.Embedding(4, c), "ref_degree": nn.Embedding(6, c), "ref_n_h": nn.Embedding(5, c), "ref_in_ring": nn.Embedding(2, c)})
        self.spd_emb, self.bond_emb = nn.Embedding(FAR + 1, cfg.pair_static), nn.Embedding(5, cfg.pair_static)
        self.s_0e, self.restype = nn.Linear(c, ms["0e"]), nn.Embedding(N_RESTYPE, ms["0e"])
        self.z_spd = nn.Embedding(FAR + 1, mz["0e"])
        self.z_res, self.z_tok = nn.Embedding(2 * R_MAX + 2, mz["0e"]), nn.Embedding(2 * R_MAX + 2, mz["0e"])

    def forward(self, mol: Dict[str, torch.Tensor]) -> Dict[str, object]:
        dev = self.s_0e.weight.device
        if "ref_spd" not in mol:                                                              # the static features are computed once with the data
            mol = add_features({k: v.cpu() for k, v in mol.items()})
        mol = {k: v.to(dev) for k, v in mol.items()}
        clamp = {"ref_charge": lambda v: (v + 2).clamp(0, 4), "ref_degree": lambda v: v.clamp(max=5), "ref_n_h": lambda v: v.clamp(max=4)}
        a0e = sum(e(clamp.get(k, lambda v: v)(mol[k])) for k, e in self.emb.items())
        ap0e = self.spd_emb(mol["ref_spd"]) + self.bond_emb(mol["ref_bond"])
        a2t, conf, t2c = mol["atom_to_token"], mol["atom_to_conformer"], mol["token_to_conformer"]
        n, nt = a2t.shape[0], t2c.shape[0]
        onehot = torch.nn.functional.one_hot(a2t, nt).T.to(a0e.dtype)                       # [Nt, Na]
        pool = onehot / onehot.sum(1, keepdim=True)
        s0e = self.s_0e(pool @ a0e) + self.restype(mol["restype"])
        res, chain, tok = mol["residue_index"], mol["asym_id"], torch.arange(nt, device=dev)
        same_chain = chain[:, None] == chain[None, :]
        other = torch.full((nt, nt), 2 * R_MAX + 1, dtype=torch.long, device=dev)
        d_res = torch.where(same_chain, (res[:, None] - res[None, :]).clamp(-R_MAX, R_MAX) + R_MAX, other)
        d_tok = torch.where(same_chain & (res[:, None] == res[None, :]), (tok[:, None] - tok[None, :]).clamp(-R_MAX, R_MAX) + R_MAX, other)
        z0e = self.z_spd(mol["token_spd"]) + self.z_res(d_res) + self.z_tok(d_tok)
        return {"a0e": a0e, "ap0e": ap0e, "s0e": s0e, "z0e": z0e, "pool": pool, "atom_to_token": a2t, "atom_to_conformer": conf, "token_to_conformer": t2c,
                "neighbour_mean": neighbour_mean(mol["ref_bond"], mol["ref_pos"].dtype), "n_atoms": n, "n_tokens": nt, "n_conformers": int(conf.max()) + 1, "mol": mol}


class EncoderTrunk(nn.Module):
    def __init__(self, cfg: Config, heads_z: int = 4, channels: int = 8, pair_channels: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        ma, mp, ms, mz = cfg.mult_a, cfg.mult_ap, cfg.mult_s, cfg.mult_z
        self.embedder = InputEmbedder(cfg)
        self.a_in_0e = nn.Linear(cfg.atom_static, ma["0e"])
        self.a_in_vec = TypedLinear({"1o": 2, "1e": 1}, {"1o": ma["1o"], "1e": ma["1e"]}, bias=False)
        self.ap_in = nn.Linear(cfg.pair_static + N_RBF, mp["0e"])
        self.ap_vol = nn.Linear(mp["0e"], mp["0o"], bias=False)
        self.a_in_0o = nn.Linear(mp["0o"], ma["0o"], bias=False)                                # 0o -> 0o: no bias
        self.atom_blocks = nn.ModuleList([SingleBlock(ma, mp, cfg.heads, cfg.product_channels) for _ in range(cfg.n_encoder_atom_blocks)])
        self.atom_pair = PairFromSingle(ma, mp, channels=pair_channels)
        self.pool = TypedPool(ma, ms)
        self.blocks = nn.ModuleList([TrunkBlock(ms, mz, cfg.heads, heads_z, channels, cfg.product_channels) for _ in range(cfg.n_encoder_blocks)])

    def forward(self, mol: Dict[str, torch.Tensor]) -> Dict[str, object]:
        st = self.embedder(mol)
        mol = st["mol"]
        ref, conf, t2c, pool = mol["ref_pos"], st["atom_to_conformer"], st["token_to_conformer"], st["pool"]
        dt, dev = ref.dtype, ref.device
        n, nt, nc = st["n_atoms"], st["n_tokens"], st["n_conformers"]
        same_atom, same_token = conf[:, None] == conf[None, :], t2c[:, None] == t2c[None, :]
        conf_mean = torch.zeros(nc, 3, dtype=dt, device=dev).index_add(0, conf, ref) / torch.bincount(conf, minlength=nc).clamp(min=1)[:, None].to(dt)
        nbr = st["neighbour_mean"].to(dt)
        has_nbr = nbr.sum(1, keepdim=True) > 0
        vec = torch.stack([ref - conf_mean[conf], torch.where(has_nbr, ref - nbr @ ref, torch.zeros_like(ref))], dim=-2) * REF_SCALE      # [Na, 2, 3] polar, relative
        d = rbf(torch.cdist(ref, ref, compute_mode="donot_use_mm_for_euclid_dist") * REF_SCALE) * same_atom[..., None].to(dt)                                # the reference distances inside a conformer only
        ap0e = self.ap_in(torch.cat([st["ap0e"].to(dt), d], -1))
        w = (ref[None, :] - ref[:, None]) * REF_SCALE
        w = w / torch.sqrt(1 + w.square().sum(-1, keepdim=True))
        vol = torch.where(same_atom, G.triple(vec[:, None, 1], "1o", vec[None, :, 1], "1o", w, "1o")[0], torch.zeros_like(w[..., 0]))         # [Na, Na] 0o
        ap = {"0e": ap0e, "0o": G.scale(self.ap_vol(ap0e), "0e", vol[..., None], "0o")[0]}
        size = same_atom.sum(1, keepdim=True).to(dt)
        a = {"0e": self.a_in_0e(st["a0e"].to(dt)), "0o": self.a_in_0o(ap["0o"].sum(1) / size),                    # the mean over the conformer of the pair pseudoscalars
             **self.a_in_vec({"1o": vec, "1e": torch.linalg.cross(vec[:, 0], vec[:, 1], dim=-1)[:, None]})}
        mask_a, mask_t = torch.ones(n, dtype=dt, device=dev), torch.ones(nt, dtype=dt, device=dev)
        for blk in self.atom_blocks:
            a = blk(a, ap, ref, mask_a, same_atom, same_atom)                                                  # the atoms of a conformer attend to each other only
        ap = add(ap, self.atom_pair(a, ref, mask_a, same_atom))
        s = self.pool(a, pool.to(dt))
        s = {**s, "0e": s["0e"] + st["s0e"].to(dt)}
        z = {"0e": st["z0e"].to(dt), "0o": torch.zeros(nt, nt, self.cfg.mult_z["0o"], dtype=dt, device=dev)}
        centre_t = pool.to(dt) @ ref                                                                           # the token centre, in the frame of its conformer
        for blk in self.blocks:
            s, z = blk(s, z, centre_t, mask_t, same_token)
        ref_rel = (ref - conf_mean[conf]) * REF_SCALE
        inv, null = Fr.reference_moments(ref_rel, conf, nc)
        frames = {"ref_rel": ref_rel, "inv": inv, "null": null}
        keep = ("pool", "atom_to_token", "atom_to_conformer", "token_to_conformer", "neighbour_mean", "n_atoms", "n_tokens", "n_conformers")
        return {**{k: st[k] for k in keep}, "a": a, "ap": ap, "s": s, "z": z, "frames": frames}
