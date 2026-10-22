"""The encoder of Pomelo, laid out as PhysDock's (PhysDock/models/encoders.py: AtomEmbedder, TokenEmbedder with the template embedder, Pairformer), with parity-equivariant
layers: the reference conformers and the template enter as vectors, and the hand of a molecule is the hand of its reference conformer (no stereo label is read). Not yet here: the MSA
module (Evoformer) and the outer product of s into z of PhysDock.

The inputs are PhysDock's features (pomelo.data.features). Tokens as in PhysDock (a standard residue is one token and one conformer; every atom of a ligand is one token).

  InputEmbedder (PhysDock's: the even, static part)
      a0e = linear_c(ref_feat without the positions and the RDKit chirality tag)
      ap0e = ffn( c_l(relu a0e)_i + c_m(relu a0e)_j + linear_d[1 / (1 + d), RBF(d)] * same conformer ),  d = |ref_i - ref_j|  (linear_p(ref_i - ref_j) of PhysDock is a vector:
             it is a_1o here)
      s0e = target_feat + key_res_feat + is_pocket (linear),  z0e = RelPos (AF3: residue offset, chain offset, same entity, ligand_pair_feat) + linear(token_bonds)
  EncoderTrunk
      atoms   a_1o = [ref - conformer centre, ref - mean of the bonded neighbours], a_1e = their cross product, ap_0o = signed volume of the bond directions of i, j and the direction
              i -> j (inside a conformer) times a linear map of ap_0e, a_0o = the mean of ap_0o over the conformer; atom blocks inside a conformer; ap += PairFromSingle(a)
      tokens  s = TypedPool(a) + s0e;  z_0e = z0e + l_i(s_0e)_i + l_j(s_0e)_j,  z_0o = l_i(s_0o)_i + l_j(s_0o)_j  (PhysDock's linear_s_i, linear_s_j, typed)
      template  z += TemplateEmbedder(z)  (TemplatePairEmbedder of PhysDock: on the protein pairs of one chain; the template positions give the directions and the out-of-plane
              vectors of the triangle updates, so the template has a hand)
      Pairformer  TrunkBlocks (single -> pair, triangle multiplication out / in, triangle attention start / end, transition; single block)

Symmetry: invariant to an independent rotation and translation of every conformer and of the template, parity-equivariant under a reflection of all of them. The output is the 12
conditions of the decoder (atoms: a 0e 0o 1o 1e, ap 0e 0o; tokens: s 0e 0o 1o 1e, z 0e 0o) and the constants that move the vectors from the frame of their conformer into the frame
of the noisy structure (pomelo.models.primitives.frames).
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo.configs import Config
from pomelo.data import features as Fe
from pomelo.models.primitives import frames as Fr
from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.irreps import Gate, NormLinear, TypedLinear, TypedNorm, TypedPool, add
from pomelo.models.primitives.transformers import AtomBlock, TemplateBlock, TrunkBlock
from pomelo.models.primitives.triangle import PairFromSingle
from pomelo.utils.tensor_utils import N_RBF, rbf

REF_SCALE = 0.25                       # reference positions in Angstrom, in units of 4 A (the decoder works in units of 4 A of the normalised structure)
ATOM_COLUMNS = [i for i in range(Fe.REF_DIM) if not (Fe.POS.start <= i < Fe.POS.stop or Fe.CHIRALITY.start <= i < Fe.CHIRALITY.stop)]


def neighbour_mean(bonded: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[Na, Na] the matrix of the mean over the bonded neighbours (a row of zeros for an atom with none), in the dtype of the coordinates"""
    adj = bonded.to(dtype)
    return adj / adj.sum(1, keepdim=True).clamp(min=1)


def conformer_mean(x: torch.Tensor, conf: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, 3, dtype=x.dtype, device=x.device).index_add(0, conf, x) / torch.bincount(conf, minlength=n).clamp(min=1)[:, None].to(x.dtype)


class SwiGLU(nn.Module):
    """x + w2(silu(w1 x) * w3 x) (PhysDock's SwiGLUFFN, with its residual)"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = 128 * ((int(8 * dim / 3) + 127) // 128)
        self.w13, self.w2 = nn.Linear(dim, 2 * hidden, bias=False), nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w13(x).chunk(2, -1)
        return x + self.w2(F.silu(a) * b)


class RelPos(nn.Module):
    """AF3's relative position features as PhysDock's RelPosEmbedder: residue offset (clipped at r_max, "other chain"), the ligand pair features, same entity, chain offset"""

    def __init__(self, c_z: int, r_max: int = 32, s_max: int = 2) -> None:
        super().__init__()
        self.r_max, self.s_max = r_max, s_max
        self.linear = nn.Linear(2 * r_max + 2 + 42 + 1 + 2 * s_max + 2, c_z, bias=False)

    def forward(self, m) -> torch.Tensor:
        asym, sym, entity, res = m["asym_id"], m["sym_id"], m["entity_id"], m["residue_index"]
        same_chain, same_entity = asym[:, None] == asym[None], entity[:, None] == entity[None]
        d_res = torch.where(same_chain, (res[:, None] - res[None] + self.r_max).clamp(0, 2 * self.r_max), 2 * self.r_max + 1)
        d_chain = torch.where(same_chain | ~same_entity, 2 * self.s_max + 1, (sym[:, None] - sym[None] + self.s_max).clamp(0, 2 * self.s_max))
        dt = m["ligand_pair_feat"].dtype
        return self.linear(torch.cat([F.one_hot(d_res.long(), 2 * self.r_max + 2).to(dt), m["ligand_pair_feat"], same_entity[..., None].to(dt),
                                      F.one_hot(d_chain.long(), 2 * self.s_max + 2).to(dt)], -1))


class InputEmbedder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        ma, mp, ms, mz = cfg.mult_a, cfg.mult_ap, cfg.mult_s, cfg.mult_z
        self.linear_c = nn.Linear(len(ATOM_COLUMNS), ma["0e"], bias=False)
        self.linear_c_l, self.linear_c_m = nn.Linear(ma["0e"], mp["0e"], bias=False), nn.Linear(ma["0e"], mp["0e"], bias=False)
        self.linear_d = nn.Linear(1 + N_RBF, mp["0e"], bias=False)
        self.ffn = SwiGLU(mp["0e"])
        self.linear_target_feat, self.linear_key_res_feat = nn.Linear(65, ms["0e"], bias=False), nn.Linear(7, ms["0e"], bias=False)
        self.linear_pocket = nn.Linear(1, ms["0e"], bias=False)
        self.rel_pos = RelPos(mz["0e"])
        self.linear_bonds = nn.Linear(1, mz["0e"], bias=False)

    def forward(self, m: Dict[str, torch.Tensor]) -> Dict[str, object]:
        """m: the features, padded or not: atom_mask [Na] / token_mask [Nt] (absent: all real), n_conformers (absent: max + 1; padded: a fixed number, the padding atoms
        in a conformer of their own)"""
        ref, conf, a2t = m["ref_pos"], m["atom_to_conformer"], m["atom_to_token"]
        dt = ref.dtype
        am = m["atom_mask"].to(dt) if "atom_mask" in m else torch.ones(ref.shape[0], dtype=dt, device=ref.device)
        nt = m["restype"].shape[0]
        tm = m["token_mask"].to(dt) if "token_mask" in m else torch.ones(nt, dtype=dt, device=ref.device)
        nc = int(m["n_conformers"]) if "n_conformers" in m else int(conf.max()) + 1
        real = am > 0
        same = conf[:, None] == conf[None]
        a0e = self.linear_c(m["ref_feat"][:, ATOM_COLUMNS].to(dt))
        d = torch.cdist(ref, ref, compute_mode="donot_use_mm_for_euclid_dist")
        geo = torch.cat([1 / (1 + d[..., None]), rbf(d * REF_SCALE)], -1) * same[..., None].to(dt)            # the reference distances inside a conformer only
        h = F.relu(a0e)
        ap0e = self.ffn(self.linear_c_l(h)[:, None] + self.linear_c_m(h)[None] + self.linear_d(geo))
        s0e = self.linear_target_feat(m["target_feat"].to(dt)) + self.linear_key_res_feat(m["key_res_feat"].to(dt)) + self.linear_pocket(m["is_pocket"].to(dt)[:, None])
        z0e = self.rel_pos({**m, "ligand_pair_feat": m["ligand_pair_feat"].to(dt)}) + self.linear_bonds(m["token_bonds"].to(dt)[..., None])
        onehot = F.one_hot(a2t, nt).T.to(dt) * am[None]                                                         # [Nt, Na]: the real atoms of every token
        t2c = torch.full((nt,), nc - 1, dtype=torch.long, device=ref.device).scatter_reduce(0, a2t, torch.where(real, conf, nc - 1), "amin", include_self=True)
        bonded = Fe.bonded(ref, conf) & real[:, None] & real[None]
        return {"a0e": a0e, "ap0e": ap0e, "s0e": s0e, "z0e": z0e, "pool": onehot / onehot.sum(1, keepdim=True).clamp(min=1), "atom_to_token": a2t, "atom_to_conformer": conf,
                "token_to_conformer": t2c, "neighbour_mean": neighbour_mean(bonded, dt), "n_atoms": a2t.shape[0], "n_tokens": nt, "n_conformers": nc,
                "atom_mask": am, "token_mask": tm}


class TemplateEmbedder(nn.Module):
    """PhysDock's TemplatePairEmbedder, typed: z_t = linear_in(norm(z)) + linear(templ_feat) (0e); the template positions give the node vectors of the triangle multiplications
    (u: the direction to the previous and to the next residue of the chain, the direction from the local centre of the template, all polar) and the directions between the
    nodes; TemplateBlocks on the pairs of one chain with a template (templ_feat mask); update = linear_out(gate(norm(z_t))) * templ_mask.

    cfg.template_unit_directions: the template enters as PhysDock's binned distogram (templ_feat) plus unit directions (the hand): every vector from the positions (the node
    vectors u, the directions between the nodes of the triangle updates) is a unit vector, so no length beyond the bins reaches the model."""

    def __init__(self, cfg: Config, heads: int, channels: int, n_blocks: int) -> None:
        super().__init__()
        mz = cfg.mult_z
        self.linear_in = NormLinear(mz, mz, bias=False)
        self.linear_templ_feat = nn.Linear(40, mz["0e"], bias=False)
        self.mult_u = {"1o": channels}
        self.linear_u = TypedLinear({"1o": 3}, self.mult_u, bias=False)
        self.unit = cfg.template_unit_directions
        ls = 1e-3 if self.unit else 8.0                         # directions(): w / sqrt(l^2 + |w|^2), with l = 1e-3 Angstrom a unit vector
        self.blocks = nn.ModuleList([TemplateBlock(mz, self.mult_u, heads, channels, length_scale=ls) for _ in range(n_blocks)])
        self.norm_out, self.gate = TypedNorm(mz), Gate(mz)
        self.linear_out = TypedLinear(mz, mz, bias=False)

    def forward(self, m, z, mask_t: torch.Tensor):
        pos = m["templ_pos"].to(z["0e"].dtype)
        chain = m["asym_id"]
        tok = (m["is_protein"] > 0) & (mask_t > 0)                                                            # the real protein tokens
        same_chain = (chain[:, None] == chain[None]) & tok[:, None] & tok[None]
        pair_mask = same_chain & (m["templ_feat"][..., 39] > 0) & (mask_t[:, None] > 0) & (mask_t[None] > 0)
        res = m["residue_index"]

        def neighbour(offset):
            nb = (res[None, :] - res[:, None] == offset) & same_chain                                          # [i, j]: j is the residue i + offset of the chain of i
            return torch.where(nb.any(1, keepdim=True), nb.to(pos.dtype) @ pos - pos, torch.zeros_like(pos))
        if self.unit:                                                                                           # the distances of the bins only (their centres)
            lo, hi, nb = Fe.TEMPL_BINS
            edges = torch.linspace(lo, hi, nb, dtype=pos.dtype, device=pos.device)
            centres = torch.cat([(edges[:-1] + edges[1:]) / 2, edges[-1:] + 2.0])
            d2 = (m["templ_feat"][..., :nb].to(pos.dtype) @ centres).square()
        else:
            d2 = (pos[:, None] - pos[None]).square().sum(-1)
        w = torch.exp(-d2 / (2 * 10.0 ** 2)) * same_chain.to(pos.dtype)
        local = pos - (w @ pos) / w.sum(1, keepdim=True).clamp(min=1e-6)
        vec = torch.stack([neighbour(-1), neighbour(1), torch.where(tok[:, None], local, torch.zeros_like(local))], -2) * REF_SCALE
        if self.unit:
            vec = vec * torch.rsqrt(vec.square().sum(-1, keepdim=True) + 1e-12)                                  # unit vectors (0 where there is none)
        else:
            vec = G.squash(vec, 2.0)
        u = self.linear_u({"1o": vec})
        zt = add(self.linear_in(z), {"0e": self.linear_templ_feat(m["templ_feat"].to(pos.dtype))})
        valid = (tok & (mask_t > 0)).to(pos.dtype)
        for blk in self.blocks:
            zt = blk(u, zt, pos, valid, same_chain, pair_mask)
        out = self.linear_out(self.gate(self.norm_out(zt)))
        return {k: v * m["templ_mask"].to(v.dtype) for k, v in out.items()}


class EncoderTrunk(nn.Module):
    def __init__(self, cfg: Config, heads_z: int = 4, channels: int = 8, pair_channels: int = 4) -> None:
        super().__init__()
        self.cfg = cfg
        ma, mp, ms, mz = cfg.mult_a, cfg.mult_ap, cfg.mult_s, cfg.mult_z
        self.embedder = InputEmbedder(cfg)
        self.a_in_vec = TypedLinear({"1o": 2, "1e": 1}, {"1o": ma["1o"], "1e": ma["1e"]}, bias=False)
        self.ap_vol = nn.Linear(mp["0e"], mp["0o"], bias=False)
        self.a_in_0o = nn.Linear(mp["0o"], ma["0o"], bias=False)                                # 0o -> 0o: no bias
        self.atom_blocks = nn.ModuleList([AtomBlock(ma, mp, cfg.heads_atom, cfg.product_channels) for _ in range(cfg.n_encoder_atom_blocks)])
        self.atom_pair = PairFromSingle(ma, mp, channels=pair_channels)
        self.pool = TypedPool(ma, ms)
        sc = {k: ms[k] for k in ("0e", "0o")}
        self.linear_s_i, self.linear_s_j = TypedLinear(sc, mz), TypedLinear(sc, mz, bias=False)
        self.template = TemplateEmbedder(cfg, heads_z, channels, cfg.n_template_blocks)
        self.blocks = nn.ModuleList([TrunkBlock(ms, mz, cfg.heads_token, heads_z, channels, cfg.product_channels) for _ in range(cfg.n_encoder_blocks)])

    def forward(self, m: Dict[str, torch.Tensor]) -> Dict[str, object]:
        st = self.embedder(m)
        ref, conf, t2c, pool = m["ref_pos"], st["atom_to_conformer"], st["token_to_conformer"], st["pool"]
        dt, dev = ref.dtype, ref.device
        n, nt, nc = st["n_atoms"], st["n_tokens"], st["n_conformers"]
        same_atom, same_token = conf[:, None] == conf[None, :], t2c[:, None] == t2c[None, :]
        cm = conformer_mean(ref, conf, nc)
        nbr = st["neighbour_mean"]
        has_nbr = nbr.sum(1, keepdim=True) > 0
        vec = torch.stack([ref - cm[conf], torch.where(has_nbr, ref - nbr @ ref, torch.zeros_like(ref))], dim=-2) * REF_SCALE        # [Na, 2, 3] polar, relative
        ap0e = st["ap0e"]
        w = G.squash((ref[None, :] - ref[:, None]) * REF_SCALE, 1.0)
        vol = torch.where(same_atom, G.triple(vec[:, None, 1], "1o", vec[None, :, 1], "1o", w, "1o")[0], torch.zeros_like(w[..., 0]))     # [Na, Na] 0o
        ap = {"0e": ap0e, "0o": G.scale(self.ap_vol(ap0e), "0e", vol[..., None], "0o")[0]}
        size = same_atom.sum(1, keepdim=True).to(dt)
        a = {"0e": st["a0e"], "0o": self.a_in_0o(ap["0o"].sum(1) / size),                                     # the mean over the conformer of the pair pseudoscalars
             **self.a_in_vec({"1o": vec, "1e": torch.linalg.cross(vec[:, 0], vec[:, 1], dim=-1)[:, None]})}
        mask_a, mask_t = st["atom_mask"], st["token_mask"]
        attend = same_atom & (mask_a[None] > 0)                                                                 # the real atoms of the own conformer (padding: a conformer of its own)
        for blk in self.atom_blocks:
            a = blk(a, ap, ref - cm[conf], attend)                                                              # the atoms of a conformer attend to each other only
        ap = add(ap, self.atom_pair(a, ref, mask_a, same_atom))
        s = self.pool(a, pool)
        s = {**s, "0e": s["0e"] + st["s0e"]}
        sc = {k: s[k] for k in ("0e", "0o")}
        si, sj = self.linear_s_i(sc), self.linear_s_j(sc)
        z = {k: si[k][:, None] + sj[k][None] for k in ("0e", "0o")}
        z = {**z, "0e": z["0e"] + st["z0e"]}
        z = add(z, self.template(m, z, mask_t))
        centre_t = pool @ ref                                                                                   # the token centre, in the frame of its conformer
        for blk in self.blocks:
            s, z = blk(s, z, centre_t, mask_t, same_token)
        ref_rel = (ref - cm[conf]) * REF_SCALE
        inv, null = Fr.reference_moments(ref_rel, conf, nc)
        keep = ("pool", "atom_to_token", "atom_to_conformer", "token_to_conformer", "neighbour_mean", "n_atoms", "n_tokens", "n_conformers", "atom_mask", "token_mask")
        return {**{k: st[k] for k in keep}, "a": a, "ap": ap, "s": s, "z": z, "frames": {"ref_rel": ref_rel, "inv": inv, "null": null}}
