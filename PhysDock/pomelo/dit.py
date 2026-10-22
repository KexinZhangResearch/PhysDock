"""The DiT of Pomelo, multi-scale as PhysDock's AllAtomDecoder and SimpleFold (atom encoder -> pool to tokens -> token trunk -> unpool -> atom decoder), parity-equivariant,
without a pair representation and without pair biases (as SimpleFold: the capacity is in the token trunk):

  atom encoder / decoder  n_atom_blocks DiT blocks each, global attention over all atoms (as PhysDock's atom DiTs)
  token trunk             n_token_blocks DiT blocks, global attention over all tokens
  RoPE (residue index, chain) on the 0e channels of q and k, the distance of the noisy sample in the logits (w_h(t)), the direction messages: the positions of the attentions are
  the noisy atoms (atom blocks) and the mean of the atoms of every token (token blocks)

It works on a batch of noisy samples of one structure, x_in [B, N_atom, 3] (PhysDock's decoder interface: forward(batch, x_in, t_in, conds)), with the conditions of
PhysDock.pomelo.embedders.ConditionEmbedder shared by the samples. Every stream holds the four kinds 0e (scalar), 0o (pseudoscalar), 1o (polar vector), 1e (axial vector).

The vectors of the inputs (every one squashed: bounded):
  atoms   1o  x - conformer centre, x - token centre, x - mean of the bonded neighbours, x - centroid (the noisy sample); R_k (r - conformer centre), R_k (r - neighbour mean)
              (the reference conformer, rotated into the frame of the sample by the proper Kabsch rotation R_k of its conformer, PhysDock.pomelo.primitives.alignment)
          1e  the cross products (x - neighbour mean) x (x - conformer centre), (x - neighbour mean) x (x - token centre), R_k (r - neighbour mean) x R_k (r - centre)
          0o  V_x: the signed volume of the neighbour triple of every atom in the noisy sample;  0e  V_ref * V_x (does the sample have the hand of the reference?), |x - token centre|
  tokens  1o  R_t (template - its centre), R_t (next - this), R_t (previous - this) (the template, rotated by one proper Kabsch rotation onto the token centres of the sample)
          1e  their cross product (the normal of the local template frame)

The exchange between the kinds (s -> s), in every block: GeometricProduct (all products of two kinds up to l = 1 and the signed volume, within a row), the attention (q . k over
all kinds, values of every kind, direction messages: across the rows), TypedTransition (the vector channels gated by the scalars); each behind a typed AdaLN-Zero of the time.
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.pomelo.configs import Config
from PhysDock.pomelo.embedders import masked_group_mean, pad_rows, signed_volume
from PhysDock.pomelo.primitives import geometry as G
from PhysDock.pomelo.primitives.alignment import kabsch, rotate
from PhysDock.pomelo.primitives.attentions import GlobalAttention
from PhysDock.pomelo.primitives.irreps import Feats, GeometricProduct, Mult, NormLinear, TypedLinear, TypedNorm, add
from PhysDock.pomelo.primitives.transformers import TypedTransition

W0, W_MAX_LOG = 0.05, math.log(40.0)   # the distance weight at the start, its largest value W0 * 40 = 2 (a logit of -2 at 1 unit, 4 Angstrom)
GEOM = 4.0                      # the normalised coordinates (sigma_data = 16 Angstrom) times 4: positions in units of 4 Angstrom


class TimestepEmbedding(nn.Module):
    """PhysDock's TimestepEmbeddings: 256 sinusoidal features of the noise conditioning (frequencies up to 1), a two-layer SiLU MLP to `dim`"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear_1, self.linear_2 = nn.Linear(256, dim), nn.Linear(dim, dim)
        self.freqs = nn.Buffer(torch.exp(-math.log(10000) * torch.arange(128) / 128), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        a = t[:, None] * self.freqs.to(t.dtype)
        return self.linear_2(F.silu(self.linear_1(torch.cat([torch.cos(a), torch.sin(a)], -1))))


class DiTBlock(nn.Module):
    """s += products, s += attention, s += transition, each behind a typed AdaLN (PhysDock's AdaLayerNormZero): the stream normalised (TypedNorm without affine), scaled
    per channel by (1 + scale) (every kind: an even number keeps the kind), shifted on 0e only, the update multiplied per channel by a gate; scale, shift, gate and the weights
    w_h of the distance term are functions of the time embedding of the sample."""

    def __init__(self, mult: Mult, heads: int, cfg: Config, product_channels: int) -> None:
        super().__init__()
        self.mult = {k: c for k, c in mult.items() if c}
        self.product = GeometricProduct(self.mult, self.mult, product_channels)
        self.product.norm = None                                                                 # the AdaLN normalises (the sublayers are called normed)
        self.attention = GlobalAttention(self.mult, heads, cfg.rope_res, cfg.rope_chain, cfg.direction_length)
        self.transition = TypedTransition(self.mult)
        self.transition.norm = None
        self.norm = TypedNorm(self.mult, affine=False)
        self.width = sum(self.mult.values())
        self.mod = nn.Linear(cfg.t_dim, 3 * (self.mult["0e"] + 2 * self.width))
        self.dist = nn.Linear(cfg.t_dim, heads)                                                    # w_h(t) = W0 exp(dist(t)) >= 0: local at small noise, global at large (learned)
        with torch.no_grad():
            self.mod.weight.mul_(0.1)
            self.mod.bias.zero_()
            for i in range(3):
                self.mod.bias[self._slices(i)[2]] = 0.0 if cfg.zero_gates else 1.0
            self.dist.weight.zero_()
            self.dist.bias.zero_()                                                               # w = W0 at the start: a logit of -1 at 4.5 units (18 Angstrom)

    def _slices(self, i: int):
        w = self.mult["0e"] + 2 * self.width
        o = i * w
        return slice(o, o + self.mult["0e"]), slice(o + self.mult["0e"], o + self.mult["0e"] + self.width), slice(o + self.mult["0e"] + self.width, o + w)

    def _per_kind(self, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(zip(self.mult, v.split(list(self.mult.values()), dim=-1)))

    def forward(self, s: Feats, pos: torch.Tensor, t: torch.Tensor, geo: Dict[str, torch.Tensor]) -> Feats:
        """s {kind: [B, N, C(, 3)]}, pos [B, N, 3] (centred), t [B, t_dim], geo: the static geometry of the attention"""
        tt = F.silu(t)
        m = self.mod(tt)[:, None]                                                                # [B, 1, ...]: per sample, the same for all rows
        w = W0 * torch.exp(self.dist(tt).clamp(max=W_MAX_LOG))                                  # log-parametrised: a relative step per update (softplus at 0.05: 20x slower)
        updates = (lambda x: self.product(x, normed=True), lambda x: self.attention(x, pos, w, geo), lambda x: self.transition(x, normed=True))
        for i, update in enumerate(updates):
            sh, sc, ga = (m[..., sl] for sl in self._slices(i))
            sc, ga = self._per_kind(sc), self._per_kind(ga)
            x = {k: G.scale(v, k, 1 + sc[k], "0e")[0] for k, v in self.norm(s).items()}
            x = {**x, "0e": x["0e"] + sh}
            s = add(s, {k: G.scale(v, k, ga[k], "0e")[0] for k, v in update(x).items()})
        return s


def pool_tokens(pool: torch.Tensor, h: Feats) -> Feats:
    """the mean over the real atoms of every token: pool [Nt, Na], h {kind: [B, Na, ...]} -> [B, Nt, ...]"""
    return {k: torch.einsum("ta,ba...->bt...", pool, v) for k, v in h.items()}


def broadcast(h: Feats, b: int) -> Feats:
    return {k: v.expand(b, *v.shape) for k, v in h.items()}


class PomeloDiT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ma, ms = cfg.mult_a, cfg.mult_s
        sa, ss = {k: ma[k] for k in G.SCALAR_KINDS}, {k: ms[k] for k in G.SCALAR_KINDS}
        self.time = TimestepEmbedding(cfg.t_dim)
        self.cond_a = NormLinear(sa, sa)                                                       # the atom conditions (0e, 0o)
        self.cond_a_token = NormLinear(ss, sa, bias=False)                                     # + the token conditions broadcast to the atoms (PhysDock: s -> a)
        self.noisy_in = TypedLinear({"0e": 2, "0o": 1, "1o": 6, "1e": 3}, ma, bias=False)                   # the noisy sample and the reference conformer
        self.templ_in = TypedLinear({"1o": 3, "1e": 1}, {"1o": ms["1o"], "1e": ms["1e"]}, bias=False)         # the template
        atom = lambda: DiTBlock(ma, cfg.heads_atom, cfg, cfg.atom_product_channels)
        self.atom_enc = nn.ModuleList([atom() for _ in range(cfg.n_atom_blocks)])
        self.pool_norm, self.pool_lin = TypedNorm(ma), TypedLinear(ma, ms)                     # PhysDock: silu(linear_downscale(x)), mean over the token
        self.cond_s = NormLinear(ss, ss)
        self.token_blocks = nn.ModuleList([DiTBlock(ms, cfg.heads_token, cfg, cfg.token_product_channels) for _ in range(cfg.n_token_blocks)])
        self.unpool = NormLinear(ms, ma)
        self.atom_dec = nn.ModuleList([atom() for _ in range(cfg.n_atom_blocks)])
        self.head_product = GeometricProduct(ma, {"1o": 8}, channels=8)
        self.head_atom = TypedLinear({"1o": 8}, {"1o": 1}, bias=False)
        self.head_token = NormLinear({"1o": ms["1o"]}, {"1o": 1}, bias=False)                  # PhysDock's token_out: the token stream predicts a displacement too

    def condition(self, batch, conds):
        """PhysDock's decoder interface: the per-structure inputs of the decoder; ConditionEmbedder already gives them"""
        return conds

    def noisy_features(self, pos: torch.Tensor, c: Dict[str, object]) -> Feats:
        """the atom inputs from the noisy sample and the reference conformer: pos [B, Na, 3] (centred on the real atoms, padding 0) -> {0e [B, Na, 2], 0o [B, Na, 1],
        1o [B, Na, 6, 3], 1e [B, Na, 3, 3]}. The reference vectors (r - conformer centre, r - mean of the bonded neighbours) rotated into the frame of the sample by the
        proper Kabsch rotation of their conformer"""
        am = c["atom_mask"][None, :, None]
        conf, nt = c["atom_to_conformer"], c["token_mask"].shape[0]                                 # at most as many conformers as tokens
        conf_c = masked_group_mean(pos, conf, c["atom_mask"], nt)
        tok_c = torch.einsum("ta,bac->btc", c["pool"], pos)[:, c["atom_to_token"]]
        nbr = c["nbr_mean"]
        v_nbr = torch.where(nbr.sum(1)[None, :, None] > 0, pos - nbr @ pos, torch.zeros_like(pos))
        u_conf, u_tok, u_nbr = G.squash(pos - conf_c, 1.0), G.squash(pos - tok_c, 1.0), G.squash(v_nbr, 1.0)
        r, r_c = kabsch(pos, c["ref"], conf, c["atom_mask"], nt)
        ra = r[:, conf]                                                                             # [B, Na, 3, 3] the rotation of the conformer of every atom
        ref_c, ref_n = G.squash(rotate(ra, r_c), 2.0), G.squash(rotate(ra, c["ref_nbr"]), 1.0)
        v_x = signed_volume(pos, c["idx"], c["has3"])                                               # [B, Na] 0o
        f = {"0e": torch.stack([c["v_ref"][None] * v_x, u_tok.norm(dim=-1)], -1),
             "0o": v_x[..., None],
             "1o": torch.stack([u_conf, u_tok, u_nbr, G.squash(pos, self.cfg.centroid_length), ref_c, ref_n], -2),
             "1e": torch.stack([torch.linalg.cross(u_nbr, u_conf, dim=-1), torch.linalg.cross(u_nbr, u_tok, dim=-1), torch.linalg.cross(ref_n, ref_c, dim=-1)], -2)}
        return {k: v * (am if k in G.SCALAR_KINDS else am[..., None]) for k, v in f.items()}

    def template_features(self, centre_t: torch.Tensor, c: Dict[str, object]) -> Feats:
        """the token inputs from the template: centre_t [B, Nt, 3] (the token centres of the sample) -> {1o [B, Nt, 3, 3], 1e [B, Nt, 1, 3]}: the template positions rotated
        into the frame of the sample by one proper Kabsch rotation (all template tokens: one frame), centred; the directions to the next and previous residue; their cross
        product (the normal of the local template frame). 0 where there is no template"""
        on, tp = c["templ_on"], c["templ_pos"]
        nt = on.shape[0]
        r, t_c = kabsch(centre_t, tp, torch.zeros(nt, dtype=torch.long, device=tp.device), on, 1)
        rt = r[:, :1].expand(-1, nt, 3, 3)
        step = lambda m: torch.where(m.sum(1, keepdim=True) > 0, (m @ tp) / m.sum(1, keepdim=True).clamp(min=1) - tp, torch.zeros_like(tp))    # the mean if an index repeats
        nxt, prv = step(c["templ_next"]), step(c["templ_prev"])
        vc, vn, vp = G.squash(rotate(rt, t_c), self.cfg.centroid_length), G.squash(rotate(rt, nxt), 1.0), G.squash(rotate(rt, prv), 1.0)
        m = on.to(tp.dtype)[None, :, None, None]
        return {"1o": torch.stack([vc, vn, vp], -2) * m, "1e": torch.linalg.cross(vn, vp, dim=-1)[:, :, None] * m}

    def forward(self, batch, x_in: torch.Tensor, t_in: torch.Tensor, c: Dict[str, object]) -> torch.Tensor:
        """x_in [B, Na, 3]: the preconditioned noisy coordinates (normalised, times c_in), t_in [B]: the noise conditioning -> the raw output F [B, Na, 3] (a polar vector per
        atom; 0 at the padding)"""
        b, na = x_in.shape[:2]
        am = c["atom_mask"].to(x_in.dtype)
        t = self.time(t_in)
        with torch.autocast(x_in.device.type, enabled=False):                                  # the geometry of the inputs in float32 (Kabsch: an SVD), under any autocast
            xg = pad_rows(x_in.to(torch.promote_types(x_in.dtype, torch.float32)).transpose(0, 1), am.shape[0]).transpose(0, 1) * GEOM * am[None, :, None]
            pos = (xg - xg.sum(1, keepdim=True) / am.sum()) * am[None, :, None]               # centred on the real atoms: translation-free
            noisy = self.noisy_features(pos, c)
            centre_t = torch.einsum("ta,bac->btc", c["pool"], pos)
            templ = self.template_features(centre_t, c)
        a = add(self.noisy_in(noisy), broadcast(add(self.cond_a(c["a"]), {k: v[c["atom_to_token"]] for k, v in self.cond_a_token(c["s"]).items()}), b))
        for blk in self.atom_enc:
            a = blk(a, pos, t, c["atom_geo"])
        h = self.pool_lin(self.pool_norm(a))
        s = pool_tokens(c["pool"], {**h, "0e": F.silu(h["0e"])})
        s = add(add(s, broadcast(self.cond_s(c["s"]), b)), self.templ_in(templ))
        for blk in self.token_blocks:
            s = blk(s, centre_t, t, c["token_geo"])
        a = add(a, {k: v[:, c["atom_to_token"]] for k, v in self.unpool(s).items()})
        for blk in self.atom_dec:
            a = blk(a, pos, t, c["atom_geo"])
        out = self.head_atom(self.head_product(a))["1o"][..., 0, :] + self.head_token({"1o": s["1o"]})["1o"][..., 0, :][:, c["atom_to_token"]]
        return (out * am[None, :, None])[:, :na]
