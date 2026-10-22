"""The decoder of Pomelo (the DiT), multi-scale as PhysDock's AllAtomDecoder (atom DiT -> tokens -> token DiT -> atoms -> atom DiT), built from the typed layers and conditioned on
the 12 outputs of the encoder trunk; equivariant under rotations, reflections and translations, so there is no augmentation of the coordinates, in training or in sampling.

It works on a batch of noisy samples of one structure, x [B, N_atom, 3], as PhysDock's: what does not depend on the noisy coordinates is computed once per structure and shared by
the samples, above all the logit biases of every block (PairBias, from the atom pairs ap and the token pairs z), and every attention is one call of the memory-efficient kernel with
that shared bias (SharedBiasAttention): the memory of a sample is O(N), no [B, H, N, N] tensor exists.

The conditions (pomelo.models.encoders.EncoderTrunk): the vectors of a and s are in the frame of the reference of their conformer; the transport T_k of every conformer (a function of the
noisy structure, pomelo.models.primitives.frames) moves them into the frame of the noisy sample. det(T_k) is an even scalar (T_k -> Q T_k P_k^T with det Q = det P_k): about +1 when
the noisy conformer has the hand of its reference, -1 when it is the mirror image of it.

The noisy coordinates x (centred, times c_in, in units of 4 Angstrom; PhysDock: linear_x(x_in), a vector, here the vectors of the kinds):
    atoms   1o  x - conformer centre, x - centroid, x - mean of the bonded neighbours (in the conformer);  1e  the cross products of the last with the first two;  0e  det(T_k)
    the positions also enter every attention (the direction messages); the pair biases are those of the structure (as PhysDock: no distance of the noisy sample in the bias)
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo.configs import Config
from pomelo.models.primitives import frames as Fr
from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.attentions import MASKED, PairBias, SharedBiasAttention
from pomelo.models.primitives.irreps import Feats, GeometricProduct, Mult, NormLinear, TypedLinear, TypedNorm, add
from pomelo.models.primitives.transformers import TypedTransition

GEOM = 4.0                      # the scaled coordinates are in units of 4 Angstrom (sigma_data = 16 Angstrom) so that vectors and distances are of order 1


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.freqs = nn.Buffer(torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2)), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        a = c_noise[..., None] * self.freqs.to(c_noise.dtype) * 100.0
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1))


class DiTBlock(nn.Module):
    """One block of a DiT stack for a batch of noisy samples: s += geometric products (the exchange between the kinds), s += attention with the shared bias, s += transition, each
    behind a typed AdaLN (PhysDock's AdaLayerNormZero): the stream normalised (TypedNorm without affine), scaled per channel by (1 + scale) (every kind: an even number keeps the kind),
    shifted on 0e only (the only kind that may be shifted), and the update multiplied per channel by a gate; scale, shift and gate are linear in the time embedding of the sample."""

    def __init__(self, mult_s: Mult, heads: int, t_dim: int, product_channels: int, direction_length: float = 2.0, distance: bool = False, zero_gates: bool = False) -> None:
        super().__init__()
        self.mult_s = {k: c for k, c in mult_s.items() if c}
        self.product = GeometricProduct(self.mult_s, self.mult_s, product_channels)
        self.product.norm = None                                                                 # the AdaLN below normalises (the layers are called normed)
        self.attention = SharedBiasAttention(self.mult_s, heads, direction_length=direction_length, distance=distance)
        if distance:                # w_h(t) >= 0 of the distance term of the logits, per head and noise level: local at small noise, global at large noise (learned)
            self.dist = nn.Linear(t_dim, heads)
            with torch.no_grad():
                self.dist.weight.zero_()
                self.dist.bias.fill_(math.log(math.expm1(0.05)))                                 # w = 0.05 at the start: a logit of -1 at 4.5 units (18 Angstrom)
        self.transition = TypedTransition(self.mult_s)
        self.transition.norm = None
        self.norm = TypedNorm(self.mult_s, affine=False)
        self.width = sum(self.mult_s.values())
        self.mod = nn.Linear(t_dim, 3 * (self.mult_s["0e"] + 2 * self.width))
        with torch.no_grad():
            self.mod.weight.mul_(0.1)
            self.mod.bias.zero_()
            for i in range(3):                                     # the scale and the shift start at about 0; the gates at about 1, or (zero_gates, AdaLN-Zero as PhysDock's)
                self.mod.bias[self._slices(i)[2]] = 0.0 if zero_gates else 1.0      # at about 0: every block starts close to the identity

    def _slices(self, i: int):
        w = self.mult_s["0e"] + 2 * self.width
        o = i * w
        return slice(o, o + self.mult_s["0e"]), slice(o + self.mult_s["0e"], o + self.mult_s["0e"] + self.width), slice(o + self.mult_s["0e"] + self.width, o + w)

    def _per_kind(self, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(zip(self.mult_s, v.split(list(self.mult_s.values()), dim=-1)))

    def forward(self, s: Feats, bias: torch.Tensor, centers: torch.Tensor, t: torch.Tensor) -> Feats:
        """s {kind: [B, N, C(, 3)]}, bias [H, N, N] (shared), centers [B, N, 3], t [B, t_dim]"""
        m = self.mod(F.silu(t))[:, None]                                                         # [B, 1, ...]: per sample, the same for all atoms
        w = F.softplus(self.dist(F.silu(t))) if hasattr(self, "dist") else None
        updates = (lambda x: self.product(x, normed=True), lambda x: self.attention(x, bias, centers, w), lambda x: self.transition(x, normed=True))
        for i, update in enumerate(updates):
            sh, sc, ga = (m[..., sl] for sl in self._slices(i))
            sc, ga = self._per_kind(sc), self._per_kind(ga)
            x = {k: G.scale(v, k, 1 + sc[k], "0e")[0] for k, v in self.norm(s).items()}
            x = {**x, "0e": x["0e"] + sh}
            s = add(s, {k: G.scale(v, k, ga[k], "0e")[0] for k, v in update(x).items()})
        return s


def move(v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """v [N, C, 3] (shared) times the matrix t [B, N, 3, 3] of every sample and row -> [B, N, C, 3]"""
    return torch.einsum("bnij,ncj->bnci", t, v)


def pool_tokens(pool: torch.Tensor, h: Feats) -> Feats:
    """the mean over the atoms of every token: pool [Nt, Na], h {kind: [B, Na, ...]} -> [B, Nt, ...]"""
    return {k: torch.einsum("ta,ba...->bt...", pool, v) for k, v in h.items()}


class PomeloDiT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ma, ms, mp, mz = cfg.mult_a, cfg.mult_s, cfg.mult_ap, cfg.mult_z
        self.time = TimeEmbedding(cfg.t_dim)
        # every condition is read through a norm and a linear map (PhysDock's cond_s / cond_z: RMSNorm + Linear): the encoder streams are not normalised and grow in training
        self.norm_a_enc, self.norm_s_enc = TypedNorm(ma), TypedNorm(ms)
        self.cond_a = TypedLinear({k: ma[k] + ms[k] for k in G.KINDS}, ma)                                       # [a_enc, s_enc broadcast to the atoms], every kind
        self.cond_ap = NormLinear(mp, mp)                                                                          # ap_enc
        self.cond_ap_z = NormLinear(mz, mp, bias=False)                                                            # + z_enc broadcast to the atom pairs (PhysDock's linear_z)
        self.noisy_in = TypedLinear({"0e": 1, "1o": 3, "1e": 2}, {k: ma[k] for k in ("0e", "1o", "1e")}, bias=False)
        self.bias_atom_enc = PairBias(mp, cfg.heads_atom, cfg.n_atom_blocks)
        self.bias_token = PairBias(mz, cfg.heads_token, cfg.n_token_blocks)
        self.bias_atom_dec = PairBias(mp, cfg.heads_atom, cfg.n_atom_blocks)
        self.atom_enc = nn.ModuleList([DiTBlock(ma, cfg.heads_atom, cfg.t_dim, cfg.product_channels, cfg.direction_length, cfg.distance_logits, cfg.zero_gates) for _ in range(cfg.n_atom_blocks)])
        self.pool_norm, self.pool_lin = TypedNorm(ma), TypedLinear(ma, ms)                                       # PhysDock: silu(linear_downscale(x)), mean over the token
        self.cond_s = TypedLinear(ms, ms)                                                                          # s_enc (normalised), its vectors moved
        self.cond_z = NormLinear(mz, mz)
        self.token_blocks = nn.ModuleList([DiTBlock(ms, cfg.heads_token, cfg.t_dim, cfg.product_channels, cfg.direction_length, cfg.distance_logits, cfg.zero_gates) for _ in range(cfg.n_token_blocks)])
        self.unpool = NormLinear(ms, ma)
        self.atom_dec = nn.ModuleList([DiTBlock(ma, cfg.heads_atom, cfg.t_dim, cfg.product_channels, cfg.direction_length, cfg.distance_logits, cfg.zero_gates) for _ in range(cfg.n_atom_blocks)])
        self.head_product = GeometricProduct(ma, {"1o": 8}, channels=8)
        self.head_atom = TypedLinear({"1o": 8}, {"1o": 1}, bias=False)
        self.head_token = NormLinear({"1o": ms["1o"]}, {"1o": 1}, bias=False)                                   # PhysDock's token_out: RMSNorm + Linear
        if cfg.vector_cond_gate:    # the encoder vectors moved into the noisy sample: their transport is fitted to the noisy coordinates, random at large noise; a gate per channel
            self.mult_vec = [ma["1o"], ma["1e"], ms["1o"], ms["1e"]]                         # of the noise level (sigmoid, 0.5 at the start) lets the model drop them there
            self.vec_gate = nn.Linear(cfg.t_dim, sum(self.mult_vec))
            with torch.no_grad():
                self.vec_gate.weight.zero_()
                self.vec_gate.bias.zero_()

    def biases(self, enc: Dict[str, object]):
        """the logit biases of the three stacks, once per structure; the padding keys masked (MASKED: no real atom / token attends to them)"""
        a2t = enc["atom_to_token"]
        zp = self.cond_ap_z(enc["z"])
        ap = add(self.cond_ap(enc["ap"]), {k: v[a2t][:, a2t] for k, v in zp.items()})
        key = lambda mask: torch.where(mask > 0, 0.0, MASKED).to(ap["0e"].dtype)[None, None, None, :]           # [1, 1, 1, N_key]
        am, tm = enc["atom_mask"], enc["token_mask"]
        return self.bias_atom_enc(ap) + key(am), self.bias_token(self.cond_z(enc["z"])) + key(tm), self.bias_atom_dec(ap) + key(am)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, enc: Dict[str, object]) -> torch.Tensor:
        """x [B, Na, 3]: the centred noisy coordinates times c_in (EDM); c_noise [B]: ln(sigma) / 4; enc: the output of the EncoderTrunk -> the raw output F [B, Na, 3] (a polar
        vector per atom)"""
        nc = enc["n_conformers"]
        pool, a2t, conf, t2c, fr = enc["pool"].to(x.dtype), enc["atom_to_token"], enc["atom_to_conformer"], enc["token_to_conformer"], enc["frames"]
        b_enc, b_tok, b_dec = enc["biases"] if "biases" in enc else self.biases(enc)
        t = self.time(c_noise)
        am = enc["atom_mask"].to(x.dtype)[None, :, None]                                                         # [1, Na, 1]
        xg = x * GEOM * am
        centroid = xg.sum(-2, keepdim=True) / am.sum(-2, keepdim=True).clamp(min=1)                               # the mean of the real atoms
        x_conf = torch.vmap(lambda xb: Fr.conformer_centred(xb, conf, nc))(xg)                                    # [B, Na, 3]
        transport = torch.vmap(lambda xb: Fr.transport(xb, fr["ref_rel"], conf, fr["inv"], fr["null"]))(x_conf)     # [B, n_conf, 3, 3]
        hand = torch.linalg.det(transport)                                                                         # even: does the noisy conformer have the hand of its reference?
        a_enc, s_enc = self.norm_a_enc(enc["a"]), self.norm_s_enc(enc["s"])
        if hasattr(self, "vec_gate"):
            ga1o, ga1e, gs1o, gs1e = (g[:, None, :, None] for g in torch.sigmoid(self.vec_gate(F.silu(t))).split(self.mult_vec, -1))       # [B, 1, C, 1]
        else:
            ga1o = ga1e = gs1o = gs1e = 1.0
        gate_a, gate_s = {"1o": ga1o, "1e": ga1e}, {"1o": gs1o, "1e": gs1e}
        cat = {}
        for k in G.KINDS:
            if k in G.SCALAR_KINDS:
                cat[k] = torch.cat([a_enc[k], s_enc[k][a2t]], -1)
            else:
                cat[k] = torch.cat([move(a_enc[k], transport[:, conf]) * gate_a[k], (move(s_enc[k], transport[:, t2c]) * gate_s[k])[:, a2t]], -2)
        a = self.cond_a({k: (v.expand(x.shape[0], *v.shape) if k in G.SCALAR_KINDS else v) for k, v in cat.items()})
        nbr = enc["neighbour_mean"].to(x.dtype)
        v_nbr = torch.where(nbr.sum(1, keepdim=True) > 0, xg - nbr @ xg, torch.zeros_like(xg))
        vec = torch.stack([G.squash(x_conf, 1.0), G.squash(xg - centroid, self.cfg.centroid_length), G.squash(v_nbr, 1.0)], dim=-2)          # [B, Na, 3, 3]  1o, relative
        axial = torch.stack([torch.linalg.cross(vec[..., 0, :], vec[..., 2, :], dim=-1), torch.linalg.cross(vec[..., 1, :], vec[..., 2, :], dim=-1) / 4.0], dim=-2)
        a = add(a, self.noisy_in({"0e": hand[:, conf][..., None], "1o": vec, "1e": axial}))
        pos = (xg - centroid) * am                                                                             # centred on the real atoms (the distance term)
        for blk, bias in zip(self.atom_enc, b_enc):
            a = blk(a, bias, pos, t)
        h = self.pool_lin(self.pool_norm(a))
        s = pool_tokens(pool, {**h, "0e": F.silu(h["0e"])})
        s_cond = {k: (v.expand(x.shape[0], *v.shape) if k in G.SCALAR_KINDS else move(v, transport[:, t2c]) * gate_s[k]) for k, v in s_enc.items()}
        s = add(s, self.cond_s(s_cond))
        centre_t = torch.einsum("ta,bac->btc", pool, pos)
        for blk, bias in zip(self.token_blocks, b_tok):
            s = blk(s, bias, centre_t, t)
        a = add(a, {k: v[:, a2t] for k, v in self.unpool(s).items()})
        for blk, bias in zip(self.atom_dec, b_dec):
            a = blk(a, bias, pos, t)
        out = self.head_atom(self.head_product(a))["1o"][..., 0, :] + self.head_token({"1o": s["1o"]})["1o"][..., 0, :][:, a2t]
        return out * am                                                                                            # the padding atoms: 0
