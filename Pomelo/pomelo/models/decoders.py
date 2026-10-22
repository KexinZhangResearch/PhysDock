"""The decoder of Pomelo (the DiT), multi-scale as in AF3 / PhysDock (atoms -> tokens -> atoms), built from the typed layers and conditioned on the 12 outputs of the encoder trunk;
equivariant under rotations, reflections and translations, so there is no augmentation of the coordinates, in training or in sampling.

One noisy sample is processed by PomeloDiT.forward; models/model.py vmaps it over the noisy samples of a structure (the conditions of the encoder are shared: what does not depend on the
noisy coordinates is computed once, vmap does not batch it).

The conditions (pomelo.models.encoders.EncoderTrunk): the vectors of a and s are in the frame of the reference of their conformer; the transport T_k of every conformer (a function of the
noisy structure, pomelo.models.primitives.frames) moves them into the frame of the noisy structure. det(T_k) is an even scalar (T_k -> Q T_k P_k^T with det Q = det P_k): about +1 when
the noisy conformer has the hand of its reference, -1 when it is the mirror image of it.

The noisy coordinates x (centred, times c_in, in units of 4 Angstrom):
    atoms        1o  x - conformer centre, x - centroid, x - mean of the bonded neighbours (in the conformer);  1e  the cross products of the last with the first two;  0e  det(T_k)
    atom pairs   0e  RBF(|x_i - x_j|)
    tokens       the pooled atoms;  token pairs  0e  RBF(|c_i - c_j|) of the token centres (the mean of their atoms): the token pair representation sees the structure too
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo.configs import Config
from pomelo.models.primitives import frames as Fr
from pomelo.models.primitives import geometry as G
from pomelo.models.primitives.irreps import Feats, GeometricProduct, Mult, NormLinear, TypedLinear, TypedNorm, TypedPool, add, mask_rows
from pomelo.models.primitives.transformers import SingleBlock
from pomelo.utils.tensor_utils import N_RBF, rbf

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
    """A SingleBlock (products, attention, transition) conditioned on the noise level by a typed AdaLN, as PhysDock's DiT blocks: before each of the three updates the stream is
    normalised (TypedNorm without affine), scaled per channel by (1 + scale) (every kind: an even number keeps the kind), shifted on 0e only (the only kind that may be shifted), and the
    update is multiplied per channel by a gate. scale, shift and gate are linear in the time embedding. The stream itself is never shifted."""

    def __init__(self, mult_s: Mult, mult_z: Mult, heads: int, t_dim: int, product_channels: int) -> None:
        super().__init__()
        self.block = SingleBlock(mult_s, mult_z, heads, product_channels)
        for layer in (self.block.product, self.block.attention, self.block.transition):
            layer.norm = None                                                                   # replaced by the AdaLN below (the layers are called with normed=True)
        self.mult_s = {k: c for k, c in mult_s.items() if c}
        self.norm = TypedNorm(self.mult_s, affine=False)
        self.width = sum(self.mult_s.values())
        self.mod = nn.Linear(t_dim, 3 * (self.mult_s["0e"] + 2 * self.width))
        with torch.no_grad():
            self.mod.weight.mul_(0.1)
            self.mod.bias.zero_()
            for i in range(3):
                self.mod.bias[self._slices(i)[2]] = 1.0                                        # the gates start at about 1, the scale and the shift at about 0

    def _slices(self, i: int):
        w = self.mult_s["0e"] + 2 * self.width
        o = i * w
        return slice(o, o + self.mult_s["0e"]), slice(o + self.mult_s["0e"], o + self.mult_s["0e"] + self.width), slice(o + self.mult_s["0e"] + self.width, o + w)

    def _per_kind(self, v: torch.Tensor) -> Dict[str, torch.Tensor]:
        return dict(zip(self.mult_s, v.split(list(self.mult_s.values()), dim=-1)))

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> Feats:
        valid = mask > 0
        m = self.mod(F.silu(t))
        s = mask_rows(s, valid)
        updates = (lambda x: self.block.product(x, normed=True), lambda x: self.block.attention(x, z, centers, mask, normed=True),
                   lambda x: self.block.transition(x, normed=True))
        for i, update in enumerate(updates):
            sh, sc, ga = (m[sl] for sl in self._slices(i))
            sc, ga = self._per_kind(sc), self._per_kind(ga)
            x = {k: G.scale(v, k, 1 + sc[k][None], "0e")[0] for k, v in self.norm(s).items()}
            x = mask_rows({**x, "0e": x["0e"] + sh}, valid)
            s = mask_rows(add(s, {k: G.scale(v, k, ga[k][None], "0e")[0] for k, v in update(x).items()}), valid)
        return s


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
        self.cond_ap_z = NormLinear(mz, mp, bias=False)                                                            # + z_enc broadcast to the atom pairs
        self.noisy_in = TypedLinear({"0e": 1, "1o": 3, "1e": 2}, {k: ma[k] for k in ("0e", "1o", "1e")}, bias=False)
        self.ap_rbf = nn.Linear(N_RBF, mp["0e"], bias=False)
        self.atom_enc = nn.ModuleList([DiTBlock(ma, mp, cfg.heads, cfg.t_dim, cfg.product_channels) for _ in range(cfg.n_atom_blocks)])
        self.pool = TypedPool(ma, ms)
        self.cond_s = TypedLinear(ms, ms)                                                                          # s_enc (normalised), its vectors moved
        self.cond_z = NormLinear(mz, mz)
        self.z_rbf = nn.Linear(N_RBF, mz["0e"], bias=False)
        self.token_blocks = nn.ModuleList([DiTBlock(ms, mz, cfg.heads, cfg.t_dim, cfg.product_channels) for _ in range(cfg.n_token_blocks)])
        self.unpool = NormLinear(ms, ma)
        self.atom_dec = nn.ModuleList([DiTBlock(ma, mp, cfg.heads, cfg.t_dim, cfg.product_channels) for _ in range(cfg.n_atom_blocks)])
        self.head_product = GeometricProduct(ma, {"1o": 8}, channels=8)
        self.head_atom = TypedLinear({"1o": 8}, {"1o": 1}, bias=False)
        self.head_token = NormLinear({"1o": ms["1o"]}, {"1o": 1}, bias=False)                                   # PhysDock's token_out: RMSNorm + Linear

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, enc: Dict[str, object]) -> torch.Tensor:
        """x [Na, 3]: the centred noisy coordinates times c_in (EDM); c_noise: ln(sigma) / 4 (a scalar); enc: the output of the EncoderTrunk -> the raw output F [Na, 3] (a polar
        vector per atom)"""
        n, nt, nc = enc["n_atoms"], enc["n_tokens"], enc["n_conformers"]
        pool, a2t, conf, fr = enc["pool"].to(x.dtype), enc["atom_to_token"], enc["atom_to_conformer"], enc["frames"]
        mask_a, mask_t = torch.ones(n, dtype=x.dtype, device=x.device), torch.ones(nt, dtype=x.dtype, device=x.device)
        t = self.time(c_noise)
        xg = x * GEOM
        x_conf = Fr.conformer_centred(xg, conf, nc)
        transport = Fr.transport(x_conf, fr["ref_rel"], conf, fr["inv"], fr["null"])                            # [n_conf, 3, 3]
        hand = torch.linalg.det(transport)                                                                         # even: does the noisy conformer have the hand of its reference?
        a_enc = {k: (Fr.move(v, transport[conf]) if k in G.VECTOR_KINDS else v) for k, v in self.norm_a_enc(enc["a"]).items()}
        s_enc = {k: (Fr.move(v, transport[enc["token_to_conformer"]]) if k in G.VECTOR_KINDS else v) for k, v in self.norm_s_enc(enc["s"]).items()}
        nbr = enc["neighbour_mean"].to(x.dtype)
        v_nbr = torch.where(nbr.sum(1, keepdim=True) > 0, xg - nbr @ xg, torch.zeros_like(xg))
        vec = torch.stack([G.squash(x_conf, 1.0), G.squash(xg - xg.mean(0), 4.0), G.squash(v_nbr, 1.0)], dim=-2)          # [Na, 3, 3]  1o, relative, bounded
        axial = torch.stack([torch.linalg.cross(vec[:, 0], vec[:, 2], dim=-1), torch.linalg.cross(vec[:, 1], vec[:, 2], dim=-1) / 4.0], dim=-2)
        a = self.cond_a({k: torch.cat([a_enc[k], s_enc[k][a2t]], dim=-1 if k in G.SCALAR_KINDS else -2) for k in G.KINDS})
        a = add(a, self.noisy_in({"0e": hand[conf][:, None], "1o": vec, "1e": axial}))
        zp = self.cond_ap_z(enc["z"])
        ap = add(self.cond_ap(enc["ap"]), {k: v[a2t][:, a2t] for k, v in zp.items()})                            # does not depend on x: computed once under vmap
        ap = {**ap, "0e": ap["0e"] + self.ap_rbf(rbf(torch.cdist(xg, xg)))}
        for blk in self.atom_enc:
            a = blk(a, ap, xg, mask_a, t)
        s = add(self.pool(a, pool), self.cond_s(s_enc))
        centre_t = pool @ xg
        z = self.cond_z(enc["z"])
        z = {**z, "0e": z["0e"] + self.z_rbf(rbf(torch.cdist(centre_t, centre_t)))}
        for blk in self.token_blocks:
            s = blk(s, z, centre_t, mask_t, t)
        a = add(a, {k: v[a2t] for k, v in self.unpool(s).items()})
        for blk in self.atom_dec:
            a = blk(a, ap, xg, mask_a, t)
        return self.head_atom(self.head_product(a))["1o"][:, 0] + self.head_token({"1o": s["1o"]})["1o"][:, 0][a2t]
