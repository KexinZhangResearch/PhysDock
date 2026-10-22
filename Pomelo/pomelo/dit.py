"""The decoder (DiT) of Pomelo, multi-scale as in AF3 / PhysDock (atoms -> units -> atoms), built from the typed layers; equivariant under rotations, reflections and translations, so
there is no augmentation of the coordinates, in training or in sampling. The stereo labels are pseudoscalars of the input: mirroring the structure and flipping the labels mirrors the output.

One sample is processed by PomeloDiT.forward; Denoiser vmaps it over the noisy samples of a structure (the static features are shared) and applies the EDM preconditioning on the
coordinates normalised by sigma_data and centred on their centroid (the noise is centred too: the translation is not part of the problem).
"""
import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo import geometry as G
from pomelo.blocks import SingleBlock, add, mask_rows
from pomelo.data import signed_volume
from pomelo.embedder import Embedder
from pomelo.irreps import Feats, GeometricProduct, Mult, TypedLinear

GEOM = 4.0                      # the scaled coordinates are in units of 4 Angstrom (sigma_data = 16 Angstrom) so that vectors and distances are of order 1
N_RBF = 8


@dataclass
class Config:
    mult_a: Mult = None          # atom stream
    mult_s: Mult = None          # unit stream (the stored features of the trunk)
    mult_ap: Mult = None         # atom pair features read by the atom attention (0e, 0o)
    mult_z: Mult = None          # unit pair features (0e, 0o)
    heads: int = 4
    n_atom_blocks: int = 1       # atom blocks before and after the unit blocks
    n_unit_blocks: int = 2
    t_dim: int = 64
    atom_static: int = 32
    pair_static: int = 16
    product_channels: int = 8

    def __post_init__(self):
        self.mult_a = self.mult_a or {"0e": 64, "0o": 16, "1o": 16, "1e": 8}
        self.mult_s = self.mult_s or {"0e": 128, "0o": 32, "1o": 32, "1e": 16}
        self.mult_ap = self.mult_ap or {"0e": 16, "0o": 8}
        self.mult_z = self.mult_z or {"0e": 32, "0o": 8}


def rbf(d: torch.Tensor, n: int = N_RBF, hi: float = 8.0) -> torch.Tensor:
    centers = torch.linspace(0, hi, n, device=d.device, dtype=d.dtype)
    return torch.exp(-((d[..., None] - centers) / (hi / n)) ** 2)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.freqs = nn.Buffer(torch.exp(-math.log(10000) * torch.arange(dim // 2) / (dim // 2)), persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        a = c_noise[..., None] * self.freqs.to(c_noise.dtype) * 100.0
        return self.mlp(torch.cat([torch.cos(a), torch.sin(a)], -1))


class TypedPool(nn.Module):
    """atoms -> units: a typed linear map (SiLU on the 0e part), then the mean over the atoms of a unit (the pool matrix [Nu, Na])"""

    def __init__(self, mult_a: Mult, mult_s: Mult) -> None:
        super().__init__()
        self.lin = TypedLinear(mult_a, mult_s)

    def forward(self, a: Feats, pool: torch.Tensor) -> Feats:
        h = self.lin(a)
        h["0e"] = F.silu(h["0e"])
        return {k: torch.einsum("ua,a...->u...", pool, v) for k, v in h.items()}


class DiTBlock(nn.Module):
    """A SingleBlock (products, attention, transition) conditioned on the noise level: a shift of the 0e channels (the only kind that may be shifted) and a gate per channel of every
    kind on each of the three updates (an AdaLN-Zero without the scale of the normalisation)"""

    def __init__(self, mult_s: Mult, mult_z: Mult, heads: int, t_dim: int, product_channels: int) -> None:
        super().__init__()
        self.block = SingleBlock(mult_s, mult_z, heads, product_channels)
        self.mult_s = dict(mult_s)
        self.shift = nn.Linear(t_dim, mult_s["0e"], bias=False)
        self.gates = nn.Linear(t_dim, 3 * sum(mult_s.values()))
        with torch.no_grad():
            self.gates.bias.fill_(1.0)

    def forward(self, s: Feats, z: Feats, centers: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> Feats:
        valid = mask > 0
        g = self.gates(F.silu(t)).split(list(self.mult_s.values()) * 3)
        gate = lambda upd, i: {k: G.scale(v, k, g[i * len(self.mult_s) + j][None, :], "0e")[0] for j, (k, v) in enumerate(upd.items())}
        s = mask_rows(s, valid)
        s = {**s, "0e": s["0e"] + self.shift(t)}
        s = add(s, gate(self.block.product(s), 0))
        s = add(s, gate(self.block.attention(s, z, centers, mask), 1))
        s = add(s, gate(self.block.transition(s), 2))
        return mask_rows(s, valid)


class PomeloDiT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ma, ms, map_, mz = cfg.mult_a, cfg.mult_s, cfg.mult_ap, cfg.mult_z
        self.time = TimeEmbedding(cfg.t_dim)
        self.a_in_0e = nn.Linear(cfg.atom_static + 1, ma["0e"])
        self.a_in_0o = TypedLinear({"0o": 2}, {"0o": ma["0o"]}, bias=False)
        self.a_in_1o = TypedLinear({"1o": 2}, {"1o": ma["1o"]}, bias=False)
        self.ap_in = nn.Linear(cfg.pair_static + N_RBF, map_["0e"])
        self.ap_tag_a, self.ap_tag_b = nn.Linear(map_["0e"], map_["0o"], bias=False), nn.Linear(map_["0e"], map_["0o"], bias=False)
        self.cond_s = TypedLinear({k: ms[k] for k in ("0e", "0o")}, {k: ms[k] for k in ("0e", "0o")})            # the static features have no vectors
        self.a_in_1e = TypedLinear({"1e": 1}, {"1e": ma["1e"]}, bias=False)
        self.atom_enc = nn.ModuleList([DiTBlock(ma, map_, cfg.heads, cfg.t_dim, cfg.product_channels) for _ in range(cfg.n_atom_blocks)])
        self.pool = TypedPool(ma, ms)
        self.unit_blocks = nn.ModuleList([DiTBlock(ms, mz, cfg.heads, cfg.t_dim, cfg.product_channels) for _ in range(cfg.n_unit_blocks)])
        self.unpool = TypedLinear(ms, ma)
        self.atom_dec = nn.ModuleList([DiTBlock(ma, map_, cfg.heads, cfg.t_dim, cfg.product_channels) for _ in range(cfg.n_atom_blocks)])
        self.head_product = GeometricProduct(ma, {"1o": 8}, channels=8)
        self.head_atom = TypedLinear({"1o": 8}, {"1o": 1}, bias=False)
        self.head_unit = TypedLinear({"1o": ms["1o"]}, {"1o": 1}, bias=False)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, st: Dict[str, object]) -> torch.Tensor:
        """x [Na, 3]: the centred noisy coordinates times c_in (EDM); c_noise: ln(sigma) / 4 (a scalar) -> the raw output F [Na, 3] (a polar vector per atom)"""
        n, nu = st["n_atoms"], st["n_units"]
        mask_a, mask_u = torch.ones(n, dtype=x.dtype, device=x.device), torch.ones(nu, dtype=x.dtype, device=x.device)
        t = self.time(c_noise)
        pool, a2u = st["pool"], st["atom_to_unit"]
        # geometry of the noisy structure: relative to the unit, relative to the centroid; the signed volumes of the stereocentres and their agreement with the labels
        centre_u = pool @ x
        vec = torch.stack([x - centre_u[a2u], x - x.mean(0)], dim=-2) * GEOM                            # [Na, 2, 3]  1o
        vol = st["scatter"] @ signed_volume(x, st["stereo_center"], st["stereo_nbrs"]) if st["stereo_center"].numel() else torch.zeros(n, dtype=x.dtype, device=x.device)
        tag = st["tag"]
        consistency = G.scale(tag, "0o", vol, "0o")[0]                                                    # tag * V: even (does the structure agree with the label?)
        a = {"0e": self.a_in_0e(torch.cat([st["a0e"], consistency[:, None]], -1)), **self.a_in_0o({"0o": torch.stack([tag, vol], -1)}), **self.a_in_1o({"1o": vec}),
             **self.a_in_1e({"1e": torch.linalg.cross(vec[:, 0], vec[:, 1], dim=-1)[:, None]})}                  # (x - unit centre) x (x - centroid): axial
        d = torch.cdist(x, x) * GEOM
        ap0e = self.ap_in(torch.cat([st["ap0e"], rbf(d)], -1))
        ap = {"0e": ap0e, "0o": G.scale(self.ap_tag_a(ap0e), "0e", tag[:, None, None], "0o")[0] + G.scale(self.ap_tag_b(ap0e), "0e", tag[None, :, None], "0o")[0]}
        centres_a = x * GEOM
        for blk in self.atom_enc:
            a = blk(a, ap, centres_a, mask_a, t)
        s = self.pool(a, pool)
        s = {**s, **{k: s[k] + v for k, v in self.cond_s({k: st["s"][k] for k in ("0e", "0o")}).items()}}
        for blk in self.unit_blocks:
            s = blk(s, st["z"], centre_u * GEOM, mask_u, t)
        a = add(a, {k: v[a2u] for k, v in self.unpool(s).items()})
        for blk in self.atom_dec:
            a = blk(a, ap, centres_a, mask_a, t)
        out = self.head_atom(self.head_product(a))["1o"][:, 0] + self.head_unit({"1o": s["1o"]})["1o"][:, 0][a2u]
        return out


class Denoiser(nn.Module):
    """EDM denoiser D(x_t, sigma) of a molecule: embedder + PomeloDiT, vmapped over the noisy samples; coordinates normalised by sigma_data, centred"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embedder = Embedder(cfg.mult_s, cfg.mult_z, cfg.atom_static, cfg.pair_static)
        self.dit = PomeloDiT(cfg)

    def forward(self, mol: Dict[str, torch.Tensor], x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """x_t [B, Na, 3] (normalised, with noise of standard deviation sigma [B]) -> the denoised coordinates [B, Na, 3]"""
        st = self.embedder(mol)
        xbar = x_t.mean(1, keepdim=True)
        xc = x_t - xbar
        s = sigma[:, None, None]
        c_in, c_skip, c_out = (s ** 2 + 1) ** -0.5, 1 / (1 + s ** 2), s * (s ** 2 + 1) ** -0.5
        raw = torch.vmap(lambda x, c: self.dit(x, c, st))(c_in * xc, torch.log(sigma) / 4)
        return xbar + c_skip * xc + c_out * raw
