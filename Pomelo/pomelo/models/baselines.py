"""PhysDock / AF3-like models with NO symmetry built in, to compare with Pomelo on the same data, inputs, loss, EDM preconditioning and sampler.

The same inputs (pomelo.models.encoders.InputEmbedder: the conformer features; no stereo label) and the same multi-scale layout (atoms -> tokens -> atoms), but the layers are the
ordinary ones: the reference conformer enters as plain numbers, as in PhysDock's ref_feat (ref - conformer centre; the frame of every conformer is random, so the model has to learn
to ignore it), and as AF3's atom pair features inside a conformer (the offset ref_j - ref_i and the distance RBF); the noisy coordinates enter through a linear layer; LayerNorm with
AdaLN-Zero (shift, scale, gate from the noise level), qk-normalised multi-head attention with a pair bias and a sigmoid output gate, SwiGLU transitions, a coordinate head from the atoms
plus one from the tokens. It learns the rotations from data: training uses a random rotation of every noisy sample. `geometry=True` also gives the pair bias the RBF of the noisy
distances (as the equivariant decoder has).
BasebaselineDenoiser puts a PhysDock-like Pairformer trunk in front.
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pomelo.configs import Config
from pomelo.models.decoders import GEOM, TimeEmbedding
from pomelo.models.encoders import REF_SCALE, InputEmbedder
from pomelo.utils.tensor_utils import N_RBF, rbf


class AdaLN(nn.Module):
    def __init__(self, dim: int, t_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Linear(t_dim, 3 * dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale, gate = self.mod(F.silu(t))[:, None].chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


class Attention(nn.Module):
    def __init__(self, dim: int, pair_dim: int, heads: int, t_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.ada = AdaLN(dim, t_dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.norm_q, self.norm_k = nn.RMSNorm(dim // heads), nn.RMSNorm(dim // heads)
        self.norm_p, self.bias = nn.RMSNorm(pair_dim), nn.Linear(pair_dim, heads, bias=False)
        self.g, self.o = nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pair: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h, gate = self.ada(x, t)
        q, k, v = (u.reshape(b, n, self.heads, -1).transpose(1, 2) for u in self.qkv(h).chunk(3, -1))
        bias = self.bias(self.norm_p(pair))                                                          # [N, N, H] shared by the samples, or [B, N, N, H]
        bias = bias.permute(2, 0, 1)[None] if bias.dim() == 3 else bias.permute(0, 3, 1, 2)
        o = F.scaled_dot_product_attention(self.norm_q(q), self.norm_k(k), v, bias.expand(b, -1, -1, -1))
        return self.o(o.transpose(1, 2).reshape(b, n, d) * torch.sigmoid(self.g(h))) * gate


class Transition(nn.Module):
    def __init__(self, dim: int, t_dim: int) -> None:
        super().__init__()
        hidden = int(8 / 3 * dim)
        self.ada = AdaLN(dim, t_dim)
        self.w1, self.w3, self.w2 = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False), nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h, gate = self.ada(x, t)
        return self.w2(F.silu(self.w1(h)) * self.w3(h)) * gate


class Block(nn.Module):
    def __init__(self, dim: int, pair_dim: int, heads: int, t_dim: int) -> None:
        super().__init__()
        self.attn, self.trans = Attention(dim, pair_dim, heads, t_dim), Transition(dim, t_dim)

    def forward(self, x, pair, t):
        x = x + self.attn(x, pair, t)
        return x + self.trans(x, t)


def reference_inputs(mol, st):
    """PhysDock's reference inputs as plain numbers: ref - conformer centre [Na, 3] (in units of 4 A), and for the atom pairs inside a conformer the offset and the distance RBF
    [Na, Na, 3 + N_RBF] (zero between conformers)"""
    ref, conf, nc = mol["ref_pos"], st["atom_to_conformer"], st["n_conformers"]
    mean = torch.zeros(nc, 3, dtype=ref.dtype, device=ref.device).index_add(0, conf, ref) / torch.bincount(conf, minlength=nc).clamp(min=1)[:, None].to(ref.dtype)
    rel = (ref - mean[conf]) * REF_SCALE
    same = (conf[:, None] == conf[None, :])[..., None].to(ref.dtype)
    off = rel[None, :] - rel[:, None]
    return rel, torch.cat([off, rbf(off.norm(dim=-1))], -1) * same


class BaselineDiT(nn.Module):
    def __init__(self, cfg: Config, c_atom: int = 128, c_token: int = 160, geometry: bool = False, cs_in: int = None, cz_in: int = None) -> None:
        super().__init__()
        self.cfg, self.geometry = cfg, geometry
        cs_in = cs_in or cfg.mult_s["0e"]                  # the width of the token features that condition it: the static s0e, z0e of the embedder, or a trunk's
        cz_in = cz_in or cfg.mult_z["0e"]
        self.cap_in = cfg.pair_static + 3 + N_RBF + (N_RBF if geometry else 0)
        self.time = TimeEmbedding(cfg.t_dim)
        self.a_in = nn.Linear(cfg.atom_static + 3, c_atom)
        self.x_in = nn.Linear(3, c_atom, bias=False)
        self.cond_s = nn.Linear(cs_in, c_token)
        self.enc = nn.ModuleList([Block(c_atom, self.cap_in, cfg.heads, cfg.t_dim) for _ in range(cfg.n_atom_blocks)])
        self.pool = nn.Linear(c_atom, c_token)
        self.tokens = nn.ModuleList([Block(c_token, cz_in, cfg.heads, cfg.t_dim) for _ in range(cfg.n_token_blocks)])
        self.unpool = nn.Linear(c_token, c_atom)
        self.dec = nn.ModuleList([Block(c_atom, self.cap_in, cfg.heads, cfg.t_dim) for _ in range(cfg.n_atom_blocks)])
        self.norm_out = nn.LayerNorm(c_atom)
        self.head_atom, self.head_token = nn.Linear(c_atom, 3), nn.Linear(c_token, 3)

    def forward(self, x: torch.Tensor, c_noise: torch.Tensor, st: Dict[str, object]) -> torch.Tensor:
        """x [B, Na, 3] centred noisy coordinates times c_in, c_noise [B] -> F [B, Na, 3]"""
        b = x.shape[0]
        t = self.time(c_noise)
        pool, a2t = st["pool"], st["atom_to_token"]
        a = self.a_in(torch.cat([st["a0e"], st["ref_rel"]], -1))[None] + self.x_in(x * GEOM)
        ap = torch.cat([st["ap0e"], st["ref_pair"]], -1)                                                        # [N, N, C] shared by the samples
        if self.geometry:
            ap = torch.cat([ap[None].expand(b, -1, -1, -1), rbf(torch.cdist(x, x) * GEOM)], -1)                 # per sample
        for blk in self.enc:
            a = blk(a, ap, t)
        s = torch.einsum("ua,bac->buc", pool, F.silu(self.pool(a)))
        s = s + self.cond_s(st.get("s_plain", st["s0e"]))[None]
        z = st.get("z_plain", st["z0e"])
        for blk in self.tokens:
            s = blk(s, z, t)
        a = a + self.unpool(s)[:, a2t]
        for blk in self.dec:
            a = blk(a, ap, t)
        return self.head_atom(self.norm_out(a)) + self.head_token(s)[:, a2t]


def static_inputs(embedder: InputEmbedder, mol) -> Dict[str, object]:
    st = embedder(mol)
    rel, pair = reference_inputs(st["mol"], st)
    return {**st, "ref_rel": rel, "ref_pair": pair}


def edm(dit, st, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    xbar = x_t.mean(1, keepdim=True)
    xc = x_t - xbar
    sg = sigma[:, None, None]
    c_in, c_skip, c_out = (sg ** 2 + 1) ** -0.5, 1 / (1 + sg ** 2), sg * (sg ** 2 + 1) ** -0.5
    return xbar + c_skip * xc + c_out * dit(c_in * xc, torch.log(sigma) / 4, st)


# ---------------------------------------------------------------------------------------------------------------- the trunk of the second baseline
class TriangleMultiplication(nn.Module):
    """AF2 / AF3 triangle multiplicative update (outgoing: sum_k a_ik b_jk, incoming: sum_k a_ki b_kj), gated, with the norm of the product"""

    def __init__(self, c: int, outgoing: bool, hidden: int = 16) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.norm_in, self.norm_out = nn.LayerNorm(c), nn.LayerNorm(hidden)
        self.ab, self.gab = nn.Linear(c, 2 * hidden, bias=False), nn.Linear(c, 2 * hidden)
        self.gate, self.out = nn.Linear(c, c), nn.Linear(hidden, c)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(z)
        a, b = (self.ab(x) * torch.sigmoid(self.gab(x))).chunk(2, -1)
        o = torch.einsum("ikc,jkc->ijc", a, b) if self.outgoing else torch.einsum("kic,kjc->ijc", a, b)
        return self.out(self.norm_out(o)) * torch.sigmoid(self.gate(x))


class TriangleAttention(nn.Module):
    """AF2 / AF3 triangle attention around the starting node (rows) or the ending node (columns), with a bias from the third edge and a sigmoid output gate"""

    def __init__(self, c: int, heads: int, starting: bool) -> None:
        super().__init__()
        self.heads, self.starting = heads, starting
        self.norm = nn.LayerNorm(c)
        self.qkv, self.bias = nn.Linear(c, 3 * c, bias=False), nn.Linear(c, heads, bias=False)
        self.g, self.o = nn.Linear(c, c), nn.Linear(c, c)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.starting:
            z = z.transpose(0, 1)
        n = z.shape[0]
        x = self.norm(z)
        q, k, v = (u.reshape(n, n, self.heads, -1).permute(0, 2, 1, 3) for u in self.qkv(x).chunk(3, -1))        # [i, H, j, d]
        bias = self.bias(x).permute(2, 0, 1)[None]                                                                 # [1, H, j, k]
        o = F.scaled_dot_product_attention(q, k, v, bias.expand(n, -1, -1, -1))
        o = self.o(o.permute(0, 2, 1, 3).reshape(n, n, -1) * torch.sigmoid(self.g(x)))
        return o if self.starting else o.transpose(0, 1)


class PlainTransition(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        h = int(8 / 3 * c)
        self.norm, self.w1, self.w3, self.w2 = nn.LayerNorm(c), nn.Linear(c, h, bias=False), nn.Linear(c, h, bias=False), nn.Linear(h, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.w2(F.silu(self.w1(h)) * self.w3(h))


class PairformerBlock(nn.Module):
    """The block of a PhysDock-like Pairformer, without any symmetry: z += outer product of s, triangle multiplication (outgoing, incoming), triangle attention (starting, ending node),
    transition; s += attention with the bias from z, transition"""

    def __init__(self, c_s: int, c_z: int, heads: int = 4, c_outer: int = 8) -> None:
        super().__init__()
        self.norm_s, self.a, self.b, self.out_outer = nn.LayerNorm(c_s), nn.Linear(c_s, c_outer, bias=False), nn.Linear(c_s, c_outer, bias=False), nn.Linear(c_outer * c_outer, c_z)
        self.mul_out, self.mul_in = TriangleMultiplication(c_z, True), TriangleMultiplication(c_z, False)
        self.att_start, self.att_end = TriangleAttention(c_z, heads, True), TriangleAttention(c_z, heads, False)
        self.z_trans = PlainTransition(c_z)
        self.heads = heads
        self.norm_p, self.bias = nn.LayerNorm(c_z), nn.Linear(c_z, heads, bias=False)
        self.qkv, self.g, self.o = nn.Linear(c_s, 3 * c_s, bias=False), nn.Linear(c_s, c_s), nn.Linear(c_s, c_s)
        self.s_trans = PlainTransition(c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        x = self.norm_s(s)
        a, b = self.a(x), self.b(x)
        z = z + self.out_outer(torch.einsum("ic,jd->ijcd", a, b).flatten(2))
        z = z + self.mul_out(z)
        z = z + self.mul_in(z)
        z = z + self.att_start(z)
        z = z + self.att_end(z)
        z = z + self.z_trans(z)
        n = s.shape[0]
        q, k, v = (u.reshape(n, self.heads, -1).transpose(0, 1) for u in self.qkv(x).chunk(3, -1))
        bias = self.bias(self.norm_p(z)).permute(2, 0, 1)
        o = F.scaled_dot_product_attention(q, k, v, bias).transpose(0, 1).reshape(n, -1)
        s = s + self.o(o * torch.sigmoid(self.g(x)))
        return s + self.s_trans(s), z


class BasebaselineDenoiser(nn.Module):
    """The second baseline: a PhysDock-like trunk too. The static token features of the embedder (s0e, z0e) run through `n_trunk` standard Pairformer blocks, and the result conditions
    the PhysDock-like DiT of BaselineDenoiser. Nothing in it is equivariant (trained with random rotations)."""

    def __init__(self, cfg: Config, c_atom: int, c_token: int, c_s: int = 64, c_z: int = 32, n_trunk: int = 2, geometry: bool = False) -> None:
        super().__init__()
        self.embedder = InputEmbedder(cfg)
        self.in_s = nn.Linear(cfg.mult_s["0e"], c_s)
        self.in_z = nn.Linear(cfg.mult_z["0e"], c_z)
        self.trunk = nn.ModuleList([PairformerBlock(c_s, c_z, cfg.heads) for _ in range(n_trunk)])
        self.dit = BaselineDiT(cfg, c_atom, c_token, geometry, cs_in=c_s, cz_in=c_z)

    def forward(self, mol, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        st = static_inputs(self.embedder, mol)
        s, z = self.in_s(st["s0e"]), self.in_z(st["z0e"])
        for blk in self.trunk:
            s, z = blk(s, z)
        return edm(self.dit, {**st, "s_plain": s, "z_plain": z}, x_t, sigma)


class BaselineDenoiser(nn.Module):
    """same interface as pomelo.models.model.Denoiser: D(mol, x_t [B, N, 3], sigma [B]) with the EDM preconditioning of the centred coordinates"""

    def __init__(self, cfg: Config, c_atom: int = 128, c_token: int = 160, geometry: bool = False) -> None:
        super().__init__()
        self.embedder = InputEmbedder(cfg)
        self.dit = BaselineDiT(cfg, c_atom, c_token, geometry)

    def forward(self, mol, x_t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return edm(self.dit, static_inputs(self.embedder, mol), x_t, sigma)
