"""The single-track DiT: the multi-scale layout of Pomelo and SimpleFold (atom encoder -> pool to tokens -> token trunk -> unpool -> atom decoder) with one plain feature
vector per atom and per token, not equivariant (as SimpleFold: the random rotations of the noising teach the symmetry).

The inputs are Pomelo's (PhysDock.pomelo.embedders.ConditionEmbedder and PomeloDiT.noisy_features / template_features: the same scalars and the same vectors, the reference
conformer and the template rotated into the frame of the sample). The simple embedder concatenates every kind, the vectors as their three components in the frame of the
sample, and maps them with one linear layer to the track:

  atoms   [cond 0e, cond 0o, noisy 0e, noisy 0o, noisy 1o (6 x 3), noisy 1e (3 x 3)] -> c_a   (+ the token conditions broadcast to the atoms)
  tokens  [cond 0e, cond 0o, template 1o (3 x 3), template 1e (1 x 3)] -> c_s                 (+ the pooled atoms)

Every block: AdaLN-Zero of the time (shift, scale, gate), attention with RMS-normed q / k, RoPE (residue index, chain) and optionally the distance term - w_h(t) |x_i - x_j|^2
of Pomelo, a sigmoid output gate; then a SwiGLU transition.
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel

from PhysDock.pomelo.dit import GEOM, W0, W_MAX_LOG, PomeloDiT, TimestepEmbedding
from PhysDock.pomelo.embedders import pad_rows
from PhysDock.pomelo.primitives.attentions import BACKENDS, apply_rope, distance_columns, pad8
from PhysDock.pomelo.relations import code_columns
from PhysDock.singletrack.configs import Config

ATOM_IN = {"0e": 2, "0o": 1, "1o": 6, "1e": 3}         # the noisy features of PomeloDiT.noisy_features
TOKEN_IN = {"1o": 3, "1e": 1}                          # the template features of PomeloDiT.template_features


def flatten(f: Dict[str, torch.Tensor]) -> torch.Tensor:
    """{kind: [..., C] or [..., C, 3]} -> [..., sum of the numbers] (the vector components as they are: not equivariant)"""
    return torch.cat([v.flatten(-2) if v.dim() > 1 and k in ("1o", "1e") else v for k, v in f.items()], -1)


class Attention(nn.Module):
    def __init__(self, c: int, heads: int, cfg: Config) -> None:
        super().__init__()
        self.heads, self.d = heads, c // heads
        if c % heads or 2 * (cfg.rope_res + cfg.rope_chain) > self.d:
            raise ValueError(f"width {c} / heads {heads}: needs a divisible width and 2 * (rope_res + rope_chain) <= the head width")
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.q_norm, self.k_norm = nn.RMSNorm(self.d, eps=1e-5), nn.RMSNorm(self.d, eps=1e-5)
        self.gate = nn.Linear(c, c)
        self.out = nn.Linear(c, c, bias=False)
        self.scale = self.d ** -0.5
        self.backend = "math"
        # w_h [same chain], w'_h [same entity] in the logits (PhysDock.pomelo.relations: codes in q and k); 0 at the start
        self.code_weight = nn.Parameter(torch.zeros(heads, 2)) if cfg.chain_codes else None

    def forward(self, x: torch.Tensor, pos: torch.Tensor, w, geo) -> torch.Tensor:
        """x [B, N, c] (normalised), pos [B, N, 3], w [B, H] or None, geo: angles [N, F], mask [N, N] bool, codes [N, 2 code_dim] (with chain_codes) -> [B, N, c]"""
        b, n, _ = x.shape
        q, k, v = self.qkv(x).unflatten(-1, (3, self.heads, self.d)).unbind(2)                  # [B, N, H, d]
        q, k = apply_rope(self.q_norm(q), geo["angles"]), apply_rope(self.k_norm(k), geo["angles"])
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)                      # [B, H, N, d]
        if w is not None:
            p = pos[:, None].expand(b, self.heads, n, 3).to(q.dtype)
            qd, kd = distance_columns(p, p, w[:, :, None, None].to(q.dtype), self.scale)
            q, k = torch.cat([q, qd], -1), torch.cat([k, kd], -1)
        if self.code_weight is not None:
            qc, kc = code_columns(geo["codes"], self.code_weight, self.scale, b, q.dtype)
            q, k = torch.cat([q, qc], -1), torch.cat([k, kc], -1)
        with sdpa_kernel(BACKENDS[self.backend]):
            o = F.scaled_dot_product_attention(pad8(q), pad8(k), pad8(v), attn_mask=geo["mask"], scale=self.scale)[..., :self.d]
        o = o.transpose(1, 2).flatten(-2)
        return self.out(o * torch.sigmoid(self.gate(x)))


class Transition(nn.Module):
    def __init__(self, c: int, expansion: float = 8 / 3) -> None:
        super().__init__()
        h = int(round(expansion * c))
        self.w13 = nn.Linear(c, 2 * h, bias=False)
        self.w2 = nn.Linear(h, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = self.w13(x).chunk(2, -1)
        return self.w2(F.silu(a) * g)


class DiTBlock(nn.Module):
    """x += gate * attention(norm(x) * (1 + scale) + shift), x += gate * transition(...), the modulation from the time embedding of the sample (AdaLN-Zero)"""

    def __init__(self, c: int, heads: int, cfg: Config) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c, elementwise_affine=False, eps=1e-5)
        self.attention = Attention(c, heads, cfg)
        self.transition = Transition(c)
        self.mod = nn.Linear(cfg.t_dim, 6 * c)
        self.dist = nn.Linear(cfg.t_dim, heads) if cfg.distance_bias else None              # w_h(t) = W0 exp(dist(t)) >= 0, as Pomelo
        with torch.no_grad():
            self.mod.weight.mul_(0.1)
            self.mod.bias.zero_()
            if not cfg.zero_gates:
                self.mod.bias[2 * c:3 * c] = 1.0
                self.mod.bias[5 * c:] = 1.0
            if self.dist is not None:
                self.dist.weight.zero_()
                self.dist.bias.zero_()

    def forward(self, x: torch.Tensor, pos: torch.Tensor, t: torch.Tensor, geo) -> torch.Tensor:
        """x [B, N, c], pos [B, N, 3] (centred), t [B, t_dim]"""
        tt = F.silu(t)
        sh1, sc1, g1, sh2, sc2, g2 = self.mod(tt)[:, None].chunk(6, -1)
        w = None if self.dist is None else W0 * torch.exp(self.dist(tt).clamp(max=W_MAX_LOG))
        x = x + g1 * self.attention(self.norm(x) * (1 + sc1) + sh1, pos, w, geo)
        return x + g2 * self.transition(self.norm(x) * (1 + sc2) + sh2)


class NormLinear(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, bias: bool = True) -> None:
        super().__init__(nn.LayerNorm(c_in), nn.Linear(c_in, c_out, bias=bias))


class SingleTrackDiT(nn.Module):
    # the vector inputs of Pomelo, unchanged (they read cfg.centroid_length only)
    noisy_features = PomeloDiT.noisy_features
    template_features = PomeloDiT.template_features

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ca, cs = cfg.c_a, cfg.c_s
        scal_a, scal_s = cfg.mult_a["0e"] + cfg.mult_a["0o"], cfg.mult_s["0e"] + cfg.mult_s["0o"]
        self.time = TimestepEmbedding(cfg.t_dim)
        self.embed_a = NormLinear(scal_a, ca)                                                  # the atom conditions (0e, 0o)
        self.embed_a_token = NormLinear(scal_s, ca, bias=False)                                # + the token conditions broadcast to the atoms
        self.noisy_in = nn.Linear(sum(c * (3 if k in ("1o", "1e") else 1) for k, c in ATOM_IN.items()), ca, bias=False)
        self.embed_s = NormLinear(scal_s, cs)
        self.templ_in = nn.Linear(sum(3 * c for c in TOKEN_IN.values()), cs, bias=False)
        self.atom_enc = nn.ModuleList([DiTBlock(ca, cfg.heads_atom, cfg) for _ in range(cfg.n_atom_blocks)])
        self.pool_lin = NormLinear(ca, cs)                                                     # PhysDock: silu(linear_downscale(x)), mean over the token
        self.token_blocks = nn.ModuleList([DiTBlock(cs, cfg.heads_token, cfg) for _ in range(cfg.n_token_blocks)])
        self.unpool = NormLinear(cs, ca)
        self.atom_dec = nn.ModuleList([DiTBlock(ca, cfg.heads_atom, cfg) for _ in range(cfg.n_atom_blocks)])
        self.head_atom = NormLinear(ca, 3, bias=False)
        self.head_token = NormLinear(cs, 3, bias=False)                                        # PhysDock's token_out
        with torch.no_grad():
            self.head_atom[1].weight.zero_()
            self.head_token[1].weight.zero_()

    def condition(self, batch, conds):
        """PhysDock's decoder interface: ConditionEmbedder already gives the per-structure inputs"""
        return conds

    def forward(self, batch, x_in: torch.Tensor, t_in: torch.Tensor, c: Dict[str, object]) -> torch.Tensor:
        """x_in [B, Na, 3] preconditioned noisy coordinates, t_in [B] noise conditioning -> the raw output [B, Na, 3] (0 at the padding)"""
        b, na = x_in.shape[:2]
        am = c["atom_mask"].to(x_in.dtype)
        a2t = c["atom_to_token"]
        t = self.time(t_in)
        with torch.autocast(x_in.device.type, enabled=False):                                  # the geometry of the inputs in float32 (Kabsch: an SVD)
            xg = pad_rows(x_in.to(torch.promote_types(x_in.dtype, torch.float32)).transpose(0, 1), am.shape[0]).transpose(0, 1) * GEOM * am[None, :, None]
            pos = (xg - xg.sum(1, keepdim=True) / am.sum()) * am[None, :, None]
            noisy = flatten(self.noisy_features(pos, c))
            centre_t = torch.einsum("ta,bac->btc", c["pool"], pos)
            templ = flatten(self.template_features(centre_t, c))
        cond_a = torch.cat([c["a"]["0e"], c["a"]["0o"]], -1)
        cond_s = torch.cat([c["s"]["0e"], c["s"]["0o"]], -1)
        a = self.noisy_in(noisy) + (self.embed_a(cond_a) + self.embed_a_token(cond_s)[a2t])[None]
        for blk in self.atom_enc:
            a = blk(a, pos, t, c["atom_geo"])
        s = torch.einsum("ta,bac->btc", c["pool"], F.silu(self.pool_lin(a))) + self.embed_s(cond_s)[None] + self.templ_in(templ)
        for blk in self.token_blocks:
            s = blk(s, centre_t, t, c["token_geo"])
        a = a + self.unpool(s)[:, a2t]
        for blk in self.atom_dec:
            a = blk(a, pos, t, c["atom_geo"])
        out = self.head_atom(a) + self.head_token(s)[:, a2t]
        return (out * am[None, :, None])[:, :na]
