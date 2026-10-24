"""SingleTrackSimple: the single-track DiT of SingleTrack with the plain inputs of SimpleFold (without the protein language model) instead of Pomelo's typed features:

  atoms   ref_feat [Na, 167] as it is (its first three columns: the reference conformer, randomly rotated and shifted per conformer by the featuriser), in 4 A units
          noisy_pos [B, Na, 3] (the preconditioned noisy sample, centred, in 4 A units)
  tokens  target_feat [Nt, 65], key_res_feat [Nt, 7], is_pocket [Nt, 1]
          templ_pos [B, Nt, 3] (the self-template: the pseudo-beta atoms, centred and randomly rotated per sample, independently of the noisy sample) and its flag [Nt, 1]
  RoPE    residue index and chain of every token (the atoms: of their token)
  t       the noise conditioning

Nothing is equivariant: the random rotations of the noisy samples (PhysDock's noising and sampler) and of the template teach the symmetry; the translations are removed by
centring. The layout and the blocks are SingleTrack's (PhysDock.singletrack.dit).
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.pomelo.dit import GEOM, TimestepEmbedding
from PhysDock.pomelo.embedders import REF_SCALE
from PhysDock.pomelo.model import Pomelo
from PhysDock.pomelo.primitives.attentions import rope_angles
from PhysDock.singletrack.configs import Config, preset
from PhysDock.singletrack.dit import DiTBlock, NormLinear
from PhysDock.utils.tensor_utils import uniform_random_rotation

REF_DIM = 167


def centre(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [..., N, 3] centred on the rows of mask [N] (0 elsewhere)"""
    m = mask.to(x.dtype)[:, None]
    return (x - (x * m).sum(-2, keepdim=True) / m.sum().clamp(min=1)) * m


class SimpleEmbedder(nn.Module):
    """the per-structure conditions: atoms a [Na, c_a], tokens s [Nt, c_s], and what the DiT needs (masks, RoPE angles, pool matrix, the template)"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.linear_ref = nn.Linear(REF_DIM, cfg.c_a)
        self.linear_token = nn.Linear(65 + 7 + 1, cfg.c_s)
        self.linear_a = nn.Linear(cfg.c_a, cfg.c_s)

    def forward(self, b: Dict[str, torch.Tensor]) -> Dict[str, object]:
        ref_feat = b["ref_feat"]
        dt = ref_feat.dtype
        am, tm, a2t = b["atom_mask"].to(dt), b["token_mask"].to(dt), b["atom_to_token"]
        na, nt = am.shape[0], tm.shape[0]
        ref_feat = torch.cat([ref_feat[:, :3] * REF_SCALE, ref_feat[:, 3:]], -1)
        a = self.linear_ref(ref_feat) * am[:, None]
        onehot = F.one_hot(a2t, nt).T.to(dt) * am[None]
        pool = onehot / onehot.sum(1, keepdim=True).clamp(min=1)
        tok = torch.cat([b["target_feat"].to(dt), b["key_res_feat"].to(dt), b["is_pocket"].to(dt)[:, None]], -1)
        s = (self.linear_token(tok) + pool @ F.silu(self.linear_a(a))) * tm[:, None]
        on = (torch.diagonal(b["templ_feat"][..., 39]) > 0) & (tm > 0) & (b["templ_mask"] > 0)
        pos = b["templ_pos"] if "templ_pos" in b else b["x_gt"][b["token_to_pseudo_beta_atom"]]
        templ = centre(pos.to(dt) * REF_SCALE, on)
        angles = rope_angles(b["residue_index"], b["asym_id"], self.cfg.rope_res, self.cfg.rope_chain, dt)
        eye_a, eye_t = torch.eye(na, dtype=torch.bool, device=am.device), torch.eye(nt, dtype=torch.bool, device=am.device)
        return {"a": a, "s": s, "pool": pool, "atom_to_token": a2t, "atom_mask": am, "token_mask": tm, "templ_pos": templ, "templ_on": on.to(dt),
                "atom_geo": {"mask": (am > 0)[None].expand(na, na) | eye_a, "angles": angles[a2t]},
                "token_geo": {"mask": (tm > 0)[None].expand(nt, nt) | eye_t, "angles": angles}}


class SimpleDiT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ca, cs = cfg.c_a, cfg.c_s
        self.time = TimestepEmbedding(cfg.t_dim)
        self.cond_a = NormLinear(ca, ca)
        self.cond_a_token = NormLinear(cs, ca, bias=False)                                     # the token conditions broadcast to the atoms
        self.noisy_in = nn.Linear(3, ca, bias=False)
        self.cond_s = NormLinear(cs, cs)
        self.templ_in = nn.Linear(3 + 1, cs, bias=False)                                       # templ_pos and its flag
        self.atom_enc = nn.ModuleList([DiTBlock(ca, cfg.heads_atom, cfg) for _ in range(cfg.n_atom_blocks)])
        self.pool_lin = NormLinear(ca, cs)
        self.token_blocks = nn.ModuleList([DiTBlock(cs, cfg.heads_token, cfg) for _ in range(cfg.n_token_blocks)])
        self.unpool = NormLinear(cs, ca)
        self.atom_dec = nn.ModuleList([DiTBlock(ca, cfg.heads_atom, cfg) for _ in range(cfg.n_atom_blocks)])
        self.head_atom = NormLinear(ca, 3, bias=False)
        self.head_token = NormLinear(cs, 3, bias=False)
        with torch.no_grad():
            self.head_atom[1].weight.zero_()
            self.head_token[1].weight.zero_()

    def condition(self, batch, conds):
        return conds

    def forward(self, batch, x_in: torch.Tensor, t_in: torch.Tensor, c: Dict[str, object]) -> torch.Tensor:
        """x_in [B, Na, 3] preconditioned noisy coordinates, t_in [B] -> the raw output [B, Na, 3] (0 at the padding)"""
        b = x_in.shape[0]
        am, a2t = c["atom_mask"].to(x_in.dtype), c["atom_to_token"]
        t = self.time(t_in)
        with torch.autocast(x_in.device.type, enabled=False):                                  # the geometry in float32 (float64 in the tests)
            dt = torch.promote_types(x_in.dtype, torch.float32)
            pos = centre(x_in.to(dt) * GEOM, am)                                               # [B, Na, 3]
            centre_t = torch.einsum("ta,bac->btc", c["pool"].to(dt), pos)
            rot = uniform_random_rotation((b,), device=x_in.device).to(dt)                     # the template in a random frame of its own per sample
            templ = torch.einsum("bij,tj->bti", rot, c["templ_pos"].to(dt))
            templ = torch.cat([templ, c["templ_on"].to(dt)[None, :, None].expand(b, -1, 1)], -1)
        a = self.noisy_in(pos) + (self.cond_a(c["a"]) + self.cond_a_token(c["s"])[a2t])[None]
        for blk in self.atom_enc:
            a = blk(a, pos, t, c["atom_geo"])
        s = torch.einsum("ta,bac->btc", c["pool"], F.silu(self.pool_lin(a))) + self.cond_s(c["s"])[None] + self.templ_in(templ)
        for blk in self.token_blocks:
            s = blk(s, centre_t, t, c["token_geo"])
        a = a + self.unpool(s)[:, a2t]
        for blk in self.atom_dec:
            a = blk(a, pos, t, c["atom_geo"])
        return (self.head_atom(a) + self.head_token(s)[:, a2t]) * am[None, :, None]


class SingleTrackSimple(Pomelo):
    """SingleTrackSimple(**model_config): PhysDock's model config (configs/model/singletrack_simple_mini.yaml) plus `singletrack_simple`: a preset of
    PhysDock.singletrack.configs or a dict. Training, sampling and compilation are Pomelo's."""

    def __init__(self, num_augmentation_sample: int, diffusion: dict, sigma_data: float, singletrack_simple="simple_mini", **physdock_only) -> None:
        nn.Module.__init__(self)
        from PhysDock.models.diffusion import build_diffuser
        self.cfg = preset(singletrack_simple)
        self.num_augmentation_sample, self.sigma_data = num_augmentation_sample, sigma_data
        self.diffuser = build_diffuser(diffusion)
        self.diffusion_conditioning = SimpleEmbedder(self.cfg)
        self.dit = SimpleDiT(self.cfg)

    @torch.no_grad()
    def stream_rms(self, c) -> torch.Tensor:
        am, tm = c["atom_mask"][:, None], c["token_mask"][:, None]
        rms = lambda x, m: ((x.float() ** 2 * m).sum() / (m.sum() * x.shape[-1])).sqrt()
        return torch.stack([rms(c["a"], am), rms(c["s"], tm), torch.zeros((), device=am.device)])
