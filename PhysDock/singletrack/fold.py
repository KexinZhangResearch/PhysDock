"""SingleTrackFold: the single-track DiT with the inputs of SimpleFold (third_party/ml-simplefold, model/torch/architecture.py FoldingDiT) as far as PhysDock's features
have them, without the protein language model:

  atom_feat  [Fourier(ref_pos / sigma_data_conformer) (771), the residue features broadcast to the atoms (is_protein, is_ligand, target_feat 65, is_pocket, key_res_feat 7),
              absolute position encoding of the token index (c_s + 1), ref_feat without the positions (164: charge, element, aromatic, degree, ... instead of
              SimpleFold's charge, element and atom name), atom_mask] -> Linear -> LayerNorm -> SiLU                                                   (atom_feat_proj)
  noisy      Fourier(noisy_pos) (771) -> Linear                                                                                                   (atom_pos_proj)
  atom_in    Linear([atom_feat, noisy])                                                                                                           (atom_in_proj)
  t          the time embedding + LayerNorm(Linear(log of the number of tokens))                                                                  (length condition)
  RoPE       axial, base 100, every channel of a head: atoms (ref_space_uid = conformer, ref_pos xyz), tokens (residue_index, entity_id, asym_id, sym_id)
  relations  the same chain / entity as codes in q and k (chain_codes), the bond graph of the ligands into atom_feat and a few rounds of messages along the bonds
             (ligand_graph; PhysDock.pomelo.relations)
  tokens     the mean of the atoms of every token (SimpleFold: concatenated with the ESM embedding; here alone) + the self-template (use_template, default on): the
             pseudo-beta xyz of every chain centred, randomly rotated and shifted on its own (PhysDock.pomelo.embedders.chain_template), / sigma_data, and its flag

ref_pos: every reference conformer centred, randomly rotated and shifted (1 A) on its own, / sigma_data_conformer (4 A; SimpleFold: the library frame / 5 A), as in the
other models of the comparison (PhysDock-base, Pomelo, SingleTrack). The layout is SimpleFold's: atom encoder -> mean over the atoms of every token -> token trunk ->
broadcast + the atom skip -> atom decoder -> head.
Not equivariant: the random rotations of the noisy samples (PhysDock's noising and sampler) teach the symmetry, the translations are removed by centring.
"""
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.pomelo.dit import TimestepEmbedding
from PhysDock.pomelo.embedders import chain_template
from PhysDock.pomelo.relations import BondMessagePassing, bond_graph, chain_entity_codes, graph_features, n_graph_features
from PhysDock.singletrack.configs import Config, preset
from PhysDock.singletrack.dit import DiTBlock, NormLinear
from PhysDock.singletrack.simple import SingleTrackSimple, centre
from PhysDock.utils.tensor_utils import group_random_augmentation

REF_TRANS = 1.0                 # Angstrom: std of the random translation of every reference conformer
RES_DIM = 2 + 65 + 1 + 7        # is_protein, is_ligand, target_feat, is_pocket, key_res_feat


class FourierEncoding(nn.Module):
    """SimpleFold's FourierPositionEncoding (include_input, log sampling): [..., d] -> [..., d + 2 d n] (the input, sin, cos of 2^linspace(min, max, n) times it)"""

    def __init__(self, in_dim: int = 3, num_freqs: int = 128, min_freq_log2: float = 0.0, max_freq_log2: float = 12.0) -> None:
        super().__init__()
        self.freqs = nn.Buffer(2.0 ** torch.linspace(min_freq_log2, max_freq_log2, num_freqs), persistent=False)
        self.dim = in_dim * (1 + 2 * num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x[..., None] * self.freqs.to(x.dtype)).flatten(-2)
        return torch.cat([x, torch.sin(a), torch.cos(a)], -1)


def absolute_encoding(pos: torch.Tensor, dim: int) -> torch.Tensor:
    """SimpleFold's AbsolutePositionEncoding (in_dim 1, include_input): pos [N] -> [N, dim + 1] (sin, cos of pi 2^linspace(0, log2(224) - 1, dim / 2), the input)"""
    omega = math.pi * 2 ** torch.linspace(0, math.log2(224) - 1, dim // 2, device=pos.device, dtype=pos.dtype)
    a = pos[:, None] * omega
    return torch.cat([torch.sin(a), torch.cos(a), pos[:, None]], -1)


def axial_rope_angles(pos: torch.Tensor, head_dim: int, base: float = 100.0) -> torch.Tensor:
    """SimpleFold's compute_axial_cis: pos [N, k] -> the RoPE angles [N, k * head_dim // (2 k)] (axis i: pos_i times 1 / base^(j 2 k / head_dim))"""
    k = pos.shape[-1]
    n = head_dim // (2 * k)
    freq = 1.0 / base ** (torch.arange(0, head_dim, 2 * k, device=pos.device, dtype=pos.dtype)[:n] / head_dim)
    return (pos[..., None] * freq).flatten(-2)


class FoldEmbedder(nn.Module):
    """the per-structure inputs: atom_feat [Na, c_a] (after atom_feat_proj), the RoPE angles of the atoms and the tokens, the length, masks and the pool matrix"""

    def __init__(self, cfg: Config, sigma_data: float = 16.0) -> None:
        super().__init__()
        self.cfg, self.sigma_data = cfg, sigma_data                                           # the template coordinates / sigma_data (as the noisy coordinates)
        self.fourier = FourierEncoding()
        self.n_graph = n_graph_features(cfg.rwse_steps) if cfg.ligand_graph else 0
        self.atom_feat_proj = nn.Sequential(nn.Linear(self.fourier.dim + RES_DIM + cfg.c_s + 1 + 164 + 1 + self.n_graph, cfg.c_a), nn.LayerNorm(cfg.c_a), nn.SiLU())
        self.bond_mp = BondMessagePassing(cfg.c_a, cfg.bond_mp_rounds) if cfg.ligand_graph and cfg.bond_mp_rounds else None

    def forward(self, b: Dict[str, torch.Tensor]) -> Dict[str, object]:
        ref_feat = b["ref_feat"]
        dt = ref_feat.dtype
        am, tm, a2t = b["atom_mask"].to(dt), b["token_mask"].to(dt), b["atom_to_token"]
        na, nt = am.shape[0], tm.shape[0]
        ref_pos = group_random_augmentation(ref_feat[:, :3], b["atom_to_conformer"], am, REF_TRANS).to(dt) / self.cfg.sigma_data_conformer   # per conformer
        res = torch.cat([b["is_protein"].to(dt)[:, None], b["is_ligand"].to(dt)[:, None], b["target_feat"].to(dt), b["is_pocket"].to(dt)[:, None],
                         b["key_res_feat"].to(dt)], -1) * tm[:, None]
        token_index = torch.arange(nt, device=a2t.device, dtype=dt)
        atom_feat = [self.fourier(ref_pos), res[a2t], absolute_encoding(token_index, self.cfg.c_s)[a2t], ref_feat[:, 3:], am[:, None]]
        if self.n_graph:                                                                               # the bond graph (per token: a ligand token is one atom)
            adj, edges = bond_graph(b, tm)
            atom_feat.append(graph_features(adj, edges, self.cfg.rwse_steps).to(dt)[a2t] * am[:, None])
        a = self.atom_feat_proj(torch.cat(atom_feat, -1)) * am[:, None]
        onehot = F.one_hot(a2t, nt).T.to(dt) * am[None]
        pool = onehot / onehot.sum(1, keepdim=True).clamp(min=1)
        if self.bond_mp is not None:                                                                   # messages along the bonds (the tokens in the graph only)
            h = pool @ a
            a = a + (self.bond_mp(h, edges) - h)[a2t] * am[:, None]
        atom_pe = torch.cat([b["atom_to_conformer"].to(dt)[:, None], ref_pos], -1)                                   # SimpleFold: ref_space_uid, ref_pos
        token_pe = torch.stack([b[k].to(dt) for k in ("residue_index", "entity_id", "asym_id", "sym_id")], -1)       # SimpleFold
        code_geo = ({}, {})
        if self.cfg.chain_codes:                                                                       # + the same chain / entity as codes in q and k
            codes = chain_entity_codes(b, tm, self.cfg.chain_codes).to(dt)
            code_geo = ({"codes": codes[a2t] * am[:, None]}, {"codes": codes})
        on = (torch.diagonal(b["templ_feat"][..., 39]) > 0) & (tm > 0) & (b["templ_mask"] > 0)
        templ = b["templ_pos"] if "templ_pos" in b else b["x_gt"][b["token_to_pseudo_beta_atom"]]
        eye_a, eye_t = torch.eye(na, dtype=torch.bool, device=am.device), torch.eye(nt, dtype=torch.bool, device=am.device)
        return {"a": a, "s": torch.zeros(nt, 1, dtype=dt, device=am.device),                                        # no token conditions (SimpleFold: ESM)
                "pool": pool, "atom_to_token": a2t, "atom_mask": am, "token_mask": tm,
                "log_length": torch.log(tm.sum().clamp(min=1))[None], "templ_pos": chain_template(templ, b["asym_id"], on).to(dt) / self.sigma_data, "templ_on": on.to(dt),
                "atom_geo": {"mask": (am > 0)[None].expand(na, na) | eye_a, "angles": axial_rope_angles(atom_pe, self.cfg.c_a // self.cfg.heads_atom), **code_geo[0]},
                "token_geo": {"mask": (tm > 0)[None].expand(nt, nt) | eye_t, "angles": axial_rope_angles(token_pe, self.cfg.c_s // self.cfg.heads_token), **code_geo[1]}}


class FoldDiT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ca, cs = cfg.c_a, cfg.c_s
        self.fourier = FourierEncoding()
        self.time = TimestepEmbedding(cfg.t_dim)
        self.length = nn.Sequential(nn.Linear(1, cfg.t_dim, bias=False), nn.LayerNorm(cfg.t_dim))
        self.atom_pos_proj = nn.Linear(self.fourier.dim, ca, bias=False)
        self.atom_in_proj = nn.Linear(2 * ca, ca, bias=False)
        self.templ_in = nn.Linear(3 + 1, cs, bias=False) if cfg.use_template else None
        self.atom_enc = nn.ModuleList([DiTBlock(ca, cfg.heads_atom, cfg) for _ in range(cfg.n_atom_blocks)])
        self.atom2latent = NormLinear(ca, cs)                                                  # SimpleFold's atom2latent_proj (Linear, LayerNorm), then the mean
        self.token_blocks = nn.ModuleList([DiTBlock(cs, cfg.heads_token, cfg) for _ in range(cfg.n_token_blocks)])
        self.latent2atom = nn.Sequential(nn.Linear(cs, cs), nn.SiLU(), nn.LayerNorm(cs), nn.Linear(cs, ca))
        self.atom_skip = nn.Linear(ca, ca, bias=False)                                          # SimpleFold adds the atom latent (of width c_s there) before latent2atom
        self.atom_dec = nn.ModuleList([DiTBlock(ca, cfg.heads_atom, cfg) for _ in range(cfg.n_atom_blocks)])
        self.head = NormLinear(ca, 3, bias=False)
        with torch.no_grad():
            self.head[1].weight.zero_()

    def condition(self, batch, conds):
        return conds

    def forward(self, batch, x_in: torch.Tensor, t_in: torch.Tensor, c: Dict[str, object]) -> torch.Tensor:
        """x_in [B, Na, 3] preconditioned noisy coordinates (normalised), t_in [B] -> the raw output [B, Na, 3] (0 at the padding)"""
        b = x_in.shape[0]
        am, a2t = c["atom_mask"].to(x_in.dtype), c["atom_to_token"]
        t = self.time(t_in) + self.length(c["log_length"].to(x_in.dtype))[None]
        with torch.autocast(x_in.device.type, enabled=False):                                  # the geometry in float32 (float64 in the tests)
            dt = torch.promote_types(x_in.dtype, torch.float32)
            pos = centre(x_in.to(dt), am)                                                      # [B, Na, 3]
            noisy = self.fourier(pos) * am[None, :, None]
        a = self.atom_in_proj(torch.cat([c["a"][None].expand(b, -1, -1), self.atom_pos_proj(noisy)], -1))
        for blk in self.atom_enc:
            a = blk(a, pos, t, c["atom_geo"])
        s = torch.einsum("ta,bac->btc", c["pool"], self.atom2latent(a))
        if self.templ_in is not None:                                                          # the same template for all samples of the structure
            s = s + self.templ_in(torch.cat([c["templ_pos"], c["templ_on"][:, None]], -1).to(s.dtype))[None]
        centre_t = torch.einsum("ta,bac->btc", c["pool"].to(pos.dtype), pos)                  # (the positions of the distance term, off in fold_mini)
        for blk in self.token_blocks:
            s = blk(s, centre_t, t, c["token_geo"])
        a = self.latent2atom(s)[:, a2t] + self.atom_skip(a)
        for blk in self.atom_dec:
            a = blk(a, pos, t, c["atom_geo"])
        return self.head(a) * am[None, :, None]


class SingleTrackFold(SingleTrackSimple):
    """SingleTrackFold(**model_config): PhysDock's model config (configs/model/singletrack_fold_mini.yaml) plus `singletrack_fold`: a preset of
    PhysDock.singletrack.configs or a dict. Training, sampling and compilation are Pomelo's."""

    def __init__(self, num_augmentation_sample: int, diffusion: dict, sigma_data: float, singletrack_fold="fold_mini", **physdock_only) -> None:
        nn.Module.__init__(self)
        from PhysDock.models.diffusion import build_diffuser
        self.cfg = preset(singletrack_fold)
        self.num_augmentation_sample, self.sigma_data = num_augmentation_sample, sigma_data
        self.diffuser = build_diffuser(diffusion)
        self.diffusion_conditioning = FoldEmbedder(self.cfg, sigma_data)
        self.dit = FoldDiT(self.cfg)
