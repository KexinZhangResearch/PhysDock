"""The condition embedder of Pomelo: PhysDock's input features -> the conditions of the DiT, once per structure, shared by all noisy samples. No pair representation, no pair
features, no attention: as SimpleFold, everything is per atom and per token.

  atoms   scalars  0e = linear(ref_feat without the positions and the RDKit chirality tag);  0o = V_ref * linear(0e), V_ref the signed volume of the first three bonded
                   neighbours (by atom index) in the reference conformer
          vectors  the reference positions, centred on their conformer and on the bonded-neighbour mean: they enter the vector channels of the atoms in the DiT, rotated
                   into the frame of every noisy sample by the proper Kabsch rotation of their conformer (PhysDock.pomelo.primitives.alignment)
  tokens  scalars  0e = linear(target_feat) + linear(key_res_feat) + linear(is_pocket) + linear(template availability) + the mean over the atoms of silu(linear(atom 0e))
                   (PhysDock's TokenEmbedder.embed);  0o = linear(the mean over the atoms of the atom 0o)
          vectors  the template positions of the tokens (PhysDock's self-template: the pseudo-beta atom of x_gt, where templ_feat says the template is there; one frame for
                   all of them): they enter the vector channels of the tokens in the DiT, rotated into the frame of every noisy sample by one Kabsch rotation
  and what the attentions of the DiT need: the boolean key masks of the atoms and of the tokens [N, N] (the padding; the diagonal always kept), the RoPE angles (residue
  index, chain) of the tokens and of the atoms (by their token), the pool matrix atoms -> tokens.

Padding: PhysDock pads the crop; a padded atom has atom_to_token 0 and atom_to_conformer 0, so every reduction goes through atom_mask / token_mask.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.pomelo.configs import Config
from PhysDock.pomelo.primitives.attentions import rope_angles

REF_SCALE = 0.25                        # Angstrom -> units of 4 Angstrom (the unit of the positions of the DiT)
BOND_CUTOFF = 2.05                      # Angstrom: a covalent bond between heavy atoms in a reference conformer (C-Br 1.94; the shortest 1-3 distance is ~2.2)
REF_DIM, POS, CHIRALITY = 167, slice(0, 3), slice(158, 161)
ATOM_COLUMNS = [i for i in range(REF_DIM) if not (POS.start <= i < POS.stop or CHIRALITY.start <= i < CHIRALITY.stop)]


def masked_group_mean(x: torch.Tensor, group: torch.Tensor, mask: torch.Tensor, n: int) -> torch.Tensor:
    """x [..., N, 3], group [N], mask [N] -> the mean of the real rows of the group of every row [..., N, 3] (a buffer of n groups: static shapes)"""
    m = mask.to(x.dtype)
    s = torch.zeros(*x.shape[:-2], n, 3, dtype=x.dtype, device=x.device).index_add(-2, group, x * m[:, None])
    c = torch.zeros(n, dtype=x.dtype, device=x.device).index_add(0, group, m)
    return (s / c.clamp(min=1)[:, None])[..., group, :]


def neighbours(ref_pos: torch.Tensor, conformer: torch.Tensor, atom_mask: torch.Tensor):
    """the bonded neighbours of every atom in its reference conformer (real atoms of one conformer closer than BOND_CUTOFF): the first three by atom index idx [Na, 3] and
    has3 [Na] (the atom has three), the mean matrix [Na, Na] (rows of zeros without neighbours)"""
    real = atom_mask > 0
    n = ref_pos.shape[0]
    d = torch.cdist(ref_pos, ref_pos, compute_mode="donot_use_mm_for_euclid_dist")
    eye = torch.eye(n, dtype=torch.bool, device=ref_pos.device)
    bonded = (d < BOND_CUTOFF) & (conformer[:, None] == conformer[None]) & real[:, None] & real[None] & ~eye
    order = torch.arange(n, 0, -1, device=ref_pos.device, dtype=ref_pos.dtype)                      # larger for a smaller index
    top = (bonded.to(ref_pos.dtype) * order[None]).topk(3, dim=1)
    adj = bonded.to(ref_pos.dtype)
    return top.indices, (top.values > 0).all(1), adj / adj.sum(1, keepdim=True).clamp(min=1)


def signed_volume(x: torch.Tensor, idx: torch.Tensor, has3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """x [..., Na, 3] -> [..., Na]: u1 . (u2 x u3) of the unit vectors from every atom to its three neighbours idx [Na, 3] (0 where has3 is False): a pseudoscalar in [-1, 1]"""
    v = x[..., idx.reshape(-1), :].unflatten(-2, idx.shape) - x[..., :, None, :]                      # [..., Na, 3, 3]
    u = v * torch.rsqrt(v.square().sum(-1, keepdim=True) + eps)
    vol = (u[..., 0, :] * torch.linalg.cross(u[..., 1, :], u[..., 2, :], dim=-1)).sum(-1)
    return vol * has3.to(x.dtype)


def pad_rows(x: torch.Tensor, n: int, value=0) -> torch.Tensor:
    """pad dim 0 to n rows"""
    return torch.cat([x, x.new_full((n - x.shape[0], *x.shape[1:]), value)]) if n > x.shape[0] else x


class ConditionEmbedder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        ma, ms = cfg.mult_a, cfg.mult_s
        self.linear_c = nn.Linear(len(ATOM_COLUMNS), ma["0e"], bias=False)
        self.linear_v = nn.Linear(ma["0e"], ma["0o"], bias=False)                                     # times V_ref: 0e -> 0o (no bias)
        self.linear_a = nn.Linear(ma["0e"], ms["0e"])
        self.linear_ao = nn.Linear(ma["0o"], ms["0o"], bias=False)
        self.linear_target_feat, self.linear_key_res_feat = nn.Linear(65, ms["0e"], bias=False), nn.Linear(7, ms["0e"], bias=False)
        self.linear_pocket, self.linear_templ = nn.Linear(1, ms["0e"], bias=False), nn.Linear(1, ms["0e"], bias=False)

    @staticmethod
    def template(b, tm: torch.Tensor):
        """the template positions [Nt, 3] (units of 4 A) and where there are (PhysDock's self-template: templ_feat[i, i, 39], templ_mask): b["templ_pos"] if given, else the
        pseudo-beta atom of x_gt (the positions of PhysDock's template distogram)"""
        dt = tm.dtype
        pos = b["templ_pos"] if "templ_pos" in b else b["x_gt"][b["token_to_pseudo_beta_atom"]]
        on = (torch.diagonal(b["templ_feat"][..., 39]) > 0) & (tm > 0) & (b["templ_mask"] > 0)
        return pos.to(dt) * REF_SCALE * on[:, None].to(dt), on

    def forward(self, b: Dict[str, torch.Tensor]) -> Dict[str, object]:
        ref, conf, a2t = b["ref_pos"], b["atom_to_conformer"], b["atom_to_token"]
        dt = ref.dtype
        am, tm = b["atom_mask"].to(dt), b["token_mask"].to(dt)
        na, nt = am.shape[0], tm.shape[0]
        npad = na
        ref_p, conf_p, am_p, a2t_p = pad_rows(ref, npad), pad_rows(conf, npad), pad_rows(am, npad), pad_rows(a2t, npad)
        with torch.autocast(ref.device.type, enabled=False):                                         # the geometry in float32, under any autocast
            ref_p = ref_p.to(torch.promote_types(ref_p.dtype, torch.float32))
            idx, has3, nbr_mean = neighbours(ref_p, conf_p, am_p)
            v_ref = signed_volume(ref_p, idx, has3)
            ref_s = ref_p * REF_SCALE * am_p[:, None]
            ref_nbr = torch.where(nbr_mean.sum(1, keepdim=True) > 0, ref_s - nbr_mean @ ref_s, torch.zeros_like(ref_s))    # r - mean of the bonded neighbours (frame-free difference)
        a0e = self.linear_c(pad_rows(b["ref_feat"][:, ATOM_COLUMNS].to(dt), npad)) * am_p[:, None]
        a0o = v_ref[:, None] * self.linear_v(a0e)
        onehot = F.one_hot(a2t_p, nt).T.to(dt) * am_p[None]                                            # [Nt, Na]: the real atoms of every token
        pool = onehot / onehot.sum(1, keepdim=True).clamp(min=1)
        templ_pos, templ_on = self.template(b, tm)
        s0e = (self.linear_target_feat(b["target_feat"].to(dt)) + self.linear_key_res_feat(b["key_res_feat"].to(dt)) + self.linear_pocket(b["is_pocket"].to(dt)[:, None])
               + self.linear_templ(templ_on.to(dt)[:, None]) + pool @ F.silu(self.linear_a(a0e))) * tm[:, None]
        s0o = self.linear_ao(pool @ a0o)
        res, chain = b["residue_index"], b["asym_id"]
        both = (chain[None] == chain[:, None]) & templ_on[None] & templ_on[:, None]
        nxt, prv = ((res[None] - res[:, None] == 1) & both).to(dt), ((res[None] - res[:, None] == -1) & both).to(dt)     # [i, j]: j is the next / previous residue of i
        angles = rope_angles(res, chain, self.cfg.rope_res, self.cfg.rope_chain, dt)
        eye = torch.eye(nt, dtype=torch.bool, device=tm.device)
        return {"a": {"0e": a0e, "0o": a0o}, "s": {"0e": s0e, "0o": s0o}, "v_ref": v_ref, "idx": idx, "has3": has3, "nbr_mean": nbr_mean, "pool": pool,
                "atom_to_token": a2t_p, "atom_to_conformer": conf_p, "atom_mask": am_p, "token_mask": tm, "n_atoms": na,
                "ref": ref_s, "ref_nbr": ref_nbr, "templ_pos": templ_pos, "templ_on": templ_on, "templ_next": nxt, "templ_prev": prv,
                "atom_geo": {"mask": (am_p > 0)[None].expand(na, na) | torch.eye(na, dtype=torch.bool, device=tm.device), "angles": angles[a2t_p]},
                "token_geo": {"mask": (tm > 0)[None].expand(nt, nt) | eye, "angles": angles}}
