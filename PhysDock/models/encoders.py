"""Encoder (the paper's DiffusionConditioning): atom embedding, Evoformer on the MSA, template embedding, Pairformer."""
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import (AtomTransformer, Evoformer, Linear, Pairformer, RMSNorm, SwiGLUFFN,
                                        Triangleformer)
from PhysDock.utils.tensor_utils import pair_mask, segment_mean_pool


class Conditions(NamedTuple):
    """Encoder output: a [N_atom, c_a], ap [N_atom, N_atom, c_ap], s [N_token, c_s], z [N_token, N_token, c_z]."""
    a: torch.Tensor
    ap: torch.Tensor
    s: torch.Tensor
    z: torch.Tensor


class TemplatePairEmbedder(nn.Module):
    """Pair update from the self-template (distogram of pseudo-beta atoms within each chain)."""

    def __init__(self, templ_dim: int, c_z: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__()
        self.norm_in = RMSNorm(c_z)
        self.linear_in = Linear(c_z, c_z, bias=False)
        self.linear_templ_feat = Linear(templ_dim, c_z, bias=False)
        self.triangleformer = Triangleformer(c_z, inf, eps, no_blocks)
        self.norm_out = RMSNorm(c_z, eps)
        self.linear_out = Linear(c_z, c_z, bias=False)

    def forward(self, batch, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch: templ_feat [N_token, N_token, 40], asym_id [N_token], token_mask [N_token], templ_mask []
            z: [N_token, N_token, c_z]

        Returns:
            [N_token, N_token, c_z] update
        """
        templ_feat = batch["templ_feat"]
        same_chain = (batch["asym_id"][None] ==
                      batch["asym_id"][:, None]).type_as(templ_feat)
        z_mask = pair_mask(batch["token_mask"]) * \
            templ_feat[..., 39] * same_chain
        z = self.linear_in(self.norm_in(z)) + \
            self.linear_templ_feat(templ_feat)
        z = self.triangleformer(z, z_mask)
        return self.linear_out(F.relu(self.norm_out(z))) * batch["templ_mask"]


class RelPosEmbedder(nn.Module):
    """Relative residue / chain position features (AF3 Alg. 3) plus the ligand pair features."""

    def __init__(self, c_z: int, r_max: int, s_max: int, ligand_pair_dim: int) -> None:
        super().__init__()
        self.r_max, self.s_max = r_max, s_max
        # residue bins (2 r_max + 2) + ligand pair features + same entity + chain bins (2 s_max + 2)
        self.linear = Linear(2 * r_max + 2 + ligand_pair_dim + 1 +
                             2 * s_max + 2, c_z, bias=False)

    def forward(self, batch) -> torch.Tensor:
        """
        Args:
            batch: asym_id, sym_id, entity_id, residue_index [N_token], ligand_pair_feat [N_token, N_token, 42]

        Returns:
            [N_token, N_token, c_z]
        """
        with torch.no_grad():
            asym, sym, entity, res = batch["asym_id"], batch["sym_id"], batch["entity_id"], batch["residue_index"]
            same_chain = asym[:, None] == asym[None, :]
            same_entity = entity[:, None] == entity[None, :]
            d_res = torch.clamp(
                res[:, None] - res[None, :] + self.r_max, 0, 2 * self.r_max)
            d_res = torch.where(same_chain, d_res, 2 * self.r_max + 1)
            d_chain = torch.clamp(
                sym[:, None] - sym[None, :] + self.s_max, 0, 2 * self.s_max)
            d_chain = torch.where(same_chain | ~same_entity,
                                  2 * self.s_max + 1, d_chain)
            rel_feat = torch.cat([
                F.one_hot(d_res.long(), 2 * self.r_max + 2).float(),
                batch["ligand_pair_feat"],
                same_entity[..., None],
                F.one_hot(d_chain.long(), 2 * self.s_max + 2).float(),
            ], dim=-1)
        return self.linear(rel_feat)


class AtomEmbedder(nn.Module):
    """Embed the reference conformers into atom (a) and atom-pair (ap) features, then run the atom transformer."""

    def __init__(self, ref_dim: int, c_a: int, c_ap: int, inf: float, eps: float, no_blocks_atom: int, local_attention: bool = False,
                 same_conformer_bias: bool = True) -> None:
        super().__init__()
        self.local_attention = local_attention
        self.linear_c = Linear(ref_dim, c_a, bias=False)
        self.linear_p = Linear(3, c_ap, bias=False)
        self.linear_d = Linear(1, c_ap, bias=False)
        if same_conformer_bias:          # a constant vector for the pairs of one conformer (the released parameters; it is a bias of the pair features)
            self.linear_v = Linear(1, c_ap, bias=False)
        self.linear_c_l = Linear(c_a, c_ap, bias=False)
        self.linear_c_m = Linear(c_a, c_ap, bias=False)
        self.ffn = SwiGLUFFN(c_ap)
        self.atom_transformer = AtomTransformer(
            c_a, c_ap, inf, eps, no_blocks_atom)

    def forward(self, batch):
        """
        Args:
            batch: ref_feat [N_atom, 167], ref_pos [N_atom, 3], atom_to_conformer [N_atom], atom_mask [N_atom]

        Returns:
            a: [N_atom, c_a], ap: [N_atom, N_atom, c_ap]
        """
        a, ap = self.embed(batch)
        mask = pair_mask(batch["atom_mask"])
        if self.local_attention:        # PhysDock2: the atoms attend only within their own conformer (residue / ligand)
            conf = batch["atom_to_conformer"]
            mask = mask * (conf[:, None] == conf[None, :])
        return self.atom_transformer(a, ap, mask), ap

    def embed(self, batch):
        """Atom and atom-pair features before the atom transformer: a [N_atom, c_a], ap [N_atom, N_atom, c_ap]"""
        ref_pos = batch["ref_pos"]
        with torch.no_grad():
            d = ref_pos[:, None, :] - ref_pos[None, :, :]
            conf = batch["atom_to_conformer"]
            # 1 for atom pairs of the same conformer
            v = (conf[:, None] == conf[None, :]).float()[..., None]
        a = self.linear_c(batch["ref_feat"])
        ap = self.linear_c_l(F.relu(a))[:, None] + \
            self.linear_c_m(F.relu(a))[None, :]
        ap = ap + (self.linear_p(d) * v + self.linear_d(1 / (1 + torch.norm(d, dim=-1)[..., None])) * v)
        if hasattr(self, "linear_v"):
            ap = ap + self.linear_v(v) * v
        return a, self.ffn(ap)


class TokenEmbedder(nn.Module):
    """Pool atoms to tokens, add token / pair / MSA features, run Evoformer, template embedder and Pairformer."""

    def __init__(self, target_dim: int, msa_dim: int, c_a: int, c_s: int, c_m: int, c_z: int, inf: float, eps: float,
                 no_blocks_evoformer: int, no_blocks_pairformer: int, no_blocks_template: int, templ_dim: int,
                 key_res_dim: int, r_max: int, s_max: int, ligand_pair_dim: int, use_msa: bool = True, pocket_dim: int = 1,
                 outer_product: bool = False) -> None:
        super().__init__()
        self.use_msa, self.pocket_dim = use_msa, pocket_dim
        self.linear_a = Linear(c_a, c_s)
        self.linear_target_feat = Linear(target_dim, c_s, bias=False)
        self.linear_key_res_feat = Linear(key_res_dim, c_s, bias=False)
        self.linear_pocket_res_feat = Linear(pocket_dim, c_s, bias=False)
        self.linear_s_i = Linear(c_s, c_z)
        self.linear_s_j = Linear(c_s, c_z)
        self.rel_pos_embedder = RelPosEmbedder(c_z, r_max, s_max, ligand_pair_dim)
        self.linear_bonds = Linear(1, c_z, bias=False)
        self.template_pair_embedder = TemplatePairEmbedder(
            templ_dim, c_z, inf, eps, no_blocks_template)
        self.pairformer = Pairformer(c_s, c_z, inf, eps, no_blocks_pairformer, outer_product)
        if use_msa:                     # the released parameters (see MSATemplEncoder)
            self.linear_msa_feat = Linear(msa_dim, c_m, bias=False)
            self.linear_s_input = Linear(c_s, c_m)
            self.evoformer = Evoformer(c_m, c_z, inf, eps, no_blocks_evoformer)
            self.linear_m = Linear(c_m, c_s, bias=False)
            self.linear_s = Linear(c_s, c_s, bias=False)

    def forward(self, batch, a: torch.Tensor):
        """
        Args:
            batch: split_sizes, target_feat, key_res_feat, is_pocket, token_bonds, msa_feat, token_mask, ...
            a: [N_atom, c_a]

        Returns:
            s: [N_token, c_s], z: [N_token, N_token, c_z]
        """
        z_mask = pair_mask(batch["token_mask"])
        s, z, m = self.embed(batch, a)
        if self.use_msa:                # released structure: Evoformer on the MSA first, the template after it
            m, z = self.evoformer(m, z, z_mask)
            z = z + self.template_pair_embedder(batch, z)
            s = self.linear_m(m[0]) + self.linear_s(s)
        else:                           # PhysDock2: no MSA, the template update comes first, then the Pairformer
            z = z + self.template_pair_embedder(batch, z)
        return self.pairformer(s, z, z_mask)

    def embed(self, batch, a: torch.Tensor):
        """Inputs of the Evoformer: s [N_token, c_s], z [N_token, N_token, c_z], m [N_msa, N_token, c_m] (None without the MSA)"""
        s = (segment_mean_pool(F.silu(self.linear_a(a)), batch["split_sizes"])
             + self.linear_target_feat(batch["target_feat"])
             + self.linear_key_res_feat(batch["key_res_feat"])
             + self.linear_pocket_res_feat(batch["is_pocket"][..., None] if self.pocket_dim == 1 else batch["pocket_feat"]))
        z = (self.linear_s_i(s)[:, None] + self.linear_s_j(s)[None, :] + self.rel_pos_embedder(batch)
             + self.linear_bonds(batch["token_bonds"][..., None]))
        m = self.linear_msa_feat(batch["msa_feat"]) + self.linear_s_input(s) if self.use_msa else None
        return s, z, m


class MSATemplEncoder(nn.Module):
    """Encoder of one structure; its output conditions every sample and denoising step of the decoder. (Nature Communications DiffusionConditioning)

    use_msa=True is the structure of the released parameters (MSA features, Evoformer, template after it); False is PhysDock2's: no MSA, the
    template update before the Pairformer."""

    def __init__(self, ref_dim: int, target_dim: int, msa_dim: int, c_a: int, c_ap: int, c_s: int, c_m: int, c_z: int,
                 inf: float, eps: float, no_blocks_atom: int, no_blocks_evoformer: int, no_blocks_pairformer: int,
                 no_blocks_template: int, templ_dim: int, key_res_dim: int, r_max: int, s_max: int,
                 ligand_pair_dim: int, use_msa: bool = True, local_atom_attention: bool = False, pocket_dim: int = 1,
                 outer_product: bool = False, same_conformer_bias: bool = True) -> None:
        super().__init__()
        self.atom_embedder = AtomEmbedder(
            ref_dim, c_a, c_ap, inf, eps, no_blocks_atom, local_atom_attention, same_conformer_bias)
        self.token_embedder = TokenEmbedder(target_dim, msa_dim, c_a, c_s, c_m, c_z, inf, eps, no_blocks_evoformer,
                                            no_blocks_pairformer, no_blocks_template, templ_dim, key_res_dim,
                                            r_max, s_max, ligand_pair_dim, use_msa, pocket_dim, outer_product)
        self.norm_s = RMSNorm(c_s, eps=eps)
        self.linear_s = Linear(c_s, c_a, bias=False)
        self.norm_z = RMSNorm(c_z, eps=eps)
        self.linear_z = Linear(c_z, c_ap, bias=False)

    def forward(self, batch) -> Conditions:
        """batch: unbatched feature dict (see PhysDock.data.FEATURES) -> Conditions(a, ap, s, z)"""
        a, ap = self.atom_embedder(batch)
        s, z = self.token_embedder(batch, a)
        return self.broadcast(batch, a, ap, s, z)

    def broadcast(self, batch, a, ap, s, z) -> Conditions:
        """Add the token (pair) representation to the atom (pair) representation."""
        idx = batch["atom_to_token"]
        a = a + self.linear_s(self.norm_s(s))[idx]                  # token -> atom
        ap = ap + self.linear_z(self.norm_z(z))[idx][:, idx]        # token pair -> atom pair
        return Conditions(a, ap, s, z)
