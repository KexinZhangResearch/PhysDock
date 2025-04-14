import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .transformers import AtomTransformer, Evoformer, Pairformer, Triangleformer
from PhysDock.models.primitives import Linear, FeedForward, RMSNorm
from PhysDock.utils.tensor_utils import one_hot
from PhysDock.data import TensorDict


class TemplatePairEmbedder(nn.Module):
    def __init__(
            self,
            templ_dim: int,
            c_z: int,
            inf: float,
            eps: float,
            no_blocks: int,
    ):
        super().__init__()

        self.no_blocks = no_blocks
        self.norm_in = RMSNorm(c_z)
        self.linear_in = Linear(c_z, c_z, bias=False)

        self.linear_templ_feat = Linear(templ_dim, c_z, bias=False)

        self.triangleformer = Triangleformer(c_z, inf, eps, no_blocks)
        self.norm_out = RMSNorm(c_z, eps)
        self.linear_out = Linear(c_z, c_z, bias=False)

    def forward(
            self,
            batch: TensorDict,
            z: torch.Tensor
    ) -> torch.Tensor:
        templ_feat = batch["templ_feat"]
        asym_id = batch["asym_id"]
        t_mask = batch["t_mask"]
        chain_same = (asym_id[None] == asym_id[:, None]).type_as(templ_feat)
        z_mask = batch["z_mask"] * templ_feat[..., 39] * chain_same

        z = self.linear_in(self.norm_in(z)) + self.linear_templ_feat(templ_feat)

        z = self.triangleformer(z, z_mask)

        z = self.linear_out(F.relu(self.norm_out(z))).float() * t_mask

        return z.float()


class RelPosEmbedder(nn.Module):
    def __init__(self, c_z: int):
        super().__init__()
        self.r_max = 32
        self.s_max = 2

        self.c_z = c_z

        self.c_rel_feat = 4 * self.r_max + 2 * self.s_max + 7 - 2 * self.r_max - 2 + 42

        self.linear = Linear(self.c_rel_feat, c_z, bias=False)

    def forward(self, batch: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            asym_id = batch["asym_id"]
            sym_id = batch["sym_id"]
            entity_id = batch["entity_id"]
            residue_index = batch["residue_index"]
            rel_tok_feat = batch["rel_tok_feat"]

            chain_same = (asym_id[..., None] == asym_id[..., None, :])
            entity_same = (entity_id[..., None] == entity_id[..., None, :])
            residue_offset = residue_index[..., None] - residue_index[..., None, :] + self.r_max
            clipped_residue_offset = torch.clamp(residue_offset, 0, 2 * self.r_max)
            d_res = torch.where(chain_same, clipped_residue_offset, 2 * self.r_max + 1)
            boundaries = torch.arange(0, 2 * self.r_max + 2, device=d_res.device)
            rel_pos_feat = one_hot(d_res, boundaries)

            boundaries = torch.arange(0, 2 * self.s_max + 2, device=d_res.device)
            chain_offset = sym_id[..., None] - sym_id[..., None, :] + self.s_max
            clipped_chain_offset = torch.clamp(chain_offset, 0, 2 * self.s_max)
            d_chain = torch.where(torch.logical_or(
                chain_same, torch.logical_not(entity_same)), 2 * self.s_max + 1, clipped_chain_offset)
            rel_chain_feat = one_hot(d_chain, boundaries)
            rel_feat = torch.cat([
                rel_pos_feat,
                rel_tok_feat,
                entity_same[..., None],
                rel_chain_feat,
            ], dim=-1)

        return self.linear(rel_feat)


class AtomEmbedder(nn.Module):
    def __init__(self, ref_dim: int, c_a: int, c_ap: int, inf: float, eps: float, no_blocks_atom: int):
        super().__init__()
        self.linear_c = Linear(ref_dim, c_a, bias=False)
        self.linear_p = Linear(3, c_ap, bias=False)
        self.linear_d = Linear(1, c_ap, bias=False)
        self.linear_v = Linear(1, c_ap, bias=False)

        self.linear_c_l = Linear(c_a, c_ap, bias=False)
        self.linear_c_m = Linear(c_a, c_ap, bias=False)
        self.ffn = FeedForward(c_ap)
        self.atom_transformer = AtomTransformer(c_a, c_ap, inf, eps, no_blocks_atom)

    def forward(self, batch: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        ref_feat = batch["ref_feat"]
        ref_pos = batch["ref_pos"]
        ref_space_uid = batch["ref_space_uid"]
        ap_mask = batch["ap_mask"]

        with torch.no_grad():
            d = (ref_pos[:, None, :] - ref_pos[None, :, :]).float()
            v = torch.eq(ref_space_uid[:, None], ref_space_uid[None, :]).float()

        a = self.linear_c(ref_feat)
        p = self.linear_p(d) * v[:, :, None]
        p = p + self.linear_d(1 / (1 + torch.norm(d, dim=-1)[:, :, None])) * v[:, :, None]
        p = p + self.linear_v(v[:, :, None]) * v[:, :, None]
        ap = self.linear_c_l(F.relu(a))[:, None, :] + self.linear_c_m(F.relu(a))[None, :, :]
        ap = ap + p
        ap = ap + self.ffn(ap)
        a = self.atom_transformer(a, ap, ap_mask)
        return a, ap


class TokenEmbedder(nn.Module):
    def __init__(
            self,
            target_dim: int,
            msa_dim: int,
            c_a: int,
            c_s: int,
            c_m: int,
            c_z: int,
            inf: float,
            eps: float,
            no_blocks_evoformer: int,
            no_blocks_pairformer: int,
    ):
        super().__init__()
        # No Recycle
        self.linear_a = Linear(c_a, c_s)

        self.linear_target_feat = Linear(target_dim, c_s, bias=False)  # 65
        self.linear_key_res_feat = Linear(7, c_s, bias=False)
        self.linear_pocket_res_feat = Linear(1, c_s, bias=False)

        self.linear_s_i = Linear(c_s, c_z)
        self.linear_s_j = Linear(c_s, c_z)
        self.rel_pos_embedder = RelPosEmbedder(c_z)
        self.linear_bonds = Linear(1, c_z, bias=False)

        self.linear_msa_feat = Linear(msa_dim, c_m, bias=False)
        self.linear_s_input = Linear(c_s, c_m)

        self.template_pair_embedder = TemplatePairEmbedder(40, c_z, inf, eps, no_blocks=2)
        self.evoformer = Evoformer(c_m, c_z, inf, eps, no_blocks_evoformer)
        self.pairformer = Pairformer(c_s, c_z, inf, eps, no_blocks_pairformer)

        self.linear_m = Linear(c_m, c_s, bias=False)
        self.linear_s = Linear(c_s, c_s, bias=False)

    def downscale(self, batch: TensorDict, a: torch.Tensor) -> torch.Tensor:
        token_id_to_chunk_sizes = batch["token_id_to_chunk_sizes"]
        a_cumsum = torch.cumsum(F.silu(self.linear_a(a)), dim=-2)
        inds = torch.cumsum(token_id_to_chunk_sizes, dim=-1) - 1

        value = a_cumsum[inds, :]
        s = torch.cat([value[0:1, :], torch.diff(value, dim=-2)], dim=-2)
        s = s / (token_id_to_chunk_sizes[:, None] + 1e-3)
        return s

    def forward(self, batch: TensorDict, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        target_feat = batch["target_feat"]
        key_res_feat = batch["key_res_feat"]
        pocket_res_feat = batch["pocket_res_feat"]
        token_bonds_feature = batch["token_bonds_feature"]
        msa_feat = batch["msa_feat"]
        z_mask = batch["z_mask"]

        s = self.downscale(batch, a) + self.linear_target_feat(target_feat) + self.linear_key_res_feat(
            key_res_feat) + self.linear_pocket_res_feat(pocket_res_feat[..., None])

        z = self.linear_s_i(s)[:, None, :] + self.linear_s_j(s)[None, :, :] + self.rel_pos_embedder(
            batch) + self.linear_bonds(token_bonds_feature[..., None])

        m = self.linear_msa_feat(msa_feat) + self.linear_s_input(s)

        m, z = self.evoformer(m, z, z_mask)

        z = z + self.template_pair_embedder(batch, z)

        s = self.linear_m(m[0]) + self.linear_s(s)

        s, z = self.pairformer(s, z, z_mask)

        return s, z


class DiffusionConditioning(nn.Module):
    def __init__(
            self,
            ref_dim: int,
            target_dim: int,
            msa_dim: int,
            c_a: int,
            c_ap: int,
            c_s: int,
            c_m: int,
            c_z: int,
            inf: float,
            eps: float,
            no_blocks_atom: int,
            no_blocks_evoformer: int,
            no_blocks_pairformer: int,
    ):
        super().__init__()

        self.atom_embedder = AtomEmbedder(ref_dim, c_a, c_ap, inf, eps, no_blocks_atom)
        self.token_embedder = TokenEmbedder(target_dim, msa_dim, c_a, c_s, c_m, c_z, inf, eps, no_blocks_evoformer,
                                            no_blocks_pairformer)
        self.norm_s = RMSNorm(c_s, eps=eps)
        self.linear_s = Linear(c_s, c_a, bias=False)
        self.norm_z = RMSNorm(c_z, eps=eps)
        self.linear_z = Linear(c_z, c_ap, bias=False)

    def forward(self, batch:TensorDict)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        atom_id_to_token_id = batch["atom_id_to_token_id"]
        a, ap = self.atom_embedder(batch)
        s, z = self.token_embedder(batch, a)
        a = a + self.linear_s(self.norm_s(s))[atom_id_to_token_id]
        ap = ap + self.linear_z(self.norm_z(z))[atom_id_to_token_id][:, atom_id_to_token_id]
        return a, ap, s, z
