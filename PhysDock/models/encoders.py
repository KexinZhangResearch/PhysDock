import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, NamedTuple, Tuple

from PhysDock.models.primitives import (AtomTransformer, Evoformer, FeedForward, Linear, Pairformer, RMSNorm,
                                        Triangleformer)
from PhysDock.utils.tensor_utils import one_hot, segment_mean_pool, segment_unpool_1d, segment_unpool_2d
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
        """Average-pool atom features into tokens (the atoms of a token are contiguous).

        Args:
            batch: Feature dict; uses ``token_id_to_chunk_sizes``, the number of atoms of every token.
            a: Atom representation, shape ``[N_atom, c_a]``.

        Returns:
            Token representation, shape ``[N_token, c_s]``.
        """
        return segment_mean_pool(F.silu(self.linear_a(a)), batch["token_id_to_chunk_sizes"])

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


class Conditions(NamedTuple):
    """Conditioning produced by the encoder for the decoder (it can be unpacked like a plain tuple).

    Attributes:
        a: Atom conditioning, shape ``[N_atom, c_a]``.
        ap: Atom-pair conditioning, shape ``[N_atom, N_atom, c_ap]``.
        s: Single (token) representation, shape ``[N_token, c_s]``.
        z: Pair (token-pair) representation, shape ``[N_token, N_token, c_z]``.
    """

    a: torch.Tensor
    ap: torch.Tensor
    s: torch.Tensor
    z: torch.Tensor


class MSATemplEncoder(nn.Module):
    """Encoder of the MSA, templates and atom/token features: the *DiffusionConditioning* of the paper.

    In the AlphaFold 3 paper, DiffusionConditioning (Algorithm 21) turns the trunk outputs into the conditioning
    of the diffusion module. In PhysDock the trunk is part of this module, hence its name: it embeds the
    reference conformers of every atom (``AtomEmbedder``), embeds the token features, relative positions and bonds
    and runs the Evoformer on the MSA and the Pairformer on the pair representation, with a template embedding
    (``TokenEmbedder``). The resulting single ``s`` and pair ``z`` representations are returned together with
    atom-level versions ``a`` and ``ap`` (the atom embeddings plus ``s`` / ``z`` broadcast to the atoms), which is
    everything :class:`~PhysDock.models.decoders.AllAtomDecoder` needs. It does not depend on the noisy
    coordinates, so it is evaluated once per structure and reused for all samples and denoising steps.

    Args:
        ref_dim: Size of the per-atom reference features.
        target_dim: Size of the per-token target features.
        msa_dim: Size of the per-MSA-row features.
        c_a: Atom channel size.
        c_ap: Atom-pair channel size.
        c_s: Single (token) channel size.
        c_m: MSA channel size.
        c_z: Pair channel size.
        inf: Magnitude of the value added to masked logits.
        eps: Epsilon of the norms.
        no_blocks_atom: Blocks of the atom transformer.
        no_blocks_evoformer: Evoformer blocks.
        no_blocks_pairformer: Pairformer blocks.
        unpool_method: How token features are broadcast to atoms: ``"index"`` (gather with ``atom_id_to_token_id``,
            default) or ``"segment"`` (from ``token_id_to_chunk_sizes``), see
            :func:`~PhysDock.utils.tensor_utils.segment_unpool_1d`.
    """

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
            unpool_method: Literal["index", "segment"] = "index",
    ) -> None:
        super().__init__()
        self.unpool_method = unpool_method

        self.atom_embedder = AtomEmbedder(ref_dim, c_a, c_ap, inf, eps, no_blocks_atom)
        self.token_embedder = TokenEmbedder(target_dim, msa_dim, c_a, c_s, c_m, c_z, inf, eps, no_blocks_evoformer,
                                            no_blocks_pairformer)
        self.norm_s = RMSNorm(c_s, eps=eps)
        self.linear_s = Linear(c_s, c_a, bias=False)
        self.norm_z = RMSNorm(c_z, eps=eps)
        self.linear_z = Linear(c_z, c_ap, bias=False)

    def forward(self, batch: TensorDict) -> Conditions:
        """Encode one structure.

        Args:
            batch: Feature dict of a single (unbatched) structure.

        Returns:
            The conditioning ``Conditions(a, ap, s, z)``.
        """
        atom_id_to_token_id = batch["atom_id_to_token_id"]
        sizes = batch["token_id_to_chunk_sizes"]
        a, ap = self.atom_embedder(batch)
        s, z = self.token_embedder(batch, a)
        n_atom = a.shape[0]
        a = a + segment_unpool_1d(self.linear_s(self.norm_s(s)), sizes, num_rows=n_atom, dim=0,
                                  index=atom_id_to_token_id, method=self.unpool_method)
        ap = ap + segment_unpool_2d(self.linear_z(self.norm_z(z)), sizes, num_rows=n_atom,
                                    row_index=atom_id_to_token_id, method=self.unpool_method)
        return Conditions(a, ap, s, z)
