import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional

from PhysDock.data import TensorDict
from PhysDock.models.primitives import Transition, DiTTransition, AttentionWithPairBias, MSARowAttentionWithPairBias, \
    MSAColumnAttention, TriangleUpdate, TriangleAttention, OuterProductMean, DiTAttention, Linear, LayerNorm, \
    TimestepEmbeddings


class AtomBlock(nn.Module):
    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float):
        super().__init__()
        self.attention = AttentionWithPairBias(c_a, c_ap, inf, eps)
        self.transition = Transition(c_a, eps)

    def forward(self, a: torch.Tensor, ap: torch.Tensor, ap_mask: torch.Tensor) -> torch.Tensor:
        a = a + self.attention(a, ap, ap_mask)
        a = a + self.transition(a)
        return a


class AtomTransformer(nn.Module):
    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float, no_blocks: int):
        super().__init__()
        self.no_blocks = no_blocks
        self.blocks = nn.ModuleList([
            AtomBlock(c_a, c_ap, inf, eps) for _ in range(self.no_blocks)
        ])

    def forward(self, a: torch.Tensor, ap: torch.Tensor, ap_mask: torch.Tensor) -> torch.Tensor:
        for block_id, block in enumerate(self.blocks):
            a = checkpoint(block, a, ap, ap_mask, use_reentrant=False)
        return a


class TriangleBlock(nn.Module):
    def __init__(self, c_z: int, inf: float, eps: float):
        super().__init__()
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        z = z + self.pair_transition(z)
        return z


class Triangleformer(nn.Module):
    def __init__(self, c_z: int, inf: float, eps: float, no_blocks: int):
        super().__init__()

        self.no_blocks = no_blocks
        self.blocks = nn.ModuleList([
            TriangleBlock(c_z, inf, eps) for _ in range(self.no_blocks)
        ])

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        for block_id, block in enumerate(self.blocks):
            z = checkpoint(block, z, z_mask, use_reentrant=False)
        return z


class EvoformerBlock(nn.Module):
    def __init__(self, c_m: int, c_z: int, inf: float, eps: float):
        super().__init__()
        self.msa_row_attention = MSARowAttentionWithPairBias(c_m, c_z, inf, eps)
        self.msa_col_attention = MSAColumnAttention(c_m, inf, eps)
        self.msa_transition = Transition(c_m, eps)
        self.opm = OuterProductMean(c_m, c_z, eps)
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = m + self.msa_row_attention(m, z, z_mask)
        m = m + self.msa_col_attention(m)
        m = m + self.msa_transition(m)
        z = z + self.opm(m)
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        z = z + self.pair_transition(z)
        return m, z


class Evoformer(nn.Module):
    def __init__(self, c_m: int, c_z: int, inf: float, eps: float, no_blocks: int = 4):
        super().__init__()
        self.no_blocks = no_blocks
        self.blocks = nn.ModuleList([
            EvoformerBlock(c_m, c_z, inf, eps) for _ in range(self.no_blocks)
        ])

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block_id, block in enumerate(self.blocks):
            m, z = checkpoint(block, m, z, z_mask, use_reentrant=False)
        return m, z


class PairFormerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float):
        super().__init__()

        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)
        self.attention = AttentionWithPairBias(c_s, c_z, inf, eps)
        self.transition = Transition(c_s, eps)

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        z = z + self.pair_transition(z)
        s = s + self.attention(s, z, z_mask)
        s = s + self.transition(s)
        return s, z


class Pairformer(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 24):
        super().__init__()
        self.no_blocks = no_blocks
        self.blocks = nn.ModuleList([
            PairFormerBlock(c_s, c_z, inf, eps) for _ in range(self.no_blocks)
        ])

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for block_id, block in enumerate(self.blocks):
            s, z = checkpoint(block, s, z, z_mask, use_reentrant=False)
        return s, z


class DiTBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float):
        super().__init__()
        self.attention = DiTAttention(c_s, c_z, inf, eps)
        self.transition = DiTTransition(c_s, eps)

    def forward(self, bs: torch.Tensor, z: torch.Tensor, t: torch.Tensor, z_mask: torch.Tensor,
                beta: Optional[torch.Tensor]) -> torch.Tensor:
        bs = bs + self.attention(bs, z, t, z_mask, beta)
        bs = bs + self.transition(bs, t)
        return bs


class DiT(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 12):
        super().__init__()

        self.no_blocks = no_blocks
        self.blocks = nn.ModuleList([
            DiTBlock(c_s, c_z, inf, eps) for _ in range(self.no_blocks)
        ])

    def forward(self, bs: torch.Tensor, z: torch.Tensor, t: torch.Tensor, z_mask: torch.Tensor,
                beta: Optional[torch.Tensor]) -> torch.Tensor:
        for block_id, block in enumerate(self.blocks):
            bs = checkpoint(block, bs, z, t, z_mask, beta, use_reentrant=False)
        return bs


class AF3DiT(nn.Module):
    def __init__(
            self,
            c_a: int,
            c_ap: int,
            c_s: int,
            c_z: int,
            inf: float,
            eps: float,
            no_blocks_atom: int,
            no_blocks_dit: int,
            sigma_data: int
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.linear_x = Linear(3, c_a)
        self.linear_downscale = Linear(c_a, c_s)
        self.linear_upscale = Linear(c_s, c_a)
        self.time_embedder = TimestepEmbeddings()

        self.atom_dit_encoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)
        self.token_dit = DiT(c_s, c_z, inf, eps, no_blocks_dit)
        self.atom_dit_decoder = DiT(c_a, c_ap, inf, eps, no_blocks_atom)

        self.norm_r = LayerNorm(c_a, eps)
        self.linear_r = Linear(c_a, 3, bias=False)

    def downscale(self, ba: torch.Tensor, s: torch.Tensor, token_id_to_chunk_sizes: torch.Tensor) -> torch.Tensor:
        ba_cumsum = torch.cumsum(F.silu(self.linear_downscale(ba)), dim=-2)
        inds = torch.cumsum(token_id_to_chunk_sizes, dim=-1) - 1
        value = ba_cumsum[:, inds, :]
        x = torch.cat([value[:, 0:1, :], torch.diff(value, dim=-2)], dim=-2)
        x = x / (token_id_to_chunk_sizes[None, :, None] + 1e-3)
        bs = x + s[None]
        return bs

    def upscale(self, ba: torch.Tensor, bs: torch.Tensor, atom_id_to_token_id: torch.Tensor) -> torch.Tensor:
        ba = ba + self.linear_upscale(bs)[:, atom_id_to_token_id].float()
        return ba

    def precond(self, x_hat: torch.Tensor, t_hat: torch.Tensor, a: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        c_in = 1 / (torch.sqrt(t_hat[:, None, None] ** 2 + self.sigma_data ** 2))
        c_noise = torch.log(t_hat / self.sigma_data) / 4.0

        ba = self.linear_x(x_hat * c_in) + a[None]
        t = self.time_embedder(t_hat * c_noise)
        beta = None
        return ba, t, beta

    def denoise(self, x_hat: torch.Tensor, t_hat: torch.Tensor, ba: torch.Tensor) -> torch.Tensor:
        c_skip = (self.sigma_data ** 2 / (self.sigma_data ** 2 + t_hat ** 2))[:, None, None]
        c_out = (self.sigma_data * t_hat / torch.sqrt(self.sigma_data ** 2 + t_hat ** 2))[:, None, None]

        x_denoised = c_skip * x_hat + c_out * self.linear_r(self.norm_r(ba))
        return x_denoised

    def forward(
            self,
            batch: TensorDict,
            x_hat: torch.Tensor,
            t_hat: torch.Tensor,
            a: torch.Tensor,
            ap: torch.Tensor,
            s: torch.Tensor,
            z: torch.Tensor
    ) -> torch.Tensor:
        ap_mask = batch["ap_mask"]
        z_mask = batch["z_mask"]
        token_id_to_chunk_sizes = batch["token_id_to_chunk_sizes"]
        atom_id_to_token_id = batch["atom_id_to_token_id"]

        ba, t, beta = self.precond(x_hat, t_hat, a)

        ba = self.atom_dit_encoder(ba, ap, t, ap_mask, beta)
        bs = self.downscale(ba, s, token_id_to_chunk_sizes)

        bs = self.token_dit(bs, z, t, z_mask, None)

        ba = self.upscale(ba, bs, atom_id_to_token_id, )

        ba = self.atom_dit_decoder(ba, ap, t, ap_mask, beta)

        x_denoised = self.denoise(x_hat, t_hat, ba)
        return x_denoised
