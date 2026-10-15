"""Transformer blocks and their stacks (Stack: atom, triangle, Evoformer, Pairformer, DiT)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from PhysDock.models.primitives import (AttentionWithPairBias, DiTAttention, DiTTransition, MSAColumnAttention,
                                        MSARowAttentionWithPairBias, OuterProductMean, Transition, TriangleAttention,
                                        TriangleUpdate)
from PhysDock.utils.tensor_utils import gen_attn_mask


def checkpointed(block, *args):
    """Activation-checkpointed block(*args). The blocks use no randomness, so the RNG state is not saved."""
    return checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False)


class Stack(nn.Module):
    """no_blocks copies of a block, applied in sequence (activation checkpointing while grads are on).

    The first n_state arguments are the state that every block updates and returns (one tensor or a tuple of
    n_state tensors); the remaining arguments (pair features, masks, timestep, ...) go unchanged to every block.

    Args:
        block: block class
        no_blocks: number of blocks
        *args, **kwargs: block constructor arguments
        n_state: number of state tensors
    """

    def __init__(self, block, no_blocks: int, *args, n_state: int = 1, **kwargs) -> None:
        super().__init__()
        self.n_state = n_state
        self.blocks = nn.ModuleList([block(*args, **kwargs) for _ in range(no_blocks)])

    def forward(self, *args):
        state, rest = args[:self.n_state], args[self.n_state:]
        for block in self.blocks:
            out = self.run_block(block, *state, *rest)
            state = out if self.n_state > 1 else (out,)
        return state if self.n_state > 1 else state[0]

    checkpointed = staticmethod(checkpointed)       # replaced by a compiled version in compile_blocks

    def run_block(self, block, *args):
        """block(*args), activation-checkpointed while grads are on"""
        if torch.is_grad_enabled():
            return self.checkpointed(block, *args)
        return block(*args)


def compile_blocks(model: nn.Module) -> int:
    """torch.compile the checkpointed training call of every block of every Stack in model (regional compilation).

    Every block call is its own small graph, so its backward recomputes one block at a time like eager checkpointing;
    compiling the whole model instead lets the backward hold the recomputed activations of several blocks at once
    (medium, bf16, crop 256: 27 GiB instead of 19 GiB). Parameters are graph inputs, so the blocks of one class share a
    compiled graph. Calls without gradients (validation, sampling) stay eager.

    Returns:
        number of compiled stacks
    """
    torch._dynamo.config.inline_inbuilt_nn_modules = True
    compiled = torch.compile(checkpointed)
    stacks = [m for m in model.modules() if isinstance(m, Stack)]
    for stack in stacks:
        stack.checkpointed = compiled
    return len(stacks)


class AtomBlock(nn.Module):
    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = AttentionWithPairBias(c_a, c_ap, inf, eps)
        self.transition = Transition(c_a, eps)

    def forward(self, a, ap, ap_mask):
        a = a + self.attention(a, ap, ap_mask)
        return a + self.transition(a)


class AtomTransformer(Stack):
    """Atom transformer: pair-biased attention + transition.

    a: [N_atom, c_a], ap: [N_atom, N_atom, c_ap], ap_mask: [N_atom, N_atom] -> a: [N_atom, c_a]
    """

    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(AtomBlock, no_blocks, c_a, c_ap, inf, eps)


class TriangleBlock(nn.Module):
    def __init__(self, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)

    def forward(self, z, z_mask):
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        return z + self.pair_transition(z)


class Triangleformer(Stack):
    """Pair-only stack (template embedder).

    z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> z
    """

    def __init__(self, c_z: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(TriangleBlock, no_blocks, c_z, inf, eps)


class EvoformerBlock(nn.Module):
    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
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

    def forward(self, m, z, z_mask):
        m = m + self.msa_row_attention(m, z, z_mask)
        m = m + self.msa_col_attention(m)
        m = m + self.msa_transition(m)
        z = z + self.opm(m)
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        return m, z + self.pair_transition(z)


class Evoformer(Stack):
    """MSA stack.

    m: [N_msa, N_token, c_m], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (m, z)
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float, no_blocks: int = 4) -> None:
        super().__init__(EvoformerBlock, no_blocks, c_m, c_z, inf, eps, n_state=2)


class PairFormerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.triangle_row_update = TriangleUpdate(c_z, eps)
        self.triangle_col_update = TriangleUpdate(c_z, eps, transpose=True)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, transpose=True)
        self.pair_transition = Transition(c_z, eps)
        self.attention = AttentionWithPairBias(c_s, c_z, inf, eps)
        self.transition = Transition(c_s, eps)

    def forward(self, s, z, z_mask):
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        z = z + self.pair_transition(z)
        s = s + self.attention(s, z, z_mask)
        return s + self.transition(s), z


class Pairformer(Stack):
    """Pair + single stack.

    s: [N_token, c_s], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (s, z)
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 24) -> None:
        super().__init__(PairFormerBlock, no_blocks, c_s, c_z, inf, eps, n_state=2)


class DiTBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = DiTAttention(c_s, c_z, inf, eps)
        self.transition = DiTTransition(c_s, eps)

    def forward(self, x, bias, t):
        x = x + self.attention(x, bias, t)
        return x + self.transition(x, t)


class DiT(Stack):
    """Diffusion transformer conditioned on a timestep embedding and per-block pair biases.

    x: [B, N, c_s], bias: [no_blocks, H, N, N] (pair_bias(z, z_mask)), t: [B, 256] -> x: [B, N, c_s]
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 12) -> None:
        super().__init__(DiTBlock, no_blocks, c_s, c_z, inf, eps)

    def forward(self, x, bias, t):
        for block, b in zip(self.blocks, bias):
            x = self.run_block(block, x, b, t)
        return x

    def pair_bias(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """Attention logit biases of all blocks, linear_z(norm_z(z)) + mask; they depend on the conditioning only,
        so the decoder computes them once per structure instead of in every block of every denoising step.

        Every block normalises the same z, so its LayerNorm statistics are computed once and each block's LayerNorm
        affine is folded into its projection: linear_z(norm_z(z)) = z_hat @ (W * gamma)^T + W @ beta, one matmul for
        all blocks (the same function and gradients as the blockwise form, up to float rounding).

        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [no_blocks, H, N, N] contiguous (the layout the fused attention kernels need)
        """
        attn = [block.attention for block in self.blocks]
        eps = attn[0].norm_z.eps
        if any(a.norm_z.eps != eps for a in attn):
            raise ValueError("pair_bias needs the same LayerNorm eps in all blocks")
        z_hat = F.layer_norm(z, z.shape[-1:], eps=eps)
        weight = torch.cat([a.linear_z.weight * a.norm_z.weight for a in attn])             # [no_blocks * H, c_z]
        bias = torch.cat([a.linear_z.weight @ a.norm_z.bias for a in attn])
        logits = F.linear(z_hat, weight, bias).permute(2, 0, 1) + gen_attn_mask(z_mask, -attn[0].inf)
        return logits.reshape(len(attn), -1, *z_mask.shape).contiguous()
