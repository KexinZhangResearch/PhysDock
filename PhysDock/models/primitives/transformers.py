"""Transformer blocks and their stacks (Stack: atom, triangle, Evoformer, Pairformer, DiT)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from PhysDock.models.primitives import (AdaLNSwiGLUFFN, AttentionWithBias, DiTAttention, MSAColumnAttention,
                                        MSARowAttentionWithBias, OuterProductMean, RMSNormSwiGLUFFN, TriangleAttention,
                                        TriangleMultiplication)
from PhysDock.models.kernels.shared_bias_attention import aligned_bias
from PhysDock.utils.tensor_utils import gen_attn_mask


def checkpointed(block, *args):
    """Activation-checkpointed block(*args). The blocks use no randomness, so the RNG state is not saved."""
    return checkpoint(block, *args, use_reentrant=False, preserve_rng_state=False)


def call(block, *args):
    """block(*args) (without checkpointing)"""
    return block(*args)


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
        for i, block in enumerate(self.blocks):
            out = self.run_block(block, *state, *rest, index=i)
            state = out if self.n_state > 1 else (out,)
        return state if self.n_state > 1 else state[0]

    checkpoint = True                               # activation checkpointing while grads are on (see set_checkpointing)
    checkpointed = staticmethod(checkpointed)       # both compiled by PhysDock.training.compile_training
    call = staticmethod(call)

    def checkpoints(self, index: int) -> bool:
        """whether block `index` is activation-checkpointed: self.checkpoint is True / False, or a fraction in (0, 1) of the
        blocks, spread evenly (round(fraction * no_blocks) blocks)"""
        c = self.checkpoint
        if isinstance(c, float):
            n = len(self.blocks)
            k = round(c * n)
            return (index + 1) * k // n > index * k // n
        return bool(c)

    def run_block(self, block, *args, index: int = None):
        """block(*args), activation-checkpointed while grads are on (if the block is checkpointed, see checkpoints)"""
        if torch.is_grad_enabled():
            ck = self.checkpoint if index is None else self.checkpoints(index)
            return (self.checkpointed if ck else self.call)(block, *args)
        return block(*args)


def set_checkpointing(model: nn.Module, modes: dict) -> None:
    """Activation checkpointing per Stack class, e.g. {"Pairformer": "full", "DiT": "attention", "Evoformer": 0.5}:
    "full" (recompute every block in the backward), "none" (keep all activations: faster, more memory), a fraction between 0 and
    1 (only that fraction of the blocks is recomputed, spread evenly: the time and the memory interpolate between none and
    full) or, for DiT, "attention" (keep the attention inputs / outputs, recompute the rest)."""
    stacks = {cls.__name__: cls for cls in (AtomTransformer, Triangleformer, Evoformer, Pairformer, DiT)}
    for name, mode in modes.items():
        if name not in stacks:
            raise ValueError(f"unknown stack {name!r}, expected one of {sorted(stacks)}")
        is_fraction = isinstance(mode, (int, float)) and not isinstance(mode, bool) and 0 <= mode <= 1
        if not is_fraction and mode not in ("full", "none") + (("attention",) if name == "DiT" else ()):
            raise ValueError(f"checkpointing of {name} must be full, none, a fraction in [0, 1]{' or attention' * (name == 'DiT')}, got {mode!r}")
    for module in model.modules():
        mode = modes.get(type(module).__name__) if isinstance(module, Stack) else None
        if mode is not None:
            module.checkpoint = {"full": True, "none": False}.get(mode, float(mode) if not isinstance(mode, str) else mode)
            if module.checkpoint in (0.0, 1.0) and not isinstance(module.checkpoint, bool):
                module.checkpoint = bool(module.checkpoint)



class AtomBlock(nn.Module):
    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = AttentionWithBias(c_a, c_ap, inf, eps)
        self.transition = RMSNormSwiGLUFFN(c_a, eps)

    def forward(self, a, ap, ap_mask):
        a = a + self.attention(a, ap, ap_mask)
        return self.transition(a)


class AtomTransformer(Stack):
    """Atom transformer: pair-biased attention + transition.

    a: [N_atom, c_a], ap: [N_atom, N_atom, c_ap], ap_mask: [N_atom, N_atom] -> a: [N_atom, c_a]
    """

    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(AtomBlock, no_blocks, c_a, c_ap, inf, eps)


class TriangleBlock(nn.Module):
    def __init__(self, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.triangle_row_update = TriangleMultiplication(c_z, eps, outgoing=True)
        self.triangle_col_update = TriangleMultiplication(c_z, eps, outgoing=False)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps, starting_node=True)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, starting_node=False)
        self.pair_transition = RMSNormSwiGLUFFN(c_z, eps)

    def forward(self, z, z_mask):
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        return self.pair_transition(z)


class Triangleformer(Stack):
    """Pair-only stack (template embedder).

    z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> z
    """

    def __init__(self, c_z: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(TriangleBlock, no_blocks, c_z, inf, eps)


class EvoformerBlock(nn.Module):
    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.msa_row_attention = MSARowAttentionWithBias(c_m, c_z, inf, eps)
        self.msa_col_attention = MSAColumnAttention(c_m, eps)
        self.msa_transition = RMSNormSwiGLUFFN(c_m, eps)
        self.opm = OuterProductMean(c_m, c_z, eps)
        self.triangle_row_update = TriangleMultiplication(c_z, eps, outgoing=True)
        self.triangle_col_update = TriangleMultiplication(c_z, eps, outgoing=False)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps, starting_node=True)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, starting_node=False)
        self.pair_transition = RMSNormSwiGLUFFN(c_z, eps)

    def forward(self, m, z, z_mask):
        m = m + self.msa_row_attention(m, z, z_mask)
        m = m + self.msa_col_attention(m)
        m = self.msa_transition(m)
        z = self.opm(m, z)
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        return m, self.pair_transition(z)


class Evoformer(Stack):
    """MSA stack.

    m: [N_msa, N_token, c_m], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (m, z)
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float, no_blocks: int = 4) -> None:
        super().__init__(EvoformerBlock, no_blocks, c_m, c_z, inf, eps, n_state=2)


class PairFormerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.triangle_row_update = TriangleMultiplication(c_z, eps, outgoing=True)
        self.triangle_col_update = TriangleMultiplication(c_z, eps, outgoing=False)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps, starting_node=True)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, starting_node=False)
        self.pair_transition = RMSNormSwiGLUFFN(c_z, eps)
        self.attention = AttentionWithBias(c_s, c_z, inf, eps)
        self.transition = RMSNormSwiGLUFFN(c_s, eps)

    def forward(self, s, z, z_mask):
        z = z + self.triangle_row_update(z, z_mask)
        z = z + self.triangle_col_update(z, z_mask)
        z = z + self.triangle_row_attention(z, z_mask)
        z = z + self.triangle_col_attention(z, z_mask)
        z = self.pair_transition(z)
        s = s + self.attention(s, z, z_mask)
        return self.transition(s), z


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
        self.transition = AdaLNSwiGLUFFN(c_s, eps)

    def forward(self, x, bias, t):
        x = x + self.attention(x, bias, t)
        return self.transition(x, t)


def _dit_attention_inputs(block: DiTBlock, x, t):
    return block.attention.project(x, t)


def _dit_block_output(block: DiTBlock, x, o, gate, t):
    x = x + block.attention.output(o, gate)
    return block.transition(x, t)


class DiT(Stack):
    """Diffusion transformer conditioned on a timestep embedding and per-block pair biases.

    x: [B, N, c_s], bias: [no_blocks, H, N, N] (pair_bias(z, z_mask)), t: [B, 256] -> x: [B, N, c_s]

    checkpoint "attention" (selective checkpointing, DiTBlock.forward split in three): the attention inputs / outputs
    are kept and only the projections, norms and the transition around them are recomputed in the backward, so the
    expensive attention (fp32, B noisy samples) runs once.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 12) -> None:
        super().__init__(DiTBlock, no_blocks, c_s, c_z, inf, eps)

    def forward(self, x, bias, t):
        for i, (block, b) in enumerate(zip(self.blocks, bias)):
            if self.checkpoint == "attention" and torch.is_grad_enabled():
                q, k, v, gate = self.checkpointed(_dit_attention_inputs, block, x, t)
                o = block.attention.attend(q, k, v, b)
                x = self.checkpointed(_dit_block_output, block, x, o, gate, t)
            else:
                x = self.run_block(block, x, b, t, index=i)
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
            [no_blocks, H, N, N] contiguous; with the Triton attention backend a view of a padded buffer whose row stride
            is a multiple of 16 (the layout the Triton kernels load fastest)
        """
        attn = [block.attention for block in self.blocks]
        eps = attn[0].norm_z.eps
        if any(a.norm_z.eps != eps for a in attn):
            raise ValueError("pair_bias needs the same LayerNorm eps in all blocks")
        if z.shape[-1] < 64:            # torch's LayerNorm kernel takes 18 ms for the 2407^2 x 16 atom pairs (90x its memory time)
            zc = z - z.mean(-1, keepdim=True)
            z_hat = zc * torch.rsqrt(zc.pow(2).mean(-1, keepdim=True) + eps)
        else:
            z_hat = F.layer_norm(z, z.shape[-1:], eps=eps)
        weight = torch.cat([a.linear_z.weight * a.norm_z.weight for a in attn])             # [no_blocks * H, c_z]
        bias = torch.cat([a.linear_z.weight @ a.norm_z.bias for a in attn])
        logits = F.linear(z_hat, weight, bias).permute(2, 0, 1) + gen_attn_mask(z_mask, -attn[0].inf)
        logits = logits.reshape(len(attn), -1, *z_mask.shape)
        # the padded layout only pays off in the Triton kernels; sdpa keeps the contiguous bias (torch.compile handles its
        # alignment itself and fails on the padded view)
        return aligned_bias(logits) if attn[0].attend.backend == "triton" else logits.contiguous()
