"""Transformer blocks and their stacks (Stack: atom, triangle, Evoformer, Pairformer, DiT)."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from PhysDock.models.primitives import (AdaLNSwiGLUFFN, AttentionWithBias, DiTAttention, LayerNormPairBias, MSAColumnAttention,
                                        MSARowAttentionWithBias, OuterProductMean, RMSNormSwiGLUFFN, TimeModulation,
                                        TriangleAttention, TriangleMultiplication)
from PhysDock.models.primitives.attentions import HEAD_DIM


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
            module.checkpoint = _checkpoint_value(mode)


def _checkpoint_value(mode):
    """Stack.checkpoint of a checkpointing mode: True, False, a fraction in (0, 1) or "attention" """
    if isinstance(mode, str):
        return {"full": True, "none": False}.get(mode, mode)
    return bool(mode) if mode in (0, 1) else float(mode)


class AtomBlock(nn.Module):
    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float) -> None:
        super().__init__()
        self.attention = AttentionWithBias(c_a, c_ap, inf, eps)
        self.transition = RMSNormSwiGLUFFN(c_a, eps)

    def forward(self, a, ap, ap_mask):
        a = self.attention(a, ap, ap_mask)
        return self.transition(a)


class AtomTransformer(Stack):
    """Atom transformer: pair-biased attention + transition.

    a: [N_atom, c_a], ap: [N_atom, N_atom, c_ap], ap_mask: [N_atom, N_atom] -> a: [N_atom, c_a]
    """

    def __init__(self, c_a: int, c_ap: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(AtomBlock, no_blocks, c_a, c_ap, inf, eps)


class PairUpdate:
    """Mixin of the blocks with the pair half of the Pairformer: two triangle multiplications, two triangle attentions and the pair
    transition. The modules are registered on the block itself (build_pair, called where the block wants them in its parameter order)."""

    def build_pair(self, c_z: int, inf: float, eps: float) -> None:
        self.triangle_row_update = TriangleMultiplication(c_z, eps, outgoing=True)
        self.triangle_col_update = TriangleMultiplication(c_z, eps, outgoing=False)
        self.triangle_row_attention = TriangleAttention(c_z, inf, eps, starting_node=True)
        self.triangle_col_attention = TriangleAttention(c_z, inf, eps, starting_node=False)
        self.pair_transition = RMSNormSwiGLUFFN(c_z, eps)

    def update_pair(self, z, z_mask):
        z = self.triangle_row_update(z, z_mask)
        z = self.triangle_col_update(z, z_mask)
        z = self.triangle_row_attention(z, z_mask)
        z = self.triangle_col_attention(z, z_mask)
        return self.pair_transition(z)


class TriangleBlock(PairUpdate, nn.Module):
    def __init__(self, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.build_pair(c_z, inf, eps)

    def forward(self, z, z_mask):
        z = self.update_pair(z, z_mask)
        # The pairs the mask excludes are still read downstream, so the forward keeps them, but no gradient flows into them: the template
        # mask leaves most rows without a valid key (186 of 384 at crop 384), their softmax is over -inf logits only, and the gradient that
        # arrives there is amplified by every block (1e3 per block, 1e35 after the stack) until the global gradient norm is inf or 1e7
        # in a third of the steps, which clipping turns into steps in a random direction.
        return torch.where(z_mask[..., None] != 0, z, z.detach())


class Triangleformer(Stack):
    """Pair-only stack (template embedder).

    z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> z
    """

    def __init__(self, c_z: int, inf: float, eps: float, no_blocks: int) -> None:
        super().__init__(TriangleBlock, no_blocks, c_z, inf, eps)


class EvoformerBlock(PairUpdate, nn.Module):
    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.msa_row_attention = MSARowAttentionWithBias(c_m, c_z, inf, eps)
        self.msa_col_attention = MSAColumnAttention(c_m, eps)
        self.msa_transition = RMSNormSwiGLUFFN(c_m, eps)
        self.opm = OuterProductMean(c_m, c_z, eps)
        self.build_pair(c_z, inf, eps)

    def forward(self, m, z, z_mask):
        m = self.msa_row_attention(m, z, z_mask)
        m = self.msa_col_attention(m)
        m = self.msa_transition(m)
        return m, self.update_pair(self.opm(m, z), z_mask)


class Evoformer(Stack):
    """MSA stack.

    m: [N_msa, N_token, c_m], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (m, z)
    """

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float, no_blocks: int = 4) -> None:
        super().__init__(EvoformerBlock, no_blocks, c_m, c_z, inf, eps, n_state=2)


class PairFormerBlock(PairUpdate, nn.Module):
    """outer_product (optional, off in the released parameters and by default): the single stream writes to the pair stream at the start of the block
    (z + the outer product of two projections of s, OuterProductMean over a one-row "MSA"); the output layer starts at zero, so the block starts as
    without it. Otherwise the information flows from z to s only (the attention bias)."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, outer_product: bool = False) -> None:
        super().__init__()
        if outer_product:
            self.outer_product = OuterProductMean(c_s, c_z, eps)
            self.outer_product.linear_o.weight.data.zero_()      # the output is RMS-normalised: with eps 1e-5 the backward gain at zero is 300
        self.build_pair(c_z, inf, eps)
        self.attention = AttentionWithBias(c_s, c_z, inf, eps)
        self.transition = RMSNormSwiGLUFFN(c_s, eps)

    def forward(self, s, z, z_mask):
        if hasattr(self, "outer_product"):
            # the padding pairs keep z and get no gradient (as in TriangleBlock): the padded s rows must not reach the real pairs or the loss
            z = torch.where(z_mask[..., None] != 0, self.outer_product(s[None], z), z)
        z = self.update_pair(z, z_mask)
        s = self.attention(s, z, z_mask)
        return self.transition(s), z


class Pairformer(Stack):
    """Pair + single stack.

    s: [N_token, c_s], z: [N_token, N_token, c_z], z_mask: [N_token, N_token] -> (s, z)
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 24, outer_product: bool = False) -> None:
        super().__init__(PairFormerBlock, no_blocks, c_s, c_z, inf, eps, n_state=2, outer_product=outer_product)


class DiTBlock(nn.Module):
    def __init__(self, c_s: int, eps: float) -> None:
        super().__init__()
        self.attention = DiTAttention(c_s, eps)
        self.transition = AdaLNSwiGLUFFN(c_s, eps)

    def forward(self, x, bias, mod, n_valid=None, pos=None, radius=None):
        x = self.attention(x, bias, mod[:, :1], n_valid, pos, radius)
        return self.transition(x, mod[:, 1:])


def _dit_attention_inputs(block: DiTBlock, x, mod):
    return block.attention.project(x, mod[:, :1])


def _dit_block_output(block: DiTBlock, x, o, gate, g, mod):
    x = block.attention.output(x, o, gate, g)
    return block.transition(x, mod[:, 1:])


class DiT(Stack):
    """Diffusion transformer conditioned on a timestep embedding and per-block pair biases.

    x: [B, N, c_s], bias: [no_blocks, H, N, N] (self.pair_bias(z, z_mask), computed once per structure), t: [B, 256] timestep embedding,
    n_valid: valid_length(z_mask) or None (the Triton attention skips the padding behind it); pos [B, N, 3] and radius [B] of the distance
    decay, when set_distance_decay was called -> x: [B, N, c_s]
    (the AdaLN modulations of all blocks come from one matmul of t, TimeModulation)

    checkpoint "attention" (selective checkpointing, DiTBlock.forward split in three): the attention inputs / outputs
    are kept and only the projections, norms and the transition around them are recomputed in the backward, so the
    expensive attention (fp32, B noisy samples) runs once.
    """

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float, no_blocks: int = 12) -> None:
        super().__init__(DiTBlock, no_blocks, c_s, eps)
        self.heads = c_s // HEAD_DIM
        self.pair_bias = LayerNormPairBias(c_z, self.heads, no_blocks, inf, eps)
        self.time_mod = TimeModulation(c_s, no_blocks)

    def set_distance_decay(self, strength: float, learnable: bool) -> None:
        """Distance decay of the attention of every block (see SharedBiasAttention); forward then needs pos and radius."""
        for block in self.blocks:
            block.attention.attend.set_distance_decay(self.heads, strength, learnable)

    def forward(self, x, bias, t, n_valid=None, pos=None, radius=None):
        for i, (block, b, mod) in enumerate(zip(self.blocks, bias, self.time_mod(t), strict=True)):
            if self.checkpoint == "attention" and torch.is_grad_enabled():
                q, k, v, gate, g = self.checkpointed(_dit_attention_inputs, block, x, mod)
                o = block.attention.attend(q, k, v, b, n_valid, pos, radius)
                x = self.checkpointed(_dit_block_output, block, x, o, gate, g, mod)
            else:
                x = self.run_block(block, x, b, mod, n_valid, pos, radius, index=i)
        return x
