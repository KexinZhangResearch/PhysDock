"""Layers of the model, bottom up.

Base layers      Linear, RMSNorm, AdaLayerNormZero.

Kernel-level modules: operators that are one fused kernel in the Triton backend and a few torch ops in the torch backend, with the
same parameters, the same function and the same matmul precision in both. Every one has a `backend` ("torch" | "triton"), set for a
whole model by set_backend. The Triton path is taken where it applies and is faster (see the can_fuse functions of
PhysDock.models.kernels); anywhere else the module runs its torch path.
  SwiGLUFFN            x + w2(silu(w1 x) * w3 x)                     (w1 and w3 merged: w13)
  RMSNormSwiGLUFFN     x + SwiGLUFFN-update(RMSNorm(x))              (the Transition of the pair, MSA, single and atom streams)
  AdaLNSwiGLUFFN       x + gate * SwiGLUFFN-update(AdaLN(x, mod))      (the transition of the DiT blocks; torch only, see its docstring)
  RMSNormLinear        RMSNorm + one linear layer (the merged q, k, v, g projection of the attentions, the q, k of the outer product mean)
  OuterProductMean     z + the pair update from the outer product of two MSA projections
  LinearGateResidual   x + linear(o) * gate: the output projection, gate and residual of every attention
  LayerNormPairBias    the logit biases of all blocks of a DiT from the pair representation, once per structure -> [blocks, H, N, N]
  RMSNormPairBias      linear(RMSNorm(z)) + attention mask -> [H, N, N]    (norm + linear + mask of a pair bias)
  SharedBiasAttention  softmax(q k^T / sqrt(32) + bias) v for a bias shared by the batch (no parameters)
  TriangleMultiplication  the gated triangular multiplicative update (outgoing / incoming edges)
Every module that has a residual stream returns it residual-added, x + update (the fused kernels add the residual in their epilogue).

Blocks built from them: attentions.py (the attention layers), transformers.py (the blocks and stacks); timestep_embeddings.py.
"""
import contextlib
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

from PhysDock.models import kernels

BACKENDS = ("torch", "triton")


def valid_length(pair_mask: torch.Tensor) -> torch.Tensor:
    """[1] int32 on the device: 1 + the last position whose diagonal entry of the [N, N] pair mask is set (0 if none) = the length of the
    part of the sequence that holds real positions when the padding is at the end, as in the training crops; no host synchronisation"""
    diag = pair_mask.diagonal() != 0
    return ((torch.arange(diag.numel(), device=diag.device, dtype=torch.int32) + 1) * diag).amax().reshape(1)
_SKIP_INIT = False


# ---------------------------------------------------------------------------------------------------------- initialisation
@contextlib.contextmanager
def skip_weight_init():
    """Skip the slow truncated-normal init of Linear (use only when all weights are loaded afterwards)."""
    global _SKIP_INIT
    _SKIP_INIT = True
    try:
        yield
    finally:
        _SKIP_INIT = False


def lecun_normal_init_(weight: torch.Tensor) -> None:
    """In-place truncated normal (2 std) with variance 1 / fan_in (AlphaFold init)."""
    fan_in = weight.shape[1]
    std = math.sqrt(1.0 / max(1, fan_in)) / truncnorm.std(a=-2, b=2, loc=0, scale=1)
    samples = truncnorm.rvs(a=-2, b=2, loc=0, scale=std, size=math.prod(weight.shape))
    with torch.no_grad():
        weight.copy_(torch.tensor(np.reshape(samples, weight.shape), device=weight.device))


# ---------------------------------------------------------------------------------------------------------- base layers
class Linear(nn.Linear):
    """nn.Linear with AlphaFold init ("default" LeCun, "final" zeros, "gating" zeros + bias 1)."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, init: str = "default") -> None:
        super().__init__(in_dim, out_dim, bias=bias)
        with torch.no_grad():
            if bias:
                self.bias.fill_(0)
            if init == "default":
                if not _SKIP_INIT:
                    lecun_normal_init_(self.weight)
            elif init in ("final", "gating"):
                self.weight.fill_(0.0)
                if init == "gating" and bias:
                    self.bias.fill_(1.0)
            else:
                raise ValueError(f"unknown init {init!r}")


class RMSNorm(nn.Module):
    """RMS norm over the last dim, computed in float32, with a learned scale."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> [..., dim]"""
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class AdaLayerNormZero(nn.Module):
    """DiT adaptive LayerNorm: LN(x) * (1 + scale) + shift, plus a gate for the following residual branch. The modulation
    [shift | scale | gate] is an argument (TimeModulation computes it for all layers of a DiT at once)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, mod: torch.Tensor):
        """
        Args:
            x: [B, N, dim]
            mod: [B, 1, 3 * dim]

        Returns:
            x: [B, N, dim] modulated input, gate: [B, 1, dim]
        """
        shift, scale, gate = mod.chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate


class TimeModulation(nn.Module):
    """The AdaLN modulations of all the blocks of a DiT from the timestep embedding in one matmul: forward(t)[i] is [B, 2, 3 * dim], the
    [shift | scale | gate] of the attention (index 0) and of the transition (index 1) of block i. They depend on t only, so the DiT
    computes them once per call instead of two small matmuls per block."""

    def __init__(self, dim: int, num_blocks: int, cond_dim: int = 256) -> None:
        super().__init__()
        self.num_blocks, self.dim = num_blocks, dim
        self.linear = Linear(cond_dim, num_blocks * 2 * 3 * dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t [B, cond_dim] -> [num_blocks, B, 2, 3 * dim]"""
        return self.linear(F.silu(t)).view(t.shape[0], self.num_blocks, 2, 3 * self.dim).transpose(0, 1)


# ---------------------------------------------------------------------------------------------------------- kernel-level modules
class KernelModule(nn.Module):
    """Base of the modules with a Triton implementation: `backend` is "torch" or "triton" (see set_backend)."""
    backend = "torch"


def set_backend(model: nn.Module, backend: str) -> None:
    """Implementation of every kernel-level module of `model`: "torch" or "triton" (PhysDock.models.kernels; the matmul precision of
    the torch path: bfloat16 under autocast, TF32 / float32 otherwise)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    for module in model.modules():
        if isinstance(module, KernelModule):
            module.backend = backend


class SwiGLUFFN(KernelModule):
    """SwiGLU feed-forward with its residual: forward(x) = x + w2(silu(w1 x) * w3 x); the hidden size is 2/3 * 4 * dim rounded up to
    a multiple of 128 and w1, w3 are one matrix (rows w1 | w3). Triton: the fused kernel of the narrow atom-pair features (16 / 32
    channels, hidden 128, kernels.swiglu_ffn), which keeps no [..., hidden] activation for the backward (7 GiB for 3072 atoms)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = int(2 * (4 * dim) / 3)
        hidden = 128 * ((hidden + 127) // 128)
        self.w13 = Linear(dim, 2 * hidden, bias=False)
        self.w2 = Linear(hidden, dim, bias=False)

    def update(self, x: torch.Tensor) -> torch.Tensor:
        """w2(silu(w1 x) * w3 x), the torch computation without the residual"""
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> x + update(x)"""
        if self.backend == "triton" and kernels.swiglu_ffn.can_fuse(x, self.w2.in_features):
            return kernels.swiglu_ffn.swiglu_ffn(x, self.w13.weight, self.w2.weight)
        return x + self.update(x)


class RMSNormSwiGLUFFN(KernelModule):
    """Pre-norm SwiGLU feed-forward with its residual: forward(x) = x + SwiGLUFFN.update(RMSNorm(x)) (the Transition of the pair, MSA,
    single and atom streams). Triton: norm, both matmul stages and the residual in one kernel pair (kernels.rmsnorm_swiglu_ffn),
    without the [..., 2 * hidden] activations the torch backward keeps (361 MiB per pair transition at 384 tokens); used where it is
    faster than torch: the 128-channel pair transition in bfloat16 (1.2x forward, 1.07x forward + backward) and, without gradients,
    in TF32."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.norm = RMSNorm(dim, eps=eps)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., dim] -> x + update"""
        if self.backend == "triton" and kernels.rmsnorm_swiglu_ffn.can_fuse(x, self.ffn.w2.in_features):
            return kernels.rmsnorm_swiglu_ffn.rmsnorm_swiglu_ffn(x, self.norm.weight, self.ffn.w13.weight, self.ffn.w2.weight, self.norm.eps)
        if self.backend == "triton" and kernels.rmsnorm_swiglu_ffn.can_split(x, self.ffn.w2.in_features):
            return kernels.rmsnorm_swiglu_ffn.rmsnorm_swiglu_ffn_split(x, self.norm.weight, self.ffn.w13.weight, self.ffn.w2.weight, self.norm.eps)
        return x + self.ffn.update(self.norm(x))


class AdaLNSwiGLUFFN(nn.Module):
    """Gated residual SwiGLU feed-forward behind an adaptive LayerNorm: forward(x, mod) = x + gate * SwiGLUFFN.update(AdaLN(x, mod)) (the
    transition of the DiT blocks). Torch only: a fused kernel pair (norm, both stages, the gate and the gradients of scale, shift and
    gate in one) was written and measured on an H20 for the atom DiT transition (3072 atoms, TF32, the decoder's precision): the
    forward is 1.1x faster, the backward 2x slower than torch (the recomputation adds three TF32 matmuls, which are slower in Triton
    than in cuBLAS, and 71% of torch's time is cuBLAS already: at most ~1 ms per block is left to fuse), so it was removed (git history:
    kernels/adaln_swiglu_ffn.py)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.norm = AdaLayerNormZero(dim, eps=eps)
        self.ffn = SwiGLUFFN(dim)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, dim]
            mod: [B, 1, 3 * dim] modulation (TimeModulation)

        Returns:
            [B, N, dim] x + update
        """
        n, gate = self.norm(x, mod)
        return x + self.ffn.update(n) * gate


class RMSNormLinear(KernelModule):
    """RMSNorm of the input and one linear layer: forward(x) = linear(RMSNorm(x)) with a bias on the last `bias_dim` output columns only
    (bias_dim=0: none; the q, k, v, g projection of the attentions has the bias of g alone, the q, k projection of the outer product mean
    a bias on all columns). The projections merged into one matmul are split by the caller. Triton (kernels.rmsnorm_linear): the norm is
    done in registers on the way into the matmul and the backward recomputes it, so no normalised copy of the [..., in_dim] input is
    kept or written."""

    def __init__(self, in_dim: int, out_dim: int, eps: float, bias_dim: int = 0) -> None:
        super().__init__()
        self.norm = RMSNorm(in_dim, eps)
        self.linear = Linear(in_dim, out_dim, bias=False)
        if bias_dim:
            self.bias = nn.Parameter(torch.zeros(bias_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., in_dim] -> [..., out_dim]"""
        bias = getattr(self, "bias", None)
        if self.backend == "triton" and kernels.rmsnorm_linear.can_fuse(x, self.linear.out_features):
            return kernels.rmsnorm_linear.rmsnorm_linear(x, self.norm.weight, self.linear.weight, bias, self.norm.eps)
        if bias is not None:
            bias = F.pad(bias, (self.linear.out_features - bias.numel(), 0))
        return F.linear(self.norm(x), self.linear.weight, bias)


class OuterProductMean(KernelModule):
    """Pair update from the outer product q_i x k_j of two MSA projections, summed (not averaged) over rows; forward(m, z) = z + update.
    q and k are one RMSNormLinear (c_m -> 2 * 32). Triton (kernels.outer_product_mean): the outer product exists only as a register tile; the
    projection, RMSNorm and residual are fused with it, and the [N, N, 32, 32] tensor (and the copies of it) of torch is never formed."""

    def __init__(self, c_m: int, c_z: int, eps: float) -> None:
        super().__init__()
        self.qk = RMSNormLinear(c_m, 2 * 32, eps, bias_dim=2 * 32)
        # LeCun init, not zeros: the output is RMS-normalised, so a zero projection has a backward gain of 1 / sqrt(eps) at initialisation
        # (gradient norm of 1e7 and non-finite steps in the first hundreds of steps); with it the initial norm is 60
        self.linear_o = Linear(32 * 32, c_z)
        self.norm_out = RMSNorm(c_z, eps)

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """m: [N_msa, N_token, c_m], z: [N_token, N_token, c_z] -> z + update"""
        n = m.shape[1]
        qk = self.qk(m)
        if self.backend == "triton" and kernels.outer_product_mean.can_fuse(qk, z):
            return kernels.outer_product_mean.outer_product_mean(qk, z, self.linear_o.weight, self.linear_o.bias, self.norm_out.weight,
                                                                 self.norm_out.eps)
        q, k = qk.chunk(2, dim=-1)
        outer = torch.einsum("bic,bjd->ijcd", q, k).reshape(n, n, -1)
        return z + self.norm_out(self.linear_o(outer))


class RMSNormPairBias(KernelModule):
    """Attention logit bias of a pair representation: linear(RMSNorm(z)) as [H, N, N] plus a large negative number where the pair is
    masked (norm + linear + mask of the single and MSA row attention). Triton: one kernel pair (kernels.pair_bias) instead of an
    RMSNorm, a matmul with 4-16 output columns, a permute and a mask add, and for the backward their six counterparts (the weight
    gradient contracts over N^2 positions); 6.2 -> 0.96 ms for the atom pairs of 2048 atoms."""

    def __init__(self, c_z: int, heads: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm = RMSNorm(c_z, eps)
        self.linear = Linear(c_z, heads, bias=False)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]
            like: a tensor of the dtype of the result (the queries of the attention: bfloat16 under autocast)

        Returns:
            [H, N, N]
        """
        if self.backend == "triton" and kernels.pair_bias.can_fuse(z, self.linear.out_features):
            return kernels.pair_bias.pair_bias(z, z_mask, self.norm.weight, self.linear.weight, self.norm.eps, self.inf,
                                               out_dtype=like.dtype, pad=z.shape[0] >= 1024)
        # masked entries are overwritten, not shifted by -inf: the logits there get no gradient (with an additive mask a fully masked row has
        # a uniform softmax and passes gradient to every masked logit, which the RMSNorm of the zero padding pairs amplifies by 1 / sqrt(eps));
        # contiguous: sdpa gives wrong gradients for a permuted bias view
        logits = self.linear(self.norm(z)).permute(2, 0, 1).to(like.dtype)
        return logits.masked_fill(z_mask[None] == 0, -self.inf).contiguous()


def merge_heads(o: torch.Tensor) -> torch.Tensor:
    """[..., H, N, D] -> [..., N, H * D]"""
    o = o.transpose(-2, -3)
    return o.reshape(*o.shape[:-2], -1)


class LinearGateResidual(KernelModule):
    """The end of every attention: forward(x, o, gate) = x + linear(merge_heads(o)) * gate, for the stream x [..., N, dim], the attention
    output o [..., H, N, 32] in the head layout and the output gate (g of the gated attentions, the AdaLN gate of the DiT). Triton
    (kernels.linear_gate_residual): reads o in the head layout (no merged copy), and the matmul, gate and residual are one pass; the
    backward is two kernels without the elementwise passes and the bias reduction of torch."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = Linear(dim, dim)

    def forward(self, x: torch.Tensor, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """x [..., N, dim], o [..., H, N, 32], gate broadcastable to x -> [..., N, dim]"""
        if self.backend == "triton" and kernels.linear_gate_residual.can_fuse(x, o):
            return kernels.linear_gate_residual.linear_gate_residual(x, o, gate, self.linear.weight, self.linear.bias)
        return x + self.linear(merge_heads(o)) * gate


class LayerNormPairBias(KernelModule):
    """The attention logit biases of every block of a DiT stack from its conditioning pair representation, in one matmul:
    forward(z, z_mask)[b] = linear_b(LayerNorm(z)) + mask, returned as [num_blocks, H, N, N] (the stack passes bias[b] to block b).
    The biases depend on the conditioning alone, so the decoder computes them once per structure and every noisy sample and
    denoising step reuses them. The LayerNorm has no affine of its own (the scale and shift of every block would be folded into its
    projection anyway): `linear` has num_blocks * H rows, the rows of block b are [b * H, (b + 1) * H). Triton (kernels.pair_bias, up to
    16 rows: the atom pairs): LayerNorm, projection, mask and the layout with a row stride that is a multiple of 16 (the layout the
    attention kernels load fastest) in one pass over z, without the normalised [N, N, c_z] copy and the [N, N, rows] logits of torch
    (3.9 GiB for the 3072 atoms of a crop)."""

    def __init__(self, c_z: int, heads: int, num_blocks: int, inf: float, eps: float) -> None:
        super().__init__()
        self.heads, self.inf, self.eps = heads, inf, eps
        self.linear = Linear(c_z, num_blocks * heads)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """z [N, N, c_z], z_mask [N, N] -> [num_blocks, H, N, N]"""
        if self.backend == "triton" and kernels.pair_bias.can_fuse(z, self.linear.out_features):
            logits = kernels.pair_bias.pair_bias(z, z_mask, z.new_ones(z.shape[-1]), self.linear.weight, self.eps, self.inf,
                                                 out_dtype=z.dtype, pad=True, lin_bias=self.linear.bias, center=True)
            return logits.reshape(-1, self.heads, *z_mask.shape)
        if z.shape[-1] < 64:            # torch's LayerNorm kernel takes 18 ms for the 2407^2 x 16 atom pairs (90x its memory time)
            zc = z - z.mean(-1, keepdim=True)
            z = zc * torch.rsqrt(zc.pow(2).mean(-1, keepdim=True) + self.eps)
        else:
            z = F.layer_norm(z, z.shape[-1:], eps=self.eps)
        logits = self.linear(z).permute(2, 0, 1).masked_fill(z_mask == 0, -self.inf)       # no gradient at the masked entries (see RMSNormPairBias)
        logits = logits.reshape(-1, self.heads, *z_mask.shape)
        # the padded layout only pays off in the Triton kernels; sdpa keeps the contiguous bias (torch.compile handles its alignment
        # itself and fails on the padded view)
        return kernels.shared_bias_attention.aligned_bias(logits) if self.backend == "triton" else logits.contiguous()


class SharedBiasAttention(KernelModule):
    """softmax(q k^T / sqrt(D) + bias) v for q, k, v [..., H, N, D] and a logit bias [H, N, N] shared by all leading (batch) entries: the
    noisy samples of the DiT, the rows of the triangle and MSA row attention. No parameters. Torch: sdpa; Triton: the fused forward and
    backward of kernels.shared_bias_attention (the bias tile is loaded once for the whole batch, dbias accumulated in registers; the
    TF32 inputs are rounded as cuBLAS does), 1.7-2x faster than sdpa in forward + backward in bfloat16 and 2-5x in the TF32 forward."""

    skip_padding = False            # set by PhysDock(skip_padding=True): the Triton kernels skip the padding at the end of the sequence

    def valid_length(self, pair_mask: torch.Tensor):
        """n_valid of a pair mask for forward; None (nothing computed, the kernels compiled without the feature) unless the Triton backend
        and skip_padding are on. Worth it only when the padding is a sizeable part and a rank does not wait for a fuller one (single GPU,
        batched inference with padding); in data-parallel training the fullest crop of the ranks sets the speed."""
        return valid_length(pair_mask) if self.backend == "triton" and self.skip_padding else None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, n_valid: torch.Tensor = None) -> torch.Tensor:
        """q, k, v: [H, N, D] or [B, H, N, D]; bias [H, N, N]; n_valid: valid_length of the mask, the Triton kernels skip the padding
        behind it (the keys there must be masked by the bias; the rows there get 0) -> o of the shape of q"""
        if self.backend == "triton":
            if q.dim() == 3:
                return kernels.shared_bias_attention.shared_bias_attention(q[None], k[None], v[None], bias, n_valid)[0]
            return kernels.shared_bias_attention.shared_bias_attention(q, k, v, bias, n_valid)
        return F.scaled_dot_product_attention(q, k, v, bias)


class TriangleMultiplication(KernelModule):
    """Gated triangular multiplicative update out[i, k] = sum_j a[i, j] * b[k, j] of the pair representation (outgoing=False: the
    incoming edges, the same on the transposed pair), returned as z + update.

    All projections of the normalised input are one Linear (linear_in, C -> 4 * 32 + C): a = qx * sigmoid(q) and b = kx * sigmoid(k)
    (32 channels each, masked) feed the product, the last C rows are the output gate. Triton: norm, projection, products and
    output projection in six kernels (kernels.triangle_update; TF32 for float32 inputs, bf16 under autocast); 2.5-4x faster than torch
    eager and ahead of torch.compile in forward + backward."""

    def __init__(self, c_z: int, eps: float, outgoing: bool = True) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.norm_in = RMSNorm(c_z, eps)
        self.norm_out = RMSNorm(32, eps)
        self.linear_in = Linear(c_z, 4 * 32 + c_z, init="final")
        self.linear_z = Linear(32, c_z, init="final")
        with torch.no_grad():                       # the initial values of the original layout, drawn in its order
            for rows in ((32, 64), (0, 32), (96, 128), (64, 96)):               # q, qx, k, kx
                self.linear_in.weight[rows[0]:rows[1]] = Linear(c_z, 32).weight
            self.linear_in.bias[4 * 32:] = 1.0                                  # gate: zero weight, bias 1

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] z + update
        """
        if self.backend == "triton":
            return z + kernels.triangle_update.triangle_update(
                z, z_mask, self.norm_in.weight, self.linear_in.weight, self.linear_in.bias, self.norm_out.weight,
                self.linear_z.weight, self.linear_z.bias, self.norm_in.eps, self.outgoing)
        z_in = z
        if not self.outgoing:
            z = z.transpose(-2, -3)
        z = self.norm_in(z)
        qx, q, kx, k, g = self.linear_in(z).split([32, 32, 32, 32, z.shape[-1]], dim=-1)
        q = qx * torch.sigmoid(q) * z_mask[..., None]
        k = kx * torch.sigmoid(k) * z_mask[..., None]
        o = self.linear_z(self.norm_out(torch.einsum("...ijc,...Ijc->...iIc", q, k))) * torch.sigmoid(g)
        if not self.outgoing:
            o = o.transpose(-2, -3)
        return z_in + o


# the blocks use the layers above, so they are imported last
from .attentions import (AttentionWithBias, MSARowAttentionWithBias, MSAColumnAttention,  # noqa: E402
                         TriangleAttention, DiTAttention)
from .timestep_embeddings import TimestepEmbeddings  # noqa: E402
from .transformers import (Stack, AtomTransformer, Triangleformer, Evoformer, Pairformer, DiT,  # noqa: E402
                           set_checkpointing)
