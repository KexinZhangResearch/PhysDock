"""Triton kernels of PhysDock's DiTTransition: a gated residual SwiGLU feed-forward behind an adaptive LayerNorm, forward and backward.

    y = x + gate * w2( silu(w1 n) * (w3 n) ),   n = LayerNorm(x) * (1 + scale) + shift       (LayerNorm without affine)

x [B, N, C] (B noisy samples of N positions), scale / shift / gate [B, C] (the three chunks of AdaLayerNormZero's Linear(silu(t)), they
are inputs here so the Linear stays an ordinary autograd op), w13 = [w1; w3] [2 * HID, C], w2 [C, HID]. The atom DiT transition (C = 128,
HID = 384, B * N = 147456 rows in training) is this operator in the decoder's TF32 matmuls.

Same structure as rmsnorm_swiglu_ffn (see there): the forward is one kernel over tiles of rows that never writes the hidden activations,
the backward recomputes them (kernel A: dH = [dh1 | dh3], s, n, dacc = gate * dy, dx, and per-tile partial sums of the gradients of
scale, shift and gate) and leaves dW13 = dH^T n, dW2 = dacc^T s to cuBLAS. A tile of BLOCK_P rows lies inside one sample (N % BLOCK_P == 0,
see can_fuse), so scale / shift / gate of the tile are one row of [B, C] and the per-sample gradients are the sums of the tile partials.
The gate gradient needs the output of the feed-forward, so the backward recomputes the second matmul as well.

Precision: see kernels/_common.py (TF32 in the decoder, which runs in float32; norms, silu, products, gates and the residual in float32).

Measured on an H20 (compiled torch as the reference, the atom DiT transition at 3072 atoms, TF32):
  forward   0.909 ms against 1.014 ms (1.11x; 1.02x for the 5 inference samples), no hidden activations kept (793 MiB in torch);
  backward  ~4.7 ms against 2.39 ms. In TF32 a Triton matmul is slower than cuBLAS, the recomputation adds three of them (6 in all), and
            71% of torch's time is the cuBLAS matmuls already: all that fusion can remove is ~1 ms per block. A variant that saves the
            hidden pre-activations instead of recomputing them (H, float32) was also tried: 2.7 ms, still slower than torch.
So the fused path loses in training and is a wash in inference: ENABLED is False and the DiT transitions run torch. The kernels are kept
(tested against torch and float64, see tests in the PR / scripts) for a GPU or precision where the balance is different.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

from ._common import MODES, compute_dtype, matmul_mode, mm, prep_weight

# (C, HID, mode) -> (forward: block, hidden chunk, warps, stages; backward kernel A: block, hidden chunk, warps, stages)
_CONFIGS = {
    (128, 384, MODES["tf32"]): ((128, 32, 8, 2), (32, 32, 4, 2)),
}
MIN_ROWS = 16384
ENABLED = False             # see the module docstring: slower than torch in training, equal in inference


def can_fuse(x: torch.Tensor, hidden: int) -> bool:
    """the kernels apply (and are enabled): a CUDA float32 x [B, N, C] with a supported (channels, hidden) for the current matmul
    precision, enough rows, and N a multiple of every block size of the configuration"""
    if not ENABLED or x.dim() != 3 or not x.is_cuda or x.dtype != torch.float32:
        return False
    cfg = _CONFIGS.get((x.shape[-1], hidden, matmul_mode()))
    if cfg is None or x.numel() // x.shape[-1] < MIN_ROWS:
        return False
    return x.shape[1] % max(cfg[0][0], cfg[1][0]) == 0


@triton.jit
def _fwd_kernel(X, SH, SC, GT, W13, W2, Y, M, NPB, eps,
                C: tl.constexpr, HID: tl.constexpr, HC: tl.constexpr, MODE: tl.constexpr, BLOCK_P: tl.constexpr):
    tile = tl.program_id(0)
    pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < M
    b = (tile * BLOCK_P) // NPB
    oc, ohc = tl.arange(0, C), tl.arange(0, HC)
    x = tl.load(X + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    xc = x - (tl.sum(x, 1) / C)[:, None]
    n0 = xc * tl.math.rsqrt(tl.sum(xc * xc, 1) / C + eps)[:, None]
    n = n0 * (1.0 + tl.load(SC + b * C + oc))[None, :] + tl.load(SH + b * C + oc)[None, :]
    acc = tl.zeros([BLOCK_P, C], tl.float32)
    for h0 in tl.range(0, HID, HC):
        w1 = tl.load(W13 + (h0 + ohc)[None, :] * C + oc[:, None])                    # [C, HC]
        w3 = tl.load(W13 + (HID + h0 + ohc)[None, :] * C + oc[:, None])
        w2 = tl.load(W2 + oc[None, :] * HID + (h0 + ohc)[:, None])                    # [HC, C]
        h1 = mm(n, w1, MODE)
        h3 = mm(n, w3, MODE)
        acc = acc + mm(h1 * tl.sigmoid(h1) * h3, w2, MODE)
    y = x + acc * tl.load(GT + b * C + oc)[None, :]
    tl.store(Y + pos[:, None] * C + oc[None, :], y.to(Y.dtype.element_ty), mask=pm[:, None])


@triton.jit
def _bwd_kernel(X, DY, SH, SC, GT, W13, W2, DX, DH, S, NO, DAC, DSC, DSH, DGT, M, NPB, eps,
                C: tl.constexpr, HID: tl.constexpr, HC: tl.constexpr, MODE: tl.constexpr, BLOCK_P: tl.constexpr):
    tile = tl.program_id(0)
    pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < M
    b = (tile * BLOCK_P) // NPB
    oc, ohc = tl.arange(0, C), tl.arange(0, HC)
    x = tl.load(X + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    dy = tl.load(DY + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    xc = x - (tl.sum(x, 1) / C)[:, None]
    r = tl.math.rsqrt(tl.sum(xc * xc, 1) / C + eps)
    n0 = xc * r[:, None]
    one_sc = 1.0 + tl.load(SC + b * C + oc)
    n = n0 * one_sc[None, :] + tl.load(SH + b * C + oc)[None, :]
    gate = tl.load(GT + b * C + oc)
    dacc = dy * gate[None, :]                                                         # gradient of the feed-forward output
    dn = tl.zeros([BLOCK_P, C], tl.float32)
    acc = tl.zeros([BLOCK_P, C], tl.float32)                                          # the feed-forward output (for the gate gradient)
    cdt = DH.dtype.element_ty
    for h0 in tl.range(0, HID, HC):
        w1n = tl.load(W13 + (h0 + ohc)[:, None] * C + oc[None, :])                    # [HC, C]
        w3n = tl.load(W13 + (HID + h0 + ohc)[:, None] * C + oc[None, :])
        w2 = tl.load(W2 + oc[:, None] * HID + (h0 + ohc)[None, :])                    # [C, HC]
        h1 = mm(n, tl.trans(w1n), MODE)
        h3 = mm(n, tl.trans(w3n), MODE)
        sg = tl.sigmoid(h1)
        a = h1 * sg
        s = a * h3
        ds = mm(dacc, w2, MODE)                                                       # [BLOCK_P, HC]
        dh1 = ds * h3 * sg * (1.0 + h1 * (1.0 - sg))
        dh3 = ds * a
        tl.store(DH + pos[:, None] * (2 * HID) + (h0 + ohc)[None, :], dh1.to(cdt), mask=pm[:, None])
        tl.store(DH + pos[:, None] * (2 * HID) + (HID + h0 + ohc)[None, :], dh3.to(cdt), mask=pm[:, None])
        tl.store(S + pos[:, None] * HID + (h0 + ohc)[None, :], s.to(cdt), mask=pm[:, None])
        dn += mm(dh1, w1n, MODE) + mm(dh3, w3n, MODE)
        acc += mm(s, tl.trans(w2), MODE)
    tl.store(NO + pos[:, None] * C + oc[None, :], n.to(NO.dtype.element_ty), mask=pm[:, None])
    tl.store(DAC + pos[:, None] * C + oc[None, :], dacc.to(cdt), mask=pm[:, None])
    dn = tl.where(pm[:, None], dn, 0.0)
    dn0 = dn * one_sc[None, :]
    dx = dy + r[:, None] * (dn0 - (tl.sum(dn0, 1) / C)[:, None] - n0 * (tl.sum(dn0 * n0, 1) / C)[:, None])
    tl.store(DX + pos[:, None] * C + oc[None, :], dx.to(DX.dtype.element_ty), mask=pm[:, None])
    tl.store(DSC + tile * C + oc, tl.sum(dn * n0, 0))
    tl.store(DSH + tile * C + oc, tl.sum(dn, 0))
    tl.store(DGT + tile * C + oc, tl.sum(tl.where(pm[:, None], dy * acc, 0.0), 0))


@torch.library.custom_op("physdock::adaln_swiglu_ffn_fwd", mutates_args=())
def _ffn_fwd(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor, eps: float,
             mode: int) -> torch.Tensor:
    B, N, C = x.shape
    hid = w2.shape[1]
    bp, hc, warps, stages = _CONFIGS[(C, hid, mode)][0]
    x = x.contiguous()
    y = torch.empty_like(x)
    _fwd_kernel[(triton.cdiv(B * N, bp),)](x, shift.contiguous(), scale.contiguous(), gate.contiguous(), prep_weight(w13, mode), prep_weight(w2, mode), y,
                                           B * N, N, eps, C=C, HID=hid, HC=hc, MODE=mode, BLOCK_P=bp, num_warps=warps, num_stages=stages)
    return y


@_ffn_fwd.register_fake
def _(x, shift, scale, gate, w13, w2, eps, mode):
    return torch.empty_like(x)


@torch.library.custom_op("physdock::adaln_swiglu_ffn_bwd", mutates_args=())
def _ffn_bwd(dy: torch.Tensor, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
             eps: float, mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, C = x.shape
    hid = w2.shape[1]
    bp, hc, warps, stages = _CONFIGS[(C, hid, mode)][1]
    x, dy = x.contiguous(), dy.contiguous()
    M = B * N
    cdt = compute_dtype(mode)
    dx = torch.empty_like(x)
    dh = torch.empty(M, 2 * hid, device=x.device, dtype=cdt)
    s = torch.empty(M, hid, device=x.device, dtype=cdt)
    n = torch.empty(M, C, device=x.device, dtype=cdt)
    dacc = torch.empty(M, C, device=x.device, dtype=cdt)
    ntiles = triton.cdiv(M, bp)
    dsc, dsh, dgt = (torch.empty(ntiles, C, device=x.device, dtype=torch.float32) for _ in range(3))
    _bwd_kernel[(ntiles,)](x, dy, shift.contiguous(), scale.contiguous(), gate.contiguous(), prep_weight(w13, mode), prep_weight(w2, mode), dx, dh, s, n,
                           dacc, dsc, dsh, dgt, M, N, eps, C=C, HID=hid, HC=hc, MODE=mode, BLOCK_P=bp, num_warps=warps, num_stages=stages)
    per_sample = lambda t: t.view(B, N // bp, C).sum(1)
    dw13 = (dh.t() @ n).float()
    dw2 = (dacc.t() @ s).float()
    return dx, per_sample(dsh), per_sample(dsc), per_sample(dgt), dw13, dw2


@_ffn_bwd.register_fake
def _(dy, x, shift, scale, gate, w13, w2, eps, mode):
    f = lambda t: x.new_empty(t.shape, dtype=torch.float32)
    return torch.empty_like(x), f(shift), f(scale), f(gate), f(w13), f(w2)


def _setup_context(ctx, inputs, output):
    x, shift, scale, gate, w13, w2, eps, mode = inputs
    ctx.save_for_backward(x, shift, scale, gate, w13, w2)
    ctx.conf = (eps, mode)


def _backward(ctx, dy):
    x, shift, scale, gate, w13, w2 = ctx.saved_tensors
    dx, dsh, dsc, dgt, dw13, dw2 = _ffn_bwd(dy, x, shift, scale, gate, w13, w2, *ctx.conf)
    return dx, dsh.to(shift.dtype), dsc.to(scale.dtype), dgt.to(gate.dtype), dw13.to(w13.dtype), dw2.to(w2.dtype), None, None


_ffn_fwd.register_autograd(_backward, setup_context=_setup_context)


def adaln_swiglu_ffn(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
                     eps: float) -> torch.Tensor:
    """x + gate * w2(silu(w1 n) * (w3 n)) with n = LayerNorm(x) * (1 + scale) + shift; differentiable with respect to every tensor input.
    The matmul precision follows torch's (see kernels/_common.py).

    Args:
        x: [B, N, C] float32; shift, scale, gate: [B, C]; w13: [2 * HID, C] (rows w1 | w3), w2: [C, HID]
        eps: LayerNorm epsilon

    Returns:
        [B, N, C]
    """
    return _ffn_fwd(x, shift, scale, gate, w13, w2, eps, matmul_mode())
