"""Triton kernels of the attention logit bias computed from a pair representation, forward and backward:

    bias[h, i, j] = sum_c W[h, c] * (norm_w[c] * z[i, j, c] * rsqrt(mean_c z[i, j, c]^2 + eps)) + b[h] + (0 if mask[i, j] else -inf)

(RMSNormPairBias: linear(RMSNorm(z)) + attention mask; with center=True the norm is a LayerNorm without affine, the normalisation of
LayerNormPairBias: the same kernels, z - mean(z) in place of z and norm_w = 1; b is the optional bias of the linear layer).
z [N, N, C], W [H, C], out [H, N, N]. Any number of heads up to 16 (the register tile is padded to a power of two).

Why a kernel: torch runs this as an RMSNorm (several passes over z), a matmul with 4-16 output columns, a permute and a mask add,
and the backward as an RMSNorm backward, two matmuls whose contraction axis is the N^2 positions (K = 4 million, 4-16 output
columns: cutlass runs them at a few percent of the GPU) and a column reduction of 1.2 ms for the 2048^2 x 16 atom pairs. Here z is
read once in the forward and z + dbias once in the backward.

Precision, the same as the torch expression it replaces (never lower): the normalisation in float32, then
  bfloat16 result (autocast): the matmul inputs rounded to bfloat16, float32 accumulation, the mask added in bfloat16
                              (what torch's autocast linear does);
  float32 result:             float32 FMAs (more accurate than torch's TF32 / float32 matmul; these paths are not hot).
The 128-channel pair representation uses tensor cores; the 16-channel atom pairs (K = 16, 4 heads) are memory bound and use per-head
float32 multiply-reduce in both cases.
"""
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ._common import MODES, mm, mm_act, num_sms, prep_weight


def _config(c: int, bf16: bool):
    """fixed launch configuration (swept on an H20): (FMA path, forward block, warps, stages, backward block, warps, stages,
    programs per SM). Wide z (c >= 64): tl.dot, small tiles in float32 / tensor-core tiles in bfloat16; narrow z (the atom
    pairs): per-head multiply-reduce, big tiles."""
    if c < 64:
        return (True, 128, 2, 2, 256, 2, 2, 4)
    return (False, 64, 4, 2, 32, 4, 3, 4) if bf16 else (False, 32, 2, 2, 16, 4, 2, 4)


def can_fuse(z: torch.Tensor, n_heads: int) -> bool:
    """the kernels apply: CUDA z with c a power of two in [16, 256] and 1 to 16 heads"""
    c = z.shape[-1]
    return z.is_cuda and z.dtype in (torch.float32, torch.bfloat16) and 16 <= c <= 256 and c & (c - 1) == 0 and 1 <= n_heads <= 16


@triton.jit
def _fwd_kernel(Z, MASK, NW, W, LB, OUT, NN, N, stride_oh, stride_oi, eps, neg,
                C: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, OUT_BF16: tl.constexpr, FMA: tl.constexpr,
                MODE: tl.constexpr, BLOCK_P: tl.constexpr, CENTER: tl.constexpr, HAS_B: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < NN
    oc, oh = tl.arange(0, C), tl.arange(0, HP)
    z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    if CENTER:
        z = z - (tl.sum(z, 1) / C)[:, None]
    r = tl.math.rsqrt(tl.sum(z * z, 1) / C + eps)
    zn = (z * r[:, None]) * tl.load(NW + oc).to(tl.float32)[None, :]
    m = tl.load(MASK + pos, mask=pm, other=1.0).to(tl.float32)
    add = tl.where(m == 0.0, neg, 0.0)
    row = pos // N
    dst = row * stride_oi + (pos - row * N)
    if FMA:
        for h in tl.static_range(H):
            lg = tl.sum(zn * tl.load(W + h * C + oc)[None, :], 1)
            if HAS_B:
                lg = lg + tl.load(LB + h).to(tl.float32)
            if OUT_BF16:
                res = lg.to(tl.bfloat16) + add.to(tl.bfloat16)
            else:
                res = lg + add
            tl.store(OUT + h * stride_oh + dst, res, mask=pm)
    else:
        w = tl.load(W + oh[None, :] * C + oc[:, None], mask=(oh < H)[None, :], other=0.0)               # [C, HP]
        logit = mm(zn, w, MODE)                                                                         # [BLOCK_P, HP]
        if HAS_B:
            logit = logit + tl.load(LB + oh, mask=oh < H, other=0.0).to(tl.float32)[None, :]
        if OUT_BF16:
            res = logit.to(tl.bfloat16) + add.to(tl.bfloat16)[:, None]
        else:
            res = logit + add[:, None]
        tl.store(OUT + oh[None, :] * stride_oh + dst[:, None], res, mask=pm[:, None] & (oh < H)[None, :])


@triton.jit
def _bwd_kernel(Z, MASK, NW, W, DB, DZ, PART, NN, N, sdh, sdi, sdj, eps, ntiles,
                C: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, NPROG: tl.constexpr, FMA: tl.constexpr,
                MODE: tl.constexpr, BLOCK_P: tl.constexpr, CENTER: tl.constexpr):
    """persistent programs over tiles of BLOCK_P positions; PART[pid] = [dW (HP x C, row-major) | dnorm_w (C) | db (HP)]. The masked entries are
    constants (-inf) and pass no gradient: db is zeroed there."""
    pid = tl.program_id(0)
    oc, oh = tl.arange(0, C), tl.arange(0, HP)
    nw = tl.load(NW + oc).to(tl.float32)
    w = tl.load(W + oh[:, None] * C + oc[None, :], mask=(oh < H)[:, None], other=0.0)                  # [HP, C]
    acc_w = tl.zeros([HP, C], tl.float32)
    acc_n = tl.zeros([C], tl.float32)
    acc_b = tl.zeros([HP], tl.float32)
    for t in range(pid, ntiles, NPROG):
        pos = t * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < NN
        row = pos // N
        col = pos - row * N
        keep = (tl.load(MASK + tl.minimum(pos, NN - 1)).to(tl.float32) != 0.0).to(tl.float32)        # (clamped index: no masked load, whose select fails to compile here)
        z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        if CENTER:
            z = z - (tl.sum(z, 1) / C)[:, None]
        r = tl.math.rsqrt(tl.sum(z * z, 1) / C + eps)
        zhat = z * r[:, None]
        zn = zhat * nw[None, :]
        if FMA:
            dzn = tl.zeros([BLOCK_P, C], tl.float32)
            for h in tl.static_range(H):
                db = tl.load(DB + h * sdh + row * sdi + col * sdj, mask=pm, other=0.0).to(tl.float32) * keep
                dzn += db[:, None] * tl.load(W + h * C + oc)[None, :]
                acc_w += tl.where(oh == h, 1.0, 0.0)[:, None] * tl.sum(db[:, None] * zn, 0)[None, :]
                acc_b += tl.where(oh == h, 1.0, 0.0) * tl.sum(db, 0)
        else:
            dbt = tl.load(DB + oh[:, None] * sdh + (row * sdi + col * sdj)[None, :], mask=(oh < H)[:, None] & pm[None, :],
                          other=0.0).to(tl.float32)                                                    # [HP, BLOCK_P]
            dbt = dbt * keep[None, :]
            acc_w += mm_act(dbt, zn, MODE)                                                             # dW += db^T zn
            acc_b += tl.sum(dbt, 1)
            dzn = mm(tl.trans(dbt), w, MODE)                                                           # [BLOCK_P, C]
        acc_n += tl.sum(dzn * zhat, 0)
        dzh = dzn * nw[None, :]
        proj = zhat * (tl.sum(dzh * zhat, 1) / C)[:, None]
        if CENTER:
            proj += (tl.sum(dzh, 1) / C)[:, None]
        dz = r[:, None] * (dzh - proj)
        tl.store(DZ + pos[:, None] * C + oc[None, :], dz.to(DZ.dtype.element_ty), mask=pm[:, None])
    base = PART + pid * (HP * C + C + HP)
    tl.store(base + oh[:, None] * C + oc[None, :], acc_w)
    tl.store(base + HP * C + oc, acc_n)
    tl.store(base + HP * C + C + oh, acc_b)


def _hp(h: int, fma: bool) -> int:
    return max(2 if fma else 16, 1 << (h - 1).bit_length())         # a power of two; tl.dot needs at least 16 rows / columns


def _mask_flat(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.reshape(-1)
    return mask if mask.dtype == torch.float32 and mask.is_contiguous() else mask.float().contiguous()


def _weight(lin_w: torch.Tensor, fma: bool, bf16: bool) -> torch.Tensor:
    """linear weight as the kernel loads it: float32 for the FMA path, the matmul operand of the mode otherwise"""
    return lin_w.float().contiguous() if fma else prep_weight(lin_w, MODES["bf16" if bf16 else "fp32"])


@torch.library.custom_op("physdock::pair_bias_fwd", mutates_args=())
def _pair_bias_fwd(z: torch.Tensor, mask: torch.Tensor, norm_w: torch.Tensor, lin_w: torch.Tensor, lin_b: Optional[torch.Tensor],
                   eps: float, neg_inf: float, bf16: bool, pad: bool, center: bool) -> torch.Tensor:
    """-> [H, N, N_out] buffer (N_out = N rounded up to 16 if pad else N), the caller slices [..., :N]"""
    N, H, C = z.shape[0], lin_w.shape[0], z.shape[-1]
    z = z.contiguous()
    n_out = (N + 15) // 16 * 16 if pad else N
    out = torch.empty(H, N, n_out, device=z.device, dtype=torch.bfloat16 if bf16 else torch.float32)
    NN = N * N
    fma, bp, warps, stages = _config(C, bf16)[:4]
    lb = lin_b if lin_b is not None else norm_w
    _fwd_kernel[(triton.cdiv(NN, bp),)](z, _mask_flat(mask), norm_w.contiguous(), _weight(lin_w, fma, bf16), lb.contiguous(), out, NN, N,
                                        out.stride(0), out.stride(1), eps, neg_inf, C=C, H=H, HP=_hp(H, fma), OUT_BF16=bf16,
                                        FMA=fma, MODE=MODES["bf16" if bf16 else "fp32"], BLOCK_P=bp, num_warps=warps,
                                        num_stages=stages, CENTER=center, HAS_B=lin_b is not None)
    return out


@_pair_bias_fwd.register_fake
def _(z, mask, norm_w, lin_w, lin_b, eps, neg_inf, bf16, pad, center):
    N, H = z.shape[0], lin_w.shape[0]
    return z.new_empty(H, N, (N + 15) // 16 * 16 if pad else N, dtype=torch.bfloat16 if bf16 else torch.float32)


@torch.library.custom_op("physdock::pair_bias_bwd", mutates_args=())
def _pair_bias_bwd(db: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, norm_w: torch.Tensor, lin_w: torch.Tensor, eps: float, bf16: bool,
                   center: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """db [H, N, N] (any strides) -> dz [N, N, C] (dtype of z), d norm_w [C], d lin_w [H, C], d lin_b [H] (float32)"""
    N, H, C = z.shape[0], lin_w.shape[0], z.shape[-1]
    z = z.contiguous()
    fma, _, _, _, bp, warps, stages, mult = _config(C, bf16)
    NN, hp = N * N, _hp(H, fma)
    ntiles = triton.cdiv(NN, bp)
    nprog = min(mult * num_sms(), ntiles)
    dz = torch.empty_like(z)
    part = torch.empty(nprog, hp * C + C + hp, device=z.device, dtype=torch.float32)
    _bwd_kernel[(nprog,)](z, _mask_flat(mask), norm_w.contiguous(), _weight(lin_w, fma, bf16), db, dz, part, NN, N, db.stride(0), db.stride(1),
                          db.stride(2), eps, ntiles, C=C, H=H, HP=hp, NPROG=nprog, FMA=fma, MODE=MODES["bf16" if bf16 else "fp32"],
                          BLOCK_P=bp, num_warps=warps, num_stages=stages, CENTER=center)
    tot = part.sum(0)
    return dz, tot[hp * C:hp * C + C].clone(), tot[:H * C].view(H, C).clone(), tot[hp * C + C:hp * C + C + H].clone()


@_pair_bias_bwd.register_fake
def _(db, z, mask, norm_w, lin_w, eps, bf16, center):
    return (z.new_empty(z.shape), z.new_empty(z.shape[-1], dtype=torch.float32), z.new_empty(lin_w.shape, dtype=torch.float32),
            z.new_empty(lin_w.shape[0], dtype=torch.float32))


def _setup_context(ctx, inputs, output):
    z, mask, norm_w, lin_w, lin_b, eps, neg_inf, bf16, pad, center = inputs
    ctx.save_for_backward(z, mask, norm_w, lin_w)
    ctx.eps, ctx.bf16, ctx.center, ctx.has_b = eps, bf16, center, lin_b is not None


def _backward(ctx, dout):
    z, mask, norm_w, lin_w = ctx.saved_tensors
    dz, dnw, dw, db = _pair_bias_bwd(dout, z, mask, norm_w, lin_w, ctx.eps, ctx.bf16, ctx.center)
    return dz, None, dnw, dw.to(lin_w.dtype), (db.to(lin_w.dtype) if ctx.has_b else None), None, None, None, None, None


_pair_bias_fwd.register_autograd(_backward, setup_context=_setup_context)


def pair_bias(z: torch.Tensor, mask: torch.Tensor, norm_weight: torch.Tensor, lin_weight: torch.Tensor, eps: float, inf: float,
              out_dtype: torch.dtype = torch.float32, pad: bool = False, lin_bias: Optional[torch.Tensor] = None,
              center: bool = False) -> torch.Tensor:
    """linear(norm(z)) + attention mask, see the module docstring (differentiable w.r.t. z, norm_weight, lin_weight and lin_bias)

    Args:
        z: [N, N, C] float32 (or bfloat16)
        mask: [N, N] pair mask (1 = keep)
        norm_weight: [C] RMSNorm scale, lin_weight: [H, C]
        eps: RMSNorm epsilon, inf: masked logits get -inf (e.g. 1e9)
        out_dtype: float32, or bfloat16 under autocast (selects the matmul precision, see the module docstring)
        pad: row stride of the result a multiple of 16 elements (the layout the Triton attention kernel loads fastest)
        lin_bias: [H] bias of the linear layer, or None
        center: LayerNorm (mean subtracted) instead of RMSNorm; norm_weight then is a vector of ones

    Returns:
        [H, N, N] out_dtype (a view of a [H, N, N_pad] buffer if pad)
    """
    N = z.shape[0]
    return _pair_bias_fwd(z, mask, norm_weight, lin_weight, lin_bias, eps, -inf, out_dtype == torch.bfloat16, pad, center)[..., :N]
