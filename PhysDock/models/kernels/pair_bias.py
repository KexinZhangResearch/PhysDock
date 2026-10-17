"""Triton kernels of the attention logit bias computed from a pair representation, forward and backward:

    bias[h, i, j] = sum_c W[h, c] * (norm_w[c] * z[i, j, c] * rsqrt(mean_c z[i, j, c]^2 + eps)) + (0 if mask[i, j] else neg_inf)

(PhysDock's AttentionWithPairBias / MSARowAttentionWithPairBias: linear_z(RMSNorm(z)) + attention mask). z [N, N, C],
W [H, C] (linear_z.weight), out [H, N, N].

Why a kernel: torch runs this as an RMSNorm (several passes over z), a matmul with 4-16 output columns, a permute and a mask
add, and the backward as an RMSNorm backward, two matmuls whose contraction axis is the N^2 positions (K = 4 million, 4-16
output columns: cutlass runs them at a few percent of the GPU) and a column reduction of 1.2 ms for the 2048^2 x 16 atom
pairs. Here z is read once in the forward and z + dbias once in the backward.

Precision, per mode the same as the torch expression it replaces (never lower): the normalisation always in float32, then
  "bf16"  (autocast): the matmul inputs rounded to bfloat16, float32 accumulation, bfloat16 output, the mask added in bfloat16
          (torch's autocast linear does exactly this),
  "fp32"  (everything else, matmul_mode): float32 FMAs, more accurate than torch's TF32 / float32 matmul,
  "tf32"  (available, not selected by matmul_mode): inputs rounded to the nearest TF32 value like cuBLAS.
Tensor-core modes for the 128-channel pair representation (K = 128); the 16-channel atom pairs (K = 16, 4 heads) are memory bound
and use per-head float32 multiply-reduce in every mode (at least as accurate as the torch path).
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

_NUM_SMS = {}


MODES = {"fp32": 0, "tf32": 1, "bf16": 2}


def _config(c: int, mode: int):
    """fixed launch configuration (found by sweeping tile size, warps and programs per SM on an H20):
    (FMA, forward block, warps, stages, backward block, warps, stages, programs per SM).
    Wide z (c >= 64, the 128-channel pair representation): tl.dot (float32 FMAs in "fp32" mode: small tiles; tensor cores in the
    other modes). Narrow z (c = 16, the atom pairs): per-head multiply-reduce, big tiles."""
    if c < 64:
        return (True, 128, 2, 2, 256, 2, 2, 4)
    return (False, 32, 2, 2, 16, 4, 2, 4) if mode == 0 else (False, 64, 4, 2, 32, 4, 3, 4)


def _nprog() -> int:
    dev = torch.cuda.current_device()
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NUM_SMS[dev]


def can_fuse(z: torch.Tensor, n_heads: int) -> bool:
    """the kernels apply: CUDA z with c a power of two in [16, 256] and a power-of-two number of heads in [1, 16]"""
    c = z.shape[-1]
    return (z.is_cuda and z.dtype in (torch.float32, torch.bfloat16) and 16 <= c <= 256 and c & (c - 1) == 0
            and 1 <= n_heads <= 16 and n_heads & (n_heads - 1) == 0)


@triton.jit
def _round_tf32(x):
    return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _dot(a, b, MODE: tl.constexpr):
    """a @ b with the matmul precision of MODE (0: float32 FMAs, 1: TF32 inputs rounded to nearest, 2: bfloat16 inputs), float32 accumulation"""
    if MODE == 0:
        return tl.dot(a, b, input_precision="ieee")
    elif MODE == 1:
        return tl.dot(_round_tf32(a), _round_tf32(b), input_precision="tf32")
    else:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))


@triton.jit
def _fwd_kernel(Z, MASK, NW, W, OUT, NN, N, stride_oh, stride_oi, eps, neg,
                C: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, OUT_BF16: tl.constexpr, FMA: tl.constexpr,
                MODE: tl.constexpr, BLOCK_P: tl.constexpr):
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < NN
    oc, oh = tl.arange(0, C), tl.arange(0, HP)
    z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(z * z, 1) / C + eps)
    zn = (z * r[:, None]) * tl.load(NW + oc).to(tl.float32)[None, :]
    m = tl.load(MASK + pos, mask=pm, other=1.0).to(tl.float32)
    add = tl.where(m == 0.0, neg, 0.0)
    row = pos // N
    dst = row * stride_oi + (pos - row * N)
    if FMA:
        for h in tl.static_range(H):
            lg = tl.sum(zn * tl.load(W + h * C + oc).to(tl.float32)[None, :], 1)
            if OUT_BF16:
                res = lg.to(tl.bfloat16) + add.to(tl.bfloat16)
            else:
                res = lg + add
            tl.store(OUT + h * stride_oh + dst, res, mask=pm)
    else:
        w = tl.load(W + oh[None, :] * C + oc[:, None], mask=(oh < H)[None, :], other=0.0).to(tl.float32)  # [C, HP]
        logit = _dot(zn, w, MODE)                                                                        # [BLOCK_P, HP]
        if OUT_BF16:
            res = logit.to(tl.bfloat16) + add.to(tl.bfloat16)[:, None]
        else:
            res = logit + add[:, None]
        tl.store(OUT + oh[None, :] * stride_oh + dst[:, None], res, mask=pm[:, None] & (oh < H)[None, :])


@triton.jit
def _bwd_kernel(Z, NW, W, DB, DZ, PART, NN, N, sdh, sdi, sdj, eps, ntiles,
                C: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, NPROG: tl.constexpr, FMA: tl.constexpr,
                MODE: tl.constexpr, BLOCK_P: tl.constexpr):
    """persistent programs over tiles of BLOCK_P positions; PART[pid] = [dW (HP x C, row-major) | dnorm_w (C)]"""
    pid = tl.program_id(0)
    oc, oh = tl.arange(0, C), tl.arange(0, HP)
    nw = tl.load(NW + oc).to(tl.float32)
    w = tl.load(W + oh[:, None] * C + oc[None, :], mask=(oh < H)[:, None], other=0.0).to(tl.float32)     # [HP, C]
    acc_w = tl.zeros([HP, C], tl.float32)
    acc_n = tl.zeros([C], tl.float32)
    for t in range(pid, ntiles, NPROG):
        pos = t * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < NN
        row = pos // N
        col = pos - row * N
        z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        r = tl.math.rsqrt(tl.sum(z * z, 1) / C + eps)
        zhat = z * r[:, None]
        zn = zhat * nw[None, :]
        if FMA:
            dzn = tl.zeros([BLOCK_P, C], tl.float32)
            for h in tl.static_range(H):
                db = tl.load(DB + h * sdh + row * sdi + col * sdj, mask=pm, other=0.0).to(tl.float32)
                dzn += db[:, None] * tl.load(W + h * C + oc).to(tl.float32)[None, :]
                acc_w += tl.where(oh == h, 1.0, 0.0)[:, None] * tl.sum(db[:, None] * zn, 0)[None, :]
        else:
            dbt = tl.load(DB + oh[:, None] * sdh + (row * sdi + col * sdj)[None, :], mask=(oh < H)[:, None] & pm[None, :],
                          other=0.0).to(tl.float32)                                                        # [HP, BLOCK_P]
            acc_w += _dot(dbt, zn, MODE)                                                                   # dW += db^T zn
            dzn = _dot(tl.trans(dbt), w, MODE)                                                             # [BLOCK_P, C]
        acc_n += tl.sum(dzn * zhat, 0)
        dzh = dzn * nw[None, :]
        dz = r[:, None] * (dzh - zhat * (tl.sum(dzh * zhat, 1) / C)[:, None])
        tl.store(DZ + pos[:, None] * C + oc[None, :], dz.to(DZ.dtype.element_ty), mask=pm[:, None])
    base = PART + pid * (HP * C + C)
    tl.store(base + oh[:, None] * C + oc[None, :], acc_w)
    tl.store(base + HP * C + oc, acc_n)


def _hp(h: int, fma: bool) -> int:
    return max(2, h) if fma else max(16, h)         # tl.dot needs at least 16 rows / columns


def _mask_flat(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.reshape(-1)
    return mask if mask.dtype == torch.float32 and mask.is_contiguous() else mask.float().contiguous()


@torch.library.custom_op("physdock::pair_bias_fwd", mutates_args=())
def _pair_bias_fwd(z: torch.Tensor, mask: torch.Tensor, norm_w: torch.Tensor, lin_w: torch.Tensor, eps: float,
                   neg_inf: float, out_bf16: bool, pad: bool, mode: int) -> torch.Tensor:
    """-> [H, N, N_out] buffer (N_out = N rounded up to 16 if pad else N), the caller slices [..., :N]"""
    N, H, C = z.shape[0], lin_w.shape[0], z.shape[-1]
    z = z.contiguous()
    n_out = (N + 15) // 16 * 16 if pad else N
    out = torch.empty(H, N, n_out, device=z.device, dtype=torch.bfloat16 if out_bf16 else torch.float32)
    NN = N * N
    fma, bp, warps, stages = _config(C, mode)[:4]
    _fwd_kernel[(triton.cdiv(NN, bp),)](z, _mask_flat(mask), norm_w.contiguous(), lin_w.contiguous(), out, NN, N,
                                        out.stride(0), out.stride(1), eps, neg_inf, C=C, H=H, HP=_hp(H, fma),
                                        OUT_BF16=out_bf16, FMA=fma, MODE=mode, BLOCK_P=bp, num_warps=warps, num_stages=stages)
    return out


@_pair_bias_fwd.register_fake
def _(z, mask, norm_w, lin_w, eps, neg_inf, out_bf16, pad, mode):
    N, H = z.shape[0], lin_w.shape[0]
    return z.new_empty(H, N, (N + 15) // 16 * 16 if pad else N, dtype=torch.bfloat16 if out_bf16 else torch.float32)


@torch.library.custom_op("physdock::pair_bias_bwd", mutates_args=())
def _pair_bias_bwd(db: torch.Tensor, z: torch.Tensor, norm_w: torch.Tensor, lin_w: torch.Tensor,
                   eps: float, mode: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """db [H, N, N] (any strides) -> dz [N, N, C] (dtype of z), d norm_w [C], d lin_w [H, C] (float32)"""
    N, H, C = z.shape[0], lin_w.shape[0], z.shape[-1]
    z = z.contiguous()
    fma, _, _, _, bp, warps, stages, mult = _config(C, mode)
    NN, hp = N * N, _hp(H, fma)
    ntiles = triton.cdiv(NN, bp)
    nprog = min(mult * _nprog(), ntiles)
    dz = torch.empty_like(z)
    part = torch.empty(nprog, hp * C + C, device=z.device, dtype=torch.float32)
    _bwd_kernel[(nprog,)](z, norm_w.contiguous(), lin_w.contiguous(), db, dz, part, NN, N, db.stride(0), db.stride(1),
                          db.stride(2), eps, ntiles, C=C, H=H, HP=hp, NPROG=nprog, FMA=fma, MODE=mode, BLOCK_P=bp, num_warps=warps,
                          num_stages=stages)
    tot = part.sum(0)
    return dz, tot[hp * C:].clone(), tot[:H * C].view(H, C).clone()


@_pair_bias_bwd.register_fake
def _(db, z, norm_w, lin_w, eps, mode):
    return z.new_empty(z.shape), z.new_empty(z.shape[-1], dtype=torch.float32), z.new_empty(lin_w.shape, dtype=torch.float32)


def _setup_context(ctx, inputs, output):
    z, mask, norm_w, lin_w, eps, neg_inf, out_bf16, pad, mode = inputs
    ctx.save_for_backward(z, norm_w, lin_w)
    ctx.eps, ctx.mode = eps, mode


def _backward(ctx, dout):
    z, norm_w, lin_w = ctx.saved_tensors
    dz, dnw, dw = _pair_bias_bwd(dout, z, norm_w, lin_w, ctx.eps, ctx.mode)
    return dz, None, dnw, dw.to(lin_w.dtype), None, None, None, None, None


_pair_bias_fwd.register_autograd(_backward, setup_context=_setup_context)


def matmul_mode(like: torch.Tensor) -> str:
    """the precision to use for a result in the dtype of `like`: "bf16" under autocast (like is bfloat16: torch's own
    linear_z is a bfloat16 matmul there, same rounding points), else "fp32" (float32 FMAs: never less accurate than torch's
    float32 / TF32 matmul, and these paths are not hot)"""
    return "bf16" if like.dtype == torch.bfloat16 else "fp32"


def pair_bias(z: torch.Tensor, mask: torch.Tensor, norm_weight: torch.Tensor, lin_weight: torch.Tensor, eps: float,
              inf: float, out_dtype: torch.dtype = torch.float32, pad: bool = False, precision: str = "fp32") -> torch.Tensor:
    """linear_z(RMSNorm(z)) + attention mask, see the module docstring (differentiable w.r.t. z, norm_weight and lin_weight)

    Args:
        z: [N, N, C] float32 (or bfloat16)
        mask: [N, N] pair mask (1 = keep)
        norm_weight: [C] RMSNorm scale, lin_weight: [H, C] (linear_z.weight)
        eps: RMSNorm epsilon, inf: masked logits get -inf (e.g. 1e9)
        out_dtype: float32 or bfloat16 (what torch gives under autocast)
        pad: row stride of the result a multiple of 16 elements (the layout the Triton attention kernel loads fastest)
        precision: "fp32", "tf32" or "bf16": the matmul precision (see the module docstring; matmul_mode gives torch's)

    Returns:
        [H, N, N] out_dtype (a view of a [H, N, N_pad] buffer if pad)
    """
    N = z.shape[0]
    return _pair_bias_fwd(z, mask, norm_weight, lin_weight, eps, -inf, out_dtype == torch.bfloat16, pad, MODES[precision])[..., :N]
