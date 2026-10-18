"""Triton kernels of the gated triangular multiplicative update (PhysDock's TriangleMultiplication), forward and backward.

For z [N, N, C], the mask [N, N] and H = 32 product channels (weights: linear_in [4H + C, C] with the rows
(qx, q, kx, k, gate), linear_z [C, H], the norm weights w_in [C] and w_out [H]):

    zn = RMSNorm_in(z)                                   u = Wi zn + bi          (one projection for everything)
    a = u_qx * sigmoid(u_q) * mask      b = u_kx * sigmoid(u_k) * mask           g = sigmoid(u_g)
    t = a b^T over j per channel:  outgoing  t[i, I] = sum_j a[i, j] b[I, j]
                                   incoming  t[p, q] = sum_j b[j, p] a[j, q]     (PhysDock: transpose, outgoing, transpose)
    out = (Wz RMSNorm_out(t) + bz) * g

Everything stays in the original [N, N] frame (the incoming direction only changes the strides of the product).

Where the time goes and what was done about it (H20: ~74 TFLOPS TF32, ~148 bf16, but 4 TB/s): the update is compute bound.
The projections are 2*NN*C*(4H + C) = 4.3 GFLOP, the weight gradients and the input gradient another 4.3 GFLOP each, the
products 1.1 GFLOP (fwd) / 2.1 GFLOP (bwd). So the forward saves what the backward would have to recompute with a GEMM
(the activations u_qx, sigmoid(u_q), u_kx, sigmoid(u_k) and the gate g, 16 / 33 MB each: a write and a read cost less than
the 2.1 GFLOP of recomputing them) and the backward is: _bwd_out (output side) -> two products -> _bwd_du (the gradients of the four
product pre-activations) -> one K = 4H + C cuBLAS product for dzn -> _bwd_norm (input norm backward) -> one cuBLAS GEMM for all weight
gradients of the merged projection.

Kernels: _fwd_in (norm + merged projection + activations: K = C at once, three dots), _bmm (the per-channel N x N products, used by the
forward product and both backward products through its strides), _fwd_out (norm + projection + gate), _bwd_out, _bwd_du, _bwd_norm. The
small parameter gradients are reduced from per-program partial sums (persistent programs, no atomics). The launch configurations are
fixed (swept on an H20), and the launches skip Triton's argument binder (_launch): an update is 3 (forward) to 8 (backward) launches of
~100 us kernels, where the binder costs 30-70 us of CPU each.

Precision: "tf32" (float32 z; matmul inputs rounded to nearest TF32 like cuBLAS does for torch's TF32 matmuls, float32
accumulation, float32 output and saved tensors) or "bf16" (bfloat16 matmuls with float32 accumulation, bfloat16 output and
saved tensors: what torch autocast computes). Norms, sigmoids and products are float32 inside the kernels. The module-level
function picks the precision from the autocast state. Under torch.compile the kernels run behind torch.library custom ops
(opaque ops with fake implementations); in eager mode a plain autograd.Function calls them directly.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.compiler.compiler import CompiledKernel

from ._common import cast_in, num_sms


def _nprog() -> int:
    """persistent programs of the position kernels (2 per SM)"""
    return 2 * num_sms()


@triton.jit
def _load_w(ptr, rows, cols, stride_r, stride_c, BF16: tl.constexpr):
    """weight tile [len(rows), len(cols)] as a matmul operand"""
    return cast_in(tl.load(ptr + rows[:, None] * stride_r + cols[None, :] * stride_c).to(tl.float32), BF16)


# Launch configurations (block, num_warps, num_stages), found by autotuning once (N = 256 on an H20; the winners of the
# neighbouring configurations are within ~3% of each other, and the persistent kernels do not depend on N at all). Fixed
# configurations: no autotuning pass (minutes of compilation per shape) and no autotuner on the launch path.
_CONFIGS = {                    # float32 / TF32
    "_fwd_in": ({"BLOCK_P": 128}, 8, 2), "_fwd_out": ({"BLOCK_P": 32}, 4, 2),
    "_bwd_out": ({"BLOCK_P": 16}, 4, 2), "_bwd_du": ({"BLOCK_P": 64}, 8, 2), "_bwd_norm": ({"BLOCK_P": 32}, 4, 1),
    "_bmm": ({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, 4, 3),
}
_CONFIGS_BF16 = {               # bfloat16
    "_fwd_in": ({"BLOCK_P": 64}, 8, 3), "_fwd_out": ({"BLOCK_P": 64}, 4, 1),
    "_bwd_out": ({"BLOCK_P": 32}, 4, 2), "_bwd_du": ({"BLOCK_P": 64}, 8, 2), "_bwd_norm": ({"BLOCK_P": 64}, 4, 2),
    "_bmm": ({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, 4, 3),
}


def _part_layout(C: int, H: int):
    """offsets of the per-program partial sums of the small gradients: dWz [C, H], dbz [C], dw_out [H], dw_in [C] and
    db_in [4H + C] (the bias of linear_in: four projections of H, then the gate)"""
    offs, o = {}, 0
    for name, n in (("wz", C * H), ("bz", C), ("wo", H), ("wi", C), ("bi", 4 * H + C)):
        offs[name] = (o, n)
        o += n
    return offs, o


# ----------------------------------------------------------------------------------------------- launching
_LAUNCH = {}


def _launch(kernel, grid_fn, gdata, args, consts, dtypes, device):
    """Launch a kernel with its fixed configuration straight from the cached binary: Triton's own launch path (argument
    binder, kernel cache lookup) costs ~30-70 us of CPU per launch, as much as the kernel itself for the small kernels
    here, and an update launches 3 (forward) to 8 (backward) of them.
    gdata: the arguments of grid_fn (the binary is specialised on the divisibility of the sizes, so they key the cache),
    args: the runtime arguments in the order of the kernel signature, consts: its constexpr arguments, dtypes: those of the
    tensors (z, compute dtype; the binary is specialised on them)."""
    key = (kernel, gdata, dtypes, tuple(consts.values()))
    hit = _LAUNCH.get(key)
    if hit is None:
        block, warps, stages = (_CONFIGS_BF16 if dtypes[1] == torch.bfloat16 else _CONFIGS)[kernel.fn.__name__]
        ck = kernel.run(*args, **consts, **block, num_warps=warps, num_stages=stages, grid=None, warmup=True)
        ck.run                                           # loads the binary (CompiledKernel.__getattribute__)
        grid = tuple(grid_fn(block, *gdata)) + (1, 1)
        hit = _LAUNCH[key] = (ck, ck.run, ck.function, ck.packed_metadata, grid[:3])
    ck, run, function, metadata, grid = hit
    stream = torch._C._cuda_getCurrentRawStream(device)
    meta = None if CompiledKernel.launch_enter_hook is None else ck.launch_metadata(grid, stream, *args)
    run(grid[0], grid[1], grid[2], stream, function, metadata, meta, CompiledKernel.launch_enter_hook,
        CompiledKernel.launch_exit_hook, *args)


# ----------------------------------------------------------------------------------------------- forward
@triton.jit
def _fwd_in(Z, MASK, W_IN, W, BIAS, AB, US, G, R, NN, eps,
              C: tl.constexpr, H: tl.constexpr, BF16: tl.constexpr, SAVE: tl.constexpr, BLOCK_P: tl.constexpr):
    """_fwd_in with the whole contraction axis at once: three dots (K = C) whose outputs are [u_qx | u_kx], [u_q | u_k] and the
    gate, so the epilogue is elementwise and needs no column slicing (the chunked version issues 20 small dots per tile and runs
    at 41 TFLOPS in bfloat16, this one at 65). The same results up to the accumulation order."""
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < NN
    oc = tl.arange(0, C)
    o2 = tl.arange(0, 2 * H)
    z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(z * z, 1) / C + eps)
    zn = cast_in(z * r[:, None] * tl.load(W_IN + oc).to(tl.float32)[None, :], BF16)
    row1 = (o2 // H) * (2 * H) + o2 % H                  # rows of qx (0 .. H - 1) and kx (2H .. 3H - 1) in W
    row2 = row1 + H                                      # rows of q (H .. 2H - 1) and k (3H .. 4H - 1)
    w1 = cast_in(tl.load(W + row1[None, :] * C + oc[:, None]).to(tl.float32), BF16)               # [C, 2H]
    w2 = cast_in(tl.load(W + row2[None, :] * C + oc[:, None]).to(tl.float32), BF16)
    wg = cast_in(tl.load(W + (4 * H + oc)[None, :] * C + oc[:, None]).to(tl.float32), BF16)       # [C, C]
    p1 = tl.dot(zn, w1, input_precision="tf32") + tl.load(BIAS + row1)[None, :]                    # [u_qx | u_kx]
    p2 = tl.dot(zn, w2, input_precision="tf32") + tl.load(BIAS + row2)[None, :]                    # [u_q | u_k]
    s = tl.sigmoid(p2)
    g = tl.sigmoid(tl.dot(zn, wg, input_precision="tf32") + tl.load(BIAS + 4 * H + oc)[None, :])
    m = tl.load(MASK + pos, mask=pm, other=0.0).to(tl.float32)
    ab = cast_in(p1 * s * m[:, None], BF16)
    tl.store(AB + (o2 // H)[None, :] * (H * NN) + (o2 % H)[None, :] * NN + pos[:, None], ab, mask=pm[:, None])
    sdt = G.dtype.element_ty
    if SAVE:
        us = US + pos[:, None] * (4 * H) + row1[None, :]
        tl.store(us, p1.to(sdt), mask=pm[:, None])
        tl.store(us + H, s.to(sdt), mask=pm[:, None])
        tl.store(R + pos, r, mask=pm)
    tl.store(G + pos[:, None] * C + oc[None, :], g.to(sdt), mask=pm[:, None])


@triton.jit
def _bwd_out(DO, T, G, W_OUT, W_Z, B_Z, DT, DU, PART, NN, eps,
             C: tl.constexpr, H: tl.constexpr, BF16: tl.constexpr, NPROG: tl.constexpr,
             PWZ: tl.constexpr, PBZ: tl.constexpr, PWO: tl.constexpr, PBG: tl.constexpr, PSTRIDE: tl.constexpr,
             BLOCK_P: tl.constexpr):
    """_bwd_out with the whole C output channels at once (contiguous 256 B stores of the gate gradient, one dot per product)
    and, in bfloat16, the column sums of dy and dgate as ones-dots instead of cross-thread reductions (they cost a third of the
    chunked kernel). The sums are over the bfloat16-rounded gradients, as torch's autocast bias gradients are; in float32 the
    sums are over the float32 values. 0.158 ms instead of 0.256 ms at 384 tokens."""
    pid = tl.program_id(0)
    ntiles = tl.cdiv(NN, BLOCK_P)
    oh, oc = tl.arange(0, H), tl.arange(0, C)
    o16 = tl.arange(0, 16)
    w_out = tl.load(W_OUT + oh).to(tl.float32)
    wz_t = cast_in(tl.load(W_Z + oc[None, :] * H + oh[:, None]).to(tl.float32), BF16)        # [H, C] = Wz^T
    wz = cast_in(tl.load(W_Z + oc[:, None] * H + oh[None, :]).to(tl.float32), BF16)          # [C, H]
    bz_w = tl.load(B_Z + oc).to(tl.float32)
    acc_wo = tl.zeros([H], tl.float32)
    acc_wz = tl.zeros([C, H], tl.float32)
    if BF16:
        acc_bz = tl.zeros([16, C], tl.float32)
        acc_bg = tl.zeros([16, C], tl.float32)
        ones = tl.full([16, BLOCK_P], 1.0, tl.bfloat16)
    else:
        acc_bz1 = tl.zeros([C], tl.float32)
        acc_bg1 = tl.zeros([C], tl.float32)
    for tile in range(pid, ntiles, NPROG):
        pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < NN
        t = tl.load(T + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32)
        rs = tl.math.rsqrt(tl.sum(t * t, 1) / H + eps)
        xhat = t * rs[:, None]
        tnc = cast_in(xhat * w_out[None, :], BF16)
        y = tl.dot(tnc, wz_t, input_precision="tf32") + bz_w[None, :]                            # [BLOCK_P, C]
        g = tl.load(G + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        do = tl.load(DO + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        dgate = do * y * g * (1.0 - g)
        tl.store(DU + pos[:, None] * (4 * H + C) + 4 * H + oc[None, :], dgate.to(DU.dtype.element_ty), mask=pm[:, None])
        dy = do * g
        dyc = cast_in(dy, BF16)
        dtn = tl.dot(dyc, wz, input_precision="tf32")                                            # [BLOCK_P, H]
        acc_wz += tl.dot(tl.trans(dyc), tnc, input_precision="tf32")                              # [C, H]
        if BF16:
            acc_bz = tl.dot(ones, dyc, acc_bz)
            acc_bg = tl.dot(ones, dgate.to(tl.bfloat16), acc_bg)
        else:
            acc_bz1 += tl.sum(dy, 0)
            acc_bg1 += tl.sum(dgate, 0)
        gw = dtn * w_out[None, :]
        dt = rs[:, None] * (gw - xhat * (tl.sum(gw * xhat, 1) / H)[:, None])
        tl.store(DT + oh[None, :] * NN + pos[:, None], dt.to(DT.dtype.element_ty), mask=pm[:, None])
        acc_wo += tl.sum(dtn * xhat, 0)
    row = PART + pid * PSTRIDE
    tl.store(row + PWZ + oc[:, None] * H + oh[None, :], acc_wz)
    if BF16:
        sel = (o16 == 0)[:, None]
        tl.store(row + PBZ + oc, tl.sum(tl.where(sel, acc_bz, 0.0), 0))
        tl.store(row + PBG + oc, tl.sum(tl.where(sel, acc_bg, 0.0), 0))
    else:
        tl.store(row + PBZ + oc, acc_bz1)
        tl.store(row + PBG + oc, acc_bg1)
    tl.store(row + PWO + oh, acc_wo)


@triton.jit
def _bmm(X, Y, OUT, N, sc,
         X_T: tl.constexpr, Y_T: tl.constexpr, BF16: tl.constexpr,
         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """OUT[c, m, n] = sum_k X[c, m, k] Y[c, n, k] for the channel-major matrices X, Y, OUT [H, N, N] (channel stride sc).
    X_T: X holds element (m, k) at k * N + m instead of m * N + k; Y_T: Y holds (n, k) at k * N + n instead of n * N + k.
    The tiles are loaded along the contiguous axis and transposed in registers."""
    pid_m, pid_n, c = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_in, n_in = offs_m < N, offs_n < N
    xb, yb = X + c.to(tl.int64) * sc, Y + c.to(tl.int64) * sc
    acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k0 in range(0, N, BLOCK_K):
        kk = k0 + offs_k
        k_in = kk < N
        if X_T:
            xt = tl.load(xb + kk[:, None] * N + offs_m[None, :], mask=k_in[:, None] & m_in[None, :], other=0.0)
            x = tl.trans(xt)
        else:
            x = tl.load(xb + offs_m[:, None] * N + kk[None, :], mask=m_in[:, None] & k_in[None, :], other=0.0)
        if Y_T:
            y = tl.load(yb + kk[:, None] * N + offs_n[None, :], mask=k_in[:, None] & n_in[None, :], other=0.0)
        else:
            yt = tl.load(yb + offs_n[:, None] * N + kk[None, :], mask=n_in[:, None] & k_in[None, :], other=0.0)
            y = tl.trans(yt)
        acc = tl.dot(x, y, acc, input_precision="tf32")
    tl.store(OUT + c.to(tl.int64) * sc + offs_m[:, None] * N + offs_n[None, :], acc.to(OUT.dtype.element_ty),
             mask=m_in[:, None] & n_in[None, :])


@triton.jit
def _fwd_out(T, G, W_OUT, W_Z, B_Z, O, NN, eps,
             C: tl.constexpr, H: tl.constexpr, BF16: tl.constexpr, BLOCK_P: tl.constexpr):
    """O = (Wz RMSNorm_out(t) + bz) * g"""
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < NN
    oc, oh = tl.arange(0, C), tl.arange(0, H)
    t = tl.load(T + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32)
    rs = tl.math.rsqrt(tl.sum(t * t, 1) / H + eps)
    tn = t * rs[:, None] * tl.load(W_OUT + oh).to(tl.float32)[None, :]
    y = tl.dot(cast_in(tn, BF16), _load_w(W_Z, oh, oc, 1, H, BF16), input_precision="tf32") + tl.load(B_Z + oc)[None, :]
    g = tl.load(G + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    tl.store(O + pos[:, None] * C + oc[None, :], (y * g).to(O.dtype.element_ty), mask=pm[:, None])


# ----------------------------------------------------------------------------------------------- backward
@triton.jit
def _bwd_du(MASK, DAB, US, DU, PART, NN,
            C: tl.constexpr, H: tl.constexpr, NPROG: tl.constexpr, PBI: tl.constexpr, PSTRIDE: tl.constexpr,
            BLOCK_P: tl.constexpr):
    """The gradients of the four product pre-activations (qx, q, kx, k) from da, db and the saved activations -> the first
    4H columns of DU [NN, 4H + C] (the gate part was written by _bwd_out); partial sums of their bias gradients."""
    pid = tl.program_id(0)
    ntiles = tl.cdiv(NN, BLOCK_P)
    oh = tl.arange(0, H)
    acc_b0 = tl.zeros([H], tl.float32)
    acc_b1 = tl.zeros([H], tl.float32)
    acc_b2 = tl.zeros([H], tl.float32)
    acc_b3 = tl.zeros([H], tl.float32)
    for tile in range(pid, ntiles, NPROG):
        pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < NN
        m = tl.load(MASK + pos, mask=pm, other=0.0).to(tl.float32)
        us = US + pos[:, None] * (4 * H) + oh[None, :]
        u_qx = tl.load(us + 0 * H, mask=pm[:, None], other=0.0).to(tl.float32)
        s_q = tl.load(us + 1 * H, mask=pm[:, None], other=0.0).to(tl.float32)
        u_kx = tl.load(us + 2 * H, mask=pm[:, None], other=0.0).to(tl.float32)
        s_k = tl.load(us + 3 * H, mask=pm[:, None], other=0.0).to(tl.float32)
        da = tl.load(DAB + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32) * m[:, None]
        db = tl.load(DAB + H * NN + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32) * m[:, None]
        d_qx = da * s_q
        d_q = da * u_qx * s_q * (1.0 - s_q)
        d_kx = db * s_k
        d_k = db * u_kx * s_k * (1.0 - s_k)
        acc_b0 += tl.sum(d_qx, 0)
        acc_b1 += tl.sum(d_q, 0)
        acc_b2 += tl.sum(d_kx, 0)
        acc_b3 += tl.sum(d_k, 0)
        du = DU + pos[:, None] * (4 * H + C) + oh[None, :]
        sdt = DU.dtype.element_ty
        tl.store(du + 0 * H, d_qx.to(sdt), mask=pm[:, None])
        tl.store(du + 1 * H, d_q.to(sdt), mask=pm[:, None])
        tl.store(du + 2 * H, d_kx.to(sdt), mask=pm[:, None])
        tl.store(du + 3 * H, d_k.to(sdt), mask=pm[:, None])
    row = PART + pid * PSTRIDE
    tl.store(row + PBI + 0 * H + oh, acc_b0)
    tl.store(row + PBI + 1 * H + oh, acc_b1)
    tl.store(row + PBI + 2 * H + oh, acc_b2)
    tl.store(row + PBI + 3 * H + oh, acc_b3)


@triton.jit
def _bwd_norm(DZ, Z, R, DZN, W_IN, ZN, PART, NN,
              C: tl.constexpr, BF16: tl.constexpr, NPROG: tl.constexpr, PWI: tl.constexpr, PSTRIDE: tl.constexpr,
              H: tl.constexpr, BLOCK_P: tl.constexpr):
    """RMSNorm backward of the input norm: dz from dzn [NN, C] (the product of the gradient du with linear_in's weight),
    the normalised input -> ZN for the weight-gradient GEMM, partial sums of dw_in."""
    pid = tl.program_id(0)
    ntiles = tl.cdiv(NN, BLOCK_P)
    oc = tl.arange(0, C)
    w_in = tl.load(W_IN + oc).to(tl.float32)
    acc_wi = tl.zeros([C], tl.float32)
    for tile in range(pid, ntiles, NPROG):
        pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < NN
        r = tl.load(R + pos, mask=pm, other=0.0)
        z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        dzn = tl.load(DZN + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        gw = dzn * w_in[None, :]
        dz = r[:, None] * gw - z * (r * r * r)[:, None] * (tl.sum(gw * z, 1) / C)[:, None]
        tl.store(DZ + pos[:, None] * C + oc[None, :], dz, mask=pm[:, None])
        acc_wi += tl.sum(dzn * z * r[:, None], 0)
        tl.store(ZN + pos[:, None] * C + oc[None, :], cast_in(z * r[:, None] * w_in[None, :], BF16).to(ZN.dtype.element_ty),
                 mask=pm[:, None])
    tl.store(PART + pid * PSTRIDE + PWI + oc, acc_wi)


# ----------------------------------------------------------------------------------------------- torch ops
def _grid_pos(meta, nn):
    return (triton.cdiv(nn, meta["BLOCK_P"]),)


def _grid_fixed(meta, n, nn):
    return (n,)


def _grid_bmm(meta, n, h):
    return (triton.cdiv(n, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]), h)


def _bmm_launch(x, y, out, N, H, x_t, y_t, bf16):
    _launch(_bmm, _grid_bmm, (N, H), (x, y, out, N, N * N), dict(X_T=x_t, Y_T=y_t, BF16=bf16), (x.dtype, x.dtype), x.device.index)


def _dims(z, w_out):
    N, N2, C = z.shape
    if N != N2:
        raise ValueError(f"z must be [N, N, C], got {tuple(z.shape)}")
    return N, C, w_out.shape[0]


def _fwd_core(z, mask, w_in, w_all, b_all, w_out, w_z, b_z, eps, outgoing, precision, save):
    """-> out [N, N, C], rstd [N*N], ab [2, H, N*N], t [H, N*N], us [N*N, 4H], g [N*N, C] (all but out are saved for the
    backward; without `save` (inference) rstd and us are not computed and returned as empty tensors)"""
    N, C, H = _dims(z, w_out)
    NN, bf16 = N * N, precision == "bf16"
    cdt = torch.bfloat16 if bf16 else torch.float32
    z, mask = z.contiguous(), mask.reshape(-1).float().contiguous()
    dev = z.device
    ab = torch.empty(2, H, NN, device=dev, dtype=cdt)
    r = torch.empty(NN if save else 1, device=dev, dtype=torch.float32)
    us = torch.empty(NN if save else 1, 4 * H, device=dev, dtype=cdt)
    g = torch.empty(NN, C, device=dev, dtype=cdt)
    t = torch.empty(H, NN, device=dev, dtype=cdt)
    out = torch.empty(N, N, C, device=dev, dtype=cdt)
    _launch(_fwd_in, _grid_pos, (NN,), (z, mask, w_in, w_all, b_all, ab, us, g, r, NN, eps), dict(C=C, H=H, BF16=bf16, SAVE=save),
            (z.dtype, cdt), dev.index)
    # outgoing: t[i, I] = sum_j a[i, j] b[I, j];  incoming: t[p, q] = sum_j b[j, p] a[j, q]
    if outgoing:
        _bmm_launch(ab[0], ab[1], t, N, H, False, False, bf16)
    else:
        _bmm_launch(ab[1], ab[0], t, N, H, True, True, bf16)
    _launch(_fwd_out, _grid_pos, (NN,), (t, g, w_out, w_z, b_z, out, NN, eps), dict(C=C, H=H, BF16=bf16), (z.dtype, cdt), dev.index)
    return out, r, ab, t, us, g


def _fwd_impl(z: torch.Tensor, mask: torch.Tensor, w_in: torch.Tensor, w_all: torch.Tensor, b_all: torch.Tensor,
              w_out: torch.Tensor, w_z: torch.Tensor, b_z: torch.Tensor, eps: float, outgoing: bool, precision: str
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _fwd_core(z, mask, w_in, w_all, b_all, w_out, w_z, b_z, eps, outgoing, precision, True)


def _bwd_impl(do: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, w_in: torch.Tensor, w_all: torch.Tensor,
              b_all: torch.Tensor, w_out: torch.Tensor, w_z: torch.Tensor, b_z: torch.Tensor, r: torch.Tensor,
              ab: torch.Tensor, t: torch.Tensor, us: torch.Tensor, g: torch.Tensor, eps: float, outgoing: bool,
              precision: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> dz [N, N, C]; the small gradients in one float32 vector (layout _part_layout); the weight gradient of linear_in
    [4H + C, C]"""
    N, C, H = _dims(z, w_out)
    NN, bf16, dev = N * N, precision == "bf16", z.device
    cdt = torch.bfloat16 if bf16 else torch.float32
    z, mask, do = z.contiguous(), mask.reshape(-1).float().contiguous(), do.contiguous()
    nprog = _nprog()
    lay, pstride = _part_layout(C, H)
    part = torch.empty(nprog, pstride, device=dev, dtype=torch.float32)
    dt = torch.empty(H, NN, device=dev, dtype=cdt)
    du = torch.empty(NN, 4 * H + C, device=dev, dtype=cdt)           # [du of the four projections | gate gradient]
    _launch(_bwd_out, _grid_fixed, (nprog, NN), (do, t, g, w_out, w_z, b_z, dt, du, part, NN, eps),
            dict(C=C, H=H, BF16=bf16, NPROG=nprog, PWZ=lay["wz"][0], PBZ=lay["bz"][0], PWO=lay["wo"][0],
                 PBG=lay["bi"][0] + 4 * H, PSTRIDE=pstride), (z.dtype, cdt), dev.index)
    dab = torch.empty(2, H, NN, device=dev, dtype=cdt)
    a, b = ab[0], ab[1]
    if outgoing:        # da[i, j] = sum_I dt[i, I] b[I, j];  db[I, j] = sum_i dt[i, I] a[i, j]
        _bmm_launch(dt, b, dab[0], N, H, False, True, bf16)
        _bmm_launch(dt, a, dab[1], N, H, True, True, bf16)
    else:               # da[j, q] = sum_p b[j, p] dt[p, q];  db[j, p] = sum_q a[j, q] dt[p, q]
        _bmm_launch(b, dt, dab[0], N, H, False, True, bf16)
        _bmm_launch(a, dt, dab[1], N, H, False, False, bf16)
    _launch(_bwd_du, _grid_fixed, (nprog, NN), (mask, dab, us, du, part, NN),
            dict(C=C, H=H, NPROG=nprog, PBI=lay["bi"][0], PSTRIDE=pstride), (z.dtype, cdt), dev.index)
    dzn = du @ w_all.to(cdt)                                                 # [NN, C]: one K = 4H + C product (cuBLAS)
    dz = torch.empty_like(z)
    zn = torch.empty(NN, C, device=dev, dtype=cdt)
    _launch(_bwd_norm, _grid_fixed, (nprog, NN), (dz, z, r, dzn, w_in, zn, part, NN),
            dict(C=C, BF16=bf16, NPROG=nprog, PWI=lay["wi"][0], PSTRIDE=pstride, H=H), (z.dtype, cdt), dev.index)
    return dz, part.sum(0), (du.t() @ zn).float()


_tu_fwd = torch.library.custom_op("physdock::triangle_update_fwd", mutates_args=())(_fwd_impl)
_tu_bwd = torch.library.custom_op("physdock::triangle_update_bwd", mutates_args=())(_bwd_impl)


@_tu_fwd.register_fake
def _(z, mask, w_in, w_all, b_all, w_out, w_z, b_z, eps, outgoing, precision):
    N, C, H = _dims(z, w_out)
    cdt = torch.bfloat16 if precision == "bf16" else torch.float32
    return (z.new_empty(N, N, C, dtype=cdt), z.new_empty(N * N, dtype=torch.float32), z.new_empty(2, H, N * N, dtype=cdt),
            z.new_empty(H, N * N, dtype=cdt), z.new_empty(N * N, 4 * H, dtype=cdt), z.new_empty(N * N, C, dtype=cdt))


@_tu_bwd.register_fake
def _(do, z, mask, w_in, w_all, b_all, w_out, w_z, b_z, r, ab, t, us, g, eps, outgoing, precision):
    N, C, H = _dims(z, w_out)
    _, pstride = _part_layout(C, H)
    return z.new_empty(z.shape), z.new_empty(pstride, dtype=torch.float32), z.new_empty(4 * H + C, C, dtype=torch.float32)


def _unpack_grads(saved, small, mats):
    """gradients of (mask, w_in, w_all, b_all, w_out, w_z, b_z) in the order of the op inputs"""
    C, H = saved[0].shape[2], saved[5].shape[0]
    lay, _ = _part_layout(C, H)
    piece = lambda name: small[lay[name][0]: lay[name][0] + lay[name][1]]
    return (None, piece("wi"), mats, piece("bi"), piece("wo"), piece("wz").view(C, H), piece("bz"))


class _TriangleMultiplicationFunction(torch.autograd.Function):
    """Eager path. The saved tensors are intermediates of the forward, not outputs of the node: an autograd node that
    saves its own outputs forms a reference cycle that a non-reentrant checkpoint keeps alive for several steps."""

    @staticmethod
    def forward(ctx, z, mask, w_in, w_all, b_all, w_out, w_z, b_z, eps, outgoing, precision):
        out, r, ab, t, us, g = _fwd_impl(z, mask, w_in, w_all, b_all, w_out, w_z, b_z, eps, outgoing, precision)
        ctx.save_for_backward(z, mask, w_in, w_all, b_all, w_out, w_z, b_z, r, ab, t, us, g)
        ctx.conf = (eps, outgoing, precision)
        return out

    @staticmethod
    def backward(ctx, do):
        saved = ctx.saved_tensors
        dz, small, mats = _bwd_impl(do, *saved, *ctx.conf)
        return (dz, *_unpack_grads(saved, small, mats), None, None, None)


def _setup_context(ctx, inputs, output):
    ctx.eps, ctx.outgoing, ctx.precision = inputs[-3:]
    ctx.save_for_backward(*inputs[:8], *output[1:])


def _backward(ctx, do, *unused):
    saved = ctx.saved_tensors                   # unpack once: a non-reentrant checkpoint forbids unpacking twice
    dz, small, mats = _tu_bwd(do, *saved, ctx.eps, ctx.outgoing, ctx.precision)
    return (dz, *_unpack_grads(saved, small, mats), None, None, None)


_tu_fwd.register_autograd(_backward, setup_context=_setup_context)


def triangle_update(z: torch.Tensor, mask: torch.Tensor, norm_in_weight: torch.Tensor, w_in: torch.Tensor, b_in: torch.Tensor,
                    norm_out_weight: torch.Tensor, w_z: torch.Tensor, b_z: torch.Tensor, eps: float, outgoing: bool) -> torch.Tensor:
    """The gated triangular multiplicative update in Triton kernels (see the module docstring).

    Args:
        z: [N, N, C] float32 (the pair representation)
        mask: [N, N] pair mask (1 / 0)
        norm_in_weight [C]; w_in [4H + C, C], b_in [4H + C] (rows qx, q, kx, k, gate); norm_out_weight [H]; w_z [C, H], b_z [C]
        eps: epsilon of both RMS norms
        outgoing: True for the outgoing (row) update, False for the incoming (column) update

    Returns:
        [N, N, C] update (the residual is added by the caller); bfloat16 under autocast, else the dtype of z
    """
    if z.dim() != 3:
        raise ValueError(f"z must be [N, N, C], got {tuple(z.shape)}")
    precision = "bf16" if (torch.is_autocast_enabled() or z.dtype == torch.bfloat16) else "tf32"
    args = (z, mask, norm_in_weight, w_in, b_in, norm_out_weight, w_z, b_z, eps, outgoing, precision)
    if torch.compiler.is_compiling():           # torch.compile: opaque custom ops (AOTAutograd handles the saved tensors)
        return _tu_fwd(*args)[0]
    if torch.is_grad_enabled() and any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
        return _TriangleMultiplicationFunction.apply(*args)
    return _fwd_core(*args, False)[0]
