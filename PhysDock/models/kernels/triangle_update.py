"""Triton kernels of PhysDock's TriangleUpdate (gated triangular multiplicative update), forward and backward.

For z [N, N, C] and the mask [N, N] (a, b: H = 32 channels):

    zn = RMSNorm_in(z)                                       (weight w_in, eps)
    a  = (Wqx zn + bqx) * sigmoid(Wq zn + bq) * mask         b = (Wkx zn + bkx) * sigmoid(Wk zn + bk) * mask
    t  = a b^T over j per channel:  outgoing  t[i, I] = sum_j a[i, j] b[I, j]
                                    incoming  t[p, q] = sum_j b[j, p] a[j, q]     (PhysDock: transpose, outgoing, transpose)
    out = (Wz RMSNorm_out(t) + bz) * sigmoid(Wg zn + bg)     (weight w_out over the H channels)

Everything stays in the original [N, N] frame (the incoming direction only changes the strides of the product), and
no [N, N, C]-sized intermediate is kept: the forward saves the per-position rstd, a, b and t (32 channels each,
channel-major [H, N*N]), the backward recomputes the normalised input and the gates from z.

Kernels: _in_proj (norm + 4 projections + gates), _bmm (the per-channel N x N products, shared by the forward product and
both backward products through its strides), _out_proj (norm + projection + output gate); backward _bwd_out (output side,
dt and the gate gradient) and _bwd_in (input side, dz); the weight gradients of the two large projections are two cuBLAS
products over the saved du / gate gradients, the small ones are reduced from per-program partial sums (no atomics).

Precision: "tf32" (float32 z, matmul inputs rounded to nearest TF32 like cuBLAS does for torch's TF32 matmuls, float32
accumulation, output float32) or "bf16" (bfloat16 matmuls with float32 accumulation, output bfloat16: what torch autocast
computes). Norms, sigmoids and products are float32 inside the kernels. The module-level function picks the precision from
the autocast state. The kernels run behind torch.library custom ops, so torch.compile treats them as opaque ops.
"""
from typing import Tuple

import torch
import triton
import triton.language as tl

_NUM_SMS = {}


def _nprog() -> int:
    """persistent programs of the position kernels (2 per SM)"""
    dev = torch.cuda.current_device()
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return 2 * _NUM_SMS[dev]


@triton.jit
def _cast_in(x, BF16: tl.constexpr):
    """x as a matmul input: bfloat16, or float32 rounded to the nearest TF32 value (add half an ulp, clear 13 bits)"""
    if BF16:
        return x.to(tl.bfloat16)
    else:
        return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def _load_w(ptr, rows, cols, stride_r, stride_c, BF16: tl.constexpr):
    """weight tile [len(rows), len(cols)] as a matmul operand"""
    w = tl.load(ptr + rows[:, None] * stride_r + cols[None, :] * stride_c)
    return _cast_in(w.to(tl.float32), BF16)


def _configs_pos():
    return [triton.Config({"BLOCK_P": p}, num_warps=w, num_stages=s) for p in (32, 64, 128) for w in (4, 8) for s in (1, 2)]


def _configs_bmm():
    out = []
    for bm, bn in ((64, 64), (128, 64), (64, 128), (128, 128), (32, 64), (64, 32)):
        for bk in (32, 64):
            for w in (4, 8):
                for s in (2, 3, 4):
                    if bm * bn >= 128 * 128 and w == 4 and bk == 64:
                        continue
                    if bm == bn == 64 and w == 8:
                        continue
                    out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s))
    return out


# ----------------------------------------------------------------------------------------------- forward
@triton.autotune(_configs_pos(), key=["NN", "C", "H", "BF16"])
@triton.jit
def _in_proj(Z, MASK, W_IN, W, BIAS, AB, R, NN, eps,
             C: tl.constexpr, H: tl.constexpr, BF16: tl.constexpr, BLOCK_P: tl.constexpr):
    """a, b -> AB [2, H, NN] (channel-major), rstd of the input norm -> R [NN]. W rows: (qx, q, kx, k) x H."""
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < NN
    oc, oh = tl.arange(0, C), tl.arange(0, H)
    z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(z * z, 1) / C + eps)
    zn = z * r[:, None] * tl.load(W_IN + oc).to(tl.float32)[None, :]
    znc = _cast_in(zn, BF16)
    m = tl.load(MASK + pos, mask=pm, other=0.0).to(tl.float32)
    u_qx = tl.dot(znc, _load_w(W, oc, 0 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 0 * H + oh)[None, :]
    u_q = tl.dot(znc, _load_w(W, oc, 1 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 1 * H + oh)[None, :]
    u_kx = tl.dot(znc, _load_w(W, oc, 2 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 2 * H + oh)[None, :]
    u_k = tl.dot(znc, _load_w(W, oc, 3 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 3 * H + oh)[None, :]
    a = u_qx * tl.sigmoid(u_q) * m[:, None]
    b = u_kx * tl.sigmoid(u_k) * m[:, None]
    dst = oh[None, :] * NN + pos[:, None]
    tl.store(AB + dst, _cast_in(a, BF16), mask=pm[:, None])
    tl.store(AB + H * NN + dst, _cast_in(b, BF16), mask=pm[:, None])
    tl.store(R + pos, r, mask=pm)


@triton.autotune(_configs_bmm(), key=["N", "X_T", "Y_T", "BF16"])
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
    tl.store(OUT + c.to(tl.int64) * sc + offs_m[:, None] * N + offs_n[None, :], acc, mask=m_in[:, None] & n_in[None, :])


@triton.autotune(_configs_pos(), key=["NN", "C", "H", "BF16"])
@triton.jit
def _out_proj(Z, R, T, W_IN, W_G, B_G, W_OUT, W_Z, B_Z, O, NN, eps,
              C: tl.constexpr, H: tl.constexpr, BF16: tl.constexpr, BLOCK_P: tl.constexpr):
    """O = (Wz RMSNorm_out(t) + bz) * sigmoid(Wg zn + bg); zn is recomputed from z and the saved rstd."""
    pos = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    pm = pos < NN
    oc, oh = tl.arange(0, C), tl.arange(0, H)
    t = tl.load(T + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32)
    rs = tl.math.rsqrt(tl.sum(t * t, 1) / H + eps)
    tn = t * rs[:, None] * tl.load(W_OUT + oh).to(tl.float32)[None, :]
    y = tl.dot(_cast_in(tn, BF16), _load_w(W_Z, oh, oc, 1, H, BF16), input_precision="tf32") + tl.load(B_Z + oc)[None, :]
    z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
    r = tl.load(R + pos, mask=pm, other=0.0)
    zn = z * r[:, None] * tl.load(W_IN + oc).to(tl.float32)[None, :]
    g = tl.sigmoid(tl.dot(_cast_in(zn, BF16), _load_w(W_G, oc, oc, 1, C, BF16), input_precision="tf32")
                   + tl.load(B_G + oc)[None, :])
    tl.store(O + pos[:, None] * C + oc[None, :], y * g, mask=pm[:, None])


# ----------------------------------------------------------------------------------------------- backward
@triton.autotune(_configs_pos(), key=["NN", "C", "H", "BF16"])
@triton.jit
def _bwd_out(DO, Z, R, T, W_IN, W_G, B_G, W_OUT, W_Z, B_Z, DT, DG, PWZ, PBZ, PWO, NN, eps,
             C: tl.constexpr, H: tl.constexpr, BF16: tl.constexpr, NPROG: tl.constexpr, BLOCK_P: tl.constexpr):
    """Output side: dt -> DT [H, NN]; the pre-sigmoid gate gradient -> DG [NN, C]; partial sums of dWz, dbz, dw_out per
    program (persistent programs loop over the position tiles)."""
    pid = tl.program_id(0)
    ntiles = tl.cdiv(NN, BLOCK_P)
    oc, oh = tl.arange(0, C), tl.arange(0, H)
    w_in = tl.load(W_IN + oc).to(tl.float32)
    w_out = tl.load(W_OUT + oh).to(tl.float32)
    b_g = tl.load(B_G + oc).to(tl.float32)
    b_z = tl.load(B_Z + oc).to(tl.float32)
    w_zt = _load_w(W_Z, oh, oc, 1, H, BF16)                    # [H, C]: (k, o)
    w_z = _load_w(W_Z, oc, oh, H, 1, BF16)                     # [C, H]: (o, k)
    acc_wz = tl.zeros([C, H], tl.float32)
    acc_bz = tl.zeros([C], tl.float32)
    acc_wo = tl.zeros([H], tl.float32)
    for tile in range(pid, ntiles, NPROG):
        pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < NN
        do = tl.load(DO + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        r = tl.load(R + pos, mask=pm, other=0.0)
        t = tl.load(T + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32)
        rs = tl.math.rsqrt(tl.sum(t * t, 1) / H + eps)
        xhat = t * rs[:, None]
        tnc = _cast_in(xhat * w_out[None, :], BF16)
        y = tl.dot(tnc, w_zt, input_precision="tf32") + b_z[None, :]
        zn = z * r[:, None] * w_in[None, :]
        g = tl.sigmoid(tl.dot(_cast_in(zn, BF16), _load_w(W_G, oc, oc, 1, C, BF16), input_precision="tf32") + b_g[None, :])
        dy = do * g
        dgate = do * y * g * (1.0 - g)
        tl.store(DG + pos[:, None] * C + oc[None, :], _cast_in(dgate, BF16), mask=pm[:, None])
        dyc = _cast_in(dy, BF16)
        dtn = tl.dot(dyc, w_z, input_precision="tf32")          # [P, H]
        gw = dtn * w_out[None, :]
        dt = rs[:, None] * (gw - xhat * (tl.sum(gw * xhat, 1) / H)[:, None])
        tl.store(DT + oh[None, :] * NN + pos[:, None], _cast_in(dt, BF16), mask=pm[:, None])
        acc_wz += tl.dot(tl.trans(dyc), tnc, input_precision="tf32")
        acc_bz += tl.sum(dy, 0)
        acc_wo += tl.sum(dtn * xhat, 0)
    tl.store(PWZ + pid * C * H + oc[:, None] * H + oh[None, :], acc_wz)
    tl.store(PBZ + pid * C + oc, acc_bz)
    tl.store(PWO + pid * H + oh, acc_wo)


@triton.autotune(_configs_pos(), key=["NN", "C", "H", "BF16"])
@triton.jit
def _bwd_in(DZ, Z, R, MASK, DAB, DG, W_IN, W, BIAS, W_G, DU, ZN, PWI, PBI, NN,
            C: tl.constexpr, H: tl.constexpr, BF16: tl.constexpr, NPROG: tl.constexpr, BLOCK_P: tl.constexpr):
    """Input side: dz; du (the gradient of the four pre-activations, (qx, q, kx, k) x H) -> DU [NN, 4H] and the normalised
    input -> ZN [NN, C] for the weight-gradient products; partial sums of dw_in and the four projection biases."""
    pid = tl.program_id(0)
    ntiles = tl.cdiv(NN, BLOCK_P)
    oc, oh = tl.arange(0, C), tl.arange(0, H)
    w_in = tl.load(W_IN + oc).to(tl.float32)
    acc_wi = tl.zeros([C], tl.float32)
    acc_b0 = tl.zeros([H], tl.float32)
    acc_b1 = tl.zeros([H], tl.float32)
    acc_b2 = tl.zeros([H], tl.float32)
    acc_b3 = tl.zeros([H], tl.float32)
    for tile in range(pid, ntiles, NPROG):
        pos = tile * BLOCK_P + tl.arange(0, BLOCK_P)
        pm = pos < NN
        z = tl.load(Z + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        r = tl.load(R + pos, mask=pm, other=0.0)
        m = tl.load(MASK + pos, mask=pm, other=0.0).to(tl.float32)
        zn = z * r[:, None] * w_in[None, :]
        znc = _cast_in(zn, BF16)
        u_qx = tl.dot(znc, _load_w(W, oc, 0 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 0 * H + oh)[None, :]
        u_q = tl.dot(znc, _load_w(W, oc, 1 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 1 * H + oh)[None, :]
        u_kx = tl.dot(znc, _load_w(W, oc, 2 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 2 * H + oh)[None, :]
        u_k = tl.dot(znc, _load_w(W, oc, 3 * H + oh, 1, C, BF16), input_precision="tf32") + tl.load(BIAS + 3 * H + oh)[None, :]
        da = tl.load(DAB + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32)
        db = tl.load(DAB + H * NN + oh[None, :] * NN + pos[:, None], mask=pm[:, None], other=0.0).to(tl.float32)
        s_q, s_k = tl.sigmoid(u_q), tl.sigmoid(u_k)
        d_qx = da * m[:, None] * s_q
        d_q = da * m[:, None] * u_qx * s_q * (1.0 - s_q)
        d_kx = db * m[:, None] * s_k
        d_k = db * m[:, None] * u_kx * s_k * (1.0 - s_k)
        dzn = tl.dot(_cast_in(d_qx, BF16), _load_w(W, 0 * H + oh, oc, C, 1, BF16), input_precision="tf32")
        dzn += tl.dot(_cast_in(d_q, BF16), _load_w(W, 1 * H + oh, oc, C, 1, BF16), input_precision="tf32")
        dzn += tl.dot(_cast_in(d_kx, BF16), _load_w(W, 2 * H + oh, oc, C, 1, BF16), input_precision="tf32")
        dzn += tl.dot(_cast_in(d_k, BF16), _load_w(W, 3 * H + oh, oc, C, 1, BF16), input_precision="tf32")
        dg = tl.load(DG + pos[:, None] * C + oc[None, :], mask=pm[:, None], other=0.0).to(tl.float32)
        dzn += tl.dot(_cast_in(dg, BF16), _load_w(W_G, oc, oc, C, 1, BF16), input_precision="tf32")
        gw = dzn * w_in[None, :]
        dz = r[:, None] * gw - z * (r * r * r)[:, None] * (tl.sum(gw * z, 1) / C)[:, None]
        tl.store(DZ + pos[:, None] * C + oc[None, :], dz, mask=pm[:, None])
        acc_wi += tl.sum(dzn * z * r[:, None], 0)
        acc_b0 += tl.sum(d_qx, 0)
        acc_b1 += tl.sum(d_q, 0)
        acc_b2 += tl.sum(d_kx, 0)
        acc_b3 += tl.sum(d_k, 0)
        du = DU + pos[:, None] * (4 * H) + oh[None, :]
        tl.store(du + 0 * H, _cast_in(d_qx, BF16), mask=pm[:, None])
        tl.store(du + 1 * H, _cast_in(d_q, BF16), mask=pm[:, None])
        tl.store(du + 2 * H, _cast_in(d_kx, BF16), mask=pm[:, None])
        tl.store(du + 3 * H, _cast_in(d_k, BF16), mask=pm[:, None])
        tl.store(ZN + pos[:, None] * C + oc[None, :], znc, mask=pm[:, None])
    tl.store(PWI + pid * C + oc, acc_wi)
    tl.store(PBI + pid * 4 * H + 0 * H + oh, acc_b0)
    tl.store(PBI + pid * 4 * H + 1 * H + oh, acc_b1)
    tl.store(PBI + pid * 4 * H + 2 * H + oh, acc_b2)
    tl.store(PBI + pid * 4 * H + 3 * H + oh, acc_b3)


# ----------------------------------------------------------------------------------------------- torch ops
def _bmm_launch(x, y, out, N, H, x_t, y_t, bf16):
    _bmm[lambda c: (triton.cdiv(N, c["BLOCK_M"]), triton.cdiv(N, c["BLOCK_N"]), H)](
        x, y, out, N, N * N, X_T=x_t, Y_T=y_t, BF16=bf16)


def _dims(z, w_out):
    N, N2, C = z.shape
    if N != N2:
        raise ValueError(f"z must be [N, N, C], got {tuple(z.shape)}")
    return N, C, w_out.shape[0]


@torch.library.custom_op("physdock::triangle_update_fwd", mutates_args=())
def _tu_fwd(z: torch.Tensor, mask: torch.Tensor, w_in: torch.Tensor, w_cat: torch.Tensor, b_cat: torch.Tensor,
            w_g: torch.Tensor, b_g: torch.Tensor, w_out: torch.Tensor, w_z: torch.Tensor, b_z: torch.Tensor,
            eps: float, outgoing: bool, precision: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> out [N, N, C], rstd [N*N], ab [2, H, N*N], t [H, N*N] (the last three are saved for the backward)"""
    N, C, H = _dims(z, w_out)
    NN, bf16 = N * N, precision == "bf16"
    cdt = torch.bfloat16 if bf16 else torch.float32
    z, mask = z.contiguous(), mask.reshape(-1).float().contiguous()
    ab = torch.empty(2, H, NN, device=z.device, dtype=cdt)
    r = torch.empty(NN, device=z.device, dtype=torch.float32)
    t = torch.empty(H, NN, device=z.device, dtype=cdt)
    out = torch.empty(N, N, C, device=z.device, dtype=cdt)
    _in_proj[lambda c: (triton.cdiv(NN, c["BLOCK_P"]),)](z, mask, w_in, w_cat, b_cat, ab, r, NN, eps, C=C, H=H, BF16=bf16)
    # outgoing: t[i, I] = sum_j a[i, j] b[I, j];  incoming: t[p, q] = sum_j b[j, p] a[j, q]
    if outgoing:
        _bmm_launch(ab[0], ab[1], t, N, H, False, False, bf16)
    else:
        _bmm_launch(ab[1], ab[0], t, N, H, True, True, bf16)
    _out_proj[lambda c: (triton.cdiv(NN, c["BLOCK_P"]),)](z, r, t, w_in, w_g, b_g, w_out, w_z, b_z, out, NN, eps, C=C, H=H, BF16=bf16)
    return out, r, ab, t


@_tu_fwd.register_fake
def _(z, mask, w_in, w_cat, b_cat, w_g, b_g, w_out, w_z, b_z, eps, outgoing, precision):
    N, C, H = _dims(z, w_out)
    cdt = torch.bfloat16 if precision == "bf16" else torch.float32
    return (z.new_empty(N, N, C, dtype=cdt), z.new_empty(N * N, dtype=torch.float32),
            z.new_empty(2, H, N * N, dtype=cdt), z.new_empty(H, N * N, dtype=cdt))


@torch.library.custom_op("physdock::triangle_update_bwd", mutates_args=())
def _tu_bwd(do: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, w_in: torch.Tensor, w_cat: torch.Tensor,
            b_cat: torch.Tensor, w_g: torch.Tensor, b_g: torch.Tensor, w_out: torch.Tensor, w_z: torch.Tensor,
            b_z: torch.Tensor, r: torch.Tensor, ab: torch.Tensor, t: torch.Tensor, eps: float, outgoing: bool,
            precision: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                     torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> dz, dw_in, dw_cat [4H, C], db_cat, dw_g, db_g, dw_out, dw_z, db_z"""
    N, C, H = _dims(z, w_out)
    NN, bf16, dev = N * N, precision == "bf16", z.device
    cdt = torch.bfloat16 if bf16 else torch.float32
    z, mask, do = z.contiguous(), mask.reshape(-1).float().contiguous(), do.contiguous()
    nprog = _nprog()
    dt = torch.empty(H, NN, device=dev, dtype=cdt)
    dg = torch.empty(NN, C, device=dev, dtype=cdt)
    p_wz = torch.empty(nprog, C, H, device=dev, dtype=torch.float32)
    p_bz = torch.empty(nprog, C, device=dev, dtype=torch.float32)
    p_wo = torch.empty(nprog, H, device=dev, dtype=torch.float32)
    _bwd_out[(nprog,)](do, z, r, t, w_in, w_g, b_g, w_out, w_z, b_z, dt, dg, p_wz, p_bz, p_wo, NN, eps,
                       C=C, H=H, BF16=bf16, NPROG=nprog)
    dab = torch.empty(2, H, NN, device=dev, dtype=cdt)
    a, b = ab[0], ab[1]
    if outgoing:        # da[i, j] = sum_I dt[i, I] b[I, j];  db[I, j] = sum_i dt[i, I] a[i, j]
        _bmm_launch(dt, b, dab[0], N, H, False, True, bf16)
        _bmm_launch(dt, a, dab[1], N, H, True, True, bf16)
    else:               # da[j, q] = sum_p b[j, p] dt[p, q];  db[j, p] = sum_q a[j, q] dt[p, q]
        _bmm_launch(b, dt, dab[0], N, H, False, True, bf16)
        _bmm_launch(a, dt, dab[1], N, H, False, False, bf16)
    dz = torch.empty_like(z)
    du = torch.empty(NN, 4 * H, device=dev, dtype=cdt)
    zn = torch.empty(NN, C, device=dev, dtype=cdt)
    p_wi = torch.empty(nprog, C, device=dev, dtype=torch.float32)
    p_bi = torch.empty(nprog, 4 * H, device=dev, dtype=torch.float32)
    _bwd_in[(nprog,)](dz, z, r, mask, dab, dg, w_in, w_cat, b_cat, w_g, du, zn, p_wi, p_bi, NN,
                      C=C, H=H, BF16=bf16, NPROG=nprog)
    dw_cat = (du.t() @ zn).float()
    dw_g = (dg.t() @ zn).float()
    return (dz, p_wi.sum(0), dw_cat, p_bi.sum(0), dw_g, dg.float().sum(0), p_wo.sum(0), p_wz.sum(0), p_bz.sum(0))


@_tu_bwd.register_fake
def _(do, z, mask, w_in, w_cat, b_cat, w_g, b_g, w_out, w_z, b_z, r, ab, t, eps, outgoing, precision):
    N, C, H = _dims(z, w_out)
    f = lambda *s: z.new_empty(*s, dtype=torch.float32)
    return (torch.empty_like(z), f(C), f(4 * H, C), f(4 * H), f(C, C), f(C), f(H), f(C, H), f(C))


def _setup_context(ctx, inputs, output):
    ctx.eps, ctx.outgoing, ctx.precision = inputs[-3:]
    ctx.save_for_backward(*inputs[:10], *output[1:])


def _backward(ctx, do, dr, dab, dt):
    saved = ctx.saved_tensors
    grads = _tu_bwd(do, *saved, ctx.eps, ctx.outgoing, ctx.precision)
    return (grads[0], None, *grads[1:], None, None, None)


_tu_fwd.register_autograd(_backward, setup_context=_setup_context)


def triangle_update(z: torch.Tensor, mask: torch.Tensor, norm_in_weight: torch.Tensor, linear_q, linear_qx, linear_k,
                    linear_kx, linear_g, norm_out_weight: torch.Tensor, linear_z, eps: float,
                    outgoing: bool) -> torch.Tensor:
    """PhysDock's TriangleUpdate in one pass of Triton kernels (see the module docstring).

    Args:
        z: [N, N, C] float32 (the pair representation)
        mask: [N, N] pair mask (1 / 0)
        norm_in_weight [C], linear_q / qx / k / kx (Linear C -> H), linear_g (Linear C -> C), norm_out_weight [H],
            linear_z (Linear H -> C): the parameters of the module
        eps: epsilon of both RMS norms
        outgoing: True for the outgoing (row) update, False for the incoming (column) update

    Returns:
        [N, N, C] update (the residual is added by the caller); bfloat16 under autocast, else the dtype of z
    """
    if z.dim() != 3:
        raise ValueError(f"z must be [N, N, C], got {tuple(z.shape)}")
    precision = "bf16" if (torch.is_autocast_enabled() or z.dtype == torch.bfloat16) else "tf32"
    w_cat = torch.cat([linear_qx.weight, linear_q.weight, linear_kx.weight, linear_k.weight])
    b_cat = torch.cat([linear_qx.bias, linear_q.bias, linear_kx.bias, linear_k.bias])
    return _tu_fwd(z, mask, norm_in_weight, w_cat, b_cat, linear_g.weight, linear_g.bias, norm_out_weight,
                   linear_z.weight, linear_z.bias, eps, outgoing, precision)[0]
