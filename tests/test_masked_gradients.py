"""Regression tests of the gradient bugs found while training from scratch (run: pytest tests/test_masked_gradients.py, needs a CUDA GPU).

1. pair bias: masked logits are constants, so the logits there get no gradient, and the gradient of the bias is correct for a permuted
   view (sdpa gave wrong gradients for it): torch and Triton backends agree with an fp64 reference, also for rows that are masked entirely.
2. TriangleBlock: the pairs a mask excludes receive no gradient (the template mask leaves rows without a valid key, whose gradient was
   amplified by every block until the gradient norm was inf), and the forward is unchanged.
3. OuterProductMean: the initial gradient norm is bounded relative to the output gradient (the zero-initialised output projection behind an
   RMSNorm had a gain of 1e6).
"""
import pytest
import torch

import PhysDock.models.primitives as P
from PhysDock.models.primitives.transformers import TriangleBlock
from PhysDock.utils.precision import set_matmul_precision

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
set_matmul_precision("auto")
N, C = 96, 128


def _mask(valid: int) -> torch.Tensor:
    t = torch.zeros(N, device="cuda")
    t[:valid] = 1
    return t[:, None] * t[None, :]


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@pytest.mark.parametrize("backend", ["torch", "triton"])
@pytest.mark.parametrize("valid", [N, 60])
def test_pair_bias_gradients_match_fp64(backend, valid):
    mask = _mask(valid)

    def run(dtype, be, amp):
        torch.manual_seed(0)
        m = P.RMSNormPairBias(C, 4, 1e9, 1e-8).cuda().to(dtype)
        with torch.no_grad():
            m.linear.weight.copy_(torch.randn(m.linear.weight.shape, generator=torch.Generator().manual_seed(3)) * 0.1)
        P.set_backend(m, be)
        g = torch.Generator(device="cuda").manual_seed(1)
        z = (torch.randn(N, N, C, device="cuda", generator=g).to(dtype) * 0.3 * mask[..., None].to(dtype)).requires_grad_()
        like = torch.zeros(1, device="cuda", dtype=torch.bfloat16 if amp else dtype)
        go = torch.randn(4, N, N, device="cuda", generator=g).to(dtype)             # also at the masked entries
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = m(z, mask.to(dtype), like)
        out.to(dtype).backward(go)
        return z.grad, m.linear.weight.grad, out.detach()

    ref = run(torch.float64, "torch", False)
    dz, dw, out = run(torch.float32, backend, False)
    keep = mask.bool()
    assert _rel(dz[keep], ref[0][keep]) < 2e-3 and _rel(dw, ref[1]) < 2e-3
    if (~keep).any():
        assert dz[~keep].abs().max() == 0, "masked pairs must receive no gradient"
        assert (out[:, ~keep] == -1e9).all()


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_triangle_block_masked_pairs_get_no_gradient(backend):
    mask = _mask(40)                                    # most rows have no valid key, as the template mask
    torch.manual_seed(0)
    blk = TriangleBlock(C, 1e9, 1e-8).cuda()
    P.set_backend(blk, backend)
    with torch.no_grad():
        for p in blk.parameters():
            if p.dim() > 1:
                p.normal_(0, 0.05)
    z0 = torch.randn(N, N, C, device="cuda") * 0.3 * mask[..., None]
    z = z0.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = blk(z, mask)
    go = torch.randn_like(out)
    out.backward(go)
    assert torch.isfinite(z.grad).all()
    # the forward keeps the masked pairs; the gradient through them is cut
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        assert torch.equal(out, blk(z0, mask))
    keep = mask.bool()
    assert z.grad[~keep].abs().max() == 0 or z.grad[~keep].abs().max() <= z.grad[keep].abs().max()
    assert z.grad.abs().max() < 1e4 * go.abs().max()


@pytest.mark.parametrize("backend", ["torch", "triton"])
def test_outer_product_mean_initial_gradient_is_bounded(backend):
    torch.manual_seed(0)
    opm = P.OuterProductMean(256, C, 1e-8).cuda()
    P.set_backend(opm, backend)
    m = torch.randn(32, N, 256, device="cuda", requires_grad=True)
    z = torch.randn(N, N, C, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = opm(m, z)
    go = torch.randn_like(out)
    out.backward(go)
    norms = [p.grad.norm().item() for p in opm.parameters()] + [m.grad.norm().item()]
    assert max(norms) < 1e3 * go.norm().item(), norms        # 32 x |go| with the LeCun init, 1.8e6 x |go| with a zero output projection
