"""shared_bias_attention with n_valid (the kernels skip the padding at the end of a padded crop): the real positions give the results of
the full computation, values and gradients, also when the upstream gradient is non-zero at the padded rows (it is, downstream of padded
attention outputs) and the activations are large; the padded rows are exactly zero and everything stays finite. Needs a CUDA GPU."""
import pytest
import torch

from PhysDock.models.kernels import shared_bias_attention as K

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _run(dtype, B, H, N, real, use_nv, scale):
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(B, H, N, 32, device="cuda", generator=g).mul(scale).to(dtype).requires_grad_() for _ in range(3))
    valid = torch.arange(N, device="cuda") < real
    bias = torch.randn(H, N, N, device="cuda", generator=g).masked_fill(~(valid[:, None] & valid[None, :]), -1e9).to(dtype)
    bias = K.aligned_bias(bias).requires_grad_()
    nv = torch.tensor([real], dtype=torch.int32, device="cuda") if use_nv else None
    o = K.shared_bias_attention(q, k, v, bias, nv)
    go = torch.randn(o.shape, device="cuda", generator=g).to(o.dtype)            # non-zero at the padded rows as well
    o.backward(go)
    return o.detach().float(), q.grad.float(), k.grad.float(), v.grad.float(), bias.grad.float(), valid


@pytest.mark.parametrize("dtype,B,H,N,real,scale", [(torch.bfloat16, 8, 16, 384, 348, 6.0), (torch.bfloat16, 24, 4, 384, 130, 1.0),
                                                    (torch.float32, 4, 4, 1100, 700, 3.0), (torch.bfloat16, 1, 16, 384, 1, 6.0)])
def test_padding_is_skipped_without_changing_the_real_positions(dtype, B, H, N, real, scale):
    a = _run(dtype, B, H, N, real, True, scale)
    assert all(torch.isfinite(t).all() for t in a[:5]), "non-finite values with n_valid"
    valid = a[5]
    assert (a[0][:, :, ~valid] == 0).all(), "padded rows must be exactly zero"
    # reference: the full computation, whose padded rows are masked out of the loss (zero upstream gradient there)
    g = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(B, H, N, 32, device="cuda", generator=g).mul(scale).to(dtype).requires_grad_() for _ in range(3))
    bias = torch.randn(H, N, N, device="cuda", generator=g).masked_fill(~(valid[:, None] & valid[None, :]), -1e9).to(dtype)
    bias = K.aligned_bias(bias).requires_grad_()
    o = K.shared_bias_attention(q, k, v, bias)
    go = torch.randn(o.shape, device="cuda", generator=g).to(o.dtype) * valid[None, None, :, None].to(o.dtype)
    o.backward(go)
    rel = lambda x, y: float((x - y).norm() / y.norm().clamp_min(1e-30))
    tol = 3e-2 if dtype == torch.bfloat16 else 1e-5
    assert rel(a[0][:, :, valid], o.detach().float()[:, :, valid]) < tol
    # gradients: the kernel ignores the padded rows' upstream gradient, the reference has none there
    assert rel(a[1][:, :, valid], q.grad.float()[:, :, valid]) < tol
    assert rel(a[2][:, :, valid], k.grad.float()[:, :, valid]) < tol
    assert rel(a[3][:, :, valid], v.grad.float()[:, :, valid]) < tol
