"""Attention layers. Every head has 32 channels; masks become additive -inf biases. Pair masks are symmetric."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import AdaLayerNormZero, Linear, RMSNorm
from PhysDock.utils.tensor_utils import gen_attn_mask


def _heads(x: torch.Tensor) -> torch.Tensor:
    """[..., N, H * 32] -> [..., H, N, 32]"""
    return x.reshape(*x.shape[:-1], -1, 32).transpose(-2, -3)


def _merge(x: torch.Tensor) -> torch.Tensor:
    """[..., H, N, 32] -> [..., N, H * 32]"""
    x = x.transpose(-2, -3)
    return x.reshape(*x.shape[:-2], -1)


ATTENTION_BACKENDS = ("sdpa", "triton")


def _shared_bias_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                           backend: str) -> torch.Tensor:
    """softmax(q k^T / sqrt(32) + bias) v for q, k, v [B, H, N, 32] and a logit bias [H, N, N] shared by the B entries
    (the noisy samples of the DiT, the rows of triangle attention). "sdpa": torch's fused attention; "triton":
    PhysDock.models.kernels.shared_bias_attention (TF32 for float32, bf16 for bfloat16 inputs)."""
    if backend == "triton":
        from PhysDock.models.kernels import shared_bias_attention
        return shared_bias_attention(q, k, v, bias)
    return F.scaled_dot_product_attention(q, k, v, bias[None])


def set_triangle_update_backend(model: nn.Module, backend: str) -> None:
    """Implementation of the triangle multiplicative updates of model: "torch" or "triton"
    (PhysDock.models.kernels.triangle_update; TF32 for float32, bf16 matmuls under autocast)."""
    if backend not in ("torch", "triton"):
        raise ValueError(f"triangle update backend must be torch or triton, got {backend!r}")
    for module in model.modules():
        if isinstance(module, TriangleUpdate):
            module.backend = backend


def set_attention_backend(model: nn.Module, backend: str) -> None:
    """Attention implementation of the triangle attention and DiT attention layers of model ("sdpa" or "triton")."""
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(f"attention backend must be one of {ATTENTION_BACKENDS}, got {backend!r}")
    for module in model.modules():
        if isinstance(module, (TriangleAttention, DiTAttention)):
            module.backend = backend


class AttentionWithPairBias(nn.Module):
    """Gated self-attention over N positions with a per-head logit bias from the pair representation."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm_s = RMSNorm(c_s, eps)
        self.norm_z = RMSNorm(c_z, eps)
        self.linear_z = Linear(c_z, c_s // 32, bias=False)
        self.linear_q = Linear(c_s, c_s, bias=False)
        self.linear_k = Linear(c_s, c_s, bias=False)
        self.linear_v = Linear(c_s, c_s, bias=False)
        self.linear_g = Linear(c_s, c_s)
        self.linear_o = Linear(c_s, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [N, c_s]
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, c_s] update
        """
        s = self.norm_s(s)
        q, k, v = _heads(self.linear_q(s)), _heads(self.linear_k(s)), _heads(self.linear_v(s))
        bias = self.linear_z(self.norm_z(z)).permute(2, 0, 1) + gen_attn_mask(z_mask.type_as(q), -self.inf)[None]
        o = F.scaled_dot_product_attention(q, k, v, bias)
        return self.linear_o(_merge(o)) * self.linear_g(s)


class MSARowAttentionWithPairBias(nn.Module):
    """Gated attention along the tokens of every MSA row, with a logit bias from the pair representation."""

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm_m = RMSNorm(c_m, eps)
        self.norm_z = RMSNorm(c_z, eps)
        self.linear_z = Linear(c_z, c_m // 32, bias=False)
        self.linear_q = Linear(c_m, c_m, bias=False)
        self.linear_k = Linear(c_m, c_m, bias=False)
        self.linear_v = Linear(c_m, c_m, bias=False)
        self.linear_g = Linear(c_m, c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: [N_msa, N_token, c_m]
            z: [N_token, N_token, c_z]
            z_mask: [N_token, N_token]

        Returns:
            [N_msa, N_token, c_m] update
        """
        m = self.norm_m(m)
        q, k, v = _heads(self.linear_q(m)), _heads(self.linear_k(m)), _heads(self.linear_v(m))
        bias = self.linear_z(self.norm_z(z)).permute(2, 0, 1)[None] + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None]
        o = F.scaled_dot_product_attention(q, k, v, bias)
        return self.linear_o(_merge(o)) * self.linear_g(m)


class MSAColumnAttention(nn.Module):
    """Gated attention along the MSA rows of every token (no mask, no bias)."""

    def __init__(self, c_m: int, inf: float, eps: float) -> None:
        super().__init__()
        self.norm_m = RMSNorm(c_m, eps)
        self.linear_q = Linear(c_m, c_m, bias=False)
        self.linear_k = Linear(c_m, c_m, bias=False)
        self.linear_v = Linear(c_m, c_m, bias=False)
        self.linear_g = Linear(c_m, c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """m: [N_msa, N_token, c_m] -> update [N_msa, N_token, c_m]"""
        m = self.norm_m(m.transpose(-2, -3))
        q, k, v = _heads(self.linear_q(m)), _heads(self.linear_k(m)), _heads(self.linear_v(m))
        o = F.scaled_dot_product_attention(q, k, v)
        return (self.linear_o(_merge(o)) * self.linear_g(m)).transpose(-2, -3)


# Row order of TriangleUpdate.linear_in: the four projections of the two 32-channel products (value, gate) x (q, k)
# and the output gate. Checkpoints of the original layout (one Linear per block of rows) are converted on loading.
_TRIANGLE_UPDATE_LEGACY_LAYERS = ("linear_qx", "linear_q", "linear_kx", "linear_k", "linear_g")


def upgrade_triangle_update_keys(state_dict: dict, prefix: str = "") -> dict:
    """In place: replace the parameters linear_qx / linear_q / linear_kx / linear_k / linear_g of every TriangleUpdate
    below `prefix` by the single linear_in (rows in that order). Other keys, and state dicts that are already in the new
    layout, are left alone. Returns state_dict."""
    for key in [k for k in state_dict if k.startswith(prefix) and k.endswith("linear_qx.weight")]:
        base = key[:-len("linear_qx.weight")]
        for kind in ("weight", "bias"):
            parts = [base + layer + "." + kind for layer in _TRIANGLE_UPDATE_LEGACY_LAYERS]
            if all(k in state_dict for k in parts):
                state_dict[base + "linear_in." + kind] = torch.cat([state_dict.pop(k) for k in parts])
    return state_dict


class TriangleUpdate(nn.Module):
    """Gated triangular multiplicative update out[i, k] = sum_j q[i, j] * k[k, j]; transpose=True for incoming edges.

    All projections of the normalised input are one Linear (linear_in, C -> 4 * 32 + C): a = (qx * sigmoid(q)) and
    b = (kx * sigmoid(k)) (32 channels each, masked) feed the product, the last C rows are the output gate."""

    def __init__(self, c_z: int, eps: float, transpose: bool = False) -> None:
        super().__init__()
        self.transpose = transpose
        self.norm_in = RMSNorm(c_z, eps)
        self.norm_out = RMSNorm(32, eps)
        self.linear_in = Linear(c_z, 4 * 32 + c_z, init="final")
        self.linear_z = Linear(32, c_z, init="final")
        with torch.no_grad():                       # the initial values of the original layout, drawn in its order
            for rows in ((32, 64), (0, 32), (96, 128), (64, 96)):               # q, qx, k, kx
                self.linear_in.weight[rows[0]:rows[1]] = Linear(c_z, 32).weight
            self.linear_in.bias[4 * 32:] = 1.0                                  # gate: zero weight, bias 1
        self.backend = "torch"                          # see set_triangle_update_backend
        self._register_load_state_dict_pre_hook(self._upgrade_legacy_keys)

    @staticmethod
    def _upgrade_legacy_keys(state_dict, prefix, *args) -> None:
        upgrade_triangle_update_keys(state_dict, prefix)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] update
        """
        if self.backend == "triton":
            from PhysDock.models.kernels.triangle_update import triangle_update
            return triangle_update(z, z_mask, self.norm_in.weight, self.linear_in, self.norm_out.weight, self.linear_z,
                                   self.norm_in.eps, outgoing=not self.transpose)
        if self.transpose:
            z = z.transpose(-2, -3)
        z = self.norm_in(z)
        qx, q, kx, k, g = self.linear_in(z).split([32, 32, 32, 32, z.shape[-1]], dim=-1)
        q = qx * torch.sigmoid(q) * z_mask[..., None]
        k = kx * torch.sigmoid(k) * z_mask[..., None]
        o = self.linear_z(self.norm_out(torch.einsum("...ijc,...Ijc->...iIc", q, k))) * torch.sigmoid(g)
        if self.transpose:
            o = o.transpose(-2, -3)
        return o


class TriangleAttention(nn.Module):
    """Gated attention along the rows of z with a bias from z; transpose=True attends along the columns."""

    def __init__(self, c_z: int, inf: float, eps: float, transpose: bool = False) -> None:
        super().__init__()
        self.inf = inf
        self.transpose = transpose
        self.norm = RMSNorm(c_z, eps=eps)
        self.linear_q = Linear(c_z, c_z, bias=False)
        self.linear_k = Linear(c_z, c_z, bias=False)
        self.linear_v = Linear(c_z, c_z, bias=False)
        self.linear_z = Linear(c_z, c_z // 32, bias=False)
        self.linear_g = Linear(c_z, c_z)
        self.linear_o = Linear(c_z, c_z)
        self.backend = "sdpa"                           # see set_attention_backend

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] update
        """
        if self.transpose:
            z = z.transpose(-2, -3)
        z = self.norm(z)
        q, k, v = _heads(self.linear_q(z)), _heads(self.linear_k(z)), _heads(self.linear_v(z))
        bias = self.linear_z(z).permute(2, 0, 1) + gen_attn_mask(z_mask.type_as(q), -self.inf)[None]       # [H, N, N]
        o = self.linear_o(_merge(_shared_bias_attention(q, k, v, bias, self.backend))) * self.linear_g(z)
        if self.transpose:
            o = o.transpose(-2, -3)
        return o


class DiTAttention(nn.Module):
    """DiT self-attention: AdaLN-Zero input, RMS-normalised q/k, pair bias shared by all samples, gated output.

    The pair bias linear_z(norm_z(z)) + mask is computed by DiT.pair_bias (once per structure for all blocks)."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.inf = inf
        self.norm_s = AdaLayerNormZero(c_s, eps)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_q = Linear(c_s, c_s, bias=False)
        self.linear_k = Linear(c_s, c_s, bias=False)
        self.linear_v = Linear(c_s, c_s, bias=False)
        self.linear_z = Linear(c_z, c_s // 32, bias=False)
        self.norm_q = RMSNorm(32, eps)
        self.norm_k = RMSNorm(32, eps)
        self.linear_o = Linear(c_s, c_s)
        self.backend = "sdpa"                           # see set_attention_backend

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """q, k, v [B, H, N, 32], bias [H, N, N] -> [B, H, N, 32]"""
        return _shared_bias_attention(q, k, v, bias, self.backend)

    def forward(self, x: torch.Tensor, bias: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, c_s]
            bias: [H, N, N] contiguous logit bias (pair bias + mask, see DiT.pair_bias)
            t: [B, 256] timestep embedding

        Returns:
            [B, N, c_s] update
        """
        q, k, v, gate = self.project(x, t)
        return self.output(self.attend(q, k, v, bias), gate)

    def project(self, x: torch.Tensor, t: torch.Tensor):
        """Attention inputs: q, k, v [B, H, N, 32] and the output gate [B, 1, c_s]"""
        x, gate = self.norm_s(x, t)
        q = self.norm_q(_heads(self.linear_q(x)))
        k = self.norm_k(_heads(self.linear_k(x)))
        return q, k, _heads(self.linear_v(x)), gate

    def output(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Attention output o [B, H, N, 32] -> gated update [B, N, c_s]"""
        return self.linear_o(_merge(o)) * gate
