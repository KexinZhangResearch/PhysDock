import torch
import torch.nn as nn
import torch.nn.functional as F

from .rms_norm import RMSNorm
from .linear import Linear
from .layer_norm import LayerNorm
from .adaptive_layer_norm_zero import AdaLayerNormZero
from PhysDock.utils.tensor_utils import gen_attn_mask


class AttentionWithPairBias(nn.Module):
    def __init__(self, c_s, c_z, inf, eps):
        super().__init__()
        assert int(c_s % 32) == 0
        self.c_s = c_s
        self.c_hidden = 32
        self.no_heads = int(self.c_s / 32)
        self.inf = inf
        self.eps = eps

        self.norm_s = RMSNorm(c_s, eps)
        self.norm_z = RMSNorm(c_z, eps)

        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.linear_q = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_g = Linear(c_s, c_s)
        self.linear_o = Linear(c_s, c_s)

    def forward(self, s, z, z_mask):
        S, _ = s.shape
        H, D = self.no_heads, self.c_hidden

        s_norm = self.norm_s(s)
        z_norm = self.norm_z(z)

        q = self.linear_q(s_norm).reshape([S, H, D]).transpose(-2, -3)
        k = self.linear_k(s_norm).reshape([S, H, D]).transpose(-2, -3)
        v = self.linear_v(s_norm).reshape([S, H, D]).transpose(-2, -3)
        g = self.linear_g(s_norm)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_bias, dropout_p=0, scale=None).transpose(-2, -3)

        o = attn_out.reshape([S, -1])
        o = self.linear_o(o) * g

        return o.float()


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c_m, c_z, inf, eps):
        super().__init__()
        assert int(c_m % 32) == 0
        self.c_m = c_m
        self.c_hidden = 32
        self.no_heads = int(self.c_m / 32)
        self.inf = inf
        self.eps = eps

        self.norm_m = RMSNorm(c_m, eps)
        self.norm_z = RMSNorm(c_z, eps)

        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.linear_q = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_g = Linear(c_m, c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m, z, z_mask):
        B, S, _ = m.shape
        H, D = self.no_heads, self.c_hidden

        m_norm = self.norm_m(m)
        z_norm = self.norm_z(z)

        q = self.linear_q(m_norm).reshape([B, S, H, D]).transpose(-2, -3)
        k = self.linear_k(m_norm).reshape([B, S, H, D]).transpose(-2, -3)
        v = self.linear_v(m_norm).reshape([B, S, H, D]).transpose(-2, -3)
        g = self.linear_g(m_norm)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])[None]
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_bias, dropout_p=0, scale=None).transpose(-2, -3)

        o = attn_out.reshape([B, S, -1])
        o = self.linear_o(o) * g

        return o.float()


class MSAColumnAttention(nn.Module):
    def __init__(self, c_m, inf, eps):
        super().__init__()
        self.c_hidden = 32
        assert int(c_m % self.c_hidden) == 0
        self.c_m = c_m
        self.no_heads = int(self.c_m / self.c_hidden)
        self.inf = inf
        self.eps = eps

        self.norm_m = RMSNorm(c_m, eps)
        self.linear_q = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_m, self.no_heads * self.c_hidden, bias=False)
        self.linear_g = Linear(c_m, c_m)
        self.linear_o = Linear(c_m, c_m)

    def forward(self, m):
        m = m.transpose(-2, -3)
        B, S, _ = m.shape
        H, D = self.no_heads, self.c_hidden

        m_norm = self.norm_m(m)

        q = self.linear_q(m_norm).reshape([B, S, H, D]).transpose(-2, -3)
        k = self.linear_k(m_norm).reshape([B, S, H, D]).transpose(-2, -3)
        v = self.linear_v(m_norm).reshape([B, S, H, D]).transpose(-2, -3)
        g = self.linear_g(m_norm)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0, scale=None).transpose(-2, -3)

        o = attn_out.reshape([B, S, -1])
        o = self.linear_o(o) * g

        o = o.transpose(-2, -3)
        return o.float()


class TriangleUpdate(nn.Module):
    def __init__(self, c_z, eps, transpose=False):
        super().__init__()
        self.c_hidden = 32
        self.c_z = c_z
        self.eps = eps
        self.transpose = transpose

        self.norm_in = RMSNorm(c_z, eps)
        self.norm_out = RMSNorm(self.c_hidden, eps)

        self.linear_q = Linear(c_z, self.c_hidden)
        self.linear_qx = Linear(c_z, self.c_hidden)
        self.linear_k = Linear(c_z, self.c_hidden)
        self.linear_kx = Linear(c_z, self.c_hidden)
        self.linear_g = Linear(c_z, c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, c_z, init="final")

    def forward(self, z, z_mask):
        if self.transpose:
            z = z.transpose(-2, -3)
        z = self.norm_in(z)
        q = self.linear_qx(z) * torch.sigmoid(self.linear_q(z)) * z_mask[..., None]
        k = self.linear_kx(z) * torch.sigmoid(self.linear_k(z)) * z_mask[..., None]
        g = torch.sigmoid(self.linear_g(z))
        attn_score = torch.einsum("...ijc,...Ijc->...iIc", q, k)
        attn_score = self.norm_out(attn_score)
        attn_score = self.linear_z(attn_score)
        o = attn_score * g
        if self.transpose:
            o = o.transpose(-2, -3)

        return o.float()


class TriangleAttention(nn.Module):
    def __init__(self, c_z, inf, eps, transpose=False):
        super().__init__()
        self.c_hidden = 32
        assert int(c_z % self.c_hidden) == 0
        self.c_z = c_z
        self.no_heads = int(c_z / self.c_hidden)
        self.inf = inf
        self.eps = eps
        self.transpose = transpose

        self.norm = RMSNorm(c_z, eps=eps)

        self.linear_q = Linear(c_z, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_z, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_z, self.no_heads * self.c_hidden, bias=False)
        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.linear_g = Linear(c_z, c_z)
        self.linear_o = Linear(c_z, c_z)

    def forward(self, z, z_mask):
        if self.transpose:
            z = z.transpose(-2, -3)

        B, S, _ = z.shape
        H, D = self.no_heads, self.c_hidden

        z_norm = self.norm(z)

        q = self.linear_q(z_norm).reshape([B, S, H, D]).transpose(-2, -3)
        k = self.linear_k(z_norm).reshape([B, S, H, D]).transpose(-2, -3)
        v = self.linear_v(z_norm).reshape([B, S, H, D]).transpose(-2, -3)
        g = self.linear_g(z_norm)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])[None]
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None, :, :]
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_bias, dropout_p=0, scale=None).transpose(-2, -3)

        o = attn_out.reshape([B, S, -1])
        o = self.linear_o(o) * g
        if self.transpose:
            o = o.transpose(-2, -3)
        return o.float()


class DiTAttention(nn.Module):
    def __init__(self, c_s, c_z, inf, eps):
        super().__init__()
        self.c_hidden = 32
        assert int(c_s % self.c_hidden) == 0
        self.c_s = c_s
        self.c_z = c_z
        self.no_heads = int(c_s / self.c_hidden)
        self.inf = inf
        self.eps = eps

        self.norm_s = AdaLayerNormZero(c_s, eps)
        self.norm_z = LayerNorm(c_z)
        self.linear_q = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_k = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_v = Linear(c_s, self.no_heads * self.c_hidden, bias=False)
        self.linear_z = Linear(c_z, self.no_heads, bias=False)
        self.norm_q = RMSNorm(self.c_hidden, eps)
        self.norm_k = RMSNorm(self.c_hidden, eps)
        self.linear_o = Linear(c_s, c_s)

    def forward(self, bs, z, t, z_mask, beta=None):
        B, S, _ = bs.shape
        H, D = self.no_heads, self.c_hidden

        bs_norm, gate = self.norm_s(bs, t)
        z_norm = self.norm_z(z)

        q = self.linear_q(bs_norm).reshape([B, S, H, D]).transpose(-2, -3)
        k = self.linear_k(bs_norm).reshape([B, S, H, D]).transpose(-2, -3)
        v = self.linear_v(bs_norm).reshape([B, S, H, D]).transpose(-2, -3)
        q = self.norm_q(q)
        k = self.norm_k(k)

        attn_bias = self.linear_z(z_norm).permute([2, 0, 1])[None, :, :, :]
        attn_bias = attn_bias + gen_attn_mask(z_mask.type_as(q), -self.inf)[None, None]
        if beta is not None:
            attn_bias = attn_bias + beta.type_as(q)[:, None, :, :]

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_bias, dropout_p=0, scale=None).transpose(-2, -3)

        o = attn_out.reshape([B, S, -1])
        o = self.linear_o(o).float() * gate.float()

        return o.float()
