"""Attention layers, built from the kernel-level modules of PhysDock.models.primitives. Every head has 32 channels; masks become
additive -inf biases. Pair masks are symmetric. Every layer returns the residual-added stream, x + update.

The output gate has two forms (set_attention_gate, model config `attention_gate`):
  legacy (the released parameters)  x + linear_o(attention) * g          g: the raw linear output, unbounded, after the output projection
  sigmoid                           x + linear_o(attention * sigmoid(g)) per head and channel, before the output projection, in (0, 1)
The parameters are the same (the DiT attention gets one more matrix, `linear_g`, with the sigmoid gate)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from PhysDock.models.primitives import (
    AdaLayerNormZero, Linear, LinearGateResidual, RMSNorm, RMSNormLinear, RMSNormPairBias, SharedBiasAttention)

HEAD_DIM = 32


def _heads(x: torch.Tensor) -> torch.Tensor:
    """[..., N, H * 32] -> [..., H, N, 32]"""
    return x.reshape(*x.shape[:-1], -1, HEAD_DIM).transpose(-2, -3)


class _GatedOutput:
    """Mixin of the attention layers with a projection gate g from the merged q, k, v, g projection: legacy_gate = True (the released
    parameters) multiplies the projected output by the raw g, False gates the attention output per head with sigmoid(g) first."""
    sigmoid_gate = False

    def finish(self, x: torch.Tensor, o: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """x [..., N, c] stream, o [..., H, N, 32] attention output, g [..., N, c] -> x + update"""
        if self.sigmoid_gate:
            return self.out(x, o * torch.sigmoid(_heads(g)).to(o.dtype))
        return self.out(x, o, g)


class AttentionWithBias(_GatedOutput, nn.Module):
    """Gated self-attention over N positions (the single representation of the Pairformer, the atoms of the AtomTransformer) with a
    per-head logit bias from the pair representation."""

    def __init__(self, c_s: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.qkvg = RMSNormLinear(c_s, 4 * c_s, eps, bias_dim=c_s)
        self.pair_bias = RMSNormPairBias(c_z, c_s // HEAD_DIM, inf, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [N, c_s]
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, c_s] s + update
        """
        q, k, v, g = self.qkvg(s).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        o = self.attend(q, k, v, self.pair_bias(z, z_mask, q), self.attend.valid_length(z_mask))
        return self.finish(s, o, g)


class MSARowAttentionWithBias(_GatedOutput, nn.Module):
    """Gated attention along the tokens of every MSA row, with a logit bias from the pair representation."""

    def __init__(self, c_m: int, c_z: int, inf: float, eps: float) -> None:
        super().__init__()
        self.qkvg = RMSNormLinear(c_m, 4 * c_m, eps, bias_dim=c_m)
        self.pair_bias = RMSNormPairBias(c_z, c_m // HEAD_DIM, inf, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            m: [N_msa, N_token, c_m]
            z: [N_token, N_token, c_z]
            z_mask: [N_token, N_token]

        Returns:
            [N_msa, N_token, c_m] m + update
        """
        q, k, v, g = self.qkvg(m).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        return self.finish(m, self.attend(q, k, v, self.pair_bias(z, z_mask, q), self.attend.valid_length(z_mask)), g)


class MSAColumnAttention(_GatedOutput, nn.Module):
    """Gated attention along the MSA rows of every token (no mask, no bias)."""

    def __init__(self, c_m: int, eps: float) -> None:
        super().__init__()
        self.qkvg = RMSNormLinear(c_m, 4 * c_m, eps, bias_dim=c_m)
        self.out = LinearGateResidual(c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """m: [N_msa, N_token, c_m] -> m + update"""
        m = m.transpose(-2, -3)
        q, k, v, g = self.qkvg(m).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        return self.finish(m, F.scaled_dot_product_attention(q, k, v), g).transpose(-2, -3)


class TriangleAttention(_GatedOutput, nn.Module):
    """Gated attention along the rows of z with a bias from z (starting node); starting_node=False attends along the columns (ending
    node). The bias is the pair bias of z (its own RMSNorm; checkpoints from before it have the bias from the norm of the q, k, v, g
    projection, see PhysDock.utils.import_old_weights)."""

    def __init__(self, c_z: int, inf: float, eps: float, starting_node: bool = True) -> None:
        super().__init__()
        self.starting_node = starting_node
        self.qkvg = RMSNormLinear(c_z, 4 * c_z, eps, bias_dim=c_z)
        self.pair_bias = RMSNormPairBias(c_z, c_z // HEAD_DIM, inf, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_z)

    def forward(self, z: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [N, N, c_z]
            z_mask: [N, N]

        Returns:
            [N, N, c_z] z + update
        """
        if not self.starting_node:
            z = z.transpose(-2, -3)
        q, k, v, g = self.qkvg(z).chunk(4, dim=-1)
        q, k, v = _heads(q), _heads(k), _heads(v)
        z = self.finish(z, self.attend(q, k, v, self.pair_bias(z, z_mask, q), self.attend.valid_length(z_mask)), g)
        return z if self.starting_node else z.transpose(-2, -3)


class DiTAttention(nn.Module):
    """DiT self-attention: AdaLN-Zero input, RMS-normalised q/k, gated output with residual. The logit bias [H, N, N] (shared by all
    samples) is an argument: it comes from the pair_bias layer of the DiT (LayerNormPairBias)."""

    def __init__(self, c_s: int, eps: float) -> None:
        super().__init__()
        self.norm_s = AdaLayerNormZero(c_s, eps)
        self.linear_qkv = Linear(
            c_s, 3 * c_s, bias=False)              # rows q, k, v
        self.norm_q = RMSNorm(HEAD_DIM, eps)
        self.norm_k = RMSNorm(HEAD_DIM, eps)
        self.attend = SharedBiasAttention()
        self.out = LinearGateResidual(c_s)

    def set_head_gate(self, on: bool) -> None:
        """Per-head sigmoid gate of the attention output (an extra matrix `linear_g`, from the AdaLN-modulated input) on top of the AdaLN gate"""
        if on and not hasattr(self, "linear_g"):
            self.linear_g = Linear(self.out.linear.in_features, self.out.linear.in_features)
        if not on and hasattr(self, "linear_g"):
            del self.linear_g

    def forward(self, x: torch.Tensor, bias: torch.Tensor, mod: torch.Tensor, n_valid: torch.Tensor = None,
                pos: torch.Tensor = None, radius: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, c_s]
            bias: [H, N, N] logit bias (pair bias + mask)
            mod: [B, 1, 3 * c_s] AdaLN modulation (TimeModulation)
            n_valid: valid_length of the mask of `bias` (the padding behind it is skipped by the Triton backend), or None
            pos: [B, N, 3] and radius: [B] of the distance decay of `attend` (None: no decay)

        Returns:
            [B, N, c_s] x + update
        """
        q, k, v, gate, g = self.project(x, mod)
        return self.output(x, self.attend(q, k, v, bias, n_valid, pos, radius), gate, g)

    def project(self, x: torch.Tensor, mod: torch.Tensor):
        """Attention inputs: q, k, v [B, H, N, 32], the AdaLN output gate [B, 1, c_s] and the per-head sigmoid gate [B, H, N, 32] (None without it)"""
        h, gate = self.norm_s(x, mod)
        q, k, v = self.linear_qkv(h).split(h.shape[-1], dim=-1)
        g = _heads(torch.sigmoid(self.linear_g(h))) if hasattr(self, "linear_g") else None
        return self.norm_q(_heads(q)), self.norm_k(_heads(k)), _heads(v), gate, g

    def output(self, x: torch.Tensor, o: torch.Tensor, gate: torch.Tensor, g: torch.Tensor = None) -> torch.Tensor:
        """Attention output o [B, H, N, 32] -> x + gated update [B, N, c_s]"""
        return self.out(x, o if g is None else o * g.to(o.dtype), gate)


def set_attention_gate(model: nn.Module, sigmoid: bool, dit_sigmoid: bool = None) -> None:
    """Form of the output gate of the attentions of `model`: sigmoid (per head, before the output projection) or legacy (the released
    parameters: the raw gate after it). `sigmoid`: the attentions of the encoder (atom transformer, Pairformer, triangle attentions, MSA);
    `dit_sigmoid` (default: the same): the DiT attentions of the decoder, which get / lose their extra head gate matrix `linear_g` (they always
    have the AdaLN gate). Call it before the model is moved to a device."""
    dit_sigmoid = sigmoid if dit_sigmoid is None else dit_sigmoid
    for module in model.modules():
        if isinstance(module, _GatedOutput):
            module.sigmoid_gate = sigmoid
        elif isinstance(module, DiTAttention):
            module.set_head_gate(dit_sigmoid)
