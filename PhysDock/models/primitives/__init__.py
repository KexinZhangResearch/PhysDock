"""Building blocks of PhysDock.

This ``__init__`` defines the lowest-level pieces that every other module is built from:

* :class:`Linear`: ``nn.Linear`` with AlphaFold-style initialisers and a bf16-safe forward.
* :class:`RMSNorm`, :class:`LayerNorm`, :class:`FP32LayerNorm`: normalisation layers.
* :class:`AdaLayerNormZero`: LayerNorm modulated by a timestep embedding (DiT conditioning).
* :func:`skip_weight_init`: context manager that skips the slow random initialisation when the
  weights are about to be overwritten by a checkpoint anyway.

The rest of the package is split by role and re-exported here:

* ``transitions``: :class:`FeedForward` (SwiGLU), :class:`Transition`, :class:`DiTTransition`.
* ``attentions``: pair-biased, MSA, triangle and DiT attention.
* ``opm``: :class:`OuterProductMean`.
* ``timestep_embeddings``: sinusoidal timestep embeddings.
* ``transformers``: Atom/Triangle/Evoformer/Pairformer/DiT stacks.

Notation used in all docstrings: ``N_token`` / ``N_atom`` / ``N_msa`` are the number of tokens, atoms and
MSA rows, ``H`` the number of attention heads, ``D`` the per-head width (always 32) and ``c_*`` channel sizes
(``c_s`` token/single, ``c_z`` token-pair, ``c_m`` MSA, ``c_a`` atom, ``c_ap`` atom-pair).
"""
from __future__ import annotations

import contextlib
import math
from typing import Callable, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

__all__ = [
    # defined here
    "Linear", "RMSNorm", "LayerNorm", "FP32LayerNorm", "AdaLayerNormZero", "skip_weight_init",
    # transitions.py
    "FeedForward", "Transition", "DiTTransition",
    # opm.py
    "OuterProductMean",
    # attentions.py
    "AttentionWithPairBias", "MSARowAttentionWithPairBias", "MSAColumnAttention", "TriangleUpdate",
    "TriangleAttention", "DiTAttention",
    # timestep_embeddings.py
    "TimestepEmbeddings",
    # transformers.py
    "AtomTransformer", "Triangleformer", "Evoformer", "Pairformer", "DiT",
]


# ======================================================================================
# Weight initialisers (AlphaFold 2 / 3 supplementary, section 1.11.4)
# ======================================================================================
def _calculate_fan(weight_shape: Tuple[int, int], fan: str = "fan_in") -> float:
    """Return the fan used to scale an initialiser.

    Args:
        weight_shape: ``(out_features, in_features)`` of a linear weight.
        fan: One of ``"fan_in"``, ``"fan_out"`` or ``"fan_avg"``.

    Returns:
        The selected fan.

    Raises:
        ValueError: If ``fan`` is not one of the supported options.
    """
    fan_out, fan_in = weight_shape
    if fan == "fan_in":
        return fan_in
    if fan == "fan_out":
        return fan_out
    if fan == "fan_avg":
        return (fan_in + fan_out) / 2
    raise ValueError(f"Invalid fan option {fan!r}")


def trunc_normal_init_(weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in") -> None:
    """Fill ``weights`` in place from a normal distribution truncated to two standard deviations.

    The variance is ``scale / max(1, fan)`` (so ``scale=1`` gives LeCun and ``scale=2`` He initialisation).
    Sampling uses ``scipy`` and therefore the global NumPy random state.

    Args:
        weights: 2-D weight tensor of shape ``(out_features, in_features)``, modified in place.
        scale: Numerator of the variance.
        fan: Which fan to divide by, see :func:`_calculate_fan`.
    """
    shape = weights.shape
    scale = scale / max(1, _calculate_fan(tuple(shape), fan))
    a, b = -2, 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=math.prod(shape))
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor) -> None:
    """LeCun (fan-in) truncated-normal initialisation, in place."""
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor) -> None:
    """He (fan-in, variance 2/fan) truncated-normal initialisation, in place."""
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor) -> None:
    """Glorot / Xavier uniform initialisation (gain 1), in place."""
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights: torch.Tensor) -> None:
    """Zero initialisation used for the last layer of a residual branch, in place."""
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor) -> None:
    """Zero weights for gating layers, in place (the bias is set to one by :class:`Linear`)."""
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights: torch.Tensor) -> None:
    """Kaiming-normal initialisation with linear gain (std ``1/sqrt(fan_in)``), in place."""
    nn.init.kaiming_normal_(weights, nonlinearity="linear")


def _no_op_init(weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in") -> None:
    """Stand-in for :func:`trunc_normal_init_` used by :func:`skip_weight_init`."""
    return None


@contextlib.contextmanager
def skip_weight_init() -> Iterator[None]:
    """Skip the truncated-normal sampling of :class:`Linear` while building a model.

    Creating the 153M-parameter model samples every weight with ``scipy`` (about 40 s on a many-core machine),
    which is wasted when a checkpoint is loaded right afterwards. Inside this context the sampling is replaced
    by a no-op; the layers keep PyTorch's default initial values, so **every parameter must subsequently be
    overwritten** (use ``load_state_dict(..., strict=True)``). Zero initialisers (``"final"``, ``"gating"``)
    and biases are unaffected. The context manager is not thread-safe.

    Example:
        >>> with skip_weight_init():
        ...     model = PhysDock(config)
        >>> model.load_state_dict(torch.load("params.pt"))
    """
    global trunc_normal_init_
    original = trunc_normal_init_
    trunc_normal_init_ = _no_op_init
    try:
        yield
    finally:
        trunc_normal_init_ = original


def _autocast_off(x: torch.Tensor) -> contextlib.AbstractContextManager:
    """Context manager disabling autocast on the device type of ``x`` (no-op for other devices)."""
    if x.device.type in ("cuda", "cpu"):
        return torch.autocast(device_type=x.device.type, enabled=False)
    return contextlib.nullcontext()


# ======================================================================================
# Linear
# ======================================================================================
class Linear(nn.Linear):
    """``nn.Linear`` with AlphaFold-style initialisers and bf16/mixed-precision handling.

    Args:
        in_dim: Size of the last input dimension.
        out_dim: Size of the last output dimension.
        bias: Whether to learn an additive bias (initialised to zero).
        init: Initialiser for the weight:

            * ``"default"``: LeCun fan-in truncated normal.
            * ``"relu"``: He fan-in truncated normal.
            * ``"glorot"``: Glorot uniform.
            * ``"gating"``: zero weight and, if there is a bias, bias one.
            * ``"normal"``: Kaiming normal with std ``1/sqrt(fan_in)``.
            * ``"final"``: all-zero weight.
            * ``"bias_fill_-2"``: bias filled with -2 and PyTorch's default weight initialisation.

            Ignored if ``init_fn`` is given.
        init_fn: Custom initialiser called as ``init_fn(weight, bias)`` (``bias`` is ``None`` without bias).
        precision: If set, the whole matmul is computed in this dtype with autocast disabled, and the
            result is cast back to the input dtype.

    Raises:
        ValueError: For an unknown ``init`` string, or ``"bias_fill_-2"`` without a bias.
    """

    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            bias: bool = True,
            init: str = "default",
            init_fn: Optional[Callable[[torch.Tensor, Optional[torch.Tensor]], None]] = None,
            precision: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(in_dim, out_dim, bias=bias)
        self.precision = precision

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        elif init == "default":
            lecun_normal_init_(self.weight)
        elif init == "relu":
            he_normal_init_(self.weight)
        elif init == "glorot":
            glorot_uniform_init_(self.weight)
        elif init == "gating":
            gating_init_(self.weight)
            if bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)
        elif init == "normal":
            normal_init_(self.weight)
        elif init == "final":
            final_init_(self.weight)
        elif init == "bias_fill_-2":
            if not bias:
                raise ValueError('init="bias_fill_-2" needs bias=True')
            with torch.no_grad():
                self.bias.fill_(-2.0)
        else:
            raise ValueError(f"Invalid init string {init!r}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the affine map.

        Args:
            input: Tensor of shape ``[..., in_dim]``.

        Returns:
            Tensor of shape ``[..., out_dim]``. Float32 in the default case; bf16 inputs are multiplied with a
            bf16 copy of the weight outside of autocast (and give a bf16 result).
        """
        dtype = input.dtype
        if self.precision is not None:
            with _autocast_off(input):
                bias = self.bias.to(dtype=self.precision) if self.bias is not None else None
                out = F.linear(input.to(dtype=self.precision), self.weight.to(dtype=self.precision), bias)
            return out.to(dtype=dtype)
        if dtype is torch.bfloat16:
            with _autocast_off(input):
                bias = self.bias.to(dtype=dtype) if self.bias is not None else None
                return F.linear(input, self.weight.to(dtype=dtype), bias)
        return F.linear(input, self.weight, self.bias)


# ======================================================================================
# Normalisation layers
# ======================================================================================
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019) with a learned scale.

    The statistics are computed in float32. The normalised tensor is cast back to the input dtype before it
    is multiplied with the float32 ``weight``, so bf16 inputs produce a float32 output (type promotion).

    Args:
        dim: Size of the last dimension.
        eps: Value added to the mean square for numerical stability.

    Attributes:
        weight: Learned scale, shape ``[dim]``, initialised to one.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension.

        Args:
            x: Tensor of shape ``[..., dim]``.

        Returns:
            Tensor of shape ``[..., dim]``.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


LayerNorm = nn.LayerNorm
"""Plain :class:`torch.nn.LayerNorm`; used with ``elementwise_affine=False`` inside :class:`AdaLayerNormZero`."""


class FP32LayerNorm(nn.LayerNorm):
    """:class:`torch.nn.LayerNorm` evaluated in float32 and cast back to the input dtype."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply layer norm over the trailing ``normalized_shape`` dimensions.

        Args:
            inputs: Tensor of shape ``[..., *normalized_shape]``.

        Returns:
            Tensor with the shape and dtype of ``inputs``.
        """
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm-Zero (DiT, Peebles & Xie 2023): conditioning produces shift, scale and gate.

    ``x_out = LayerNorm(x) * (1 + scale) + shift`` and the returned ``gate`` is meant to multiply the output of
    the sub-layer that follows (attention or transition) before the residual addition.

    Args:
        embedding_dim: Channel size of ``x``.
        eps: Epsilon of the (non-affine) LayerNorm.
        cond_dim: Size of the conditioning vector; must equal the timestep-embedding size (256).
    """

    def __init__(self, embedding_dim: int, eps: float, cond_dim: int = 256) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = Linear(cond_dim, 3 * embedding_dim, bias=True)
        self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=eps)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Modulate ``x`` with the conditioning ``t``.

        Args:
            x: Activations, shape ``[B, N, embedding_dim]``.
            t: Conditioning (timestep embedding), shape ``[B, cond_dim]``.

        Returns:
            Tuple ``(x_modulated, gate)``; ``x_modulated`` has the shape of ``x`` and ``gate`` has shape
            ``[B, 1, embedding_dim]``.
        """
        shift, scale, gate = self.linear(self.silu(t[..., None, :])).chunk(3, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x, gate


# Imported last: the modules below use Linear / RMSNorm / AdaLayerNormZero defined above.
from .transitions import DiTTransition, FeedForward, Transition  # noqa: E402
from .opm import OuterProductMean  # noqa: E402
from .attentions import (AttentionWithPairBias, DiTAttention, MSAColumnAttention,  # noqa: E402
                         MSARowAttentionWithPairBias, TriangleAttention, TriangleUpdate)
from .timestep_embeddings import TimestepEmbeddings  # noqa: E402
from .transformers import AtomTransformer, DiT, Evoformer, Pairformer, Triangleformer  # noqa: E402
