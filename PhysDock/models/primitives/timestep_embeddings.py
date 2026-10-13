"""Sinusoidal timestep embeddings followed by a small MLP (adapted from HuggingFace ``diffusers``)."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

__all__ = ["get_activation", "get_timestep_embedding", "Timesteps", "TimestepEmbedding", "TimestepEmbeddings"]

_ACTIVATIONS: Dict[str, Callable[[], nn.Module]] = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Create an activation module from its name.

    Args:
        act_fn: One of ``"swish"``, ``"silu"``, ``"mish"``, ``"gelu"``, ``"relu"`` (case-insensitive).

    Returns:
        A new activation module.

    Raises:
        ValueError: If the name is not supported.
    """
    try:
        return _ACTIVATIONS[act_fn.lower()]()
    except KeyError:
        raise ValueError(f"Unsupported activation function: {act_fn}") from None


def get_timestep_embedding(
        timesteps: torch.Tensor,
        embedding_dim: int,
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1,
        scale: float = 1,
        max_period: int = 10000,
) -> torch.Tensor:
    """Create DDPM-style sinusoidal timestep embeddings.

    Args:
        timesteps: 1-D tensor of ``N`` (possibly fractional) timesteps.
        embedding_dim: Output channel size.
        flip_sin_to_cos: Order the output as ``[cos, sin]`` instead of ``[sin, cos]``.
        downscale_freq_shift: Offset subtracted from ``embedding_dim // 2`` in the frequency spacing.
        scale: Factor applied to the angles.
        max_period: Controls the lowest frequency.

    Returns:
        Float32 tensor of shape ``[N, embedding_dim]`` (zero-padded by one channel if ``embedding_dim`` is odd).
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Module wrapper around :func:`get_timestep_embedding`.

    Args:
        num_channels: Embedding size.
        flip_sin_to_cos: Order the output as ``[cos, sin]``.
        downscale_freq_shift: See :func:`get_timestep_embedding`.
        scale: See :func:`get_timestep_embedding`.
    """

    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            timesteps: Tensor of shape ``[N]``.

        Returns:
            Tensor of shape ``[N, num_channels]``.
        """
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    """Two-layer MLP ``linear_1 -> act -> linear_2 [-> post_act]`` applied to a timestep projection.

    Args:
        in_channels: Input size.
        time_embed_dim: Hidden size.
        act_fn: Activation between the two linears.
        out_dim: Output size; defaults to ``time_embed_dim``.
        post_act_fn: Optional activation after the second linear.
        cond_proj_dim: If set, an extra condition of this size is projected and added to the input.
        sample_proj_bias: Whether the two linears have a bias.
    """

    def __init__(
            self,
            in_channels: int,
            time_embed_dim: int,
            act_fn: str = "silu",
            out_dim: Optional[int] = None,
            post_act_fn: Optional[str] = None,
            cond_proj_dim: Optional[int] = None,
            sample_proj_bias: bool = True,
    ) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        self.post_act = get_activation(post_act_fn) if post_act_fn is not None else None

    def forward(self, sample: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply the MLP.

        Args:
            sample: Tensor of shape ``[N, in_channels]``.
            condition: Optional tensor of shape ``[N, cond_proj_dim]``.

        Returns:
            Tensor of shape ``[N, out_dim]``.
        """
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class TimestepEmbeddings(nn.Module):
    """256-channel sinusoidal projection (``[cos, sin]``) followed by a 256 -> ``embedding_dim`` MLP.

    The output feeds :class:`~PhysDock.models.primitives.AdaLayerNormZero`, whose conditioning size is 256.

    Args:
        embedding_dim: Size of the output embedding.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """Embed a batch of (noise-level) timesteps.

        Args:
            timestep: Tensor of shape ``[B]``.

        Returns:
            Tensor of shape ``[B, embedding_dim]`` in the dtype of ``timestep``.
        """
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=timestep.dtype))
