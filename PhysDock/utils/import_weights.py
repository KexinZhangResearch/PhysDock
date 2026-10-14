"""Loading PhysDock checkpoints.

:func:`load_pretrained` is the fast path for inference: the model is built on the ``meta`` device (no memory is
allocated and no initialiser runs), the checkpoint is memory-mapped, every tensor is copied straight to the target
device and the copies are *assigned* to the parameters (no second copy). For the 153M-parameter model this takes
about 1 s instead of 4-5 s (or about 40 s with the random initialisation).
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Union

import torch

__all__ = ["load_pretrained", "load_checkpoint_state_dict", "import_state_dict", "import_unicore_ckpt"]

PathLike = Union[str, os.PathLike]


def _strip_prefix(params: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in params.items()}


def load_checkpoint_state_dict(ckpt_path: PathLike, mmap: bool = True, use_ema: bool = True) -> Dict[str, torch.Tensor]:
    """Read the model parameters of a checkpoint (on the CPU), without the ``model.`` / ``_orig_mod.`` prefixes.

    Accepts the released ``params.pt`` (a flat ``{"model.<name>": tensor}`` dict) and Uni-Core checkpoints
    (``{"ema": {"params": ...}, "model": ...}``).

    Args:
        ckpt_path: Checkpoint file.
        mmap: Memory-map the file instead of reading it into memory (tensors are read on first use).
        use_ema: For Uni-Core checkpoints: prefer the EMA parameters.

    Returns:
        ``{parameter name: tensor}``.

    Raises:
        ValueError: If the file holds no parameter dict.
    """
    try:
        params: Any = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=mmap)
    except Exception:  # noqa: BLE001 - pickles with non-tensor objects (e.g. Uni-Core args) need the full unpickler
        params = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=mmap)
    if isinstance(params, Mapping) and ("ema" in params or "model" in params) and not any(
            isinstance(v, torch.Tensor) for v in params.values()):
        params = params["ema"]["params"] if use_ema and "ema" in params else params["model"]
    if not isinstance(params, Mapping) or not all(isinstance(v, torch.Tensor) for v in params.values()):
        raise ValueError(f"{ckpt_path} does not contain a parameter dict")
    return _strip_prefix(_strip_prefix(params, "model."), "_orig_mod.")


def load_pretrained(
        ckpt_path: PathLike,
        config: Optional[Any] = None,
        device: Union[str, torch.device] = "cuda",
        dtype: Optional[torch.dtype] = None,
        eval_mode: bool = True,
) -> torch.nn.Module:
    """Build PhysDock and load a checkpoint as fast as possible (no initialisation, one copy per tensor).

    Args:
        ckpt_path: Checkpoint (``params.pt`` or a Uni-Core checkpoint).
        config: :class:`~PhysDock.configs.PhysDockConfig`; defaults to ``PhysDockConfig(model_name="medium")``.
        device: Target device.
        dtype: Cast floating-point parameters to this dtype (default: keep the checkpoint dtype, float32).
        eval_mode: Return the model in eval mode.

    Returns:
        The model on ``device``; every parameter comes from the checkpoint.

    Raises:
        RuntimeError: If parameters are missing from or unexpected in the checkpoint.
    """
    from PhysDock import PhysDock, PhysDockConfig
    from PhysDock.models.primitives import skip_weight_init

    device = torch.device(device)
    if config is None:
        config = PhysDockConfig(model_name="medium")
    with skip_weight_init(), torch.device("meta"):
        model = PhysDock(config)
    params = load_checkpoint_state_dict(ckpt_path, mmap=True)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(params))
    unexpected = sorted(set(params) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"{ckpt_path} does not match the model: {len(missing)} missing (e.g. {missing[:3]}), "
                           f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    on_device = {}
    for name, value in params.items():
        target_dtype = dtype if dtype is not None and value.is_floating_point() else expected[name].dtype
        if value.shape != expected[name].shape:
            raise RuntimeError(f"{name}: checkpoint shape {tuple(value.shape)}, model shape {tuple(expected[name].shape)}")
        on_device[name] = value.to(device=device, dtype=target_dtype, non_blocking=True)
    model.load_state_dict(on_device, strict=True, assign=True)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if left:
        raise RuntimeError(f"tensors not in the checkpoint are still uninitialised: {left[:5]}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model.eval() if eval_mode else model.train()


def import_unicore_ckpt(
        model: torch.nn.Module,
        ckpt_path: PathLike,
        load_ema_state: bool = True,
        remove_compile_prefix: bool = False,
) -> None:
    """Load a Uni-Core checkpoint into an existing model (strict).

    Args:
        model: The model.
        ckpt_path: Checkpoint file.
        load_ema_state: Prefer the EMA parameters.
        remove_compile_prefix: Kept for compatibility; ``_orig_mod.`` prefixes are always removed.
    """
    model.load_state_dict(load_checkpoint_state_dict(ckpt_path, mmap=False, use_ema=load_ema_state))
    print("Weights loaded")


def import_state_dict(model: torch.nn.Module, ckpt_path: PathLike) -> None:
    """Load ``params.pt`` into an existing model (strict)."""
    model.load_state_dict(load_checkpoint_state_dict(ckpt_path, mmap=False))
    print("Weights loaded")
