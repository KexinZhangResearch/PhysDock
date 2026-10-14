"""Checkpoint loading. load_pretrained builds the model on the meta device and assigns the checkpoint tensors
directly (no initialisation, one copy per tensor): about 1 s for the released 153M-parameter model."""
import torch


def load_checkpoint_state_dict(ckpt_path, mmap: bool = True, use_ema: bool = True) -> dict:
    """Parameters of params.pt or a Uni-Core checkpoint (EMA if use_ema), on the CPU, without "model." / "_orig_mod."."""
    try:
        params = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=mmap)
    except Exception:  # noqa: BLE001 - Uni-Core checkpoints also pickle non-tensor objects
        params = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=mmap)
    if isinstance(params, dict) and ("ema" in params or "model" in params) and \
            not any(isinstance(v, torch.Tensor) for v in params.values()):
        params = params["ema"]["params"] if use_ema and "ema" in params else params["model"]
    if not isinstance(params, dict) or not all(isinstance(v, torch.Tensor) for v in params.values()):
        raise ValueError(f"{ckpt_path} does not contain a parameter dict")
    strip = lambda d, p: {(k[len(p):] if k.startswith(p) else k): v for k, v in d.items()}
    return strip(strip(params, "model."), "_orig_mod.")


def load_pretrained(ckpt_path, config=None, device="cuda", dtype: torch.dtype = None, eval_mode: bool = True):
    """PhysDock with all parameters from a checkpoint.

    Args:
        ckpt_path: params.pt or a Uni-Core checkpoint
        config: PhysDockConfig (default: "medium")
        device: target device
        dtype: cast floating-point parameters (default: keep float32)
        eval_mode: return the model in eval mode
    """
    from PhysDock import PhysDock, PhysDockConfig
    from PhysDock.models.primitives import skip_weight_init

    device = torch.device(device)
    with skip_weight_init(), torch.device("meta"):
        model = PhysDock(config if config is not None else PhysDockConfig(model_name="medium"))
    params = load_checkpoint_state_dict(ckpt_path, mmap=True)
    expected = model.state_dict()
    missing, unexpected = sorted(set(expected) - set(params)), sorted(set(params) - set(expected))
    if missing or unexpected:
        raise RuntimeError(f"{ckpt_path} does not match the model: {len(missing)} missing (e.g. {missing[:3]}), "
                           f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
    on_device = {}
    for name, value in params.items():
        if value.shape != expected[name].shape:
            raise RuntimeError(f"{name}: checkpoint shape {tuple(value.shape)}, model shape {tuple(expected[name].shape)}")
        target = dtype if dtype is not None and value.is_floating_point() else expected[name].dtype
        on_device[name] = value.to(device=device, dtype=target, non_blocking=True)
    model.load_state_dict(on_device, strict=True, assign=True)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("some tensors are not in the checkpoint and stay uninitialised")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model.eval() if eval_mode else model.train()


def import_unicore_ckpt(model: torch.nn.Module, ckpt_path, load_ema_state: bool = True, remove_compile_prefix: bool = False):
    """Load a Uni-Core checkpoint into an existing model (strict); "_orig_mod." prefixes are always removed."""
    model.load_state_dict(load_checkpoint_state_dict(ckpt_path, mmap=False, use_ema=load_ema_state))
    print("Weights loaded")
