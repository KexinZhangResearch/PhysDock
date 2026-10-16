"""Checkpoint loading. load_pretrained builds the model on the meta device and assigns the checkpoint tensors
directly (no initialisation, one copy per tensor): about 1 s for the released 153M-parameter model."""
import torch


def load_checkpoint_state_dict(ckpt_path, mmap: bool = True, use_ema: bool = True) -> dict:
    """Model parameters of params.pt, a Lightning checkpoint (scripts/train/train.py) or a Uni-Core checkpoint, on the CPU,
    without the "model." / "_orig_mod." prefixes. With use_ema the EMA weights are taken if the checkpoint has them."""
    try:
        params = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=mmap)
    except Exception:  # noqa: BLE001 - training checkpoints also pickle non-tensor objects
        params = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=mmap)
    if isinstance(params, dict) and "state_dict" in params:                      # Lightning
        params = params["ema"]["weights"] if use_ema and "ema" in params else params["state_dict"]
        params = {k: v for k, v in params.items() if k.startswith("model.")}
    elif isinstance(params, dict) and ("ema" in params or "model" in params) and \
            not any(isinstance(v, torch.Tensor) for v in params.values()):     # Uni-Core
        params = params["ema"]["params"] if use_ema and "ema" in params else params["model"]
    if not isinstance(params, dict) or not all(isinstance(v, torch.Tensor) for v in params.values()):
        raise ValueError(f"{ckpt_path} does not contain a parameter dict")
    strip = lambda d, p: {(k[len(p):] if k.startswith(p) else k): v for k, v in d.items()}
    from PhysDock.models.primitives.checkpoint_keys import upgrade_legacy_keys
    return upgrade_legacy_keys(strip(strip(params, "model."), "_orig_mod."))                 # the layout of the released checkpoint


def load_pretrained(ckpt_path, config=None, device="cuda", eval_mode: bool = True):
    """PhysDock with all parameters from a checkpoint, in float32 (use bf16 autocast for low precision: the decoder,
    distogram head and loss stay float32 under it).

    Args:
        ckpt_path: params.pt, a Lightning or a Uni-Core checkpoint
        config: model config (configs/model/*.yaml; default: "medium")
        device: target device
        eval_mode: return the model in eval mode
    """
    from PhysDock.configs import model_config
    from PhysDock.models.model import PhysDock
    from PhysDock.models.primitives import skip_weight_init

    device = torch.device(device)
    with skip_weight_init(), torch.device("meta"):
        model = PhysDock.from_config(config if config is not None else model_config("medium"))
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
        on_device[name] = value.to(device=device, dtype=expected[name].dtype, non_blocking=True)
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
