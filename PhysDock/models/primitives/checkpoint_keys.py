"""Checkpoint keys of the merged projections. Several layers were one Linear each in the released checkpoint and are one
matmul now (the attention q, k, v and gate, the DiT q, k, v, the feed-forward w1 and w3, the TriangleUpdate projections).
upgrade_legacy_keys converts a state dict of the old layout in place. It is applied to the parameter dicts that are read
without a model (import_weights, EMA weights) and registered as a load pre-hook of the root model, so every
load_state_dict accepts both layouts. Converting a state dict that is already in the new layout changes nothing."""
import torch


def _bases(state_dict: dict, prefix: str, layer: str, kind: str = "weight") -> list:
    """Module prefixes below `prefix` that own the parameter <layer>.<kind> (e.g. "blocks.0.attention.")"""
    suffix = layer + "." + kind
    return [k[:-len(suffix)] for k in state_dict
            if k.startswith(prefix) and (k == suffix or k.endswith("." + suffix))]


def _merge(state_dict: dict, base: str, layers: tuple, target: str, kind: str) -> None:
    """target.<kind> = the rows of layers[0].<kind>, layers[1].<kind], ... stacked, if all of them are present"""
    keys = [base + layer + "." + kind for layer in layers]
    if all(k in state_dict for k in keys):
        state_dict[base + target + "." + kind] = torch.cat([state_dict.pop(k) for k in keys])


def upgrade_legacy_keys(state_dict: dict, prefix: str = "") -> dict:
    """In place, for the keys below `prefix`:
      TriangleUpdate  linear_qx, linear_q, linear_kx, linear_k, linear_g (weight, bias)  -> linear_in (rows in that order)
      gated attention linear_q, linear_k, linear_v, linear_g.weight -> linear_qkvg.weight (rows q, k, v, g);
                      linear_g.bias -> linear_qkvg.gate_bias
      DiT attention   linear_q, linear_k, linear_v (no gate)        -> linear_qkv.weight
      FeedForward     w1, w3.weight                                 -> w13.weight (rows w1, w3)
    Returns state_dict."""
    for base in _bases(state_dict, prefix, "linear_qx"):
        layers = ("linear_qx", "linear_q", "linear_kx", "linear_k", "linear_g")
        for kind in ("weight", "bias"):
            _merge(state_dict, base, layers, "linear_in", kind)
    for base in _bases(state_dict, prefix, "linear_g"):
        if all(base + layer + ".weight" in state_dict for layer in ("linear_q", "linear_k", "linear_v")):
            _merge(state_dict, base, ("linear_q", "linear_k", "linear_v", "linear_g"), "linear_qkvg", "weight")
            if base + "linear_g.bias" in state_dict:
                state_dict[base + "linear_qkvg.gate_bias"] = state_dict.pop(base + "linear_g.bias")
    for base in _bases(state_dict, prefix, "linear_v"):
        if base + "linear_g.weight" not in state_dict and all(base + l + ".weight" in state_dict for l in ("linear_q", "linear_k")):
            _merge(state_dict, base, ("linear_q", "linear_k", "linear_v"), "linear_qkv", "weight")
    for base in _bases(state_dict, prefix, "w1"):
        _merge(state_dict, base, ("w1", "w3"), "w13", "weight")
    return state_dict
