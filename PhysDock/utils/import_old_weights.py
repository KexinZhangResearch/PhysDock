"""Parameter names of older checkpoints -> the names of the current modules. The only place that knows the old layouts: the model
loads (and saves) the current names only; a checkpoint written by an older version goes through import_old_weights first
(load_checkpoint_state_dict does it for params.pt, Lightning and Uni-Core checkpoints).

Two older layouts are understood, and any mix of them:
  the released params.pt   q, k, v, g (and the five projections of the triangle update, w1 and w3 of the feed-forward) are separate
                           Linear layers; the transitions are ffn_norm + feed_forward;
  intermediate versions    the projections merged (linear_qkvg [+ gate_bias], linear_in, w13), the other names old.
Optimizer states of old Lightning checkpoints do not fit the merged parameters (other number and shapes): start a fresh optimizer.

Conversion is by owner module: an owner is a module prefix that has the parameter named in the rule (so the same layer name in two kinds of
module is told apart by what its siblings are, e.g. the pair bias norm_z of an attention against the LayerNorm norm_z of a DiT attention).
"""
import re

import torch


def _owners(sd: dict, member: str) -> list:
    """Module prefixes ("blocks.0.attention.", "" for the root) that own the parameter `member` ("linear_g.weight")"""
    return [k[:-len(member)] for k in sd if k == member or k.endswith("." + member)]


def _stack(sd: dict, base: str, layers: tuple, target: str, kinds: tuple = ("weight",)) -> bool:
    """base+target.<kind> = the rows of base+layers[i].<kind> stacked, for every kind in which all layers are there"""
    done = False
    for kind in kinds:
        keys = [f"{base}{layer}.{kind}" for layer in layers]
        if all(k in sd for k in keys):
            sd[f"{base}{target}.{kind}"] = torch.cat([sd.pop(k) for k in keys])
            done = True
    return done


def _move(sd: dict, base: str, moves: dict) -> None:
    for old, new in moves.items():
        if base + old in sd:
            sd[base + new] = sd.pop(base + old)


# the transitions (pre-norm SwiGLU feed-forwards): ffn_norm -> norm, feed_forward -> ffn
_TRANSITIONS = [(re.compile(r"(^|\.)((?:pair_|msa_)?transition)\.ffn_norm\."), r"\1\2.norm."),
                (re.compile(r"(^|\.)((?:pair_|msa_)?transition)\.feed_forward\."), r"\1\2.ffn.")]


def import_old_weights(state_dict: dict) -> dict:
    """Convert `state_dict` (keys without the "model." prefix) to the current parameter names, in place; returns it. Keys that are in the
    current layout are left alone (the conversion is idempotent)."""
    sd = state_dict
    # 1. separate Linear layers -> merged matmuls
    for base in _owners(sd, "linear_qx.weight"):                                    # triangle multiplication: qx, q, kx, k, gate
        _stack(sd, base, ("linear_qx", "linear_q", "linear_kx", "linear_k", "linear_g"), "linear_in", ("weight", "bias"))
    for base in _owners(sd, "linear_g.weight"):                                     # gated attention: q, k, v, g
        if _stack(sd, base, ("linear_q", "linear_k", "linear_v", "linear_g"), "linear_qkvg"):
            _move(sd, base, {"linear_g.bias": "linear_qkvg.bias"})
    for base in _owners(sd, "norm_in.weight"):                                      # outer product mean: q, k (the only owner of norm_in)
        if base + "linear_q.weight" in sd:
            _stack(sd, base, ("linear_q", "linear_k"), "qk.linear", ("weight",))
            sd[base + "qk.bias"] = torch.cat([sd.pop(base + "linear_q.bias"), sd.pop(base + "linear_k.bias")])
            _move(sd, base, {"norm_in.weight": "qk.norm.weight"})
    for base in _owners(sd, "linear_v.weight"):                                     # DiT attention: q, k, v
        _stack(sd, base, ("linear_q", "linear_k", "linear_v"), "linear_qkv")
    for base in _owners(sd, "w1.weight"):                                           # SwiGLU feed-forward: w1, w3
        _stack(sd, base, ("w1", "w3"), "w13")
    # 2. module names
    for key in list(sd):
        new = key
        for pattern, repl in _TRANSITIONS:
            new = pattern.sub(repl, new)
        if new != key:
            sd[new] = sd.pop(key)
    for base in _owners(sd, "linear_qkvg.weight"):                                  # the attentions with a gate
        if base + "norm_z.weight" in sd and base + "norm_z.bias" not in sd:         # pair bias from an RMSNorm-ed pair input
            _move(sd, base, {"norm_z.weight": "pair_bias.norm.weight", "linear_z.weight": "pair_bias.linear.weight"})
        norm = next(n for n in ("norm_s", "norm_m", "norm") if base + n + ".weight" in sd)          # the norm of the stream
        _move(sd, base, {norm + ".weight": "qkvg.norm.weight", "linear_qkvg.weight": "qkvg.linear.weight",
                         "linear_qkvg.bias": "qkvg.bias", "linear_qkvg.gate_bias": "qkvg.bias"})
    return sd
