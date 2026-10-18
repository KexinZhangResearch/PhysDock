"""Triton kernels: functions with torch.library custom ops (a fused forward and backward, opaque for torch.compile) and the
conditions under which they apply (can_fuse). The nn.Module wrappers with their torch fallbacks are in PhysDock.models.primitives."""
from PhysDock.models.kernels import (block_neighbor_attention, linear_gate_residual, outer_product_mean, pair_bias, rmsnorm_linear, rmsnorm_swiglu_ffn, shared_bias_attention, swiglu_ffn,  # noqa: F401
                                     triangle_update)

__all__ = ["block_neighbor_attention", "linear_gate_residual", "outer_product_mean", "pair_bias", "rmsnorm_linear", "rmsnorm_swiglu_ffn", "shared_bias_attention", "swiglu_ffn", "triangle_update"]
