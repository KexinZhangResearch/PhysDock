"""Triton kernels: functions with torch.library custom ops (a fused forward and backward, opaque for torch.compile) and the
conditions under which they apply (can_fuse). The nn.Module wrappers with their torch fallbacks are in PhysDock.models.primitives."""
from PhysDock.models.kernels import pair_bias, rmsnorm_swiglu_ffn, shared_bias_attention, swiglu_ffn, triangle_update  # noqa: F401

__all__ = ["pair_bias", "rmsnorm_swiglu_ffn", "shared_bias_attention", "swiglu_ffn", "triangle_update"]
