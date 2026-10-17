"""Helpers shared by the Triton kernels.

Matmul precision of the fused SwiGLU kernels and pair_bias (the same as the torch expressions they replace, never lower):
  "bf16"  under autocast: the matmul inputs are rounded to bfloat16, products and sums in float32;
  "tf32"  float32 inputs with torch.set_float32_matmul_precision("high" / "medium"): inputs rounded to the nearest TF32 value
          (what cuBLAS does; the Triton default truncates), float32 accumulation;
  "fp32"  torch.set_float32_matmul_precision("highest"): float32 FMAs.
Everything between the matmuls (norms, silu, products, gates, residuals) is float32 in every mode.
"""
import torch
import triton
import triton.language as tl

MODES = {"fp32": 0, "tf32": 1, "bf16": 2}
_NUM_SMS = {}


def num_sms() -> int:
    dev = torch.cuda.current_device()
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NUM_SMS[dev]


def matmul_mode() -> int:
    """MODES value of the matmuls under the current autocast / float32 matmul precision state"""
    if torch.is_autocast_enabled():
        return MODES["bf16"]
    return MODES["fp32"] if torch.get_float32_matmul_precision() == "highest" else MODES["tf32"]


def round_tf32(w: torch.Tensor) -> torch.Tensor:
    """w (float32) rounded to the nearest TF32 value: add half an ulp of the 10-bit mantissa, clear the low 13 bits"""
    return ((w.float().contiguous().view(torch.int32) + 0x1000) & -8192).view(torch.float32)


def prep_weight(w: torch.Tensor, mode: int) -> torch.Tensor:
    """weight as the matmul operand of `mode` (the kernels load it without any conversion)"""
    if mode == MODES["bf16"]:
        return w.to(torch.bfloat16).contiguous()
    return round_tf32(w) if mode == MODES["tf32"] else w.float().contiguous()


def compute_dtype(mode: int) -> torch.dtype:
    return torch.bfloat16 if mode == MODES["bf16"] else torch.float32


@triton.jit
def round_tf32_(x):
    return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def cast_in(x, BF16: tl.constexpr):
    """x as a matmul input: bfloat16, or float32 rounded to the nearest TF32 value (TriangleMultiplication's two modes)"""
    if BF16:
        return x.to(tl.bfloat16)
    else:
        return round_tf32_(x)


@triton.jit
def mm(a, b, MODE: tl.constexpr):
    """a @ b of the precision MODE, float32 accumulation. a is a float32 value (converted here); b is a weight tile that prep_weight
    already converted (bfloat16 / TF32-rounded float32 / float32)"""
    if MODE == 0:
        return tl.dot(a, b, input_precision="ieee")
    elif MODE == 1:
        return tl.dot(round_tf32_(a), b, input_precision="tf32")
    else:
        return tl.dot(a.to(tl.bfloat16), b)


@triton.jit
def mm_act(a, b, MODE: tl.constexpr):
    """a @ b of the precision MODE for two float32 activations (both converted here)"""
    if MODE == 0:
        return tl.dot(a, b, input_precision="ieee")
    elif MODE == 1:
        return tl.dot(round_tf32_(a), round_tf32_(b), input_precision="tf32")
    else:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
