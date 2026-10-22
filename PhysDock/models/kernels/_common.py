"""Helpers shared by the Triton kernels.

Matmul precision of the kernels (the same as the torch expressions they replace, never lower); the mode follows the state torch would use:
  "tf32x3"  float32 inputs with torch.set_float32_matmul_precision("highest"): every float32 operand is split into a TF32 value and the TF32
            value of the remainder and the product is the sum of three TF32 tensor-core products (the small x small term dropped), float32
            accumulation: the accuracy of float32 FMAs (relative error ~3e-7 against 2e-7 for torch's float32) at tensor-core speed;
  "tf32"    float32 inputs with torch.set_float32_matmul_precision("high" / "medium"): inputs rounded to the nearest TF32 value
            (what cuBLAS does; the Triton default truncates), float32 accumulation;
  "bf16"    under torch.autocast(dtype=bfloat16): the matmul inputs are rounded to bfloat16, products and sums in float32;
  "fp16"    under torch.autocast(dtype=float16): the same with float16.
Everything between the matmuls (norms, silu, products, gates, residuals) is float32 in every mode. The launch configurations were swept for TF32
and bfloat16: tf32x3 uses those of TF32 and fp16 those of bfloat16 (cfg_mode).
"""
import torch
import triton
import triton.language as tl

MODES = {"tf32x3": 0, "tf32": 1, "bf16": 2, "fp16": 3}
MODE_NAMES = {v: k for k, v in MODES.items()}
_NUM_SMS = {}


def num_sms() -> int:
    dev = torch.cuda.current_device()
    if dev not in _NUM_SMS:
        _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NUM_SMS[dev]


def float32_mode() -> int:
    """MODES value of the matmuls of float32 operands: 3xTF32 under torch.set_float32_matmul_precision("highest"), TF32 under "high" / "medium" """
    return MODES["tf32x3"] if torch.get_float32_matmul_precision() == "highest" else MODES["tf32"]


def matmul_mode() -> int:
    """MODES value of the matmuls under the current autocast / float32 matmul precision state"""
    if torch.is_autocast_enabled():
        return MODES["fp16"] if torch.get_autocast_dtype("cuda") == torch.float16 else MODES["bf16"]
    return float32_mode()


def is_half(mode: int) -> bool:
    """bfloat16 or float16 matmuls"""
    return mode >= MODES["bf16"]


def is_tf32(mode: int) -> bool:
    """TF32 or 3xTF32 matmuls (float32 inputs)"""
    return mode <= MODES["tf32"]


def half_dtype(mode: int) -> torch.dtype:
    return torch.float16 if mode == MODES["fp16"] else torch.bfloat16


def cfg_mode(mode: int) -> int:
    """the mode whose swept launch configuration `mode` uses: tf32x3 the TF32 one, fp16 the bfloat16 one"""
    return {MODES["tf32x3"]: MODES["tf32"], MODES["fp16"]: MODES["bf16"]}.get(mode, mode)


def round_tf32(w: torch.Tensor) -> torch.Tensor:
    """w (float32) rounded to the nearest TF32 value: add half an ulp of the 10-bit mantissa, clear the low 13 bits"""
    return ((w.float().contiguous().view(torch.int32) + 0x1000) & -8192).view(torch.float32)


def prep_weight(w: torch.Tensor, mode: int) -> torch.Tensor:
    """weight as the matmul operand of `mode` (the kernels load it without any conversion)"""
    if is_half(mode):
        return w.to(half_dtype(mode)).contiguous()
    return round_tf32(w) if mode == MODES["tf32"] else w.float().contiguous()


def compute_dtype(mode: int) -> torch.dtype:
    """dtype of the tensors a kernel stores as matmul operands for its backward"""
    return half_dtype(mode) if is_half(mode) else torch.float32


@triton.jit
def round_tf32_(x):
    return ((x.to(tl.int32, bitcast=True) + 0x1000) & -8192).to(tl.float32, bitcast=True)


@triton.jit
def cast_in(x, MODE: tl.constexpr):
    """x (float32) as a matmul input of the mode: bfloat16 / float16, float32 rounded to the nearest TF32 value, or (3xTF32: split by the matmul) as is"""
    if MODE == 2:
        return x.to(tl.bfloat16)
    elif MODE == 3:
        return x.to(tl.float16)
    elif MODE == 1:
        return round_tf32_(x)
    else:
        return x


@triton.jit
def mm(a, b, MODE: tl.constexpr):
    """a @ b of the precision MODE, float32 accumulation. a is a float32 value (converted here); b is a weight tile that prep_weight
    already converted (bfloat16 / float16 / TF32-rounded float32 / float32)"""
    if MODE == 0:
        return tl.dot(a, b, input_precision="tf32x3")
    elif MODE == 1:
        return tl.dot(round_tf32_(a), b, input_precision="tf32")
    elif MODE == 2:
        return tl.dot(a.to(tl.bfloat16), b)
    else:
        return tl.dot(a.to(tl.float16), b)


@triton.jit
def mm_act(a, b, MODE: tl.constexpr):
    """a @ b of the precision MODE for two float32 activations (both converted here)"""
    if MODE == 0:
        return tl.dot(a, b, input_precision="tf32x3")
    elif MODE == 1:
        return tl.dot(round_tf32_(a), round_tf32_(b), input_precision="tf32")
    elif MODE == 2:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    else:
        return tl.dot(a.to(tl.float16), b.to(tl.float16))


@triton.jit
def dot_in(a, b, MODE: tl.constexpr):
    """a @ b of operands that cast_in already converted to the matmul input of MODE (the half types and TF32-rounded values go to the tensor
    cores as they are; 3xTF32 splits the float32 operands)"""
    if MODE == 0:
        return tl.dot(a, b, input_precision="tf32x3")
    else:
        return tl.dot(a, b, input_precision="tf32")


@triton.jit
def dot_acc(a, b, acc, MODE: tl.constexpr):
    """acc + a @ b, operands as for dot_in"""
    if MODE == 0:
        return tl.dot(a, b, acc, input_precision="tf32x3")
    else:
        return tl.dot(a, b, acc, input_precision="tf32")
