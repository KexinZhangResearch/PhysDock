"""Float32 matmul precision. TF32 (10-bit mantissa inputs, float32 accumulation) is the default wherever the GPU
supports it (compute capability >= 8.0); the model was released with it and its error is ~1e-3 of bf16's."""
import torch


def tf32_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def set_matmul_precision(precision: str = "auto") -> str:
    """Set the precision of float32 matmuls / convolutions.

    Args:
        precision: "auto" (TF32 if supported, else full float32), "high" (TF32), "highest" (full float32) or
            "medium" (bf16 inputs, see torch.set_float32_matmul_precision)

    Returns:
        the precision that was set
    """
    if precision == "auto":
        precision = "high" if tf32_supported() else "highest"
    if precision not in ("highest", "high", "medium"):
        raise ValueError(f"matmul precision must be auto, highest, high or medium, got {precision!r}")
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = precision != "highest"
    return precision
