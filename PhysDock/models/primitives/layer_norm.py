import torch
import torch.nn as nn
import torch.nn.functional as F

LayerNorm = nn.LayerNorm


class FP32LayerNorm(nn.LayerNorm):
    def forward(
            self,
            inputs: torch.Tensor
    ) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)
