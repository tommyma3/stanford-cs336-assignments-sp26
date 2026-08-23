import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=device if device else "cpu", dtype=dtype if dtype else torch.float))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in, out in -> ... out")



        