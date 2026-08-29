import torch
import torch.nn as nn
import math
from einops import einsum, reduce, rearrange

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()

        std = math.sqrt(2.0 / (in_features + out_features))
        bound = 3.0 * std
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-bound,
            b=bound,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in, out in -> ... out")


class Embedding(nn.Module):
    def __init__(
            self, 
            num_embeddings: int, 
            embedding_dim: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):

        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.num_embeddings = num_embeddings
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch_size, sequence_length, d_model)
        assert x.shape[-1] == self.d_model

        in_dtype = x.dtype
        x = x.to(dtype=torch.float32)
        squared_x = x * x
        rms = torch.sqrt(reduce(squared_x, "... hidden -> ...", 'mean') + self.eps).unsqueeze(-1)
        rms = rms.expand(*rms.shape[:-1], self.d_model)
        norm = x / rms
        norm = norm * self.weight
        return norm.to(dtype=in_dtype)

        