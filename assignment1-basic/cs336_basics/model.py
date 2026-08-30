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


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_feedforward: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_feedforward = round(d_model / 24) * 64 if d_feedforward is None else d_feedforward
        self.w1 = nn.Parameter(torch.empty(d_feedforward, d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(d_feedforward, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(d_model, d_feedforward, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w1)
        nn.init.trunc_normal_(self.w2)
        nn.init.trunc_normal_(self.w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff")
        print(result.shape)
        result *= torch.sigmoid(result)
        result *= einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
        result = einsum(self.w2, result, "d_model d_ff, ... d_ff -> ... d_model")
        return result


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        sins = torch.empty(max_seq_len, d_k // 2, device=device)
        coss = torch.empty(max_seq_len, d_k // 2, device=device)

        for i in range(max_seq_len):
            for k in range(d_k // 2):
                angle = i / (theta ** (2 * k / d_k))
                coss[i][k] = math.cos(angle)
                sins[i][k] = math.sin(angle)
        
        self.register_buffer("sins", sins, persistent=False)
        self.register_buffer("coss", coss, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        coss_sliced = self.coss[token_positions]
        sins_sliced = self.sins[token_positions]

        cos = self.coss[token_positions]
        sin = self.sins[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)

        out[..., ::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos

        return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_entry = torch.amax(x, dim=dim, keepdim=True)
    exp_input = torch.exp(x - max_entry)
    exp_sum = torch.sum(exp_input, dim=dim, keepdim=True)
    return exp_input / exp_sum


def scaled_dot_product_attention(
        key: torch.Tensor,
        query: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None
):
    # key & query: (batch_size, ..., seq_len, d_k)
    # value: (batch_size, ..., seq_len, d_v)
    # mask: (seq_len, seq_len)
    dot_product = einsum(query, key, "... s1 d_k, ... s2 d_k -> ... s1 s2")
    scaled_dot_product = dot_product / math.sqrt(key.shape[-1])

    if mask is not None:
        numeric_mask = torch.where(mask, 0.0, float("-inf"))
        scaled_dot_product += numeric_mask

    scaled_dot_product = softmax(scaled_dot_product, dim=-1)
    return einsum(scaled_dot_product, value, "... s1 s2, ... s2 d_v -> ... s1 d_v")



class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            d_key: int | None = None,
            d_value: int | None = None,
            enable_position_embedding: bool = True,
            rope_theta: int = 10000, 
            max_seq_len: int = 2048,
            device=None,
            dtype=None
            ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_key = d_model // num_heads if d_key is None else d_key
        self.d_value = d_model // num_heads if d_value is None else d_value
        self.enable_position_embedding = enable_position_embedding

        assert self.d_key == self.d_value

        self.qkv_proj = nn.Parameter(torch.empty(2 * self.num_heads * self.d_key + self.num_heads * self.d_value, self.d_model, device=device, dtype=dtype))
        self.out_proj = nn.Parameter(torch.empty(self.d_model, self.num_heads * self.d_value, device=device, dtype=dtype))
        self.position_embedding = RotaryPositionalEmbedding(rope_theta, d_k=self.d_key, max_seq_len=max_seq_len, device=device) if enable_position_embedding else None
        nn.init.trunc_normal_(self.qkv_proj)
        nn.init.trunc_normal_(self.out_proj)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = einsum(self.qkv_proj, x, "three_hd_k d_model, b seq d_model -> b seq three_hd_k")
        qkv = rearrange(qkv, "b seq (channel h d_k) -> channel b h seq d_k", channel=3, h=self.num_heads)

        if self.enable_position_embedding:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).broadcast_to((batch_size, self.num_heads, seq_len))
            Q = self.position_embedding(qkv[0], token_positions)
            K = self.position_embedding(qkv[1], token_positions)
        else:
            Q = qkv[0]
            K = qkv[1]
        V = qkv[2] 
        
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn = scaled_dot_product_attention(K, Q, V, mask) # (batch, head, seq_len, d_v)
        attn = rearrange(attn, "b h seq d_v -> b seq (h d_v)")
        return einsum(self.out_proj, attn, "d_model hd_v, b seq hd_v -> b seq d_model")
    
        
