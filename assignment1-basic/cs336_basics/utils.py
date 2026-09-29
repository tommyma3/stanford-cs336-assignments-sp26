import torch
import math
from typing import Iterable

def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int
):
    assert max_learning_rate >= min_learning_rate
    assert warmup_iters < cosine_cycle_iters
    assert it >= 0

    if it < warmup_iters:
        return it * max_learning_rate / warmup_iters
    elif it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def gradient_clipping(
        parameters: Iterable[torch.nn.Parameter], 
        max_l2_norm: float,
        eps: float = 1e-6
):
    for p in parameters:
        if p.grad is not None and p.grad.norm(p=2) >= max_l2_norm:
            p.grad = max_l2_norm