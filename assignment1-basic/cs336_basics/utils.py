import torch
import math
from typing import Iterable
import numpy.typing as npt
import numpy as np
import os
import typing

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
    params = [p for p in parameters if p.grad is not None]

    total_norm = torch.sqrt(
        sum(p.grad.norm(2) ** 2 for p in params)
    )

    clip_coef = max_l2_norm / (total_norm + eps)

    if clip_coef < 1:
        for p in params:
            p.grad.mul_(clip_coef)


def load_data(
    dataset: np.memmap,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:

    starts = np.random.randint(
        0,
        len(dataset) - context_length,
        size=batch_size,
    )

    data_np = np.stack([
        dataset[i:i + context_length + 1]
        for i in starts
    ])

    # Usually token IDs need torch.long
    data_np = data_np.astype(np.int32)

    data = torch.from_numpy(data_np)

    x = data[:, :-1].to(device)
    y = data[:, 1:].to(device)

    return x, y

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, out)


def load_checkpoint(
        src,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer
):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]