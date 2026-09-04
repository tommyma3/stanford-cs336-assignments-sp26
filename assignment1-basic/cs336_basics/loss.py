import torch
from einops import reduce

def cross_entropy(logit: torch.Tensor, target: torch.Tensor):
    target_logits = torch.gather(logit, dim=-1, index=target.unsqueeze(-1))
    max_elements = torch.amax(logit, dim=-1, keepdim=True)
    exponents = torch.exp(logit - max_elements)
    loss = max_elements + torch.log(reduce(exponents, "... vocab -> ... 1", "sum")) - target_logits
    return loss.mean()

