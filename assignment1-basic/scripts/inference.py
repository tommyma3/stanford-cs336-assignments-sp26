import torch
import numpy as np
from cs336_basics.model import TransformerLM, softmax
from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import load_checkpoint
import warnings
import yaml

def _get_eos_token_id(
        tokenizer: Tokenizer,
        eos_token_bytes: bytes = b"<|endoftext|>"
) -> int:
    return tokenizer.token_to_id.get(eos_token_bytes, -1)


@torch.no_grad()
def decode(
        model: TransformerLM,
        tokenizer: Tokenizer,
        prompt: list[int],
        max_generated_tokens: int | None = None,
        temperature: float = 0,
        top_p : float | None = None
) -> list[int]:
    assert temperature >= 0
    assert top_p is None or 0 < top_p <= 1

    if max_generated_tokens is None:
        max_generated_tokens = model.context_length - len(prompt)

    if max_generated_tokens + len(prompt) > model.context_length:
        max_generated_tokens = model.context_length - len(prompt)
        warnings.warn(f"Max new tokens exceeding context length, reducing to {max_generated_tokens}.")

    use_top_p = top_p is not None
    num_new_tokens = 0
    eos_token_id = _get_eos_token_id(tokenizer=tokenizer)

    prompt_tensor = torch.from_numpy(np.asarray(prompt, dtype=np.int32)).unsqueeze(0).to(model.device)
    current_len = len(prompt)
    buf = torch.full((1, current_len + max_generated_tokens), eos_token_id, dtype=torch.int32, device=model.device)
    buf[:, :current_len] = prompt_tensor

    with torch.no_grad():
        for _ in range(max_generated_tokens):
            next_token_logits = model(buf[:, :current_len])[:, -1]
            if temperature == 0:
                next_token = next_token_logits.argmax(-1, keepdim=True)
            else:
                next_token_logits /= temperature
                next_token_probs = softmax(next_token_logits, dim=-1)
                if use_top_p:
                    sorted_probs, sorted_idx = torch.sort(next_token_probs, dim=-1, descending=True)
                    cum_excl = sorted_probs.cumsum(-1) - sorted_probs
                    remove = cum_excl >= top_p
                    remove = remove.scatter(-1, sorted_idx, remove)
                    next_token_probs = next_token_probs.masked_fill(remove, 0)
                    next_token_probs = next_token_probs / next_token_probs.sum()

                next_token = torch.multinomial(next_token_probs, num_samples=1)

            buf[:, current_len] = next_token.squeeze(-1)
            current_len += 1
            
            if next_token == eos_token_id:
                break

        return buf[0, len(prompt):current_len].tolist()


def inference(
        model: TransformerLM,
        tokenizer: Tokenizer,
        prompt: str,
        max_generated_tokens: int | None = None,
        temperature: float = 0,
        top_p : float | None = None
) -> str:
    prompt_ids = tokenizer.encode(prompt)
    completion_ids = decode(model, tokenizer, prompt_ids, max_generated_tokens, temperature, top_p)
    completion = tokenizer.decode(completion_ids)
    return completion


if __name__ == "__main__":
    prompt = "Once upon a time"
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer = Tokenizer.from_files(config["tokenizer"]["vocab_filepath"], config["tokenizer"]["merges_filepath"], config["tokenizer"]["special_tokens"])
    vocab_size = len(tokenizer.vocab)
    context_length = config["model"]["context_length"]
    d_model = config["model"]["d_model"]
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]
    d_feedforward = config["model"]["d_feedforward"]
    lr = 0.0
    betas = (config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    eps = float(config["optimizer"]["eps"])
    weight_decay = config["optimizer"]["weight_decay"]
    max_l2_norm = config["training"]["max_l2_norm"]

    device = torch.device(config["model"]["device"])
    dtype = getattr(torch, config["model"]["dtype"])

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_feedforward, device=device, dtype=dtype)
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    load_checkpoint("artifacts/train/iter49999.pt", model, optimizer)

    completion = inference(model, tokenizer, prompt)
    print(prompt, completion)