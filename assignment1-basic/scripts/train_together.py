import yaml
import os
from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.utils import load_data, gradient_clipping, get_lr_cosine_schedule, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.loss import cross_entropy
import torch
import wandb

def main():
    with open("configs/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_data_path = config["data"]["train"]
    eval_data_path = config["data"]["eval"]
    tokenizer = Tokenizer.from_files(config["tokenizer"]["vocab_filepath"], config["tokenizer"]["merges_filepath"], config["tokenizer"]["special_tokens"])
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_data_string = f.read()
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data_string = f.read()

    # 1. Generate dataset
    train_dataset = tokenizer.encode(train_data_string)
    eval_dataset = tokenizer.encode(eval_data_string)
    batch_size = config["training"]["batch_size"]
    context_length = config["model"]["context_length"]
    device = config["model"]["device"]

    # 2. Model initialization
    vocab_size = len(tokenizer.vocab)
    context_length = config["model"]["context_length"]
    d_model = config["model"]["d_model"]
    num_layers = config["model"]["num_layers"]
    num_heads = config["model"]["num_heads"]
    d_feedforward = config["model"]["d_feedforward"]
    dtype = config["model"]["dtype"]

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_feedforward, device=device, dtype=dtype)
    model = torch.compile(model)

    # 3. Training loop
    iterations = config["training"]["iterations"]
    min_lr = config["training"]["min_lr"]
    max_lr = config["training"]["max_lr"]
    warmup_iters = config["training"]["warmup_iters"]
    cosine_cycle_iters = config["training"]["cosine_cycle_iters"]
    checkpoint_interval = config["training"]["checkpoint_interval"]
    eval_interval = config["training"]["eval_interval"]
    log_interval = config["training"]["log_interval"]
    ckpt_dir = config["training"]["ckpt_dir"]
    lr = 0.0
    betas = (config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    eps = config["optimizer"]["eps"]
    weight_decay = config["optimizer"]["weight_decay"]
    max_l2_norm = config["training"]["max_l2_norm"]

    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    run = wandb.init(
        project="cs336-assignment1",
        config=config
    )
    run.define_metric("iteration")
    run.define_metric("train/*", step_metric="iteration")
    run.define_metric("eval/*", step_metric="iteration")

    os.makedirs(ckpt_dir, exist_ok=True)

    for it in range(iterations):
        lr = get_lr_cosine_schedule(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters)
        
        for group in optimizer.param_groups:
            group["lr"] = lr

        data, labels = load_data(train_dataset, batch_size, context_length, device)
        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(logits, labels)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)
        optimizer.step()

        if it == 0 or (it + 1) % log_interval == 0:
            run.log({
                "iteration": it,
                "train/loss": loss.item(),
                "train/perplexity": torch.exp(loss).item(),
                "train/lr": lr,
            })
            print(f"TRAIN: Iteration {it} loss: {loss.item()}")

        if (it + 1) % eval_interval == 0:
            with torch.no_grad():
                eval_data, eval_labels = load_data(eval_dataset, batch_size, context_length, device)
                eval_logits = model(eval_data)
                eval_loss = cross_entropy(eval_logits, eval_labels)
                print(f"EVAL: Iteration {it} eval loss: {eval_loss.item()}")
                run.log({
                    "iteration": it,
                    "eval/loss": eval_loss.item(),
                })

        if (it + 1) % checkpoint_interval == 0:
            output_path = f"{ckpt_dir}/iter{it}.pt"
            save_checkpoint(model, optimizer, iteration=it, out=output_path)
            print(f"Iteration {it} checkpoint saved.")
    

if __name__ == "__main__":
    main()