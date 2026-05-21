import re
import math

import torch


def parse_dtype(dtype):
    if dtype == "fp16":
        return torch.float16
    elif dtype == "fp32":
        return torch.float32
    elif dtype == "bf16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_lr(config, lr, step):
    lr_schedule = config.training.lr_schedule
    warmup_steps = config.training.warmup_steps
    num_train_steps = config.training.num_train_steps

    if lr_schedule == "constant":
        return lr
    elif lr_schedule == "linear":
        return lr * min(1, step / warmup_steps, 1 - (step - warmup_steps) / (num_train_steps - warmup_steps))
    elif lr_schedule == "cosine":
        if step < warmup_steps:
            return lr * step / warmup_steps
        else:
            return lr * (0.1 + 0.9*(1 + math.cos(math.pi * (step - warmup_steps) / (num_train_steps - warmup_steps))) / 2)
    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")


@torch.no_grad()
def sample_categorical(probs, generator=None):
    return torch.distributions.Categorical(probs=probs).sample()


def calculate_flops_per_batch(config, model, vocab_size, non_emb_params=None, method="hoffmann"):
    if method == "kaplan":
        assert non_emb_params is not None
        flops_per_token = 2 * (non_emb_params + config.model.n_blocks * config.model.hidden_size * config.model.max_seq_len)
        flops_per_sample = 3 * config.model.max_seq_len * flops_per_token
    elif method == "hoffmann":
        seq_len = config.model.max_seq_len
        d_model = config.model.hidden_size
        num_heads = config.model.n_heads
        mlp_ratio = 4
        num_layers = config.model.n_blocks

        emb_flops = 2 * seq_len * vocab_size * d_model
        attn_flops = (
            2 * 3 * seq_len * d_model**2
            + 2 * seq_len**2 * d_model
            + 3 * num_heads * seq_len**2
            + 2 * seq_len**2 * d_model
            + 2 * seq_len * d_model**2
        )
        mlp_flops = 2 * seq_len * 2 * d_model * (mlp_ratio * d_model)
        layer_flops = attn_flops + mlp_flops
        final_flops = 2 * seq_len * d_model * vocab_size

        if config.model.type == "diffusion":
            freq_dim = model.sigma_map.mlp[0].in_features
            cond_dim = config.model.cond_dim
            emb_flops += 2 * (freq_dim * d_model + d_model * d_model)
            layer_flops += 2 * (cond_dim * 6 * d_model)
            final_flops += 2 * (cond_dim * 2 * d_model)

        flops_per_sample = 3 * (emb_flops + num_layers * layer_flops + final_flops)
    else:
        raise ValueError(f"Unknown method: {method}")
    flops_per_batch = flops_per_sample * config.training.train_batch_size
    return flops_per_batch