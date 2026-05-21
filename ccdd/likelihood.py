import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
import torch.distributed as dist
from transformers import PreTrainedModel


class ELBO(nn.Module):
    def __init__(self, config, model, noise_schedule, loss_fn):
        super().__init__()
        self.config = config
        self.model = model
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn

    def forward(self, input_ids, attention_mask, t):
        # z_t, tgt_features = self.noise_schedule.sample_zt(input_ids, t)
        z_t = self.noise_schedule.sample_zt(input_ids, t)
        outputs = self.model(z_t, t)
        # _, elbo, _ = self.loss_fn(outputs, tgt_features, input_ids, attention_mask, z_t, t, reduction="none")
        _, elbo, _ = self.loss_fn(outputs, input_ids, attention_mask, z_t, t, reduction="none")

        return elbo


def compute_elbo(elbo_fn: ELBO, batch, num_samples=128, t_eps=1e-4, return_token_nlls=False, reduce_metrics=False, show_progress=True):
    device = batch["input_ids"].device
    ts = torch.linspace(t_eps, 1 - t_eps, num_samples, device=device)

    elbos = []
    for i in tqdm.trange(num_samples, disable=not show_progress):
        t = ts[i, None].expand(batch["input_ids"].shape[0])
        elbo = elbo_fn(batch["input_ids"], batch["attention_mask"], t)
        elbos.append(elbo.cpu())
    elbos = torch.stack(elbos, dim=0).to(device)

    token_nlls = elbos.mean(dim=0)
    total_nll = (token_nlls * batch["attention_mask"]).sum()
    total_tokens = batch["attention_mask"].sum()
    total_batch_size = torch.tensor(batch["input_ids"].size(0), device=device)
    if reduce_metrics and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_nll)
        dist.all_reduce(total_tokens)
        dist.all_reduce(total_batch_size)

    nll = total_nll / total_tokens
    seq_nll = total_nll / total_batch_size

    metrics = {
        "nll": nll,
        "ppl": nll.exp(),
        "seq_nll": seq_nll,
    }

    return (metrics, token_nlls) if return_token_nlls else metrics


def compute_causal_nll(model: PreTrainedModel, batch, reduce_metrics=False, return_token_nlls=False):
    labels = batch["input_ids"][:, 1:]
    loss_mask = batch["attention_mask"][:, :-1]

    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[:, :-1]
    loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    
    total_nll = (loss * loss_mask).sum()
    total_tokens = loss_mask.sum()
    total_batch_size = torch.tensor(batch["input_ids"].size(0), device=loss.device)

    if reduce_metrics and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_nll)
        dist.all_reduce(total_tokens)
        dist.all_reduce(total_batch_size)

    nll = total_nll / total_tokens
    seq_nll = total_nll / total_batch_size

    metrics = {
        "nll": nll,
        "ppl": nll.exp(),
        "seq_nll": seq_nll,
        "seq_ppl": seq_nll.exp(),
    }

    return (metrics, loss) if return_token_nlls else metrics