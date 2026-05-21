import torch
import torch.nn as nn
import torch.distributed as dist

from ccdd.diffusion_process import sample_t, NoiseSchedule
from ccdd.loss import Loss


class DiffusionTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, loss_fn: Loss, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn
        self.dtype = dtype

        self.device = next(model.parameters()).device

        self.register_buffer("pad_id", torch.tensor(tokenizer.pad_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("mask_id", torch.tensor(tokenizer.mask_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("t0", torch.zeros(1, device=self.device))
        self.register_buffer("t1", torch.ones(1, device=self.device))

    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch):
        batch_size = batch["input_ids"].size(0)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            t = sample_t(self.config, batch_size, device=self.device)
            z_t = self.noise_schedule.sample_zt(batch["input_ids"], t)

            logits = self.model(z_t, t)
            loss, _, metrics = self.loss_fn.forward(
                logits=logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                z_t=z_t,
                t=t,
                reduction=self.config.loss.reduction,
            )
        return loss, metrics
    

class BertTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, loss_fn: Loss, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn
        self.dtype = dtype

        self.device = next(model.parameters()).device

        self.register_buffer("pad_id", torch.tensor(tokenizer.pad_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("mask_id", torch.tensor(tokenizer.mask_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("t0", torch.zeros(1, device=self.device))
        self.register_buffer("t1", torch.ones(1, device=self.device))

    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch):
        batch_size = batch["input_ids"].size(0)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            t = sample_t(self.config, batch_size, device=self.device)
            z_t = self.noise_schedule.sample_zt(batch["input_ids"], t)

            logits = self.model(z_t, attention_mask=batch["attention_mask"]).logits
            loss, _, metrics = self.loss_fn.forward(
                logits=logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                x_t=z_t,
                t=t,
                reduction=self.config.loss.reduction,
            )
        return loss, metrics
    

class CoevolutionaryDiffusionTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, noise_schedule: NoiseSchedule, loss_fn: Loss, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn
        self.dtype = dtype

        self.device = next(model.parameters()).device

        self.register_buffer("pad_id", torch.tensor(tokenizer.pad_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("mask_id", torch.tensor(tokenizer.mask_token_id, device=self.device, dtype=torch.long))
        self.register_buffer("t0", torch.zeros(1, device=self.device))
        self.register_buffer("t1", torch.ones(1, device=self.device))

    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch, p_u=None, p_r=None, resample_t=None):
        batch_size = batch["input_ids"].size(0)

        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            with torch.no_grad():
                t = sample_t(self.config, batch_size, device=self.device).to(self.dtype)
                
                # Handle resample_t to avoid torch.compile issues
                if resample_t is not None:
                    if isinstance(resample_t, str) and resample_t == "zero":
                        resample_t = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
                    elif isinstance(resample_t, torch.Tensor):
                        resample_t = resample_t.to(self.dtype)
                    elif isinstance(resample_t, bool) and resample_t:
                        resample_t = sample_t(self.config, batch_size, device=self.device).to(self.dtype)
                    else:
                        resample_t = t.clone()
                else:
                    resample_t = None
                    
                x_t, z_t, latents, p_u_sample = self.noise_schedule.sample_zt(batch["input_ids"], t, attention_mask=batch["attention_mask"], p_u=p_u, p_r=p_r, resample_t=resample_t)
                
                if (self.training and self.config.training.get("p_drop_continuous", 0.0) > 0) or self.config.training.get("p_drop_continuous", 0.0) == 1:
                    cont_mask = torch.rand(z_t.shape[0], device=z_t.device) >= self.config.training.get("p_drop_continuous", 0.0)
                    z_t[~cont_mask] = 0
                else:
                    cont_mask = torch.ones(z_t.shape[0], device=z_t.device, dtype=torch.bool)
                if (self.training and self.config.training.get("p_drop_discrete", 0.0) > 0) or self.config.training.get("p_drop_discrete", 0.0) == 1:
                    disc_mask = torch.rand(x_t.shape[0], device=x_t.device) >= self.config.training.get("p_drop_discrete", 0.0)
                    x_t[~disc_mask] = self.tokenizer.mask_token_id
                else:
                    disc_mask = torch.ones(x_t.shape[0], device=x_t.device, dtype=torch.bool)

            logits, z_pred = self.model(x_t, z_t, t, attention_mask=batch["attention_mask"], c_cont=resample_t)
            z_pred[~cont_mask] = 0
            loss, _, metrics = self.loss_fn.forward(
                logits=logits,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                x_t=x_t,
                t=t,
                z_pred=z_pred,
                latents=latents,
                z_t=z_t,
                p_u=p_u_sample,
                reduction=self.config.loss.reduction,
            )
        return loss, metrics


class AutoregressiveTrainer(nn.Module):
    def __init__(self, config, model, tokenizer, loss_fn, dtype=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.dtype = dtype
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.device = next(model.parameters()).device
    
    def to(self, device=None, dtype=None):
        self.device = device if device else self.device
        self.dtype = dtype if dtype else self.dtype
        return super().to(device, dtype)

    def forward(self, batch):
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            labels = batch["input_ids"][:, 1:]
            loss_mask = batch["attention_mask"][:, :-1]

            logits = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            logits = logits[:, :-1]
            loss = self.loss_fn(logits.transpose(1, 2), labels)
            total_loss = (loss * loss_mask).sum()
            total_tokens = loss_mask.sum().float()

            if self.world_size > 1:
                dist.all_reduce(total_tokens)
                total_tokens /= self.world_size

            loss = total_loss / total_tokens

        return loss, {
            "elbo": loss.detach(),
            "nll": loss.detach(),
            "ppl": loss.detach().exp(),
        }


def get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype=None):
    if config.model.type == "diffusion":
        return DiffusionTrainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    elif config.model.type == "autoregressive":
        return AutoregressiveTrainer(config, model, tokenizer, loss_fn, dtype)
    elif config.model.type in ["coevolutionary", "mmdit", "moedit", 'mdit']:
        return CoevolutionaryDiffusionTrainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    elif config.model.type == "bert":
        return BertTrainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")
