from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm

from ccdd.diffusion_process import NoiseSchedule
from ccdd.utils import sample_categorical


class Sampler(nn.Module):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps: float = 1e-4):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.t_eps = t_eps

    @abstractmethod
    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, num_samples=1, num_denoising_steps=1000, max_length=None, decode=True, show_progress=True):
        max_length = max_length or self.model.config.max_seq_len
        device = next(self.model.parameters()).device

        z_t = self._do_generate(num_samples, num_denoising_steps, max_length, show_progress=show_progress, device=device)

        if decode:
            texts = self.tokenizer.batch_decode(z_t, skip_special_tokens=True)
            return texts
        else:
            return z_t


class GiddSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, tokenizer, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.tokenizer = tokenizer
            self.min_p = min_p

        def forward(self, z_t, t, s):
            logits = self.model(z_t, t)
            logits[..., self.tokenizer.mask_token_id] = -1e6

            q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s)
            q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t)
            q_zt = q_t.gather(-1, z_t.unsqueeze(-1))

            alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t)
            alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s)

            alpha_ts = alpha_t / alpha_s
            beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

            vz_t = F.one_hot(z_t, num_classes=len(self.tokenizer))
            beta_pi_ts_at_zt = beta_pi_ts.unsqueeze(1).expand_as(vz_t).gather(-1, z_t.unsqueeze(-1))
            q_ts = (alpha_ts * vz_t + beta_pi_ts_at_zt)

            q_st = q_ts * q_s / q_zt
            if self.min_p > 0.0:
                is_small = (q_st < self.min_p).float()
                q_st = (1 - is_small) * q_st
                q_st = q_st / q_st.sum(-1, keepdim=True)
            return sample_categorical(q_st)

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):

        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)]).clone()
        return z_t


class MDLMSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, mask_id, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.mask_id = mask_id
            self.min_p = min_p

        def get_sigmas(self, t, eps=1e-4):
            dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
            sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
            return dsigma, sigma

        def forward(self, z_t, t, tm1, i=None, eps=1e-4):
            logits = self.model(z_t, t)
            logits[..., self.mask_id] = -1e6

            if i == 0:
                z_tm1 = logits.argmax(-1)
            else:
                _, sigma_t = self.get_sigmas(t, eps=eps)
                _, sigma_tm1 = self.get_sigmas(tm1, eps=eps)

                move_chance_t = 1 - torch.exp(-sigma_t)
                move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                move_chance_t = move_chance_t[:, None, None]
                move_chance_tm1 = move_chance_tm1[:, None, None]
                probs = logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                probs[:, :, self.mask_id] = move_chance_tm1[:, :, 0]
                probs /= move_chance_t
                if self.min_p > 0.0:
                    is_small = (probs < self.min_p).float()
                    probs = (1 - is_small) * probs
                    probs = probs / probs.sum(-1, keepdim=True)
                z_tm1 = sample_categorical(probs)

            copy_flag = (z_t != self.mask_id).to(z_t.dtype)
            z_t = copy_flag * z_t + (1 - copy_flag) * z_tm1
            return z_t

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer.mask_token_id, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        z_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)

        ts = torch.linspace(self.t_eps, 1 - self.t_eps, num_denoising_steps + 1, device=device).unsqueeze(-1)

        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress):
            z_t = self.sampling_step(z_t, ts[i], ts[max(0, i-1)], i=i, eps=self.t_eps).clone()

        return z_t
    

class CCDDSampler(Sampler):
    class DenoisingStep(nn.Module):
        def __init__(self, model, noise_schedule, tokenizer, min_p=0.0):
            super().__init__()
            self.model = model
            self.noise_schedule = noise_schedule
            self.tokenizer = tokenizer
            self.min_p = min_p
            self.latent_dim = model.config.model.get("latent_dim", model.config.model.hidden_size)
            self.mask_id = tokenizer.mask_token_id

        def get_sigmas(self, t, eps=1e-4):
            dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
            sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
            return dsigma, sigma

        def forward(self, x_t, z_t, t, s, disc_type="ccdd", cont_type="ddpm", last_step=False, w=1.0):
            logits, z_pred = self.model(x_t, z_t, t)
            logits = logits[:, :, :len(self.tokenizer)]
            logits[..., self.tokenizer.mask_token_id] = -1e6

            if w != 1.0:
                logits_single, _ = self.model(x_t, torch.zeros_like(z_t), t)
                logits_single = logits_single[:, :, :len(self.tokenizer)]
                logits_single[..., self.tokenizer.mask_token_id] = -1e6
                logits = w * logits + (1 - w) * logits_single

            if disc_type == "ccdd":
                c_t = self.noise_schedule.get_c_t(t)
                c_s = self.noise_schedule.get_c_t(s)
                q_s = self.noise_schedule.probs_at_t(logits.softmax(-1), s, c_t=c_s)
                q_t = self.noise_schedule.probs_at_t(logits.softmax(-1), t, c_t=c_t)
                q_xt = q_t.gather(-1, x_t.unsqueeze(-1))

                alpha_t, beta_pi_t = self.noise_schedule.get_alpha_betapi(t, c_t=c_t)
                alpha_s, beta_pi_s = self.noise_schedule.get_alpha_betapi(s, c_t=c_s)

                alpha_ts = alpha_t / alpha_s
                beta_pi_ts = beta_pi_t - alpha_t / alpha_s * beta_pi_s

                vx_t = F.one_hot(x_t, num_classes=len(self.tokenizer))  # len(self.tokenizer)
                beta_pi_ts_at_xt = beta_pi_ts.expand_as(vx_t).gather(-1, x_t.unsqueeze(-1))
                q_ts = (alpha_ts * vx_t + beta_pi_ts_at_xt)

                q_st = q_ts * q_s / q_xt
                if self.min_p > 0.0:
                    is_small = (q_st < self.min_p).float()
                    q_st = (1 - is_small) * q_st
                    q_st = q_st / q_st.sum(-1, keepdim=True)
                x_s = sample_categorical(q_st)
            elif disc_type == "mdlm":
                if last_step:
                    x_tm1 = logits.argmax(-1)
                else:
                    _, sigma_t = self.get_sigmas(t)
                    _, sigma_tm1 = self.get_sigmas(s)
    
                    move_chance_t = 1 - torch.exp(-sigma_t)
                    move_chance_tm1 = 1 - torch.exp(-sigma_tm1)
                    move_chance_t = move_chance_t[:, None, None]
                    move_chance_tm1 = move_chance_tm1[:, None, None]
                    probs = logits.softmax(-1) * (move_chance_t - move_chance_tm1)
                    probs[:, :, self.mask_id] = move_chance_tm1[:, :, 0]
                    probs /= move_chance_t
                    if self.min_p > 0.0:
                        is_small = (probs < self.min_p).float()
                        probs = (1 - is_small) * probs
                        probs = probs / probs.sum(-1, keepdim=True)
                    x_tm1 = sample_categorical(probs)

                copy_flag = (x_t != self.mask_id).to(x_t.dtype)
                x_s = copy_flag * x_t + (1 - copy_flag) * x_tm1
            elif disc_type == "metric":
                ar_prior = 0.05
            else:
                raise ValueError(f"Unsupported discrete type: {disc_type}")
            
            if self.noise_schedule.cont_schedule in ["old", "vp_linear"]:
                if self.noise_schedule.cont_schedule == "old":
                    alpha_t, alpha_s = 1 - t, 1 - s
                else:
                    alpha_t, alpha_s = torch.sqrt(1 - t), torch.sqrt(1 - s)
                sigma_t, sigma_s = torch.sqrt(1 - alpha_t ** 2), torch.sqrt(1 - alpha_s ** 2)
                if cont_type == "ddpm":
                    noise = torch.randn_like(z_pred) if not last_step else 0
                    z_s = alpha_s * (alpha_s ** 2 - alpha_t ** 2) / (alpha_s ** 2 * (1 - alpha_t ** 2)) * z_t + alpha_t * (1 - alpha_s ** 2) / (1 - alpha_t ** 2) * z_pred + (1 - alpha_s ** 2) * (alpha_s ** 2 - alpha_t ** 2) / (alpha_s ** 2 * (1 - alpha_t ** 2)) * noise
                elif cont_type == "ddim":
                    z_s = alpha_s * z_pred + sigma_s / sigma_t * (z_t - alpha_t * z_pred)
                else:
                    raise ValueError(f"Unsupported continuous type: {cont_type}")
            else:
                raise ValueError(f"Unsupported continuous type: {self.noise_schedule.cont_schedule}")
            return x_s, z_s

    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, t_eps=1e-4, compile_step=True, min_p=0.0):
        super().__init__(model, tokenizer, noise_schedule, t_eps=t_eps)
        self.sampling_step = self.DenoisingStep(model, noise_schedule, tokenizer, min_p=min_p)
        if compile_step:
            self.sampling_step = torch.compile(self.sampling_step)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None, dtype=torch.bfloat16, disc_type="ccdd", cont_type="ddpm", w=1.0):

        ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
        ts = (1 - 2 * self.t_eps) * ts + self.t_eps

        x_t = self.noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
        z_t = torch.randn(num_samples, max_length, self.sampling_step.latent_dim, device=device, dtype=dtype)
        
        for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Generating samples", disable=not show_progress, dynamic_ncols=True):
            x_t, z_t = self.sampling_step(x_t, z_t, ts[i], ts[max(0, i-1)], disc_type=disc_type, cont_type=cont_type, last_step=(i == 0), w=w)
        return x_t, z_t
    
    @torch.no_grad()
    def generate(self, num_samples=1, num_denoising_steps=1000, max_length=None, decode=True, show_progress=True, dtype=torch.bfloat16, disc_type="ccdd", cont_type="ddpm", w=1.0):
        max_length = max_length or self.model.config.model.max_seq_len
        device = next(self.model.parameters()).device

        x_t, z_t = self._do_generate(num_samples, num_denoising_steps, max_length, show_progress=show_progress, device=device, dtype=dtype, disc_type=disc_type, cont_type=cont_type, w=w)

        if decode:
            texts = self.tokenizer.batch_decode(x_t, skip_special_tokens=True)
            return texts
        else:
            return x_t, z_t


class AutoregressiveSampler(Sampler):
    def __init__(self, model, tokenizer, noise_schedule: NoiseSchedule, compile_step=True):
        super().__init__(model, tokenizer, noise_schedule)
        if compile_step:
            self.model = torch.compile(model)

    def _do_generate(self, num_samples, num_denoising_steps, max_length, show_progress=False, device=None):
        bos_token_id = self.tokenizer.cls_token_id or self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id

        input_ids = torch.full((num_samples, max_length), eos_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((num_samples, max_length), dtype=torch.long, device=device)
        input_ids[:, 0] = bos_token_id
        attention_mask[:, 0] = 1

        done = torch.zeros(num_samples, device=device)
        for i in tqdm.trange(1, max_length, desc="Generating samples", disable=not show_progress):
            logits = self.model(input_ids, use_cache=False).logits[:, i-1]
            probs = logits.softmax(-1)
            next_x = (1 - done) * sample_categorical(probs) + done * self.tokenizer.pad_token_id
            input_ids[:, i] = next_x.to(input_ids.dtype)
            done += (1 - done) * (next_x == eos_token_id).to(done.dtype)
            if (done == 1).all():
                break

        return input_ids


def get_sampler(config, model, tokenizer, noise_schedule: NoiseSchedule, compile_step=True, min_p=0.0):
    if config.model.type in ["diffusion", "mmdit", "moedit", "mdit"]:
        if config.model.diffusion_process == "ccdd":
            return GiddSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p)
        elif config.model.diffusion_process == "mdlm":
            return MDLMSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p)
        elif config.model.diffusion_process == "coevolutionary":
            return CCDDSampler(model, tokenizer, noise_schedule, t_eps=config.model.t_eps, compile_step=compile_step, min_p=min_p)
        else:
            raise ValueError(f"Unsupported forward process: {config.model.diffusion_process}")
    elif config.model.type == "autoregressive":
        return AutoregressiveSampler(model, tokenizer, noise_schedule, compile_step=True)
    else:
        raise ValueError(f"Unsupported model type: {config.model.type}")
