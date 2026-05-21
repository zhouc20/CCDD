from abc import abstractmethod, ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(torch.nn.Module, ABC):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.noise_schedule = noise_schedule
        self.vocab_size = len(tokenizer)

    @abstractmethod
    def loss(self, logits, input_ids, attention_mask, x_t, t):
        raise NotImplementedError

    def forward(self, logits, input_ids, attention_mask, x_t, t, z_pred=None, latents=None, z_t=None, p_u=0.0, reduction="tokenmean"):
        loss, elbo, metrics = self.loss(logits, input_ids, attention_mask, x_t, t, z_pred, latents, z_t, p_u)

        if reduction == "tokenmean":
            loss = (loss * attention_mask).sum() / attention_mask.sum()
        else:
            pass

        return loss, elbo, metrics


class GiddLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id
        self.loss_weighting = config.loss.loss_weighting
        self.min_loss_weight = config.loss.min_loss_weight
        self.max_loss_weight = config.loss.max_loss_weight
        assert self.max_loss_weight > 0, "max_loss_weight must be positive"

    def get_weights(self, t, z_t, input_ids):
        orig_dtype = t.dtype
        t = t.unsqueeze(-1).to(torch.float64)
        t1m = (1 - t)

        gamma = self.noise_schedule.log_gamma.exp()
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)
        B = self.noise_schedule.log_B.exp()

        c_t = t_gamma.sqrt() * t1m_gamma.sqrt() * B
        c_t_prime = (gamma / 2) * (1 - 2 * t) / (t * t1m) * c_t

        is_mask = (z_t == self.mask_id).to(t.dtype)
        is_x = (z_t == input_ids).to(t.dtype)

        alpha_ratio = -1 / (1 - t) - c_t_prime / (1 + c_t)
        N = self.vocab_size - 1
        weight_on_x = (c_t + (1 - t) * c_t_prime) / N / ((1 - t)*(1 - t + c_t/N))
        weight_on_u = (c_t + (1 - t) * c_t_prime) / ((1 - t)*c_t)
        weight_on_m = 1 / ((1 - t)*t)

        elbo_weights = is_x * weight_on_x + is_mask * weight_on_m + (1 - is_x - is_mask) * weight_on_u

        loss_weights = elbo_weights.clone()
        if self.loss_weighting == "clip":
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "dynamic":
            log_snr = torch.sigmoid(-t).clip(-20, 20)  # not exactly the log-SNR, but close enough if C_t is close to 1
            x_scale = B / self.vocab_size * torch.exp(gamma / 2 * log_snr)
            loss_weights = (1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)

        return alpha_ratio.to(orig_dtype), elbo_weights.to(orig_dtype), loss_weights.to(orig_dtype)

    def loss(self, logits, input_ids, attention_mask, z_t, t):
        dtype = logits.dtype
        alpha_ratio, elbo_weights, ws = self.get_weights(t, z_t, input_ids)

        logits[..., self.mask_id] = torch.finfo(dtype).min

        x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
        x_hat = logits.softmax(-1).to(dtype)  # prevent automatic upcasting
        log_q_t = self.noise_schedule.probs_at_t(x, t).log_().clip_(min=-1e6)
        log_p_t = self.noise_schedule.probs_at_t(x_hat, t).log_().clip_(min=-1e6)

        kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)

        log_q_zt = log_q_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_p_zt = log_p_t.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
        log_ratio = log_q_zt - log_p_zt

        is_loss = log_ratio.exp() - log_ratio - 1
        elbo = elbo_weights * (kl_loss + is_loss)

        loss = ws * (kl_loss + is_loss)

        metrics = {
            "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "is_loss": (ws * is_loss.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
        }

        return loss, elbo, metrics
    

class CoevolutionaryGiddLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id
        self.loss_weighting = config.loss.loss_weighting
        self.min_loss_weight = config.loss.min_loss_weight
        self.max_loss_weight = config.loss.max_loss_weight
        assert self.max_loss_weight > 0, "max_loss_weight must be positive"
        self.weight_type = config.loss.weight_type
        self.cont_weight = config.loss.get("cont_weight", 1.0)

    def get_weights(self, t, z_t, input_ids, p_u=0.0):
        orig_dtype = t.dtype
        t = t[:, None]
        t1m = (1 - t)

        gamma = self.noise_schedule.gamma.to(orig_dtype)
        t_gamma = t.pow(gamma)
        t1m_gamma = t1m.pow(gamma)
        B = 2.0**gamma * p_u / (1 - p_u)

        c_t = (t_gamma.sqrt() * t1m_gamma.sqrt() * B).clip(min=1e-30).to(orig_dtype)
        c_t_prime = (gamma / 2) * (1 - 2 * t) / (t * t1m).clip(min=1e-30) * c_t

        is_mask = (z_t == self.mask_id).to(t.dtype)
        is_x = (z_t == input_ids).to(t.dtype)

        alpha_ratio = -1 / (1 - t) - c_t_prime / (1 + c_t)
        N = self.vocab_size - 1
        weight_on_x = (c_t + (1 - t) * c_t_prime) / N / ((1 - t)*(1 - t + c_t/N)).clip(min=1e-30)
        weight_on_u = (c_t + (1 - t) * c_t_prime) / ((1 - t)*c_t).clip(min=1e-30)
        weight_on_m = 1 / ((1 - t)*t).clip(min=1e-30)

        elbo_weights = is_x * weight_on_x + is_mask * weight_on_m + (1 - is_x - is_mask) * weight_on_u

        loss_weights = elbo_weights.clone()
        if self.loss_weighting == "clip":
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)
        elif self.loss_weighting == "dynamic":
            log_snr = torch.sigmoid(-t).clip(-20, 20)  # not exactly the log-SNR, but close enough if C_t is close to 1
            x_scale = B / self.vocab_size * torch.exp(gamma / 2 * log_snr)
            loss_weights = (1 - is_x) * ((1 - is_mask) + 2 * is_mask) + is_x * x_scale
            loss_weights.clip_(self.min_loss_weight, self.max_loss_weight)

        return alpha_ratio.to(orig_dtype), elbo_weights.to(orig_dtype), loss_weights.to(orig_dtype)
    
    def get_weight_function(self, weight_type, t):
        # Add small epsilon to prevent division by zero and extreme values
        eps = 1e-8
        t_safe = torch.clamp(t, min=eps, max=1-eps)
        
        if weight_type == "uniform":
            return torch.ones_like(t)

        elif weight_type == "snr":
            snr = (1 - t_safe) / t_safe
            return torch.clamp(snr, max=10)  # Prevent extreme values

        elif weight_type == "min_snr":
            gamma = 5.0
            snr = (1 - t_safe) / t_safe
            return torch.clamp(snr, max=gamma)

        elif weight_type == "inv_snr":
            inv_snr = t_safe / (1 - t_safe)
            return torch.clamp(inv_snr, max=10)  # Prevent extreme values

        elif weight_type == "sqrt_snr":
            snr = (1 - t_safe) / t_safe
            sqrt_snr = torch.sqrt(torch.clamp(snr, max=10))
            return sqrt_snr

        elif weight_type == "ddpm_x":
            ddpm_weight = 1 / (1 - t_safe)
            return torch.clamp(ddpm_weight, max=10)  # Prevent extreme values

        elif weight_type == "karras":
            c = 0.1  # variance estimation
            karras_weight = 1 / (t_safe + c)
            return torch.clamp(karras_weight, max=10)  # Prevent extreme values
        
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

    def loss(self, logits, input_ids, attention_mask, x_t, t, z_pred, latents, z_t, p_u=0.0):
        dtype = logits.dtype
        alpha_ratio, elbo_weights, ws = self.get_weights(t, x_t, input_ids, p_u)

        logits[..., self.mask_id] = torch.finfo(dtype).min

        x = F.one_hot(input_ids, logits.shape[-1]).to(dtype)
        x_hat = logits.softmax(-1).to(dtype)  # prevent automatic upcasting
        c_t = self.noise_schedule.get_c_t(t, p_u=p_u)
        log_q_t = torch.log(self.noise_schedule.probs_at_t(x, t, c_t=c_t).clip(min=1e-30)).to(dtype)
        log_p_t = torch.log(self.noise_schedule.probs_at_t(x_hat, t, c_t=c_t).clip(min=1e-30)).to(dtype)

        kl_loss = F.kl_div(log_p_t, log_q_t, reduction="none", log_target=True).sum(-1)
    
        log_q_zt = log_q_t.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
        log_p_zt = log_p_t.gather(-1, x_t.unsqueeze(-1)).squeeze(-1)
        log_ratio = log_q_zt - log_p_zt

        is_loss = log_ratio.exp() - log_ratio - 1
        elbo = elbo_weights * (kl_loss + is_loss)
        is_loss = is_loss.clip(max=10)
        loss_disc = ws * (kl_loss + is_loss)
        
        weight_cont = self.get_weight_function(self.weight_type, t).unsqueeze(-1).to(dtype)
        
        # Handle NaN and infinity values before MSE computation
        z_pred_safe = torch.where(torch.isfinite(z_pred), z_pred, torch.zeros_like(z_pred))
        latents_safe = torch.where(torch.isfinite(latents), latents, torch.zeros_like(latents))
        
        # Clip values to prevent overflow
        z_pred_safe = torch.clamp(z_pred_safe, min=-50, max=50)
        latents_safe = torch.clamp(latents_safe, min=-50, max=50)
        
        per_dim_loss = F.mse_loss(z_pred_safe, latents_safe, reduction="none")
        
        # Additional safety: replace any remaining NaN/inf with small value
        per_dim_loss = torch.where(torch.isfinite(per_dim_loss), per_dim_loss, torch.zeros_like(per_dim_loss))
        per_dim_loss = per_dim_loss.clip(max=10)
        
        per_seq_loss = per_dim_loss.mean(-1)
        loss_cont = weight_cont * per_seq_loss
        
        # Ensure continuous loss is finite before combining
        loss_cont_safe = torch.where(torch.isfinite(loss_cont), loss_cont, torch.zeros_like(loss_cont))
        
        loss = loss_disc + self.cont_weight * loss_cont_safe
        
        # Final safety check: replace any NaN or inf in total loss
        loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

        metrics = {
            "kl_loss": (ws * kl_loss.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "is_loss": (ws * is_loss.detach() * attention_mask).sum() / (ws * attention_mask).sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
            "loss_disc": (loss_disc.detach() * attention_mask).sum() / attention_mask.sum(),
            "loss_cont": (loss_cont.detach() * attention_mask).sum() / attention_mask.sum(),
            "loss_total": (loss.detach() * attention_mask).sum() / attention_mask.sum(),
            "z_pred_has_nan": torch.isnan(z_pred).any().float(),
            "latents_has_nan": torch.isnan(latents).any().float(),
            "z_pred_has_inf": torch.isinf(z_pred).any().float(),
            "latents_has_inf": torch.isinf(latents).any().float(),
            "z_pred_max": torch.abs(z_pred).max(),
            "latents_max": torch.abs(latents).max(),
        }

        return loss, elbo, metrics


class MDLMLoss(Loss):
    def __init__(self, config, tokenizer, noise_schedule):
        super().__init__(config, tokenizer, noise_schedule)
        self.mask_id = tokenizer.mask_token_id
        self.neg_infty = -1e6

    def get_sigmas(self, t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    def loss(self, logits, input_ids, attention_mask, x_t, t, *args, **kwargs):
        dsigma, sigma_t = self.get_sigmas(t)

        logits[..., self.mask_id] = self.neg_infty
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        mask_ids = (x_t == self.mask_id)
        logits[~mask_ids] = self.neg_infty
        logits = torch.where(~mask_ids.unsqueeze(-1).expand_as(logits), logits.scatter(-1, x_t.unsqueeze(-1), 0), logits)

        rec_loss = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")

        weights = dsigma.unsqueeze(-1) / torch.expm1(sigma_t).unsqueeze(-1)
        weights = weights * mask_ids.to(weights.dtype)

        elbo = weights * rec_loss

        metrics = {
            "rec_loss": (weights * rec_loss.detach() * attention_mask).sum() / attention_mask.sum(),
            "elbo": (elbo.detach() * attention_mask).sum() / attention_mask.sum(),
        }

        return elbo, elbo, metrics


def get_loss(config, tokenizer, noise_schedule):
    if config.loss.loss_type == "ccdd":
        return GiddLoss(config, tokenizer, noise_schedule)
    elif config.loss.loss_type == "mdlm":
        return MDLMLoss(config, tokenizer, noise_schedule)
    elif config.loss.loss_type == "ar":
        return nn.CrossEntropyLoss(reduction="none")
    elif config.loss.loss_type == "ccdd":
        return CoevolutionaryGiddLoss(config, tokenizer, noise_schedule)
    else:
        raise ValueError(f"Unknown loss_type: {config.loss.loss_type}")
