from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Model, RobertaModel, Qwen3Model

from ccdd.utils import sample_categorical


def sample_t(config, batch_size, eps=None, device=None):
    if eps is None:
        eps = config.model.t_eps

    if config.training.low_discrepancy_sampling:
        t = torch.arange(batch_size, device=device) / batch_size
        t = (t + torch.rand(1, device=device)).fmod(1.0)
    else:
        t = torch.rand(batch_size, device=device)

    t = (1 - 2 * eps) * t + eps
    return t


class NoiseSchedule(nn.Module, ABC):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

        self.register_buffer("log_prior", self.get_log_prior())

    def get_log_prior(self):
        pr = torch.full((self.vocab_size,), -1e3)
        pr[self.mask_id] = 0
        return pr - pr.logsumexp(-1, keepdim=True)
    
    def sample_prior(self, shape):
        return torch.full(shape, self.mask_id, dtype=torch.long, device=self.log_prior.device)
    
    @abstractmethod
    def logits_at_t(self, features, t):
        raise NotImplementedError
    
    @abstractmethod
    def probs_at_t(self, prs, t):
        raise NotImplementedError

    @abstractmethod
    def sample_zt(self, input_ids, t):
        raise NotImplementedError


class HybridDiffusion(NoiseSchedule):
    def __init__(self, tokenizer, clip_noise=20, gamma=1.0, p_uniform=0.0):
        super().__init__(tokenizer)
        self.clip_noise = clip_noise
        self.p_uniform = max(np.exp(-clip_noise), p_uniform)

        log_B = gamma*np.log(2) + np.log(self.p_uniform) - np.log(1 - self.p_uniform)
        self.register_buffer("log_B", torch.tensor(float(log_B)).clip(-clip_noise))
        self.register_buffer("log_gamma", torch.tensor(float(gamma)).log())

        mask = torch.zeros(self.vocab_size)
        mask[self.mask_id] = 1
        self.register_buffer("mask", mask, persistent=False)

        unif = (1 - self.mask) / (self.vocab_size - 1)
        self.register_buffer("unif", unif, persistent=False)
    
    def get_alpha_betapi(self, t, eps=1e-4):
        t = t[:, None]
        t1m = 1 - t

        gamma = self.log_gamma.exp()
        B = self.log_B.exp()
        # .pow() autocasts to fp32
        c_t = t.pow(gamma/2) * t1m.pow(gamma/2) * B
        C_t = 1 + c_t
        # C_t should never be much smaller than 1,
        # but just in case it is, we clip it to avoid numerical instability
        C_t = C_t.clip(eps)

        alpha_t = t1m / C_t
        beta_pi = (t * self.mask + c_t * self.unif) / C_t
        return alpha_t, beta_pi

    def logits_at_t(self, features, t):
        raise NotImplementedError("logits_at_t is not implemented for HybridDiffusion. Use probs_at_t instead.")
    
    def probs_at_t(self, prs, t, eps=1e-4):
        orig_dtype = prs.dtype
        alpha_t, beta_pi = self.get_alpha_betapi(t, eps=eps)

        probs = prs.mul(alpha_t.unsqueeze(-1))
        probs[..., :beta_pi.shape[-1]].add_(beta_pi.unsqueeze(1))
        return probs.to(orig_dtype)

    def sample_zt(self, input_ids, t):
        x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=t.dtype)
        probs = self.probs_at_t(x, t)
        z_t = sample_categorical(probs)
        return z_t
    

class CoevolutionaryDiffusion(NoiseSchedule):
    def __init__(self, tokenizer, vae_model, gamma=1.0, p_u_max=0.0, p_u_min=0.0, p_r_max=0.0, p_r_min=0.0, 
                 forcing_factor=0.0, ar_prior_factor=0.0, latent_dim=None, vae_normalize=True, cont_schedule='flow_linear', layer=None):
        super().__init__(tokenizer)
        self.vae_model = vae_model
        self.vae_model.eval()
        for param in self.vae_model.parameters():
            param.requires_grad = False

        self.register_buffer("gamma", torch.tensor(float(gamma)))

        mask = torch.zeros(self.vocab_size)
        mask[self.mask_id] = 1
        self.register_buffer("mask", mask, persistent=False)

        unif = (1 - self.mask) / (self.vocab_size - 1)
        self.register_buffer("unif", unif, persistent=False)
        self.p_u_max = p_u_max
        self.p_u_min = p_u_min
        self.p_r_max = p_r_max
        self.p_r_min = p_r_min
        self.forcing_factor = forcing_factor
        self.ar_prior_factor = ar_prior_factor
        self.latent_dim = latent_dim
        self.vae_normalize = vae_normalize
        self.cont_schedule = cont_schedule
        self.layer = layer

    def logits_at_t(self, features, t):
        raise NotImplementedError("logits_at_t is not implemented for CoevolutionaryDiffusion. Use probs_at_t instead.")

    def get_alpha_betapi(self, t, eps=0.0, c_t=0.0):
        """Get alpha and beta_pi parameters for diffusion."""
        t = t.unsqueeze(1) if t.ndim == 1 else t
        t1m = 1 - t
        C_t = 1 + c_t

        alpha_t = t1m / C_t
        beta_pi = (t.unsqueeze(-1) * self.mask.unsqueeze(0).unsqueeze(0) + c_t.unsqueeze(-1) * self.unif.unsqueeze(0).unsqueeze(0)) / C_t.unsqueeze(-1)

        return alpha_t.unsqueeze(-1), beta_pi

    def get_c_t(self, t, p_u=0.0):
        """Compute c_t parameter."""
        t = t.unsqueeze(1) if t.ndim == 1 else t
        t1m = 1 - t
        gamma = self.gamma
        
        if p_u > 0:
            B = 2.0**gamma * p_u / (1 - p_u)
            c_t = t.pow(gamma/2) * t1m.pow(gamma/2) * B
        else:
            c_t = torch.zeros_like(t)
        
        return c_t
    
    def probs_at_t(self, prs, t, c_t=0.0, eps=0.0):
        """Compute probability distribution at time t."""
        orig_dtype = prs.dtype
        alpha_t, beta_pi = self.get_alpha_betapi(t, eps=eps, c_t=c_t)

        probs = prs.mul(alpha_t)
        probs[..., :beta_pi.shape[-1]].add_(beta_pi)
        return probs.to(orig_dtype)

    @torch.no_grad()
    def sample_zt_easy(self, input_ids, t, p_u=0.0):
        x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=t.dtype)
        c_t = self.get_c_t(t, p_u=p_u)
        probs = self.probs_at_t(x, t, c_t=c_t)
        x_t = sample_categorical(probs)
        z_t = self.vae_model(input_ids) * torch.sqrt((1 - t)[:, None] / (1 + c_t)).to(t.dtype).unsqueeze(1)
        u_t = sample_categorical(self.unif.unsqueeze(0).unsqueeze(0).repeat(input_ids.shape[0], input_ids.shape[1], 1))
        z_t.add_(self.vae_model(u_t) * torch.sqrt(c_t / (1 + c_t)).to(t.dtype).unsqueeze(1))
        z_t.add_(torch.randn_like(z_t) * torch.sqrt(t[:, None] / (1 + c_t)).to(t.dtype).unsqueeze(1))
        return x_t, z_t

    @torch.no_grad()
    def sample_zt(self, input_ids, t, attention_mask=None, p_u=None, p_r=None, resample_t=None, training=False):
        """
        Sample noisy versions of input for both discrete and continuous components.
        
        Returns:
            x_t: Noisy discrete tokens
            z_t: Noisy continuous latents  
            latents: Clean continuous latents
            p_u: Uniform mixing probability used
        """
        device = input_ids.device
        dtype = t.dtype
        
        # Sample random p_u and p_r if not provided
        if p_u is None:
            p_u = (self.p_u_min + torch.rand(1, device=device, dtype=dtype) * 
                   (self.p_u_max - self.p_u_min))
        else:
            p_u = torch.tensor(p_u, device=device, dtype=dtype)
            
        if p_r is None:
            p_r = (self.p_r_min + torch.rand(1, device=device, dtype=dtype) * 
                   (self.p_r_max - self.p_r_min))
        else:
            p_r = torch.tensor(p_r, device=device, dtype=dtype)
            
        p_u = torch.clamp(p_u, min=0.0, max=1.0)
        p_r = torch.clamp(p_r, min=0.0, max=1.0)
        
        # Continuous component: Sample z_t from q(z_t | z_0) 
        t_cont = (resample_t if resample_t is not None else t).to(dtype)
        
        if (self.forcing_factor > 0 or self.ar_prior_factor > 0) and training:
            if t.ndim == 1:
                t = t.unsqueeze(1)
                t_cont = t_cont.unsqueeze(1)
            t = t.repeat(1, input_ids.size(1))
            t = t + self.forcing_factor * torch.randn_like(t)
            t = t + self.ar_prior_factor * (torch.arange(input_ids.size(1), device=input_ids.device) / input_ids.size(1) - 0.5).unsqueeze(0)
            t = t.clamp(min=0.0, max=1.0).to(dtype)
            if resample_t is not None:
                t_cont = t_cont.unsqueeze(1).repeat(1, input_ids.size(1))
                t_cont = t_cont + self.forcing_factor * torch.randn_like(t_cont)
                t_cont = t_cont + self.ar_prior_factor * (torch.arange(input_ids.size(1), device=input_ids.device) / input_ids.size(1) - 0.5).unsqueeze(0)
                t_cont = t_cont.clamp(min=0.0, max=1.0).to(dtype)
            else:
                t_cont = t
            
        # Discrete component: Sample x_t from q(x_t | x_0)
        x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=dtype)
        c_t = self.get_c_t(t, p_u=p_u)
        probs = self.probs_at_t(x, t, c_t=c_t)
        x_t = sample_categorical(probs)
        
        if p_r > 0:
            # Sample y_t as a mixture of x_t and input_ids based on p_r
            mask = torch.rand_like(input_ids.float()) < p_r
            y_t = torch.where(mask, x_t, input_ids)
        else:
            y_t = input_ids
            
        # continuous component: clean target
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            z_t = None
            latents = self.vae_encode(input_ids, attention_mask, self.vae_normalize)
            if not torch.equal(y_t, input_ids):
                z_t = self.vae_encode(y_t, attention_mask, self.vae_normalize, alter_input_ids=input_ids)
            if z_t is None:
                z_t = latents.clone().detach()
            pca_dim = self.latent_dim
            if pca_dim is not None and pca_dim < z_t.shape[-1]:
                z_t = self.pca(z_t, pca_dim)
                latents = self.pca(latents, pca_dim)
    
        # Add noise to continuous component
        t_cont = t_cont[:, None] if t_cont.ndim == 1 else t_cont
        z_t = self.add_cont_noise(z_t, t_cont)
        
        return x_t, z_t, latents, p_u
    
    @torch.no_grad()
    def vae_encode(self, input_ids, attention_mask, normalize=True, alter_input_ids=None):
        if self.vae_model is not None:
            if isinstance(self.vae_model, GPT2Model) or isinstance(self.vae_model, Qwen3Model):
                if (input_ids == self.mask_id).any():
                    if alter_input_ids is not None:
                        z_t = self.vae_model(alter_input_ids, attention_mask=attention_mask, output_hidden_states=True)
                        z_t = z_t.last_hidden_state if self.layer is None else z_t.hidden_states[self.layer]
                        z_t = torch.where(input_ids[..., None] == self.mask_id, 0., z_t)
                    else:
                        return None
                else:
                    z_t = self.vae_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    z_t = z_t.last_hidden_state if self.layer is None else z_t.hidden_states[self.layer]
            elif isinstance(self.vae_model, RobertaModel):
                z_t = self.vae_model(input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
            else:
                z_t = self.vae_model(input_ids)
            if isinstance(self.vae_model, Qwen3Model) and self.latent_dim is not None:
                z_t = z_t[..., :self.latent_dim]
        else:
            # If no VAE model, use random latents
            raise ValueError("No VAE model provided")
        if normalize:
            mean, var = z_t.mean(dim=-1, keepdim=True), z_t.var(dim=-1, keepdim=True)
            z_t = (z_t - mean) / torch.sqrt(var + 1e-6)
        return z_t
    
    @torch.no_grad()
    def pca(self, z_t, pca_dim):
        batch_size, seq_length, hidden_dim = z_t.shape
        # Reshape for PCA: [batch_size * seq_length, hidden_dim]
        latents_2d = z_t.view(-1, hidden_dim)
        # Perform PCA using SVD
        U, S, V = torch.pca_lowrank(latents_2d, q=pca_dim)
        # Project to reduced space: [batch_size * seq_length, pca_dim]
        z_t_reduced = torch.mm(latents_2d, V)
        # Reshape back to [batch_size, seq_length, pca_dim]
        z_t = z_t_reduced.view(batch_size, seq_length, pca_dim)
        return z_t
    
    @torch.no_grad()
    def add_cont_noise(self, z_t, t):
        if self.cont_schedule == 'old':
            alpha_t = 1 - t
            beta_t = torch.sqrt((2 - t) * t)
        elif self.cont_schedule == 'vp_linear':
            alpha_t = torch.sqrt(1 - t)
            beta_t = torch.sqrt(t)
        elif self.cont_schedule == 'flow_linear':
            alpha_t = 1 - t
            beta_t = t
        elif self.cont_schedule == 'edm':
            raise NotImplementedError
        else:
            raise ValueError
        z_t.mul_(alpha_t.unsqueeze(-1))
        z_t.add_(torch.randn_like(z_t) * beta_t.unsqueeze(-1))
        return z_t
    

class MaskedDiffusion(NoiseSchedule):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # required to be able to interchangeably mix our/mdlm schedule/loss
        self.register_buffer("log_gamma", torch.tensor(0.0))
        self.register_buffer("log_B", torch.tensor(-20.0))

    def get_sigmas(self, t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    def logits_at_t(self, features, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        log_1m_move_chance = -sigma
        logits = (features + 1e-8).clip(1e-8).log().log_softmax(-1) + log_1m_move_chance[..., None, None]
        logits[:, :, self.mask_id] = move_chance.log().clip(-1e6)[..., None]
        return logits
    
    def probs_at_t(self, prs, t):
        _, sigma = self.get_sigmas(t)
        alpha_t = torch.exp(-sigma)
        probs = alpha_t[..., None, None] * prs
        probs[..., self.mask_id] = 1 - alpha_t.unsqueeze(-1)
        return probs

    def sample_zt(self, input_ids, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        is_masked = torch.rand_like(input_ids.float()) < move_chance.unsqueeze(-1)
        z_t = torch.where(is_masked, self.mask_id, input_ids)
        return z_t


def get_noise_schedule(config, tokenizer, vae_model=None):
    if config.model.type == "autoregressive":
        return None
    elif config.model.diffusion_process == "ccdd":
        noise_schedule = HybridDiffusion(tokenizer, p_uniform=config.model.p_uniform)
    elif config.model.diffusion_process == "mdlm":
        noise_schedule = MaskedDiffusion(tokenizer)
    elif config.model.diffusion_process == "coevolutionary":
        noise_schedule = CoevolutionaryDiffusion(tokenizer, vae_model, gamma=config.model.get("gamma", 1.0), p_u_max=config.training.get("p_u_max", 0.0), p_u_min=config.training.get("p_u_min", 0.0), p_r_max=config.training.get("p_r_max", 0.0), p_r_min=config.training.get("p_r_min", 0.0), 
                                                 forcing_factor=config.training.get("forcing_factor", 0.0), ar_prior_factor=config.training.get("ar_prior_factor", 0.0), latent_dim=config.model.get("latent_dim", None), vae_normalize=config.model.get("vae_normalize", True), cont_schedule=config.model.get("cont_schedule", "flow_linear"), layer=config.model.get("layer", None))
    else:
        raise ValueError(f"Unknown diffusion process: {config.model.diffusion_process}")

    return noise_schedule
