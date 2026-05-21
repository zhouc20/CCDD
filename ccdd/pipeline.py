import torch
import torch.nn as nn
import tqdm.auto as tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer

from ccdd.diffusion_process import HybridDiffusion
from ccdd.sampling import GiddSampler
from ccdd.utils import sample_categorical


class GiddPipeline(nn.Module):
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        compile_step = kwargs.pop("compile_step", False)
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        config = model.config
        noise_schedule = HybridDiffusion(tokenizer, p_uniform=config.p_uniform)
        return cls(model, noise_schedule, tokenizer, config, compile_step=compile_step)
    
    def __init__(self, model, noise_schedule, tokenizer, config, compile_step: bool = False):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.config = config

        self.sampler = GiddSampler(model, tokenizer, noise_schedule, t_eps=config.t_eps, compile_step=compile_step)

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm.tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm.tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        num_inference_steps: int = 128,
        show_progress: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[str]:
        device = next(self.model.parameters()).device
        with torch.autocast(device.type, dtype):
            return self.sampler.generate(
                num_samples=num_samples,
                num_denoising_steps=num_inference_steps,
                max_length=self.config.max_seq_len,
                decode=True,
                show_progress=show_progress,
            )
    
    @torch.no_grad()
    def self_correction(
        self,
        texts: list[str],
        num_inference_steps: int = 128,
        temperature: float = 0.1,
        t0: float = 0.01,
        early_stopping: bool = True,
        early_stopping_patience: int = 32,
        show_progress: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[str]:
        def _correction_step(model, tokenizer, z_t, t, temp):
            logits = model(z_t, t)
            logits[..., tokenizer.mask_token_id] = -1e6

            p_t = (logits / temp).softmax(-1)

            z_tm1 = sample_categorical(p_t)
            score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1)

            ids = torch.topk(score, 1, dim=-1).indices
            z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

            acc = (z_tm1 == logits.argmax(-1)).float().mean().item()
            return z_tm1, acc

        device = next(self.model.parameters()).device
        z_ts = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.max_seq_len)["input_ids"]
        corrected_zts = []
        with tqdm.tqdm(total=len(texts) * num_inference_steps, disable=not show_progress) as pbar:
            for z_t in z_ts:
                max_acc = 0
                curr_patience = 0

                z_t = z_t.unsqueeze(0).to(device)
                t = torch.full((z_t.shape[0],), device=device, fill_value=t0)

                logits = self.model(z_t, t)
                logits[..., self.tokenizer.mask_token_id] = -1e6
                
                for i in range(num_inference_steps):
                    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                        z_t_next, acc = _correction_step(self.model, self.tokenizer, z_t, t, temperature)

                        if early_stopping:
                            if acc > max_acc:
                                max_acc = acc
                                curr_patience = 0
                            else:
                                curr_patience += 1
                                if curr_patience > early_stopping_patience:
                                    break

                            if (z_t == z_t_next).all():
                                break
                        z_t = z_t_next
                    pbar.update(1)

                corrected_zts.append(z_t)

            corrected_zts = torch.cat(corrected_zts, dim=0)
            corrected_samples = self.tokenizer.batch_decode(corrected_zts, skip_special_tokens=True)
            return corrected_samples
