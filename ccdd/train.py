import datetime
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import hydra
import tqdm
import wandb
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel, GPT2Model, RobertaModel, AutoTokenizer, AutoModel

from ccdd.models.dit import DIT, MMDIT, MDIT
from ccdd.models.ccdd_moe2s import MoeDiT
from ccdd.checkpoints import (
    save_checkpoint,
    load_checkpoint_for_training,
    TrainingState,
    save_rng_state,
    load_rng_state,
)
from ccdd.diffusion_process import get_noise_schedule
from ccdd.modeling import get_tokenizer, get_model
from ccdd.data import get_dataloaders
from ccdd.loss import get_loss
from ccdd.trainer import get_trainer
from ccdd.optimizer import get_optimizer
from ccdd.pipeline import GiddPipeline
from ccdd.utils import (
    get_lr,
    parse_dtype,
    calculate_flops_per_batch,
)


class Logger:
    def __init__(self, is_main_process):
        self.is_main_process = is_main_process

    def init(self, *args, **kwargs):
        if self.is_main_process:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_main_process:
            wandb.log(*args, **kwargs)


@contextmanager
def main_process_first():
    if dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
    else:
        yield


@hydra.main(config_path="configs", config_name="ccdd", version_base="1.1")
def main(config):
    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=30),
            device_id=torch.device("cuda", local_rank),
        )
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()  # only a single group, don't have to worry about local vs. global rank
        is_main_process = (global_rank == 0)
    except RuntimeError:
        print("Distributed training not available, running on single device.")
        world_size = 1
        local_rank = 0
        global_rank = 0
        is_main_process = True
    with open_dict(config):
        config.training.world_size = world_size

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    seed = config.training.seed + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.enable_flash_sdp(enabled=True)

    dtype = parse_dtype(config.training.dtype)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=} and {dtype=}")

    if "gidd" in config.model.pretrained_model_name:
        pipe = GiddPipeline.from_pretrained(config.model.pretrained_model_name, trust_remote_code=True)
        embeddings = pipe.model.vocab_embed.embedding.detach()
        embeddings[pipe.tokenizer.mask_token_id] = 0
        embeddings[pipe.tokenizer.pad_token_id] = 0
        mean, var = embeddings.mean(dim=0), embeddings.var(dim=0)
        embeddings = (embeddings - mean) / torch.sqrt(var + 1e-6)
        vocab_size = len(pipe.tokenizer)
        rounded_vocab_size = vocab_size + (128 - vocab_size % 128) % 128
        vae_model = torch.nn.Embedding(rounded_vocab_size, config.model.hidden_size)
        vae_model.weight.data.copy_(torch.cat([embeddings[:vocab_size], torch.zeros(rounded_vocab_size - vocab_size, embeddings.shape[1])], dim=0))
        vae_model.requires_grad_ = False
    elif "gpt2" in config.model.pretrained_model_name:
        gpt2_model = GPT2Model.from_pretrained(config.model.pretrained_model_name)
        if config.model.get("contextualize_vae", False):
            vae_model = gpt2_model
        else:
            vae_model = gpt2_model.get_input_embeddings()
        vae_model.requires_grad_ = False
    elif "roberta" in config.model.pretrained_model_name:
        roberta_model = RobertaModel.from_pretrained(config.model.pretrained_model_name)
        if config.model.get("contextualize_vae", False):
            vae_model = roberta_model
        else:
            vae_model = roberta_model.get_input_embeddings()
        vae_model.requires_grad_ = False
    elif "qwen" in config.model.pretrained_model_name.lower():
        vae_model = AutoModel.from_pretrained(config.model.pretrained_model_name)
        vae_model.requires_grad_ = False
    else:
        raise ValueError(f"Unknown pretrained model name: {config.model.pretrained_model_name}")

    if config.training.resume is None:
        tokenizer = get_tokenizer(config)
        if "gpt2" in config.model.pretrained_model_name:
            tokenizer.pad_token = tokenizer.eos_token
            
        if "roberta" in config.model.pretrained_model_name:
            tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)

        if "qwen" in  config.model.pretrained_model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name)
            special_tokens = {}
            if tokenizer.mask_token_id is None:
                special_tokens["mask_token"] = "[MASK]"
            if tokenizer.pad_token_id is None:
                special_tokens["pad_token"] = "<|endoftext|>"
            if special_tokens:
                tokenizer.add_special_tokens(special_tokens)
            config.data.tokenizer_name = config.model.pretrained_model_name

        model = get_model(config, tokenizer, dtype=dtype)
        noise_schedule = get_noise_schedule(config, tokenizer, vae_model)
        loss_fn = get_loss(config, tokenizer, noise_schedule)
        trainer = get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
        trainer = trainer.to(device)

        optimizer = get_optimizer(config, trainer)

        state = TrainingState(
            epoch=0,
            epoch_start_step=0,
            step=0,
        )
    else:
        (
            model,
            noise_schedule,
            tokenizer,
            checkpoint_config,
            trainer,
            optimizer,
            state
        ) = load_checkpoint_for_training(config.training.resume, device=device, dtype=dtype)

    print("Done! Loading dataloaders...")
    with main_process_first():
        train_dl, test_dl = get_dataloaders(config, tokenizer)

    max_lr = config.optimizer.lr

    logger = Logger(is_main_process)
    logger.init(
        name=config.logging.run_name,
        entity=config.logging.wandb_entity,
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )

    if is_main_process:
        pwd = Path(".").resolve()
        wandb.config.update({"pwd": pwd})
        print(f"Working directory: {pwd}")

    if isinstance(model, DIT) or isinstance(model, MMDIT) or isinstance(model, MoeDiT) or isinstance(model, MDIT):
        non_emb_params = sum(p.numel() for p in model.blocks.parameters())
    else:  # Llama
        non_emb_params = sum(p.numel() for p in model.model.layers.parameters())

    flops_per_batch = calculate_flops_per_batch(config, model, len(tokenizer), non_emb_params, method="hoffmann")

    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)

    if config.training.compile_model:
        opt_trainer = torch.compile(trainer)
    else:
        opt_trainer = trainer

    if is_distributed:
        ddp_trainer = DDP(opt_trainer, device_ids=[device.index], find_unused_parameters=True)
    else:
        ddp_trainer = opt_trainer

    if is_main_process:
        non_emb_params_str = f"{non_emb_params / 1e6:.1f}M" if non_emb_params < 500 * 1e6 else f"{non_emb_params / 1e9:.1f}B"
        trainable_params_str = f"{trainable_params / 1e6:.1f}M" if trainable_params < 500 * 1e6 else f"{trainable_params / 1e9:.1f}B"
        grad_accum_steps = config.training.get("grad_accum_steps", 1)
        effective_batch_size = config.training.train_batch_size * grad_accum_steps
        print(f"*** Starting training ***")
        print(f"* World size: {world_size}")
        print(f"* FLOPS per batch: {flops_per_batch:.3g}")
        print(f"* Per-device batch size: {config.training.train_batch_size}")
        print(f"* Gradient accumulation steps: {grad_accum_steps}")
        print(f"* Effective per-device batch size: {effective_batch_size}")
        print(f"* Total effective batch size: {effective_batch_size * world_size}")
        print(f"* Non-embedding parameters: {non_emb_params_str}")
        print(f"* Trainable parameters: {trainable_params_str}")
        print(f"* Model dtype: {next(iter(model.parameters())).dtype}")
        print(f"*************************")

    if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
        train_dl.sampler.set_epoch(state.epoch)
    batch_iterator = iter(train_dl)

    # initialize eval dataloader to prevent new processes getting started during training
    # (without this crashes can occur if the code changes before the first eval step)
    _ = next(iter(test_dl))

    if state.step - state.epoch_start_step > 0:
        for _ in tqdm.trange(state.step - state.epoch_start_step, desc="Skipping batches", dynamic_ncols=True, disable=not is_main_process):
            next(batch_iterator)

    curr_time = time.time()
    # adjust start time in case we're resuming training
    trained_time = 0 if config.training.resume is None else (state.start_time - state.curr_time)
    state.start_time = curr_time - trained_time
    state.curr_time = curr_time
    prev_time = curr_time

    log_buffer = []

    if config.training.resume is not None:
        load_rng_state(config.training.resume, global_rank)

    with tqdm.tqdm(total=config.training.num_train_steps, initial=state.step, desc="Training", dynamic_ncols=True, disable=not is_main_process) as pbar:
        for step in range(state.step, config.training.num_train_steps):
                
            ### TRAIN ###

            # Gradient accumulation loop
            grad_accum_steps = config.training.get("grad_accum_steps", 1)
            accumulated_loss = 0.0
            
            for accum_step in range(grad_accum_steps):
                try:
                    batch = next(batch_iterator)
                except StopIteration:
                    state.epoch += 1
                    state.epoch_start_step = step
                    if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
                        train_dl.sampler.set_epoch(state.epoch)
                    batch_iterator = iter(train_dl)
                    batch = next(batch_iterator)

                curr_lr = get_lr(config, max_lr, step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                loss, metrics = ddp_trainer(batch, resample_t=config.training.get("resample_t", None))
                
                # Scale loss for gradient accumulation
                scaled_loss = (loss * config.loss.loss_scale) / grad_accum_steps
                scaled_loss.backward()
                
                accumulated_loss += loss.item()

            # After gradient accumulation, handle gradients
            if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)
                if torch.isnan(norm):
                    print(f"Warning: NaN gradient detected at step {step}")
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.zero_()
            else:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)
            
            optimizer.step()
            optimizer.zero_grad()

            # Use accumulated loss for logging
            avg_loss = accumulated_loss / grad_accum_steps

            # Calculate effective batch size considering gradient accumulation
            effective_batch_size = batch["input_ids"].size(0) * grad_accum_steps
            batch_tokens = batch["attention_mask"].sum().item() * grad_accum_steps * config.training.world_size
            batch_flops = flops_per_batch * grad_accum_steps * config.training.world_size
            total_batch_size = effective_batch_size * config.training.world_size
            state.total_tokens += batch_tokens
            state.total_flops += batch_flops

            curr_time = time.time()
            step_time = curr_time - prev_time
            prev_time = curr_time

            # no need to all_reduce metrics since these are not that important
            log_buffer.append({
                "train/loss": avg_loss,
                "train/lr": curr_lr,
                "train/step": step + 1,
                "train/grad_norm": norm.item(),
                "train/epoch": step / len(train_dl),
                "train/total_tokens": state.total_tokens,
                "train/total_flops": state.total_flops,
                "train/tokens_per_sec": batch_tokens / step_time,
                "train/flops_per_sec": batch_flops / step_time,
                "train/samples_per_sec": total_batch_size / step_time,
                "train/it_per_sec": 1 / step_time,
                "train/avg_it_per_sec": (step + 1) / (curr_time - state.start_time),
                "train/grad_accum_steps": grad_accum_steps,
                **{f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()},
            })

            if ((step + 1) % config.logging.log_freq) == 0:
                metrics = {k: sum(d[k] for d in log_buffer) / len(log_buffer) for k in log_buffer[0]}
                logger.log({k: v for k, v in metrics.items()}, step=step)
                logger.log({"trainer/global_step": step}, step=step)
                log_buffer = []

            ### EVAL ###

            if ((step + 1) % config.logging.eval_freq) == 0:
                with torch.no_grad():
                    eval_start_time = time.time()
                    model.eval()

                    eval_metrics = {}
                    eval_loss = 0
                    num_eval_samples = 0
                    for i, test_batch in enumerate(tqdm.tqdm(test_dl, desc="Eval", dynamic_ncols=True, total=config.logging.num_eval_batches, disable=not is_main_process)):
                        bs = test_batch["input_ids"].size(0)

                        test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
                        # loss, metrics = ddp_trainer(test_batch, p_u=0.0, p_r=1.0, resample_t=torch.zeros(bs, device=device, dtype=torch.bfloat16))
                        loss, metrics = ddp_trainer(test_batch, p_r=1.0)  # use p_1=0.0 for real tasks

                        for k, v in metrics.items():
                            eval_metrics[k] = eval_metrics.get(k, 0) + (v.item() if isinstance(v, torch.Tensor) else v) * bs

                        eval_loss += loss.item() * bs
                        num_eval_samples += bs

                        if i >= config.logging.num_eval_batches - 1:
                            break

                    for key in ["nll", "ppl"]:
                        if key in eval_metrics:
                            del eval_metrics[key]

                    dist.barrier()

                    eval_elapsed_time = time.time() - eval_start_time
                    logger.log({
                        "eval/loss": eval_loss / num_eval_samples,
                        "eval/time_taken": eval_elapsed_time,
                        **{f"eval/{k}": v / num_eval_samples for k, v in eval_metrics.items()},
                    }, step=step)
                    model.train()

            ### SAVE ###

            # increment step before saving so that resuming from the checkpoint will start at the next step
            state.step += 1
            if ((step + 1) % config.logging.save_freq) == 0:
                dist.barrier()
                output_path = Path(config.logging.save_dir, config.logging.run_name)
                if (step + 1) == 500000:
                    suffix = "-500k"
                elif (step + 1) == 1000000:
                    suffix = "-1M"
                elif (step + 1) == 250000:
                    suffix = "-250k"
                elif (step + 1) == 750000:
                    suffix = "-750k"
                elif (step + 1) == 565500:
                    suffix = "-565k"
                elif (step + 1) == 1131000:
                    suffix = "-1.131M"
                else:
                    suffix = "latest"
                output_path = output_path / suffix
                if is_main_process:
                    save_checkpoint(output_path, trainer, optimizer, state)
                dist.barrier()
                output_path.mkdir(exist_ok=True, parents=True)
                save_rng_state(output_path, global_rank)
                dist.barrier()

            pbar.update(1)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
