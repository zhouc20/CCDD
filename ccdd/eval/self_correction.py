import pandas as pd
import hydra
import tqdm
import torch

from ccdd.utils import parse_dtype
from ccdd.checkpoints import load_checkpoint
from ccdd.utils import sample_categorical


def correction_step(model, tokenizer, z_t, t, temp, tokens_per_step):
    logits = model(z_t, t)
    logits[..., tokenizer.mask_token_id] = -1e6

    p_t = (logits / temp).softmax(-1)

    z_tm1 = sample_categorical(p_t)
    score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1)

    ids = torch.topk(score, tokens_per_step, dim=-1).indices
    z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))

    acc = (z_tm1 == logits.argmax(-1)).float().mean().item()
    return z_tm1, acc


@hydra.main(config_path="../configs", config_name="self_correction", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(args.path)

    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)

    model = torch.compile(model)

    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    z_ts = torch.load(samples_path, weights_only=True)

    metrics = []
    samples = []
    z_t: torch.Tensor
    for z_t in tqdm.tqdm(z_ts, desc="Correction", dynamic_ncols=True, smoothing=0.0):
        max_acc = 0
        curr_patience = 0

        z_t = z_t.unsqueeze(0).to(device)
        z_t_init = z_t.clone()
        t = torch.full((z_t.shape[0],), device=device, fill_value=args.t0)

        logits = model(z_t, t)
        logits[..., tokenizer.mask_token_id] = -1e6
        init_acc = (z_t == logits.argmax(-1)).float().mean().item()
        
        converged = 0
        early_stopped = 0
        for i in range(args.num_denoising_steps):
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                z_t_next, acc = correction_step(model, tokenizer, z_t, t, args.temp, args.tokens_per_step)

                if acc > max_acc:
                    max_acc = acc
                    curr_patience = 0
                else:
                    curr_patience += 1
                    if curr_patience > args.max_patience:
                        early_stopped = 1
                        break

                if (z_t == z_t_next).all():
                    converged = 1
                    break
                z_t = z_t_next

        num_changes = (z_t_init != z_t).sum().item()
        samples.append(z_t)
        metrics.append({
            "init_acc": init_acc,
            "final_acc": acc,
            "num_changes": num_changes,
            "converged": converged,
            "early_stopped": early_stopped,
        })
    samples = torch.cat(samples, dim=0).cpu()

    torch.save(samples, args.corrected_samples_path)

    df = pd.DataFrame(metrics)
    df["improvement"] = df["final_acc"] - df["init_acc"]

    df.to_csv(hydra.utils.to_absolute_path(args.metrics_path), index=False)

    # compute mean and std of all metrics, print as markdown table
    print(f"Results for {args.path} (temp={args.temp}, max_patience={args.max_patience}):\n{df.describe().to_markdown()}")

if __name__ == "__main__":
    main()
