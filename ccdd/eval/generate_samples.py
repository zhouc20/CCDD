import hydra
import tqdm
import torch

from ccdd.utils import parse_dtype
from ccdd.checkpoints import load_checkpoint
from ccdd.sampling import get_sampler


@hydra.main(config_path="../configs", config_name="generate", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Generating {args.num_samples} samples from {args.path}")

    ckpt_path = hydra.utils.to_absolute_path(args.path)

    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)

    sampler = get_sampler(config, model, tokenizer, noise_schedule, min_p=args.min_p, compile_step=args.get("compile_step", False))
    model.eval()

    disc_type = args.get("disc_type", "ccdd")
    cont_type = args.get("cont_type", "ddpm")
    w = args.get("w", 1.0)
    print(f"Disc type: {disc_type}, cont type: {cont_type}, w: {w}")

    samples = []
    with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, args.num_samples, args.batch_size):
                bs = min(args.batch_size, args.num_samples - i)
                x_t, z_t = sampler.generate(bs, args.num_denoising_steps, max_length=args.max_length, decode=False, show_progress=False, dtype=dtype, disc_type=disc_type, cont_type=cont_type, w=w)
                samples.append(x_t)
                pbar.update(bs)
    samples = torch.cat(samples, dim=0).cpu()

    torch.save(samples, hydra.utils.to_absolute_path(args.samples_path))


if __name__ == "__main__":
    main()
