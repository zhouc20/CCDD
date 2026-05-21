import json

import numpy as np
import hydra
import tqdm
import torch
from transformers import AutoModelForCausalLM

from ccdd.data import get_dataloaders
from ccdd.utils import parse_dtype
from ccdd.loss import get_loss
from ccdd.checkpoints import load_checkpoint
from ccdd.trainer import get_trainer


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    ckpt_path = hydra.utils.to_absolute_path(args.path)

    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)
    if args.use_gpt2:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)

    loss_fn = get_loss(config, tokenizer, noise_schedule)
    _, test_dl = get_dataloaders(config, tokenizer)

    trainer = get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
    trainer.to(device)
    trainer = torch.compile(trainer)
    model.eval()

    eval_metrics = {}
    with torch.no_grad():
        eval_loss = 0
        num_eval_samples = 0
        for test_batch in tqdm.tqdm(test_dl, desc="Eval", dynamic_ncols=True):
            bs = test_batch["input_ids"].size(0)

            test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
            loss, metrics = trainer(test_batch)

            for k, v in metrics.items():
                eval_metrics[k] = eval_metrics.get(k, 0) + (v.item() if isinstance(v, torch.Tensor) else v) * bs

            eval_loss += loss.item() * bs
            num_eval_samples += bs

    eval_metrics = {
        "loss": eval_loss / num_eval_samples,
        **{k: v / num_eval_samples for k, v in eval_metrics.items()},
    }
    eval_metrics["ppl"] = np.exp(eval_metrics["elbo"])

    eval_metrics["path"] = ckpt_path

    print(json.dumps(eval_metrics, indent=2))

    with open("metrics.json", "a") as f:
        json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()
