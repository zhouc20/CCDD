from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

import ccdd.models.dit as dit
import ccdd.models.ccdd_moe2s as moe2s

try:
    import flash_attn
    has_flash_attn = True
except ImportError:
    has_flash_attn = False


def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer_name)
    
    # Handle special tokens based on model type
    special_tokens = {}
    if tokenizer.pad_token_id is None:
        if "qwen" in config.data.tokenizer_name.lower():
            special_tokens["pad_token"] = "<|endoftext|>"
        else:
            special_tokens["pad_token"] = "[PAD]"
    if tokenizer.mask_token_id is None:
        special_tokens["mask_token"] = "[MASK]"
    
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    
    tokenizer.model_max_length = config.model.max_seq_len
    return tokenizer


def get_model(config, tokenizer, device=None, dtype=None):
    if config.model.type == "diffusion":
        model = dit.DIT(config, len(tokenizer))
    elif config.model.type == "autoregressive":
        cfg = LlamaConfig(
            vocab_size=len(tokenizer),
            num_hidden_layers=config.model.n_blocks,
            hidden_size=config.model.hidden_size,
            intermediate_size=4*config.model.hidden_size,
            num_attention_heads=config.model.n_heads,
            max_position_embeddings=config.model.max_seq_len,
            attn_implementation="flash_attention_2" if has_flash_attn else "sdpa",
            torch_dtype=dtype,
        )
        model = LlamaForCausalLM(cfg)
    elif config.model.type == "mmdit":
        model = dit.MMDIT(config, len(tokenizer))
    elif config.model.type == "mdit":
        model = dit.MDIT(config, len(tokenizer))
    elif config.model.type == "moedit":
        model = moe2s.MoeDiT(config, len(tokenizer))
    elif config.model.type == "coevolutionary":
        raise NotImplementedError("Coevolutionary model not implemented yet")
    else:
        raise ValueError(f"Unknown model type: {config.model.type}")

    if device is not None:
        model = model.to(device, dtype=dtype)
    elif dtype is not None:
        model = model.to(dtype=dtype)

    return model
