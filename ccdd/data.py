import functools
import hashlib
import json
import os
import shutil
from typing import Callable

import numpy as np
import torch
import hydra
from transformers import BatchEncoding, PreTrainedTokenizer
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataset(config, num_proc=32):
    test_size = int(config.data.test_size)
    n_proc = min(os.cpu_count(), num_proc)
    train_ds = load_dataset(
        config.data.dataset_name,
        config.data.dataset_subset,
        split=f"train[:-{test_size}]",
        trust_remote_code=config.data.trust_remote_code,
        num_proc=n_proc,
        cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
    )
    test_ds = load_dataset(
        config.data.dataset_name,
        config.data.dataset_subset,
        split=f"train[-{test_size}:]",
        trust_remote_code=config.data.trust_remote_code,
        num_proc=n_proc,
        cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
    )

    return train_ds, test_ds


def cached_dataset(cache_dir: str, file_name: str, generate_fn: Callable[[], Dataset]) -> Dataset:
    if cache_dir is None:
        return generate_fn()

    cache_path = os.path.join(cache_dir, file_name)
    if os.path.exists(cache_path):
        ds = Dataset.load_from_disk(cache_path)
        return ds
    else:
        ds = generate_fn()
        os.makedirs(cache_dir, exist_ok=True)
        try:
            ds.save_to_disk(cache_path)
        except Exception as e:
            shutil.rmtree(cache_path)
            raise e
        return ds


def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int = 512,
    sequence_packing: bool = False,
    batch_size: int = 1024,
    num_proc: int = 32,
):
    n_proc = min(os.cpu_count(), num_proc)
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    tokenizer_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = 10_000_000

    def tokenize_fn(examples):
        tokens = tokenizer(examples["text"], truncation=False, padding=False)["input_ids"]
        # # Use return_tensors="pt" then convert to avoid numpy.object_ issues

        tokens = [[bos_token_id] + x + ([] if sequence_packing else [eos_token_id]) for x in tokens]
        if sequence_packing:
            tokens = np.concatenate(tokens, axis=0)
            tokens = tokens[: len(tokens) - len(tokens) % max_seq_len]
            tokens = tokens.reshape(-1, max_seq_len)
        else:
            tokens = [
                np.pad(x, (0, max_seq_len - len(x) % max_seq_len), mode="constant", constant_values=tokenizer.pad_token_id)
                for x in tokens
            ]
            tokens = [x.reshape(-1, max_seq_len) for x in tokens]
            tokens = np.concatenate(tokens, axis=0)
        return {"input_ids": tokens}

    ds = ds.map(
        tokenize_fn,
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
        num_proc=n_proc,
    )

    tokenizer.model_max_length = tokenizer_max_len
    return ds


def default_collator(config, tokenizer, examples, text_key="text"):
    examples = [x[text_key] for x in examples]
    return tokenizer(examples, padding="max_length", truncation=True, max_length=config.model.max_seq_len, return_tensors="pt")


def pretokenized_collator(examples, pad_token_id=0, tokens_key="input_ids"):
    input_ids_list = []
    for x in examples:
        tokens = x[tokens_key]
        
        # Handle different token data types and structures
        if isinstance(tokens, np.ndarray) and tokens.dtype == np.object_:
            # Extract the actual array from object array
            if tokens.size == 1:
                tokens = tokens.item()
                if isinstance(tokens, np.ndarray):
                    # It was a nested array, now we have the actual array
                    tokens = tokens
                elif isinstance(tokens, (list, tuple)):
                    # Convert list/tuple to numpy array
                    tokens = np.array(tokens, dtype=np.int64)
                else:
                    # Single value, wrap in array
                    tokens = np.array([tokens], dtype=np.int64)
            else:
                # Try to extract valid numeric data from object array
                valid_tokens = []
                for item in tokens.flat:
                    if item is not None:
                        if isinstance(item, np.ndarray):
                            valid_tokens.extend(item.tolist())
                        elif isinstance(item, (list, tuple)):
                            valid_tokens.extend(item)
                        elif isinstance(item, (int, np.integer)):
                            valid_tokens.append(int(item))
                if valid_tokens:
                    tokens = np.array(valid_tokens, dtype=np.int64)
                else:
                    # If no valid tokens found, create empty array
                    tokens = np.array([], dtype=np.int64)
        elif not isinstance(tokens, np.ndarray):
            # Convert list/tuple to numpy array, filtering out None values
            if isinstance(tokens, (list, tuple)):
                tokens = [t for t in tokens if t is not None]
                tokens = np.array(tokens, dtype=np.int64)
            else:
                # Single value
                tokens = np.array([tokens] if tokens is not None else [], dtype=np.int64)
        
        # Ensure tokens are integers - but handle object arrays carefully
        if tokens.dtype == np.object_:
            # Still object array, try to manually convert
            valid_tokens = []
            for item in tokens.flat:
                if item is not None and isinstance(item, (int, np.integer, float)):
                    valid_tokens.append(int(item))
            tokens = np.array(valid_tokens, dtype=np.int64)
        elif tokens.dtype not in [np.int64, np.int32]:
            try:
                tokens = tokens.astype(np.int64)
            except (ValueError, TypeError):
                # Filter out invalid values and convert
                valid_tokens = []
                for item in tokens.flat:
                    try:
                        if item is not None:
                            valid_tokens.append(int(item))
                    except (ValueError, TypeError):
                        continue
                tokens = np.array(valid_tokens, dtype=np.int64)
        
        input_ids_list.append(tokens)
    
    if input_ids_list:
        max_len = max(len(tokens) for tokens in input_ids_list)
        padded_tokens = []
        for tokens in input_ids_list:
            if len(tokens) < max_len:
                # Pad with pad_token_id
                padded = np.pad(tokens, (0, max_len - len(tokens)), 
                              mode='constant', constant_values=pad_token_id)
                padded_tokens.append(padded)
            else:
                padded_tokens.append(tokens[:max_len])  # Truncate if too long
        
        input_ids = np.stack(padded_tokens, axis=0)
    else:
        input_ids = np.array([[]], dtype=np.int64)
    
    attn_masks = (input_ids != pad_token_id).astype(np.int32)
    input_ids = torch.from_numpy(input_ids).to(torch.long)
    attn_masks = torch.from_numpy(attn_masks).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def subsample_collator(config, tokenizer, examples, text_key="text"):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id

    examples = [x[text_key] for x in examples]
    tokens = tokenizer(examples, truncation=False, return_tensors="pt")
    # Convert to numpy for processing
    tokens = {k: v.numpy() for k, v in tokens.items()}
    max_length = config.model.max_seq_len
    input_ids = []
    attn_masks = []
    for i in range(len(examples)):
        toks = tokens["input_ids"][i]
        attn_mask = tokens["attention_mask"][i]
        if toks[0] != bos_token_id:
            toks = np.concatenate([[bos_token_id], toks])
            attn_mask = np.concatenate([[1], attn_mask])
        if toks[-1] != eos_token_id:
            toks = np.concatenate([toks, [eos_token_id]])
            attn_mask = np.concatenate([attn_mask, [1]])

        if len(toks) > max_length:
            overflow = len(toks) - max_length
            start_idx = np.random.randint(0, overflow + config.data.max_add_padding)
            toks = toks[start_idx : start_idx + max_length]
            attn_mask = attn_mask[start_idx : start_idx + max_length]
        if len(toks) < max_length:
            underflow = max_length - len(toks)
            toks = np.pad(toks, (0, underflow), mode="constant", constant_values=tokenizer.pad_token_id)
            attn_mask = np.pad(attn_mask, (0, underflow), mode="constant", constant_values=0)
        assert len(toks) == max_length
        assert len(attn_mask) == max_length
        input_ids.append(toks)
        attn_masks.append(attn_mask)
    input_ids = torch.from_numpy(np.array(input_ids)).to(torch.long)
    attn_masks = torch.from_numpy(np.array(attn_masks)).to(torch.long)
    return BatchEncoding({"input_ids": input_ids, "attention_mask": attn_masks}, tensor_type="pt", n_sequences=len(input_ids))


def split_sequence_collator(config, tokenizer, examples, text_key="text"):
    bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
    max_length = config.model.max_seq_len

    # Tokenize all examples
    examples = [x[text_key] for x in examples]
    tokens = tokenizer(examples, truncation=False, return_tensors="pt")
    # Convert to numpy for processing
    tokens = {k: v.numpy() for k, v in tokens.items()}
    
    # Process each example
    all_chunks = []
    for i in range(len(examples)):
        toks = tokens["input_ids"][i]
        attn_mask = tokens["attention_mask"][i]
        
        # Add BOS and EOS tokens
        if toks[0] != bos_token_id:
            toks = np.concatenate([[bos_token_id], toks])
            attn_mask = np.concatenate([[1], attn_mask])
        if toks[-1] != eos_token_id:
            toks = np.concatenate([toks, [eos_token_id]])
            attn_mask = np.concatenate([attn_mask, [1]])
            
        # Split into chunks of max_length
        n_chunks = len(toks) // max_length
        if n_chunks > 0:
            # Reshape into chunks, dropping the last incomplete chunk
            chunks = toks[:n_chunks * max_length].reshape(-1, max_length)
            mask_chunks = attn_mask[:n_chunks * max_length].reshape(-1, max_length)
            all_chunks.extend([(chunk, mask) for chunk, mask in zip(chunks, mask_chunks)])
    
    # If we have more chunks than batch_size, randomly select batch_size chunks
    if len(all_chunks) > config.training.train_batch_size:
        indices = np.random.choice(len(all_chunks), config.training.train_batch_size, replace=False)
        all_chunks = [all_chunks[i] for i in indices]
    
    # Stack the chunks
    input_ids = torch.from_numpy(np.stack([chunk for chunk, _ in all_chunks])).to(torch.long)
    attn_masks = torch.from_numpy(np.stack([mask for _, mask in all_chunks])).to(torch.long)
    
    return BatchEncoding(
        {"input_ids": input_ids, "attention_mask": attn_masks},
        tensor_type="pt",
        n_sequences=len(input_ids)
    )


def _get_dataloader(config, ds, shuffle, drop_last, batch_size, collate_fn):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        sampler = DistributedSampler(ds, seed=config.training.seed, shuffle=shuffle)
        _shuffle = False
    else:
        sampler = None
        _shuffle = shuffle

    return DataLoader(
        ds,
        collate_fn=collate_fn,
        batch_size=batch_size,
        drop_last=drop_last,
        sampler=sampler,
        num_workers=config.data.num_workers,
        shuffle=_shuffle,
        pin_memory=True,
        persistent_workers=True,
    )


def get_dataloaders(config, tokenizer, train_batch_size=None, eval_batch_size=None):
    if train_batch_size is None:
        train_batch_size = config.training.train_batch_size
    if eval_batch_size is None:
        eval_batch_size = config.training.eval_batch_size

    train_ds, test_ds = get_dataset(config)

    if config.data.pre_tokenize:
        max_seq_len = config.model.max_seq_len
        sequence_packing = config.data.sequence_packing
        cache_key = hashlib.sha256(
            json.dumps(
                {
                    "dataset_name": config.data.dataset_name,
                    "subset": config.data.dataset_subset,
                    "tokenizer_name": config.data.tokenizer_name,
                    "max_seq_len": max_seq_len,
                    "sequence_packing": sequence_packing,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        train_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-train-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=train_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing),
        )
        test_ds = cached_dataset(
            cache_dir=hydra.utils.to_absolute_path(config.data.cache_dir),
            file_name=f"cache-{config.data.dataset_name.replace('/', '--')}-test-{cache_key}",
            generate_fn=functools.partial(tokenize_dataset, ds=test_ds, tokenizer=tokenizer, max_seq_len=max_seq_len, sequence_packing=sequence_packing),
        )

        collate_fn = functools.partial(pretokenized_collator, pad_token_id=tokenizer.pad_token_id, tokens_key="input_ids")
    else:
        if config.data.sequence_packing:
            raise ValueError("Sequence packing requires pre-tokenization.")
        
        if config.data.get("split_sequences", False):
            collate_fn = functools.partial(split_sequence_collator, config, tokenizer, text_key="text")
        else:
            collate_fn = functools.partial(subsample_collator, config, tokenizer, text_key="text")

    train_dl = _get_dataloader(config, train_ds, shuffle=True, drop_last=True, batch_size=train_batch_size, collate_fn=collate_fn)
    test_dl = _get_dataloader(config, test_ds, shuffle=False, drop_last=False, batch_size=eval_batch_size, collate_fn=collate_fn)

    return train_dl, test_dl
