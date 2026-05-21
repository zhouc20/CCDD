import re
import sys

import torch
from transformers import AutoTokenizer


def compute_diff(tokenizer, xa, xb):
    tokens_a = tokenizer(xa, return_tensors="pt")
    tokens_b = tokenizer(xb, return_tensors="pt")

    out_a, out_b = "", ""
    idx_a, idx_b = 0, 0
    is_highlighting = False
    for i in range(min(tokens_a["input_ids"].size(1), tokens_b["input_ids"].size(1))):
        is_same = tokens_a["input_ids"][0, i] == tokens_b["input_ids"][0, i]
        span_a = tokens_a.token_to_chars(0, i)
        span_b = tokens_b.token_to_chars(0, i)
        if not is_same:
            if not is_highlighting:
                out_a += xa[idx_a:span_a.start]
                out_b += xb[idx_b:span_b.start]
                idx_a = span_a.start
                idx_b = span_b.start
                is_highlighting = True
            else:
                continue
        else:
            if is_highlighting:
                out_a += "\\hlred{" + xa[idx_a:span_a.start] + "}"
                out_b += "\\hlgreen{" + xb[idx_b:span_b.start] + "}"
                is_highlighting = False
                idx_a = span_a.start
                idx_b = span_b.start
            else:
                continue
    if is_highlighting:
        out_a += "\\hlred{" + xa[idx_a:] + "}"
        out_b += "\\hlgreen{" + xb[idx_b:] + "}"
    else:
        out_a += xa[idx_a:]
        out_b += xb[idx_b:]
    return out_a, out_b


template = """
\\midrule
% ----- Before -----
{} &
% ----- After ------
{} \\\\
""".strip()


def sanitize(x):
    x = re.sub(r"\&", r"\\&", x)
    x = re.sub(r"\%", r"\\%", x)
    x = re.sub(r"\$", r"\\$", x)
    x = re.sub(r"\#", r"\\#", x)
    x = re.sub(r"\_", r"\\_", x)
    x = re.sub(r"\{", r"\\{", x)
    x = re.sub(r"\}", r"\\}", x)
    x = re.sub(r"\^", r"\\^", x)
    x = re.sub(r"\~", r"\\~", x)
    return x


def main():
    file_a = sys.argv[1]
    file_b = sys.argv[2]

    zs_a = torch.load(file_a, weights_only=True)
    zs_b = torch.load(file_b, weights_only=True)

    if zs_a.size(1) == 1:
        zs_a = zs_a.squeeze(1)
    if zs_b.size(1) == 1:
        zs_b = zs_b.squeeze(1)

    assert zs_a.shape == zs_b.shape

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    xs_a = tokenizer.batch_decode(zs_a, skip_special_tokens=True)[:20]
    xs_b = tokenizer.batch_decode(zs_b, skip_special_tokens=True)[:20]

    # sanitize strings
    xs_a = [sanitize(x) for x in xs_a]
    xs_b = [sanitize(x) for x in xs_b]

    for xa, xb in zip(xs_a, xs_b):
        out_a, out_b = compute_diff(tokenizer, xa, xb)
        out_a = re.sub(r"[\n\s]+", " ", out_a)
        out_b = re.sub(r"[\n\s]+", " ", out_b)

        print(template.format(out_a.strip(), out_b.strip()))
        print("\n-----------------------------------\n")


if __name__ == "__main__":
    main()
