"""Dump real GPT-2 attention weights for the demo prompt; write data/attention.json.

Run from the repository root:
    uv run --extra cuda python presentations/01-gpt2-inference/attention_map.py
"""

import json
from pathlib import Path

import tiktoken
import torch

from learn_gpt2.gpt2 import GPT2

OUT = Path(__file__).parent / "data" / "attention.json"
PROMPT = "Hello, I'm a language model,"


@torch.no_grad()
def main() -> None:
    model = GPT2.from_pretrained("gpt2").eval()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(PROMPT)
    idx = torch.tensor([tokens])
    T = idx.size(1)
    x = model.transformer.wte(idx) + model.transformer.wpe(torch.arange(T))
    maps = []
    for block in model.transformer.h:
        attn = block.attn
        h = block.ln_1(x)
        q, k, _ = attn.c_attn(h).split(attn.n_embd, dim=2)
        hs = attn.n_embd // attn.n_head
        q = q.view(1, T, attn.n_head, hs).transpose(1, 2)
        k = k.view(1, T, attn.n_head, hs).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / hs**0.5
        scores = scores.masked_fill(torch.ones(T, T).triu(1).bool(), float("-inf"))
        maps.append(scores.softmax(-1)[0].tolist())  # (n_head, T, T)
        x = block(x)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"tokens": [enc.decode([t]) for t in tokens], "maps": maps}))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
