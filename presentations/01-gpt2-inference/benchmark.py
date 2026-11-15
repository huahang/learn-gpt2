"""Measure GPT-2 decoding with and without the KV cache; write data/benchmark.json.

Run from the repository root:
    uv run --extra cuda python presentations/01-gpt2-inference/benchmark.py
"""

import json
import statistics
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from learn_gpt2.gpt2 import GPT2, KVCache

OUT = Path(__file__).parent / "data" / "benchmark.json"
PROMPT = "Hello, I'm a language model,"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def per_step_latency(model: GPT2, prompt: torch.Tensor, n_steps: int, use_cache: bool) -> list:
    """Greedy-decode n_steps tokens, timing each step (forward + argmax + append)."""
    idx = prompt
    cache = KVCache(model.config.n_layer, prompt.size(1) + n_steps) if use_cache else None
    times = []
    for _ in range(n_steps):
        sync()
        t0 = time.perf_counter()
        logits = model(idx) if cache is None else model(idx[:, cache.length :], cache)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
        sync()
        times.append(time.perf_counter() - t0)
    return [times, idx]


def run_model(model_type: str, n_steps: int, repeats: int = 3) -> dict:
    model = GPT2.from_pretrained(model_type).to(DEVICE).eval()
    enc = tiktoken.get_encoding("gpt2")
    prompt = torch.tensor([enc.encode(PROMPT)], device=DEVICE)
    per_step_latency(model, prompt, 32, True)  # warm up kernels
    per_step_latency(model, prompt, 32, False)
    result = {"model": model_type, "prompt_len": prompt.size(1), "n_steps": n_steps}
    outputs = {}
    for use_cache in (True, False):
        runs = [per_step_latency(model, prompt, n_steps, use_cache) for _ in range(repeats)]
        # Median over repeats for every step smooths out scheduler noise.
        steps = [statistics.median(r[0][i] for r in runs) for i in range(n_steps)]
        key = "cached" if use_cache else "uncached"
        result[f"{key}_step_ms"] = [round(s * 1000, 4) for s in steps]
        result[f"{key}_total_s"] = round(sum(steps), 4)
        outputs[key] = runs[0][1]
    result["identical_tokens"] = bool(torch.equal(outputs["cached"], outputs["uncached"]))
    result["speedup"] = round(result["uncached_total_s"] / result["cached_total_s"], 2)
    result["continuation"] = enc.decode(outputs["cached"][0, prompt.size(1) :][:40].tolist())
    print(
        f"{model_type}: cached {result['cached_total_s']:.2f}s, "
        f"uncached {result['uncached_total_s']:.2f}s, x{result['speedup']}, "
        f"identical={result['identical_tokens']}"
    )
    del model
    torch.cuda.empty_cache()
    return result


@torch.no_grad()
def next_token_distribution() -> dict:
    model = GPT2.from_pretrained("gpt2").to(DEVICE).eval()
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(PROMPT)
    logits = model(torch.tensor([tokens], device=DEVICE))[0, -1].float()
    dists = {}
    for temperature in (1.0, 0.7):
        probs = F.softmax(logits / temperature, dim=-1)
        top = torch.topk(probs, 10)
        dists[str(temperature)] = [
            {"token": enc.decode([i]), "prob": round(p, 5)}
            for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True)
        ]
    return {
        "prompt": PROMPT,
        "token_ids": tokens,
        "token_strs": [enc.decode([t]) for t in tokens],
        "top10": dists,
        "greedy": enc.decode([int(logits.argmax())]),
    }


def flops_model(n_layer: int, d: int, vocab: int, prompt_len: int, n_steps: int) -> dict:
    """Matmul FLOPs per decoding step (2 per multiply-add), following gpt2.py's forward."""
    linear = 24 * n_layer * d * d  # qkv 3d^2 + proj d^2 + mlp 8d^2, times 2
    head = 2 * d * vocab  # lm_head runs on every position fed in

    def attn(q: int, k: int) -> int:
        return 4 * n_layer * q * k * d  # QK^T and AV, all heads together

    uncached, cached = [], []
    for step in range(n_steps):
        t = prompt_len + step  # tokens visible at this step
        uncached.append(t * (linear + head) + attn(t, t))
        if step == 0:
            cached.append(t * (linear + head) + attn(t, t))  # prefill
        else:
            cached.append(linear + head + attn(1, t))
    return {"uncached": uncached, "cached": cached}


def main() -> None:
    torch.manual_seed(0)
    data = {
        "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
        "torch": torch.__version__,
        "distribution": next_token_distribution(),
        "runs": [run_model("gpt2", 1000), run_model("gpt2-xl", 1000, repeats=1)],
        "flops_gpt2": flops_model(12, 768, 50257, 8, 1000),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
