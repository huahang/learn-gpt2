"""Turn measurements and code snippets into build/deck_data.json for deck.js."""

import json
from pathlib import Path

from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Token

HERE = Path(__file__).parent
BUILD = HERE / "build"

# Syntax colours for code on the dark ink panels.
CODE_COLORS = [
    (Comment, "8D99AE"),
    (Keyword, "C792EA"),
    (Name.Builtin, "82AAFF"),
    (Name.Function, "82AAFF"),
    (String, "C3E88D"),
    (Number, "F78C6C"),
    (Operator, "89DDFF"),
    (Punctuation, "89DDFF"),
]
DEFAULT_CODE = "E6E9F2"

# Excerpts of src/learn_gpt2/gpt2.py, dedented, with comments shortened for the slides.
SNIPPETS = {
    "loop": """\
for _ in range(max_new_tokens):
    logits = self(idx)           # 整段前缀重跑
    logits = logits[:, -1, :]    # 只要最后位置
    next_token = sample(logits)  # (B, 1)
    idx = torch.cat((idx, next_token), dim=1)""",
    "forward": """\
pos = torch.arange(start, start + T, device=idx.device)
tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
x = tok_emb + pos_emb
for i, block in enumerate(self.transformer.h):
    x = block(x, kv_cache.layers[i] if kv_cache else None)
x = self.transformer.ln_f(x)
logits = self.lm_head(x)             # (B, T, vocab_size)""",
    "block": """\
# Block.forward
x = x + self.attn(self.ln_1(x), kv_cache)
x = x + self.mlp(self.ln_2(x))

# MLP.forward：768 → 3072 → GELU → 768
return self.c_proj(self.gelu(self.c_fc(x)))""",
    "qkv": """\
B, T, C = x.shape
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)""",
    "cache": """\
def update(self, k, v):  # k, v: (B, nh, T, hs)
    B, nh, T, hs = k.shape
    end = self.length + T
    if end > self.max_len:
        raise ValueError("KV cache overflow ...")
    if self.k is None:  # 首次调用：惰性分配
        self.k = k.new_empty(B, nh, self.max_len, hs)
        self.v = v.new_empty(B, nh, self.max_len, hs)
    # （省略：校验 batch、头数、dtype、device）
    self.k[:, :, self.length : end] = k
    self.v[:, :, self.length : end] = v
    self.length = end
    return self.k[:, :, :end], self.v[:, :, :end]""",
    "masks": """\
if kv_cache is not None:
    k, v = kv_cache.update(k, v)    # (B, nh, L, hs)
L = k.size(2)
if T == L:     # ① 没有旧位置
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
elif T == 1:   # ② 单个新 token 看全部缓存
    y = F.scaled_dot_product_attention(q, k, v)
else:          # ③ 第 i 个新 token 位于 L - T + i
    mask = torch.ones(T, L, dtype=torch.bool).tril(diagonal=L - T)
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)""",
    "pos": """\
# forward()
start = kv_cache.length if kv_cache else 0
pos = torch.arange(start, start + T)""",
    "generate": """\
# generate()
for _ in range(max_new_tokens):
    if kv_cache is not None and idx.size(1) > kv_cache.max_len:
        kv_cache = None          # 超过 block_size：丢弃缓存
    if kv_cache is None:
        logits = self(idx[:, -self.block_size :])
    else:
        logits = self(idx[:, kv_cache.length :], kv_cache)""",
}


def color_of(ttype) -> tuple[str, bool]:
    for base, color in CODE_COLORS:
        if ttype in base:
            return color, base is Comment
    return DEFAULT_CODE, False


def highlight(code: str) -> list[list[dict]]:
    """Return lines of runs: [{"text", "color", "italic"}]."""
    lines: list[list[dict]] = [[]]
    for ttype, value in PythonLexer().get_tokens(code):
        if ttype is Token.Text.Whitespace and value == "\n":
            lines.append([])
            continue
        color, italic = color_of(ttype)
        for i, part in enumerate(value.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append({"text": part, "color": color, "italic": italic})
    while lines and not lines[-1]:
        lines.pop()
    return lines


def bucket_mean(values: list[float], size: int) -> list[float]:
    return [
        sum(values[i : i + size]) / len(values[i : i + size]) for i in range(0, len(values), size)
    ]


def main() -> None:
    bench = json.loads((HERE / "data" / "benchmark.json").read_text())
    runs = {r["model"]: r for r in bench["runs"]}
    flops = bench["flops_gpt2"]
    step = 10
    labels = [str(i + 1) for i in range(0, 1000, step)]
    data = {
        "code": {k: highlight(v) for k, v in SNIPPETS.items()},
        "device": bench["device"],
        "distribution": bench["distribution"],
        "latency": {
            "labels": labels,
            "gpt2_cached": [round(v, 2) for v in bucket_mean(runs["gpt2"]["cached_step_ms"], step)],
            "gpt2_uncached": [
                round(v, 2) for v in bucket_mean(runs["gpt2"]["uncached_step_ms"], step)
            ],
            "xl_cached": [
                round(v, 1) for v in bucket_mean(runs["gpt2-xl"]["cached_step_ms"], step)
            ],
            "xl_uncached": [
                round(v, 1) for v in bucket_mean(runs["gpt2-xl"]["uncached_step_ms"], step)
            ],
        },
        "runs": {
            m: {
                k: r[k]
                for k in ("cached_total_s", "uncached_total_s", "speedup", "identical_tokens")
            }
            for m, r in runs.items()
        },
        "flops": {
            "labels": labels,
            "uncached": [round(v / 1e9, 2) for v in bucket_mean(flops["uncached"], step)],
            "cached": [round(v / 1e9, 3) for v in bucket_mean(flops["cached"], step)],
            "total_uncached_tflops": round(sum(flops["uncached"]) / 1e12, 1),
            "total_cached_tflops": round(sum(flops["cached"]) / 1e12, 2),
            "ratio": round(sum(flops["uncached"]) / sum(flops["cached"])),
        },
        "narration": {
            n["id"]: n["text"] for n in json.loads((HERE / "narration.json").read_text())
        },
    }
    BUILD.mkdir(exist_ok=True)
    (BUILD / "deck_data.json").write_text(json.dumps(data, ensure_ascii=False))
    print(
        "wrote build/deck_data.json;",
        data["flops"]["total_uncached_tflops"],
        data["flops"]["total_cached_tflops"],
        data["flops"]["ratio"],
    )


if __name__ == "__main__":
    main()
