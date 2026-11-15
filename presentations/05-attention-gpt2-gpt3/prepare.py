"""Turn code snippets into syntax-highlighted runs in build/deck_data.json for deck.js."""

import json
from pathlib import Path

from pygments.lexers import PythonLexer
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, String, Token

HERE = Path(__file__).parent
BUILD = HERE / "build"

# Syntax colours for code on the dark ink panels (same as deck 01).
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

# Excerpts of src/learn_gpt2/gpt2.py (comments shortened), plus a hand-written equivalent.
SNIPPETS = {
    "attn": """\
# CausalSelfAttention.forward（无缓存时）
B, T, C = x.shape
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
q = q.view(B, T, nh, C // nh).transpose(1, 2)
k = k.view(B, T, nh, C // nh).transpose(1, 2)
v = v.view(B, T, nh, C // nh).transpose(1, 2)
y = F.scaled_dot_product_attention(
    q, k, v, is_causal=True)    # 下三角掩码
y = y.transpose(1, 2).reshape(B, T, C)
return self.c_proj(y)           # 输出投影""",
    "manual": """\
# 等价的手写版本
att = q @ k.transpose(-2, -1)
att = att / math.sqrt(k.size(-1))
mask = torch.ones(T, T).tril().bool()
att = att.masked_fill(~mask, float("-inf"))
att = F.softmax(att, dim=-1)    # 每行和为 1
y = att @ v                     # 加权求和""",
    "kvloop": """\
# GPT2.generate：只送缓存里还没有的 token
logits = self(idx[:, kv_cache.length :], kv_cache)

# GPT2.forward：位置编号接在缓存之后
start = kv_cache.length
pos = torch.arange(start, start + T)

# LayerKVCache.update
end = self.length + T
if self.k is None:              # 首次调用：一次分配
    self.k = k.new_empty(B, nh, self.max_len, hs)
    self.v = v.new_empty(B, nh, self.max_len, hs)
self.k[:, :, self.length : end] = k
self.v[:, :, self.length : end] = v
self.length = end
return self.k[:, :, :end], self.v[:, :, :end]""",
}


def color_of(ttype) -> str:
    for base, color in CODE_COLORS:
        if ttype in base:
            return color
    return DEFAULT_CODE


def highlight(code: str) -> list[list[dict]]:
    """Return lines of runs: [{"text", "color"}]."""
    lines: list[list[dict]] = [[]]
    for ttype, value in PythonLexer().get_tokens(code):
        if ttype is Token.Text.Whitespace and value == "\n":
            lines.append([])
            continue
        color = color_of(ttype)
        for i, part in enumerate(value.split("\n")):
            if i:
                lines.append([])
            if part:
                lines[-1].append({"text": part, "color": color})
    while lines and not lines[-1]:
        lines.pop()
    return lines


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    data = {"code": {k: highlight(v) for k, v in SNIPPETS.items()}}
    (BUILD / "deck_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"wrote {BUILD / 'deck_data.json'}")


if __name__ == "__main__":
    main()
