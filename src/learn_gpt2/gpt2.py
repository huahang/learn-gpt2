import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


GPT2_CONFIGS: dict[str, GPT2Config] = {
    "gpt2": GPT2Config(n_layer=12, n_head=12, n_embd=768),  # 124M
    "gpt2-medium": GPT2Config(n_layer=24, n_head=16, n_embd=1024),  # 355M
    "gpt2-large": GPT2Config(n_layer=36, n_head=20, n_embd=1280),  # 774M
    "gpt2-xl": GPT2Config(n_layer=48, n_head=25, n_embd=1600),  # 1558M
}

SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Hugging Face stores these as Conv1D weights of shape (in, out); torch.nn.Linear expects (out, in).
_TRANSPOSED_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


class LayerKVCache:
    """Key/value buffers of one attention layer, appended to as tokens are decoded.

    Buffers are allocated lazily on the first update so they match the batch size, dtype and
    device of the activations. ``length`` counts the positions already stored.
    """

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.length = 0
        self.k: torch.Tensor | None = None  # (B, nh, max_len, hs)
        self.v: torch.Tensor | None = None  # (B, nh, max_len, hs)

    def update(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``k`` and ``v`` of shape (B, nh, T, hs); return all cached keys and values."""
        B, nh, T, hs = k.shape
        end = self.length + T
        if end > self.max_len:
            raise ValueError(f"KV cache overflow: {end} positions exceed max_len {self.max_len}")
        if self.k is None or self.v is None:
            self.k = k.new_empty(B, nh, self.max_len, hs)
            self.v = v.new_empty(B, nh, self.max_len, hs)
        else:
            # Slice assignment would silently broadcast or cast mismatched activations.
            cached = (self.k.size(0), self.k.size(1), self.k.size(3), self.k.dtype, self.k.device)
            new = (B, nh, hs, k.dtype, k.device)
            if new != cached:
                raise ValueError(
                    "KV cache mismatch: (batch, heads, head_size, dtype, device) is "
                    f"{cached} in the cache but {new} for the new tokens"
                )
        self.k[:, :, self.length : end] = k
        self.v[:, :, self.length : end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]


class KVCache:
    """Per-layer key/value caches that let decoding feed only the new tokens to the model.

    Without a cache, every decoding step recomputes keys and values for the whole prefix, so
    generating N tokens costs O(N^2) token forwards. With a cache, each step only runs the
    newest token through the network and attends to the stored keys and values.
    """

    def __init__(self, n_layer: int, max_len: int):
        self.layers = [LayerKVCache(max_len) for _ in range(n_layer)]

    @property
    def length(self) -> int:
        return self.layers[0].length

    @property
    def max_len(self) -> int:
        return self.layers[0].max_len


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor, kv_cache: LayerKVCache | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)  # (B, nh, L, hs), L = cached + T
        L = k.size(2)
        if T == L:
            # No earlier positions: plain causal attention.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif T == 1:
            # A single new token may attend to every cached position.
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # Query i sits at absolute position L - T + i and sees keys up to that position.
            # is_causal=True would align the mask to the top-left corner, which is wrong here.
            mask = torch.ones(T, L, dtype=torch.bool, device=x.device).tril(diagonal=L - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.c_proj(y)


class MLP(torch.nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = torch.nn.GELU(approximate="tanh")
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(torch.nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = torch.nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, kv_cache: LayerKVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), kv_cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(torch.nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(config.vocab_size, config.n_embd),
                wpe=torch.nn.Embedding(config.block_size, config.n_embd),
                h=torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=torch.nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # GPT-2 scales residual projections by 1/sqrt(N), where N counts the two residual
        # additions (attention and MLP) in each block.
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, kv_cache: KVCache | None = None) -> torch.Tensor:
        """Return logits for ``idx``.

        With ``kv_cache``, ``idx`` holds only the tokens that are not cached yet; they are placed
        after the cached positions, and their keys and values are appended to the cache.
        """
        B, T = idx.shape
        start = 0
        if kv_cache is not None:
            if len(kv_cache.layers) != len(self.transformer.h):
                raise ValueError(
                    f"KV cache has {len(kv_cache.layers)} layers, "
                    f"model has {len(self.transformer.h)}"
                )
            start = kv_cache.length
        assert start + T <= self.block_size, (
            f"Cannot forward sequence of length {start + T}, block size is only {self.block_size}"
        )
        pos = torch.arange(start, start + T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb
        for i, block in enumerate(self.transformer.h):
            x = block(x, kv_cache.layers[i] if kv_cache is not None else None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Extend ``idx`` of shape (B, T) by ``max_new_tokens`` sampled tokens.

        ``temperature=0`` decodes greedily. With ``use_cache``, the prompt is run once to fill a
        KV cache and each later step feeds only the newest token. Learned absolute position
        embeddings cap the context at ``block_size``; once the sequence outgrows it, cached
        positions would shift, so generation falls back to recomputing the cropped context.
        """
        B, T = idx.shape
        kv_cache = (
            KVCache(self.config.n_layer, min(T + max_new_tokens, self.block_size))
            if use_cache
            else None
        )
        for _ in range(max_new_tokens):
            if kv_cache is not None and idx.size(1) > kv_cache.max_len:
                kv_cache = None
            if kv_cache is None:
                logits = self(idx[:, -self.block_size :])
            else:
                logits = self(idx[:, kv_cache.length :], kv_cache)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type: str, *, dtype: torch.dtype = torch.float32) -> GPT2:
        from transformers import GPT2LMHeadModel

        if model_type not in GPT2_CONFIGS:
            raise ValueError(
                f"Unknown model type {model_type!r}, expected one of {list(GPT2_CONFIGS)}"
            )
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype}, expected one of {SUPPORTED_DTYPES}")

        model = cls(GPT2_CONFIGS[model_type]).to(dtype)
        sd = model.state_dict()
        sd_hf = {
            k: v
            for k, v in GPT2LMHeadModel.from_pretrained(model_type).state_dict().items()
            if not k.endswith((".attn.bias", ".attn.masked_bias"))
        }

        missing = sd.keys() - sd_hf.keys() - {"lm_head.weight"}
        unexpected = sd_hf.keys() - sd.keys()
        if missing or unexpected:
            raise ValueError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")

        with torch.no_grad():
            for k, v in sd_hf.items():
                if k.endswith(_TRANSPOSED_WEIGHTS):
                    v = v.t()
                if sd[k].shape != v.shape:
                    raise ValueError(f"Shape mismatch for {k}: {sd[k].shape} vs {v.shape}")
                sd[k].copy_(v)
        return model
