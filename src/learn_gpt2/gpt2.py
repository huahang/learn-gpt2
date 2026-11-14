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


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        )
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        return logits

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
