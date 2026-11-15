"""Qwen3.5-0.8B text model for inference, mirroring Hugging Face ``Qwen3_5ForCausalLM``.

Qwen3.5 is a hybrid decoder. Three of every four layers mix tokens with Gated DeltaNet, a linear
attention whose state has a fixed size no matter how long the sequence grows; every fourth layer
uses softmax attention with an output gate, grouped-query heads, QK-norm and partial rotary
position embeddings. Module and parameter names follow Hugging Face, so a checkpoint's state dict
loads without renaming:

    Qwen3_5ForCausalLM                    Qwen3_5
    ├── model (Qwen3_5TextModel)          ├── model (ModuleDict)
    │   ├── embed_tokens                  │   ├── embed_tokens
    │   ├── layers[i]                     │   ├── layers[i] (DecoderLayer)
    │   │   ├── linear_attn | self_attn   │   │   ├── linear_attn (GatedDeltaNet) | self_attn
    │   │   ├── mlp                       │   │   ├── mlp (MLP)
    │   │   ├── input_layernorm           │   │   ├── input_layernorm (RMSNorm)
    │   │   └── post_attention_layernorm  │   │   └── post_attention_layernorm (RMSNorm)
    │   ├── norm                          │   ├── norm (RMSNorm)
    │   └── rotary_emb                    │   └── rotary_emb (RotaryEmbedding)
    └── lm_head (tied to embed_tokens)    └── lm_head (tied to embed_tokens)

The file name contains dots, so ``import`` statements cannot reach it; load it with
``importlib.util.spec_from_file_location`` instead (see ``tests/test_qwen3_5.py``).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from learn_gpt2.gpt2 import SUPPORTED_DTYPES, LayerKVCache

LINEAR_ATTENTION = "linear_attention"
FULL_ATTENTION = "full_attention"
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


@dataclass(frozen=True)
class Qwen3_5Config:
    """Text model hyperparameters; the defaults are those of Qwen3.5-0.8B."""

    vocab_size: int = 248320
    hidden_size: int = 1024
    intermediate_size: int = 3584
    num_hidden_layers: int = 24
    layer_types: tuple[str, ...] = ((LINEAR_ATTENTION,) * 3 + (FULL_ATTENTION,)) * 6
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    # Full attention layers.
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25  # RoPE rotates only the first quarter of each head
    # Gated DeltaNet (linear attention) layers.
    linear_conv_kernel_dim: int = 4
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries, "
                f"num_hidden_layers is {self.num_hidden_layers}"
            )
        if unknown := set(self.layer_types) - {LINEAR_ATTENTION, FULL_ATTENTION}:
            raise ValueError(f"Unknown layer types {unknown}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be a multiple of linear_num_key_heads")
        if self.rotary_dim % 2 != 0 or not 0 < self.rotary_dim <= self.head_dim:
            raise ValueError(f"Rotary dimension {self.rotary_dim} must be even and <= head_dim")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_hf(cls, hf_config: Any) -> Qwen3_5Config:
        """Build a config from a Hugging Face ``Qwen3_5TextConfig``."""
        rope = hf_config.rope_parameters
        if rope["rope_type"] != "default":
            raise ValueError(f"Unsupported rope_type {rope['rope_type']!r}")
        if hf_config.hidden_act != "silu" or hf_config.attention_bias:
            raise ValueError("Only SiLU activations and bias-free attention are supported")
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            layer_types=tuple(hf_config.layer_types),
            rms_norm_eps=hf_config.rms_norm_eps,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            initializer_range=hf_config.initializer_range,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=hf_config.head_dim,
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            linear_conv_kernel_dim=hf_config.linear_conv_kernel_dim,
            linear_num_key_heads=hf_config.linear_num_key_heads,
            linear_num_value_heads=hf_config.linear_num_value_heads,
            linear_key_head_dim=hf_config.linear_key_head_dim,
            linear_value_head_dim=hf_config.linear_value_head_dim,
        )


class LayerLinearCache:
    """Fixed-size state of one Gated DeltaNet layer, carried from one forward call to the next.

    A KV cache grows by one entry per token. This state does not: it keeps the last ``K - 1``
    inputs of the causal convolution (kernel size ``K``) and a ``(dk, dv)`` memory matrix per
    value head. ``length`` counts the tokens folded into the state.
    """

    def __init__(self):
        self.length = 0
        self.conv_state: torch.Tensor | None = None  # (B, conv_dim, K - 1), model dtype
        self.recurrent_state: torch.Tensor | None = None  # (B, nv, dk, dv), float32

    def get(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the cached states after checking they fit the new activations."""
        if self.conv_state is not None:
            # Broadcasting would silently combine a state of batch 1 with a batch of 2.
            cached = (self.conv_state.size(0), self.conv_state.dtype, self.conv_state.device)
            new = (batch_size, dtype, device)
            if new != cached:
                raise ValueError(
                    "Linear attention cache mismatch: (batch, dtype, device) is "
                    f"{cached} in the cache but {new} for the new tokens"
                )
        return self.conv_state, self.recurrent_state

    def update(self, conv_state: torch.Tensor, recurrent_state: torch.Tensor, n_new: int) -> None:
        self.conv_state = conv_state
        self.recurrent_state = recurrent_state
        self.length += n_new


class HybridCache:
    """Per-layer caches that let decoding feed only the new tokens to the model.

    Full attention layers keep growing key/value buffers (``LayerKVCache``); Gated DeltaNet layers
    keep a fixed-size state (``LayerLinearCache``). ``max_len`` bounds the KV buffers.
    """

    def __init__(self, config: Qwen3_5Config, max_len: int):
        self.layer_types = config.layer_types
        self.max_len = max_len
        self.layers: list[LayerKVCache | LayerLinearCache] = [
            LayerKVCache(max_len) if t == FULL_ATTENTION else LayerLinearCache()
            for t in config.layer_types
        ]

    @property
    def length(self) -> int:
        return self.layers[0].length


class RMSNorm(torch.nn.Module):
    """RMSNorm with a zero-centred weight: the scale is ``1 + weight``, applied in float32."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * (1.0 + self.weight.float())).type_as(x)


class RMSNormGated(torch.nn.Module):
    """RMSNorm (plain ``weight`` scale) followed by a SiLU gate, as in Gated DeltaNet's output."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        y = self.weight * y.to(dtype)
        y = y * F.silu(gate.float())
        return y.to(dtype)


class RotaryEmbedding(torch.nn.Module):
    """Cosine and sine tables for rotary position embeddings over the first ``rotary_dim`` dims.

    Qwen3.5 uses interleaved multimodal RoPE, which splits the frequencies between time, height
    and width positions. For text all three positions equal the token index, so it reduces to
    ordinary 1D RoPE.

    ``inv_freq`` stays in float32 because ``from_pretrained`` builds the model directly in the
    target dtype. Casting a built model with ``.to(torch.bfloat16)`` would round it too.
    """

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        dim = config.rotary_dim
        exponent = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        self.register_buffer("inv_freq", 1.0 / (config.rope_theta**exponent), persistent=False)

    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions[:, None].float() * self.inv_freq.float()  # (T, rotary_dim / 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, rotary_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the first ``rotary_dim`` dims of ``q`` and ``k`` of shape (B, nh, T, hd)."""
    rotary_dim = cos.size(-1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _heads_first(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Share key heads across value heads, move heads before time, upcast, L2-normalize q and k."""
    n_rep = v.size(2) // k.size(2)
    if n_rep > 1:
        # Each query/key head serves n_rep consecutive value heads.
        q = q.repeat_interleave(n_rep, dim=2)
        k = k.repeat_interleave(n_rep, dim=2)
    q, k, v, g, beta = (
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (q, k, v, g, beta)
    )
    return l2norm(q), l2norm(k), v, g, beta


def recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the gated delta rule one token at a time; this is the definition of Gated DeltaNet.

    Takes q, k of shape (B, T, nk, dk), v of shape (B, T, nv, dv), and the log decay ``g`` and
    write strength ``beta`` of shape (B, T, nv). Each head keeps a memory S of shape (dk, dv),
    and for every token:

        S   <- exp(g_t) * S                          forget a little of everything
        S   <- S + beta_t * k_t (v_t - S^T k_t)^T    move the value stored under k_t toward v_t
        o_t  = S^T q_t                               read the memory with the query

    Returns the outputs (B, T, nv, dv) in the input dtype and the final state (B, nv, dk, dv)
    in float32.
    """
    dtype = q.dtype
    q, k, v, g, beta = _heads_first(q, k, v, g, beta)
    B, H, T, dk = k.shape
    q = q / (dk**0.5)
    state = v.new_zeros(B, H, dk, v.size(-1)) if initial_state is None else initial_state.to(v)
    out = torch.zeros_like(v)
    for t in range(T):
        q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
        state = state * g[:, :, t].exp()[..., None, None]
        kv_mem = (state * k_t[..., None]).sum(dim=-2)  # S^T k_t, (B, H, dv)
        delta = (v_t - kv_mem) * beta[:, :, t, None]
        state = state + k_t[..., None] * delta[..., None, :]
        out[:, :, t] = (state * q_t[..., None]).sum(dim=-2)
    return out.transpose(1, 2).contiguous().to(dtype), state


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``recurrent_gated_delta_rule`` with matmuls over chunks of ``chunk_size`` tokens.

    Inside a chunk, the delta rule's sequential writes form a unit lower-triangular linear system
    (the UT transform of DeltaNet). Solving it gives the corrected values ``u`` and the decayed
    keys ``w`` that read the state carried in from earlier chunks. Only the scan over chunks
    remains sequential, which makes prefill fast on GPUs.
    """
    dtype = q.dtype
    q, k, v, g, beta = _heads_first(q, k, v, g, beta)
    B, H, T, dk = k.shape
    dv = v.size(-1)
    q = q * dk**-0.5

    # Pad the sequence to whole chunks, then split it into (n_chunks, chunk_size).
    pad = (chunk_size - T % chunk_size) % chunk_size
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    g, beta = (F.pad(x, (0, pad)) for x in (g, beta))
    k_beta = k * beta[..., None]
    v_beta = v * beta[..., None]
    q, k, k_beta, v_beta = (
        x.reshape(B, H, -1, chunk_size, x.size(-1)) for x in (q, k, k_beta, v_beta)
    )
    g_cum = g.reshape(B, H, -1, chunk_size).cumsum(dim=-1)  # log decay from chunk start to t

    # decay[i, j] = exp(g_cum[i] - g_cum[j]) for j <= i: how much a write at j fades by i.
    future = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
    decay = (g_cum[..., :, None] - g_cum[..., None, :]).masked_fill(future, float("-inf")).exp()
    ut_system = (k_beta @ k.transpose(-1, -2)) * decay
    intra_attn = (q @ k.transpose(-1, -2)) * decay
    u = torch.linalg.solve_triangular(ut_system, v_beta, upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(
        ut_system, k_beta * g_cum.exp()[..., None], upper=False, unitriangular=True
    )

    state = u.new_zeros(B, H, dk, dv) if initial_state is None else initial_state.to(u)
    out = torch.zeros_like(u)
    q = q * g_cum.exp()[..., None]  # decay from chunk start, for reading the carried-in state
    k = k * (g_cum[..., -1:] - g_cum).exp()[..., None]  # decay to chunk end, for writing it
    chunk_decay = g_cum[..., -1].exp()[..., None, None]
    for i in range(q.size(2)):
        v_new = u[:, :, i] - w[:, :, i] @ state  # the part of each value the state cannot predict
        out[:, :, i] = q[:, :, i] @ state + intra_attn[:, :, i] @ v_new
        state = state * chunk_decay[:, :, i] + k[:, :, i].transpose(-1, -2) @ v_new

    out = out.reshape(B, H, -1, dv)[:, :, :T]
    return out.transpose(1, 2).contiguous().to(dtype), state


class GatedDeltaNet(torch.nn.Module):
    """Linear attention layer: causal conv, then the gated delta rule, then a gated RMSNorm."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim

        # Depthwise causal conv over q, k and v. The left context is concatenated explicitly in
        # forward, from the cache or as zeros, so the layer itself needs no padding.
        self.conv1d = torch.nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        # Per-head decay rate: g = -exp(A_log) * softplus(a + dt_bias) is always negative.
        self.dt_bias = torch.nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = torch.nn.Parameter(torch.empty(self.num_v_heads).uniform_(0.01, 16).log())
        self.norm = RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = torch.nn.Linear(self.value_dim, config.hidden_size, bias=False)
        self.in_proj_qkv = torch.nn.Linear(config.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = torch.nn.Linear(config.hidden_size, self.value_dim, bias=False)  # gate
        self.in_proj_b = torch.nn.Linear(config.hidden_size, self.num_v_heads, bias=False)  # beta
        self.in_proj_a = torch.nn.Linear(config.hidden_size, self.num_v_heads, bias=False)  # decay

    def forward(self, x: torch.Tensor, cache: LayerLinearCache | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        mixed_qkv = self.in_proj_qkv(x).transpose(1, 2)  # (B, conv_dim, T)
        z = self.in_proj_z(x).view(B, T, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)  # (B, T, nv)
        a = self.in_proj_a(x)  # (B, T, nv)

        conv_state, recurrent_state = (
            cache.get(B, mixed_qkv.dtype, mixed_qkv.device) if cache is not None else (None, None)
        )
        if conv_state is None:
            conv_state = mixed_qkv.new_zeros(B, self.conv_dim, self.conv_kernel_size - 1)
        conv_input = torch.cat((conv_state, mixed_qkv), dim=-1)  # (B, conv_dim, K - 1 + T)
        mixed_qkv = F.silu(self.conv1d(conv_input)).transpose(1, 2)  # (B, T, conv_dim)

        q, k, v = mixed_qkv.split((self.key_dim, self.key_dim, self.value_dim), dim=-1)
        q = q.reshape(B, T, self.num_k_heads, self.head_k_dim)
        k = k.reshape(B, T, self.num_k_heads, self.head_k_dim)
        v = v.reshape(B, T, self.num_v_heads, self.head_v_dim)
        beta = b.sigmoid()
        # Upcast before exp: in float16, A_log.exp() can overflow to inf.
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if recurrent_state is not None and T == 1:
            y, state = recurrent_gated_delta_rule(q, k, v, g, beta, recurrent_state)
        else:
            y, state = chunk_gated_delta_rule(q, k, v, g, beta, recurrent_state)
        if cache is not None:
            cache.update(conv_input[:, :, T:].clone(), state, T)

        y = self.norm(y, z).reshape(B, T, self.value_dim)  # (B, T, nv * dv)
        return self.out_proj(y)


class Attention(torch.nn.Module):
    """Softmax attention with grouped-query heads, QK-norm, partial RoPE and an output gate."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        # q_proj yields a query and an output gate per head, laid out as [q_h, gate_h] per head.
        self.q_proj = torch.nn.Linear(
            config.hidden_size, 2 * self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: LayerKVCache | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, gate = self.q_proj(x).view(B, T, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)  # (B, nkv, T, hd)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)  # (B, nkv, L, hd), L = cached + T
        L = k.size(2)
        if T == L:
            # No earlier positions: plain causal attention.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        elif T == 1:
            # A single new token may attend to every cached position.
            y = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        else:
            # Query i sits at absolute position L - T + i and sees keys up to that position.
            mask = torch.ones(T, L, dtype=torch.bool, device=x.device).tril(diagonal=L - T)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        y = y.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        y = y * torch.sigmoid(gate.reshape(B, T, self.num_heads * self.head_dim))
        return self.o_proj(y)


class MLP(torch.nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.gate_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(torch.nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == LINEAR_ATTENTION:
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: LayerKVCache | LayerLinearCache | None = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        if self.layer_type == LINEAR_ATTENTION:
            assert cache is None or isinstance(cache, LayerLinearCache)
            h = self.linear_attn(h, cache)
        else:
            assert cache is None or isinstance(cache, LayerKVCache)
            h = self.self_attn(h, cos, sin, cache)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class Qwen3_5(torch.nn.Module):
    def __init__(self, config: Qwen3_5Config):
        super().__init__()
        self.config = config
        self.model = torch.nn.ModuleDict(
            dict(
                embed_tokens=torch.nn.Embedding(config.vocab_size, config.hidden_size),
                layers=torch.nn.ModuleList(
                    [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
                ),
                norm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
                rotary_emb=RotaryEmbedding(config),
            )
        )
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:
        # Mirrors Hugging Face: norms that scale by (1 + weight) start at zero, the rest at one.
        std = self.config.initializer_range
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.zeros_(module.weight)
        elif isinstance(module, RMSNormGated):
            torch.nn.init.ones_(module.weight)

    def forward(self, idx: torch.Tensor, kv_cache: HybridCache | None = None) -> torch.Tensor:
        """Return logits for ``idx``.

        With ``kv_cache``, ``idx`` holds only the tokens that are not cached yet; they are placed
        after the cached positions, and every layer's cache absorbs them.
        """
        B, T = idx.shape
        start = 0
        if kv_cache is not None:
            if kv_cache.layer_types != self.config.layer_types:
                raise ValueError("Cache layer types do not match the model's layer types")
            start = kv_cache.length
            # Check before any layer runs; linear layers have no max_len of their own and
            # would otherwise absorb the tokens before a later KV layer rejects them.
            if start + T > kv_cache.max_len:
                raise ValueError(
                    f"KV cache overflow: {start + T} positions exceed max_len {kv_cache.max_len}"
                )
        x = self.model.embed_tokens(idx)  # (B, T, hidden_size)
        positions = torch.arange(start, start + T, dtype=torch.long, device=idx.device)
        cos, sin = self.model.rotary_emb(positions, x.dtype)  # (T, rotary_dim)
        for i, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, kv_cache.layers[i] if kv_cache is not None else None)
        x = self.model.norm(x)
        return self.lm_head(x)  # (B, T, vocab_size)

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

        ``temperature=0`` decodes greedily. With ``use_cache``, the prompt is run once to fill the
        caches and each later step feeds only the newest token. RoPE has no learned position
        table, so unlike GPT-2 the context is never cropped.
        """
        kv_cache = HybridCache(self.config, idx.size(1) + max_new_tokens) if use_cache else None
        for _ in range(max_new_tokens):
            logits = self(idx) if kv_cache is None else self(idx[:, kv_cache.length :], kv_cache)
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
    def from_pretrained(
        cls, model_type: str = DEFAULT_MODEL, *, dtype: torch.dtype = torch.float32
    ) -> Qwen3_5:
        """Load the text model weights of a Hugging Face Qwen3.5 checkpoint.

        The vision tower and multi-token prediction weights in the checkpoint are skipped by
        ``transformers.Qwen3_5ForCausalLM``; its state dict then matches this model key for key.
        """
        from transformers import Qwen3_5ForCausalLM

        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype}, expected one of {SUPPORTED_DTYPES}")

        model_hf = Qwen3_5ForCausalLM.from_pretrained(model_type, dtype=dtype)
        # Build in the target dtype rather than casting afterwards, so inv_freq stays float32.
        with _default_dtype(dtype):
            model = cls(Qwen3_5Config.from_hf(model_hf.config))
        sd = model.state_dict()
        sd_hf = model_hf.state_dict()

        missing = sd.keys() - sd_hf.keys()
        unexpected = sd_hf.keys() - sd.keys()
        if missing or unexpected:
            raise ValueError(f"State dict mismatch: missing={missing}, unexpected={unexpected}")

        with torch.no_grad():
            for k, v in sd_hf.items():
                if sd[k].shape != v.shape:
                    raise ValueError(f"Shape mismatch for {k}: {sd[k].shape} vs {v.shape}")
                sd[k].copy_(v)
        return model
