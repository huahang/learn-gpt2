"""GPT-2 inference in float16 whose operators are written in the Triton language.

``gpt2.py`` is the reference: it builds the model from PyTorch modules. This file computes the
same function with hand-written GPU kernels and loads the weights from a safetensors file. Only
token sampling still uses PyTorch operators.

Optimizations, and where each one lives:

- Zero-copy load: safetensors memory-maps the file and copies tensors straight to the GPU.
- Half precision with float32 accumulation: weights and activations are float16, which halves
  the memory traffic and lets ``tl.dot`` use tensor cores. Sums, the softmax and the logits are
  float32.
- GEMM: every product of two matrices is a tiled ``tl.dot``. That covers the four linear layers
  of a block, the LM head and the two products inside attention (scores, weighted values).
  ``_matmul_tile`` is the shared inner loop and lists its techniques; ``_matmul_tiling`` picks
  the tile sizes for each matrix shape. A decode step puts all sequences into one row tile, so B
  matrix-vector products become one matrix-matrix product.
- Weight layout: matrices are transposed once at load time to (out, in), so each program reads
  whole contiguous rows. The LM head reads the token embedding in that layout as it is, so the
  tied weight exists once.
- Constant folding: the attention scale 1/sqrt(head_size) is multiplied into the query
  projection when the weights are loaded (``_fold_attention_scale``).
- Operator fusion: every kernel does the work of several PyTorch operators.
  ``_embed_kernel`` gathers token and position embeddings and adds them.
  ``_qkv_kernel`` projects to query/key/value, splits the heads and appends to the KV cache.
  ``_attention_kernel`` is flash attention: scores, causal mask, softmax and the weighted sum of
  values in one pass, without storing the (T, L) score matrix.
  ``_linear_kernel`` adds the bias, applies GELU and adds the residual in its epilogue.
- KV cache: decoding feeds one token per step and attends to stored keys and values.
- Static memory plan: ``_Workspace`` allocates every buffer once. Query, key and value live in
  one arena, so no step allocates, concatenates or transposes.
- Phase-specific kernels: the prompt pass (many rows) and the decode step (one row per
  sequence) use different tile sizes. The decode step also folds LayerNorm into the following
  matrix kernel to launch fewer kernels.
- Last-token logits: generation computes logits only for the last position.
- CUDA graph: one decode step is captured once and replayed. The cache length lives in GPU
  memory, so one graph serves every step and a step costs Python a single call.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors.torch import load_file

from learn_gpt2.gpt2 import GPT2_CONFIGS, GPT2Config

DTYPE = torch.float16
LAYER_NORM_EPS = 1e-5

# Legacy causal-mask buffers and the tied LM head carry no information of their own.
_IGNORED_SUFFIXES = (".attn.bias", ".attn.masked_bias", "lm_head.weight")


# --------------------------------------------------------------------------------------------
# Triton kernels
# --------------------------------------------------------------------------------------------


@triton.jit
def _embed_kernel(idx_ptr, wte_ptr, wpe_ptr, x_ptr, pos_ptr, T, C, BLOCK_C: tl.constexpr):
    """x[b, t] = wte[idx[b, t]] + wpe[pos + t]. One program per row."""
    row = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    token = tl.load(idx_ptr + row)
    position = tl.load(pos_ptr) + row % T
    tok_emb = tl.load(wte_ptr + token * C + offs_c, mask=mask)
    pos_emb = tl.load(wpe_ptr + position * C + offs_c, mask=mask)
    tl.store(x_ptr + row.to(tl.int64) * C + offs_c, tok_emb + pos_emb, mask=mask)


@triton.jit
def _layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, C, eps, BLOCK_C: tl.constexpr):
    """out[row] = (x[row] - mean) / sqrt(var + eps) * w + b. One program per row."""
    row = tl.program_id(0).to(tl.int64) * C
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    x = tl.load(x_ptr + row + offs_c, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    centered = tl.where(mask, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(centered * centered, axis=0) / C + eps)
    w = tl.load(w_ptr + offs_c, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs_c, mask=mask).to(tl.float32)
    out = centered * rstd * w + b
    tl.store(out_ptr + row + offs_c, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tile_ids(M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    """Map the program id to a (row tile, column tile) pair.

    Programs are ordered so that consecutive ones cover GROUP_M row tiles of the same columns.
    Programs that run together then read the same weight rows, which keeps those rows in the
    L2 cache when the whole weight does not fit. GPT-2 small weights fit, and the order made no
    measurable difference there.
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group_span = GROUP_M * num_n
    first_m = (pid // group_span) * GROUP_M
    group_m = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_span) % group_m
    pid_n = (pid % group_span) // group_m
    return pid_m, pid_n


@triton.jit
def _load(ptrs, mask, MASKED: tl.constexpr):
    """Load ``ptrs``; where ``mask`` is false read zero, but only if ``MASKED`` asks for it."""
    if MASKED:  # noqa: SIM108 - Triton resolves this branch at compile time
        tile = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tile = tl.load(ptrs)
    return tile


@triton.jit
def _matmul_tile(
    a_ptr,
    w_ptr,
    ln_w_ptr,
    ln_b_ptr,
    offs_m,
    offs_n,
    M,
    N,
    K,
    stride_am,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    MASK_K: tl.constexpr,
    FUSE_LN: tl.constexpr,
):
    """Return one float32 (BLOCK_M, BLOCK_N) tile of ``a @ w.T``, or ``layer_norm(a) @ w.T``.

    This is the GEMM inner loop shared by every matrix kernel:

    - ``w`` has shape (N, K), one contiguous row per output column, so a program streams
      BLOCK_N whole rows and every load is a wide vector load.
    - ``tl.dot`` multiplies a (BLOCK_M, BLOCK_K) tile by a (BLOCK_K, BLOCK_N) tile on tensor
      cores. The inputs are float16; the accumulator is float32.
    - The loads carry no mask. A tile that sticks out past the last row re-reads that row, and
      the caller masks the store instead. Only if BLOCK_K does not divide K (``MASK_K``) do the
      loads mask the tail of each row.
    - ``num_stages`` at launch makes the compiler fetch the tiles of later iterations while the
      current ones are multiplied.
    """
    a_rows = a_ptr + tl.minimum(offs_m, M - 1)[:, None].to(tl.int64) * stride_am
    w_rows = w_ptr + tl.minimum(offs_n, N - 1)[:, None] * K

    if FUSE_LN:
        # Each program owns whole rows, so it can compute the row statistics itself. This costs
        # one more read of ``a``, which is cheap when there are few rows (the decode step).
        total = tl.zeros([BLOCK_M], dtype=tl.float32)
        total_sq = tl.zeros([BLOCK_M], dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)
            a = _load(a_rows + offs_k[None, :], offs_k[None, :] < K, MASK_K).to(tl.float32)
            total += tl.sum(a, axis=1)
            total_sq += tl.sum(a * a, axis=1)
        mean = total / K
        rstd = tl.rsqrt(tl.maximum(total_sq / K - mean * mean, 0.0) + eps)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = _load(a_rows + offs_k[None, :], k_mask[None, :], MASK_K)
        # Loaded as (BLOCK_N, BLOCK_K) so that the last axis is contiguous in memory.
        w = _load(w_rows + offs_k[None, :], k_mask[None, :], MASK_K)
        if FUSE_LN:
            # Masked columns load a zero gain and shift, so they stay zero after normalizing.
            gain = _load(ln_w_ptr + offs_k, k_mask, MASK_K).to(tl.float32)
            shift = _load(ln_b_ptr + offs_k, k_mask, MASK_K).to(tl.float32)
            normed = (a.to(tl.float32) - mean[:, None]) * rstd[:, None] * gain[None, :]
            a = (normed + shift[None, :]).to(w.dtype)
        acc = tl.dot(a, tl.trans(w), acc)
    return acc


@triton.jit
def _linear_kernel(
    a_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    ln_w_ptr,
    ln_b_ptr,
    M,
    N,
    K,
    stride_am,
    stride_om,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MASK_K: tl.constexpr,
    FUSE_LN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    GELU: tl.constexpr,
    RESIDUAL: tl.constexpr,
):
    """out = [out +] [gelu](layer_norm?(a) @ w.T [+ bias]), fused in one kernel."""
    pid_m, pid_n = _tile_ids(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = _matmul_tile(
        a_ptr, w_ptr, ln_w_ptr, ln_b_ptr, offs_m, offs_n, M, N, K, stride_am, eps,
        BLOCK_M, BLOCK_N, BLOCK_K, MASK_K, FUSE_LN,
    )  # fmt: skip
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    if GELU:
        # tanh approximation: 0.5 * x * (1 + tanh(z)) with z = sqrt(2 / pi) * (x + 0.044715 x^3).
        # Since 0.5 * (1 + tanh(z)) == sigmoid(2 z), this needs one exponential per element.
        acc = acc * tl.sigmoid(1.5957691216057308 * (acc + 0.044715 * acc * acc * acc))
    out_ptrs = out_ptr + offs_m[:, None].to(tl.int64) * stride_om + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if RESIDUAL:
        # In place: this program is the only one that reads or writes its tile of ``out``.
        acc += tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


# kv_slot is 1 for the first layer; Triton would otherwise compile a variant for that value.
@triton.jit(do_not_specialize=["kv_slot"])
def _qkv_kernel(
    a_ptr,
    w_ptr,
    bias_ptr,
    arena_ptr,
    pos_ptr,
    ln_w_ptr,
    ln_b_ptr,
    M,
    C,
    T,
    kv_slot,
    stride_am,
    stride_slot,
    stride_b,
    stride_h,
    stride_l,
    eps,
    HEAD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    MASK_K: tl.constexpr,
    FUSE_LN: tl.constexpr,
):
    """Project to query/key/value and store them per head, appending to the KV cache.

    The arena has shape (slot, batch, head, position, head_size). Slot 0 holds the queries of the
    current pass; slots ``kv_slot`` and ``kv_slot + 1`` hold this layer's keys and values. Writing
    there directly replaces split, view, transpose and the cache copy.
    """
    N = 3 * C
    pid_m, pid_n = _tile_ids(M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = _matmul_tile(
        a_ptr, w_ptr, ln_w_ptr, ln_b_ptr, offs_m, offs_n, M, N, C, stride_am, eps,
        BLOCK_M, BLOCK_N, BLOCK_K, MASK_K, FUSE_LN,
    )  # fmt: skip
    acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]

    # Column n is element ``dim`` of head ``head`` of part ``part`` (0 query, 1 key, 2 value).
    part = offs_n // C
    head = (offs_n % C) // HEAD_SIZE
    dim = offs_n % HEAD_SIZE
    slot = tl.where(part == 0, 0, kv_slot + part - 1)
    col = slot.to(tl.int64) * stride_slot + head * stride_h + dim
    # Row m is token t of sequence b; it lands at position ``pos + t``, after the cached tokens.
    position = tl.load(pos_ptr) + offs_m % T
    row = (offs_m // T).to(tl.int64) * stride_b + position * stride_l
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out = acc.to(arena_ptr.dtype.element_ty)
    tl.store(arena_ptr + row[:, None] + col[None, :], out, mask=mask)


@triton.jit(do_not_specialize=["kv_slot"])
def _attention_kernel(
    arena_ptr,
    y_ptr,
    pos_ptr,
    T,
    n_head,
    kv_slot,
    stride_slot,
    stride_b,
    stride_h,
    stride_l,
    stride_ym,
    HEAD_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    MASK_D: tl.constexpr,
):
    """Causal flash attention of the T new queries over all ``pos + T`` cached keys.

    One program handles BLOCK_Q queries of one head and walks over the keys in tiles. Each tile
    takes two ``tl.dot`` products: queries times keys for the scores, then probabilities times
    values. The program keeps a running maximum ``m`` and a running sum ``l`` of exp(score - m),
    so the softmax needs one pass and the (T, L) score matrix is never stored. The query
    projection already contains the 1/sqrt(head_size) scale. The output goes straight to the
    (B * T, C) layout.
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // n_head
    h = pid_bh % n_head
    pos = tl.load(pos_ptr)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d[None, :] < HEAD_SIZE
    q_pos = pos + offs_q  # absolute positions of the queries
    head_base = arena_ptr + b.to(tl.int64) * stride_b + h * stride_h
    k_base = head_base + kv_slot.to(tl.int64) * stride_slot
    v_base = k_base + stride_slot
    # As in ``_matmul_tile``, tiles that stick out re-read the last valid row instead of using
    # masked loads. Such keys lie after every query, so the causal mask removes them, and such
    # queries are not stored.
    q_rows = (pos + tl.minimum(offs_q, T - 1)) * stride_l
    q = _load(head_base + q_rows[:, None] + offs_d[None, :], d_mask, MASK_D)

    m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_Q], dtype=tl.float32)  # noqa: E741
    acc = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
    # The last query of this tile sees keys up to its own position; later keys are skipped.
    n_keys = pos + tl.minimum((pid_q + 1) * BLOCK_Q, T)
    for start in range(0, n_keys, BLOCK_KV):
        offs_k = start + tl.arange(0, BLOCK_KV)
        kv_offs = (tl.minimum(offs_k, n_keys - 1) * stride_l)[:, None] + offs_d[None, :]
        k = _load(k_base + kv_offs, d_mask, MASK_D)
        v = _load(v_base + kv_offs, d_mask, MASK_D)
        s = tl.dot(q, tl.trans(k))
        s = tl.where(offs_k[None, :] <= q_pos[:, None], s, float("-inf"))  # causal mask
        # Online softmax: rescale what was accumulated under the old maximum, then add the tile.
        m_new = tl.maximum(m, tl.max(s, axis=1))
        rescale = tl.exp(m - m_new)
        p = tl.exp(s - m_new[:, None])
        l = l * rescale + tl.sum(p, axis=1)  # noqa: E741
        acc = acc * rescale[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m = m_new
    acc = acc / l[:, None]

    rows = (b * T + offs_q).to(tl.int64) * stride_ym
    y_ptrs = y_ptr + rows[:, None] + h * HEAD_SIZE + offs_d[None, :]
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=(offs_q[:, None] < T) & d_mask)


# --------------------------------------------------------------------------------------------
# Host side: tile sizes, weights, buffers and the launch sequence
# --------------------------------------------------------------------------------------------


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class _MatmulTiling:
    """Tile sizes and launch options of a matrix kernel."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _AttentionTiling:
    """Tile sizes and launch options of the attention kernel."""

    block_q: int
    block_kv: int
    num_warps: int
    num_stages: int


# Tile sizes were measured on an RTX PRO 4000 Blackwell (70 SMs) with GPT-2 small.
#
# Decode step (at most _DECODE_ROWS rows): all rows of the batch share one row tile, so a step
# reads each weight once however many sequences it decodes. That read bounds the speed (memory
# bandwidth, not arithmetic), so wide K tiles keep many loads in flight.
_DECODE_ROWS = 16
_DECODE_MATMUL = _MatmulTiling(16, 64, 256, num_warps=4, num_stages=3)
_DECODE_ATTENTION = _AttentionTiling(16, 64, num_warps=2, num_stages=3)
# Prompt pass: arithmetic bounds the speed. A larger tile reuses every loaded value in more
# products, but the kernel then has fewer programs, and it needs enough of them to keep every
# streaming multiprocessor (SM) busy. Entries are (tiling, minimum programs per SM), from the
# largest tile to the smallest. The largest one nearly fills the shared memory, where the tiles
# of the pipelined iterations wait.
_PROMPT_MATMULS = (
    (_MatmulTiling(256, 128, 64, num_warps=8, num_stages=3), 2.0),
    (_MatmulTiling(128, 128, 64, num_warps=4, num_stages=3), 0.5),
    (_MatmulTiling(64, 64, 64, num_warps=4, num_stages=3), 0.5),
)
_SMALL_PROMPT_MATMUL = _MatmulTiling(32, 64, 128, num_warps=4, num_stages=3)
_PROMPT_ATTENTION = _AttentionTiling(64, 64, num_warps=4, num_stages=3)
_GROUP_M = 8  # see _tile_ids


def _matmul_tiling(rows: int, n: int, sm_count: int) -> _MatmulTiling:
    """Pick the tiling of a matrix kernel that produces a (rows, n) output."""
    if rows <= _DECODE_ROWS:
        return _DECODE_MATMUL
    for tiling, programs_per_sm in _PROMPT_MATMULS:
        programs = triton.cdiv(rows, tiling.block_m) * triton.cdiv(n, tiling.block_n)
        # A row tile more than twice as tall as the input would be mostly padding.
        if tiling.block_m <= _next_power_of_two(rows) and programs >= programs_per_sm * sm_count:
            return tiling
    return _SMALL_PROMPT_MATMUL


@dataclass(frozen=True)
class _LayerWeights:
    """Weights of one block. Matrices are stored as (out, in)."""

    ln_1: tuple[torch.Tensor, torch.Tensor]
    qkv_w: torch.Tensor  # (3C, C), query rows pre-scaled
    qkv_b: torch.Tensor
    attn_proj_w: torch.Tensor  # (C, C)
    attn_proj_b: torch.Tensor
    ln_2: tuple[torch.Tensor, torch.Tensor]
    fc_w: torch.Tensor  # (4C, C)
    fc_b: torch.Tensor
    mlp_proj_w: torch.Tensor  # (C, 4C)
    mlp_proj_b: torch.Tensor


def _fold_attention_scale(
    weight: torch.Tensor, bias: torch.Tensor, n_embd: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply the query third of c_attn, given as (3C, C), by 1/sqrt(head_size).

    softmax((q / sqrt(hs)) k^T) equals softmax(q k^T / sqrt(hs)), so the kernel never scales the
    scores. GPT-2 uses head_size 64, so the factor is 1/8 and the scaled float16 values are exact.
    """
    scale = 1.0 / math.sqrt(head_size)
    weight, bias = weight.float(), bias.float()  # float32 copies
    weight[:n_embd] *= scale
    bias[:n_embd] *= scale
    return weight.to(DTYPE).contiguous(), bias.to(DTYPE).contiguous()


def _infer_config(sd: dict[str, torch.Tensor], n_head: int | None) -> GPT2Config:
    for key in ("wte.weight", "wpe.weight"):
        if key not in sd:
            raise ValueError(f"State dict has no {key!r}; is this a GPT-2 checkpoint?")
    vocab_size, n_embd = sd["wte.weight"].shape
    block_size = sd["wpe.weight"].shape[0]
    n_layer = len({key.split(".")[1] for key in sd if key.startswith("h.")})
    if n_head is None:
        # The head count leaves no trace in the tensor shapes.
        known = {
            c.n_head for c in GPT2_CONFIGS.values() if (c.n_layer, c.n_embd) == (n_layer, n_embd)
        }
        if len(known) != 1:
            raise ValueError(
                f"Cannot infer n_head for n_layer={n_layer}, n_embd={n_embd}; pass n_head"
            )
        n_head = known.pop()
    if n_embd % n_head != 0:
        raise ValueError(f"n_embd {n_embd} is not divisible by n_head {n_head}")
    return GPT2Config(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )


def _check_state_dict(sd: dict[str, torch.Tensor], config: GPT2Config) -> None:
    C = config.n_embd
    shapes: dict[str, tuple[int, ...]] = {
        "wte.weight": (config.vocab_size, C),
        "wpe.weight": (config.block_size, C),
        "ln_f.weight": (C,),
        "ln_f.bias": (C,),
    }
    for i in range(config.n_layer):
        shapes.update(
            {
                f"h.{i}.ln_1.weight": (C,),
                f"h.{i}.ln_1.bias": (C,),
                f"h.{i}.attn.c_attn.weight": (C, 3 * C),
                f"h.{i}.attn.c_attn.bias": (3 * C,),
                f"h.{i}.attn.c_proj.weight": (C, C),
                f"h.{i}.attn.c_proj.bias": (C,),
                f"h.{i}.ln_2.weight": (C,),
                f"h.{i}.ln_2.bias": (C,),
                f"h.{i}.mlp.c_fc.weight": (C, 4 * C),
                f"h.{i}.mlp.c_fc.bias": (4 * C,),
                f"h.{i}.mlp.c_proj.weight": (4 * C, C),
                f"h.{i}.mlp.c_proj.bias": (C,),
            }
        )

    missing = shapes.keys() - sd.keys()
    unexpected = sd.keys() - shapes.keys()
    if missing or unexpected:
        raise ValueError(
            f"State dict mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for key, shape in shapes.items():
        if tuple(sd[key].shape) != shape:
            raise ValueError(f"Shape mismatch for {key}: {tuple(sd[key].shape)} vs {shape}")


class _Workspace:
    """Every buffer one batch size needs, allocated once and reused by every pass.

    ``arena`` has shape (1 + 2 * n_layer, B, n_head, capacity, head_size): slot 0 holds the
    queries of the current pass, and each layer owns one slot for keys and one for values. These
    two slots are the KV cache. ``pos`` counts the cached positions and stays on the GPU, so a
    captured decode step reads the current value when it is replayed.
    """

    def __init__(self, config: GPT2Config, batch_size: int, capacity: int, device: torch.device):
        rows, C = batch_size * capacity, config.n_embd

        def buffer(*shape: int) -> torch.Tensor:
            return torch.empty(*shape, dtype=DTYPE, device=device)

        self.batch_size = batch_size
        self.capacity = capacity
        self.pos = torch.zeros(1, dtype=torch.int32, device=device)
        self.tokens = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.x = buffer(rows, C)  # residual stream
        self.xn = buffer(rows, C)  # layer norm output
        self.y = buffer(rows, C)  # attention output before c_proj
        self.h = buffer(rows, 4 * C)  # MLP hidden activations
        self.arena = buffer(
            1 + 2 * config.n_layer, batch_size, config.n_head, capacity, C // config.n_head
        )
        self.logits = torch.empty(batch_size, config.vocab_size, dtype=torch.float32, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None


class GPT2Inference:
    """GPT-2 forward pass and sampling loop built from Triton kernels, computed in float16.

    ``state_dict`` uses the Hugging Face layout: Conv1D weights of shape (in, out), with or
    without the ``transformer.`` prefix. Weights stored in another dtype are converted to
    float16. Logits are float32, because the kernels accumulate in float32 anyway.
    """

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        n_head: int | None = None,
        device: torch.device | str = "cuda",
        cuda_graph: bool | None = None,
    ):
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"Triton kernels need a GPU, got device {device}")
        sd = {
            key.removeprefix("transformer."): value
            for key, value in state_dict.items()
            if not key.endswith(_IGNORED_SUFFIXES)
        }
        config = _infer_config(sd, n_head)
        _check_state_dict(sd, config)

        def vector(key: str) -> torch.Tensor:
            return sd[key].to(device=device, dtype=DTYPE).contiguous()

        def matrix(key: str) -> torch.Tensor:
            # Hugging Face stores Conv1D weights as (in, out). The kernels want (out, in): each
            # program then reads whole contiguous rows instead of one short piece of every row.
            return sd[key].to(device=device, dtype=DTYPE).t().contiguous()

        C, head_size = config.n_embd, config.n_embd // config.n_head
        layers = []
        for i in range(config.n_layer):
            # Fold before the conversion to float16, so the stored values are rounded only once.
            qkv_w, qkv_b = _fold_attention_scale(
                sd[f"h.{i}.attn.c_attn.weight"].to(device).t(),
                sd[f"h.{i}.attn.c_attn.bias"].to(device),
                C,
                head_size,
            )
            layers.append(
                _LayerWeights(
                    ln_1=(vector(f"h.{i}.ln_1.weight"), vector(f"h.{i}.ln_1.bias")),
                    qkv_w=qkv_w,
                    qkv_b=qkv_b,
                    attn_proj_w=matrix(f"h.{i}.attn.c_proj.weight"),
                    attn_proj_b=vector(f"h.{i}.attn.c_proj.bias"),
                    ln_2=(vector(f"h.{i}.ln_2.weight"), vector(f"h.{i}.ln_2.bias")),
                    fc_w=matrix(f"h.{i}.mlp.c_fc.weight"),
                    fc_b=vector(f"h.{i}.mlp.c_fc.bias"),
                    mlp_proj_w=matrix(f"h.{i}.mlp.c_proj.weight"),
                    mlp_proj_b=vector(f"h.{i}.mlp.c_proj.bias"),
                )
            )

        self.config = config
        self.device = device
        self.head_size = head_size
        # CUDA graphs are an NVIDIA feature; ROCm builds of PyTorch report no CUDA version.
        self.cuda_graph = torch.version.cuda is not None if cuda_graph is None else cuda_graph
        self.wte = vector("wte.weight")  # (vocab_size, C); already (out, in) for the LM head
        self.wpe = vector("wpe.weight")  # (block_size, C)
        self.ln_f = (vector("ln_f.weight"), vector("ln_f.bias"))
        self.layers = layers
        self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self._workspace_cache: _Workspace | None = None

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        n_head: int | None = None,
        device: torch.device | str = "cuda",
        cuda_graph: bool | None = None,
    ) -> GPT2Inference:
        """Load weights saved from a Hugging Face ``GPT2LMHeadModel`` state dict.

        ``n_head`` is needed only for sizes other than the four published GPT-2 models.
        """
        state_dict = load_file(path, device=str(torch.device(device)))
        return cls(state_dict, n_head=n_head, device=device, cuda_graph=cuda_graph)

    # ---------------------------------------------------------------------------------------
    # Kernel launches
    # ---------------------------------------------------------------------------------------

    def _embed(self, ws: _Workspace, idx: torch.Tensor, T: int) -> None:
        C = self.config.n_embd
        _embed_kernel[(idx.numel(),)](
            idx, self.wte, self.wpe, ws.x, ws.pos, T, C,
            BLOCK_C=_next_power_of_two(C), enable_fp_fusion=True,
        )  # fmt: skip

    def _normed(
        self, ws: _Workspace, ln: tuple[torch.Tensor, torch.Tensor], rows: int
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Return the input of the next matrix kernel and the layer norm it must still apply."""
        if rows <= _DECODE_ROWS:
            return ws.x, ln  # the matrix kernel normalizes its own rows
        C = self.config.n_embd
        _layernorm_kernel[(rows,)](
            ws.x, ln[0], ln[1], ws.xn, C, LAYER_NORM_EPS,
            BLOCK_C=_next_power_of_two(C), enable_fp_fusion=True,
        )  # fmt: skip
        return ws.xn, None

    def _linear(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        bias: torch.Tensor | None,
        out: torch.Tensor,
        rows: int,
        *,
        a_stride: int | None = None,
        ln: tuple[torch.Tensor, torch.Tensor] | None = None,
        gelu: bool = False,
        residual: bool = False,
    ) -> None:
        N, K = w.shape
        tiling = _matmul_tiling(rows, N, self._sm_count)
        ln_w, ln_b = ln if ln is not None else (w, w)  # unused pointers still need a value
        grid = (triton.cdiv(rows, tiling.block_m) * triton.cdiv(N, tiling.block_n),)
        _linear_kernel[grid](
            a, w, bias if bias is not None else w, out, ln_w, ln_b,
            rows, N, K, a_stride or K, N, LAYER_NORM_EPS,
            BLOCK_M=tiling.block_m, BLOCK_N=tiling.block_n, BLOCK_K=tiling.block_k,
            GROUP_M=_GROUP_M, MASK_K=K % tiling.block_k != 0,
            FUSE_LN=ln is not None, HAS_BIAS=bias is not None, GELU=gelu, RESIDUAL=residual,
            num_warps=tiling.num_warps, num_stages=tiling.num_stages, enable_fp_fusion=True,
        )  # fmt: skip

    def _qkv(
        self,
        ws: _Workspace,
        a: torch.Tensor,
        ln: tuple[torch.Tensor, torch.Tensor] | None,
        layer: _LayerWeights,
        kv_slot: int,
        rows: int,
        T: int,
    ) -> None:
        C = self.config.n_embd
        tiling = _matmul_tiling(rows, 3 * C, self._sm_count)
        ln_w, ln_b = ln if ln is not None else (a, a)
        grid = (triton.cdiv(rows, tiling.block_m) * triton.cdiv(3 * C, tiling.block_n),)
        _qkv_kernel[grid](
            a, layer.qkv_w, layer.qkv_b, ws.arena, ws.pos, ln_w, ln_b,
            rows, C, T, kv_slot, C, *ws.arena.stride()[:4], LAYER_NORM_EPS,
            HEAD_SIZE=self.head_size,
            BLOCK_M=tiling.block_m, BLOCK_N=tiling.block_n, BLOCK_K=tiling.block_k,
            GROUP_M=_GROUP_M, MASK_K=C % tiling.block_k != 0, FUSE_LN=ln is not None,
            num_warps=tiling.num_warps, num_stages=tiling.num_stages, enable_fp_fusion=True,
        )  # fmt: skip

    def _attention(self, ws: _Workspace, kv_slot: int, T: int) -> None:
        n_head = self.config.n_head
        tiling = _DECODE_ATTENTION if T <= _DECODE_ROWS else _PROMPT_ATTENTION
        grid = (triton.cdiv(T, tiling.block_q), ws.batch_size * n_head)
        # ``tl.dot`` needs power-of-two tiles of at least 16, so other head sizes are padded.
        block_d = max(16, _next_power_of_two(self.head_size))
        _attention_kernel[grid](
            ws.arena, ws.y, ws.pos, T, n_head, kv_slot, *ws.arena.stride()[:4], self.config.n_embd,
            HEAD_SIZE=self.head_size, BLOCK_D=block_d,
            BLOCK_Q=tiling.block_q, BLOCK_KV=tiling.block_kv, MASK_D=block_d != self.head_size,
            num_warps=tiling.num_warps, num_stages=tiling.num_stages, enable_fp_fusion=True,
        )  # fmt: skip

    def _pass(
        self, ws: _Workspace, idx: torch.Tensor, T: int, logits: torch.Tensor, *, last_only: bool
    ) -> None:
        """Run ``idx`` of shape (B * T,) through the model, after the ``ws.pos`` cached tokens.

        Keys and values of the new tokens are appended to the cache. ``logits`` receives one row
        per sequence if ``last_only``, else one row per token.
        """
        B, C = ws.batch_size, self.config.n_embd
        rows = B * T
        self._embed(ws, idx, T)
        for i, layer in enumerate(self.layers):
            kv_slot = 1 + 2 * i
            a, ln = self._normed(ws, layer.ln_1, rows)
            self._qkv(ws, a, ln, layer, kv_slot, rows, T)
            self._attention(ws, kv_slot, T)
            self._linear(ws.y, layer.attn_proj_w, layer.attn_proj_b, ws.x, rows, residual=True)
            a, ln = self._normed(ws, layer.ln_2, rows)
            self._linear(a, layer.fc_w, layer.fc_b, ws.h, rows, ln=ln, gelu=True)
            self._linear(ws.h, layer.mlp_proj_w, layer.mlp_proj_b, ws.x, rows, residual=True)
        if last_only:
            # Row T - 1 of every sequence: start at that row and step T rows at a time.
            self._linear(ws.x[T - 1 :], self.wte, None, logits, B, a_stride=T * C, ln=self.ln_f)
        else:
            a, ln = self._normed(ws, self.ln_f, rows)
            self._linear(a, self.wte, None, logits, rows, ln=ln)

    # ---------------------------------------------------------------------------------------
    # Prompt pass and decode step
    # ---------------------------------------------------------------------------------------

    def _workspace(self, batch_size: int, length: int) -> _Workspace:
        ws = self._workspace_cache
        if ws is None or ws.batch_size != batch_size or ws.capacity < length:
            self._workspace_cache = ws = None  # release the old buffers first
            # Round up so that slightly longer requests reuse the buffers and the captured graph.
            capacity = min(self.config.block_size, max(64, _next_power_of_two(length)))
            ws = self._workspace_cache = _Workspace(self.config, batch_size, capacity, self.device)
        return ws

    @staticmethod
    def _flat(idx: torch.Tensor) -> torch.Tensor:
        """Return ``idx`` as one contiguous row of tokens, which is what the kernels index.

        ``reshape(-1)`` is not enough: a column slice such as ``out[:, :1]`` flattens to a
        strided view, and a kernel would then read the tokens between the wanted ones.
        """
        return idx.contiguous().view(-1)

    def _prefill(self, ws: _Workspace, idx: torch.Tensor) -> torch.Tensor:
        """Restart the cache with ``idx`` of shape (B, T); return the last-token logits."""
        T = idx.size(1)
        ws.pos.zero_()
        self._pass(ws, self._flat(idx), T, ws.logits, last_only=True)
        ws.pos.fill_(T)
        return ws.logits

    def _decode_step(self, ws: _Workspace) -> None:
        self._pass(ws, ws.tokens, 1, ws.logits, last_only=True)
        ws.pos += 1

    def _decode(self, ws: _Workspace, tokens: torch.Tensor) -> torch.Tensor:
        """Append one token per sequence to the cache; return the logits of the next token."""
        ws.tokens.copy_(tokens)
        if ws.graph is not None:
            ws.graph.replay()
            return ws.logits
        # The first step runs eagerly, which also compiles the kernels.
        self._decode_step(ws)
        if self.cuda_graph:
            # Record the same launches for later steps. Capturing does not run them. Buffer
            # addresses and tile counts are fixed, and the position is read from ``ws.pos``.
            ws.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(ws.graph):
                self._decode_step(ws)
        return ws.logits

    def _check_tokens(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.dim() != 2 or idx.size(0) == 0 or idx.size(1) == 0:
            raise ValueError(f"Expected tokens of shape (B, T) with B, T > 0, got {idx.shape}")
        idx = idx.to(device=self.device, dtype=torch.long)
        # The kernels index the embedding with raw pointers, so a bad id would read out of bounds.
        if idx.min() < 0 or idx.max() >= self.config.vocab_size:
            raise ValueError(f"Token ids must be in [0, {self.config.vocab_size})")
        return idx

    @torch.no_grad()
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Return float32 logits of shape (B, T, vocab_size) for ``idx`` of shape (B, T)."""
        idx = self._check_tokens(idx)
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"
            )
        ws = self._workspace(B, T)
        logits = torch.empty(B * T, self.config.vocab_size, dtype=torch.float32, device=self.device)
        ws.pos.zero_()
        self._pass(ws, self._flat(idx), T, logits, last_only=False)
        return logits.view(B, T, -1)

    __call__ = forward

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Extend ``idx`` of shape (B, T) by ``max_new_tokens`` sampled tokens.

        Sampling follows ``GPT2.generate``: ``temperature=0`` decodes greedily. The prompt runs
        once to fill the KV cache, then each step feeds only the newest token. Once the sequence
        outgrows ``block_size``, cached positions would shift, so every step recomputes the
        cropped context.
        """
        idx = self._check_tokens(idx)
        B, T = idx.shape
        block_size = self.config.block_size
        ws = self._workspace(B, min(T + max_new_tokens, block_size))
        out = torch.empty(B, T + max_new_tokens, dtype=torch.long, device=self.device)
        out[:, :T] = idx
        cached = False
        for length in range(T, T + max_new_tokens):
            if length > block_size:
                logits = self._prefill(ws, out[:, length - block_size : length])
            elif not cached:
                logits = self._prefill(ws, out[:, :length])
                cached = True
            else:
                logits = self._decode(ws, out[:, length - 1])
            out[:, length] = _sample(logits, temperature, top_k)
        return out


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    """Pick one token per row of ``logits`` (B, vocab_size); return shape (B,)."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)
