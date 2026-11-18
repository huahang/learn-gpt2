"""GPT-2 inference whose operators are written in the CuTe DSL (the CUTLASS 4 Python DSL).

``gpt2.py`` is the reference: it builds the model from PyTorch modules. This file computes the
same function in float16 with hand-written GPU kernels and loads the weights from a safetensors
file. Only token sampling still uses PyTorch operators.

Every matrix product runs on tensor cores as a GEMM: the four linear layers of each block, the LM
head, and both products inside attention (Q K^T and P V). The kernels target the warp-level
``mma.sync`` instruction (SM80 and newer, including the SM120 Blackwell GPUs that have no
``tcgen05``), with float16 inputs and float32 accumulators.

Optimizations, and where each one lives:

- Tensor-core GEMM (``_GemmKernel``): a CTA computes one (BM, BN) output tile. ``cp.async`` copies
  (BM, BK) and (BN, BK) tiles from global to shared memory through a multi-stage ring, so several
  tiles are in flight while tensor cores work on an older one. Shared memory tiles use an XOR
  swizzle, so neither the 128-bit writes nor the ``ldmatrix`` reads hit bank conflicts.
  ``ldmatrix`` moves operands from shared memory to registers, double buffered against ``mma``.
- Fused epilogues: the accumulator tile is staged in shared memory and leaves the CTA in
  contiguous vectors. On the way out the GEMM applies the folded layer norm, adds the bias,
  applies GELU, adds the residual, or scatters query/key/value into the KV cache. Each element is
  rounded to float16 once. Per-row and per-column operands are loaded once per tile, and GELU
  uses ``ex2.approx`` and ``rcp.approx`` instead of a division.
- Layer norm folding: ``ln_1`` and ``ln_2`` never write a normalized copy of the activations.
  Their gain is multiplied into the next weight matrix and their shift into its bias when the
  weights load, and the GEMM epilogue applies the row statistics:
  ``LN(x) @ W = rstd * (x @ W' - mean * colsum(W')) + b'``. In the decode step the GEMM computes
  the statistics itself, from the A fragments it already holds in registers, so the layer norm
  costs no kernel at all. The prompt pass computes them once per row in a small kernel instead,
  because there every column tile would repeat the work.
- Constant folding: the attention scale 1/sqrt(head_size) is folded into the query weights.
- Flash attention (``_AttentionKernel``): scores, causal mask, online softmax and the weighted sum
  of values in one pass, without storing the (T, L) score matrix. Q stays in registers. The
  softmax output P is converted in registers from the accumulator layout to the A operand layout
  of the next ``mma``, so it never touches shared memory. K and V tiles load while the other is
  being multiplied.
- Split-KV decoding (flash decoding): a decode step has one query per sequence, so one CTA per
  head would leave most of the GPU idle. The keys are split across CTAs, and
  ``_AttentionCombineKernel`` merges the partial results by their log-sum-exp.
- Layouts that need no transposes: the KV cache is stored as (batch, position, n_embd), exactly
  the layout the QKV GEMM produces, and attention reads and writes the (rows, n_embd) layout of
  the GEMMs.
- Static memory plan and padding: ``_Workspace`` allocates every buffer once, with rows padded to
  whole tiles and the vocabulary padded to a multiple of 128. Loads therefore need no bounds
  checks; only stores are predicated.
- Last-token logits: generation normalizes and projects only the last position of each sequence.
- KV cache: decoding feeds one token per step and attends to stored keys and values.
- Programmatic dependent launch (PDL): a kernel may start while the previous one finishes. GEMMs
  prefetch their first weight tiles before ``griddepcontrol.wait``, because weights do not depend
  on the previous kernel.
- CUDA graph: one decode step is captured once and replayed. The cache length lives in GPU memory,
  so the same graph serves every step and Python launches nothing per kernel.
- Low launch overhead: kernels are compiled once per shape with dynamic row counts, and are
  called through TVM-FFI on the current PyTorch stream.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cuda.bindings.driver import CUstream
from cutlass import Float16, Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.runtime import make_fake_stream
from safetensors.torch import load_file

from learn_gpt2.gpt2 import GPT2_CONFIGS, GPT2Config

LAYER_NORM_EPS = 1e-5
LOG2_E = 1.4426950408889634
LN_2 = 0.6931471805599453
SQRT_2_OVER_PI = 0.7978845608028654

_LAYER_KEYS = (
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
)
# Legacy causal-mask buffers and the tied LM head carry no information of their own.
_IGNORED_SUFFIXES = (".attn.bias", ".attn.masked_bias", "lm_head.weight")

# Every activation buffer has this many spare rows, so that a tile that starts at the last valid
# row still reads allocated memory.
_ROW_SLACK = 128
# The vocabulary is padded to a multiple of this, so that the LM head needs no partial tiles.
_VOCAB_MULTIPLE = 128
# Each 128-bit copy moves 8 float16 values.
_VEC16 = 8


def _round_up(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fake(dtype: type, shape: tuple) -> object:
    """A placeholder for a row-major tensor argument of ``cute.compile``."""
    order = tuple(range(len(shape) - 1, -1, -1))
    return make_fake_compact_tensor(dtype, shape, stride_order=order, assumed_align=16)


def _compile(kernel: object, *args: object) -> object:
    # TVM-FFI accepts torch tensors directly and launches on the current torch stream, which
    # keeps the Python cost of a launch to a few microseconds and works under graph capture.
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, *args, stream, options="--enable-tvm-ffi")


@cute.jit
def _quad_sum(value: Float32) -> Float32:
    """Sum over the 4 lanes of a quad. An ``mma`` row lives in the 4 lanes of one quad."""
    value = value + cute.arch.shuffle_sync_bfly(value, offset=1)
    return value + cute.arch.shuffle_sync_bfly(value, offset=2)


@cute.jit
def _quad_max(value: Float32) -> Float32:
    value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=1))
    return cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, offset=2))


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for offset in cutlass.range_constexpr(5):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=16 >> offset)
    return value


def _swizzled_smem_layout(rows: int, cols: int, stages: int | None = None) -> object:
    """Shared memory layout of a K-major (rows, cols[, stages]) float16 tile.

    The atom is 8 rows of up to 64 values (128 bytes). ``Swizzle(b, 3, 3)`` XORs the 16-byte chunk
    index with the row index, so the 8 rows of an ``ldmatrix`` 8x8 matrix fall into different
    banks, and so do the 128-bit ``cp.async`` writes of one row.
    """
    major = min(cols, 64)
    bits = min(int(math.log2(major * 2 // 16)), 3)
    atom = cute.make_composed_layout(
        cute.make_swizzle(bits, 3, 3), 0, cute.make_layout((8, major), stride=(major, 1))
    )
    if stages is None:
        return cute.tile_to_shape(atom, (rows, cols), (0, 1))
    return cute.tile_to_shape(atom, (rows, cols, stages), (0, 1, 2))


def _g2s_tiled_copy(threads: int, cols: int) -> object:
    """``cp.async`` copy of a K-major tile: each thread moves 128 bits per instruction.

    ``LoadCacheMode.GLOBAL`` (``cp.async.cg``) bypasses L1, since every tile is read once per CTA.
    """
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        Float16,
        num_bits_per_copy=128,
    )
    per_row = cols // _VEC16
    threads_layout = cute.make_layout((threads // per_row, per_row), stride=(per_row, 1))
    return cute.make_tiled_copy_tv(atom, threads_layout, cute.make_layout((1, _VEC16)))


def _warp_mma(warps_m: int, warps_n: int) -> object:
    """Tensor-core MMA: m16n8k16, float16 inputs, float32 accumulators.

    Warps are laid out (warps_m, warps_n). The N permutation lets each warp own 16 columns per
    step instead of 8, so one ``ldmatrix.x4`` feeds two ``mma`` instructions.
    """
    return cute.make_tiled_mma(
        warp.MmaF16BF16Op(Float16, Float32, (16, 8, 16)),
        cute.make_layout((warps_m, warps_n, 1)),
        permutation_mnk=(warps_m * 16, warps_n * 16, 16),
    )


def _ldmatrix(transpose: bool) -> object:
    return cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose, 4), Float16)


# --------------------------------------------------------------------------------------------
# GEMM
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _GemmTile:
    """CTA tile (bm, bn, bk), pipeline depth, and warp layout of a GEMM."""

    bm: int
    bn: int
    bk: int
    stages: int
    warps_m: int
    warps_n: int

    @property
    def threads(self) -> int:
        return self.warps_m * self.warps_n * 32

    def fits(self, n: int, k: int) -> bool:
        rows_per_copy = self.threads // (self.bk // _VEC16)
        rows_per_store = self.threads // (self.bn // _EPILOGUE_VEC)
        return (
            n % self.bn == 0
            and k % self.bk == 0
            and self.bm % (self.warps_m * 16) == 0
            and self.bn % (self.warps_n * 16) == 0
            and self.bm % rows_per_copy == 0
            and self.bn % rows_per_copy == 0
            and self.threads % (self.bn // _EPILOGUE_VEC) == 0
            and self.bm % rows_per_store == 0
            and self.bm <= self.threads
        )


@dataclass(frozen=True)
class _Epilogue:
    """What the GEMM does to its float32 accumulator before the tile leaves the CTA.

    In order: ``ln_fold`` applies the folded layer norm, ``bias`` adds the bias, ``gelu`` applies
    GELU, ``residual`` adds the output buffer to the result (in place), and ``qkv`` scatters the
    columns into the query buffer and the KV cache. ``out_fp32`` writes float32 instead of float16.

    The folded layer norm needs the mean and 1/std of every row. With ``own_stats`` the GEMM
    computes them from its A fragments. That is free while memory bandwidth bounds the GEMM (the
    decode step), but every column tile repeats the work, so the prompt pass reads them from a
    buffer that ``_LayerNormKernel(stats_only=True)`` fills once.
    """

    ln_fold: bool = False
    own_stats: bool = False
    bias: bool = False
    gelu: bool = False
    residual: bool = False
    qkv: bool = False
    out_fp32: bool = False


# The float32 accumulator tile is staged in shared memory with rows padded by 8 values. A row
# then starts 8 banks after the previous one, so the 64-bit fragment writes of 8 rows hit 32
# different banks.
_EPILOGUE_PAD = 8
# The epilogue moves 4 float32 values (128 bits of shared memory) per thread and step.
_EPILOGUE_VEC = 4


class _GemmKernel:
    """``out[m, n] = epilogue(sum_k a[m, k] * w[n, k])`` on tensor cores.

    ``a`` is (rows, K) and ``w`` is (N, K), both K-major. Only the first ``m`` rows of ``a`` are
    valid, but every tile may read up to ``_ROW_SLACK`` rows past them.
    """

    def __init__(self, n: int, k: int, tile: _GemmTile, epilogue: _Epilogue, *, n_embd: int, pdl):
        assert tile.fits(n, k), (n, k, tile)
        assert epilogue.ln_fold or not epilogue.own_stats
        if epilogue.qkv:
            assert n == 3 * n_embd and n_embd % tile.bn == 0
        self.n, self.k, self.tile, self.epi = n, k, tile, epilogue
        self.n_embd = n_embd
        self.pdl = pdl
        self.k_tiles = k // tile.bk
        # The ring holds ``stages`` tiles; ``stages - 1`` are loaded before the main loop.
        self.stages = min(tile.stages, self.k_tiles + 1)
        self.out_dtype = Float32 if epilogue.out_fp32 else Float16

    def compile(self) -> object:
        rows, rows_out, rows_kv = cute.sym_int32(), cute.sym_int32(), cute.sym_int32()
        out_cols = self.n_embd if self.epi.qkv else self.n
        kv_cols = self.n_embd if self.epi.qkv else _VEC16
        return _compile(
            self,
            _fake(Float16, (rows, self.k)),
            _fake(Float16, (self.n, self.k)),
            _fake(Float32, (self.n,)),
            _fake(Float32, (self.n,)),
            _fake(self.out_dtype, (rows_out, out_cols)),
            _fake(Float16, (2, rows_kv, kv_cols)),
            _fake(Float32, (cute.sym_int32(), 2)),
            _fake(Int32, (1,)),
            Int32(1),
            Int32(1),
            Int32(1),
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mW: cute.Tensor,
        mBias: cute.Tensor,
        mColSum: cute.Tensor,
        mOut: cute.Tensor,
        mKV: cute.Tensor,
        mStats: cute.Tensor,
        mPos: cute.Tensor,
        m: Int32,
        seq_len: Int32,
        cap: Int32,
        stream: CUstream,
    ):
        t = self.tile
        tiled_mma = _warp_mma(t.warps_m, t.warps_n)
        sA_layout = _swizzled_smem_layout(t.bm, t.bk, self.stages)
        sW_layout = _swizzled_smem_layout(t.bn, t.bk, self.stages)
        sC_layout = cute.make_layout((t.bm, t.bn), stride=(t.bn + _EPILOGUE_PAD, 1))
        copy_g2s = _g2s_tiled_copy(t.threads, t.bk)
        # blockIdx.x walks the row tiles fastest. CTAs that run together therefore share one
        # column tile of the weight, which is read from DRAM once and then hits in L2.
        grid = (cute.ceil_div(m, t.bm), self.n // t.bn, 1)
        self.kernel(
            mA, mW, mBias, mColSum, mOut, mKV, mStats, mPos, m, seq_len, cap,
            sA_layout, sW_layout, sC_layout, copy_g2s, tiled_mma,
        ).launch(grid=grid, block=[t.threads, 1, 1], stream=stream, use_pdl=self.pdl)  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mW: cute.Tensor,
        mBias: cute.Tensor,
        mColSum: cute.Tensor,
        mOut: cute.Tensor,
        mKV: cute.Tensor,
        mStats: cute.Tensor,
        mPos: cute.Tensor,
        m: Int32,
        seq_len: Int32,
        cap: Int32,
        sA_layout: cute.ComposedLayout,
        sW_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        copy_g2s: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        t = self.tile
        bm, bn, bk, stages = t.bm, t.bn, t.bk, self.stages
        k_tiles = self.k_tiles
        tidx, _, _ = cute.arch.thread_idx()
        tile_m, tile_n, _ = cute.arch.block_idx()

        gA = cute.local_tile(mA, (bm, bk), (tile_m, None))  # (bm, bk, k_tiles)
        gW = cute.local_tile(mW, (bn, bk), (tile_n, None))  # (bn, bk, k_tiles)

        # One shared memory buffer serves the main loop (A and W rings) and then the epilogue.
        smem = cutlass.utils.SmemAllocator()
        main_elems = cute.cosize(sA_layout) + cute.cosize(sW_layout)
        epi_bytes = bm * (bn + _EPILOGUE_PAD) * 4
        buf = smem.allocate(max(main_elems * 2, epi_bytes), byte_alignment=1024)
        stats = smem.allocate_tensor(Float32, cute.make_layout((bm, 2), stride=(2, 1)), 16)
        row_map = smem.allocate_tensor(Int32, cute.make_layout(bm), 16)
        half_ptr = cute.recast_ptr(buf, dtype=Float16)
        sA = cute.make_tensor(half_ptr, sA_layout)
        sW = cute.make_tensor(half_ptr + cute.cosize(sA_layout), sW_layout)
        sC = cute.make_tensor(cute.recast_ptr(buf, dtype=Float32), sC_layout)

        thr_g2s = copy_g2s.get_slice(tidx)
        tAgA = thr_g2s.partition_S(gA)  # (CPY, CPY_M, CPY_K, k_tiles)
        tAsA = thr_g2s.partition_D(sA)  # (CPY, CPY_M, CPY_K, stages)
        tWgW = thr_g2s.partition_S(gW)
        tWsW = thr_g2s.partition_D(sW)

        # Prologue. The weights do not depend on earlier kernels, so their first tiles are
        # requested before waiting for the previous kernel (PDL). They form one extra commit
        # group, older than all others, so the group counting below still holds.
        for s in cutlass.range_constexpr(stages - 1):
            cute.copy(copy_g2s, tWgW[None, None, None, s], tWsW[None, None, None, s])
        cute.arch.cp_async_commit_group()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        for s in cutlass.range_constexpr(stages - 1):
            cute.copy(copy_g2s, tAgA[None, None, None, s], tAsA[None, None, None, s])
            cute.arch.cp_async_commit_group()

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsW = thr_mma.partition_B(sW)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])  # (MMA, MMA_M, MMA_K)
        tCrW = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])  # (MMA, MMA_N, MMA_K)
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((bm, bn)), Float32)
        acc.fill(0.0)
        # Per-thread partial sums of x and x^2 for the folded layer norm, shaped like an A
        # fragment of one k-block.
        row_sum = cute.make_rmem_tensor(tCrA[None, None, 0].layout, Float32)
        row_sq = cute.make_rmem_tensor(tCrA[None, None, 0].layout, Float32)
        row_sum.fill(0.0)
        row_sq.fill(0.0)

        s2r_a = cute.make_tiled_copy_A(_ldmatrix(False), tiled_mma).get_slice(tidx)
        s2r_w = cute.make_tiled_copy_B(_ldmatrix(False), tiled_mma).get_slice(tidx)
        tCsA_view = s2r_a.partition_S(sA)
        tCrA_view = s2r_a.retile(tCrA)
        tCsW_view = s2r_w.partition_S(sW)
        tCrW_view = s2r_w.retile(tCrW)

        read = Int32(0)
        write = Int32(stages - 1)
        tCsA_p = tCsA_view[None, None, None, read]
        tCsW_p = tCsW_view[None, None, None, read]
        num_kb = cute.size(tCrA, mode=[2])
        cute.arch.cp_async_wait_group(stages - 2)
        cute.arch.sync_threads()
        cute.copy(s2r_a, tCsA_p[None, None, 0], tCrA_view[None, None, 0])
        cute.copy(s2r_w, tCsW_p[None, None, 0], tCrW_view[None, None, 0])

        # Main loop. Two pipelines overlap:
        # - shared memory: tile k + stages - 1 is requested while tile k is multiplied;
        # - registers: k-block kb + 1 is loaded with ldmatrix while k-block kb is multiplied.
        for k_tile in range(k_tiles):
            for kb in cutlass.range_constexpr(num_kb):
                if kb == num_kb - 1:
                    # Move to the next stage of the ring once it has landed.
                    tCsA_p = tCsA_view[None, None, None, read]
                    tCsW_p = tCsW_view[None, None, None, read]
                    cute.arch.cp_async_wait_group(stages - 2)
                    cute.arch.sync_threads()
                kb_next = (kb + 1) % num_kb
                cute.copy(s2r_a, tCsA_p[None, None, kb_next], tCrA_view[None, None, kb_next])
                cute.copy(s2r_w, tCsW_p[None, None, kb_next], tCrW_view[None, None, kb_next])
                # A is requested before the MMA and W after it, so the copies interleave with
                # tensor-core work. ``kb == 0`` is decided while tracing, the tile check on the GPU.
                if kb == 0:  # noqa: SIM102
                    if k_tile + stages - 1 < k_tiles:
                        next_tile = k_tile + stages - 1
                        cute.copy(
                            copy_g2s,
                            tAgA[None, None, None, next_tile],
                            tAsA[None, None, None, write],
                        )
                cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrW[None, None, kb], acc)
                if cutlass.const_expr(self.epi.own_stats):
                    a = tCrA[None, None, kb].load().to(Float32)
                    row_sum.store(row_sum.load() + a)
                    row_sq.store(row_sq.load() + a * a)
                if kb == 0:
                    if k_tile + stages - 1 < k_tiles:
                        next_tile = k_tile + stages - 1
                        cute.copy(
                            copy_g2s,
                            tWgW[None, None, None, next_tile],
                            tWsW[None, None, None, write],
                        )
                    # Commit even when nothing was copied, so the group count stays regular.
                    cute.arch.cp_async_commit_group()
                    write = read
                    read = read + 1
                    if read == stages:
                        read = Int32(0)

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if cutlass.const_expr(self.epi.own_stats):
            self._row_statistics(row_sum, row_sq, stats, thr_mma, tidx)

        # Stage the float32 accumulators in shared memory (the ring is no longer needed).
        cute.autovec_copy(acc, thr_mma.partition_C(sC))
        # Per-row values of the epilogue, prepared once per row instead of once per chunk.
        if tidx < bm:
            row = tile_m * bm + tidx
            if cutlass.const_expr(self.epi.ln_fold and not self.epi.own_stats):
                # Rows past m read the zeroed padding of the buffer; their results are dropped.
                stats[tidx, 0] = mStats[row, 0]
                stats[tidx, 1] = mStats[row, 1]
            if cutlass.const_expr(self.epi.qkv):
                # Token t of sequence b lands at cache position pos + t.
                b = row // seq_len
                row_map[tidx] = b * cap + mPos[0] + row - b * seq_len
        cute.arch.sync_threads()
        self._store_tile(sC, stats, row_map, mBias, mColSum, mOut, mKV, m, tidx)

    @cute.jit
    def _row_statistics(
        self,
        row_sum: cute.Tensor,
        row_sq: cute.Tensor,
        stats: cute.Tensor,
        thr_mma: cute.ThrMma,
        tidx: Int32,
    ):
        """Turn per-thread partial sums into the mean and 1/std of every row of the tile.

        An m16n8k16 A fragment holds 8 values per thread, indexed (col pair, row half, col half):
        rows ``lane / 4`` and ``lane / 4 + 8`` of the 16-row atom. The 4 lanes of a quad hold
        different columns of the same rows, so a quad reduction completes the row sums. Warps
        that share rows compute the same numbers and write the same values.
        """
        coords = thr_mma.partition_A(cute.make_identity_tensor((self.tile.bm, self.tile.bk)))
        for mi in cutlass.range_constexpr(cute.size(row_sum, mode=[1])):
            for half in cutlass.range_constexpr(2):
                total = Float32(0.0)
                total_sq = Float32(0.0)
                for pair in cutlass.range_constexpr(2):
                    for col_half in cutlass.range_constexpr(2):
                        total = total + row_sum[(pair, half, col_half), mi]
                        total_sq = total_sq + row_sq[(pair, half, col_half), mi]
                mean = _quad_sum(total) / self.k
                var = _quad_sum(total_sq) / self.k - mean * mean
                rstd = cute.math.rsqrt(cutlass.max(var, 0.0) + LAYER_NORM_EPS)
                row = coords[(0, half, 0), mi, 0][0]
                if tidx % 4 == 0:
                    stats[row, 0] = mean
                    stats[row, 1] = rstd
        cute.arch.sync_threads()

    @cute.jit
    def _store_tile(
        self,
        sC: cute.Tensor,
        stats: cute.Tensor,
        sRowMap: cute.Tensor,
        mBias: cute.Tensor,
        mColSum: cute.Tensor,
        mOut: cute.Tensor,
        mKV: cute.Tensor,
        m: Int32,
        tidx: Int32,
    ):
        """Apply the epilogue to the staged tile and write it out, 4 values per thread and step.

        Consecutive threads take consecutive 4-value chunks of a row, so both the shared memory
        reads and the global stores of a warp are contiguous. A thread keeps the same columns in
        every step, so it loads the column vectors (bias, column sums) once.
        """
        t, epi, vec = self.tile, self.epi, _EPILOGUE_VEC
        tile_m, tile_n, _ = cute.arch.block_idx()
        chunks_per_row = t.bn // vec
        rows_per_step = t.threads // chunks_per_row
        c = tidx % chunks_per_row
        col_chunk = tile_n * chunks_per_row + c
        if cutlass.const_expr(epi.ln_fold):
            col_sum = cute.local_tile(mColSum, (vec,), (col_chunk,)).load()
        if cutlass.const_expr(epi.bias):
            bias = cute.local_tile(mBias, (vec,), (col_chunk,)).load()
        if cutlass.const_expr(epi.qkv):
            # A tile lies inside one of query, key or value, because n_embd % bn == 0.
            part = tile_n * t.bn // self.n_embd
            dst_chunk = col_chunk - part * (self.n_embd // vec)
        for step in cutlass.range_constexpr(t.bm // rows_per_step):
            r = tidx // chunks_per_row + step * rows_per_step
            row = tile_m * t.bm + r
            if row < m:
                v = cute.local_tile(sC[r, None], (vec,), (c,)).load()
                if cutlass.const_expr(epi.ln_fold):
                    v = (v - col_sum * stats[r, 0]) * stats[r, 1]
                if cutlass.const_expr(epi.bias):
                    v = v + bias
                if cutlass.const_expr(epi.gelu):
                    # tanh approximation. 0.5 * (1 + tanh(z)) == 1 / (1 + exp(-2 z)), so the
                    # activation costs one exp2 and one approximate reciprocal per value.
                    z = v * (v * v * 0.044715 + 1.0) * (-2.0 * SQRT_2_OVER_PI * LOG2_E)
                    v = v * cute.math.rcp(
                        cute.math.exp2(z, fastmath=True) + 1.0, approx=True, ftz=True
                    )
                if cutlass.const_expr(epi.qkv):
                    # Both destinations are built outside the branch: views compute addresses,
                    # they access nothing.
                    kv = mKV[cutlass.max(part - 1, 0), sRowMap[r], None]
                    dst_kv = cute.local_tile(kv, (vec,), (dst_chunk,))
                    dst_q = cute.local_tile(mOut[row, None], (vec,), (dst_chunk,))
                    if part == 0:
                        dst_q.store(v.to(Float16))
                    else:
                        dst_kv.store(v.to(Float16))
                else:
                    dst = cute.local_tile(mOut[row, None], (vec,), (col_chunk,))
                    if cutlass.const_expr(epi.residual):
                        v = v + dst.load().to(Float32)
                    dst.store(v.to(self.out_dtype))


# --------------------------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _AttentionTile:
    """Queries per CTA (bm), keys per step (bn) and warps; each warp owns 16 query rows."""

    bm: int
    bn: int
    warps: int


class _AttentionKernel:
    """Causal flash attention of T new queries per sequence over ``pos + T`` cached keys.

    q is (rows, n_embd) with row ``b * T + t``; the KV cache is (2, batch * cap, n_embd) with row
    ``b * cap + position``. Head h owns columns ``h * head_size`` to ``(h + 1) * head_size``. The
    query projection already contains the 1/sqrt(head_size) scale.

    With ``splits > 1`` the keys of a query tile are divided among ``splits`` CTAs. Each writes its
    normalized output and the log-sum-exp of its scores, and ``_AttentionCombineKernel`` merges
    them. With ``splits == 1`` the CTA writes the float16 output directly.
    """

    def __init__(self, n_head: int, head_size: int, tile: _AttentionTile, splits: int, *, pdl):
        assert head_size % 16 == 0 and tile.bm % (tile.warps * 16) == 0
        self.nh, self.hs, self.c = n_head, head_size, n_head * head_size
        self.tile, self.splits, self.pdl = tile, splits, pdl
        self.threads = tile.warps * 32

    def compile(self) -> object:
        rows, rows_kv, rows_part = cute.sym_int32(), cute.sym_int32(), cute.sym_int32()
        return _compile(
            self,
            _fake(Float16, (rows, self.c)),
            _fake(Float16, (2, rows_kv, self.c)),
            _fake(Float16, (rows, self.c)),
            _fake(Float32, (self.splits, rows_part, self.c)),
            _fake(Float32, (self.splits, rows_part, self.nh)),
            _fake(Int32, (1,)),
            Int32(1),
            Int32(1),
            Int32(1),
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mY: cute.Tensor,
        mOpart: cute.Tensor,
        mLse: cute.Tensor,
        mPos: cute.Tensor,
        n_seq: Int32,
        seq_len: Int32,
        cap: Int32,
        stream: CUstream,
    ):
        t = self.tile
        sQ_layout = _swizzled_smem_layout(t.bm, self.hs)
        sKV_layout = _swizzled_smem_layout(t.bn, self.hs)
        copy_g2s = _g2s_tiled_copy(self.threads, self.hs)
        tiled_mma = _warp_mma(t.warps, 1)
        grid = (cute.ceil_div(seq_len, t.bm), n_seq * self.nh, self.splits)
        block = [self.threads, 1, 1]
        self.kernel(
            mQ, mKV, mY, mOpart, mLse, mPos, seq_len, cap,
            sQ_layout, sKV_layout, copy_g2s, tiled_mma,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=self.pdl)  # fmt: skip

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mY: cute.Tensor,
        mOpart: cute.Tensor,
        mLse: cute.Tensor,
        mPos: cute.Tensor,
        seq_len: Int32,
        cap: Int32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        copy_g2s: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        bm, bn, hs, nh = self.tile.bm, self.tile.bn, self.hs, self.nh
        tidx, _, _ = cute.arch.thread_idx()
        q_tile, bh, split = cute.arch.block_idx()
        b = bh // nh
        h = bh % nh
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        pos = mPos[0]
        q0 = q_tile * bm  # first query of the tile, counted within its sequence
        kv_len = pos + seq_len
        # The last query of the tile sees keys up to its own position; later keys are skipped.
        kv_end = cutlass.min(pos + q0 + bm, kv_len)
        n_blocks = cute.ceil_div(kv_end, bn)
        per_split = cute.ceil_div(n_blocks, self.splits)
        blk_begin = split * per_split
        blk_end = cutlass.min(blk_begin + per_split, n_blocks)
        # Key blocks that end at or before the first query of the tile need no causal mask.
        n_unmasked = (pos + q0 + 1) // bn

        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(Float16, sQ_layout, 1024)
        sK = smem.allocate_tensor(Float16, sKV_layout, 1024)
        sV = smem.allocate_tensor(Float16, sKV_layout, 1024)
        # P @ V needs V as the K-major B operand (head_size, keys): a transposed view, read with
        # the transposing ldmatrix.
        sVt = cute.composition(sV, cute.make_layout((hs, bn), stride=(bn, 1)))

        q_row0 = b * seq_len + q0
        gQ = cute.local_tile(cute.domain_offset((q_row0, h * hs), mQ), (bm, hs), (0, 0))
        gQ = cute.make_tensor(gQ.iterator.align(16), gQ.layout)
        cache_origin = (b * cap, h * hs)
        gK = cute.local_tile(
            cute.domain_offset(cache_origin, mKV[0, None, None]), (bn, hs), (None, 0)
        )
        gK = cute.make_tensor(gK.iterator.align(16), gK.layout)  # (bn, hs, blocks)
        gV = cute.local_tile(
            cute.domain_offset(cache_origin, mKV[1, None, None]), (bn, hs), (None, 0)
        )
        gV = cute.make_tensor(gV.iterator.align(16), gV.layout)

        thr_g2s = copy_g2s.get_slice(tidx)
        tQgQ = thr_g2s.partition_S(gQ)
        tQsQ = thr_g2s.partition_D(sQ)
        tKgK = thr_g2s.partition_S(gK)
        tKsK = thr_g2s.partition_D(sK)
        tVgV = thr_g2s.partition_S(gV)
        tVsV = thr_g2s.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((bm, hs)), Float32)
        acc_O.fill(0.0)

        s2r_q = cute.make_tiled_copy_A(_ldmatrix(False), tiled_mma).get_slice(tidx)
        s2r_k = cute.make_tiled_copy_B(_ldmatrix(False), tiled_mma).get_slice(tidx)
        s2r_v = cute.make_tiled_copy_B(_ldmatrix(True), tiled_mma).get_slice(tidx)
        tSsQ = s2r_q.partition_S(sQ)
        tSrQ_view = s2r_q.retile(tSrQ)
        tSsK = s2r_k.partition_S(sK)
        tSrK_view = s2r_k.retile(tSrK)
        tOsVt = s2r_v.partition_S(sVt)
        tOrVt_view = s2r_v.retile(tOrVt)

        # Online softmax state per row of this thread, in log2 units.
        acc_O_mn = _mn_view(acc_O)
        n_rows = cute.size(acc_O_mn, mode=[0])
        row_max = cute.make_rmem_tensor((n_rows,), Float32)
        row_sum = cute.make_rmem_tensor((n_rows,), Float32)
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)
        # (query, key) coordinates of every score this thread holds, within the tile.
        tScS_mn = _mn_view(thr_mma.partition_C(cute.make_identity_tensor((bm, bn))))

        if blk_begin < blk_end:
            cute.copy(copy_g2s, tQgQ, tQsQ)
            cute.copy(copy_g2s, tKgK[None, None, None, blk_begin], tKsK)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            cute.copy(s2r_q, tSsQ, tSrQ_view)  # Q stays in registers from here on

            for n in range(blk_begin, blk_end):
                if n > blk_begin:
                    # K of this block has landed, and every warp is done with V of the last one.
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                cute.copy(copy_g2s, tVgV[None, None, None, n], tVsV)
                cute.arch.cp_async_commit_group()

                # S = Q K^T on tensor cores.
                acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((bm, bn)), Float32)
                acc_S.fill(0.0)
                cute.copy(s2r_k, tSsK, tSrK_view)
                for kb in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
                    cute.gemm(tiled_mma, acc_S, tSrQ[None, None, kb], tSrK[None, None, kb], acc_S)

                acc_S_mn = _mn_view(acc_S)
                if n >= n_unmasked:
                    for r in cutlass.range_constexpr(n_rows):
                        # Query q0 + i sits at position pos + q0 + i and sees keys up to there.
                        limit = cutlass.min(pos + q0 + tScS_mn[r, 0][0] + 1, kv_len) - n * bn
                        for c in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[1])):
                            if tScS_mn[0, c][1] >= limit:
                                acc_S_mn[r, c] = -Float32.inf

                for r in cutlass.range_constexpr(n_rows):
                    scores = acc_S_mn[r, None].load() * LOG2_E
                    new_max = _quad_max(scores.reduce(cute.ReductionOp.MAX, -Float32.inf, 0))
                    new_max = cute.arch.fmax(new_max, row_max[r])
                    # A row that has seen only masked keys keeps reference 0, so exp2 gives 0.
                    ref = new_max if new_max != -Float32.inf else Float32(0.0)
                    p = cute.math.exp2(scores - ref, fastmath=True)
                    rescale = cute.math.exp2(row_max[r] - ref, fastmath=True)
                    row_sum[r] = row_sum[r] * rescale + p.reduce(cute.ReductionOp.ADD, 0.0, 0)
                    row_max[r] = new_max
                    acc_S_mn[r, None] = p
                    acc_O_mn[r, None] = acc_O_mn[r, None].load() * rescale

                # The C fragment of S, (4, MMA_M, MMA_N), is the A fragment of P @ V once the
                # N mode is split in pairs: ((4, 2), MMA_M, MMA_N / 2). No data moves.
                rP = cute.make_fragment_like(acc_S, Float16)
                rP.store(acc_S.load().to(Float16))
                rP_div = cute.logical_divide(rP.layout, (None, None, 2))
                tOrP = cute.make_tensor(
                    rP.iterator,
                    cute.make_layout(
                        (
                            (rP_div.shape[0], rP_div.shape[2][0]),
                            rP_div.shape[1],
                            rP_div.shape[2][1],
                        ),
                        stride=(
                            (rP_div.stride[0], rP_div.stride[2][0]),
                            rP_div.stride[1],
                            rP_div.stride[2][1],
                        ),
                    ),
                )

                # V has landed, and every warp is done with K: fetch the next K during P @ V.
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                if n + 1 < blk_end:
                    cute.copy(copy_g2s, tKgK[None, None, None, n + 1], tKsK)
                    cute.arch.cp_async_commit_group()
                cute.copy(s2r_v, tOsVt, tOrVt_view)
                for kb in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
                    cute.gemm(tiled_mma, acc_O, tOrP[None, None, kb], tOrVt[None, None, kb], acc_O)

        # Normalize by the row sums and compute the log-sum-exp of every row.
        lse = cute.make_rmem_tensor((n_rows,), Float32)
        for r in cutlass.range_constexpr(n_rows):
            total = _quad_sum(row_sum[r])
            inv = Float32(0.0)
            lse[r] = -Float32.inf
            if total > 0.0:
                inv = cute.arch.rcp_approx(total)
                lse[r] = (row_max[r] + cute.math.log2(total, fastmath=True)) * LN_2
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv

        tOcO = thr_mma.partition_C(cute.make_identity_tensor((bm, hs)))
        out_origin = (q_row0, h * hs)
        if cutlass.const_expr(self.splits == 1):
            gY = cute.local_tile(cute.domain_offset(out_origin, mY), (bm, hs), (0, 0))
            tOgY = thr_mma.partition_C(gY)
            rO = cute.make_fragment_like(acc_O, Float16)
            rO.store(acc_O.load().to(Float16))
            for i in cutlass.range_constexpr(cute.size(rO)):
                if q0 + tOcO[i][0] < seq_len:
                    tOgY[i] = rO[i]
        else:
            part = mOpart[split, None, None]
            gO = cute.local_tile(cute.domain_offset(out_origin, part), (bm, hs), (0, 0))
            tOgO = thr_mma.partition_C(gO)
            for i in cutlass.range_constexpr(cute.size(acc_O)):
                if q0 + tOcO[i][0] < seq_len:
                    tOgO[i] = acc_O[i]
            tOcO_mn = _mn_view(tOcO)
            if tidx % 4 == 0:
                for r in cutlass.range_constexpr(n_rows):
                    t = q0 + tOcO_mn[r, 0][0]
                    if t < seq_len:
                        mLse[split, q_row0 - q0 + t, h] = lse[r]


@cute.jit
def _mn_view(acc: cute.Tensor) -> cute.Tensor:
    """View an accumulator fragment ((2, 2), MMA_M, MMA_N) as (rows, cols) of this thread."""
    shape = cute.make_layout(acc.layout.shape)
    mn = cute.make_layout(
        ((shape.shape[0][1], shape.shape[1]), (shape.shape[0][0], shape.shape[2])),
        stride=((shape.stride[0][1], shape.stride[1]), (shape.stride[0][0], shape.stride[2])),
    )
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))


class _AttentionCombineKernel:
    """Merge the split-KV partial outputs: ``y = sum_s exp(lse_s) o_s / sum_s exp(lse_s)``.

    One warp handles one (row, head) pair; each lane owns head_size / 32 values.
    """

    WARPS = 4

    def __init__(self, n_head: int, head_size: int, splits: int, *, pdl):
        assert head_size % 32 == 0
        self.nh, self.hs, self.c = n_head, head_size, n_head * head_size
        self.splits, self.pdl = splits, pdl

    def compile(self) -> object:
        rows, rows_part = cute.sym_int32(), cute.sym_int32()
        return _compile(
            self,
            _fake(Float32, (self.splits, rows_part, self.c)),
            _fake(Float32, (self.splits, rows_part, self.nh)),
            _fake(Float16, (rows, self.c)),
            Int32(1),
        )

    @cute.jit
    def __call__(
        self, mOpart: cute.Tensor, mLse: cute.Tensor, mY: cute.Tensor, m: Int32, stream: CUstream
    ):
        grid = (cute.ceil_div(m * self.nh, self.WARPS), 1, 1)
        self.kernel(mOpart, mLse, mY, m).launch(
            grid=grid, block=[self.WARPS * 32, 1, 1], stream=stream, use_pdl=self.pdl
        )

    @cute.kernel
    def kernel(self, mOpart: cute.Tensor, mLse: cute.Tensor, mY: cute.Tensor, m: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        pair = bidx * self.WARPS + tidx // 32
        lane = tidx % 32
        row = pair // self.nh
        h = pair % self.nh
        per_lane = self.hs // 32
        if row < m:
            top = -Float32.inf
            for s in cutlass.range_constexpr(self.splits):
                top = cute.arch.fmax(top, mLse[s, row, h])
            chunk = h * (self.hs // per_lane) + lane
            acc = cute.make_rmem_tensor((per_lane,), Float32)
            acc.fill(0.0)
            total = Float32(0.0)
            for s in cutlass.range_constexpr(self.splits):
                # Empty splits have lse = -inf and weight 0.
                weight = cute.math.exp2((mLse[s, row, h] - top) * LOG2_E, fastmath=True)
                total = total + weight
                part = cute.local_tile(mOpart[s, row, None], (per_lane,), (chunk,)).load()
                acc.store(acc.load() + part * weight)
            out = acc.load() * cute.arch.rcp_approx(total)
            cute.local_tile(mY[row, None], (per_lane,), (chunk,)).store(out.to(Float16))


# --------------------------------------------------------------------------------------------
# Embedding and the final layer norm
# --------------------------------------------------------------------------------------------


class _EmbedKernel:
    """``x[r] = wte[idx[r]] + wpe[pos + r % T]``; one warp per row, 128-bit loads."""

    WARPS = 4

    def __init__(self, n_embd: int, *, pdl):
        assert n_embd % _VEC16 == 0
        self.c, self.pdl = n_embd, pdl

    def compile(self) -> object:
        # Separate symbols: TVM-FFI checks that tensors sharing a symbol agree on its value.
        tokens, vocab, block, rows = (cute.sym_int32() for _ in range(4))
        return _compile(
            self,
            _fake(Int32, (tokens,)),
            _fake(Float16, (vocab, self.c)),
            _fake(Float16, (block, self.c)),
            _fake(Float16, (rows, self.c)),
            _fake(Int32, (1,)),
            Int32(1),
            Int32(1),
        )

    @cute.jit
    def __call__(
        self,
        mIdx: cute.Tensor,
        mWte: cute.Tensor,
        mWpe: cute.Tensor,
        mX: cute.Tensor,
        mPos: cute.Tensor,
        m: Int32,
        seq_len: Int32,
        stream: CUstream,
    ):
        grid = (cute.ceil_div(m, self.WARPS), 1, 1)
        self.kernel(mIdx, mWte, mWpe, mX, mPos, m, seq_len).launch(
            grid=grid, block=[self.WARPS * 32, 1, 1], stream=stream, use_pdl=self.pdl
        )

    @cute.kernel
    def kernel(
        self,
        mIdx: cute.Tensor,
        mWte: cute.Tensor,
        mWpe: cute.Tensor,
        mX: cute.Tensor,
        mPos: cute.Tensor,
        m: Int32,
        seq_len: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        row = bidx * self.WARPS + tidx // 32
        lane = tidx % 32
        chunks = self.c // _VEC16
        if row < m:
            token = mIdx[row]
            position = mPos[0] + row % seq_len
            for i in cutlass.range_constexpr((chunks + 31) // 32):
                chunk = lane + 32 * i
                if chunk < chunks:
                    tok = cute.local_tile(mWte[token, None], (_VEC16,), (chunk,)).load()
                    pe = cute.local_tile(mWpe[position, None], (_VEC16,), (chunk,)).load()
                    out = (tok.to(Float32) + pe.to(Float32)).to(Float16)
                    cute.local_tile(mX[row, None], (_VEC16,), (chunk,)).store(out)


class _LayerNormKernel:
    """Layer norm of whole rows; one warp per row.

    By default ``out[r] = layer_norm(x[r * step + offset])`` in float16; ``step`` and ``offset``
    let the kernel gather the last token of every sequence. With ``stats_only``, ``out[r]`` is
    ``(mean, 1/std)`` of ``x[r]`` in float32, for GEMMs with a folded layer norm.

    The row stays in registers, so the variance is computed around the mean in a second pass
    over registers rather than as E[x^2] - E[x]^2.
    """

    WARPS = 4

    def __init__(self, n_embd: int, *, stats_only: bool = False, pdl):
        assert n_embd % _VEC16 == 0
        self.c, self.stats_only, self.pdl = n_embd, stats_only, pdl

    def compile(self) -> object:
        rows_in, rows_out = cute.sym_int32(), cute.sym_int32()
        out = (
            _fake(Float32, (rows_out, 2)) if self.stats_only else _fake(Float16, (rows_out, self.c))
        )
        return _compile(
            self,
            _fake(Float16, (rows_in, self.c)),
            _fake(Float16, (self.c,)),
            _fake(Float16, (self.c,)),
            out,
            Int32(1),
            Int32(1),
            Int32(1),
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mGain: cute.Tensor,
        mShift: cute.Tensor,
        mOut: cute.Tensor,
        m: Int32,
        step: Int32,
        offset: Int32,
        stream: CUstream,
    ):
        grid = (cute.ceil_div(m, self.WARPS), 1, 1)
        self.kernel(mX, mGain, mShift, mOut, m, step, offset).launch(
            grid=grid, block=[self.WARPS * 32, 1, 1], stream=stream, use_pdl=self.pdl
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mGain: cute.Tensor,
        mShift: cute.Tensor,
        mOut: cute.Tensor,
        m: Int32,
        step: Int32,
        offset: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()
        row = bidx * self.WARPS + tidx // 32
        lane = tidx % 32
        chunks = self.c // _VEC16
        per_lane = (chunks + 31) // 32
        if row < m:
            src = mX[row * step + offset, None]
            vals = cute.make_rmem_tensor((_VEC16, per_lane), Float32)
            vals.fill(0.0)
            total = Float32(0.0)
            for i in cutlass.range_constexpr(per_lane):
                chunk = lane + 32 * i
                if chunk < chunks:
                    v = cute.local_tile(src, (_VEC16,), (chunk,)).load().to(Float32)
                    vals[None, i] = v
                    total = total + v.reduce(cute.ReductionOp.ADD, 0.0, 0)
            mean = _warp_sum(total) / self.c
            total_sq = Float32(0.0)
            for i in cutlass.range_constexpr(per_lane):
                if lane + 32 * i < chunks:
                    d = vals[None, i].load() - mean
                    total_sq = total_sq + (d * d).reduce(cute.ReductionOp.ADD, 0.0, 0)
            rstd = cute.math.rsqrt(_warp_sum(total_sq) / self.c + LAYER_NORM_EPS)
            if cutlass.const_expr(self.stats_only):
                if lane == 0:
                    mOut[row, 0] = mean
                    mOut[row, 1] = rstd
            else:
                for i in cutlass.range_constexpr(per_lane):
                    chunk = lane + 32 * i
                    if chunk < chunks:
                        gain = cute.local_tile(mGain, (_VEC16,), (chunk,)).load().to(Float32)
                        shift = cute.local_tile(mShift, (_VEC16,), (chunk,)).load().to(Float32)
                        out = (vals[None, i].load() - mean) * rstd * gain + shift
                        dst = cute.local_tile(mOut[row, None], (_VEC16,), (chunk,))
                        dst.store(out.to(Float16))


# --------------------------------------------------------------------------------------------
# Host side: weights, buffers, tile choices and the launch sequence
# --------------------------------------------------------------------------------------------

# Tile sizes were measured on an RTX PRO 4000 Blackwell (SM120, 70 SMs) with GPT-2 small.
#
# Decode step: a 16-row tile holds every sequence of the batch, so a step reads each weight
# once. Memory bandwidth bounds the speed, so deep pipelines with wide K tiles keep many loads in
# flight, and narrow N tiles give enough CTAs to keep the SMs busy.
_DECODE_ROWS = 16
_DECODE_GEMMS = (
    _GemmTile(16, 64, 128, 4, 1, 4),
    _GemmTile(16, 32, 128, 4, 1, 2),
    _GemmTile(16, 64, 64, 6, 1, 4),
    _GemmTile(16, 32, 64, 6, 1, 2),
)
# Prompt pass: arithmetic bounds the speed. A larger tile reuses every loaded value in more
# products, but needs enough tiles to fill the SMs. The tiles run from the largest to the
# smallest. The 128 x 128 tile has 8 warps while it makes at most 3 waves: the extra warps hide
# load latency and share the epilogue. With more waves, 4 warps are slightly faster.
_PROMPT_GEMMS = (
    _GemmTile(128, 128, 64, 3, 2, 4),
    _GemmTile(64, 128, 64, 4, 2, 4),
    _GemmTile(128, 64, 64, 3, 2, 2),
    _GemmTile(64, 64, 64, 4, 2, 2),
    _GemmTile(32, 64, 64, 4, 1, 4),
    _GemmTile(16, 64, 64, 4, 1, 4),
)
_PROMPT_GEMM_MANY_WAVES = _GemmTile(128, 128, 64, 3, 2, 2)
_DECODE_ATTENTION = _AttentionTile(16, 64, 1)
_PROMPT_ATTENTION = _AttentionTile(64, 64, 4)


# Compiled kernels of this process, keyed by kernel class, shapes, tile and options.
_COMPILED_KERNELS: dict[tuple, object] = {}


def _gemm_tile(rows: int, n: int, k: int, sm_count: int) -> _GemmTile:
    """Pick the tile of a GEMM with ``rows`` valid rows and an (n, k) weight."""
    if rows <= _DECODE_ROWS:
        fitting = [t for t in _DECODE_GEMMS if t.fits(n, k)]
        # Prefer wide tiles if they still give half a CTA per SM.
        for tile in fitting:
            if n // tile.bn >= sm_count // 2:
                return tile
        return max(fitting, key=lambda t: n // t.bn)
    fitting = [t for t in _PROMPT_GEMMS if t.fits(n, k) and t.bm <= _next_power_of_two(rows)]
    if not fitting:
        raise ValueError(f"No GEMM tile fits n={n}, k={k}")
    for tile in fitting:
        ctas = math.ceil(rows / tile.bm) * (n // tile.bn)
        # Take the largest tile that gives half the SMs a CTA. Skip a tile that needs between
        # one and two waves: its second wave leaves most SMs idle.
        if ctas >= sm_count / 2 and not sm_count < ctas < 2 * sm_count:
            if ctas > 3 * sm_count and tile == _PROMPT_GEMMS[0]:
                return _PROMPT_GEMM_MANY_WAVES
            return tile
    return fitting[-1]


@dataclass(frozen=True)
class _LayerWeights:
    """Weights of one block. Matrices are (out, in) float16; vectors are float32.

    ``qkv`` and ``fc`` absorb ``ln_1`` and ``ln_2``: ``*_w`` holds gain * W, ``*_b`` holds
    shift @ W + bias, and ``*_colsum`` holds the row sums of the float16 ``*_w``.
    """

    qkv_w: torch.Tensor  # (3C, C), query rows also scaled by 1/sqrt(head_size)
    qkv_b: torch.Tensor
    qkv_colsum: torch.Tensor
    attn_proj_w: torch.Tensor  # (C, C)
    attn_proj_b: torch.Tensor
    fc_w: torch.Tensor  # (4C, C)
    fc_b: torch.Tensor
    fc_colsum: torch.Tensor
    mlp_proj_w: torch.Tensor  # (C, 4C)
    mlp_proj_b: torch.Tensor


def _fold_layer_norm(
    gain: torch.Tensor, shift: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold ``layer_norm(x) @ weight + bias`` into one float16 matrix and two vectors.

    ``weight`` is (in, out) as Hugging Face stores it. With ``n(x) = (x - mean) * rstd``:

        (n(x) * gain + shift) @ W + bias = rstd * (x @ W' - mean * colsum) + b'

    where ``W' = gain[:, None] * W``, ``b' = shift @ W + bias`` and ``colsum`` sums the columns
    of the rounded ``W'``, so that ``x - mean`` cancels exactly in the GEMM.
    """
    weight, bias = weight.double(), bias.double()
    folded = (gain.double()[:, None] * weight).t().to(torch.float16).contiguous()  # (out, in)
    colsum = folded.double().sum(dim=1).float()
    folded_bias = (shift.double() @ weight + bias).float()
    return folded, folded_bias, colsum


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
    head_size = n_embd // n_head
    # GEMM tiles are 64 wide, and attention feeds a head to mma in 16-wide k-blocks and gives
    # each lane head_size / 32 values when it merges splits. GPT-2 uses n_embd % 64 == 0 and
    # head_size 64.
    if n_embd % 64 != 0 or head_size % 32 != 0:
        raise ValueError(
            f"CuTe kernels need n_embd % 64 == 0 and head_size % 32 == 0, got n_embd={n_embd}, "
            f"head_size={head_size}"
        )
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
    layer_shapes: dict[str, tuple[int, ...]] = {
        "ln_1.weight": (C,),
        "ln_1.bias": (C,),
        "attn.c_attn.weight": (C, 3 * C),
        "attn.c_attn.bias": (3 * C,),
        "attn.c_proj.weight": (C, C),
        "attn.c_proj.bias": (C,),
        "ln_2.weight": (C,),
        "ln_2.bias": (C,),
        "mlp.c_fc.weight": (C, 4 * C),
        "mlp.c_fc.bias": (4 * C,),
        "mlp.c_proj.weight": (4 * C, C),
        "mlp.c_proj.bias": (C,),
    }
    assert tuple(layer_shapes) == _LAYER_KEYS
    for i in range(config.n_layer):
        shapes.update({f"h.{i}.{key}": shape for key, shape in layer_shapes.items()})

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

    Activation buffers have ``batch_size * capacity`` rows plus ``_ROW_SLACK``, so a tile that
    starts at a valid row never leaves the buffer. ``kv`` is the KV cache, (n_layer, 2,
    batch_size * cache_stride, n_embd) with ``cache_stride`` a multiple of the attention key
    block. Buffers start zeroed, so the padding that tiles read is always finite. ``pos`` counts
    the cached positions and stays on the GPU, so a captured decode step reads the current value
    when it is replayed.
    """

    def __init__(
        self,
        config: GPT2Config,
        batch_size: int,
        capacity: int,
        vocab_padded: int,
        splits: int,
        device: torch.device,
    ):
        C = config.n_embd
        rows = batch_size * capacity + _ROW_SLACK
        self.batch_size = batch_size
        self.capacity = capacity
        self.cache_stride = _round_up(capacity, _PROMPT_ATTENTION.bn)
        self.splits = splits

        def zeros(*shape: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
            return torch.zeros(*shape, dtype=dtype, device=device)

        self.pos = zeros(1, dtype=torch.int32)
        self.tokens = zeros(batch_size, dtype=torch.int32)
        self.x = zeros(rows, C)  # residual stream
        self.q = zeros(rows, C)  # queries of the current pass
        self.y = zeros(rows, C)  # attention output before c_proj
        self.h = zeros(rows, 4 * C)  # MLP hidden activations
        self.xf = zeros(rows, C)  # ln_f output, the LM head input
        self.stats = zeros(rows, 2, dtype=torch.float32)  # (mean, 1/std) of each row of x
        self.kv = zeros(config.n_layer, 2, batch_size * self.cache_stride, C)
        decode_rows = batch_size * _DECODE_ROWS
        self.o_part = zeros(splits, decode_rows, C, dtype=torch.float32)
        self.lse = zeros(splits, decode_rows, config.n_head, dtype=torch.float32)
        self.logits = zeros(batch_size, vocab_padded, dtype=torch.float32)
        self.graph: torch.cuda.CUDAGraph | None = None


class GPT2Inference:
    """GPT-2 forward pass and sampling loop built from CuTe DSL kernels, in float16.

    ``state_dict`` uses the Hugging Face layout: Conv1D weights of shape (in, out), with or
    without the ``transformer.`` prefix, in any float dtype. Weights are converted to float16.
    Logits are float32, because the kernels accumulate in float32 anyway.
    """

    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        n_head: int | None = None,
        device: torch.device | str = "cuda",
        cuda_graph: bool | None = None,
        pdl: bool | None = None,
    ):
        device = torch.device(device)
        if device.type != "cuda" or torch.version.cuda is None:
            raise ValueError(f"CuTe DSL kernels need an NVIDIA GPU, got device {device}")
        sd = {
            key.removeprefix("transformer."): value
            for key, value in state_dict.items()
            if not key.endswith(_IGNORED_SUFFIXES)
        }
        config = _infer_config(sd, n_head)
        _check_state_dict(sd, config)
        major, _ = torch.cuda.get_device_capability(device)
        if major < 8:
            raise ValueError("CuTe kernels use mma.sync and cp.async, which need SM80 or newer")

        def half(key: str) -> torch.Tensor:
            return sd[key].to(device=device, dtype=torch.float16).contiguous()

        def single(key: str) -> torch.Tensor:
            return sd[key].to(device=device, dtype=torch.float32).contiguous()

        def matrix(key: str) -> torch.Tensor:
            # Hugging Face stores Conv1D weights as (in, out); the GEMM wants K-major (out, in).
            return sd[key].to(device=device, dtype=torch.float16).t().contiguous()

        def params(key: str) -> torch.Tensor:
            return sd[key].to(device=device)

        C, head_size = config.n_embd, config.n_embd // config.n_head
        layers = []
        for i in range(config.n_layer):
            p = f"h.{i}."
            qkv_w = params(p + "attn.c_attn.weight").double()
            qkv_bias = params(p + "attn.c_attn.bias").double()
            # softmax((q / sqrt(hs)) k^T) == softmax(q k^T / sqrt(hs)): the kernel never scales
            # the scores. GPT-2 uses head_size 64, so the factor is exactly 1/8.
            scale = 1.0 / math.sqrt(head_size)
            qkv_w[:, :C] *= scale
            qkv_bias[:C] *= scale
            qkv = _fold_layer_norm(
                params(p + "ln_1.weight"), params(p + "ln_1.bias"), qkv_w, qkv_bias
            )
            fc = _fold_layer_norm(
                params(p + "ln_2.weight"),
                params(p + "ln_2.bias"),
                params(p + "mlp.c_fc.weight"),
                params(p + "mlp.c_fc.bias"),
            )
            layers.append(
                _LayerWeights(
                    qkv_w=qkv[0],
                    qkv_b=qkv[1],
                    qkv_colsum=qkv[2],
                    attn_proj_w=matrix(p + "attn.c_proj.weight"),
                    attn_proj_b=single(p + "attn.c_proj.bias"),
                    fc_w=fc[0],
                    fc_b=fc[1],
                    fc_colsum=fc[2],
                    mlp_proj_w=matrix(p + "mlp.c_proj.weight"),
                    mlp_proj_b=single(p + "mlp.c_proj.bias"),
                )
            )

        self.config = config
        self.device = device
        self.head_size = head_size
        self.cuda_graph = True if cuda_graph is None else cuda_graph
        # PDL needs SM90 or newer.
        self.pdl = major >= 9 if pdl is None else pdl
        self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        # The token embedding doubles as the LM head. Its rows are padded with zeros to a whole
        # number of GEMM tiles; the padded logits are never returned.
        self.vocab_padded = _round_up(config.vocab_size, _VOCAB_MULTIPLE)
        self.wte = torch.zeros(self.vocab_padded, C, dtype=torch.float16, device=device)
        self.wte[: config.vocab_size] = half("wte.weight")
        self.wpe = half("wpe.weight")
        self.ln_f = (half("ln_f.weight"), half("ln_f.bias"))
        self.layers = layers
        # Placeholders for kernel arguments that a variant does not read.
        self._no_bias = torch.zeros(
            max(self.vocab_padded, 4 * C), dtype=torch.float32, device=device
        )
        self._no_kv = torch.zeros(2, 1, _VEC16, dtype=torch.float16, device=device)
        self._no_part = torch.zeros(1, 1, C, dtype=torch.float32, device=device)
        self._no_lse = torch.zeros(1, 1, config.n_head, dtype=torch.float32, device=device)
        self._workspace_cache: _Workspace | None = None

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        n_head: int | None = None,
        device: torch.device | str = "cuda",
        cuda_graph: bool | None = None,
        pdl: bool | None = None,
    ) -> GPT2Inference:
        """Load weights saved from a Hugging Face ``GPT2LMHeadModel`` state dict.

        ``n_head`` is needed only for sizes other than the four published GPT-2 models.
        """
        state_dict = load_file(path, device=str(torch.device(device)))
        return cls(state_dict, n_head=n_head, device=device, cuda_graph=cuda_graph, pdl=pdl)

    # ---------------------------------------------------------------------------------------
    # Kernel launches
    # ---------------------------------------------------------------------------------------

    def _kernel(self, key: tuple, build: object) -> object:
        """Compile on first use. Row counts are dynamic, so each variant compiles once.

        Kernels depend on shapes and options, not on weights, so all models share the cache.
        """
        key = (*key, self.pdl, torch.cuda.get_device_capability(self.device))
        compiled = _COMPILED_KERNELS.get(key)
        if compiled is None:
            compiled = _COMPILED_KERNELS[key] = build().compile()
        return compiled

    def _gemm(
        self,
        ws: _Workspace,
        a: torch.Tensor,
        w: torch.Tensor,
        out: torch.Tensor,
        rows: int,
        epilogue: _Epilogue,
        *,
        bias: torch.Tensor | None = None,
        colsum: torch.Tensor | None = None,
        kv: torch.Tensor | None = None,
        seq_len: int = 1,
    ) -> None:
        n, k = w.shape
        tile = _gemm_tile(rows, n, k, self._sm_count)
        C = self.config.n_embd
        kernel = self._kernel(
            ("gemm", n, k, C, tile, epilogue),
            lambda: _GemmKernel(n, k, tile, epilogue, n_embd=C, pdl=self.pdl),
        )
        bias = bias if bias is not None else self._no_bias[:n]
        kernel(
            a, w, bias, colsum if colsum is not None else bias, out,
            kv if kv is not None else self._no_kv, ws.stats, ws.pos, rows, seq_len,
            ws.cache_stride,
        )  # fmt: skip

    def _attention(self, ws: _Workspace, kv: torch.Tensor, T: int) -> None:
        B, nh, hs = ws.batch_size, self.config.n_head, self.head_size
        tile = _DECODE_ATTENTION if T <= _DECODE_ROWS else _PROMPT_ATTENTION
        splits = ws.splits if T <= _DECODE_ROWS else 1
        kernel = self._kernel(
            ("attention", nh, hs, tile, splits),
            lambda: _AttentionKernel(nh, hs, tile, splits, pdl=self.pdl),
        )
        if splits == 1:
            kernel(ws.q, kv, ws.y, self._no_part, self._no_lse, ws.pos, B, T, ws.cache_stride)
            return
        kernel(ws.q, kv, ws.y, ws.o_part, ws.lse, ws.pos, B, T, ws.cache_stride)
        combine = self._kernel(
            ("combine", nh, hs, splits),
            lambda: _AttentionCombineKernel(nh, hs, splits, pdl=self.pdl),
        )
        combine(ws.o_part, ws.lse, ws.y, B * T)

    def _pass(
        self, ws: _Workspace, idx: torch.Tensor, T: int, logits: torch.Tensor, *, last_only: bool
    ) -> None:
        """Run ``idx`` of shape (B * T,) through the model, after the ``ws.pos`` cached tokens.

        Keys and values of the new tokens are appended to the cache. ``logits`` receives one row
        per sequence if ``last_only``, else one row per token.
        """
        B, C = ws.batch_size, self.config.n_embd
        rows = B * T
        embed = self._kernel(("embed", C), lambda: _EmbedKernel(C, pdl=self.pdl))
        embed(idx, self.wte, self.wpe, ws.x, ws.pos, rows, T)
        own_stats = rows <= _DECODE_ROWS
        stats = self._kernel(
            ("stats", C), lambda: _LayerNormKernel(C, stats_only=True, pdl=self.pdl)
        )
        qkv = _Epilogue(ln_fold=True, own_stats=own_stats, bias=True, qkv=True)
        fc = _Epilogue(ln_fold=True, own_stats=own_stats, bias=True, gelu=True)
        for i, layer in enumerate(self.layers):
            kv = ws.kv[i]
            # ln_1 is folded into the QKV GEMM, which also appends keys and values to the cache.
            if not own_stats:
                stats(ws.x, *self.ln_f, ws.stats, rows, 1, 0)
            self._gemm(
                ws, ws.x, layer.qkv_w, ws.q, rows, qkv,
                bias=layer.qkv_b, colsum=layer.qkv_colsum, kv=kv, seq_len=T,
            )  # fmt: skip
            self._attention(ws, kv, T)
            self._gemm(
                ws, ws.y, layer.attn_proj_w, ws.x, rows, _Epilogue(bias=True, residual=True),
                bias=layer.attn_proj_b,
            )  # fmt: skip
            # ln_2 is folded into the FC GEMM.
            if not own_stats:
                stats(ws.x, *self.ln_f, ws.stats, rows, 1, 0)
            self._gemm(
                ws, ws.x, layer.fc_w, ws.h, rows, fc, bias=layer.fc_b, colsum=layer.fc_colsum
            )
            self._gemm(
                ws, ws.h, layer.mlp_proj_w, ws.x, rows, _Epilogue(bias=True, residual=True),
                bias=layer.mlp_proj_b,
            )  # fmt: skip
        # ln_f gathers the rows that need logits into a compact buffer for the LM head.
        out_rows, step, offset = (B, T, T - 1) if last_only else (rows, 1, 0)
        ln_f = self._kernel(("ln_f", C), lambda: _LayerNormKernel(C, pdl=self.pdl))
        ln_f(ws.x, *self.ln_f, ws.xf, out_rows, step, offset)
        self._gemm(ws, ws.xf, self.wte, logits, out_rows, _Epilogue(out_fp32=True))

    # ---------------------------------------------------------------------------------------
    # Prompt pass and decode step
    # ---------------------------------------------------------------------------------------

    def _workspace(self, batch_size: int, length: int) -> _Workspace:
        ws = self._workspace_cache
        if ws is None or ws.batch_size != batch_size or ws.capacity < length:
            self._workspace_cache = ws = None  # release the old buffers first
            # Round up so that slightly longer requests reuse the buffers and the captured graph.
            capacity = min(self.config.block_size, max(64, _next_power_of_two(length)))
            # Split the keys of a decode step so that every SM gets about two CTAs.
            heads = batch_size * self.config.n_head
            max_blocks = math.ceil(capacity / _DECODE_ATTENTION.bn)
            splits = max(1, min(max_blocks, math.ceil(2 * self._sm_count / heads)))
            ws = self._workspace_cache = _Workspace(
                self.config, batch_size, capacity, self.vocab_padded, splits, self.device
            )
        return ws

    def _prefill(self, ws: _Workspace, idx: torch.Tensor) -> torch.Tensor:
        """Restart the cache with ``idx`` of shape (B, T); return the last-token logits."""
        T = idx.size(1)
        ws.pos.zero_()
        self._pass(ws, idx.reshape(-1).to(torch.int32), T, ws.logits, last_only=True)
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
        logits = torch.empty(B * T, self.vocab_padded, dtype=torch.float32, device=self.device)
        ws.pos.zero_()
        self._pass(ws, idx.reshape(-1).to(torch.int32), T, logits, last_only=False)
        ws.pos.fill_(T)
        return logits.view(B, T, -1)[..., : self.config.vocab_size]

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
        vocab = self.config.vocab_size
        cached = False
        for length in range(T, T + max_new_tokens):
            if length > block_size:
                logits = self._prefill(ws, out[:, length - block_size : length])
            elif not cached:
                logits = self._prefill(ws, out[:, :length])
                cached = True
            else:
                logits = self._decode(ws, out[:, length - 1])
            out[:, length] = _sample(logits[:, :vocab], temperature, top_k)
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
