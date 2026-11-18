"""用 CuTe DSL 编写 GPT-2 推理 kernel, 并导出为 cubin. 接口见 ``kernel_abi.py``.

每个 kernel 的参数都是 ``cute.Pointer``, ``Int32`` 或 ``Float32``, 顺序与 ABI 相同. PTX 中
它们分别是 ``.u64``, ``.u32`` 和 ``.f32`` 参数, 与 TileLang 版本的 C++ 签名一一对应.
kernel 在内部用运行时的形状构造张量, 并用 ``cute.assume`` 声明 ABI 保证的整除关系, 使
128 位访问仍能向量化.

CuTe DSL 自带编译器, 直接产出 cubin. 本脚本用 ``set_name_prefix`` 把入口符号设为
``<name>_0``, 检查 PTX 入口与 ABI 一致, 再写出 cubin 和 PTX.

矩阵乘法的 Tensor Core 构件 (swizzle 共享内存布局, cp.async 拷贝, ldmatrix, mma) 复用
``learn_gpt2.gpt2_inference_cute_dsl``. 这里的 kernel 是它的简化版: 没有 LayerNorm 折叠,
PDL 和 split-KV, 形状全部是运行时参数.

用法: ``python build_cute_dsl.py --arch sm_120a --out build/kernels/cute_dsl``
"""

import argparse
import math
import tempfile
from pathlib import Path

import cutlass
import cutlass.cute as cute
import kernel_abi as abi
from cuda.bindings.driver import CUstream
from cutlass import Float16, Float32, Int32
from cutlass.cute.runtime import make_ptr
from cutlass.runtime import make_fake_stream

from learn_gpt2.gpt2_inference_cute_dsl import (
    _g2s_tiled_copy,
    _ldmatrix,
    _mn_view,
    _quad_max,
    _quad_sum,
    _swizzled_smem_layout,
    _warp_mma,
    _warp_sum,
)

THREADS = abi.THREADS
WARPS = THREADS // 32
BM, BN, D = abi.TILE_M, abi.TILE_N, abi.HEAD_DIM
LOG2_E = math.log2(math.e)
SQRT_2_OVER_PI = math.sqrt(2 / math.pi)
# 每次 128 位访问 8 个 FP16.
VEC = 8

# GEMM: 2 x 2 个 warp, K 方向每块 64, 3 级 cp.async 流水线.
GEMM_BK = abi.TILE_K
GEMM_STAGES = 3
# 累加器先暂存到共享内存再整行写出. 每行补 8 个 float, 使相邻行错开 8 个 bank.
EPILOGUE_PAD = 8
# 写出时每个线程每步处理 4 个 float (128 位).
EPILOGUE_VEC = 4
GEMM_SMEM = max(2 * GEMM_STAGES * BM * GEMM_BK * 2, BM * (BN + EPILOGUE_PAD) * 4)
# Attention: Q, K, V 各一个 (64, 64) 的 FP16 块.
ATTENTION_SMEM = 3 * BM * D * 2
# LayerNorm: 两次块内归约, 每次每个 warp 一个 float.
LAYERNORM_SMEM = 2 * WARPS * 4
# LayerNorm 中每个线程最多保存的 8 元素组数.
LAYERNORM_CHUNKS = abi.MAX_EMBD // VEC // THREADS

SMEM = {
    "gpt2_embedding": 0,
    "gpt2_layernorm": LAYERNORM_SMEM,
    "gpt2_linear_bias": GEMM_SMEM,
    "gpt2_linear_bias_gelu": GEMM_SMEM,
    "gpt2_linear_bias_residual": GEMM_SMEM,
    "gpt2_lm_head": GEMM_SMEM,
    "gpt2_attention": ATTENTION_SMEM,
}


def _row(ptr: cute.Pointer, row: Int32, width: Int32) -> cute.Tensor:
    """行主序矩阵中一行的视图. width 是 8 的倍数, 所以每行起点按 16 字节对齐."""
    return cute.make_tensor((ptr + row * width).align(16), cute.make_layout(width))


# --------------------------------------------------------------------------------------------
# Embedding 与 LayerNorm: 每个 block 处理一行, 每个线程处理若干个 8 元素组.
# --------------------------------------------------------------------------------------------


@cute.kernel
def embedding(
    x: cute.Pointer,
    ids: cute.Pointer,
    wte: cute.Pointer,
    wpe: cute.Pointer,
    rows: Int32,
    pos0: Int32,
    n_embd: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    width = cute.assume(n_embd, divby=VEC)
    token = cute.make_tensor(ids, cute.make_layout(rows))[r]
    token_row = _row(wte, token, width)
    position_row = _row(wpe, pos0 + r, width)
    out_row = _row(x, r, width)
    chunks = width // VEC
    for i in range(cute.ceil_div(chunks, THREADS)):
        chunk = tidx + i * THREADS
        if chunk < chunks:
            a = cute.local_tile(token_row, (VEC,), (chunk,)).load().to(Float32)
            b = cute.local_tile(position_row, (VEC,), (chunk,)).load().to(Float32)
            cute.local_tile(out_row, (VEC,), (chunk,)).store((a + b).to(Float16))


@cute.jit
def _block_sum(value: Float32, partial: cute.Tensor, slot: int, tidx: Int32) -> Float32:
    """block 内 4 个 warp 求和. ``slot`` 选择共享内存中的一组位置, 两次归约互不覆盖."""
    value = _warp_sum(value)
    if tidx % 32 == 0:
        partial[slot * WARPS + tidx // 32] = value
    cute.arch.sync_threads()
    total = Float32(0.0)
    for w in cutlass.range_constexpr(WARPS):
        total = total + partial[slot * WARPS + w]
    return total


@cute.kernel
def layernorm(
    y: cute.Pointer,
    x: cute.Pointer,
    gamma: cute.Pointer,
    beta: cute.Pointer,
    rows: Int32,
    n_embd: Int32,
    eps: Float32,
):
    tidx, _, _ = cute.arch.thread_idx()
    r, _, _ = cute.arch.block_idx()
    width = cute.assume(n_embd, divby=VEC)
    chunks = width // VEC
    count = Float32(n_embd)
    smem = cutlass.utils.SmemAllocator()
    partial = smem.allocate_tensor(Float32, cute.make_layout(2 * WARPS), 16)
    src = _row(x, r, width)

    # 整行留在寄存器中, 方差按 "减去均值后的平方和" 计算, 比 E[x^2] - E[x]^2 更稳定.
    values = cute.make_rmem_tensor((VEC, LAYERNORM_CHUNKS), Float32)
    values.fill(0.0)
    total = Float32(0.0)
    for i in cutlass.range_constexpr(LAYERNORM_CHUNKS):
        chunk = tidx + i * THREADS
        if chunk < chunks:
            v = cute.local_tile(src, (VEC,), (chunk,)).load().to(Float32)
            values[None, i] = v
            total = total + v.reduce(cute.ReductionOp.ADD, 0.0, 0)
    mean = _block_sum(total, partial, 0, tidx) / count
    total_sq = Float32(0.0)
    for i in cutlass.range_constexpr(LAYERNORM_CHUNKS):
        if tidx + i * THREADS < chunks:
            d = values[None, i].load() - mean
            total_sq = total_sq + (d * d).reduce(cute.ReductionOp.ADD, 0.0, 0)
    rstd = cute.math.rsqrt(_block_sum(total_sq, partial, 1, tidx) / count + eps)

    gain = cute.make_tensor(gamma, cute.make_layout(width))
    shift = cute.make_tensor(beta, cute.make_layout(width))
    dst = _row(y, r, width)
    for i in cutlass.range_constexpr(LAYERNORM_CHUNKS):
        chunk = tidx + i * THREADS
        if chunk < chunks:
            g = cute.local_tile(gain, (VEC,), (chunk,)).load().to(Float32)
            b = cute.local_tile(shift, (VEC,), (chunk,)).load().to(Float32)
            out = (values[None, i].load() - mean) * rstd * g + b
            cute.local_tile(dst, (VEC,), (chunk,)).store(out.to(Float16))


# --------------------------------------------------------------------------------------------
# GEMM: y = epilogue(x @ w^T). 一个 block 计算一个 (BM, BN) 输出块.
# --------------------------------------------------------------------------------------------


@cute.jit
def _gemm(
    y: cute.Pointer,
    x: cute.Pointer,
    w: cute.Pointer,
    b: cute.Pointer,
    m: Int32,
    n: Int32,
    k: Int32,
    epilogue: cutlass.Constexpr,
):
    """epilogue 为 ``bias``, ``gelu``, ``residual`` 或 ``logits`` (无 bias, FP32 输出).

    ``cp.async`` 把 (BM, BK) 与 (BN, BK) 的块经多级环形缓冲读入共享内存, ``ldmatrix``
    把操作数读入寄存器, ``mma.sync`` 在 Tensor Core 上累加. x 的行数按 ABI 补齐,
    读取不检查边界; 只有写出检查 ``row < m``.
    """
    bk, stages = GEMM_BK, GEMM_STAGES
    tidx, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()
    k_ = cute.assume(k, divby=abi.TILE_K)
    n_ = cute.assume(n, divby=abi.TILE_N)
    k_tiles = k_ // bk
    mA = cute.make_tensor(x, cute.make_layout((m, k_), stride=(k_, 1)))
    mW = cute.make_tensor(w, cute.make_layout((n_, k_), stride=(k_, 1)))
    mOut = cute.make_tensor(y, cute.make_layout((m, n_), stride=(n_, 1)))
    mBias = cute.make_tensor(b, cute.make_layout(n_))

    tiled_mma = _warp_mma(2, 2)
    sA_layout = _swizzled_smem_layout(BM, bk, stages)
    sW_layout = _swizzled_smem_layout(BN, bk, stages)
    sC_layout = cute.make_layout((BM, BN), stride=(BN + EPILOGUE_PAD, 1))
    copy_g2s = _g2s_tiled_copy(THREADS, bk)

    gA = cute.local_tile(mA, (BM, bk), (tile_m, None))  # (BM, bk, k_tiles)
    gW = cute.local_tile(mW, (BN, bk), (tile_n, None))  # (BN, bk, k_tiles)

    # 一块共享内存先用作 A 与 W 的环形缓冲, 主循环结束后再暂存累加器.
    smem = cutlass.utils.SmemAllocator()
    buf = smem.allocate(GEMM_SMEM, byte_alignment=1024)
    half_ptr = cute.recast_ptr(buf, dtype=Float16)
    sA = cute.make_tensor(half_ptr, sA_layout)
    sW = cute.make_tensor(half_ptr + cute.cosize(sA_layout), sW_layout)
    sC = cute.make_tensor(cute.recast_ptr(buf, dtype=Float32), sC_layout)

    thr_g2s = copy_g2s.get_slice(tidx)
    tAgA = thr_g2s.partition_S(gA)
    tAsA = thr_g2s.partition_D(sA)
    tWgW = thr_g2s.partition_S(gW)
    tWsW = thr_g2s.partition_D(sW)

    # 预取前 stages - 1 个块, 每块一个提交组. 块数不足时也提交空组, 使计数保持规则.
    for s in cutlass.range_constexpr(stages - 1):
        if s < k_tiles:
            cute.copy(copy_g2s, tAgA[None, None, None, s], tAsA[None, None, None, s])
            cute.copy(copy_g2s, tWgW[None, None, None, s], tWsW[None, None, None, s])
        cute.arch.cp_async_commit_group()

    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsW = thr_mma.partition_B(sW)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrW = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
    acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((BM, BN)), Float32)
    acc.fill(0.0)

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

    # 两条流水线重叠: 计算第 k_tile 块时请求第 k_tile + stages - 1 块;
    # 计算第 kb 个 k 子块时用 ldmatrix 读入第 kb + 1 个.
    for k_tile in range(k_tiles):
        for kb in cutlass.range_constexpr(num_kb):
            if kb == num_kb - 1:
                tCsA_p = tCsA_view[None, None, None, read]
                tCsW_p = tCsW_view[None, None, None, read]
                cute.arch.cp_async_wait_group(stages - 2)
                cute.arch.sync_threads()
            kb_next = (kb + 1) % num_kb
            cute.copy(s2r_a, tCsA_p[None, None, kb_next], tCrA_view[None, None, kb_next])
            cute.copy(s2r_w, tCsW_p[None, None, kb_next], tCrW_view[None, None, kb_next])
            if kb == 0:  # noqa: SIM102
                if k_tile + stages - 1 < k_tiles:
                    nxt = k_tile + stages - 1
                    cute.copy(copy_g2s, tAgA[None, None, None, nxt], tAsA[None, None, None, write])
                    cute.copy(copy_g2s, tWgW[None, None, None, nxt], tWsW[None, None, None, write])
            cute.gemm(tiled_mma, acc, tCrA[None, None, kb], tCrW[None, None, kb], acc)
            if kb == 0:
                cute.arch.cp_async_commit_group()
                write = read
                read = read + 1
                if read == stages:
                    read = Int32(0)

    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()

    # 暂存累加器, 再由连续的线程写出连续的 4 元素组, 使全局内存写入合并.
    cute.autovec_copy(acc, thr_mma.partition_C(sC))
    cute.arch.sync_threads()
    chunks_per_row = BN // EPILOGUE_VEC
    rows_per_step = THREADS // chunks_per_row
    c = tidx % chunks_per_row
    col_chunk = tile_n * chunks_per_row + c
    if cutlass.const_expr(epilogue != "logits"):
        bias = cute.local_tile(mBias, (EPILOGUE_VEC,), (col_chunk,)).load().to(Float32)
    for step in cutlass.range_constexpr(BM // rows_per_step):
        r = tidx // chunks_per_row + step * rows_per_step
        row = tile_m * BM + r
        if row < m:
            v = cute.local_tile(sC[r, None], (EPILOGUE_VEC,), (c,)).load()
            dst = cute.local_tile(mOut[row, None], (EPILOGUE_VEC,), (col_chunk,))
            if cutlass.const_expr(epilogue == "logits"):
                dst.store(v)
            else:
                v = v + bias
                if cutlass.const_expr(epilogue == "gelu"):
                    # tanh 近似. 0.5 * (1 + tanh(z)) = 1 / (1 + exp(-2z)).
                    z = v * (v * v * 0.044715 + 1.0) * (-2.0 * SQRT_2_OVER_PI * LOG2_E)
                    v = v * cute.math.rcp(
                        cute.math.exp2(z, fastmath=True) + 1.0, approx=True, ftz=True
                    )
                if cutlass.const_expr(epilogue == "residual"):
                    v = v + dst.load().to(Float32)
                dst.store(v.to(Float16))


def _linear_kernel(epilogue: str) -> object:
    @cute.kernel
    def linear(
        y: cute.Pointer,
        x: cute.Pointer,
        w: cute.Pointer,
        b: cute.Pointer,
        m: Int32,
        n: Int32,
        k: Int32,
    ):
        _gemm(y, x, w, b, m, n, k, epilogue)

    return linear


@cute.kernel
def lm_head(y: cute.Pointer, x: cute.Pointer, w: cute.Pointer, m: Int32, n: Int32, k: Int32):
    # 没有 bias. 传入 x 只是为了满足 _gemm 的签名, 它不会被当作 bias 读取.
    _gemm(y, x, w, x, m, n, k, "logits")


# --------------------------------------------------------------------------------------------
# Attention: 因果 FlashAttention, 一个 block 处理一个 head 的 64 个 query.
# --------------------------------------------------------------------------------------------


@cute.kernel
def attention(
    y: cute.Pointer,
    qkv: cute.Pointer,
    k_cache: cute.Pointer,
    v_cache: cute.Pointer,
    m: Int32,
    pos0: Int32,
    n_head: Int32,
    scale: Float32,
):
    """Q K^T 与 P V 都在 Tensor Core 上计算. Q 留在寄存器中; softmax 的结果 P 直接在寄存器
    中从累加器布局转换为下一个 mma 的输入布局, 不经过共享内存."""
    bm, bn, hs = BM, BN, D
    tidx, _, _ = cute.arch.thread_idx()
    q_tile, h, _ = cute.arch.block_idx()
    c = cute.assume(n_head * hs, divby=hs)
    kv_len = pos0 + m
    mQ = cute.make_tensor(qkv, cute.make_layout((m, c), stride=(cute.assume(3 * c, divby=hs), 1)))
    mY = cute.make_tensor(y, cute.make_layout((m, c), stride=(c, 1)))
    mK = cute.make_tensor(k_cache, cute.make_layout((kv_len, c), stride=(c, 1)))
    mV = cute.make_tensor(v_cache, cute.make_layout((kv_len, c), stride=(c, 1)))

    sQ_layout = _swizzled_smem_layout(bm, hs)
    sKV_layout = _swizzled_smem_layout(bn, hs)
    copy_g2s = _g2s_tiled_copy(THREADS, hs)
    tiled_mma = _warp_mma(WARPS, 1)

    q0 = q_tile * bm
    # 块内最后一个 query 位于 pos0 + q0 + bm - 1, 之后的 key 都被遮住.
    kv_end = cutlass.min(pos0 + q0 + bm, kv_len)
    n_blocks = cute.ceil_div(kv_end, bn)
    # 结束位置不超过块内第一个 query 的 key 块不需要因果掩码.
    n_unmasked = (pos0 + q0 + 1) // bn
    scale_log2 = scale * LOG2_E

    smem = cutlass.utils.SmemAllocator()
    buf = smem.allocate(ATTENTION_SMEM, byte_alignment=1024)
    half_ptr = cute.recast_ptr(buf, dtype=Float16)
    sQ = cute.make_tensor(half_ptr, sQ_layout)
    sK = cute.make_tensor(half_ptr + cute.cosize(sQ_layout), sKV_layout)
    sV = cute.make_tensor(half_ptr + cute.cosize(sQ_layout) + cute.cosize(sKV_layout), sKV_layout)
    # P V 需要 V 作为 K-major 的 B 操作数 (head_dim, keys): 用转置视图, 由转置 ldmatrix 读取.
    sVt = cute.composition(sV, cute.make_layout((hs, bn), stride=(bn, 1)))

    gQ = cute.local_tile(cute.domain_offset((q0, h * hs), mQ), (bm, hs), (0, 0))
    gQ = cute.make_tensor(gQ.iterator.align(16), gQ.layout)
    gK = cute.local_tile(cute.domain_offset((0, h * hs), mK), (bn, hs), (None, 0))
    gK = cute.make_tensor(gK.iterator.align(16), gK.layout)  # (bn, hs, blocks)
    gV = cute.local_tile(cute.domain_offset((0, h * hs), mV), (bn, hs), (None, 0))
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

    # 每行的在线 softmax 状态, 以 2 为底.
    acc_O_mn = _mn_view(acc_O)
    n_rows = cute.size(acc_O_mn, mode=[0])
    row_max = cute.make_rmem_tensor((n_rows,), Float32)
    row_sum = cute.make_rmem_tensor((n_rows,), Float32)
    row_max.fill(-Float32.inf)
    row_sum.fill(0.0)
    # 本线程持有的每个分数在块内的 (query, key) 坐标.
    tScS_mn = _mn_view(thr_mma.partition_C(cute.make_identity_tensor((bm, bn))))

    cute.copy(copy_g2s, tQgQ, tQsQ)
    cute.copy(copy_g2s, tKgK[None, None, None, 0], tKsK)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    cute.copy(s2r_q, tSsQ, tSrQ_view)

    for blk in range(n_blocks):
        if blk > 0:
            # 本块的 K 已到达, 所有 warp 都已用完上一块的 V.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
        cute.copy(copy_g2s, tVgV[None, None, None, blk], tVsV)
        cute.arch.cp_async_commit_group()

        # S = Q K^T.
        acc_S = cute.make_rmem_tensor(thr_mma.partition_shape_C((bm, bn)), Float32)
        acc_S.fill(0.0)
        cute.copy(s2r_k, tSsK, tSrK_view)
        for kb in cutlass.range_constexpr(cute.size(tSrQ, mode=[2])):
            cute.gemm(tiled_mma, acc_S, tSrQ[None, None, kb], tSrK[None, None, kb], acc_S)

        acc_S_mn = _mn_view(acc_S)
        if blk >= n_unmasked:
            for r in cutlass.range_constexpr(n_rows):
                # query q0 + i 位于 pos0 + q0 + i, 看到不超过这个位置的 key.
                limit = cutlass.min(pos0 + q0 + tScS_mn[r, 0][0] + 1, kv_len) - blk * bn
                for col in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[1])):
                    if tScS_mn[0, col][1] >= limit:
                        acc_S_mn[r, col] = -Float32.inf

        for r in cutlass.range_constexpr(n_rows):
            scores = acc_S_mn[r, None].load() * scale_log2
            new_max = _quad_max(scores.reduce(cute.ReductionOp.MAX, -Float32.inf, 0))
            new_max = cute.arch.fmax(new_max, row_max[r])
            # 只见过被遮住的 key 的行用 0 作参考值, exp2 得到 0.
            ref = new_max if new_max != -Float32.inf else Float32(0.0)
            p = cute.math.exp2(scores - ref, fastmath=True)
            rescale = cute.math.exp2(row_max[r] - ref, fastmath=True)
            row_sum[r] = row_sum[r] * rescale + p.reduce(cute.ReductionOp.ADD, 0.0, 0)
            row_max[r] = new_max
            acc_S_mn[r, None] = p
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * rescale

        # S 的累加器 fragment (4, MMA_M, MMA_N) 把 N 维两两分组后, 就是 P V 的 A fragment
        # ((4, 2), MMA_M, MMA_N / 2). 数据不移动.
        rP = cute.make_fragment_like(acc_S, Float16)
        rP.store(acc_S.load().to(Float16))
        rP_div = cute.logical_divide(rP.layout, (None, None, 2))
        tOrP = cute.make_tensor(
            rP.iterator,
            cute.make_layout(
                ((rP_div.shape[0], rP_div.shape[2][0]), rP_div.shape[1], rP_div.shape[2][1]),
                stride=(
                    (rP_div.stride[0], rP_div.stride[2][0]),
                    rP_div.stride[1],
                    rP_div.stride[2][1],
                ),
            ),
        )

        # V 已到达, 所有 warp 都已用完 K: 在计算 P V 时预取下一块 K.
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if blk + 1 < n_blocks:
            cute.copy(copy_g2s, tKgK[None, None, None, blk + 1], tKsK)
            cute.arch.cp_async_commit_group()
        cute.copy(s2r_v, tOsVt, tOrVt_view)
        for kb in cutlass.range_constexpr(cute.size(tOrP, mode=[2])):
            cute.gemm(tiled_mma, acc_O, tOrP[None, None, kb], tOrVt[None, None, kb], acc_O)

    for r in cutlass.range_constexpr(n_rows):
        inv = cute.arch.rcp_approx(_quad_sum(row_sum[r]))
        acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv

    tOcO = thr_mma.partition_C(cute.make_identity_tensor((bm, hs)))
    gY = cute.local_tile(cute.domain_offset((q0, h * hs), mY), (bm, hs), (0, 0))
    tOgY = thr_mma.partition_C(gY)
    rO = cute.make_fragment_like(acc_O, Float16)
    rO.store(acc_O.load().to(Float16))
    for i in cutlass.range_constexpr(cute.size(rO)):
        if q0 + tOcO[i][0] < m:
            tOgY[i] = rO[i]


# --------------------------------------------------------------------------------------------
# Host 端启动函数. grid 规则与 kernel_abi.py 一致; 只用于编译, 本脚本不运行 kernel.
# --------------------------------------------------------------------------------------------


def _launchers() -> dict[str, object]:
    kernels = {
        "gpt2_embedding": embedding,
        "gpt2_layernorm": layernorm,
        "gpt2_linear_bias": _linear_kernel("bias"),
        "gpt2_linear_bias_gelu": _linear_kernel("gelu"),
        "gpt2_linear_bias_residual": _linear_kernel("residual"),
        "gpt2_lm_head": lm_head,
        "gpt2_attention": attention,
    }
    # CuTe DSL 生成的名字是 "<前缀>_<序号>". 每个 cubin 只有一个 kernel, 序号为 0.
    for name, kernel in kernels.items():
        kernel.set_name_prefix(name, remove_cutlass_symbol=True, keep_mangled_name=False)

    def launch(name: str, grid: tuple, *args: object, stream: CUstream) -> None:
        kernels[name](*args).launch(
            grid=grid, block=(THREADS, 1, 1), smem=SMEM[name], stream=stream
        )

    @cute.jit
    def gpt2_embedding(x, ids, wte, wpe, rows: Int32, pos0: Int32, n_embd: Int32, stream: CUstream):
        launch("gpt2_embedding", (rows, 1, 1), x, ids, wte, wpe, rows, pos0, n_embd, stream=stream)

    @cute.jit
    def gpt2_layernorm(
        y, x, gamma, beta, rows: Int32, n_embd: Int32, eps: Float32, stream: CUstream
    ):
        launch("gpt2_layernorm", (rows, 1, 1), y, x, gamma, beta, rows, n_embd, eps, stream=stream)

    def linear(name: str) -> object:
        @cute.jit
        def gemm(y, x, w, b, m: Int32, n: Int32, k: Int32, stream: CUstream):
            grid = (cute.ceil_div(m, BM), n // BN, 1)
            launch(name, grid, y, x, w, b, m, n, k, stream=stream)

        return gemm

    @cute.jit
    def gpt2_lm_head(y, x, w, m: Int32, n: Int32, k: Int32, stream: CUstream):
        launch("gpt2_lm_head", (cute.ceil_div(m, BM), n // BN, 1), y, x, w, m, n, k, stream=stream)

    @cute.jit
    def gpt2_attention(
        y,
        qkv,
        k_cache,
        v_cache,
        m: Int32,
        pos0: Int32,
        n_head: Int32,
        scale: Float32,
        stream: CUstream,
    ):
        grid = (cute.ceil_div(m, BM), n_head, 1)
        launch(
            "gpt2_attention", grid, y, qkv, k_cache, v_cache, m, pos0, n_head, scale, stream=stream
        )

    return {
        "gpt2_embedding": gpt2_embedding,
        "gpt2_layernorm": gpt2_layernorm,
        "gpt2_linear_bias": linear("gpt2_linear_bias"),
        "gpt2_linear_bias_gelu": linear("gpt2_linear_bias_gelu"),
        "gpt2_linear_bias_residual": linear("gpt2_linear_bias_residual"),
        "gpt2_lm_head": gpt2_lm_head,
        "gpt2_attention": gpt2_attention,
    }


_POINTEE = {"f16*": Float16, "const f16*": Float16, "f32*": Float32, "const i32*": Int32}


def _example_arg(param: abi.Param) -> object:
    """编译用的占位参数. 指针地址只用于编译, 不会被访问."""
    if param.type in _POINTEE:
        return make_ptr(_POINTEE[param.type], 256, cute.AddressSpace.gmem, assumed_align=16)
    return Int32(1) if param.type == "i32" else Float32(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="把 CuTe DSL kernel 导出为 cubin")
    parser.add_argument("--arch", required=True, help="GPU 架构, 例如 sm_120a 或 sm_86")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    launchers = _launchers()
    with tempfile.TemporaryDirectory() as dump_dir:
        options = f"--gpu-arch {args.arch} --keep-ptx --keep-cubin --dump-dir {dump_dir}"
        for kernel in abi.KERNELS:
            example = [_example_arg(p) for p in kernel.params]
            compiled = cute.compile(
                launchers[kernel.name], *example, make_fake_stream(), options=options
            )
            abi.check_ptx_entry(kernel, compiled.__ptx__)
            (args.out / f"{kernel.name}.ptx").write_text(compiled.__ptx__)
            (args.out / f"{kernel.name}.cubin").write_bytes(compiled.__cubin__)
            print(f"cute_dsl {kernel.symbol:32s} smem {SMEM[kernel.name]:6d} B")
    abi.write_manifest(args.out, "cute_dsl", args.arch, SMEM)


if __name__ == "__main__":
    main()
