"""用 TileLang 编写 GPT-2 推理 kernel, 并导出为 cubin. 接口见 ``kernel_abi.py``.

每个 kernel 的导出流程:

1. TileLang 把 prim_func 降级 (lower) 为 CUDA C++ 源码. 这一步不需要 GPU.
2. TileLang 生成的入口函数按自己的规则排列参数: 先是按名字排序的指针, 再是按名字排序的
   整数, 没有用到的形状变量不出现. 本脚本把这个入口改成 ``__device__`` 函数, 再加一个
   按 ABI 排列参数的 ``extern "C"`` 入口, 由它转发调用.
3. nvcc 把源码编译为 cubin. 源码 ``<name>.cu`` 保留在输出目录中, 便于阅读.

形状全部是运行时参数, 同一组 cubin 可用于 GPT-2 的四种尺寸.
GEMM 的 ``T.copy`` 带边界检查, 所以 TileLang 版本不依赖 ABI 规定的补齐行.

用法: ``python build_tilelang.py --arch sm_120a --out build/kernels/tilelang``
"""

import argparse
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import kernel_abi as abi
import tilelang
import tilelang.language as T
import tvm
from tilelang.contrib.nvcc import get_nvcc_compiler
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.transform.pass_config import normalize_pass_configs

THREADS = abi.THREADS
BM, BN, D = abi.TILE_M, abi.TILE_N, abi.HEAD_DIM
BK = 64
# TileLang 0.1.15 在 K 为运行时值时, 无法为 3 级以上的 cp.async 流水线生成代码.
STAGES = 2
LOG2_E = math.log2(math.e)
# GELU 的 tanh 近似: 0.5 * (1 + tanh(z)) = 1 / (1 + exp(-2z)), z = sqrt(2/pi) * (v + 0.044715 v^3).
GELU_2_SQRT_2_OVER_PI = 2 * math.sqrt(2 / math.pi)
# 关闭 warp 特化. 搬运数据的 copy 另外关闭 TMA, 使所有参数都是普通指针和标量.
PASS_CONFIGS = {"tl.disable_warp_specialized": True}
# TileLang 默认为依赖数据的下标插入边界检查, 例如 wte[ids[r]] 和 k_cache[key]. 这需要把
# vocab, context 和缓存行数也作为 kernel 参数. C++ 端已检查 token 编号与位置, 缓存行数由
# ABI 保证, 所以这两个 kernel 关闭这项检查. 它们的代码中已有需要的边界条件.
EXTRA_PASS_CONFIGS = dict.fromkeys(
    ("gpt2_embedding", "gpt2_attention"), {"tl.disable_safe_memory_legalize": True}
)

F16, F32 = "float16", "float32"


def embedding():
    rows, n_embd, vocab, context = T.dynamic("rows, n_embd, vocab, context")

    @T.prim_func
    def kernel(
        x: T.Tensor((rows, n_embd), F16),
        ids: T.Tensor((rows,), "int32"),
        wte: T.Tensor((vocab, n_embd), F16),
        wpe: T.Tensor((context, n_embd), F16),
        pos0: T.int32,
    ):
        # 每个 block 处理一行, 每个线程处理间隔为 THREADS 的列.
        with T.Kernel(rows, threads=THREADS) as r:
            tx = T.get_thread_binding()
            for i in T.serial(T.ceildiv(n_embd, THREADS)):
                c = i * THREADS + tx
                if c < n_embd:
                    x[r, c] = T.cast(
                        T.cast(wte[ids[r], c], F32) + T.cast(wpe[pos0 + r, c], F32), F16
                    )

    return kernel


def layernorm():
    rows, n_embd = T.dynamic("rows, n_embd")
    width = abi.MAX_EMBD

    @T.prim_func
    def kernel(
        y: T.Tensor((rows, n_embd), F16),
        x: T.Tensor((rows, n_embd), F16),
        gamma: T.Tensor((n_embd,), F16),
        beta: T.Tensor((n_embd,), F16),
        eps: T.float32,
    ):
        # 每个 block 处理一行. fragment 按最大宽度分配, 超出 n_embd 的位置补零.
        with T.Kernel(rows, threads=THREADS) as r:
            # ABI 保证 n_embd 是 64 的倍数, 使 "c < n_embd" 在每组 8 个元素内相同, 读写可以向量化.
            T.assume(n_embd % 64 == 0)
            values = T.alloc_fragment((width,), F32)
            centered = T.alloc_fragment((width,), F32)
            squares = T.alloc_fragment((width,), F32)
            total = T.alloc_fragment((1,), F32)
            variance = T.alloc_fragment((1,), F32)
            count = T.cast(n_embd, F32)
            for c in T.Parallel(width):
                values[c] = T.if_then_else(c < n_embd, T.cast(x[r, c], F32), 0.0)
            T.reduce_sum(values, total, dim=0)
            # 先减去均值再求平方和, 比 E[x^2] - E[x]^2 更稳定.
            for c in T.Parallel(width):
                centered[c] = T.if_then_else(c < n_embd, values[c] - total[0] / count, 0.0)
                squares[c] = centered[c] * centered[c]
            T.reduce_sum(squares, variance, dim=0)
            for c in T.Parallel(width):
                if c < n_embd:
                    y[r, c] = T.cast(
                        centered[c] * T.rsqrt(variance[0] / count + eps) * T.cast(gamma[c], F32)
                        + T.cast(beta[c], F32),
                        F16,
                    )

    return kernel


@T.macro
def _assume_gemm_shape(n, k):
    # ABI 保证 n 与 k 是 64 的倍数. 没有这两个提示时, TileLang 无法证明每行按 16 字节对齐,
    # T.copy 退化为逐个元素的读取, GEMM 慢数倍. 有了提示, 它生成 128 位的 cp.async.
    T.assume(n % abi.TILE_N == 0)
    T.assume(k % abi.TILE_K == 0)


def linear(epilogue: str):
    """``y = epilogue(x @ w^T + b)``. epilogue 为 ``bias``, ``gelu`` 或 ``residual``."""
    m, n, k = T.dynamic("m, n, k")
    gelu, residual = epilogue == "gelu", epilogue == "residual"

    @T.prim_func
    def kernel(
        y: T.Tensor((m, n), F16),
        x: T.Tensor((m, k), F16),
        w: T.Tensor((n, k), F16),
        b: T.Tensor((n,), F16),
    ):
        # blockIdx.x 遍历行块, 同时运行的 block 共享同一列块的权重, 权重从 L2 复用.
        with T.Kernel(T.ceildiv(m, BM), T.ceildiv(n, BN), threads=THREADS) as (bx, by):
            _assume_gemm_shape(n, k)
            xs = T.alloc_shared((BM, BK), F16)
            ws = T.alloc_shared((BN, BK), F16)
            acc = T.alloc_fragment((BM, BN), F32)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, BK), num_stages=STAGES):
                # T.copy 为越界的行补零, 所以不依赖补齐行.
                T.copy(x[bx * BM, ko * BK], xs, disable_tma=True)
                T.copy(w[by * BN, ko * BK], ws, disable_tma=True)
                T.gemm(xs, ws, acc, transpose_B=True)
            for i, j in T.Parallel(BM, BN):
                if bx * BM + i < m:
                    acc[i, j] += T.cast(b[by * BN + j], F32)
                    if gelu:
                        v = acc[i, j]
                        acc[i, j] = v / (
                            1 + T.exp2(-GELU_2_SQRT_2_OVER_PI * LOG2_E * (v + 0.044715 * v * v * v))
                        )
                    if residual:
                        acc[i, j] += T.cast(y[bx * BM + i, by * BN + j], F32)
                    y[bx * BM + i, by * BN + j] = T.cast(acc[i, j], F16)

    return kernel


def lm_head():
    m, n, k = T.dynamic("m, n, k")

    @T.prim_func
    def kernel(
        y: T.Tensor((m, n), F32),
        x: T.Tensor((m, k), F16),
        w: T.Tensor((n, k), F16),
    ):
        with T.Kernel(T.ceildiv(m, BM), T.ceildiv(n, BN), threads=THREADS) as (bx, by):
            _assume_gemm_shape(n, k)
            xs = T.alloc_shared((BM, BK), F16)
            ws = T.alloc_shared((BN, BK), F16)
            acc = T.alloc_fragment((BM, BN), F32)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, BK), num_stages=STAGES):
                T.copy(x[bx * BM, ko * BK], xs, disable_tma=True)
                T.copy(w[by * BN, ko * BK], ws, disable_tma=True)
                T.gemm(xs, ws, acc, transpose_B=True)
            T.copy(acc, y[bx * BM, by * BN])

    return kernel


def attention():
    """因果 FlashAttention: 两个 GEMM 计算 Q K^T 与 P V, 在线 softmax, 不保存注意力矩阵."""
    m, n_head, cache_rows = T.dynamic("m, n_head, cache_rows")

    @T.prim_func
    def kernel(
        y: T.Tensor((m, n_head * D), F16),
        qkv: T.Tensor((m, 3 * n_head * D), F16),
        k_cache: T.Tensor((cache_rows, n_head * D), F16),
        v_cache: T.Tensor((cache_rows, n_head * D), F16),
        pos0: T.int32,
        scale: T.float32,
    ):
        # blockIdx.x 是 query 块, blockIdx.y 是 head.
        with T.Kernel(T.ceildiv(m, BM), n_head, threads=THREADS) as (qt, h):
            q = T.alloc_shared((BM, D), F16)
            kt = T.alloc_shared((BN, D), F16)
            vt = T.alloc_shared((BN, D), F16)
            # 共享内存把 softmax 结果从累加器布局转换为第二个 GEMM 的输入布局.
            p = T.alloc_shared((BM, BN), F16)
            s = T.alloc_fragment((BM, BN), F32)
            acc = T.alloc_fragment((BM, D), F32)
            maximum = T.alloc_fragment((BM,), F32)
            previous = T.alloc_fragment((BM,), F32)
            rescale = T.alloc_fragment((BM,), F32)
            denominator = T.alloc_fragment((BM,), F32)
            tile_sum = T.alloc_fragment((BM,), F32)
            total = pos0 + m
            # 块内最后一个 query 位于 pos0 + (qt + 1) * BM - 1, 之后的 key 都被遮住.
            end = T.min(total, pos0 + (qt + 1) * BM)
            for i, d in T.Parallel(BM, D):
                q[i, d] = T.if_then_else(qt * BM + i < m, qkv[qt * BM + i, h * D + d], 0)
            T.clear(acc)
            T.clear(denominator)
            # 有限的初值避免 -inf - -inf. 第一个 key 块总含位置 0, 所以每行都有可见的 key.
            T.fill(maximum, -1e30)
            for t in T.Pipelined(T.ceildiv(end, BN), num_stages=1):
                for j, d in T.Parallel(BN, D):
                    key = t * BN + j
                    kt[j, d] = T.if_then_else(key < total, k_cache[key, h * D + d], 0)
                    vt[j, d] = T.if_then_else(key < total, v_cache[key, h * D + d], 0)
                T.clear(s)
                T.gemm(q, kt, s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # 在以 2 为底的指数域中计算 softmax. 越界的 query 行沿用最后一个有效行的掩码.
                for i, j in T.Parallel(BM, BN):
                    visible = pos0 + T.min(qt * BM + i, m - 1)
                    s[i, j] = T.if_then_else(
                        t * BN + j <= visible, s[i, j] * (scale * LOG2_E), -1e30
                    )
                T.copy(maximum, previous)
                T.reduce_max(s, maximum, dim=1, clear=False)
                for i in T.Parallel(BM):
                    rescale[i] = T.exp2(previous[i] - maximum[i])
                for i, j in T.Parallel(BM, BN):
                    visible = pos0 + T.min(qt * BM + i, m - 1)
                    s[i, j] = T.if_then_else(t * BN + j <= visible, T.exp2(s[i, j] - maximum[i]), 0)
                T.reduce_sum(s, tile_sum, dim=1)
                for i in T.Parallel(BM):
                    denominator[i] = denominator[i] * rescale[i] + tile_sum[i]
                for i, d in T.Parallel(BM, D):
                    acc[i, d] *= rescale[i]
                T.copy(s, p)
                T.gemm(p, vt, acc, policy=T.GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(BM, D):
                if qt * BM + i < m:
                    y[qt * BM + i, h * D + d] = T.cast(acc[i, d] / denominator[i], F16)

    return kernel


FACTORIES = {
    "gpt2_embedding": embedding,
    "gpt2_layernorm": layernorm,
    "gpt2_linear_bias": lambda: linear("bias"),
    "gpt2_linear_bias_gelu": lambda: linear("gelu"),
    "gpt2_linear_bias_residual": lambda: linear("residual"),
    "gpt2_lm_head": lm_head,
    "gpt2_attention": attention,
}


def lower(name: str, arch: str) -> tuple[str, int, int]:
    """返回 TileLang 生成的 CUDA 源码, 动态共享内存字节数和每个 block 的线程数."""
    func = FACTORIES[name]()
    target = tvm.target.Target(json.dumps({"kind": "cuda", "arch": arch}))
    config = normalize_pass_configs(PASS_CONFIGS | EXTRA_PASS_CONFIGS.get(name, {}))
    with tvm.transform.PassContext(opt_level=3, config=config), target:
        artifact = tilelang.lower(func, target=target)
    functions = list(artifact.device_mod.functions.values())
    if len(functions) != 1:
        raise ValueError(f"预期 1 个设备函数, 实际 {len(functions)} 个")
    attrs = functions[0].attrs
    smem = int(attrs["dyn_shared_memory_buf"]) if "dyn_shared_memory_buf" in attrs else 0
    # grid 的维度可能是运行时表达式, 线程数总是常量.
    threads = math.prod(
        int(extent)
        for tag, extent in attrs["thread_extent"].items()
        if str(tag).startswith("threadIdx")
    )
    return artifact.kernel_source, smem, threads


def add_abi_entry(kernel: abi.Kernel, source: str) -> str:
    """把 TileLang 的入口改为 ``__device__`` 函数, 再加一个按 ABI 排列参数的入口."""
    definition = re.compile(
        r'extern "C" __global__ void __launch_bounds__\((\d+), 1\) (\w+)\(([^)]*)\) \{'
    )
    matches = definition.findall(source)
    if len(matches) != 1:
        raise ValueError(f"{kernel.name}: 预期 1 个 kernel 定义, 实际 {len(matches)} 个")
    threads, name, params = matches[0]
    if int(threads) != THREADS:
        raise ValueError(f"{kernel.name}: launch bounds 为 {threads}, 预期 {THREADS}")
    impl_params = [p.split()[-1] for p in params.split(",")]
    unknown = set(impl_params) - {p.name for p in kernel.params}
    if unknown:
        raise ValueError(f"{kernel.name}: TileLang 生成了 ABI 之外的参数 {sorted(unknown)}")

    impl = f"{kernel.name}_impl"
    source = re.sub(rf'extern "C" __global__ void {name}\([^)]*\);\n', "", source)
    source = definition.sub(
        f"static __device__ __forceinline__ void {impl}({params}) {{", source, count=1
    )
    abi_params = ", ".join(f"{abi.CUDA_TYPES[p.type]} {p.name}" for p in kernel.params)
    return (
        source
        + f"""
// ABI 入口, 见 cpp/kernels/kernel_abi.py: {kernel.doc}
// TileLang 生成的函数按名字排列参数; 这里按 ABI 的顺序接收参数, 再转发给它.
extern "C" __global__ void __launch_bounds__({THREADS}, 1) {kernel.symbol}({abi_params}) {{
  {impl}({", ".join(impl_params)});
}}
"""
    )


def compile_cubin(source: Path, cubin: Path, arch: str) -> None:
    command = [
        get_nvcc_compiler(),
        "--cubin",
        "-O3",
        f"-arch={arch}",
        "-std=c++20",
        # TileLang 的头文件会触发大量与本项目无关的弃用警告.
        "-w",
        f"-I{TILELANG_TEMPLATE_PATH}",
        f"-I{CUTLASS_INCLUDE_DIR}",
        "-o",
        str(cubin),
        str(source),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc 编译 {source} 失败:\n{result.stdout}{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="把 TileLang kernel 导出为 cubin")
    parser.add_argument("--arch", required=True, help="GPU 架构, 例如 sm_120a 或 sm_86")
    parser.add_argument("--out", type=Path, required=True, help="输出目录")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    smem: dict[str, int] = {}
    for kernel in abi.KERNELS:
        source, smem[kernel.name], threads = lower(kernel.name, args.arch)
        if threads != THREADS:
            raise ValueError(f"{kernel.name}: 每个 block {threads} 个线程, 预期 {THREADS}")
        (args.out / f"{kernel.name}.cu").write_text(add_abi_entry(kernel, source))

    # nvcc 编译较慢, 并行执行.
    with ThreadPoolExecutor() as pool:
        jobs = [
            pool.submit(
                compile_cubin,
                args.out / f"{k.name}.cu",
                args.out / f"{k.name}.cubin",
                args.arch,
            )
            for k in abi.KERNELS
        ]
        for job in jobs:
            job.result()

    abi.write_manifest(args.out, "tilelang", args.arch, smem)
    for kernel in abi.KERNELS:
        print(f"tilelang {kernel.symbol:32s} smem {smem[kernel.name]:6d} B")


if __name__ == "__main__":
    main()
