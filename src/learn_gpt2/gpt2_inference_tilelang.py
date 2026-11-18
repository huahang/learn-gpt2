"""使用 safetensors 权重和 TileLang 算子的 GPT-2 FP16 推理。

参考 ``gpt2.py``，权重默认采用 Hugging Face Conv1D 的 (in, out) 布局。
模型计算使用 TileLang；PyTorch 仅负责张量、设备、CUDA Graph 和随机采样。
需要支持 FP16 Tensor Core 的 NVIDIA GPU、TileLang >= 0.1.15 和 CUDA 编译工具链。

优化策略：

* 加载时转置矩阵，使 GEMM 按连续的 K 维读取权重；LM head 共享词嵌入。
* FP16 输入配合 FP32 累加；LayerNorm 和在线 softmax 的归约也使用 FP32。
* QKV 合并为一个 GEMM，直接写入按 head 排列的 KV 缓存，不再转置或复制。
* FlashAttention 用两个 GEMM 计算 QK^T 和 PV，不保存完整的注意力矩阵。K/V 用异步复制并分两级
  流水；P 留在寄存器中，直接作为第二个 GEMM 的输入；因果掩码写入分数的初值。
* 解码时按实际长度把 key 块分给多个 split，再合并在线 softmax 的结果，提高小 batch 的并行度。
* GEMM 使用共享内存、软件流水线和 L2 swizzle。块大小按矩阵形状和 SM 数选择，
  预填充最大用 128 x 128 的块和 8 个 warp。
* 在 GEMM 的后处理中融合 bias、GELU 或 residual；融合两种嵌入的读取和相加。
* 预分配工作区和输出序列；生成时只计算最后一个位置的 logits。
* SM90 及以上用程序化依赖启动 (PDL)，下一个内核在本内核运行时完成启动。
* 将缓存位置放在 GPU 上，捕获并复用固定形状的单步解码 CUDA Graph。

LayerNorm 单独运行，保持与 ``gpt2.py`` 相同的 FP16 舍入点。块大小在 RTX PRO 4000 Blackwell
上测得；其他 GPU 需要重新测量。工作区由实例复用，同一实例不能并发处理多个请求。
"""

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import tilelang
import tilelang.language as T
import torch
from safetensors.torch import load_file

from learn_gpt2.gpt2 import GPT2_CONFIGS, GPT2Config

LAYER_NORM_EPS = 1e-5
_MATRIX_SUFFIXES = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


@T.macro
def _wait_for_previous_kernel() -> None:
    """程序化依赖启动 (PDL)：等前一个内核完成并写回结果，再允许下一个内核提前启动。

    下一个内核因此能在本内核运行时完成启动和 CTA 调度，只在读取数据前等待。
    PDL 需要 SM90 或更新的 GPU；内核工厂的 ``pdl`` 参数控制是否插入这一步。
    """
    T.evaluate(T.pdl_sync())
    T.evaluate(T.pdl_trigger())


# 返回 PrimFunc 而不在定义时编译，便于在无 GPU 的环境中检查生成的 IR。
# 每种形状只编译一次；所有输出都由调用者预分配。
def _embedding_kernel(
    batch: int, length: int, channels: int, vocab: int, context: int, pdl: bool = False
) -> Any:
    @T.prim_func
    def main(
        ids: T.Tensor((batch * length,), "int64"),
        wte: T.Tensor((vocab, channels), "float16"),
        wpe: T.Tensor((context, channels), "float16"),
        position: T.Tensor((1,), "int32"),
        output: T.Tensor((batch * length, channels), "float16"),
    ):
        with T.Kernel(batch * length, threads=128) as row:
            if pdl:
                _wait_for_previous_kernel()
            for c in T.Parallel(channels):
                output[row, c] = T.cast(wte[ids[row], c], "float32") + T.cast(
                    wpe[position[0] + row % length, c], "float32"
                )

    return main


def _layernorm_kernel(
    batch: int, length: int, channels: int, last_only: bool, pdl: bool = False
) -> Any:
    rows = batch if last_only else batch * length
    block = _power_of_two(channels)

    @T.prim_func
    def main(
        x: T.Tensor((batch * length, channels), "float16"),
        gain: T.Tensor((channels,), "float16"),
        bias: T.Tensor((channels,), "float16"),
        output: T.Tensor((rows, channels), "float16"),
    ):
        with T.Kernel(rows, threads=128) as row:
            if pdl:
                _wait_for_previous_kernel()
            values = T.alloc_fragment((block,), "float32")
            centered = T.alloc_fragment((block,), "float32")
            squares = T.alloc_fragment((block,), "float32")
            total = T.alloc_fragment((1,), "float32")
            variance = T.alloc_fragment((1,), "float32")
            source = row * length + length - 1 if last_only else row
            for c in T.Parallel(block):
                values[c] = T.if_then_else(c < channels, T.cast(x[source, c], "float32"), 0)
            T.reduce_sum(values, total, dim=0)
            for c in T.Parallel(block):
                centered[c] = T.if_then_else(c < channels, values[c] - total[0] / channels, 0)
                squares[c] = centered[c] * centered[c]
            T.reduce_sum(squares, variance, dim=0)
            # 与 centered 的形状一致，再跳过补齐的通道。
            for c in T.Parallel(block):
                if c < channels:
                    output[row, c] = centered[c] * T.rsqrt(
                        variance[0] / channels + LAYER_NORM_EPS
                    ) * T.cast(gain[c], "float32") + T.cast(bias[c], "float32")

    return main


# 块大小在 RTX PRO 4000 Blackwell (SM120，70 个 SM) 上用 GPT-2 small 测得。
# 元素为 (bm, bn, bk, stages, threads)。
# 解码时一个 16 行的块容纳整个 batch，每个权重只读一次，速度受显存带宽限制。
# N 较小时用窄块，使 CTA 数至少达到 SM 数的一半。
_DECODE_TILES = ((16, 64, 64, 4, 128), (16, 32, 128, 4, 64))
# 预填充受算力限制。块越大，每次读取的数据参与的乘加越多，但块数要足够填满 SM。
# 候选块从大到小排列；128 x 256 只在块数很多时使用。
_PROMPT_TILES = (
    (128, 128, 64, 3, 256),
    (64, 128, 64, 3, 256),
    (128, 64, 64, 3, 128),
    (64, 64, 64, 4, 128),
    (32, 64, 64, 4, 128),
    (16, 64, 64, 4, 128),
)
_WIDE_PROMPT_TILE = (128, 256, 32, 3, 256)
# 离线编译检查没有 GPU，使用这个 SM 数选择块大小。
_DEFAULT_SM_COUNT = 64


def _gemm_tiling(rows: int, n: int, sm_count: int) -> tuple[int, int, int, int, int]:
    if rows <= 16:
        wide, narrow = _DECODE_TILES
        return wide if n // wide[1] >= sm_count // 2 else narrow
    if _ceildiv(rows, 128) * _ceildiv(n, 256) >= 4 * sm_count:
        return _WIDE_PROMPT_TILE
    fitting = [tile for tile in _PROMPT_TILES if tile[0] <= _power_of_two(rows)]
    for tile in fitting:
        ctas = _ceildiv(rows, tile[0]) * _ceildiv(n, tile[1])
        # 选择能让一半 SM 分到 CTA 的最大块。需要 1 到 2 波的块不选：第二波只占用少数 SM。
        if ctas >= sm_count / 2 and not sm_count < ctas < 2 * sm_count:
            return tile
    return fitting[-1]


def _linear_kernel(
    rows: int,
    n: int,
    k: int,
    *,
    has_bias: bool = True,
    gelu: bool = False,
    residual: bool = False,
    logits: bool = False,
    sm_count: int = _DEFAULT_SM_COUNT,
    pdl: bool = False,
) -> Any:
    bm, bn, bk, stages, threads = _gemm_tiling(rows, n, sm_count)
    output_dtype = "float32" if logits else "float16"

    @T.prim_func
    def main(
        x: T.Tensor((rows, k), "float16"),
        weight: T.Tensor((n, k), "float16"),
        bias: T.Tensor((n,), "float16"),
        output: T.Tensor((rows, n), output_dtype),
    ):
        with T.Kernel(T.ceildiv(n, bn), T.ceildiv(rows, bm), threads=threads) as (col, row):
            if pdl:
                _wait_for_previous_kernel()
            a = T.alloc_shared((bm, bk), "float16")
            b = T.alloc_shared((bn, bk), "float16")
            acc = T.alloc_fragment((bm, bn), "float32")
            T.use_swizzle(panel_size=8)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                # T.copy 的边界合法化会为尾块补零。存储同样受形状边界保护。
                T.copy(x[row * bm, ko * bk], a)
                T.copy(weight[col * bn, ko * bk], b)
                T.gemm(a, b, acc, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                if has_bias:
                    acc[i, j] += T.cast(bias[col * bn + j], "float32")
                if gelu:
                    # GPT-2 使用 tanh 近似；sigmoid(2z) = (1 + tanh(z)) / 2。
                    acc[i, j] = T.cast(T.cast(acc[i, j], "float16"), "float32")
                    acc[i, j] = acc[i, j] / (
                        1
                        + T.exp(
                            -1.5957691216057308
                            * (acc[i, j] + 0.044715 * acc[i, j] * acc[i, j] * acc[i, j])
                        )
                    )
                if residual:
                    # 先舍入投影，再相加，保持 gpt2.py 的 FP16 运算边界。
                    acc[i, j] = T.cast(T.cast(acc[i, j], "float16"), "float32") + T.cast(
                        output[row * bm + i, col * bn + j], "float32"
                    )
                output[row * bm + i, col * bn + j] = acc[i, j]

    return main


def _qkv_kernel(
    config: GPT2Config,
    batch: int,
    length: int,
    capacity: int,
    sm_count: int = _DEFAULT_SM_COUNT,
    pdl: bool = False,
) -> Any:
    c, heads = config.n_embd, config.n_head
    dim, slots = c // heads, 1 + 2 * config.n_layer
    rows = batch * length
    bm, bn, bk, stages, threads = _gemm_tiling(rows, 3 * c, sm_count)

    @T.prim_func
    def main(
        x: T.Tensor((rows, c), "float16"),
        weight: T.Tensor((3 * c, c), "float16"),
        bias: T.Tensor((3 * c,), "float16"),
        arena: T.Tensor((slots, batch, heads, capacity, dim), "float16"),
        position: T.Tensor((1,), "int32"),
        kv_slot: T.int32,
    ):
        with T.Kernel(T.ceildiv(3 * c, bn), T.ceildiv(rows, bm), threads=threads) as (
            col,
            row,
        ):
            if pdl:
                _wait_for_previous_kernel()
            a = T.alloc_shared((bm, bk), "float16")
            b = T.alloc_shared((bn, bk), "float16")
            acc = T.alloc_fragment((bm, bn), "float32")
            T.use_swizzle(panel_size=8)
            T.clear(acc)
            for ko in T.Pipelined(T.ceildiv(c, bk), num_stages=stages):
                T.copy(x[row * bm, ko * bk], a)
                T.copy(weight[col * bn, ko * bk], b)
                T.gemm(a, b, acc, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                m, n = row * bm + i, col * bn + j
                if m < rows and n < 3 * c:
                    part = n // c
                    slot = T.if_then_else(part == 0, 0, kv_slot + part - 1)
                    arena[slot, m // length, n % c // dim, position[0] + m % length, n % dim] = acc[
                        i, j
                    ] + T.cast(bias[n], "float32")

    return main


def _attention_tiling(length: int) -> tuple[int, int, int, int]:
    """返回注意力的 (bm, bn, threads, stages)。

    每个 warp 负责 16 个完整的 query 行，P 才能留在寄存器中作为第二个 GEMM 的输入。
    解码只有一个 query，所以只用一个 warp。长提示词用 128 行的块，每次读取的 K/V
    供更多 query 使用；短提示词用 64 行的块，块数更多。
    """
    if length == 1:
        return (16, 64, 32, 2)
    return (128, 128, 256, 2) if length >= 256 else (64, 64, 128, 2)


def _attention_kernel(
    config: GPT2Config, batch: int, length: int, capacity: int, splits: int, pdl: bool = False
) -> Any:
    heads, dim = config.n_head, config.n_embd // config.n_head
    slots, bd = 1 + 2 * config.n_layer, max(16, _power_of_two(dim))
    bm, bn, threads, stages = _attention_tiling(length)
    if dim != bd:
        # 补齐的 head 维度用逐元素加载，不能与计算流水重叠。
        stages = 1
    split_kv = length == 1 and splits > 1
    partitions = splits if split_kv else 1
    # Query 权重已乘入 1/sqrt(dim)，这里仅转换为 exp2 的底数。
    log2e = math.log2(math.e)

    @T.prim_func
    def main(
        arena: T.Tensor((slots, batch, heads, capacity, dim), "float16"),
        position: T.Tensor((1,), "int32"),
        output: T.Tensor((batch * length, config.n_embd), "float16"),
        partial: T.Tensor((batch, heads, splits, bd), "float32"),
        lse: T.Tensor((batch, heads, splits), "float32"),
        kv_slot: T.int32,
    ):
        with T.Kernel(T.ceildiv(length, bm), batch * heads, partitions, threads=threads) as (
            query_tile,
            batch_head,
            partition,
        ):
            if pdl:
                _wait_for_previous_kernel()
            q = T.alloc_shared((bm, bd), "float16")
            k = T.alloc_shared((bn, bd), "float16")
            v = T.alloc_shared((bn, bd), "float16")
            scores = T.alloc_fragment((bm, bn), "float32")
            # P 留在寄存器中，直接作为第二个 GEMM 的 A 操作数。
            probability = T.alloc_fragment((bm, bn), "float16")
            acc = T.alloc_fragment((bm, bd), "float32")
            maximum = T.alloc_fragment((bm,), "float32")
            previous = T.alloc_fragment((bm,), "float32")
            rescale = T.alloc_fragment((bm,), "float32")
            denominator = T.alloc_fragment((bm,), "float32")
            tile_sum = T.alloc_fragment((bm,), "float32")
            b, h = batch_head // heads, batch_head % heads
            total = position[0] + length
            first = position[0] + query_tile * bm
            # 按实际长度把 key 块均分给各个 split，短前缀也能用上所有 split。
            per_split = T.ceildiv(T.ceildiv(total, bn), partitions) * bn
            begin = partition * per_split
            end = T.min(T.min(total, begin + per_split), first + bm)
            # 越界的行和补齐的 head 维度填零。缓存初始化为零，
            # 所以已分配但尚未写入的位置也是有限值，掩码后不会产生 NaN。
            if dim == bd:
                T.copy(arena[0, b, h, first, 0], q)
            else:
                for i, d in T.Parallel(bm, bd):
                    q[i, d] = T.if_then_else(
                        first + i < capacity and d < dim, arena[0, b, h, first + i, d], 0
                    )
            T.clear(acc)
            T.clear(denominator)
            # 有限初值使空的 split 不产生 -inf - -inf。
            T.fill(maximum, -1e30)
            for tile in T.Pipelined(T.max(T.ceildiv(end - begin, bn), 0), num_stages=stages):
                key = begin + tile * bn
                if dim == bd:
                    # head_size 是 2 的幂时用异步复制，由流水线与计算重叠。
                    T.copy(arena[kv_slot, b, h, key, 0], k)
                    T.copy(arena[kv_slot + 1, b, h, key, 0], v)
                else:
                    for j, d in T.Parallel(bn, bd):
                        inside = key + j < capacity and d < dim
                        k[j, d] = T.if_then_else(inside, arena[kv_slot, b, h, key + j, d], 0)
                        v[j, d] = T.if_then_else(inside, arena[kv_slot + 1, b, h, key + j, d], 0)
                # 掩码写入初值，GEMM 在其上累加：被遮住的分数仍为 -1e30。
                # 无效 query 行沿用最后一个真实 query 的掩码，避免全掩码行。
                for i, j in T.Parallel(bm, bn):
                    visible = position[0] + T.min(query_tile * bm + i, length - 1)
                    scores[i, j] = T.if_then_else(key + j <= visible, 0, -1e30)
                T.gemm(q, k, scores, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(maximum, previous)
                T.reduce_max(scores, maximum, dim=1, clear=False)
                for i in T.Parallel(bm):
                    rescale[i] = T.exp2((previous[i] - maximum[i]) * log2e)
                # 每行在第一个块中至少有一个可见 key，此后被遮住的分数换算后为 0。
                for i, j in T.Parallel(bm, bn):
                    scores[i, j] = T.exp2((scores[i, j] - maximum[i]) * log2e)
                T.reduce_sum(scores, tile_sum, dim=1)
                for i in T.Parallel(bm):
                    denominator[i] = denominator[i] * rescale[i] + tile_sum[i]
                for i, d in T.Parallel(bm, bd):
                    acc[i, d] *= rescale[i]
                T.copy(scores, probability)
                T.gemm(probability, v, acc, policy=T.GemmWarpPolicy.FullRow)
            if split_kv:
                # 保留 GEMM 的行布局，不能直接在 fragment 上固定读取第 0 行。
                for i, d in T.Parallel(bm, bd):
                    if i == 0:
                        partial[b, h, partition, d] = acc[i, d] / T.max(denominator[i], 1e-20)
                for i in T.Parallel(bm):
                    if i == 0:
                        lse[b, h, partition] = T.if_then_else(
                            denominator[i] > 0,
                            maximum[i] * log2e + T.log2(T.max(denominator[i], 1e-20)),
                            -T.infinity("float32"),
                        )
            else:
                # 循环必须覆盖整个 fragment：只遍历前 dim 列会让布局推断给出错误的线程映射。
                for i, d in T.Parallel(bm, bd):
                    if query_tile * bm + i < length and d < dim:
                        output[b * length + query_tile * bm + i, h * dim + d] = (
                            acc[i, d] / denominator[i]
                        )

    return main


def _merge_attention_kernel(
    batch: int, heads: int, dim: int, splits: int, pdl: bool = False
) -> Any:
    bd = max(16, _power_of_two(dim))

    @T.prim_func
    def main(
        partial: T.Tensor((batch, heads, splits, bd), "float32"),
        lse: T.Tensor((batch, heads, splits), "float32"),
        output: T.Tensor((batch, heads * dim), "float16"),
    ):
        with T.Kernel(batch, heads, threads=128) as (b, h):
            if pdl:
                _wait_for_previous_kernel()
            logs = T.alloc_fragment((splits,), "float32")
            weights = T.alloc_fragment((splits,), "float32")
            values = T.alloc_fragment((splits, bd), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            total = T.alloc_fragment((1,), "float32")
            result = T.alloc_fragment((bd,), "float32")
            for s in T.Parallel(splits):
                logs[s] = lse[b, h, s]
            T.reduce_max(logs, maximum, dim=0)
            for s in T.Parallel(splits):
                weights[s] = T.exp2(logs[s] - maximum[0])
            T.reduce_sum(weights, total, dim=0)
            for s, d in T.Parallel(splits, bd):
                values[s, d] = weights[s] * partial[b, h, s, d]
            T.reduce_sum(values, result, dim=0)
            for d in T.Parallel(bd):
                if d < dim:
                    output[b, h * dim + d] = result[d] / total[0]

    return main


@lru_cache(maxsize=256)
def _compile(factory: Any, *args: Any, target: str, **kwargs: Any) -> Any:
    return tilelang.compile(factory(*args, **kwargs), target=target)


def _load_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], int | None]:
    """读取单文件或 Hugging Face 分片目录；只访问本地文件。"""
    path = Path(path)
    n_head = None
    directory = path if path.is_dir() else path.parent
    config_path = directory / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        n_head = config.get("n_head", config.get("num_attention_heads"))
    if not path.is_dir():
        return load_file(str(path), device="cpu"), n_head
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        return load_file(str(path / "model.safetensors"), device="cpu"), n_head
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    state_dict: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        shard_path = (path / shard).resolve()
        if not shard_path.is_relative_to(path.resolve()):
            raise ValueError(f"权重分片必须位于检查点目录中：{shard}")
        tensors = load_file(str(shard_path), device="cpu")
        duplicates = state_dict.keys() & tensors.keys()
        if duplicates:
            raise ValueError(f"分片中存在重复权重：{sorted(duplicates)}")
        for key in tensors:
            if weight_map.get(key) != shard:
                raise ValueError(f"分片索引与权重不一致：{key}")
        state_dict.update(tensors)
    if state_dict.keys() != weight_map.keys():
        raise ValueError("分片索引中的部分权重缺失")
    return state_dict, n_head


def _prepare_state_dict(
    state_dict: dict[str, torch.Tensor], n_head: int | None, weight_layout: str
) -> tuple[GPT2Config, dict[str, torch.Tensor]]:
    """验证键名和形状。此函数不需要 GPU，也不会修改传入的权重。"""
    if weight_layout not in ("hf", "pytorch"):
        raise ValueError("weight_layout 必须是 'hf' 或 'pytorch'")
    sd: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if name.endswith((".attn.bias", ".attn.masked_bias")) or name == "lm_head.weight":
            continue
        key = name.removeprefix("transformer.")
        if key in sd:
            raise ValueError(f"重复权重：{key}")
        sd[key] = tensor
    for key in ("wte.weight", "wpe.weight"):
        if key not in sd or sd[key].ndim != 2:
            raise ValueError(f"检查点缺少二维权重 {key}")
    vocab, c = sd["wte.weight"].shape
    context = sd["wpe.weight"].shape[0]
    layer_ids = {key.split(".")[1] for key in sd if key.startswith("h.")}
    layers = len(layer_ids)
    if not layers or layer_ids != {str(i) for i in range(layers)}:
        raise ValueError("Transformer 层编号必须从 0 开始并且连续")
    if n_head is None:
        candidates = {
            cfg.n_head for cfg in GPT2_CONFIGS.values() if (cfg.n_layer, cfg.n_embd) == (layers, c)
        }
        if len(candidates) != 1:
            raise ValueError("无法从权重推断 n_head；请显式传入 n_head")
        n_head = candidates.pop()
    if isinstance(n_head, bool) or not isinstance(n_head, int) or n_head <= 0 or c % n_head:
        raise ValueError("n_head 必须为正整数，并且能整除 n_embd")
    if min(vocab, context, c) <= 0:
        raise ValueError("词表、上下文和隐藏维度必须大于 0")
    config = GPT2Config(context, vocab, layers, n_head, c)
    shapes = {
        "wte.weight": (vocab, c),
        "wpe.weight": (context, c),
        "ln_f.weight": (c,),
        "ln_f.bias": (c,),
    }
    layer_shapes = {
        "ln_1.weight": (c,),
        "ln_1.bias": (c,),
        "attn.c_attn.weight": (c, 3 * c),
        "attn.c_attn.bias": (3 * c,),
        "attn.c_proj.weight": (c, c),
        "attn.c_proj.bias": (c,),
        "ln_2.weight": (c,),
        "ln_2.bias": (c,),
        "mlp.c_fc.weight": (c, 4 * c),
        "mlp.c_fc.bias": (4 * c,),
        "mlp.c_proj.weight": (4 * c, c),
        "mlp.c_proj.bias": (c,),
    }
    for i in range(layers):
        for name, shape in layer_shapes.items():
            if weight_layout == "pytorch" and name.endswith(_MATRIX_SUFFIXES):
                shape = shape[::-1]
            shapes[f"h.{i}.{name}"] = shape
    missing, extra = shapes.keys() - sd.keys(), sd.keys() - shapes.keys()
    if missing or extra:
        raise ValueError(f"权重键不匹配：missing={sorted(missing)}, unexpected={sorted(extra)}")
    for key, shape in shapes.items():
        if tuple(sd[key].shape) != shape:
            raise ValueError(f"权重形状不匹配 {key}：{tuple(sd[key].shape)}，预期 {shape}")
        if not sd[key].is_floating_point():
            raise ValueError(f"权重必须是浮点张量：{key}")
    return config, sd


@dataclass(frozen=True)
class _LayerWeights:
    ln_1: tuple[torch.Tensor, torch.Tensor]
    qkv_w: torch.Tensor
    qkv_b: torch.Tensor
    attn_w: torch.Tensor
    attn_b: torch.Tensor
    ln_2: tuple[torch.Tensor, torch.Tensor]
    fc_w: torch.Tensor
    fc_b: torch.Tensor
    proj_w: torch.Tensor
    proj_b: torch.Tensor


class _Workspace:
    def __init__(
        self,
        config: GPT2Config,
        batch: int,
        capacity: int,
        device: torch.device,
        sm_count: int = _DEFAULT_SM_COUNT,
    ):
        c, dim = config.n_embd, config.n_embd // config.n_head
        rows = batch * capacity
        self.batch = batch
        self.capacity = capacity
        # 解码时每个 (序列, head) 拆成若干 split，使每个 SM 约有一个 CTA。
        # 拆得更细时每个 split 只有一两个 key 块，固定开销反而更多。
        key_blocks = _ceildiv(capacity, _attention_tiling(1)[1])
        self.splits = max(1, min(key_blocks, _ceildiv(sm_count, batch * config.n_head)))
        self.position = torch.zeros(1, dtype=torch.int32, device=device)
        self.tokens = torch.empty(batch, dtype=torch.int64, device=device)
        self.x = torch.empty(rows, c, dtype=torch.float16, device=device)
        self.norm = torch.empty_like(self.x)
        self.attention = torch.empty_like(self.x)
        self.hidden = torch.empty(rows, 4 * c, dtype=torch.float16, device=device)
        # 注意力按整块读取缓存，未写入的位置必须是有限值，所以用零初始化。
        self.arena = torch.zeros(
            1 + 2 * config.n_layer,
            batch,
            config.n_head,
            capacity,
            dim,
            dtype=torch.float16,
            device=device,
        )
        self.partial = torch.empty(
            batch,
            config.n_head,
            self.splits,
            max(16, _power_of_two(dim)),
            dtype=torch.float32,
            device=device,
        )
        self.lse = torch.empty(
            batch, config.n_head, self.splits, dtype=torch.float32, device=device
        )
        self.logits = torch.empty(batch, config.vocab_size, dtype=torch.float32, device=device)
        self.graph: torch.cuda.CUDAGraph | None = None


class GPT2Inference:
    """仅用于 FP16 推理的 GPT-2，不构建 nn.Module，也不执行自动微分。

    ``weight_layout='hf'`` 用于 HF 的 Conv1D 权重；``'pytorch'`` 用于本仓库
    ``GPT2.state_dict()`` 的 Linear 权重。两种布局必须显式区分，因为 c_proj 是方阵。
    每次 ``forward`` 都重置缓存；``generate`` 在内部维护缓存并支持滑动窗口重算。
    """

    @torch.no_grad()
    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        n_head: int | None = None,
        device: torch.device | str = "cuda",
        weight_layout: str = "hf",
        cuda_graph: bool = True,
    ):
        config, sd = _prepare_state_dict(state_dict, n_head, weight_layout)
        device = torch.device(device)
        if device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("此实现需要 NVIDIA CUDA GPU")
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到可用的 CUDA GPU")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        major, minor = torch.cuda.get_device_capability(device)
        if (major, minor) < (7, 5):
            raise ValueError("此实现需要计算能力 >= 7.5 的 FP16 Tensor Core GPU")
        # 显式绑定设备目标，避免多 GPU 环境按当前设备误编译。
        # Hopper 及后续架构的专用指令需要带 a 的目标，例如 sm_90a。
        suffix = "a" if major >= 9 else ""
        self.target = json.dumps({"kind": "cuda", "arch": f"sm_{major}{minor}{suffix}"})
        self.sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        # 程序化依赖启动 (PDL) 需要 SM90 或更新的 GPU。
        self.pdl = major >= 9
        self.config = config
        self.device = device
        self.dtype = torch.float16
        self.cuda_graph = cuda_graph

        def weight(key: str) -> torch.Tensor:
            return sd[key].to(device=device, dtype=torch.float16).contiguous()

        def matrix(key: str) -> torch.Tensor:
            tensor = sd[key].t() if weight_layout == "hf" else sd[key]
            return tensor.to(device=device, dtype=torch.float16).contiguous()

        self.wte = weight("wte.weight")
        self.wpe = weight("wpe.weight")
        self.ln_f = (weight("ln_f.weight"), weight("ln_f.bias"))
        # 无 bias 的 LM head 仍传入有效形状的占位指针，内核不读取它。
        self._head_bias = torch.empty(config.vocab_size, dtype=torch.float16, device=device)
        self.layers = []
        c, dim = config.n_embd, config.n_embd // config.n_head
        for i in range(config.n_layer):
            prefix = f"h.{i}."
            # 在 FP32 中折叠缩放再转 FP16。clone 保证不修改调用者持有的 state_dict。
            qkv = sd[prefix + "attn.c_attn.weight"]
            qkv = (qkv.t() if weight_layout == "hf" else qkv).float().clone()
            bias = sd[prefix + "attn.c_attn.bias"].float().clone()
            qkv[:c] *= 1 / math.sqrt(dim)
            bias[:c] *= 1 / math.sqrt(dim)
            self.layers.append(
                _LayerWeights(
                    ln_1=(weight(prefix + "ln_1.weight"), weight(prefix + "ln_1.bias")),
                    qkv_w=qkv.to(device=device, dtype=torch.float16).contiguous(),
                    qkv_b=bias.to(device=device, dtype=torch.float16).contiguous(),
                    attn_w=matrix(prefix + "attn.c_proj.weight"),
                    attn_b=weight(prefix + "attn.c_proj.bias"),
                    ln_2=(weight(prefix + "ln_2.weight"), weight(prefix + "ln_2.bias")),
                    fc_w=matrix(prefix + "mlp.c_fc.weight"),
                    fc_b=weight(prefix + "mlp.c_fc.bias"),
                    proj_w=matrix(prefix + "mlp.c_proj.weight"),
                    proj_b=weight(prefix + "mlp.c_proj.bias"),
                )
            )
        self._ws: _Workspace | None = None

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        n_head: int | None = None,
        device: torch.device | str = "cuda",
        weight_layout: str = "hf",
        cuda_graph: bool = True,
    ) -> GPT2Inference:
        """从单个 safetensors 或含分片索引的 HF 目录读取权重。

        优先使用显式 n_head，其次读取同目录的 config.json，再匹配四种标准 GPT-2。
        safetensors 在 CPU 上映射；整理权重后只保留 GPU 上的 FP16 副本。
        """
        state_dict, stored_heads = _load_checkpoint(path)
        return cls(
            state_dict,
            n_head=stored_heads if n_head is None else n_head,
            device=device,
            weight_layout=weight_layout,
            cuda_graph=cuda_graph,
        )

    def _workspace(self, batch: int, length: int) -> _Workspace:
        ws = self._ws
        if ws is None or ws.batch != batch or ws.capacity < length:
            self._ws = None
            ws = None
            capacity = min(self.config.block_size, max(64, _power_of_two(length)))
            ws = _Workspace(self.config, batch, capacity, self.device, self.sm_count)
            self._ws = ws
        return ws

    def _norm(
        self,
        ws: _Workspace,
        length: int,
        weights: tuple[torch.Tensor, torch.Tensor],
        *,
        last_only: bool = False,
    ) -> torch.Tensor:
        rows = ws.batch if last_only else ws.batch * length
        output = ws.norm[:rows]
        _compile(
            _layernorm_kernel,
            ws.batch,
            length,
            self.config.n_embd,
            last_only,
            pdl=self.pdl,
            target=self.target,
        )(ws.x[: ws.batch * length], *weights, output)
        return output

    def _linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        output: torch.Tensor,
        *,
        gelu: bool = False,
        residual: bool = False,
        logits: bool = False,
    ) -> None:
        n, k = weight.shape
        _compile(
            _linear_kernel,
            x.shape[0],
            n,
            k,
            has_bias=not logits,
            gelu=gelu,
            residual=residual,
            logits=logits,
            sm_count=self.sm_count,
            pdl=self.pdl,
            target=self.target,
        )(x, weight, bias, output)

    def _pass(
        self,
        ws: _Workspace,
        tokens: torch.Tensor,
        length: int,
        logits: torch.Tensor,
        *,
        last_only: bool,
    ) -> None:
        rows = ws.batch * length
        x, attention, hidden = ws.x[:rows], ws.attention[:rows], ws.hidden[:rows]
        _compile(
            _embedding_kernel,
            ws.batch,
            length,
            self.config.n_embd,
            self.config.vocab_size,
            self.config.block_size,
            pdl=self.pdl,
            target=self.target,
        )(tokens, self.wte, self.wpe, ws.position, x)
        qkv = _compile(
            _qkv_kernel,
            self.config,
            ws.batch,
            length,
            ws.capacity,
            sm_count=self.sm_count,
            pdl=self.pdl,
            target=self.target,
        )
        attend = _compile(
            _attention_kernel,
            self.config,
            ws.batch,
            length,
            ws.capacity,
            ws.splits,
            pdl=self.pdl,
            target=self.target,
        )
        for i, layer in enumerate(self.layers):
            slot = 1 + 2 * i
            norm = self._norm(ws, length, layer.ln_1)
            qkv(norm, layer.qkv_w, layer.qkv_b, ws.arena, ws.position, slot)
            attend(ws.arena, ws.position, attention, ws.partial, ws.lse, slot)
            if length == 1 and ws.splits > 1:
                _compile(
                    _merge_attention_kernel,
                    ws.batch,
                    self.config.n_head,
                    self.config.n_embd // self.config.n_head,
                    ws.splits,
                    pdl=self.pdl,
                    target=self.target,
                )(ws.partial, ws.lse, attention)
            self._linear(attention, layer.attn_w, layer.attn_b, x, residual=True)
            norm = self._norm(ws, length, layer.ln_2)
            self._linear(norm, layer.fc_w, layer.fc_b, hidden, gelu=True)
            self._linear(hidden, layer.proj_w, layer.proj_b, x, residual=True)
        norm = self._norm(ws, length, self.ln_f, last_only=last_only)
        self._linear(norm, self.wte, self._head_bias, logits, logits=True)

    def _prefill(self, ws: _Workspace, tokens: torch.Tensor) -> torch.Tensor:
        ws.position.zero_()
        self._pass(ws, tokens.reshape(-1), tokens.shape[1], ws.logits, last_only=True)
        ws.position.fill_(tokens.shape[1])
        return ws.logits

    def _decode_step(self, ws: _Workspace) -> None:
        self._pass(ws, ws.tokens, 1, ws.logits, last_only=True)
        ws.position.add_(1)

    def _decode(self, ws: _Workspace, tokens: torch.Tensor) -> torch.Tensor:
        ws.tokens.copy_(tokens)
        if ws.graph is not None:
            ws.graph.replay()
        else:
            # 先执行一次以编译所有内核；捕获期间仅记录相同的启动，不再执行。
            self._decode_step(ws)
            if self.cuda_graph:
                ws.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(ws.graph):
                    self._decode_step(ws)
        return ws.logits

    def _check_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or min(tokens.shape) <= 0:
            raise ValueError("tokens 必须为非空的 (batch, length) 张量")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("token id 必须使用 int32 或 int64")
        # 每个请求只校验一次；生成循环中不会读取 GPU 标量。
        if tokens.min().item() < 0 or tokens.max().item() >= self.config.vocab_size:
            raise ValueError(f"token id 必须位于 [0, {self.config.vocab_size})")
        return tokens.to(device=self.device, dtype=torch.int64).contiguous()

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """返回 (batch, length, vocab) 的 FP32 logits；中间计算和权重均为 FP16。"""
        tokens = self._check_tokens(tokens)
        batch, length = tokens.shape
        if length > self.config.block_size:
            raise ValueError(f"输入长度 {length} 超出上下文长度 {self.config.block_size}")
        with torch.cuda.device(self.device):
            ws = self._workspace(batch, length)
            logits = torch.empty(
                batch * length, self.config.vocab_size, dtype=torch.float32, device=self.device
            )
            ws.position.zero_()
            self._pass(ws, tokens.reshape(-1), length, logits, last_only=False)
            ws.position.fill_(length)
            return logits.view(batch, length, -1)

    __call__ = forward

    @torch.inference_mode()
    def generate(
        self,
        tokens: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """预填充后复用 KV 缓存。temperature=0 为贪心解码，其他值使用随机采样。

        GPT-2 使用绝对位置嵌入。超出上下文长度后，逐步重算截断后的上下文，
        与 ``GPT2.generate`` 的行为一致。返回的序列包含原始提示词。
        """
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens 必须为非负整数")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须为有限的非负数")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise ValueError("top_k 必须为正整数或 None")
        tokens = self._check_tokens(tokens)
        if max_new_tokens == 0:
            return tokens.clone()
        batch, length = tokens.shape
        context = self.config.block_size
        with torch.cuda.device(self.device):
            ws = self._workspace(batch, min(length + max_new_tokens, context))
            output = torch.empty(
                batch, length + max_new_tokens, dtype=torch.int64, device=self.device
            )
            output[:, :length] = tokens
            for step in range(max_new_tokens):
                end = length + step
                if step == 0 or end > context:
                    logits = self._prefill(ws, output[:, max(0, end - context) : end])
                else:
                    logits = self._decode(ws, output[:, end - 1])
                output[:, end] = _sample(logits, temperature, top_k)
            return output


def _sample(logits: torch.Tensor, temperature: float, top_k: int | None) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None:
        threshold = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -torch.inf)
    return torch.multinomial(logits.softmax(dim=-1), 1).squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="safetensors 文件或 HF 检查点目录")
    parser.add_argument("--prompt", default="Hello, I am a language model,")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-layout", choices=("hf", "pytorch"), default="hf")
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = tokenizer.encode(args.prompt)
    if not tokens:
        tokens = [tokenizer.eot_token]
    model = GPT2Inference.from_safetensors(
        args.checkpoint,
        n_head=args.n_head,
        device=args.device,
        weight_layout=args.weight_layout,
        cuda_graph=args.cuda_graph,
    )
    output = model.generate(
        torch.tensor([tokens], dtype=torch.int64),
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
