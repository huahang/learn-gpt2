"""用 NumPy 演示因果自注意力、逐层无泄漏与增量 KV Cache。

运行：uv run python presentations/03-gpt2-gpt3-overview/causal_attention_demo.py

不下载权重，不调用模型服务。四个汉字只是教学 token，输入与权重由固定
随机种子生成。演示使用单头、两层 Pre-LN 残差结构和逐位置 GELU MLP；
它用于验证信息流，不是完整 GPT-2，也不产生有意义的语言预测。
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
SEED = 20261115
WIDTH = 4
ATOL = 1e-12


@dataclass(frozen=True)
class AttentionResult:
    """保留中间结果，便于检查 softmax 之前究竟屏蔽了什么。"""

    scores: Array
    mask: Array
    masked_scores: Array
    weights: Array
    output: Array


@dataclass(frozen=True)
class Layer:
    """一个教学用单头 Transformer block 的固定随机权重。"""

    query: Array
    key: Array
    value: Array
    output: Array
    expand: Array
    contract: Array


def row_softmax(scores: Array) -> Array:
    """沿每行归一化；先减最大值以避免指数溢出。"""
    if np.any(np.all(np.isneginf(scores), axis=-1)):
        raise ValueError("每个查询必须至少有一个可见键，不能整行都是负无穷。")
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def attention(
    query: Array,
    key: Array,
    value: Array,
    *,
    causal: bool = True,
    query_start: int = 0,
) -> AttentionResult:
    """严格执行 scale → 加掩码 → 行 softmax → 乘 V。

    行是查询位置，列是键位置。query_start 是当前查询的全局起始位置：
    全序列计算时为 0；第 t 个 token 增量计算时为 t。缓存键从位置 0 开始。
    因此增量计算不能对形状 (1, t+1) 直接套普通左上角 tril，否则会误遮历史。
    """
    scores = (query @ key.T) / np.sqrt(query.shape[-1])
    query_positions = np.arange(query_start, query_start + len(query))[:, None]
    key_positions = np.arange(len(key))[None, :]
    visible = key_positions <= query_positions
    mask = np.where(visible, 0.0, -np.inf) if causal else np.zeros_like(scores)
    masked_scores = scores + mask
    weights = row_softmax(masked_scores)
    output = weights @ value
    return AttentionResult(scores, mask, masked_scores, weights, output)


def layer_norm(x: Array) -> Array:
    """只在最后一维（隐藏特征）归一化，绝不跨 token 混合信息。"""
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance + 1e-5)


def gelu(x: Array) -> Array:
    """常用的 tanh 近似 GELU，每个元素独立计算。"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def make_layer(rng: np.random.Generator) -> Layer:
    """按输入宽度缩放随机权重；并非训练得到的语言模型参数。"""

    def matrix(rows: int, columns: int) -> Array:
        return rng.normal(size=(rows, columns)) / np.sqrt(rows)

    return Layer(
        query=matrix(WIDTH, WIDTH),
        key=matrix(WIDTH, WIDTH),
        value=matrix(WIDTH, WIDTH),
        output=matrix(WIDTH, WIDTH),
        expand=matrix(WIDTH, 4 * WIDTH),
        contract=matrix(4 * WIDTH, WIDTH),
    )


def block(x: Array, layer: Layer, *, causal: bool = True) -> Array:
    """Pre-LN attention + residual，再接逐位置 MLP + residual。"""
    normalized = layer_norm(x)
    result = attention(
        normalized @ layer.query,
        normalized @ layer.key,
        normalized @ layer.value,
        causal=causal,
    )
    hidden = x + result.output @ layer.output
    return hidden + gelu(layer_norm(hidden) @ layer.expand) @ layer.contract


def full_forward(x: Array, layers: list[Layer], *, causal: bool = True) -> Array:
    """一次处理所有 token；各层仍然通过掩码限制信息依赖。"""
    hidden = x
    for layer in layers:
        hidden = block(hidden, layer, causal=causal)
    return hidden


def incremental_forward(x: Array, layers: list[Layer]) -> Array:
    """逐 token 计算，且为每层单独缓存 K/V；旧位置的 Q 不再使用。

    固定权重、固定位置编号且无 dropout。第 t 步每层只处理一个新位置，
    但这个位置读取该层 0..t 的全部缓存 K/V。模型与 full_forward 完全相同。
    """
    cached_keys = [np.empty((0, WIDTH), dtype=np.float64) for _ in layers]
    cached_values = [np.empty((0, WIDTH), dtype=np.float64) for _ in layers]
    outputs = []
    for position in range(len(x)):
        hidden = x[position : position + 1]
        for index, layer in enumerate(layers):
            normalized = layer_norm(hidden)
            query = normalized @ layer.query
            new_key = normalized @ layer.key
            new_value = normalized @ layer.value
            cached_keys[index] = np.concatenate((cached_keys[index], new_key), axis=0)
            cached_values[index] = np.concatenate((cached_values[index], new_value), axis=0)
            result = attention(
                query,
                cached_keys[index],
                cached_values[index],
                query_start=position,
            )
            hidden = hidden + result.output @ layer.output
            hidden = hidden + gelu(layer_norm(hidden) @ layer.expand) @ layer.contract
        outputs.append(hidden)
    return np.concatenate(outputs, axis=0)


def max_difference(left: Array, right: Array) -> float:
    return float(np.max(np.abs(left - right)))


def show_matrix(name: str, values: Array) -> None:
    print(f"\n{name}\n{values}")


def main() -> None:
    np.set_printoptions(precision=4, suppress=True, linewidth=110)
    rng = np.random.default_rng(SEED)
    # 将 token embedding 和固定位置 embedding 合在 x 中；后续绝不重新编号。
    x = rng.normal(size=(4, WIDTH))
    layers = [make_layer(rng), make_layer(rng)]
    first_layer = layers[0]
    normalized = layer_norm(x)
    query = normalized @ first_layer.query
    key = normalized @ first_layer.key
    value = normalized @ first_layer.value
    result = attention(query, key, value)

    print(f"固定随机种子：{SEED}；教学 token：[我, 爱, 学, 习]")
    print("行 = 查询位置；列 = 被读取位置；以下为第一层单头注意力。")
    show_matrix("1. scaled scores = QK^T / sqrt(d_k)", result.scores)
    show_matrix("2. additive causal mask（对角线允许，未来为 -inf）", result.mask)
    show_matrix("3. masked scores = scores + mask", result.masked_scores)
    show_matrix("4. weights = row_softmax(masked scores)", result.weights)
    show_matrix("5. output = weights @ V（尚未输出投影和残差相加）", result.output)
    print("每行权重之和：", result.weights.sum(axis=-1))
    assert np.allclose(result.weights.sum(axis=-1), 1.0, atol=ATOL, rtol=0)
    assert np.all(result.weights[np.triu_indices(len(x), k=1)] == 0)

    # 只改位置 2、3 的输入，位置 0、1 的查询与前缀保持不变。
    changed = x.copy()
    changed[2:] += rng.normal(size=(2, WIDTH)) * 4.0
    changed_normalized = layer_norm(changed)
    changed_query = changed_normalized @ first_layer.query
    changed_key = changed_normalized @ first_layer.key
    changed_value = changed_normalized @ first_layer.value
    changed_causal = attention(changed_query, changed_key, changed_value).output
    original_unmasked = attention(query, key, value, causal=False).output
    changed_unmasked = attention(changed_query, changed_key, changed_value, causal=False).output
    causal_error = max_difference(result.output[:2], changed_causal[:2])
    unmasked_change = max_difference(original_unmasked[:2], changed_unmasked[:2])
    assert np.allclose(result.output[:2], changed_causal[:2], atol=ATOL, rtol=0)
    assert not np.allclose(original_unmasked[:2], changed_unmasked[:2], atol=ATOL, rtol=0)

    original_stack = full_forward(x, layers)
    changed_stack = full_forward(changed, layers)
    stack_error = max_difference(original_stack[:2], changed_stack[:2])
    assert np.allclose(original_stack[:2], changed_stack[:2], atol=ATOL, rtol=0)

    cached_stack = incremental_forward(x, layers)
    cache_error = max_difference(original_stack, cached_stack)
    assert np.allclose(original_stack, cached_stack, atol=ATOL, rtol=0)
    print("\n只修改未来的第 3、4 个 token 后：")
    print(f"  单层因果注意力：前两个位置最大变化 = {causal_error:.3e}")
    print(f"  无掩码注意力：  前两个位置最大变化 = {unmasked_change:.3e}")
    print(f"  两层因果堆叠：  前两个位置最大变化 = {stack_error:.3e}")
    print(f"\n两层 full forward 与逐 token KV cache 最大差异 = {cache_error:.3e}")
    print(f"全部断言通过：allclose 的 atol={ATOL:g}, rtol=0。")
    print("无掩码时一般会改变；特殊的输入或退化权重可能巧合不变。")
    print("这里验证的是确定性因果信息流，不代表对真实语言模型能力的评测。")


if __name__ == "__main__":
    main()
