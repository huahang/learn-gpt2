"""Teaching numbers for causal self-attention and the KV cache.

The vectors are fixed, not trained. They exist so the slides, the review, and this
script all show the same mask, the same softmax row, and the same cache check.
"""

from __future__ import annotations

import math

import numpy as np

TOKENS = ["今天", "天气", "真", "好"]
NEW_TOKEN = "呀"


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Non-finite entries (the masked future) become weight 0."""
    out = np.zeros_like(scores, dtype=np.float64)
    for i, row in enumerate(scores):
        finite = np.isfinite(row)
        if not finite.any():
            continue
        shifted = row[finite] - np.max(row[finite])
        exp = np.exp(shifted)
        out[i, finite] = exp / exp.sum()
    return out


def teaching_qkv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four tokens, one head, d_k = 4. Each query leans toward recent keys."""
    query = np.array(
        [
            [1.2, 0.1, 0.0, 0.0],
            [0.8, 1.1, 0.1, 0.0],
            [0.1, 0.7, 1.2, 0.2],
            [0.0, 0.2, 0.8, 1.3],
        ],
        dtype=np.float64,
    )
    key = np.array(
        [
            [1.1, 0.2, 0.0, 0.1],
            [0.6, 1.2, 0.2, 0.0],
            [0.1, 0.5, 1.1, 0.3],
            [0.0, 0.1, 0.4, 1.2],
        ],
        dtype=np.float64,
    )
    value = np.eye(4, dtype=np.float64)
    return query, key, value


def new_token_qkv() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([0.2, 0.3, 0.4, 1.1], dtype=np.float64),
        np.array([0.1, 0.2, 0.5, 1.0], dtype=np.float64),
        np.array([0.2, 0.0, 0.1, 0.9], dtype=np.float64),
    )


def causal_example() -> dict[str, np.ndarray]:
    query, key, value = teaching_qkv()
    scale = math.sqrt(query.shape[1])
    scores = query @ key.T / scale
    length = scores.shape[0]
    allowed = np.tril(np.ones((length, length), dtype=bool))
    masked = scores.copy()
    masked[~allowed] = -np.inf
    weights = softmax_rows(masked)
    return {
        "scores": scores,
        "masked": masked,
        "weights": weights,
        "context": weights @ value,
        "scale": np.array(scale),
    }


def future_invariance() -> float:
    """Changing a future key must not move earlier causal outputs."""
    query, key, value = teaching_qkv()
    scale = math.sqrt(query.shape[1])

    def outputs(key_matrix: np.ndarray, value_matrix: np.ndarray) -> np.ndarray:
        scores = query @ key_matrix.T / scale
        allowed = np.tril(np.ones_like(scores, dtype=bool))
        scores = scores.copy()
        scores[~allowed] = -np.inf
        return softmax_rows(scores) @ value_matrix

    baseline = outputs(key, value)
    altered_key = key.copy()
    altered_value = value.copy()
    altered_key[3] = np.array([4.0, -3.0, 2.0, -1.0])
    altered_value[3] = np.array([9.0, 9.0, 9.0, 9.0])
    altered = outputs(altered_key, altered_value)
    return float(np.max(np.abs(baseline[:3] - altered[:3])))


def kv_equivalence() -> float:
    """The last row of a full causal pass equals one cached query over all keys."""
    query, key, value = teaching_qkv()
    query_new, key_new, value_new = new_token_qkv()
    scale = math.sqrt(query.shape[1])
    query_all = np.vstack([query, query_new])
    key_all = np.vstack([key, key_new])
    value_all = np.vstack([value, value_new])

    scores = query_all @ key_all.T / scale
    allowed = np.tril(np.ones_like(scores, dtype=bool))
    scores = scores.copy()
    scores[~allowed] = -np.inf
    full_last = softmax_rows(scores)[-1] @ value_all

    cached_scores = (query_new @ key_all.T) / scale
    cached_weights = softmax_rows(cached_scores[None, :])[0]
    cached_last = cached_weights @ value_all
    return float(np.max(np.abs(full_last - cached_last)))


def score_counts(n: int = 1024) -> dict[str, int]:
    """How many attention scores a single head computes while generating n tokens."""
    full = sum(length * length for length in range(1, n + 1))
    triangle = sum(length * (length + 1) // 2 for length in range(1, n + 1))
    cached = sum(range(1, n + 1))
    return {"n": n, "full": full, "triangle": triangle, "cached": cached, "token_forwards": cached}


def kv_bytes(n_layer: int, n_embd: int, length: int, dtype_bytes: int, batch: int = 1) -> int:
    """Bytes for K and V. Head count times head size cancels to n_embd."""
    return n_layer * 2 * batch * length * n_embd * dtype_bytes


def fmt(matrix: np.ndarray, digits: int = 2) -> str:
    rows = []
    for row in matrix:
        cells = []
        for value in row:
            cells.append("  -inf" if not np.isfinite(value) else f"{value:6.{digits}f}")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def main() -> None:
    example = causal_example()
    counts = score_counts()
    print("scale", float(example["scale"]))
    print("scores\n" + fmt(example["scores"]))
    print("weights\n" + fmt(example["weights"]))
    print("row sums", example["weights"].sum(axis=1))
    print("future invariance", future_invariance())
    print("kv equivalence", kv_equivalence())
    print("counts", counts)
    print(
        "gpt2 fp16 MiB",
        kv_bytes(12, 768, 1024, 2) / 1024**2,
    )
    print(
        "gpt3 fp16 GiB",
        kv_bytes(96, 12288, 2048, 2) / 1024**3,
    )


if __name__ == "__main__":
    main()
