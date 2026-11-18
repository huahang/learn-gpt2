"""端到端检查 C++ 推理程序 ``cpp/build/bin/gpt2``.

测试用随机初始化的小模型, 以 Hugging Face 的布局写入 safetensors, 再让 C++ 程序用两个
cubin 后端分别计算 logits 和生成 token, 与 ``learn_gpt2.gpt2.GPT2`` 的 FP32 结果比较.

需要先构建: ``make -C cpp``. 没有构建的部分自动跳过.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from learn_gpt2.gpt2 import GPT2, GPT2Config

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "cpp" / "build"
BIN = BUILD / "bin" / "gpt2"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None or not BIN.is_file(),
    reason="需要 NVIDIA GPU 和 make -C cpp 构建的 C++ 程序",
)

# kernel 需要 n_embd 为 64 的倍数, 每个 head 64 维. 词表不是 64 的倍数, 用于检查补齐;
# 70 个 token 跨过两个 64 行的块; 80 的上下文使生成超过上下文长度.
CONFIG = GPT2Config(block_size=80, vocab_size=300, n_layer=2, n_head=2, n_embd=128)
CONV1D_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
# C++ 程序用 FP16 计算, 参考模型用 FP32.
RTOL, ATOL = 2e-2, 5e-2
# 生成的 token 与 FP32 的选择只可能在 logits 几乎相等时不同.
LOGIT_TOLERANCE = 5e-2


@pytest.fixture(scope="module")
def reference() -> GPT2:
    torch.manual_seed(0)
    model = GPT2(CONFIG)
    # 默认初始化使 logits 几乎均匀, bias 全为零. 放大权重并填充 bias 与 LayerNorm 参数,
    # 使每一项计算都影响结果.
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() == 2:
                param.mul_(5)
            else:
                param.add_(torch.randn_like(param) * 0.2)
        model.transformer.wpe.weight.add_(0.5)
    return model.eval()


@pytest.fixture(scope="module")
def checkpoint(reference: GPT2, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Hugging Face 布局的检查点目录: Conv1D 权重形状为 (输入, 输出), 不保存 lm_head."""
    directory = tmp_path_factory.mktemp("gpt2_tiny")
    state = {
        key: value.t().contiguous() if key.endswith(CONV1D_WEIGHTS) else value.clone()
        for key, value in reference.state_dict().items()
        if key != "lm_head.weight"
    }
    save_file(state, directory / "model.safetensors")
    config = {"n_head": CONFIG.n_head, "layer_norm_epsilon": 1e-5}
    (directory / "config.json").write_text(json.dumps(config))
    return directory


@pytest.fixture(params=["tilelang", "cute_dsl"])
def backend(request: pytest.FixtureRequest) -> str:
    if not (BUILD / "kernels" / request.param / "manifest.txt").is_file():
        pytest.skip(f"没有构建 {request.param} kernel")
    return request.param


def run(model: Path, backend: str, *args: str) -> str:
    result = subprocess.run(
        [str(BIN), "--model", str(model), "--backend", backend, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def tokens_arg(tokens: list[int]) -> str:
    return ",".join(map(str, tokens))


def random_tokens(length: int, seed: int = 1) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, CONFIG.vocab_size, (length,), generator=generator).tolist()


@torch.no_grad()
def next_token_logits(reference: GPT2, sequence: list[int], position: int) -> torch.Tensor:
    """预测 sequence[position] 的 FP32 logits, 使用 GPT2.generate 会用的上下文窗口."""
    start = max(0, position - CONFIG.block_size)
    return reference(torch.tensor([sequence[start:position]]))[0, -1]


def assert_near_top_k(reference: GPT2, sequence: list[int], prompt_len: int, top_k: int) -> None:
    for position in range(prompt_len, len(sequence)):
        logits = next_token_logits(reference, sequence, position)
        kth = torch.topk(logits, top_k).values[-1]
        chosen = logits[sequence[position]]
        assert chosen >= kth - LOGIT_TOLERANCE, f"位置 {position} 的 token 不在前 {top_k} 个中"


@pytest.mark.parametrize("length", [1, 7, 64, 70, 80])
def test_logits_match_reference(
    reference: GPT2, checkpoint: Path, backend: str, length: int, tmp_path: Path
) -> None:
    tokens = random_tokens(length)
    path = tmp_path / "logits.bin"
    run(checkpoint, backend, "--tokens", tokens_arg(tokens), "--dump-logits", str(path))
    logits = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(length, -1))
    with torch.no_grad():
        expected = reference(torch.tensor([tokens]))[0]
    torch.testing.assert_close(logits, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(("prompt_len", "new_tokens"), [(1, 30), (10, 20), (70, 20)])
def test_greedy_generation(
    reference: GPT2, checkpoint: Path, backend: str, prompt_len: int, new_tokens: int
) -> None:
    """解码路径使用 KV 缓存; 70 + 20 个 token 超过 80 的上下文, 之后每步重算窗口."""
    prompt = random_tokens(prompt_len, seed=prompt_len)
    output = run(
        checkpoint,
        backend,
        "--tokens",
        tokens_arg(prompt),
        "--max-new-tokens",
        str(new_tokens),
        "--print-tokens",
    )
    sequence = [int(x) for x in output.split()]
    assert sequence[:prompt_len] == prompt
    assert len(sequence) == prompt_len + new_tokens
    assert_near_top_k(reference, sequence, prompt_len, top_k=1)


def test_sampling_is_reproducible(reference: GPT2, checkpoint: Path, backend: str) -> None:
    prompt = random_tokens(5)
    args = ["--tokens", tokens_arg(prompt), "--max-new-tokens", "15", "--print-tokens"]
    args += ["--temperature", "1.0", "--top-k", "5", "--seed", "7"]
    first = run(checkpoint, backend, *args)
    assert first == run(checkpoint, backend, *args)
    assert_near_top_k(reference, [int(x) for x in first.split()], len(prompt), top_k=5)


def test_pytorch_weight_layout(reference: GPT2, backend: str, tmp_path: Path) -> None:
    """GPT2.state_dict() 的 nn.Linear 布局, 键带 transformer. 前缀, 含共享的 lm_head."""
    save_file(
        {k: v.contiguous() for k, v in reference.state_dict().items() if k != "lm_head.weight"},
        tmp_path / "model.safetensors",
    )
    tokens = random_tokens(9)
    path = tmp_path / "logits.bin"
    run(
        tmp_path / "model.safetensors",
        backend,
        "--weight-layout",
        "pytorch",
        "--tokens",
        tokens_arg(tokens),
        "--dump-logits",
        str(path),
    )
    logits = torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(len(tokens), -1))
    with torch.no_grad():
        expected = reference(torch.tensor([tokens]))[0]
    torch.testing.assert_close(logits, expected, rtol=RTOL, atol=ATOL)
