"""检查权重加载，并在 CUDA 可用时对照 gpt2.py 验证 FP16 算子。"""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

tilelang = pytest.importorskip("tilelang")

from learn_gpt2 import gpt2_inference_tilelang as tl_gpt2  # noqa: E402
from learn_gpt2.gpt2 import GPT2, GPT2Config  # noqa: E402

TINY = GPT2Config(block_size=512, vocab_size=97, n_layer=2, n_head=4, n_embd=64)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="需要 NVIDIA CUDA GPU",
)


@pytest.fixture
def checkpoint() -> dict[str, torch.Tensor]:
    torch.manual_seed(42)
    model = GPT2(TINY).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim == 1:
                if "ln_" in name and name.endswith("weight"):
                    param.uniform_(0.7, 1.3)
                else:
                    param.normal_(std=0.03)
    return {
        name: tensor.t().contiguous().clone()
        if name.endswith(tl_gpt2._MATRIX_SUFFIXES)
        else tensor.clone()
        for name, tensor in model.state_dict().items()
        if name != "lm_head.weight"
    }


def test_checkpoint_layouts(checkpoint: dict[str, torch.Tensor]) -> None:
    config, hf = tl_gpt2._prepare_state_dict(checkpoint, TINY.n_head, "hf")
    assert config == TINY
    pytorch = {
        key: tensor.t().contiguous() if key.endswith(tl_gpt2._MATRIX_SUFFIXES) else tensor
        for key, tensor in hf.items()
    }
    assert tl_gpt2._prepare_state_dict(pytorch, TINY.n_head, "pytorch")[0] == TINY
    with pytest.raises(ValueError, match="形状不匹配"):
        tl_gpt2._prepare_state_dict(checkpoint, TINY.n_head, "pytorch")
    with pytest.raises(ValueError, match="n_head"):
        tl_gpt2._prepare_state_dict(checkpoint, None, "hf")


@pytest.mark.parametrize("heads", [0, -1, 3, True])
def test_invalid_head_count(checkpoint: dict[str, torch.Tensor], heads: int) -> None:
    with pytest.raises(ValueError, match="n_head"):
        tl_gpt2._prepare_state_dict(checkpoint, heads, "hf")


def test_invalid_weights(checkpoint: dict[str, torch.Tensor]) -> None:
    missing = checkpoint.copy()
    del missing["transformer.h.0.ln_1.weight"]
    with pytest.raises(ValueError, match="missing"):
        tl_gpt2._prepare_state_dict(missing, TINY.n_head, "hf")
    invalid = checkpoint | {"transformer.wpe.weight": torch.zeros(512, 63)}
    with pytest.raises(ValueError, match="形状不匹配"):
        tl_gpt2._prepare_state_dict(invalid, TINY.n_head, "hf")
    with pytest.raises(ValueError, match="重复权重"):
        tl_gpt2._prepare_state_dict(
            checkpoint | {"wte.weight": checkpoint["transformer.wte.weight"]}, 4, "hf"
        )


def test_safetensors_and_shards(tmp_path: Path, checkpoint: dict[str, torch.Tensor]) -> None:
    single = tmp_path / "model.safetensors"
    save_file(checkpoint, str(single))
    (tmp_path / "config.json").write_text(json.dumps({"n_head": TINY.n_head}))
    loaded, heads = tl_gpt2._load_checkpoint(single)
    assert heads == TINY.n_head
    for name, expected in checkpoint.items():
        torch.testing.assert_close(loaded[name], expected)
    assert tl_gpt2._load_checkpoint(tmp_path)[0].keys() == checkpoint.keys()
    names = list(checkpoint)
    weight_map = {}
    for i in range(2):
        shard_name = f"model-{i}.safetensors"
        shard = {key: checkpoint[key] for key in names[i::2]}
        save_file(shard, str(tmp_path / shard_name))
        weight_map.update(dict.fromkeys(shard, shard_name))
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))
    loaded, heads = tl_gpt2._load_checkpoint(tmp_path)
    assert heads == TINY.n_head
    assert tl_gpt2._prepare_state_dict(loaded, heads, "hf")[0] == TINY
    weight_map[names[0]] = "../outside.safetensors"
    index.write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(ValueError, match="检查点目录"):
        tl_gpt2._load_checkpoint(tmp_path)


@pytest.mark.parametrize("length,last_only", [(7, False), (7, True), (1, True)])
def test_layernorm_ir(length: int, last_only: bool) -> None:
    # IR 构造不需要 GPU，检查不同输出形状的分支均可解析。
    kernel = tl_gpt2._layernorm_kernel(2, length, 60, last_only)
    assert kernel is not None


@pytest.mark.parametrize(
    "tokens",
    [
        torch.tensor([1]),
        torch.empty(1, 0, dtype=torch.int64),
        torch.tensor([[1.5]]),
        torch.tensor([[-1]]),
        torch.tensor([[TINY.vocab_size]]),
    ],
)
def test_invalid_tokens(tokens: torch.Tensor) -> None:
    model = object.__new__(tl_gpt2.GPT2Inference)
    model.config, model.device = TINY, torch.device("cpu")
    with pytest.raises(ValueError):
        model._check_tokens(tokens)


@pytest.mark.skipif(
    os.environ.get("LEARN_GPT2_TILELANG_COMPILE_TESTS") != "1",
    reason="设置 LEARN_GPT2_TILELANG_COMPILE_TESTS=1 运行离线 CUDA 编译检查",
)
@pytest.mark.parametrize("kind", ["embed", "norm", "linear", "qkv", "attention", "decode", "merge"])
def test_offline_cuda_compilation(kind: str) -> None:
    # 尾块覆盖非整除的 M/N/K 和 head_size；不需要驱动，也不加载 GPU 模块。
    config = GPT2Config(block_size=512, vocab_size=97, n_layer=2, n_head=3, n_embd=60)
    factories = {
        "embed": lambda: tl_gpt2._embedding_kernel(2, 17, 60, 97, 512),
        "norm": lambda: tl_gpt2._layernorm_kernel(2, 17, 60, True),
        "linear": lambda: tl_gpt2._linear_kernel(34, 97, 60, gelu=True, residual=True),
        "qkv": lambda: tl_gpt2._qkv_kernel(config, 2, 17, 512),
        "attention": lambda: tl_gpt2._attention_kernel(config, 2, 17, 512, 4),
        "decode": lambda: tl_gpt2._attention_kernel(config, 2, 1, 512, 4),
        "merge": lambda: tl_gpt2._merge_attention_kernel(2, 3, 20, 4),
    }
    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        artifact = tilelang.lower(factories[kind](), target=target, enable_device_compile=True)
    assert artifact.kernel_source


@requires_cuda
@pytest.mark.parametrize("length", [1, 7, 37])
@torch.inference_mode()
def test_forward_matches_reference(
    tmp_path: Path, checkpoint: dict[str, torch.Tensor], length: int
) -> None:
    path = tmp_path / "tiny.safetensors"
    save_file(checkpoint, str(path))
    model = tl_gpt2.GPT2Inference.from_safetensors(path, n_head=TINY.n_head, cuda_graph=False)
    reference = _reference(checkpoint)
    tokens = torch.randint(0, TINY.vocab_size, (2, length), device="cuda")
    actual = model(tokens)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, reference(tokens).float(), atol=2e-3, rtol=2e-2)


def _reference(checkpoint: dict[str, torch.Tensor], config: GPT2Config = TINY) -> GPT2:
    state_dict = {
        key: tensor.t().contiguous() if key.endswith(tl_gpt2._MATRIX_SUFFIXES) else tensor
        for key, tensor in checkpoint.items()
    }
    state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
    reference = GPT2(config).cuda().half().eval()
    reference.load_state_dict(state_dict)
    return reference


@requires_cuda
@pytest.mark.parametrize("cuda_graph", [False, True])
@torch.inference_mode()
def test_split_kv_decode_and_graph(checkpoint: dict[str, torch.Tensor], cuda_graph: bool) -> None:
    reference = _reference(checkpoint)
    model = tl_gpt2.GPT2Inference(checkpoint, n_head=TINY.n_head, cuda_graph=cuda_graph)
    # 256 以上的容量启用 split-KV；短前缀会产生空的 split。
    ws = model._workspace(2, 400)
    tokens = torch.randint(0, TINY.vocab_size, (2, 270), device="cuda")
    expected = reference(tokens).float()
    torch.testing.assert_close(
        model._prefill(ws, tokens[:, :3]), expected[:, 2], atol=2e-3, rtol=2e-2
    )
    for i in range(3, 7):
        torch.testing.assert_close(
            model._decode(ws, tokens[:, i]), expected[:, i], atol=2e-3, rtol=2e-2
        )
    assert ws.position.item() == 7
    # 多 token 的追加必须使用带偏移的因果掩码。
    logits = torch.empty(2 * 263, TINY.vocab_size, dtype=torch.float32, device="cuda")
    model._pass(ws, tokens[:, 7:].reshape(-1), 263, logits, last_only=False)
    torch.testing.assert_close(logits.view(2, 263, -1), expected[:, 7:], atol=2e-3, rtol=2e-2)
    ws.position.fill_(270)
    torch.testing.assert_close(
        model._decode(ws, tokens[:, 0]),
        reference(torch.cat((tokens, tokens[:, :1]), 1))[:, -1].float(),
        atol=2e-3,
        rtol=2e-2,
    )
    # 同一工作区重置位置后继续复用已捕获的图。
    model._prefill(ws, tokens[:, :3])
    torch.testing.assert_close(
        model._decode(ws, tokens[:, 3]), expected[:, 3], atol=2e-3, rtol=2e-2
    )


@requires_cuda
@torch.inference_mode()
def test_generation_rollover_and_repeated_requests(checkpoint: dict[str, torch.Tensor]) -> None:
    checkpoint = checkpoint | {"transformer.wpe.weight": checkpoint["transformer.wpe.weight"][:8]}
    config = GPT2Config(8, TINY.vocab_size, TINY.n_layer, TINY.n_head, TINY.n_embd)
    reference = _reference(checkpoint, config)
    model = tl_gpt2.GPT2Inference(checkpoint, n_head=TINY.n_head)
    tokens = torch.randint(0, TINY.vocab_size, (2, 6), device="cuda")
    expected = reference.generate(tokens, 7, temperature=0)
    actual = model.generate(tokens, 7, temperature=0)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.generate(tokens, 7, temperature=0), actual)
    torch.testing.assert_close(model.generate(tokens, 0), tokens)
    # 初始提示词已超出上下文时，也应从最后一个窗口开始。
    long_tokens = tokens.repeat(1, 2)
    torch.testing.assert_close(
        model.generate(long_tokens, 2, temperature=0),
        reference.generate(long_tokens, 2, temperature=0),
    )


@requires_cuda
@torch.inference_mode()
def test_unaligned_dimensions_and_pytorch_layout() -> None:
    torch.manual_seed(123)
    config = GPT2Config(block_size=64, vocab_size=97, n_layer=1, n_head=3, n_embd=60)
    reference = GPT2(config).cuda().half().eval()
    model = tl_gpt2.GPT2Inference(
        reference.state_dict(), n_head=3, weight_layout="pytorch", cuda_graph=False
    )
    tokens = torch.randint(0, config.vocab_size, (2, 33), device="cuda")
    torch.testing.assert_close(model(tokens), reference(tokens).float(), atol=2e-3, rtol=2e-2)
