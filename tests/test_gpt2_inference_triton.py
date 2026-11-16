from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from learn_gpt2.gpt2 import GPT2, GPT2Config

# The module imports Triton, which the mps backend does not install.
gpt2_inference_triton = pytest.importorskip("learn_gpt2.gpt2_inference_triton")
GPT2Inference = gpt2_inference_triton.GPT2Inference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need a GPU")

# head_size 16 is the smallest tile ``tl.dot`` accepts. head_size 10 and n_embd 40 fit no tile
# exactly, so they exercise the padded and masked code paths.
CONFIGS = {
    "head16": GPT2Config(block_size=48, vocab_size=97, n_layer=2, n_head=4, n_embd=64),
    "head10": GPT2Config(block_size=48, vocab_size=97, n_layer=2, n_head=4, n_embd=40),
}
# torch.nn.Linear stores (out, in); Hugging Face Conv1D stores (in, out).
CONV1D_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)
# The engine computes in float16; the reference runs in float32.
RTOL, ATOL = 2e-2, 2e-2
# A greedy or top-k choice may differ from the float32 choice only between near-equal logits.
LOGIT_TOLERANCE = 5e-2


def hf_state_dict(model: GPT2) -> dict[str, torch.Tensor]:
    """Return the weights of ``model`` as ``GPT2LMHeadModel.state_dict()`` would store them."""
    return {
        key: value.t().contiguous() if key.endswith(CONV1D_WEIGHTS) else value.clone()
        for key, value in model.state_dict().items()
        if key != "lm_head.weight"
    }


@pytest.fixture(params=list(CONFIGS))
def config(request: pytest.FixtureRequest) -> GPT2Config:
    return CONFIGS[request.param]


@pytest.fixture
def reference(config: GPT2Config) -> GPT2:
    torch.manual_seed(0)
    model = GPT2(config)
    # The default init gives near-uniform logits and zero biases. Spread the logits and fill
    # every bias and layer norm parameter, so that each term of the computation matters.
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() == 2:
                param.mul_(5)
            else:
                param.add_(torch.randn_like(param) * 0.2)
    return model.cuda().eval()


@pytest.fixture
def weights_path(reference: GPT2, tmp_path: Path) -> Path:
    path = tmp_path / "gpt2_tiny.safetensors"
    save_file(hf_state_dict(reference), path)
    return path


@pytest.fixture
def engine(weights_path: Path, config: GPT2Config) -> GPT2Inference:
    return GPT2Inference.from_safetensors(weights_path, n_head=config.n_head)


def random_tokens(config: GPT2Config, batch_size: int, length: int) -> torch.Tensor:
    return torch.randint(0, config.vocab_size, (batch_size, length), device="cuda")


@torch.no_grad()
def next_token_logits(reference: GPT2, seq: torch.Tensor, position: int) -> torch.Tensor:
    """Float32 logits for ``seq[:, position]``, from the context ``GPT2.generate`` would use."""
    start = max(0, position - reference.block_size)
    return reference(seq[:, start:position])[:, -1]


def assert_near_top_k(reference: GPT2, seq: torch.Tensor, prompt_len: int, top_k: int) -> None:
    """Check that every generated token is among the ``top_k`` best float32 logits."""
    for position in range(prompt_len, seq.size(1)):
        logits = next_token_logits(reference, seq, position)
        chosen = logits.gather(1, seq[:, position : position + 1]).squeeze(1)
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1]
        assert (chosen >= kth - LOGIT_TOLERANCE).all(), f"unlikely token at position {position}"


@pytest.mark.parametrize("shape", [(1, 1), (2, 5), (3, 16), (3, 33), (2, 48), (20, 2)])
@torch.no_grad()
def test_forward_matches_reference(
    reference: GPT2, engine: GPT2Inference, config: GPT2Config, shape: tuple[int, int]
) -> None:
    idx = random_tokens(config, *shape)
    logits = engine(idx)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, reference(idx), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("prompt_len", [1, 5, 40, 51])
@pytest.mark.parametrize("cuda_graph", [True, False])
def test_greedy_generate_follows_reference(
    reference: GPT2, weights_path: Path, config: GPT2Config, prompt_len: int, cuda_graph: bool
) -> None:
    # Prompt lengths 40 and 51 make the sequence outgrow block_size 48 during or before decoding.
    engine = GPT2Inference.from_safetensors(
        weights_path, n_head=config.n_head, cuda_graph=cuda_graph
    )
    idx = random_tokens(config, 3, prompt_len)

    out = engine.generate(idx, 20, temperature=0)

    assert out.shape == (3, prompt_len + 20)
    assert torch.equal(out[:, :prompt_len], idx)
    assert_near_top_k(reference, out, prompt_len, top_k=1)


def test_graph_replay_matches_eager_launches(weights_path: Path, config: GPT2Config) -> None:
    idx = random_tokens(config, 3, 5)
    outputs = [
        GPT2Inference.from_safetensors(
            weights_path, n_head=config.n_head, cuda_graph=cuda_graph
        ).generate(idx, 20, temperature=0)
        for cuda_graph in (True, False)
    ]

    assert torch.equal(*outputs)


def test_decode_batch_larger_than_one_row_tile(
    reference: GPT2, engine: GPT2Inference, config: GPT2Config
) -> None:
    # More than 16 sequences leave the single-tile decode path.
    idx = random_tokens(config, 20, 4)

    out = engine.generate(idx, 12, temperature=0)

    assert_near_top_k(reference, out, 4, top_k=1)


def test_generate_does_not_depend_on_earlier_calls(
    engine: GPT2Inference, config: GPT2Config
) -> None:
    idx = random_tokens(config, 3, 6)
    first = engine.generate(idx, 10, temperature=0)
    # Same batch size, so this call reuses the buffers and the captured graph.
    engine.generate(random_tokens(config, 3, 9), 15, temperature=0)

    assert torch.equal(engine.generate(idx, 10, temperature=0), first)


def test_sampled_generate_is_seeded_and_respects_top_k(
    reference: GPT2, engine: GPT2Inference, config: GPT2Config
) -> None:
    idx = random_tokens(config, 4, 5)
    torch.manual_seed(1234)
    out = engine.generate(idx, 20, temperature=0.8, top_k=5)
    torch.manual_seed(1234)
    again = engine.generate(idx, 20, temperature=0.8, top_k=5)

    assert torch.equal(out, again)
    assert torch.equal(out[:, :5], idx)
    assert not torch.equal(out, engine.generate(idx, 20, temperature=0))
    assert_near_top_k(reference, out, 5, top_k=5)


def test_zero_new_tokens_returns_prompt(engine: GPT2Inference, config: GPT2Config) -> None:
    idx = random_tokens(config, 2, 7)

    assert torch.equal(engine.generate(idx, 0), idx)


@torch.no_grad()
def test_state_dict_variants_load_identically(
    reference: GPT2, engine: GPT2Inference, config: GPT2Config
) -> None:
    # The checkpoint on the Hugging Face hub has no ``transformer.`` prefix and still carries the
    # causal-mask buffers. A state dict taken from a live model carries the tied LM head.
    state_dict = {
        key.removeprefix("transformer."): value.double()
        for key, value in hf_state_dict(reference).items()
    }
    state_dict["h.0.attn.bias"] = torch.ones(1, 1, config.block_size, config.block_size)
    state_dict["lm_head.weight"] = state_dict["wte.weight"]
    other = GPT2Inference(state_dict, n_head=config.n_head)
    idx = random_tokens(config, 2, 9)

    assert other.wte.dtype == torch.float16
    assert torch.equal(other(idx), engine(idx))


def test_published_sizes_need_no_head_count() -> None:
    config = GPT2Config(block_size=8, vocab_size=11, n_layer=12, n_head=12, n_embd=768)
    torch.manual_seed(0)
    engine = GPT2Inference(hf_state_dict(GPT2(config)))

    assert engine.config == config


def test_unknown_size_needs_head_count(weights_path: Path) -> None:
    with pytest.raises(ValueError, match="pass n_head"):
        GPT2Inference.from_safetensors(weights_path)


def test_incomplete_state_dict_raises(reference: GPT2, config: GPT2Config) -> None:
    state_dict = hf_state_dict(reference)
    del state_dict["transformer.h.1.mlp.c_fc.bias"]
    state_dict["transformer.extra"] = torch.zeros(1)

    with pytest.raises(
        ValueError, match=r"missing=\['h.1.mlp.c_fc.bias'\], unexpected=\['extra'\]"
    ):
        GPT2Inference(state_dict, n_head=config.n_head)


def test_wrong_layout_raises(reference: GPT2, config: GPT2Config) -> None:
    # A state dict in torch.nn.Linear layout: (out, in) instead of (in, out).
    state_dict = {k: v for k, v in reference.state_dict().items() if k != "lm_head.weight"}

    with pytest.raises(ValueError, match="Shape mismatch for h.0.attn.c_attn.weight"):
        GPT2Inference(state_dict, n_head=config.n_head)


def test_cpu_device_raises(reference: GPT2, config: GPT2Config) -> None:
    with pytest.raises(ValueError, match="need a GPU"):
        GPT2Inference(hf_state_dict(reference), n_head=config.n_head, device="cpu")


@pytest.mark.parametrize("bad_token", [-1, 97])
def test_out_of_range_token_raises(engine: GPT2Inference, bad_token: int) -> None:
    idx = torch.tensor([[1, bad_token, 2]])

    with pytest.raises(ValueError, match="Token ids"):
        engine(idx)
    with pytest.raises(ValueError, match="Token ids"):
        engine.generate(idx, 2)


def test_forward_longer_than_block_size_raises(engine: GPT2Inference, config: GPT2Config) -> None:
    with pytest.raises(ValueError, match="block size"):
        engine(random_tokens(config, 1, config.block_size + 1))
