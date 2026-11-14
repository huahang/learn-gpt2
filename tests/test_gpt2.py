import pytest
import torch
import torch.nn.functional as F
from transformers import DynamicCache, GPT2LMHeadModel
from transformers import GPT2Config as HFGPT2Config

from learn_gpt2.gpt2 import GPT2, GPT2_CONFIGS, SUPPORTED_DTYPES, GPT2Config, KVCache

TINY_CONFIG = GPT2Config(block_size=32, vocab_size=97, n_layer=2, n_head=4, n_embd=32)
DTYPE_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-2, 1e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


@pytest.fixture
def model() -> GPT2:
    torch.manual_seed(0)
    return GPT2(TINY_CONFIG).eval()


@pytest.fixture
def hf_model(monkeypatch: pytest.MonkeyPatch) -> GPT2LMHeadModel:
    torch.manual_seed(0)
    hf_config = HFGPT2Config(
        n_positions=TINY_CONFIG.block_size,
        vocab_size=TINY_CONFIG.vocab_size,
        n_layer=TINY_CONFIG.n_layer,
        n_head=TINY_CONFIG.n_head,
        n_embd=TINY_CONFIG.n_embd,
        bos_token_id=None,
        eos_token_id=None,
    )
    model_hf = GPT2LMHeadModel(hf_config).eval()
    monkeypatch.setitem(GPT2_CONFIGS, "gpt2-tiny", TINY_CONFIG)
    monkeypatch.setattr(GPT2LMHeadModel, "from_pretrained", lambda _: model_hf)
    return model_hf


@pytest.fixture
def hf_pair(hf_model: GPT2LMHeadModel) -> tuple[GPT2LMHeadModel, GPT2]:
    return hf_model, GPT2.from_pretrained("gpt2-tiny").eval()


def random_tokens(batch_size: int, length: int) -> torch.Tensor:
    return torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, length))


@torch.no_grad()
def test_cached_forward_matches_full_forward(model: GPT2) -> None:
    idx = random_tokens(3, 20)
    expected = model(idx)

    kv_cache = KVCache(TINY_CONFIG.n_layer, max_len=idx.size(1))
    # Prefill, then single tokens, then a multi-token chunk that needs an offset causal mask.
    chunks = [(0, 8), (8, 9), (9, 10), (10, 16), (16, 20)]
    for start, end in chunks:
        logits = model(idx[:, start:end], kv_cache)
        torch.testing.assert_close(logits, expected[:, start:end])
    assert kv_cache.length == idx.size(1)


@torch.no_grad()
def test_cache_overflow_raises(model: GPT2) -> None:
    kv_cache = KVCache(TINY_CONFIG.n_layer, max_len=4)
    model(random_tokens(1, 4), kv_cache)
    with pytest.raises(ValueError, match="KV cache overflow"):
        model(random_tokens(1, 1), kv_cache)


@torch.no_grad()
def test_cache_batch_mismatch_raises(model: GPT2) -> None:
    kv_cache = KVCache(TINY_CONFIG.n_layer, max_len=8)
    model(random_tokens(2, 3), kv_cache)
    with pytest.raises(ValueError, match="KV cache mismatch"):
        model(random_tokens(1, 1), kv_cache)
    assert all(layer.length == 3 for layer in kv_cache.layers)


@pytest.mark.parametrize("prompt_len", [1, 5, TINY_CONFIG.block_size + 3])
def test_greedy_generate_matches_uncached(model: GPT2, prompt_len: int) -> None:
    # 40 new tokens outgrow block_size, which exercises the uncached fallback too.
    idx = random_tokens(2, prompt_len)
    cached = model.generate(idx, 40, temperature=0, use_cache=True)
    uncached = model.generate(idx, 40, temperature=0, use_cache=False)
    assert cached.shape == (2, prompt_len + 40)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


def test_sampled_generate_matches_uncached(model: GPT2) -> None:
    idx = random_tokens(4, 6)
    torch.manual_seed(1234)
    cached = model.generate(idx, 20, temperature=0.8, top_k=10, use_cache=True)
    torch.manual_seed(1234)
    uncached = model.generate(idx, 20, temperature=0.8, top_k=10, use_cache=False)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


def test_parameters_and_legacy_buffers_match_hf(hf_model: GPT2LMHeadModel) -> None:
    # Older Hugging Face checkpoints store causal mask buffers in each attention layer.
    for block in hf_model.transformer.h:
        block.attn.register_buffer(
            "bias", torch.tril(torch.ones(1, 1, TINY_CONFIG.block_size, TINY_CONFIG.block_size))
        )
        block.attn.register_buffer("masked_bias", torch.tensor(-1e4))

    custom_model = GPT2.from_pretrained("gpt2-tiny").eval()
    assert custom_model.transformer.wte.weight is custom_model.lm_head.weight
    assert dict(custom_model.named_parameters()).keys() == dict(hf_model.named_parameters()).keys()
    assert sum(p.numel() for p in custom_model.parameters()) == sum(
        p.numel() for p in hf_model.parameters()
    )


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [1, 13, TINY_CONFIG.block_size])
def test_forward_matches_hf(hf_pair: tuple[GPT2LMHeadModel, GPT2], seq_len: int) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(2, seq_len)
    torch.testing.assert_close(custom_model(idx), hf_model(idx).logits)


@torch.no_grad()
def test_cached_forward_matches_hf(hf_pair: tuple[GPT2LMHeadModel, GPT2]) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(3, 20)
    kv_cache = KVCache(TINY_CONFIG.n_layer, max_len=idx.size(1))
    hf_cache = DynamicCache()
    chunks = [(0, 8), (8, 9), (9, 10), (10, 16), (16, 20)]
    for start, end in chunks:
        custom_logits = custom_model(idx[:, start:end], kv_cache)
        hf_logits = hf_model(idx[:, start:end], past_key_values=hf_cache, use_cache=True).logits
        torch.testing.assert_close(custom_logits, hf_logits)


@pytest.mark.parametrize("prompt_len", [1, 8])
def test_greedy_generate_matches_hf(hf_pair: tuple[GPT2LMHeadModel, GPT2], prompt_len: int) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(2, prompt_len)
    max_new_tokens = TINY_CONFIG.block_size - prompt_len
    hf_out = hf_model.generate(idx, max_new_tokens=max_new_tokens, do_sample=False)
    custom_cached = custom_model.generate(idx, max_new_tokens, temperature=0, use_cache=True)
    custom_uncached = custom_model.generate(idx, max_new_tokens, temperature=0, use_cache=False)
    torch.testing.assert_close(custom_cached, hf_out, rtol=0, atol=0)
    torch.testing.assert_close(custom_uncached, hf_out, rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=str)
def test_from_pretrained_dtypes_match_hf(hf_model: GPT2LMHeadModel, dtype: torch.dtype) -> None:
    custom_model = GPT2.from_pretrained("gpt2-tiny", dtype=dtype).eval()
    assert all(p.dtype == dtype for p in custom_model.parameters())
    idx = random_tokens(2, 12)
    custom_logits = custom_model(idx)
    hf_logits = hf_model.to(dtype)(idx).logits
    assert custom_logits.dtype == dtype
    rtol, atol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(custom_logits, hf_logits, rtol=rtol, atol=atol)


def test_backward_gradients_match_hf(hf_pair: tuple[GPT2LMHeadModel, GPT2]) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(3, 16)
    targets = random_tokens(3, 16)

    hf_loss = F.cross_entropy(
        hf_model(idx).logits.view(-1, TINY_CONFIG.vocab_size), targets.view(-1)
    )
    custom_loss = F.cross_entropy(
        custom_model(idx).view(-1, TINY_CONFIG.vocab_size), targets.view(-1)
    )
    torch.testing.assert_close(custom_loss, hf_loss)

    hf_loss.backward()
    custom_loss.backward()
    hf_params = dict(hf_model.named_parameters())
    for name, custom_param in custom_model.named_parameters():
        hf_grad = hf_params[name].grad
        assert custom_param.grad is not None and hf_grad is not None
        if hf_grad.ndim == 2 and not name.endswith(("wte.weight", "wpe.weight")):
            hf_grad = hf_grad.t()
        torch.testing.assert_close(custom_param.grad, hf_grad)
