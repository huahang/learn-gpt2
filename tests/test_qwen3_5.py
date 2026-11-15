import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from transformers import DynamicCache, Qwen3_5ForCausalLM, Qwen3_5TextConfig

import learn_gpt2


def load_qwen_module() -> ModuleType:
    # The dots in "qwen3.5_0.8b" make the module unreachable by an import statement.
    name = "learn_gpt2.qwen3_5_0_8b"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(learn_gpt2.__file__).with_name("qwen3.5_0.8b.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qwen = load_qwen_module()
Qwen3_5 = qwen.Qwen3_5
Qwen3_5Config = qwen.Qwen3_5Config
HybridCache = qwen.HybridCache

# Four query heads share two KV heads; two key heads serve four value heads in linear attention.
HF_TINY_CONFIG = dict(
    vocab_size=97,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
    linear_conv_kernel_dim=4,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    tie_word_embeddings=True,
    bos_token_id=None,
    eos_token_id=None,
    pad_token_id=None,
)
TINY_CONFIG = Qwen3_5Config.from_hf(Qwen3_5TextConfig(**HF_TINY_CONFIG))
# Longer than one 64-token chunk of the chunked delta rule.
LONG_LEN = 150
DTYPE_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1.3e-6, 1e-5),
    torch.float16: (1e-2, 1e-3),
    torch.bfloat16: (2e-2, 1e-2),
}


@pytest.fixture
def model() -> Qwen3_5:
    torch.manual_seed(0)
    return Qwen3_5(TINY_CONFIG).eval()


@pytest.fixture
def hf_model(monkeypatch: pytest.MonkeyPatch) -> Qwen3_5ForCausalLM:
    torch.manual_seed(0)
    hf_config = Qwen3_5TextConfig(**HF_TINY_CONFIG)
    model_hf = Qwen3_5ForCausalLM(hf_config).eval()
    with torch.no_grad():
        # The default init sets norm weights to 0 or 1, which would hide a norm that ignores them.
        for p in model_hf.parameters():
            p.add_(0.1 * torch.randn_like(p))
    state_dict = model_hf.state_dict()

    def fake_from_pretrained(
        model_type: str, *, dtype: torch.dtype = torch.float32
    ) -> Qwen3_5ForCausalLM:
        # Like a real checkpoint load: build in the requested dtype, then load the weights.
        loaded = Qwen3_5ForCausalLM._from_config(hf_config, dtype=dtype)
        loaded.load_state_dict(state_dict)
        return loaded.eval()

    monkeypatch.setattr(Qwen3_5ForCausalLM, "from_pretrained", fake_from_pretrained)
    return model_hf


@pytest.fixture
def hf_pair(hf_model: Qwen3_5ForCausalLM) -> tuple[Qwen3_5ForCausalLM, Qwen3_5]:
    return hf_model, Qwen3_5.from_pretrained("qwen-tiny").eval()


def random_tokens(batch_size: int, length: int) -> torch.Tensor:
    return torch.randint(0, TINY_CONFIG.vocab_size, (batch_size, length))


def test_default_config_matches_qwen3_5_0_8b() -> None:
    hf_config = Qwen3_5TextConfig(
        hidden_size=1024,
        intermediate_size=3584,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        linear_num_value_heads=16,
        max_position_embeddings=262144,
        tie_word_embeddings=True,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000},
    )
    assert Qwen3_5Config.from_hf(hf_config) == Qwen3_5Config()


@pytest.mark.parametrize("initial_state", [False, True])
def test_chunked_delta_rule_matches_recurrent(initial_state: bool) -> None:
    torch.manual_seed(0)
    B, T, nk, nv, dk, dv = 2, LONG_LEN, 2, 4, 8, 6
    q = torch.randn(B, T, nk, dk)
    k = torch.randn(B, T, nk, dk)
    v = torch.randn(B, T, nv, dv)
    g = -torch.rand(B, T, nv)
    beta = torch.rand(B, T, nv)
    state = torch.randn(B, nv, dk, dv) if initial_state else None

    out_rec, state_rec = qwen.recurrent_gated_delta_rule(q, k, v, g, beta, state)
    out_chunk, state_chunk = qwen.chunk_gated_delta_rule(q, k, v, g, beta, state, chunk_size=16)
    assert out_chunk.shape == (B, T, nv, dv)
    torch.testing.assert_close(out_chunk, out_rec, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(state_chunk, state_rec, rtol=1e-4, atol=1e-4)


@torch.no_grad()
def test_cached_forward_matches_full_forward(model: Qwen3_5) -> None:
    idx = random_tokens(3, LONG_LEN)
    expected = model(idx)

    kv_cache = HybridCache(TINY_CONFIG, max_len=idx.size(1))
    # Prefill, single tokens, a chunk that needs an offset causal mask, then one past 64 tokens.
    chunks = [(0, 8), (8, 9), (9, 10), (10, 16), (16, 20), (20, LONG_LEN)]
    for start, end in chunks:
        logits = model(idx[:, start:end], kv_cache)
        torch.testing.assert_close(logits, expected[:, start:end])
    assert kv_cache.length == idx.size(1)
    assert all(layer.length == idx.size(1) for layer in kv_cache.layers)


@torch.no_grad()
def test_cache_overflow_raises(model: Qwen3_5) -> None:
    kv_cache = HybridCache(TINY_CONFIG, max_len=4)
    model(random_tokens(1, 4), kv_cache)
    with pytest.raises(ValueError, match="KV cache overflow"):
        model(random_tokens(1, 1), kv_cache)
    # The check runs before any layer, so linear attention states are untouched too.
    assert all(layer.length == 4 for layer in kv_cache.layers)


@torch.no_grad()
def test_cache_batch_mismatch_raises(model: Qwen3_5) -> None:
    kv_cache = HybridCache(TINY_CONFIG, max_len=8)
    model(random_tokens(2, 3), kv_cache)
    with pytest.raises(ValueError, match="cache mismatch"):
        model(random_tokens(1, 1), kv_cache)
    assert all(layer.length == 3 for layer in kv_cache.layers)


@pytest.mark.parametrize("prompt_len", [1, 5])
def test_greedy_generate_matches_uncached(model: Qwen3_5, prompt_len: int) -> None:
    idx = random_tokens(2, prompt_len)
    cached = model.generate(idx, 40, temperature=0, use_cache=True)
    uncached = model.generate(idx, 40, temperature=0, use_cache=False)
    assert cached.shape == (2, prompt_len + 40)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


def test_sampled_generate_matches_uncached(model: Qwen3_5) -> None:
    idx = random_tokens(4, 6)
    torch.manual_seed(1234)
    cached = model.generate(idx, 20, temperature=0.8, top_k=10, use_cache=True)
    torch.manual_seed(1234)
    uncached = model.generate(idx, 20, temperature=0.8, top_k=10, use_cache=False)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)


def test_structure_matches_hf(hf_pair: tuple[Qwen3_5ForCausalLM, Qwen3_5]) -> None:
    hf_model, custom_model = hf_pair
    assert custom_model.lm_head.weight is custom_model.model.embed_tokens.weight
    hf_sd, custom_sd = hf_model.state_dict(), custom_model.state_dict()
    assert list(custom_sd) == list(hf_sd)
    assert all(custom_sd[k].shape == v.shape for k, v in hf_sd.items())
    assert list(dict(custom_model.named_parameters())) == list(dict(hf_model.named_parameters()))
    assert sum(p.numel() for p in custom_model.parameters()) == sum(
        p.numel() for p in hf_model.parameters()
    )


@torch.no_grad()
@pytest.mark.parametrize("seq_len", [1, 13, LONG_LEN])
def test_forward_matches_hf(hf_pair: tuple[Qwen3_5ForCausalLM, Qwen3_5], seq_len: int) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(2, seq_len)
    torch.testing.assert_close(custom_model(idx), hf_model(idx).logits)


@torch.no_grad()
def test_cached_forward_matches_hf(hf_pair: tuple[Qwen3_5ForCausalLM, Qwen3_5]) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(3, 20)
    kv_cache = HybridCache(TINY_CONFIG, max_len=idx.size(1))
    hf_cache = DynamicCache(config=hf_model.config)
    chunks = [(0, 8), (8, 9), (9, 10), (10, 16), (16, 20)]
    for start, end in chunks:
        custom_logits = custom_model(idx[:, start:end], kv_cache)
        hf_logits = hf_model(idx[:, start:end], past_key_values=hf_cache, use_cache=True).logits
        torch.testing.assert_close(custom_logits, hf_logits)


@pytest.mark.parametrize("prompt_len", [1, 8])
def test_greedy_generate_matches_hf(
    hf_pair: tuple[Qwen3_5ForCausalLM, Qwen3_5], prompt_len: int
) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(2, prompt_len)
    max_new_tokens = 24
    hf_out = hf_model.generate(idx, max_new_tokens=max_new_tokens, do_sample=False)
    custom_cached = custom_model.generate(idx, max_new_tokens, temperature=0, use_cache=True)
    custom_uncached = custom_model.generate(idx, max_new_tokens, temperature=0, use_cache=False)
    torch.testing.assert_close(custom_cached, hf_out, rtol=0, atol=0)
    torch.testing.assert_close(custom_uncached, hf_out, rtol=0, atol=0)


@torch.no_grad()
@pytest.mark.parametrize("dtype", qwen.SUPPORTED_DTYPES, ids=str)
def test_from_pretrained_dtypes_match_hf(hf_model: Qwen3_5ForCausalLM, dtype: torch.dtype) -> None:
    custom_model = Qwen3_5.from_pretrained("qwen-tiny", dtype=dtype).eval()
    assert all(p.dtype == dtype for p in custom_model.parameters())
    assert custom_model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.get_default_dtype() == torch.float32
    hf_model = Qwen3_5ForCausalLM.from_pretrained("qwen-tiny", dtype=dtype)
    idx = random_tokens(2, 12)
    custom_logits = custom_model(idx)
    assert custom_logits.dtype == dtype
    rtol, atol = DTYPE_TOLERANCES[dtype]
    torch.testing.assert_close(custom_logits, hf_model(idx).logits, rtol=rtol, atol=atol)


def test_backward_gradients_match_hf(hf_pair: tuple[Qwen3_5ForCausalLM, Qwen3_5]) -> None:
    hf_model, custom_model = hf_pair
    idx = random_tokens(3, 16)
    targets = random_tokens(3, 16)

    hf_loss = F.cross_entropy(
        hf_model(idx, use_cache=False).logits.view(-1, TINY_CONFIG.vocab_size), targets.view(-1)
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
        torch.testing.assert_close(custom_param.grad, hf_grad)
