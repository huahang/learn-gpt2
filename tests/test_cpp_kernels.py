"""逐个检查 ``cpp/kernels`` 导出的 cubin kernel.

测试用 CUDA driver API 加载 cubin, 按 ``kernel_abi.py`` 的接口启动 kernel, 再与 PyTorch 的
FP32 结果比较. 两个后端共用同一套启动代码, 所以测试也验证了两者的接口一致.

需要先构建 kernel: ``make -C cpp kernels``. 某个后端没有构建时, 它的测试自动跳过.
"""

import ctypes
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = ROOT / "cpp" / "build" / "kernels"
sys.path.insert(0, str(ROOT / "cpp" / "kernels"))
import kernel_abi as abi  # noqa: E402

driver = pytest.importorskip("cuda.bindings.driver")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="cubin kernel 需要 NVIDIA GPU",
)

BACKENDS = ("tilelang", "cute_dsl")
RTOL, ATOL = 2e-2, 2e-2


def _check(result: tuple) -> object:
    err, *values = result
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver error: {err}")
    return values[0] if len(values) == 1 else values


class Kernels:
    """按 manifest 加载一个后端的全部 kernel. 与 C++ 端 ``KernelLibrary`` 的做法相同."""

    def __init__(self, directory: Path):
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # 确保当前线程有 CUDA context
        self.entries = {}
        for line in (directory / "manifest.txt").read_text().splitlines():
            fields = line.split()
            if not fields or fields[0] != "kernel":
                continue
            name, symbol, threads, smem, signature = fields[1:]
            assert signature == abi.KERNELS_BY_NAME[name].signature
            module = _check(driver.cuModuleLoad(str(directory / f"{name}.cubin").encode()))
            function = _check(driver.cuModuleGetFunction(module, symbol.encode()))
            if int(smem) > 48 * 1024:
                attribute = (
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
                )
                _check(driver.cuFuncSetAttribute(function, attribute, int(smem)))
            self.entries[name] = (module, function, int(threads), int(smem))

    def launch(self, name: str, **args: object) -> None:
        kernel = abi.KERNELS_BY_NAME[name]
        _, function, threads, smem = self.entries[name]
        scope = {p.name: args[p.name] for p in kernel.params if p.type in abi.SCALAR_TYPES}
        scope |= {"ceil": math.ceil, "TILE_M": abi.TILE_M, "TILE_N": abi.TILE_N}
        # grid 规则是 kernel_abi.py 中的表达式, 例如 "ceil(m / TILE_M)".
        grid = [int(eval(expr, {}, scope)) for expr in kernel.grid]
        values, types = [], []
        for p in kernel.params:
            value = args[p.name]
            if p.type in abi.POINTER_TYPES:
                values.append(value.data_ptr())
                types.append(ctypes.c_void_p)
            elif p.type == "i32":
                values.append(int(value))
                types.append(ctypes.c_int)
            else:
                values.append(float(value))
                types.append(ctypes.c_float)
        stream = torch.cuda.current_stream().cuda_stream
        _check(
            driver.cuLaunchKernel(
                function, *grid, threads, 1, 1, smem, stream, (tuple(values), tuple(types)), 0
            )
        )
        torch.cuda.synchronize()


@pytest.fixture(scope="module", params=BACKENDS)
def kernels(request: pytest.FixtureRequest) -> Kernels:
    directory = KERNEL_ROOT / request.param
    if not (directory / "manifest.txt").is_file():
        pytest.skip(f"没有构建 {request.param} kernel: make -C cpp kernels")
    return Kernels(directory)


def _round_up(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


def _padded(rows: int, cols: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """按 ABI 补齐到 TILE_M 的倍数行. 补齐的行填入有限的随机值, 检查 kernel 不写入它们."""
    return torch.randn(_round_up(rows, abi.TILE_M), cols, device="cuda").to(dtype)


@pytest.mark.parametrize("rows", [1, 7, 70])
@pytest.mark.parametrize("n_embd", [128, 768])
def test_embedding(kernels: Kernels, rows: int, n_embd: int) -> None:
    wte = torch.randn(300, n_embd, device="cuda", dtype=torch.float16)
    wpe = torch.randn(200, n_embd, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, 300, (rows,), device="cuda", dtype=torch.int32)
    x = _padded(rows, n_embd)
    pad = x[rows:].clone()
    kernels.launch(
        "gpt2_embedding", x=x, ids=ids, wte=wte, wpe=wpe, rows=rows, pos0=5, n_embd=n_embd
    )
    expected = wte[ids.long()].float() + wpe[5 : 5 + rows].float()
    torch.testing.assert_close(x[:rows].float(), expected, rtol=RTOL, atol=ATOL)
    assert torch.equal(x[rows:], pad)


@pytest.mark.parametrize("rows", [1, 70])
@pytest.mark.parametrize("n_embd", [128, 768, 1600])
def test_layernorm(kernels: Kernels, rows: int, n_embd: int) -> None:
    x = _padded(rows, n_embd) * 3 + 1
    gamma = torch.randn(n_embd, device="cuda", dtype=torch.float16)
    beta = torch.randn(n_embd, device="cuda", dtype=torch.float16)
    y = _padded(rows, n_embd)
    kernels.launch(
        "gpt2_layernorm", y=y, x=x, gamma=gamma, beta=beta, rows=rows, n_embd=n_embd, eps=1e-5
    )
    expected = F.layer_norm(x[:rows].float(), (n_embd,), gamma.float(), beta.float(), 1e-5)
    torch.testing.assert_close(y[:rows].float(), expected, rtol=RTOL, atol=ATOL)


def _gelu(v: torch.Tensor) -> torch.Tensor:
    return F.gelu(v, approximate="tanh")


@pytest.mark.parametrize("epilogue", ["bias", "gelu", "residual"])
@pytest.mark.parametrize(("m", "n", "k"), [(1, 192, 128), (70, 128, 512), (130, 2304, 768)])
def test_linear(kernels: Kernels, epilogue: str, m: int, n: int, k: int) -> None:
    x = _padded(m, k)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) / math.sqrt(k)
    b = torch.randn(n, device="cuda", dtype=torch.float16)
    y = _padded(m, n)
    before = y.clone()
    name = {"bias": "gpt2_linear_bias", "gelu": "gpt2_linear_bias_gelu"}.get(
        epilogue, "gpt2_linear_bias_residual"
    )
    kernels.launch(name, y=y, x=x, w=w, b=b, m=m, n=n, k=k)
    expected = x[:m].float() @ w.float().T + b.float()
    if epilogue == "gelu":
        expected = _gelu(expected)
    if epilogue == "residual":
        expected = expected + before[:m].float()
    torch.testing.assert_close(y[:m].float(), expected, rtol=RTOL, atol=ATOL)
    assert torch.equal(y[m:], before[m:])


@pytest.mark.parametrize(("m", "n", "k"), [(1, 320, 128), (70, 50304, 768)])
def test_lm_head(kernels: Kernels, m: int, n: int, k: int) -> None:
    x = _padded(m, k)
    w = torch.randn(n, k, device="cuda", dtype=torch.float16) / math.sqrt(k)
    y = _padded(m, n, torch.float32)
    kernels.launch("gpt2_lm_head", y=y, x=x, w=w, m=m, n=n, k=k)
    torch.testing.assert_close(y[:m], x[:m].float() @ w.float().T, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(("m", "pos0"), [(1, 0), (1, 100), (7, 0), (70, 0), (70, 58), (3, 125)])
@pytest.mark.parametrize("n_head", [2, 12])
def test_attention(kernels: Kernels, m: int, pos0: int, n_head: int) -> None:
    c = n_head * abi.HEAD_DIM
    total = pos0 + m
    qkv = _padded(m, 3 * c)
    # 缓存中 pos0 之前是更早的 token. 本次的 key 和 value 已写入 pos0 .. pos0 + m - 1.
    rows = _round_up(total, abi.TILE_M)
    k_cache = torch.zeros(rows, c, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    k_cache[:pos0] = torch.randn(pos0, c, device="cuda", dtype=torch.float16)
    v_cache[:pos0] = torch.randn(pos0, c, device="cuda", dtype=torch.float16)
    k_cache[pos0:total] = qkv[:m, c : 2 * c]
    v_cache[pos0:total] = qkv[:m, 2 * c :]
    y = _padded(m, c)
    scale = 1 / math.sqrt(abi.HEAD_DIM)
    kernels.launch(
        "gpt2_attention",
        y=y,
        qkv=qkv,
        k_cache=k_cache,
        v_cache=v_cache,
        m=m,
        pos0=pos0,
        n_head=n_head,
        scale=scale,
    )
    heads = (n_head, abi.HEAD_DIM)
    q = qkv[:m, :c].float().view(m, *heads).transpose(0, 1)
    k = k_cache[:total].float().view(total, *heads).transpose(0, 1)
    v = v_cache[:total].float().view(total, *heads).transpose(0, 1)
    # query i 位于 pos0 + i, 看到位置 0 .. pos0 + i.
    mask = (
        torch.arange(total, device="cuda")[None, :]
        <= pos0 + torch.arange(m, device="cuda")[:, None]
    )
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    expected = expected.transpose(0, 1).reshape(m, c)
    torch.testing.assert_close(y[:m].float(), expected, rtol=RTOL, atol=ATOL)
