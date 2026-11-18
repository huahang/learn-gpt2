"""C++ 推理框架与 cubin kernel 之间的接口 (ABI).

TileLang 与 CuTe DSL 两个后端都按这里的定义导出 kernel, C++ 端也按这里的定义启动 kernel.
因此两个后端的 cubin 可以互换, C++ 代码不需要知道 kernel 来自哪个后端.

接口包含以下内容, 两个后端完全相同:

* 每个 kernel 编译成一个 cubin 文件 ``<name>.cubin``.
* 入口符号为 ``<name>_0``. CuTe DSL 总会在 kernel 名后加 ``_<序号>``, 每个 cubin
  只有一个 kernel, 所以序号总是 0. TileLang 端导出同名符号.
* 参数的个数, 顺序和类型. 见 ``KERNELS``.
* 每个 block 的线程数 (``THREADS``) 和 grid 的计算规则 (``Kernel.grid``).

每个后端的 kernel 需要的动态共享内存大小不同, 这个值写在 ``manifest.txt`` 中.

所有矩阵都是行主序 (row-major), 元素连续存放. ``half`` 是 IEEE FP16, 累加使用 FP32.
C++ 端保证下列前提, kernel 不再检查:

* ``n_embd`` 是 64 的倍数, 不超过 ``MAX_EMBD``; 每个 head 的维度是 ``HEAD_DIM``.
* GEMM 的 ``n`` 是 ``TILE_N`` 的倍数, ``k`` 是 ``TILE_K`` 的倍数.
* 激活缓冲区至少有 ``round_up(m, TILE_M)`` 行. 多出的行可以是任意有限值,
  kernel 可能读取这些行, 但不会写入.
* KV 缓存至少有 ``round_up(pos0 + m, TILE_M)`` 行, 未写入的位置是有限值 (C++ 端清零).
* 所有指针按 16 字节对齐.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

ABI_VERSION = 1
THREADS = 128
TILE_M = 64
TILE_N = 64
TILE_K = 64
HEAD_DIM = 64
MAX_EMBD = 2048

# 参数类型. 指针是 64 位设备地址, 标量是 32 位.
POINTER_TYPES = ("f16*", "const f16*", "f32*", "const i32*")
SCALAR_TYPES = ("i32", "f32")
# PTX 中每种参数类型对应的类型名, 用于检查 CuTe DSL 生成的入口.
PTX_TYPES = {"i32": "u32", "f32": "f32", **dict.fromkeys(POINTER_TYPES, "u64")}
# TileLang 生成的 CUDA C++ 中每种参数类型对应的声明, 用于生成入口函数.
CUDA_TYPES = {
    "f16*": "half_t* __restrict__",
    "const f16*": "const half_t* __restrict__",
    "f32*": "float* __restrict__",
    "const i32*": "const int* __restrict__",
    "i32": "int",
    "f32": "float",
}


@dataclass(frozen=True)
class Param:
    name: str
    type: str

    @property
    def size(self) -> int:
        return 8 if self.type in POINTER_TYPES else 4


@dataclass(frozen=True)
class Kernel:
    name: str
    params: tuple[Param, ...]
    # grid 的三个维度, 用参数名写成表达式. C++ 端 (src/kernels.cc) 实现同样的规则.
    grid: tuple[str, str, str]
    doc: str

    @property
    def symbol(self) -> str:
        return f"{self.name}_0"

    @property
    def signature(self) -> str:
        """写入 manifest 的参数签名, C++ 端加载时逐字比较."""
        return ",".join(p.type.replace(" ", "_") for p in self.params)

    def param(self, name: str) -> Param:
        return next(p for p in self.params if p.name == name)


def _params(spec: str) -> tuple[Param, ...]:
    """把 ``"f16* y, const f16* x, i32 m"`` 解析为参数列表."""
    params = []
    for item in spec.split(","):
        *type_words, name = item.split()
        params.append(Param(name, " ".join(type_words)))
    for p in params:
        assert p.type in PTX_TYPES, p
    return tuple(params)


_GEMM = "f16* y, const f16* x, const f16* w, const f16* b, i32 m, i32 n, i32 k"
_GEMM_GRID = ("ceil(m / TILE_M)", "n / TILE_N", "1")

KERNELS = (
    Kernel(
        "gpt2_embedding",
        _params(
            "f16* x, const i32* ids, const f16* wte, const f16* wpe, i32 rows, i32 pos0, i32 n_embd"
        ),
        ("rows", "1", "1"),
        "x[r, :] = wte[ids[r], :] + wpe[pos0 + r, :]",
    ),
    Kernel(
        "gpt2_layernorm",
        _params(
            "f16* y, const f16* x, const f16* gamma, const f16* beta, i32 rows, i32 n_embd, f32 eps"
        ),
        ("rows", "1", "1"),
        "y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * gamma + beta",
    ),
    Kernel(
        "gpt2_linear_bias",
        _params(_GEMM),
        _GEMM_GRID,
        "y = x @ w^T + b; x 为 (m, k), w 为 (n, k), b 为 (n,)",
    ),
    Kernel(
        "gpt2_linear_bias_gelu",
        _params(_GEMM),
        _GEMM_GRID,
        "y = gelu_tanh(x @ w^T + b)",
    ),
    Kernel(
        "gpt2_linear_bias_residual",
        _params(_GEMM),
        _GEMM_GRID,
        "y = y + (x @ w^T + b); 原地加上残差",
    ),
    Kernel(
        "gpt2_lm_head",
        _params("f32* y, const f16* x, const f16* w, i32 m, i32 n, i32 k"),
        _GEMM_GRID,
        "y = x @ w^T, 输出 FP32 logits; w 为补齐到 TILE_N 倍数的词嵌入",
    ),
    Kernel(
        "gpt2_attention",
        _params(
            "f16* y, const f16* qkv, const f16* k_cache, const f16* v_cache, "
            "i32 m, i32 pos0, i32 n_head, f32 scale"
        ),
        ("ceil(m / TILE_M)", "n_head", "1"),
        "因果注意力. 第 i 个 query 位于 pos0 + i, 看到缓存中位置 0 .. pos0 + i 的 key. "
        "qkv 为 (m, 3 * n_head * HEAD_DIM); 缓存为 (位置, n_head * HEAD_DIM); "
        "调用前 C++ 端已把本次的 key 与 value 写入缓存",
    ),
)
KERNELS_BY_NAME = {k.name: k for k in KERNELS}


def write_manifest(out_dir: Path, backend: str, arch: str, smem: dict[str, int]) -> None:
    """写入 ``manifest.txt``. 格式简单, C++ 端逐行解析 (见 src/kernels.cc)."""
    lines = [
        "# learn-gpt2 cubin manifest",
        "# kernel <name> <symbol> <threads> <dynamic smem bytes> <param signature>",
        f"abi {ABI_VERSION}",
        f"backend {backend}",
        f"arch {arch}",
    ]
    for k in KERNELS:
        lines.append(f"kernel {k.name} {k.symbol} {THREADS} {smem[k.name]} {k.signature}")
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")


def check_ptx_entry(kernel: Kernel, ptx: str) -> None:
    """检查 PTX 中入口函数的名字和参数类型与 ABI 一致."""
    match = re.search(r"\.entry\s+(\w+)\s*\(([^)]*)\)", ptx)
    if match is None:
        raise ValueError(f"{kernel.name}: PTX 中没有 .entry")
    if match.group(1) != kernel.symbol:
        raise ValueError(f"{kernel.name}: 入口名为 {match.group(1)}, 预期 {kernel.symbol}")
    types = re.findall(r"\.param\s+\.(\w+)", match.group(2))
    expected = [PTX_TYPES[p.type] for p in kernel.params]
    if types != expected:
        raise ValueError(f"{kernel.name}: PTX 参数类型为 {types}, 预期 {expected}")


def cubin_param_sizes(cuobjdump_elf: str, symbol: str) -> list[tuple[int, int, int]]:
    """从 ``cuobjdump -elf`` 的输出中读取 kernel 参数的 (序号, 偏移, 字节数)."""
    # 段内容是段名下方缩进的行. 段表和 .nv.merc.nv.info.* 也含有段名, 用行首锚定排除.
    section = re.search(
        rf"^\.nv\.info\.{re.escape(symbol)}\n((?:[ \t].*\n?)*)", cuobjdump_elf, re.MULTILINE
    )
    if section is None:
        raise ValueError(f"cubin 中没有符号 {symbol} 的 .nv.info 段")
    params = re.findall(
        r"Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)",
        section.group(1),
    )
    return sorted((int(o, 16), int(off, 16), int(s, 16)) for o, off, s in params)


def expected_param_sizes(kernel: Kernel) -> list[tuple[int, int, int]]:
    """按 C ABI 排列参数: 每个参数按自身大小对齐."""
    result, offset = [], 0
    for i, p in enumerate(kernel.params):
        offset = (offset + p.size - 1) // p.size * p.size
        result.append((i, offset, p.size))
        offset += p.size
    return result


def check_cubins(kernel_dirs: list[Path]) -> None:
    """检查每个后端的 cubin 都导出 ABI 规定的符号和参数布局. 需要 ``cuobjdump``."""
    import subprocess

    for directory in kernel_dirs:
        manifest = (directory / "manifest.txt").read_text()
        for k in KERNELS:
            line = next(
                (x for x in manifest.splitlines() if x.startswith(f"kernel {k.name} ")), None
            )
            if line is None:
                raise ValueError(f"{directory}: manifest 中缺少 {k.name}")
            fields = line.split()
            if fields[2] != k.symbol or int(fields[3]) != THREADS or fields[5] != k.signature:
                raise ValueError(f"{directory}: manifest 中 {k.name} 的接口与 ABI 不一致")
            elf = subprocess.run(
                ["cuobjdump", "-elf", str(directory / f"{k.name}.cubin")],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            actual = cubin_param_sizes(elf, k.symbol)
            if actual != expected_param_sizes(k):
                raise ValueError(
                    f"{directory / k.name}.cubin: 参数布局 {actual}, 预期 {expected_param_sizes(k)}"
                )
        print(f"{directory}: {len(KERNELS)} 个 kernel 与 ABI v{ABI_VERSION} 一致")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 cubin 是否符合 kernel ABI")
    parser.add_argument("kernel_dirs", type=Path, nargs="+", help="含 manifest.txt 的目录")
    args = parser.parse_args()
    check_cubins(args.kernel_dirs)


if __name__ == "__main__":
    main()
