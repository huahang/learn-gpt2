# learn-gpt2

用于学习 GPT-2 的 Python 项目, 使用 Transformers / PyTorch 和 Jupyter 笔记本开展实验.

项目目前处于初始化阶段. 命令行入口仅输出问候信息; `gpt2_hf.ipynb` 目前只有导入 `GPT2LMHeadModel` 的代码单元, 尚未包含模型加载, 文本生成或训练流程.

## 环境要求

- Python >= 3.14. 仓库通过 `.python-version` 指定 Python 3.14.
- 已安装 `uv`, 用于管理 Python 环境, 依赖和构建.
- 运行 PyTorch 实验时, 根据本机硬件选择对应后端, 并准备兼容的驱动与运行环境.

以下命令均在仓库根目录执行.

## 安装依赖

只运行项目入口或使用基础开发工具时:

```bash
uv sync
```

基础依赖包含 Transformers, JupyterLab, Notebook, Voilà 和 ipywidgets. PyTorch 通过可选依赖安装. 根据硬件选择下列一条命令:

| 硬件 / 系统 | 安装命令 | 项目配置的包来源 |
| --- | --- | --- |
| NVIDIA GPU, Linux / Windows | `uv sync --extra cuda` | PyTorch CUDA 13.2 索引 |
| AMD GPU, Linux | `uv sync --extra rocm` | PyTorch ROCm 7.2 索引, 包含 `triton-rocm` |
| Apple Silicon, macOS | `uv sync --extra mps` | 默认包索引中的 PyTorch |

`cuda` / `rocm` / `mps` 三个可选依赖互斥, 不要同时启用. 安装结果取决于对应索引是否提供适合本机平台和 Python 版本的软件包. 当前未定义单独的 CPU 可选依赖.

## 运行项目

检查命令行入口:

```bash
uv run learn-gpt2
```

预期输出:

```text
Hello from learn-gpt2!
```

启动基础笔记本环境:

```bash
uv run jupyter lab
```

运行需要 PyTorch 的实验时, 在运行命令中显式指定所选后端. 例如, AMD GPU 环境使用:

```bash
uv run --extra rocm jupyter lab
```

NVIDIA GPU 或 Apple Silicon 环境分别将 `rocm` 替换为 `cuda` 或 `mps`. 在 JupyterLab 中打开 `gpt2_hf.ipynb`, 选择项目环境的 Python 内核并执行代码单元. 选择安装后端不会自动为实验代码设置计算设备.

## 项目结构

```text
.
├── src/learn_gpt2/
│   └── __init__.py    # main() 命令行入口
├── gpt2_hf.ipynb      # GPT-2 实验笔记本
├── pyproject.toml     # 依赖, 后端与构建配置
├── uv.lock            # 依赖锁文件
├── .python-version    # Python 版本
└── AGENTS.md          # 贡献与开发约定
```

## 开发与验证

将可复用逻辑放入 `src/learn_gpt2/`, 将实验过程与展示保留在笔记本中. 修改依赖后, 同步更新 `uv.lock`.

构建源码包和 wheel:

```bash
uv build
```

构建产物位于 `dist/`. 当前没有自动化测试框架或覆盖率门槛. 修改入口后检查命令行输出; 修改笔记本后重启内核, 按顺序执行代码单元并检查结果. 更多约定见 [AGENTS.md](AGENTS.md).

## 论文讲解材料

三篇论文综述包含中文综述、54 页 PPT、约 49 分钟的普通话讲解和数值演示。视频从 05:11 讲因果自注意力，从 28:47 讲 KV Cache。[注意力机制、GPT-2 与 GPT-3 三篇论文综述](presentations/03-gpt2-gpt3-overview/README.md)

[《展开讲讲 Causal Self Attention：GPT-2 因果自注意力》](presentations/04-gpt2-causal-self-attention/README.md)：17 页 PPT 与 12:30 中文配音视频，讲解 `src/learn_gpt2/gpt2.py` 的 `CausalSelfAttention`（L92-122）。

[《从注意力到大语言模型：Transformer、GPT-2 与 GPT-3》](presentations/06-transformer-gpt2-gpt3/README.md)：中文综述、37 页 PPT 与 24 分 13 秒中文配音视频，重点展开因果自注意力和 KV Cache。

## Triton FP16 推理

`src/learn_gpt2/gpt2_inference_triton.py` 用 Triton 语言实现 GPT-2 的推理算子, 并从 safetensors 文件加载权重. 权重和激活固定为 FP16; GEMM 与归约用 FP32 累加, 输出的 logits 为 FP32. 该实现需要 NVIDIA GPU 和 `cuda` 后端.

先运行 `gpt2_hf.ipynb` 的 Cell #1, 生成 `gpt2_hf.safetensors`. 然后:

```python
import tiktoken
import torch

from learn_gpt2.gpt2_inference_triton import GPT2Inference

enc = tiktoken.get_encoding("gpt2")
model = GPT2Inference.from_safetensors("gpt2_hf.safetensors")
tokens = torch.tensor([enc.encode("Hello, I'm a language model,")])
logits = model(tokens)  # (batch, length, vocab), FP32
output = model.generate(tokens, max_new_tokens=64, temperature=0.8, top_k=50)
print(enc.decode(output[0].tolist()))
```

`generate()` 的参数含义与 `GPT2.generate()` 相同. 权重文件采用 Hugging Face 布局 (Conv1D 权重为 `(in, out)`). 四种公开尺寸之外的模型需要传入 `n_head`. 首次运行会编译内核 (约 3 秒), 之后使用 Triton 的磁盘缓存.

主要优化:

- GEMM: 所有矩阵乘都用分块 `tl.dot`, 在 Tensor Core 上计算. 这包括每个 block 的 4 个线性层, LM head 和注意力中的两次乘法. 内层循环的加载不带掩码. 权重按 `(out, in)` 存放, 每个程序读连续的行. 块大小按矩阵形状选择.
- 算子融合: QKV 投影直接写入 KV cache. 注意力采用 FlashAttention. bias, GELU 和残差并入 GEMM 的尾处理. decode 步把 LayerNorm 并入后续 GEMM.
- KV cache 与静态工作区: 所有缓冲区只分配一次.
- 生成时只计算末位 logits.
- CUDA Graph: 一个 decode 步只需回放一次图.

实测数据 (GPT-2 small, RTX PRO 4000 Blackwell SFF, 对照 FP16 的 `gpt2.py`):

| 场景 | `gpt2.py` | Triton |
| --- | --- | --- |
| decode, 5 条序列, 每步 | 1.19 ms | 0.96 ms |
| 提示 1 x 1024 | 9.9 ms | 4.3 ms |
| 提示 8 x 1024 | 69 ms | 27 ms |
| 提示 32 x 128 | 33 ms | 12 ms |

decode 步受显存带宽限制: 每步要读一遍全部权重 (约 250 MB). 这块 GPU 的功耗上限是 70 W, 持续负载下数据约有 10% 的波动.

验证命令:

```bash
uv run --extra cuda pytest tests/test_gpt2_inference_triton.py
```

测试使用随机初始化的小模型, 不下载权重. 没有 GPU 或没有 Triton 时, 测试自动跳过.

## TileLang FP16 推理

`src/learn_gpt2/gpt2_inference_tilelang.py` 从本地 safetensors 加载 GPT-2 权重。
该实现需要 NVIDIA CUDA GPU、FP16 Tensor Core（计算能力至少 7.5）和可用的 `nvcc`。
安装依赖后，可使用仓库中的 Hugging Face 布局权重生成文本：

```bash
uv sync --extra cuda
uv run --extra cuda python -m learn_gpt2.gpt2_inference_tilelang \
  gpt2_hf.safetensors --prompt "Hello, I am" --max-new-tokens 64
```

默认使用贪心解码和 CUDA Graph。随机采样可加 `--temperature 0.8 --top-k 40`。
首次运行会编译内核；首次获取 GPT-2 tokenizer 时，tiktoken 可能需要下载编码文件。
可用 `--no-cuda-graph` 关闭图捕获。模型权重和中间激活固定为 FP16。
GEMM 与归约使用 FP32 累加，输出 logits 保留为 FP32。

Python 接口：

```python
import torch
from learn_gpt2.gpt2_inference_tilelang import GPT2Inference

model = GPT2Inference.from_safetensors("gpt2_hf.safetensors")
tokens = torch.tensor([[15496, 11, 314, 716]], dtype=torch.int64)
logits = model(tokens)  # (batch, length, vocab)
output = model.generate(tokens, max_new_tokens=64, temperature=0)
```

也可以传入包含 `model.safetensors` 或 `model.safetensors.index.json` 的 HF 目录。
非标准模型需提供 `n_head`，或在同目录的 `config.json` 中保存该值。
HF Conv1D 权重默认采用 `(in, out)` 布局；从本仓库 `GPT2.state_dict()` 保存的权重
必须指定 `weight_layout="pytorch"`（命令行使用 `--weight-layout pytorch`）。

主要优化包括合并 QKV GEMM 并直接写缓存、FlashAttention、split-KV 解码、
融合 bias/GELU/residual、GEMM 软件流水线和 L2 swizzle、工作区复用、
仅计算末位 logits，以及 CUDA Graph。超过 GPT-2 的绝对位置范围后，逐步重算最后一个上下文窗口。
同一个模型实例复用工作区，不能并发处理多个请求。块大小使用启发式配置，性能需在目标 GPU 上测量。
实现参考 [TileLang GEMM 示例](https://github.com/tile-ai/tilelang/blob/main/examples/quickstart.py)
和 [FlashAttention 示例](https://github.com/tile-ai/tilelang/blob/main/examples/flash_attention/example_mha_fwd_bshd.py)。

验证命令：

```bash
uv run --extra cuda pytest tests/test_gpt2_inference_tilelang.py
# 无 GPU 时，仍可使用本地 CUDA 工具链检查所有内核的编译。
LEARN_GPT2_TILELANG_COMPILE_TESTS=1 uv run --extra cuda pytest tests/test_gpt2_inference_tilelang.py
```

测试使用随机初始化的小模型，不下载模型权重。未检测到 CUDA GPU 时，跳过数值对照与图捕获测试。

## CuTe DSL FP16 推理

`src/learn_gpt2/gpt2_inference_cute_dsl.py` 用 CuTe DSL (CUTLASS 4 的 Python DSL, `import cutlass`) 实现 GPT-2 的推理算子, 并从 safetensors 文件加载权重. 权重和激活固定为 FP16; GEMM 与归约用 FP32 累加, 输出的 logits 为 FP32. 内核使用 warp 级 `mma.sync` 和 `cp.async`, 需要 SM80 或更新的 NVIDIA GPU 和 `cuda` 后端. 模型尺寸需满足 `n_embd % 64 == 0` 与 `head_size % 32 == 0`; 四种公开的 GPT-2 尺寸都满足.

先运行 `gpt2_hf.ipynb` 的 Cell #1, 生成 `gpt2_hf.safetensors`. 然后:

```python
import tiktoken
import torch

from learn_gpt2.gpt2_inference_cute_dsl import GPT2Inference

enc = tiktoken.get_encoding("gpt2")
model = GPT2Inference.from_safetensors("gpt2_hf.safetensors")
tokens = torch.tensor([enc.encode("Hello, I'm a language model,")])
logits = model(tokens)  # (batch, length, vocab), FP32
output = model.generate(tokens, max_new_tokens=64, temperature=0.8, top_k=50)
print(enc.decode(output[0].tolist()))
```

`generate()` 的参数含义与 `GPT2.generate()` 相同. 权重文件采用 Hugging Face 布局 (Conv1D 权重为 `(in, out)`), 也可以直接加载 Hugging Face 缓存中的 `model.safetensors`. 四种公开尺寸之外的模型需要传入 `n_head`. 每种内核在进程中首次使用时编译 (GPT-2 small 共约 3 秒), 同一进程中的模型共享编译结果. `cuda_graph=False` 关闭图捕获, `pdl=False` 关闭 PDL.

主要优化:

- GEMM: 所有矩阵乘都在 Tensor Core 上计算 (`mma.sync` m16n8k16). 这包括每个 block 的 4 个线性层, LM head 和注意力中的两次乘法. `cp.async` 多级流水线把分块从显存搬到共享内存. 共享内存采用 swizzle 布局, 避免 bank 冲突. `ldmatrix` 把操作数读入寄存器, 并与 `mma` 双缓冲. 块大小按矩阵形状选择.
- LayerNorm 折叠: 加载权重时, `ln_1` 与 `ln_2` 的增益和偏移折进下一个 GEMM 的权重和 bias. GEMM 的尾处理用每行的均值和 1/std 完成归一化, 不写出归一化后的激活. decode 步的 GEMM 直接从寄存器中的 A 分片计算行统计量, 不需要额外内核.
- 尾处理融合: bias, GELU 和残差并入 GEMM 的尾处理. QKV 投影直接写入 KV cache. 累加结果先暂存在共享内存, 再按连续向量写出; 每个元素只舍入一次.
- 注意力: FlashAttention 式在线 softmax. P 矩阵留在寄存器中, 直接作为下一次 `mma` 的输入. decode 步把 KV 序列分给多个 CTA (split-KV), 再按 log-sum-exp 合并.
- 常量折叠: 注意力缩放 1/sqrt(head_size) 折进 query 权重.
- 布局与内存: KV cache 的布局与 QKV GEMM 的输出相同, 全程没有转置. 静态工作区只分配一次. 行数和词表都补齐到整块, 所以加载不需要边界检查.
- 生成时只计算末位 logits.
- PDL (programmatic dependent launch): 后一个 GEMM 在前一个内核结束前预取权重.
- CUDA Graph: 一个 decode 步只需回放一次图. 内核通过 TVM-FFI 调用, Python 端开销很小.

实测数据 (GPT-2 small, RTX PRO 4000 Blackwell SFF, 三种实现在同一个脚本中依次运行). decode 为每步时间, 提示长 16, 生成 256 个 token. 提示一栏为一次预填充的时间; `gpt2.py` 在这一栏计算全部位置的 logits, 另外两种实现只计算末位 logits.

| 场景 | `gpt2.py` FP16 | Triton | CuTe DSL |
| --- | --- | --- | --- |
| decode, 1 条序列, 每步 | 1.07 ms | 0.85 ms | 0.81 ms |
| decode, 8 条序列, 每步 | 1.19 ms | 0.96 ms | 0.93 ms |
| 提示 1 x 1024 | 9.76 ms | 4.32 ms | 4.02 ms |
| 提示 8 x 512 | 34.33 ms | 13.52 ms | 12.57 ms |

精度: 以 FP32 的 `gpt2.py` 为基准, logits 的平均绝对误差小于 PyTorch 的 FP16 推理 (GPT-2 small: 0.023 对 0.032). 四种公开尺寸各用 2 条随机提示贪心生成 40 个 token, 结果与 FP32 逐 token 一致.

decode 步受显存带宽限制: 每步要读一遍全部权重 (约 250 MB), 其中 LM head 约占 190 µs.

验证命令:

```bash
uv run --extra cuda pytest tests/test_gpt2_inference_cute_dsl.py
```

测试使用随机初始化的小模型, 不下载权重. 没有 NVIDIA GPU 或没有安装 CuTe DSL 时, 测试自动跳过.
