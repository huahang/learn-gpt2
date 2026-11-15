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

[注意力机制、GPT-2 与 GPT-3 三篇论文综述](presentations/03-gpt2-gpt3-overview/README.md)：中文综述、46 页 PPT、约 42 分钟配音视频和可运行的因果自注意力示例。

[《展开讲讲 Causal Self Attention：GPT-2 因果自注意力》](presentations/04-gpt2-causal-self-attention/README.md)：17 页 PPT 与 12:30 中文配音视频，讲解 `src/learn_gpt2/gpt2.py` 的 `CausalSelfAttention`（L92-122）。

[《从注意力到大语言模型：Transformer、GPT-2 与 GPT-3》](presentations/05-transformer-gpt2-gpt3/README.md)：中文综述、37 页 PPT 与 24 分 13 秒中文配音视频，重点展开因果自注意力和 KV Cache。
