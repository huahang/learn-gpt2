# Repository Guidelines

## 项目结构与模块组织

本仓库用于学习 GPT-2, 使用 Python / Transformers / PyTorch. 当前代码仍处于初始阶段.

- `src/learn_gpt2/`: Python 包. `__init__.py` 中的 `main()` 是命令行入口, 调用 `pipeline.py` 中的交互式文本生成循环. `gpt2.py` 是自行实现的 GPT-2, 含推理用的 `KVCache` 与 `GPT2.generate()`. `qwen3.5_0.8b.py` 是自行实现的 Qwen3.5-0.8B 文本模型 (Gated DeltaNet 与全注意力混合), 结构与参数名同 HF `Qwen3_5ForCausalLM` 一致, 含 `HybridCache` 与 `Qwen3_5.from_pretrained()`; 文件名含点号, 需用 `importlib.util.spec_from_file_location` 加载 (见 `tests/test_qwen3_5.py`).
- `presentations/`: 讲解材料 (PPT, 视频及其生成脚本). 其中 `*.mp4` 与 `*.pptx` 由 Git LFS 管理 (见 `.gitattributes`), 克隆后需 `git lfs install` 再 `git lfs pull`.
- `gpt2_hf.ipynb` / `qwen3.5_0.8b.ipynb`: 根目录中的实验笔记本. 将可复用逻辑放入包中, 让笔记本专注于实验与展示.
- `pyproject.toml`: 项目元数据, 依赖, 构建配置与硬件后端配置.
- `uv.lock`: 依赖锁文件. 修改依赖时同步更新.
- `README.md`: 当前为空, 新增使用流程时补充说明.

- `tests/`: pytest 测试, 使用随机初始化的小模型, 不需要下载权重.

当前没有独立资源目录.

## 构建与开发命令

使用 Python 3.14 或更新版本, 通过 `uv` 管理环境.

- `uv sync`: 安装基础依赖并同步虚拟环境.
- `uv sync --extra cuda`: 安装 NVIDIA GPU 后端.
- `uv sync --extra rocm`: 安装 Linux AMD GPU 后端.
- `uv sync --extra mps`: 安装 macOS Apple Silicon 后端.
- `uv run learn-gpt2`: 运行已配置的命令行入口.
- `uv run pytest`: 运行 `tests/` 中的测试. 测试依赖 PyTorch, 需先安装一个硬件后端, 或直接使用 `uv run --extra cuda pytest` (按硬件换成 `rocm` / `mps`).
- `uv run jupyter lab`: 启动本地笔记本开发环境.
- `uv build`: 使用 `uv_build` 构建源码包与 wheel.
- `uv run ruff check`: 对 `.py` 与 `.ipynb` 运行 lint; 加 `--fix` 自动修复可修复的问题.
- `uv run ruff format`: 格式化 `.py` 与 `.ipynb`; 加 `--check` 仅检查不修改.

三个硬件后端互斥, 只选择一个. 不需要 PyTorch 的操作可仅安装基础依赖.

## 语言与文风

- 对话, 工作记录, 文档和报告默认用中文.
- 专业术语按需附英文, 例如: 工作树 (worktree).
- 借鉴 ASD-STE100 (简化技术英语), 用自然中文表达.
- 用短句, 常用词和主动语态. 一句话只表达一个主要意思.
- 写清主体, 动作和条件, 避免指代不明.
- 同一概念始终用同一术语, 不引入多余术语.
- 删掉多余修饰, 不在名词前堆叠定语.

## 代码风格与命名

Python 使用 4 个空格缩进, 保持现有双引号风格. 函数和变量使用 `snake_case`, 类使用 `PascalCase`, 常量使用 `UPPER_SNAKE_CASE`. 新增函数应提供清晰的类型注解, 与现有 `main() -> None` 保持一致. 使用 Ruff 作为 lint 与格式化工具, 配置位于 `pyproject.toml` 的 `[tool.ruff]`, 行宽 100, 同时覆盖 `.py` 与 `.ipynb`. `ruff` 在 `dev` 依赖组中, `uv sync` 默认安装. 提交前确保 `uv run ruff check` 与 `uv run ruff format --check` 通过. 当前未配置 pre-commit 或 CI; 不要声称相关检查会自动执行.

## 测试与验证

测试框架为 pytest (在 `dev` 依赖组中), 当前没有覆盖率门槛. 修改 `gpt2.py` 后运行 `uv run pytest`; 修改入口后运行 `uv run learn-gpt2`; 修改打包配置后运行 `uv build`. 修改笔记本后重启内核, 按顺序执行相关单元并检查结果. 新增测试使用 `tests/test_*.py` 命名.

## 提交与 Pull Request

提交信息必须遵循 [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/), 格式为 `<type>[optional scope][!]: <description>`. 类型后必须有冒号和空格; scope 和 `!` 可省略.

- 新功能使用 `feat`, 缺陷修复使用 `fix`. 其他常用类型包括 `docs` / `chore` / `refactor` / `test` / `build` / `ci`.
- 类型统一使用小写. 描述沿用现有提交的英文祈使句风格, 每个提交聚焦一个目的. 例如: `docs: update project README` 或 `fix(cli): handle invalid arguments`.
- 不兼容变更必须在冒号前加 `!`, 或在 footer 中使用 `BREAKING CHANGE: <description>` 标记. 正文和 footer 为可选项, 与前面的内容用空行分隔.

PR 应说明改动目的, 涉及的模块, 实际执行的验证命令及结果. 涉及 GPU 行为时注明硬件后端; 有关联 issue 时附链接. 不要提交 `.venv/`, 密钥, 下载的模型权重或大型实验输出.
