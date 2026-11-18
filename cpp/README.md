# C++ 推理框架

本目录用 C++ 实现 GPT-2 的 FP16 推理. 程序从 safetensors 文件加载权重, 用预先编译的
cubin kernel 计算. kernel 有两个版本: 一个用 TileLang 编写, 一个用 CuTe DSL 编写.
两个版本的 cubin 接口相同, 运行时用 `--backend` 选择.

运行时只依赖 CUDA 驱动 (libcuda) 和仓库中的 [safetensors-cpp](third_party/safetensors-cpp/).
运行时不需要 Python, PyTorch, CUDA runtime 或 nvcc. Python 只在构建 cubin 时使用.

## 目录结构

```text
cpp/
├── Makefile                  构建规则
├── build.sh                  检查工具后调用 make
├── kernels/
│   ├── kernel_abi.py         kernel 接口 (ABI) 的定义, 以及检查 cubin 的工具
│   ├── build_tilelang.py     TileLang 版本的 kernel, 导出为 cubin
│   └── build_cute_dsl.py     CuTe DSL 版本的 kernel, 导出为 cubin
├── src/
│   ├── main.cc               命令行程序: 分词, 采样, 生成循环
│   ├── model.{h,cc}          加载权重, 管理 KV 缓存, 计算前向
│   ├── kernels.{h,cc}        按 manifest.txt 加载 cubin, 按 ABI 启动 kernel
│   ├── tokenizer.{h,cc}      GPT-2 的字节级 BPE 分词器
│   ├── json.{h,cc}, utf8.h   读取 config.json 与 vocab.json
│   ├── half.h                FP32 / BF16 到 FP16 的转换 (舍入到最近的偶数)
│   └── cuda_utils.{h,cc}     driver API 的错误检查, context, 设备内存
├── third_party/safetensors-cpp/   上游代码, 来源见 VENDOR.md
└── build/                    构建输出 (不入库)
    ├── bin/gpt2
    └── kernels/{tilelang,cute_dsl}/   每个 kernel 一个 .cubin, 另有 manifest.txt 和源码
```

## 构建

需要:

- NVIDIA GPU, 计算能力 8.0 或更新 (Tensor Core 的 `mma.sync`), 以及驱动.
- CUDA Toolkit: 提供 `cuda.h` 和 libcuda 的 stub; TileLang 后端还需要 `nvcc`.
- 支持 C++20 与 `std::format` 的编译器, 例如 GCC 13 或 Clang 17 及更新的版本.
- [uv](https://docs.astral.sh/uv/) 与本项目的 `cuda` 依赖 (TileLang, CuTe DSL):
  `uv sync --extra cuda`.

在仓库根目录执行:

```bash
cpp/build.sh
```

`build.sh` 的选项:

| 选项 | 说明 |
| --- | --- |
| `--arch sm_XX` | GPU 架构. 默认读取第 0 号 GPU; SM90 及更新的架构加后缀 `a`, 例如 `sm_120a` |
| `--backend NAME` | 只构建一个后端: `tilelang` 或 `cute_dsl` |
| `--clean` | 先删除 `build/` |
| `--check` | 构建后检查 cubin 接口, 并运行测试 |

也可以直接使用 make:

```bash
make -C cpp                    # cubin 与程序
make -C cpp kernels            # 只构建 cubin
make -C cpp check              # 检查接口并运行测试
make -C cpp ARCH=sm_86         # 为其他架构构建 cubin
```

修改 kernel 脚本或 `ARCH` 后, make 会重新生成 cubin.

## 运行

`--model` 接受 safetensors 文件, 或 Hugging Face 检查点目录. 目录中的 `vocab.json` 与
`merges.txt` 用于分词, `config.json` 提供 `n_head` 与 `layer_norm_epsilon`.

```bash
MODEL=$(ls -d ~/.cache/huggingface/hub/models--gpt2/snapshots/*/)
cpp/build/bin/gpt2 --model "$MODEL" --backend cute_dsl --prompt "Hello, I am a language model,"
```

输出:

```text
[gpt2] 设备 NVIDIA RTX PRO 4000 Blackwell SFF Edition, kernel 后端 cute_dsl (sm_120a), n_layer=12 n_head=12 n_embd=768 n_ctx=1024, 加载用时 715 ms
Hello, I am a language model, not a language model. I am a language model. ...
[gpt2] 提示词 8 个 token, 预填充 1.1 ms, 解码 0.99 ms/token (1006.2 token/s)
```

常用选项, 完整列表见 `gpt2 --help`:

| 选项 | 说明 |
| --- | --- |
| `--backend tilelang\|cute_dsl` | 选择 `build/kernels/` 下的 cubin 目录. `--kernels DIR` 可指定任意目录 |
| `--prompt TEXT` / `--tokens 1,2,3` | 文本提示词, 或直接给出 token 编号 |
| `--max-new-tokens N` | 生成的 token 数, 默认 32 |
| `--temperature T`, `--top-k K`, `--seed S` | 采样参数. 温度为 0 时贪心解码 (默认) |
| `--weight-layout pytorch` | 权重为 `nn.Linear` 布局, 例如本仓库 `GPT2.state_dict()` 保存的文件 |
| `--print-tokens` | 输出 token 编号, 不输出文本 |
| `--dump-logits FILE` | 把提示词每个位置的 logits 以 FP32 写入文件, 用于测试 |

## kernel 接口

接口在 [`kernels/kernel_abi.py`](kernels/kernel_abi.py) 中定义. 两个后端都按它导出 cubin,
C++ 端 ([`src/kernels.cc`](src/kernels.cc)) 也按它启动 kernel.

| kernel | 参数 | grid | 计算 |
| --- | --- | --- | --- |
| `gpt2_embedding` | `f16* x, const i32* ids, const f16* wte, const f16* wpe, i32 rows, i32 pos0, i32 n_embd` | `(rows)` | `x[r] = wte[ids[r]] + wpe[pos0 + r]` |
| `gpt2_layernorm` | `f16* y, const f16* x, const f16* gamma, const f16* beta, i32 rows, i32 n_embd, f32 eps` | `(rows)` | LayerNorm |
| `gpt2_linear_bias` | `f16* y, const f16* x, const f16* w, const f16* b, i32 m, i32 n, i32 k` | `(⌈m/64⌉, n/64)` | `y = x wᵀ + b` |
| `gpt2_linear_bias_gelu` | 同上 | 同上 | `y = gelu(x wᵀ + b)` |
| `gpt2_linear_bias_residual` | 同上 | 同上 | `y += x wᵀ + b` |
| `gpt2_lm_head` | `f32* y, const f16* x, const f16* w, i32 m, i32 n, i32 k` | `(⌈m/64⌉, n/64)` | `y = x wᵀ`, FP32 输出 |
| `gpt2_attention` | `f16* y, const f16* qkv, const f16* k_cache, const f16* v_cache, i32 m, i32 pos0, i32 n_head, f32 scale` | `(⌈m/64⌉, n_head)` | 因果 FlashAttention |

两个后端的以下内容完全相同:

- 文件名 `<kernel>.cubin` 与入口符号 `<kernel>_0`. CuTe DSL 总会在 kernel 名后加
  `_<序号>`; 每个 cubin 只有一个 kernel, 所以序号总是 0, TileLang 端导出同名符号.
- 参数的个数, 顺序和类型. 指针是 64 位设备地址, 整数与浮点数是 32 位.
- 每个 block 128 个线程, grid 按上表计算.

每个后端需要的动态共享内存不同, 由同目录的 `manifest.txt` 记录. 所有形状都是运行时参数,
所以同一组 cubin 可用于 GPT-2 的四种尺寸. C++ 端保证 kernel 的前提条件:
`n_embd` 是 64 的倍数且不超过 2048, 每个 head 64 维; GEMM 的 `n` 与 `k` 是 64 的倍数
(词表补齐到 64 的倍数); 激活缓冲区的行数补齐到 64 的倍数; KV 缓存清零.

两个后端用不同的方法满足同一个接口:

- **TileLang**: TileLang 生成的入口函数按名字排列参数, 并省去没有用到的参数.
  `build_tilelang.py` 把它改为 `__device__` 函数, 再加一个按 ABI 排列参数的
  `extern "C"` 入口, 然后用 nvcc 编译. 生成的 `.cu` 保留在输出目录中.
  数据搬运关闭了 TMA, 所以参数中没有 `CUtensorMap`. 形状是运行时参数, 所以 kernel 用
  `T.assume(k % 64 == 0)` 等提示告诉 TileLang ABI 保证的整除关系; 没有这些提示时,
  TileLang 无法证明每行按 16 字节对齐, 只能逐个元素读取, GEMM 慢数倍.
- **CuTe DSL**: kernel 的参数是 `cute.Pointer`, `Int32` 与 `Float32`, 在 PTX 中就是
  `.u64`, `.u32` 与 `.f32`. kernel 内部用运行时的形状构造张量, 用 `cute.assume` 声明整除关系,
  使 128 位访问仍能向量化. `set_name_prefix` 设置入口符号. CuTe DSL 自带编译器, 直接产出 cubin.

接口在三处检查:

1. 构建时: `build_cute_dsl.py` 检查 PTX 入口的名字和参数类型; `build_tilelang.py` 检查
   TileLang 生成的参数都在 ABI 中, 线程数为 128.
2. `make check-abi`: 用 `cuobjdump` 读出每个 cubin 的参数布局, 与 ABI 比较.
3. 加载时: C++ 端比较 manifest 中的参数签名, 再用 `cuFuncGetParamInfo` 读出 cubin 中每个参数
   的字节数.

## 前向计算

[`Gpt2Model::forward`](src/model.cc) 对 `m` 个新 token (位置 `pos0` 到 `pos0 + m - 1`) 依次启动:

```text
embedding
每层: layernorm -> linear_bias (QKV) -> 复制 K, V 到缓存 -> attention
      -> linear_bias_residual (投影) -> layernorm -> linear_bias_gelu -> linear_bias_residual
layernorm (生成时只算最后一个位置) -> lm_head
```

K 与 V 用 `cuMemcpy2DAsync` 从 QKV 输出复制到缓存, 不需要 kernel. 生成循环与
`learn_gpt2.gpt2.GPT2.generate` 相同: 先预填充整个提示词, 之后每步只计算一个 token;
序列超过上下文长度后, 每步重新计算最后 `n_ctx` 个 token.

## 验证

- `tests/test_cpp_kernels.py`: 用 CUDA driver API 加载两个后端的 cubin, 逐个与 PyTorch 比较.
  两个后端共用同一套启动代码.
- `tests/test_cpp_inference.py`: 用随机初始化的小模型运行 `gpt2`, 检查 logits, 贪心生成,
  采样与 `nn.Linear` 布局的权重.

两个测试文件在没有构建产物时自动跳过. 运行: `make -C cpp check`.

另外用 Hugging Face 缓存中的四种 GPT-2 权重与 `transformers` 的 FP32 结果比较过 (不在测试中,
因为需要下载权重):

- 分词: 含英文标点, 缩写, 连续空白, 带重音的拉丁字母, 中文和 emoji 的 5 段文本, token 编号完全相同.
- logits: 135 个位置的 argmax 全部一致. GPT-2 small 的平均绝对误差约 0.1, 与仓库中的 Python
  FP16 实现相当, 小于 PyTorch FP16 推理的 0.19.
- 贪心生成 40 个 token: 两个后端, 四种尺寸, 两条提示词, 结果与 `transformers` 逐 token 相同.

## 性能

RTX PRO 4000 Blackwell SFF (SM120, 70 W). 两个后端交替运行 4 轮, 每格取最小值.
decode 为每步时间, "16 + 256" 表示提示长 16, 生成 256 个 token. 这块 GPU 持续负载时
数据约有 10% 的波动.

| 场景 | GPT-2 small, TileLang | GPT-2 small, CuTe DSL | GPT-2 XL, TileLang | GPT-2 XL, CuTe DSL |
| --- | --- | --- | --- | --- |
| decode, 16 + 256 | 1.10 ms | 1.02 ms | 9.82 ms | 9.74 ms |
| decode, 768 + 128 | 1.39 ms | 1.20 ms | 11.14 ms | 10.76 ms |
| 预填充 1 x 128 | 1.30 ms | 1.20 ms | 12.20 ms | 11.90 ms |
| 预填充 1 x 1024 | 4.90 ms | 4.70 ms | 72.70 ms | 69.60 ms |

decode 受显存带宽限制, 两个后端的 GEMM 速度接近. TileLang 的注意力慢一些: K 为运行时值时,
TileLang 0.1.15 无法为两级以上的 `cp.async` 流水线生成代码, 所以注意力的读取与计算不重叠,
GEMM 也只用两级流水线.

## 限制

- 一次只处理一个序列 (batch 为 1).
- 只读取单个 safetensors 文件, 不支持分片检查点. GPT-2 的四种尺寸都是单个文件.
- 分词器不带 Unicode 数据表. ASCII 文本的切分是精确的; 其他字符按常见区段近似分类,
  罕见字符的切分可能与 Hugging Face 不同.
- 解码时注意力也按 64 个 query 一块计算, 没有 split-KV, 没有 CUDA Graph, 没有 LayerNorm 折叠.
  仓库中的 Python 实现有这些优化, GPT-2 small 的 decode 约快 20%.
