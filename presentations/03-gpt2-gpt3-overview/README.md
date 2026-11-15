# 注意力机制、GPT-2 与 GPT-3 三篇论文综述

从向量、加权平均与下一 token 预测出发，解释三篇论文的研究问题、方法、训练数据、实验与局限。适合了解基本向量和概率的学习者。第 7–14 页讲因果自注意力，第 33–39 页讲 KV Cache。

## 成品

| 文件 | 内容 |
| --- | --- |
| [review.html](review.html) | 中文综述，带离线目录、表格及公式 |
| [review.md](review.md) | 综述源文，含详细引用定位、自测题与答案 |
| [review.pdf](review.pdf) | 19 页可打印的综述 PDF |
| [overview.pptx](overview.pptx) | 54 页可编辑 PPT，演讲者备注含完整旁白与原文链接 |
| [overview.pdf](overview.pdf) | 54 页幻灯片阅读版，使用视频中的课件画面 |
| [overview.mp4](overview.mp4) | 48 分 53 秒普通话配音，1920×1080、30 fps，带烧录字幕和逐页章节 |
| [overview.srt](overview.srt) | 622 条外置中文字幕 |
| [narration.md](narration.md) | 逐页中文讲稿与开始时间 |
| [chapters.json](chapters.json) | 54 个章节的时间戳 |
| [causal_attention_demo.py](causal_attention_demo.py) | NumPy 演示，检验未来屏蔽与三种输入方式的缓存等价性 |
| [source_notes.md](source_notes.md) | 原文版本、哈希、章节与表号定位、统计口径 |
| [fact_check.md](fact_check.md) | GPT-2 / GPT-3 幻灯片逐项事实核对 |

课件与视频的手算数据、中文提示及示意图均为教学构造。论文实验数值保留对应指标、数据划分和提示条件。配音使用普通话合成音色 `zh-CN-YunxiNeural`。句级字幕依据语音服务时间戳，长句内分段按字长分配时间。

## 视频章节

| 开始时间 | PPT 页码 | 内容 |
| --- | --- | --- |
| 00:00 | 1–22 | 注意力与原始 Transformer |
| 19:20 | 23–32 | GPT-2 的训练、结构与多任务迁移实验 |
| 28:47 | 33–39 | KV Cache 专讲 |
| 34:36 | 40–52 | GPT-3 的规模实验与上下文学习 |
| 47:01 | 53–54 | 三篇论文的联系与自测 |

### 因果自注意力

第 4–6 页先讲 Q/K/V 和 softmax，第 7–14 页对应综述 §2.7。

| 视频时间 | PPT 页码 | 内容 |
| --- | --- | --- |
| 05:11 | 7 | 因果可见性与当前位置 |
| 06:03 | 8 | 查询行、键列和含对角线的下三角掩码 |
| 06:56 | 9 | softmax 前屏蔽分数，再按行归一化 |
| 07:52 | 10 | 输入和标签错开一位，当前位置预测下一个 token |
| 08:44 | 11 | Teacher forcing 如何并行计算训练位置 |
| 09:37 | 12 | 各层的因果可见性 |
| 10:26 | 13 | 从最后一个位置生成新 token |
| 11:22 | 14 | 追加 token 后复用历史 K/V 的条件 |

![下三角掩码示意](causal_attention_preview.png)

### KV Cache

第 33–39 页对应综述 §4.8。算例沿用第 6 页的键和值。

| 视频时间 | PPT 页码 | 内容 |
| --- | --- | --- |
| 28:47 | 33 | 预填充和解码分别送入哪些 token |
| 29:34 | 34 | 用当前 Q 和缓存 K/V 手算输出 |
| 30:26 | 35 | 一次送入多项时按绝对位置构造掩码 |
| 31:19 | 36 | 重复计算、注意力工作量与实际延迟 |
| 32:12 | 37 | GPT-2 Small 的缓存形状和内存估算 |
| 32:59 | 38 | 对照本仓库的推理代码和数值演示 |
| 33:49 | 39 | 超过 1,024 个位置后截取窗口和重算 |

## 重新生成

`slides.json` 提供课件文字、数据、图表类型和旁白。`build_deck.py` 定义课件布局。`review.md` 是综述的内容输入。中间 PNG、音频缓存、依赖与日志放在已忽略的 `build/`。构建所用媒体依赖单独安装。

以下从本目录运行，需要 Python 3.14+、`uv`、Node.js/npm。当前构建使用 WSL 中的微软雅黑字体：`/mnt/c/Windows/Fonts/msyh.ttc` 与 `msyhbd.ttc`。其他系统需在 `build_deck.py` 配置可用中文字体，并给 `video.py --font` 传入中文字体文件。

```bash
uv venv --python 3.14 build/.venv
uv pip install --python build/.venv/bin/python -r requirements-media.txt

# 生成 PPT、视频画面、幻灯片 PDF 和布局检查结果
build/.venv/bin/python build_deck.py

# 中文配音需要联网；按正文、声音与语速缓存，未改动页面复用缓存
build/.venv/bin/python video.py --audio-only
# 有缓存后，这一步离线合成视频，同时刷新字幕、讲稿与章节
build/.venv/bin/python video.py --video-only --threads 16 --preset superfast

# 综述离线 HTML：KaTeX 只在构建时需要，成品没有外部渲染依赖
npm install --prefix build/review_deps --no-save --no-package-lock katex@0.16.22
build/.venv/bin/python build_review.py
# 安装了 Chrome 的当前 WSL 环境可直接导出 PDF
build/.venv/bin/python build_review.py --pdf
# 也可以在浏览器打开 review.html，打印为 PDF

# 验证手算、因果掩码和整段与增量输入的等价性
build/.venv/bin/python causal_attention_demo.py
```

`build_deck.py` 在同一套布局对象上生成 PPT 原生形状和 PNG，因此没有 LibreOffice 依赖。`overview.pdf` 是这些 PNG 的阅读版；可编辑对象位于 PPTX。`video.py` 使用 `imageio-ffmpeg` 自带的 FFmpeg；不需要系统 FFmpeg。修改旁白后应重跑音频阶段，再生成视频，以保持讲稿、备注和字幕一致。不同语音服务版本可能改变重生成的音色或时长。

演示材料中的 `*.pptx`、`*.mp4`、`*.pdf`、`*.png`，以及本目录内嵌数学字体的生成版 `review.html`，已被仓库根目录的 `.gitattributes` 纳入 Git LFS。克隆后运行 `git lfs install` 与 `git lfs pull` 获取完整文件；Markdown、JSON 与生成脚本仍由普通 Git 管理。原文下载副本在 `sources/`，属于忽略的阅读缓存，永久来源链接与哈希保留在 `source_notes.md`。

## 原论文

1. Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)。
2. Radford et al. (2019), [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。
3. Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)。

KV Cache 专讲还对照了 OpenAI 发布的 GPT-2 推理代码。[官方 model.py](https://github.com/openai/gpt-2/blob/master/src/model.py)

容易混淆的口径已单独说明：GPT-2 的论文参数量与发布命名；LAMBADA 的停用词过滤；GPT-2 部分任务已有提示示例；GPT-3 的 499B 数据池与 300B 训练用量；不同指标与数据划分；原始 Transformer 英法 BLEU 在当前 PDF 不同位置的数值差异。

## 已执行的验证

- PPTX 共 54 页。生成脚本检查了全部页面的文本布局。演讲者备注可与 `slides.json` 逐页核对。
- 综述 HTML 已排版 127 处公式。PDF 共 19 页。
- 数值演示通过所有断言。修改未来输入后，两层因果结构的前两个位置输出保持一致。整段前向与逐项缓存输出最大误差约为 1.33×10⁻¹⁵，分块缓存的最大误差为 0。
- 成品视频通过完整音视频解码。已逐页对照课件画面和烧录字幕，并核对 54 个章节与 622 条外置字幕的时间。
