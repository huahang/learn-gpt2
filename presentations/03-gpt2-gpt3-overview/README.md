# 注意力机制、GPT-2 与 GPT-3 三篇论文综述

从向量、加权平均与下一 token 预测出发，串起三篇论文的研究问题、方法、训练数据、实验与局限。重点展开因果自注意力，适合具备基本数学和 Python 知识的学习者。

## 成品

| 文件 | 内容 |
| --- | --- |
| [review.html](review.html) | 推荐阅读版：约一万字中文综述，离线目录、表格及 82 处排版公式 |
| [review.md](review.md) | 综述源文，含详细引用定位、自测题与答案 |
| [review.pdf](review.pdf) | 可打印的综述 PDF |
| [overview.pptx](overview.pptx) | 46 页可编辑 PPT，文字与图形为原生对象，演讲者备注含完整旁白与原文链接 |
| [overview.pdf](overview.pdf) | 与视频画面一致的 46 页幻灯片阅读版 |
| [overview.mp4](overview.mp4) | 42 分 33 秒中文讲解，1920×1080、30fps，烧录字幕、内嵌逐页章节 |
| [overview.srt](overview.srt) | 538 条外置中文字幕 |
| [narration.md](narration.md) | 逐页中文讲稿与开始时间 |
| [chapters.json](chapters.json) | 46 页章节时间，便于播放器定位或二次剪辑 |
| [causal_attention_demo.py](causal_attention_demo.py) | 不下载权重的 NumPy 因果注意力、未来屏蔽与 KV Cache 数值演示 |
| [source_notes.md](source_notes.md) | 原文版本、哈希、章节与表号定位、统计口径 |
| [fact_check.md](fact_check.md) | GPT-2 / GPT-3 幻灯片逐项事实核对 |

课件与视频的手算数据、中文提示及示意图均为教学构造；论文实验数值明确保留对应指标、数据划分和提示条件。视频使用合成中文旁白 `zh-CN-YunxiNeural`，画面为逐页课件；句级字幕依据语音服务时间戳，长句内分段按字长分配时间。

## 因果自注意力重点观看位置

综述详见 **§2.7**；课件第 **7–14 页**集中讲解，前面的第 4–6 页先建立 Q/K/V 和 softmax 基础。

| 视频时间 | PPT 页码 | 内容 |
| --- | --- | --- |
| 05:14 | 7 | 因果可见性，为什么允许读取当前位置 |
| 06:06 | 8 | 行是查询、列是键；包含对角线的下三角掩码 |
| 06:59 | 9 | softmax 前加负无穷与重新归一化 |
| 07:55 | 10 | 输入和标签错开一位，当前位置预测下一个 token |
| 08:47 | 11 | Teacher forcing，训练并行与信息依赖的区别 |
| 09:40 | 12 | 逐层归纳：堆叠后也不能绕路读取未来 |
| 10:29 | 13 | 生成时未来未知，必须逐步追加 token |
| 11:24 | 14 | 因果性为何允许缓存历史 K/V，缓存的代价与边界 |

其余章节：00:00 注意力与 Transformer；19:30 GPT-2；28:57 GPT-3；40:30 三篇论文的联系与自测。精确逐页边界以 `chapters.json` 为准。

![下三角掩码示意](causal_attention_preview.png)

## 重新生成

`slides.json` 是课件文字、数据、图表类型及旁白的唯一内容输入；`review.md` 是综述的内容输入。所有中间 PNG、音频缓存、依赖与日志放在已忽略的 `build/`。不需要下载 GPT 权重，也不需要修改项目的训练依赖。

以下从本目录运行，需要 Python 3.14+、`uv`、Node.js/npm。当前构建使用 WSL 中的微软雅黑字体：`/mnt/c/Windows/Fonts/msyh.ttc` 与 `msyhbd.ttc`。其他系统需在 `build_deck.py` 配置可用中文字体，并给 `video.py --font` 传入中文字体文件。

```bash
uv venv --python 3.14 build/.venv
uv pip install --python build/.venv/bin/python -r requirements-media.txt

# 46 页 PPT、同源 PNG、幻灯片 PDF、联系表和布局检查结果
build/.venv/bin/python build_deck.py

# 中文配音需要联网；按正文、声音与语速缓存，未改动页面复用缓存
build/.venv/bin/python video.py --audio-only
# 有缓存后，这一步离线合成视频，同时刷新字幕、讲稿与章节
build/.venv/bin/python video.py --video-only --threads 12

# 综述离线 HTML：KaTeX 只在构建时需要，成品没有外部渲染依赖
npm install --prefix build/review_deps --no-save --no-package-lock katex@0.16.22
build/.venv/bin/python build_review.py
# 安装了 Chrome 的当前 WSL 环境可直接导出 PDF
build/.venv/bin/python build_review.py --pdf
# 也可以在浏览器打开 review.html，打印为 PDF

# 单头两层教学网络，显示矩阵并验证因果性和缓存等价性
build/.venv/bin/python causal_attention_demo.py
```

`build_deck.py` 在同一套布局对象上生成 PPT 原生形状和 PNG，因此没有 LibreOffice 依赖。`overview.pdf` 是这些 PNG 的阅读版；可编辑对象位于 PPTX。`video.py` 使用 `imageio-ffmpeg` 自带的 FFmpeg；不需要系统 FFmpeg。修改旁白后应重跑音频阶段，再生成视频，以保持讲稿、备注和字幕一致。不同语音服务版本可能改变重生成的音色或时长。

演示材料中的 `*.pptx`、`*.mp4`、`*.pdf`、`*.png`，以及本目录内嵌数学字体的生成版 `review.html`，已被仓库根目录的 `.gitattributes` 纳入 Git LFS。克隆后运行 `git lfs install` 与 `git lfs pull` 获取完整文件；Markdown、JSON 与生成脚本仍由普通 Git 管理。原文下载副本在 `sources/`，属于忽略的阅读缓存，永久来源链接与哈希保留在 `source_notes.md`。

## 原论文

1. Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)。
2. Radford et al. (2019), [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。
3. Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)。

容易混淆的口径已单独说明：GPT-2 的论文参数量与发布命名；LAMBADA 的停用词过滤；GPT-2 部分任务已有提示示例；GPT-3 的 499B 数据池与 300B 训练用量；不同指标与数据划分；原始 Transformer 英法 BLEU 在当前 PDF 不同位置的数值差异。

## 已执行的验证

- 46 页 PPTX 可重新读取，压缩包完整；每页备注与 `slides.json` 旁白逐字一致，来源链接齐全。
- 46 张 1920×1080 画面已生成，布局检查无文本越界；检查了全部联系表和重点页面。
- 综述 HTML 的 82 处公式均完成 KaTeX 排版；17 页 PDF 已检查边界，并抽查手算与因果掩码页面。
- `causal_attention_demo.py`：修改未来输入后，单层与两层因果结构的前两位置差异均为 0；无掩码对照变化约 0.7975；完整前向与增量 KV Cache 最大误差约 1.33×10⁻¹⁵。
- 仓库级 `uv run --no-sync ruff check`、`uv run --no-sync ruff format --check` 通过；本任务没有改动模型实现。
- 视频完整音视频解码通过，无解码错误；76,595 帧，视频 2553.166667 秒、音轨 2553.168 秒；46 个内嵌章节、538 条字幕均已校验。抽查第 1、8、26、37、42、46 页，字幕未遮挡课件内容。
