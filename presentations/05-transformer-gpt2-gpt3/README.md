# 从注意力到大语言模型：Transformer、GPT-2 与 GPT-3

给想入门 LLM 的读者。三篇论文各自做了什么、为什么重要、关键方法和实验结论，重点展开因果自注意力和 KV Cache。视觉沿用 [PR #13](https://github.com/huahang/learn-gpt2/pull/13) 的深色主视觉：背景 `#090715`，强调色 `#916EF0`，中文 Noto Sans CJK SC，拉丁文 Inter，代码 JetBrains Mono，页面 10×5.625 英寸。

## 成品

| 文件 | 内容 |
| --- | --- |
| [review.md](review.md) | 中文综述 |
| [transformer_gpt2_gpt3.pptx](transformer_gpt2_gpt3.pptx) | 37 页幻灯片，演讲者备注即旁白（Git LFS） |
| [transformer_gpt2_gpt3.mp4](transformer_gpt2_gpt3.mp4) | 24 分 13 秒中文配音，1920×1080、30fps，烧录字幕（Git LFS） |
| [transformer_gpt2_gpt3.srt](transformer_gpt2_gpt3.srt) | 外置字幕 |
| [narration.md](narration.md) | 逐页讲解稿 |
| [chapters.json](chapters.json) | 逐页起始时间 |
| [demo.py](demo.py) | 不下载权重的因果掩码与 KV Cache 数值演示 |

`*.pptx` 与 `*.mp4` 由仓库根目录 `.gitattributes` 交给 Git LFS。克隆后执行 `git lfs install` 与 `git lfs pull`。

## 幻灯片

1. 封面
2. 路线图
3. 先预测下一个词
4. 输入和标签错开一位
5. 为什么要换成注意力
6. Transformer
7. 缩放点积注意力
8. 多头
9. 三种注意力
10. 位置、前馈网络和训练配方
11. 翻译结果
12. 因果自注意力
13. 为什么不能看未来
14. 掩码是包含对角线的下三角
15. 负无穷变成 0
16. 并行的是计算，不是偷看
17. 对照 gpt2.py 的形状
18. 三种掩码分支
19. KV Cache
20. 没有缓存就要重算过去
21. 预填充，再解码一步
22. 缓存的形状和显存
23. 复杂度对比
24. 没有因果掩码，缓存就是错的
25. GPT-2
26. 零样本的口径
27. WebText
28. Pre-LN 与初始化
29. 四个尺寸与语言建模
30. LAMBADA、CBT、Winograd
31. GPT-3
32. zero / one / few-shot
33. 1750 亿参数与数据配比
34. 关键实验
35. 局限
36. 三篇论文如何相连
37. 四句话

因果自注意力集中在第 12–18 页，KV Cache 在第 19–24 页。

## 重新生成

课件和视频不下载模型权重。构建机需要 Python 3.12+、ffmpeg，以及下面三种字体：

- Noto Sans CJK SC：`fonts-noto-cjk` 里的 `NotoSansCJK-Regular.ttc` / `Bold.ttc`，面索引 2
- Inter：`Inter-Regular.otf` 与 `Inter-SemiBold.otf`
- JetBrains Mono：`fonts-jetbrains-mono`

字体不在默认路径时，用环境变量 `NOTO_REGULAR`、`NOTO_BOLD`、`INTER_REGULAR`、`INTER_SEMIBOLD`、`MONO_REGULAR`、`MONO_BOLD` 指向文件。

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-media.txt
.venv/bin/python demo.py
.venv/bin/python build_deck.py
.venv/bin/python video.py
```

`build_deck.py` 同时写出 PPTX 和 1920×1080 PNG，布局溢出会直接失败。`video.py` 用 edge-tts 合成旁白，需要联网；音频按正文、音色和语速缓存。配音是 `zh-CN-XiaoxiaoNeural`，语速 `-8%`。

## 原论文

1. Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)。
2. Radford et al. (2019), [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。
3. Brown et al. (2020), [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)。

英法 BLEU 在原 PDF 的摘要、表 2 与正文 6.1 节不一致，综述和课件都同时写出 41.8 与 41.0。GPT-2 论文表 2 的参数量与本仓库 `GPT2_CONFIGS` 注释的计数口径不同，课件注明了二者。
