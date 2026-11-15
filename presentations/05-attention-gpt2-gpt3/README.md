# 《注意力、GPT-2 与 GPT-3：三篇论文精读》

精读三篇奠定大语言模型基础的论文，并重点展开**因果自注意力**与 **KV Cache**：

1. Vaswani 等，[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（2017）
2. Radford 等，[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)（GPT-2，2019）
3. Brown 等，[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)（GPT-3，2020）

## 产物

- `review.md`：综述文章（三篇论文 + 因果自注意力 + KV Cache 专题）
- `attention_gpt2_gpt3.pptx`：45 页幻灯片，演讲者备注即旁白稿
- `attention_gpt2_gpt3.mp4`：39:49 讲解视频（1920×1080、30fps，中文配音，烧录字幕）
- `attention_gpt2_gpt3.srt`：视频字幕
- `narration.json`：逐页旁白稿与动画切换点（`cues` 为句子序号，可带小数表示句内位置）

## 幻灯片

| 部分 | 页码 | 内容 |
| --- | --- | --- |
| 开场 | 1–3 | 封面、三篇论文的主线、语言模型在做什么 |
| 一、注意力机制 | 4–13 | 循环网络的瓶颈、Q/K/V、缩放点积注意力（动画）、为什么除以 √d_k、多头注意力、Transformer 结构、前馈网络与位置编码、复杂度比较（表 1）、实验结果（表 2） |
| 二、因果自注意力 | 14–19 | 训练时如何防止“偷看答案”、因果掩码（动画）、因果性逐层保持、真实 GPT-2 的注意力图、`is_causal=True` 的代码与手写等价实现 |
| 三、GPT-2 | 20–26 | 任务即文本、WebText 与字节级 BPE、模型修改（Pre-LN 等）、零样本语言建模、下游任务、记忆与去重分析 |
| 四、GPT-3 | 27–35 | 动机、微调 / 零样本 / 单样本 / 少样本、8 个模型规模（表 2.1）、训练数据、主要结果、算术与新闻生成、规模规律、局限与风险 |
| 五、KV Cache | 36–43 | 不用缓存时的 O(N²) 重算（动画）、旧位置 K/V 不变的核心洞察、预填充 + 逐步解码（动画）、`LayerKVCache` 代码与掩码偏移、实测加速、显存代价、后续改进（MQA/GQA、PagedAttention 等） |
| 结尾 | 44–45 | 总结、致谢 |

第 41 页的实测数据与第 18 页的真实注意力图来自 `../01-gpt2-inference/data/`（`benchmark.json`、`attention.json`）。

## 视频

- 分辨率：1920×1080，30fps；画面四周留边，下方烧录字幕
- 时长：39:49
- 配音：本地 TTS [Kokoro multi-lang v1.1](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html)（sherpa-onnx，说话人 3）
- 第 7、16、37、39 页为逐帧动画，按 `narration.json` 中的 `cues` 与旁白同步

## 重新生成

脚本均在本目录运行，中间产物写入 `build/`（已忽略）。

```bash
# 1. 插图与动画帧 -> build/figures/
python figures.py
# 2. 代码高亮片段 -> build/deck_data.json
python prepare.py
# 3. 幻灯片 -> attention_gpt2_gpt3.pptx（需要 pptxgenjs）
NODE_PATH=/path/to/node_modules node deck.js
# 4. 用 LibreOffice 渲染为 build/render/slide-NN.png（字体映射见 tools/fonts.conf）
SOFFICE=/path/to/soffice ./render.sh
# 5. 配音、合成动画、字幕与视频 -> attention_gpt2_gpt3.{mp4,srt}
python video.py --model /path/to/kokoro-multi-lang-v1_1 --check   # 列出 TTS 词典缺失的词
python video.py --model /path/to/kokoro-multi-lang-v1_1
```

依赖：

- Python：`matplotlib`、`numpy`、`pygments`、`pillow`、`sherpa-onnx`、`soundfile`
- Node.js：`pptxgenjs`
- LibreOffice、`pdftoppm`（poppler）、带 `libx264` 与 `libass` 的 `ffmpeg`
- 字体：Noto Sans CJK SC、DejaVu Sans Mono

逐句音频缓存在 `build/audio/`，修改旁白后只会重新合成改动过的句子。
TTS 会静默跳过词典里没有的英文词，`video.py` 中的 `SPOKEN` 表把这类术语改写成可读形式，并把四位及以上的整数转成中文读法。
