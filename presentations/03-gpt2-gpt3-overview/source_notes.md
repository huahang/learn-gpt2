# 来源定位与数据口径

本目录的综述和教学材料依据以下三篇原文。核查日期为 2026-11-15；“2017／2019／2020”是论文工作的年份，不是下载日期。PDF 页码从第一页计数，以下页码对应本次下载版本。所有手算注意力、中文任务示例和显存估算均为教学构造，不能当作实测模型输出。

## 原文记录

| 缩写 | 论文与作者 | 获取地址 | 本次版本 |
|---|---|---|---|
| T | Vaswani et al., Attention Is All You Need | https://arxiv.org/pdf/1706.03762 | arXiv v7，2023-08-02，15 页；原工作发表于 2017 |
| G2 | Radford et al., Language Models are Unsupervised Multitask Learners | https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | 2019 技术报告，24 页 |
| G3 | Brown et al., Language Models are Few-Shot Learners | https://arxiv.org/pdf/2005.14165 | arXiv v4，2020-07-22，75 页 |

本地阅读副本位于 `sources/attention.pdf`、`sources/gpt2.pdf`、`sources/gpt3.pdf`；对应 `.txt` 为 pypdf 提取文本，带有逐页标记。PDF 与提取文本作为下载缓存，不必提交到 Git。原文图表的数值与结构用于核查；成品教学图应自绘并注明来源。

SHA-256：

```text
attention.pdf  bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697
gpt2.pdf       d9d852e2894556e73f53cb22b7c605a9643d6f0b19bf604b429ed6192fa24f4e
gpt3.pdf       97fd272f1fdfc18677462d0292f5fbf26ca86b4d1b485c2dba03269b643a0e83
```

## 方法与数字的原文位置

| 核查事项 | 位置 | 口径或注意事项 |
|---|---|---|
| Transformer 总体结构、6 层 encoder 与 6 层 decoder、Post-LN | T §3.1，图 1，PDF pp. 2–3 | 原论文不是 GPT-2 的 Pre-LN |
| Scaled dot-product attention 与缩放原因 | T §3.2.1，式 1，脚注 4，p. 4 | softmax 按 key 维度；缩放因子为根号 key 维度 |
| Multi-head 与 base 的 8 头／每头 64 维 | T §3.2.2，pp. 4–5 | 每个头使用独立投影 |
| 三种 attention 与未来屏蔽 | T §3.2.3，p. 5 | future logits 在 softmax 之前设为负无穷 |
| FFN、正弦位置编码 | T §3.3–3.5，pp. 5–6 | ReLU；主模型正弦，可学习版本消融接近 |
| 复杂度与路径长度 | T 表 1，§4，pp. 6–7 | 表中注意力项不应替代完整 block 成本 |
| 训练数据、GPU、warmup、正则化 | T §5，pp. 7–8 | 8 张 P100；base 12 小时，big 3.5 天 |
| 翻译与消融结果 | T 表 2–3，pp. 8–9 | 英德 big 28.4；英法以摘要和表 2 的 41.8 为准 |
| 句法分析 | T §6.3，表 4，pp. 9–10 | 针对句法任务训练，不是翻译模型直接 zero-shot |
| GPT-2 自回归目标与隐式任务假设 | G2 §2，pp. 2–3 | 自监督下一 token 预测；目标任务不微调 |
| WebText | G2 §2.1，pp. 3–4 | Reddit 外链至少 3 karma，约 800 万文档，40 GB；截止 2017-12；移除 Wikipedia |
| byte-level BPE | G2 §2.2，p. 4 | 256 字节基础单位；最终词表 50,257 |
| GPT-2 规模、Pre-LN、初始化、上下文 | G2 §2.3，表 2，p. 4 | 117M/345M/762M/1542M；1024 token；batch 512 |
| GPT-2 语言建模结果 | G2 表 3，p. 5 | PPL/BPB/BPC/accuracy 不可当同一指标 |
| LAMBADA 过滤 | G2 §3.3，pp. 5–6 | 52.66% 为未加 stop-word 过滤；63.24% 含过滤 |
| CBT 重合及验证集口径 | G2 §3.2，p. 5 | 因一本测试书出现在 WebText，报告验证集 |
| Winograd、CoQA | G2 §3.4–3.5，p. 6 | Winograd 70.70%，273 例；CoQA 开发集 F1 55 |
| GPT-2 摘要 | G2 §3.6，表 4，p. 6 | R-1/R-2/R-L 为 29.34/8.27/26.58，TL;DR 提示 |
| GPT-2 翻译与 QA | G2 §3.7–3.8，pp. 6–7 | 输入中带示范；英法 5 BLEU，法英 11.5；NQ EM 4.1% |
| GPT-2 污染分析 | G2 §4，表 6，p. 8 | 8-gram；无重合 LAMBADA 62.9%，不能证明绝无记忆 |
| zero/one/few-shot 定义 | G3 §2，图 2.1，pp. 6–7 | 所有这些设置测试时无梯度更新 |
| GPT-3 模型尺寸与训练 token | G3 表 2.1，§2.1，p. 8 | 8 个规模；175B=96 层/12288 宽/96 头/128 head dim；300B 训练 token |
| 稠密与局部带状稀疏交替 | G3 §2.1，p. 8 | 沿用 GPT-2 架构的重要例外 |
| GPT-3 语料量、混合比例 | G3 表 2.2，§2.2，pp. 8–9 | 可用约 499B，实际训练 300B；整数权重合计 101% 是近似取整 |
| GPT-3 英语比例 | G3 §3.3，p. 14 | 按词数约 93%，不是按 token 统计 |
| GPT-3 训练工程 | G3 §2.3，p. 9；附录 B，p. 43 | V100；矩阵内与层间并行；Adam、warmup、cosine decay |
| GPT-3 评估协议 | G3 §2.4，p. 10；附录 G | 根据可用窗口与验证集选择 K；任务有不同评分和解码 |
| LAMBADA、StoryCloze、HellaSwag | G3 表 3.2，p. 12 | LAMBADA 76.2/72.5/86.4；zero 与 few-shot 提示格式不同 |
| GPT-3 闭卷问答 | G3 表 3.3，p. 13 | TriviaQA FS 71.2；NaturalQS FS 29.9 |
| GPT-3 阅读理解 | G3 表 3.7，p. 18 | CoQA 81.5/84/85；DROP 与 RACE 有明显差距 |
| SuperGLUE | G3 表 3.8，p. 19 | 测试集；FS 71.8；BERT-Large 69.0；SOTA 89.0；WiC 49.4 |
| SuperGLUE 规模与示例数曲线 | G3 图 3.8，p. 20 | 开发集，不可与表 3.8 测试集混用 |
| 污染与过滤 bug | G3 §4，pp. 29–33；附录 C，pp. 43–45 | PIQA、Winograd 需特别标注；背景文本重合不等于标签泄漏 |
| 局限与 broader impacts | G3 §5–6，pp. 33–39 | ICL 机制未完全解释；偏见、流畅性与事实性、成本 |
| 训练总 FLOPs | G3 附录 D，表 D.1，p. 46 | 175B 约 3.14×10^23 FLOPs；估算忽略部分 attention 成本 |

## 特别容易写错的地方

### GPT-2 的 zero-shot 与 GPT-3 的定义不完全相同

GPT-2 的论点是无目标任务参数／架构更新的迁移，但其翻译、Natural Questions 输入中带示例。GPT-3 把 zero-shot 限定为无示范，one/few-shot 允许上下文示范。成品应保留这一区别，不要用后来的定义重写全部早期实验。

### 参数量命名

GPT-2 论文表 2 列出 117M、345M、762M、1542M；发布模型生态常见 124M、355M、774M、1558M 等命名。综述中的规模表采用论文口径。GPT-3 表 2.1 和附录 D 的部分中间模型亦有近似数差别，绘图不应把这些轻微差异解释为额外实验。最大模型 175B 的含义始终为约 1750 亿参数。

### Transformer 英法 BLEU 的原文不一致

本次下载的 T v7 摘要与表 2 写 41.8，§6.1 的正文仍写 41.0。成品采用摘要与表 2 的 41.8，并注明来源。不要声称三个位置完全一致；也不要将两者当作自行测得的不同结果。

### GPT-3 主表与污染附表的差异

附录 C 表 C.1 的注释说明其结果使用不同的上下文随机样本种子，因此会与正文数值略有不同。比如 CoQA、RACE 等存在差别。主结果图应采用正文主表；清洁子集比较才使用表 C.1。表 3.8 是 SuperGLUE 测试集，图 3.8 是开发集，不能混合画同一条曲线。

### 自回归掩码的对角线与标签

第 i 个位置包含当前 token x_i，预测 x_(i+1)，所以允许矩阵包含对角线。训练输入已知但每一层都屏蔽未来，才能并行而不泄漏。softmax 行沿 key 归一化；padding mask 与 causal mask 不同；全遮罩行需特别处理。

### 教学算例核对

单个 q=(1,0)，K=[(1,0),(0,1),(1,1)]，dk=2，权重约为 (0.401112,0.197776,0.401112)。V=[(1,0),(0,2),(2,1)] 时输出约为 (1.203336,0.796664)。屏蔽第三列后权重为 (0.669762,0.330238,0)，输出为 (0.669762,0.660477)。第二个掩码示例 scores=(2,1,0,3) 只允许前两列时，权重约 (0.731059,0.268941,0,0)。所有数字均为手工构造并计算，不代表真实头的实测权重。

PPL 教学例中正确 token 概率 0.8、0.5、0.25，乘积 0.1，几何平均概率约 0.464159，平均 NLL 约 0.767528，PPL 约 2.154435。FP16 权重 350 GB 是 175×10^9×2 字节的简化估算，未包括任何运行额外开销。
