# GPT-2 与 GPT-3 幻灯片事实核对

页码对应 `overview.pptx`。表中实验数字沿用原文的定义。手算、教学提示和缓存工作量均采用构造示例。

论文简称：

- GPT-2：Radford 等，2019，[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。
- GPT-3：Brown 等，2020，[Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)，arXiv v4，75 页版本。
- Transformer：Vaswani 等，2017，[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)。

| PPT 页码 | 核对内容 | 原文定位与边界 |
|---|---|---|
| 23 | GPT-2 通过语言建模研究无任务微调的迁移 | GPT-2 §2，式 (1)；§3。无微调不等于每个实验没有提示示例。 |
| 24 | WebText 外链门槛 3 karma，候选约 4500 万链接，800 多万文档，40 GB | GPT-2 §2.1，PDF 第 3 页。时间截至 2017 年 12 月，排除 Wikipedia。karma 是粗略筛选信号，不保证准确性。 |
| 25 | 字节级 BPE，词表 50,257，窗口 1,024 | GPT-2 §2.2–2.3，PDF 第 4 页。字节覆盖任意字符串不等于充分训练所有语言。 |
| 26–27 | 因果单栈、pre-LN、最终 LN、初始化缩放 | GPT-2 §2.3；原始 Transformer §3.1。可学习位置向量及 block 细节也可对照 [OpenAI 官方 model.py](https://github.com/openai/gpt-2/blob/master/src/model.py) 的 wpe 与 block 定义。缩放发生在初始化；论文 N 指残差层。 |
| 28 | 117M/345M/762M/1542M；层数 12/24/36/48；宽度 768/1024/1280/1600 | GPT-2 Table 2，PDF 第 4 页。发布模型命名不同，亦可对照本仓库 `src/learn_gpt2/gpt2.py` 配置注释。 |
| 29 | batch 512、5% WebText 留出集调参、尚未充分拟合 | GPT-2 §2.3 与 §3，PDF 第 4 页。移位标签例子为教学示意，不是真实分词输出。 |
| 30 | CoQA 文档与问答历史；摘要 TL;DR；翻译与 NQ 含示例对 | GPT-2 §3.5–3.8，PDF 第 6–7 页。避免把 GPT-2 全部实验概括为严格零提示样本。 |
| 31 | WikiText-103 PPL 17.48；LAMBADA 63.24%；CoQA dev F1 55 | GPT-2 Table 3，§3.3、§3.5。LAMBADA 原始准确率 52.66%，63.24% 使用停用词过滤。论文称 8 个语言建模数据集中的 7 个为当时最佳。 |
| 32 | 摘要 R-AVG 21.40 vs Random-3 20.98；翻译 5/11.5 BLEU；NQ EM 4.1% | GPT-2 Table 4，§3.6–3.8。摘要是 CNN/Daily Mail、k=2、生成 100 token 取前三句的设置。 |
| 33–35 | 预填充、单项解码手算、分块解码掩码 | 原始 Transformer §3.2.3 给出可见性。发布实现用 `past` 接收旧 K/V，用 `present` 返回当轮 K/V。已缓存 C 项时，本轮查询第 i 行可读键列 j ≤ C + i。[OpenAI 官方 model.py](https://github.com/openai/gpt-2/blob/master/src/model.py) |
| 36–39 | 稠密注意力的复杂度、36 MiB 估算、窗口截取与重算 | 复杂度以整段重跑和单项缓存为对照。GPT-2 Small 共有 12 层、12 头，每头 64 维。估算采用 B=1、容量 1024、FP16，计两份 K/V 张量。第 39 页对应本仓库的 `generate` 实现。[仓库代码](../../src/learn_gpt2/gpt2.py) |
| 40 | 8 种规模，从 125M 到 175B | GPT-3 Table 2.1，PDF 第 8 页。论文中更多极小模型仅用于 Figure 3.1 的训练趋势补充分析，不改变主要 8 模型系列。 |
| 41 | 175B：96 层、12288 宽、96 头、每头 128、FFN 49152、窗口 2048 | GPT-3 Table 2.1 与 §2.1。FFN 为 4×d_model 的计算结果。密集与局部带状稀疏注意力交替，不是 MoE。 |
| 42 | 数据池 410+19+12+55+3=499B；权重 60/22/8/8/3% | GPT-3 Table 2.2，PDF 第 9 页。499B 是表内数据相加的近似值；权重四舍五入和约 101%；实际训练 300B token。 |
| 43 | 所有模型训练 300B token；最大 batch 3.2M token，学习率 6e-5 | GPT-3 Table 2.1；§2.3、Appendix B。相同 token 数不等于相同 FLOPs。 |
| 44–45 | zero/one/few 与微调区别；通常 K=10–100 | GPT-3 §2，Figure 2.1。不同任务例数不同，受窗口约束；例如 CoQA few-shot K=5。中文情感示例为原创教学，未运行模型。 |
| 46 | LAMBADA 76.2/72.5/86.4% | GPT-3 Table 3.2，§3.1.2，Appendix H。175B，test；zero-shot 与 one/few-shot 格式不同；few-shot K=15；污染影响另见 §4。 |
| 47 | TriviaQA 64.3/68.0/71.2%；CoQA 81.5/84/85 F1 | GPT-3 Table 3.3、Table 3.7、Appendix H。TriviaQA 表中 few-shot 为 wiki split test server，附录中 dev 亦是 71.2；CoQA 为 dev。不可混同闭卷问答与带文档问答，也不可直接比较 accuracy 与 F1。 |
| 48 | WMT14 英译法 25.2/28.3/32.6，法译英 21.2/33.7/39.2 | GPT-3 §3.3，Table 3.4。175B，零样本、单样本、少样本。少样本 K=64，采用 XLM 分词下的 multi-bleu.perl。 |
| 49 | SuperGLUE 69.0/71.8/89.0；WiC 49.4% | GPT-3 Table 3.8，PDF 第 19 页。分别是微调 BERT-Large/GPT-3 few-shot/当时微调 SOTA，都是 test；GPT-3 为每任务 K=32。正文 Figure 3.8 是 dev，不用于此柱图。 |
| 50 | 加法 100/80.4/25.5/9.3% | GPT-3 Table 3.9，Appendix H。175B few-shot，K=50。2D 是输入整数采样 [0,100)，含不足两位数；3D/4D/5D 相应扩展区间，名称沿用论文。 |
| 51 | 过滤 bug、未重训、clean/dirty 子集比较 | GPT-3 §4，Appendix C。原始过滤用 13-gram；后续分析依任务长度使用 8–13 gram。未检测重叠不保证完全独立。 |
| 52 | 重复矛盾、比较任务弱、偏见与成本 | GPT-3 §5–6。基础模型与后续指令微调/RLHF 系统的区别是对论文方法范围的说明。 |
| 53–54 | 三文研究问题与学习检查 | 教学归纳；不是三论文原文声明的统一因果结论。 |
