# 《展开讲讲 Causal Self Attention：GPT-2 因果自注意力》

讲解 `src/learn_gpt2/gpt2.py` 的 `CausalSelfAttention`（L92-122）。

- `gpt2_causal_self_attention.pptx`：17 页幻灯片，演讲者备注即旁白稿
- `gpt2_causal_self_attention.mp4`：12:30 讲解视频（3840×2160、30fps，中文配音，烧录字幕）
- `gpt2_causal_self_attention.srt`：视频字幕
- `narration.md`：旁白稿

## 幻灯片

1. 封面
2. 路线图
3. 注意力在哪里（Pre-LN Block）
4. Q、K、V：软查表
5. forward 全貌
6. 第 1 步：c_attn 投影出 q、k、v
7. 第 2 步：切分多头（768 = 12 × 64）
8. 第 3 步：打分与缩放（Q·Kᵀ/√hs）
9. 第 4 步：因果掩码
10. 第 5 步：softmax 与加权求和
11. 第 6 步：合并多头与 c_proj
12. 一次调用：scaled_dot_product_attention
13. 第二部分：注意力遇上 KV Cache
14. 带缓存：只算新 token
15. 三种掩码分支
16. 计算与参数
17. 总结：形状一图流

## 视频

- 分辨率：3840×2160
- 帧率：30fps
- 时长：12:30
- 配音：edge-tts `zh-CN-XiaoxiaoNeural`（语速 -8%）
- 字幕：烧录中文字幕，另附 `gpt2_causal_self_attention.srt`
- 内容对照 `gpt2.py` 提交 `f585a1d` 核对
