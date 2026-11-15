# 《GPT-2 推理原理与实现：为什么 KV Cache 能加速》

讲解 `src/learn_gpt2/gpt2.py` 的推理过程，重点说明 KV Cache 为什么能加速。

- `gpt2_kvcache.pptx`：16 页幻灯片，演讲者备注即旁白稿
- `gpt2_kvcache.mp4`：12:38 讲解视频（3840×2160、60fps，中文配音，烧录字幕）
- `gpt2_kvcache.srt`：视频字幕
- `narration.md`：旁白稿

## 幻灯片

1. 封面
2. 推理全景图
3. GPT2Config 四种规模
4. 输入层
5. Transformer Block
6. 注意力（上）
7. 注意力（下）三种掩码
8. MLP 与输出头
9. 采样 generate()
10. 无缓存 O(N²)
11. KV Cache 核心观察
12. 预填充与逐步解码
13. LayerKVCache 代码
14. 加速与显存代价
15. 边界 block_size=1024
16. 总结

## 视频

- 分辨率：3840×2160
- 帧率：60fps
- 时长：12:38
- 配音：edge-tts `zh-CN-YunxiNeural`
- 字幕：烧录中文字幕，另附 `gpt2_kvcache.srt`
