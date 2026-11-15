# GPT-2 推理：原理与实现

讲解 `src/learn_gpt2/gpt2.py` 的推理过程，重点说明 KV Cache 为什么能加速。

- `gpt2_inference.pptx`：27 页幻灯片，演讲者备注即旁白稿
- `gpt2_inference.mp4`：约 16 分钟的讲解视频（中文配音，烧录字幕，含两段动画）
- `gpt2_inference.srt`：视频字幕

## 重新生成

依赖：本项目的 CUDA 环境（测量数据）、一个装有 `matplotlib pygments python-pptx sherpa-onnx soundfile` 的 Python 环境、`pptxgenjs`、Docker（镜像内含 LibreOffice 与 ffmpeg）、Kokoro 语音模型 [`kokoro-multi-lang-v1_1`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models)。

```bash
docker build -t learn-gpt2-media presentations/01-gpt2-inference/tools
uv run --extra cuda python presentations/01-gpt2-inference/benchmark.py      # data/benchmark.json
uv run --extra cuda python presentations/01-gpt2-inference/attention_map.py  # data/attention.json
cd presentations/01-gpt2-inference
python figures.py --frames                                 # build/figures, build/frames
python prepare.py                                          # build/deck_data.json
NODE_PATH=<pptxgenjs 所在 node_modules> node deck.js <pptx 技能的 apply_theme.js>
./render.sh                                                # build/render/slide-NN.png
python video.py --model <kokoro-multi-lang-v1_1 目录>       # mp4 + srt
```

讲稿在 `narration.json`，修改后重新运行 `prepare.py`、`deck.js` 与 `video.py` 即可；语音按句缓存在 `build/audio`，只会重新合成改动过的句子。
