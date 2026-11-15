"""Narrate the deck with local TTS and assemble gpt2_inference.mp4 (+ .srt).

    python video.py --model <kokoro-multi-lang-v1_1 dir> [--check]

Needs build/render/slide-NN.png (render.sh), build/frames/ (figures.py --frames) and the
learn-gpt2-media Docker image for ffmpeg. --check only lists words the TTS lexicon lacks.
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
BUILD = HERE / "build"
AUDIO = BUILD / "audio"
VOICE_SID = 3  # Kokoro v1.1 speaker id; 3 and up are Mandarin voices
SPEED = 1.0
SR = 24000
LEAD, GAP, TAIL = 0.45, 0.28, 0.75  # seconds of silence: before, between and after sentences
FPS = 30

# Spoken forms for terms the TTS lexicon does not know. Order matters: longer first.
SPOKEN = [
    (r"gpt2\.py", "G P T 2 点 P Y"),
    (r"gpt2-xl", "G P T 2 X L"),
    (r"GPT2Config", "G P T 2 configuration"),
    (r"(?i)gpt-?2", "G P T 2"),
    (r"LayerKVCache", "layer K V cache"),
    (r"KVCache", "K V cache"),
    (r"KV Cache", "K V cache"),
    (r"\bKV\b", "K V"),
    (r"LayerNorm", "layer norm"),
    (r"PyTorch", "pie torch"),
    (r"tiktoken", "tick token"),
    (r"softmax", "soft max"),
    (r"argmax", "arg max"),
    (r"logits", "lodge its"),
    (r"lm_head", "L M head"),
    (r"c_attn", "C attention"),
    (r"torch\.cat", "torch cat"),
    (r"max len", "max length"),
    (r"top-k", "top K"),
    (r"\btril\b", "T R I L"),
    (r"\bidx\b", "I D X"),
    (r"\bwte\b", "W T E"),
    (r"\bwpe\b", "W P E"),
    (r"\b(MLP|GELU|GPU|GQA|MQA|RTX|GB|KB|MB)\b", lambda m: " ".join(m.group(1))),
    (r"\btokens\b", "token"),
]


def spoken(text: str) -> str:
    for pattern, repl in SPOKEN:
        text = re.sub(pattern, repl, text)
    return text


def sentences(text: str) -> list[str]:
    parts = re.findall(r"[^。！？；]+[。！？；]?", text)
    return [p.strip() for p in parts if p.strip()]


def weight(s: str) -> float:
    return sum(1.0 if re.match(r"[⺀-￯]", ch) else 0.5 for ch in s if not ch.isspace())


def subtitle_chunks(sentence: str, limit: float = 24) -> list[str]:
    """Split a sentence at commas and colons into chunks short enough for one subtitle line."""
    pieces = [p for p in re.split(r"(?<=[，：、])", sentence) if p]
    chunks: list[str] = []
    for p in pieces:
        if chunks and weight(chunks[-1] + p) <= limit:
            chunks[-1] += p
        else:
            chunks.append(p)
    return [c.strip().rstrip("，。；、：") for c in chunks if c.strip().rstrip("，。；、：")]


def make_tts(model_dir: Path):
    import sherpa_onnx

    m = model_dir
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(m / "model.onnx"),
                voices=str(m / "voices.bin"),
                tokens=str(m / "tokens.txt"),
                data_dir=str(m / "espeak-ng-data"),
                dict_dir=str(m / "dict"),
                lexicon=f"{m / 'lexicon-us-en.txt'},{m / 'lexicon-zh.txt'}",
            ),
            num_threads=12,
        ),
        rule_fsts=f"{m / 'date-zh.fst'},{m / 'phone-zh.fst'},{m / 'number-zh.fst'}",
        max_num_sentences=1,
    )
    return sherpa_onnx.OfflineTts(cfg)


def synth(tts, sentence: str) -> np.ndarray:
    """Synthesize one sentence, cached on disk by its spoken text and voice."""
    text = spoken(sentence)
    key = hashlib.sha1(f"{VOICE_SID}|{SPEED}|{text}".encode()).hexdigest()[:16]
    path = AUDIO / f"{key}.wav"
    if not path.exists():
        audio = tts.generate(text, sid=VOICE_SID, speed=SPEED)
        assert audio.sample_rate == SR
        sf.write(path, np.asarray(audio.samples, dtype=np.float32), SR)
    samples, _ = sf.read(path, dtype="float32")
    return samples


# Which frames an animation shows during which sentences: (sentence indices, frame numbers).
# Frames in a group are spread evenly over the time those sentences take.
ANIMATION_SCHEDULE = {
    "compare": [((0, 1), [0]), ((2,), [1]), ((3, 4), list(range(2, 11))), ((5, 6), [10])],
    "decode": [((0, 1), [0]), ((2, 3), list(range(1, 7)))],
}


def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def check_lexicon(model_dir: Path, segments: list[dict]) -> None:
    words = {
        line.split(" ", 1)[0].lower()
        for line in (model_dir / "lexicon-us-en.txt").read_text().splitlines()
        if line
    }
    missing = set()
    for seg in segments:
        for w in re.findall(r"[A-Za-z][A-Za-z']*", spoken(seg["text"])):
            if w.lower() not in words:
                missing.add(w)
    print("not in lexicon:", sorted(missing) or "none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    segments = json.loads((HERE / "narration.json").read_text())
    if args.check:
        check_lexicon(args.model, segments)
        return

    AUDIO.mkdir(parents=True, exist_ok=True)
    tts = make_tts(args.model)
    audio_parts: list[np.ndarray] = []
    shots: list[tuple[str, float]] = []  # (image path relative to HERE, seconds)
    subs: list[tuple[float, float, str]] = []
    clock = 0.0
    slide_no = 0

    def silence(sec: float) -> np.ndarray:
        return np.zeros(int(round(sec * SR)), dtype=np.float32)

    for seg in segments:
        sents = sentences(seg["text"])
        spans = []  # (start, end) of each sentence within the segment
        seg_audio = [silence(LEAD)]
        t = LEAD
        for i, s in enumerate(sents):
            a = synth(tts, s)
            spans.append((t, t + len(a) / SR))
            seg_audio.append(a)
            t += len(a) / SR
            gap = GAP if i < len(sents) - 1 else TAIL
            seg_audio.append(silence(gap))
            t += gap
        seg_len = sum(len(a) for a in seg_audio) / SR
        audio_parts.extend(seg_audio)

        for s, (a, b) in zip(sents, spans, strict=True):
            chunks = subtitle_chunks(s)
            total = sum(weight(c) for c in chunks)
            ca = a
            for c in chunks:
                cb = ca + (b - a) * weight(c) / total
                subs.append((clock + ca, clock + cb, c))
                ca = cb

        if seg.get("type") == "animation":
            frames = sorted((BUILD / "frames" / seg["frames"]).glob("*.png"))
            timeline = []
            for idxs, frame_ids in ANIMATION_SCHEDULE[seg["frames"]]:
                start = spans[idxs[0]][0] if timeline else 0.0
                end = spans[idxs[-1]][1]
                step = (end - start) / len(frame_ids)
                for k, f in enumerate(frame_ids):
                    timeline.append((f, start + k * step))
            for k, (f, start) in enumerate(timeline):
                end = timeline[k + 1][1] if k + 1 < len(timeline) else seg_len
                shots.append((str(frames[f].relative_to(HERE)), end - start))
        else:
            slide_no += 1
            shots.append((f"build/render/slide-{slide_no:02d}.png", seg_len))
        clock += seg_len
        print(f"{seg['id']:18s} {seg_len:6.1f}s  (total {clock / 60:5.1f} min)")

    audio = np.concatenate(audio_parts)
    sf.write(BUILD / "narration.wav", audio, SR)

    # Snap shot boundaries to whole frames so video and audio stay in sync over the long run.
    lines, t_shown = [], 0.0
    t_target = 0.0
    for path, dur in shots:
        t_target += dur
        frames = round(t_target * FPS) - round(t_shown * FPS)
        lines += [f"file '../{path}'", f"duration {frames / FPS:.6f}"]
        t_shown = t_target
    lines.append(f"file '../{shots[-1][0]}'")
    (BUILD / "shots.txt").write_text("\n".join(lines) + "\n")

    srt = []
    for i, (a, b, text) in enumerate(subs, 1):
        srt += [str(i), f"{srt_time(a)} --> {srt_time(b - 0.02)}", text, ""]
    (HERE / "gpt2_inference.srt").write_text("\n".join(srt), encoding="utf-8")

    style = (
        "FontName=Noto Sans CJK SC,FontSize=11,PrimaryColour=&H00FFFFFF,"
        "BorderStyle=1,Outline=0,Shadow=0,MarginV=7,Alignment=2"
    )
    vf = (
        "scale=1728:972:flags=lanczos,pad=1920:1080:96:12:color=0x1B1F3B,"
        f"subtitles=gpt2_inference.srt:force_style='{style}',format=yuv420p"
    )
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{_id('-u')}:{_id('-g')}",
        "-e",
        "HOME=/tmp",
        "-v",
        f"{HERE}:/work",
        "-w",
        "/work",
        "learn-gpt2-media",
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-stats",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "build/shots.txt",
        "-i",
        "build/narration.wav",
        "-vf",
        vf,
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-t",
        f"{len(audio) / SR:.3f}",
        "-movflags",
        "+faststart",
        "gpt2_inference.mp4",
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote gpt2_inference.mp4 ({clock / 60:.1f} min) and gpt2_inference.srt")


def _id(flag: str) -> str:
    return subprocess.run(["id", flag], capture_output=True, text=True, check=True).stdout.strip()


if __name__ == "__main__":
    main()
