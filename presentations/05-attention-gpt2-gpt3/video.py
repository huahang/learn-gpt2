"""Narrate the deck with local TTS and assemble attention_gpt2_gpt3.mp4 (+ .srt).

    python video.py --model <kokoro-multi-lang-v1_1 dir> [--check]

Needs build/render/slide-NN.png (render.sh), build/frames/ (figures.py), build/anim_boxes.json
(deck.js) and ffmpeg with libx264 and libass on PATH. --check only lists words the TTS lexicon
lacks, after the spoken-form rewrites below.
"""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

HERE = Path(__file__).parent
BUILD = HERE / "build"
AUDIO = BUILD / "audio"
NAME = "attention_gpt2_gpt3"
VOICE_SID = 3  # Kokoro v1.1 speaker id; 3 and up are Mandarin voices
SPEED = 1.0
SR = 24000
LEAD, GAP, TAIL = 0.45, 0.28, 0.75  # seconds of silence: before, between and after sentences
FPS = 30
PX_PER_IN = 1920 / 13.333  # slide renders are 1920 px wide for a 13.333 in slide


def _letters(m: re.Match) -> str:
    return " ".join(m.group(0).replace("-", ""))


# Spoken forms for terms the TTS lexicon lacks (it silently skips unknown English words).
# Order matters: longer patterns first.
SPOKEN = [
    (r"gpt2\.py", "G P T 2 点 P Y"),
    (r"gpt2-xl", "G P T 2 X L"),
    (r"(?i)\bgpt-?([123])\b", r"G P T \1"),
    (r"\bGPT\b", "G P T"),
    (r"presentations", "presentation"),
    (r"LayerKVCache", "layer K V cache"),
    (r"KV Cache", "K V cache"),
    (r"\bKV\b", "K V"),
    (r"\bQKV\b", "Q K V"),
    (r"CausalSelfAttention", "causal self attention"),
    (r"PagedAttention", "paged attention"),
    (r"PyTorch", "pie torch"),
    (r"softmax", "soft max"),
    (r"logits", "lodge its"),
    (r"c attn", "C attention"),
    (r"\bidx\b", "I D X"),
    (r"WebText2", "web text 2"),
    (r"WebText", "web text"),
    (r"SuperGLUE", "super glue"),
    (r"TriviaQA", "trivia Q A"),
    (r"CoQA", "co Q A"),
    (r"LAMBADA", "lambada"),
    (r"BLEU", "blue"),
    (r"Reddit", "red it"),
    (r"Pre-LN", "pre L N"),
    (r"ReLU", "R E L U"),
    (r"RoPE", "rope"),
    (r"\bTL DR\b", "T L D R"),
    (r"\bWiC\b", "W I C"),
    (r"UTF-8", "U T F 8"),
    (r"\bMiB\b", "M B"),
    (r"\bGiB\b", "G B"),
    (r"WMT 2014", "W M T 二零一四"),
    (r"\b([PV])(100)\b", r"\1 \2"),
    (r"\b(T5|F1)\b", _letters),
    (r"\b(LSTM|RNN|BPE|WMT|GPU|MQA|GQA|ANLI|RTX|BERT|GB|MB)\b", _letters),
    (r"\btokens\b", "token"),
    (r"(\d+(?:\.\d+)?)%", r"百分之\1"),
]

DIGITS = "零一二三四五六七八九"


def zh_number(n: int) -> str:
    """Chinese numeral for 0 <= n < 10**8, e.g. 1750 -> 一千七百五十, 524800 -> 五十二万四千八百."""
    if n == 0:
        return DIGITS[0]
    if n >= 10000:
        high, low = divmod(n, 10000)
        out = zh_number(high) + "万"
        if low:
            out += (DIGITS[0] if low < 1000 else "") + zh_number(low)
        return out
    out, zero = "", False
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // value % 10
        if d:
            if zero and out:
                out += DIGITS[0]
            out += DIGITS[d] + unit
            zero = False
        else:
            zero = True
    return out


def _big_number(m: re.Match) -> str:
    # Years ("2017 年") stay as they are; other integers >= 1000 get read digit by digit otherwise.
    if m.group(2):
        return m.group(0)
    return re.sub(r"^二(?=[千万])", "两", zh_number(int(m.group(1))))


def spoken(text: str) -> str:
    for pattern, repl in SPOKEN:
        text = re.sub(pattern, repl, text)
    return re.sub(r"(?<![\d.])([1-9]\d{3,7})(?![\d.])(\s*年)?", _big_number, text)


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
            num_threads=4,
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
    missing: dict[str, str] = {}
    for seg in segments:
        for w in re.findall(r"[A-Za-z][A-Za-z']*", spoken(seg["text"])):
            if w.lower() not in words:
                missing.setdefault(w, seg["id"])
    print("not in lexicon:", missing or "none")


def composite_frames(anim: str, slide_png: Path, box: dict) -> list[Path]:
    """Paste each animation frame onto the rendered slide at the figure's position."""
    out_dir = BUILD / "anim" / anim
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Image.open(slide_png).convert("RGB")
    x, y = round(box["x"] * PX_PER_IN), round(box["y"] * PX_PER_IN)
    w, h = round(box["w"] * PX_PER_IN), round(box["h"] * PX_PER_IN)
    paths = []
    for frame in sorted((BUILD / "frames" / anim).glob("*.png")):
        img = base.copy()
        fr = Image.open(frame).convert("RGB").resize((w, h), Image.LANCZOS)
        img.paste(fr, (x, y))
        path = out_dir / frame.name
        img.save(path)
        paths.append(path)
    return paths


def cue_time(cue: float, spans: list[tuple[float, float]]) -> float:
    """Sentence index (fraction = position within the sentence) -> seconds into the segment."""
    if cue == 0:
        return 0.0
    i = math.floor(cue)
    a, b = spans[i]
    return a + (cue - i) * (b - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    segments = json.loads((HERE / "narration.json").read_text())
    if args.check:
        check_lexicon(args.model, segments)
        return

    boxes = json.loads((BUILD / "anim_boxes.json").read_text())
    AUDIO.mkdir(parents=True, exist_ok=True)
    tts = make_tts(args.model)
    audio_parts: list[np.ndarray] = []
    shots: list[tuple[str, float]] = []  # (image path relative to HERE, seconds)
    subs: list[tuple[float, float, str]] = []
    clock = 0.0

    def silence(sec: float) -> np.ndarray:
        return np.zeros(int(round(sec * SR)), dtype=np.float32)

    for slide_no, seg in enumerate(segments, 1):
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

        slide_png = BUILD / "render" / f"slide-{slide_no:02d}.png"
        if "anim" in seg:
            anim = seg["anim"]
            box = boxes[str(slide_no)]
            assert box["name"] == anim["name"], (slide_no, box, anim)
            frames = composite_frames(anim["name"], slide_png, box)
            cues = anim["cues"]
            assert len(frames) == len(cues), (anim["name"], len(frames), len(cues))
            starts = [cue_time(c, spans) for c in cues]
            for k, frame in enumerate(frames):
                end = starts[k + 1] if k + 1 < len(starts) else seg_len
                shots.append((str(frame.relative_to(HERE)), end - starts[k]))
        else:
            shots.append((str(slide_png.relative_to(HERE)), seg_len))
        clock += seg_len
        print(f"{slide_no:2d} {seg['id']:20s} {seg_len:6.1f}s  (total {clock / 60:5.1f} min)")

    audio = np.concatenate(audio_parts)
    sf.write(BUILD / "narration.wav", audio, SR)

    # Snap shot boundaries to whole frames so video and audio stay in sync over the long run.
    lines, t_shown, t_target = [], 0.0, 0.0
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
    (HERE / f"{NAME}.srt").write_text("\n".join(srt), encoding="utf-8")

    style = (
        "FontName=Noto Sans CJK SC,FontSize=11,PrimaryColour=&H00FFFFFF,"
        "BorderStyle=1,Outline=0,Shadow=0,MarginV=7,Alignment=2"
    )
    vf = (
        "scale=1728:972:flags=lanczos,pad=1920:1080:96:12:color=0x1B1F3B,"
        f"subtitles={NAME}.srt:force_style='{style}',format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-stats",
        "-f", "concat", "-safe", "0", "-i", "build/shots.txt",
        "-i", "build/narration.wav",
        "-vf", vf, "-r", str(FPS),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "160k",
        "-t", f"{len(audio) / SR:.3f}", "-movflags", "+faststart",
        f"{NAME}.mp4",
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=HERE)
    print(f"wrote {NAME}.mp4 ({clock / 60:.1f} min) and {NAME}.srt")


if __name__ == "__main__":
    main()
