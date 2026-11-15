"""Turn the slide PNGs and narrations into one narrated MP4.

Voice is edge-tts zh-CN-XiaoxiaoNeural at -8%, the same voice family used for the
causal-attention deck. Subtitles are drawn with Noto Sans CJK SC, so the video
does not depend on libass finding a Chinese font.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from pathlib import Path

import build_deck
import edge_tts
from engine import BG, BUILD, PNG_DIR, WHITE, lh, load_font, wrap
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
AUDIO = BUILD / "audio"
FRAMES = BUILD / "frames"
SEGMENTS = BUILD / "segments"
VOICE = "zh-CN-XiaoxiaoNeural"
RATE = "-8%"
PAD = 0.4
FPS = 30


def _sec(ticks: int) -> float:
    return ticks / 10_000_000


def _stamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    if millis == 1000:
        whole += 1
        millis = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _cues(boundaries: list[dict], audio_end: float) -> list[tuple[float, float, str]]:
    sentences = [item for item in boundaries if item["type"] == "SentenceBoundary"]
    cues: list[tuple[float, float, str]] = []
    if not sentences:
        return []
    for item in sentences:
        start = _sec(item["offset"])
        end = start + _sec(item["duration"])
        text = item["text"].strip()
        # One visual line per cue. The bottom band is only tall enough for a single line.
        lines = [line for line in wrap(text, 22, False, False, 1600) if line]
        if not lines:
            continue
        weights = [max(len(line), 1) for line in lines]
        span = max(end - start, 0.2)
        cursor = start
        for line, weight in zip(lines, weights, strict=True):
            piece = span * weight / sum(weights)
            cues.append((cursor, cursor + piece, line))
            cursor += piece
    # Sentence boundaries sometimes overlap by a few tens of milliseconds.
    # The frame list is sequential, so an overlap would stretch the picture past the audio.
    ordered: list[tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, text in cues:
        start = max(start, cursor)
        if end <= start + 0.04 or not text:
            continue
        ordered.append((start, end, text))
        cursor = end
    if not ordered:
        return []
    last_start, _, last_text = ordered[-1]
    ordered[-1] = (last_start, max(ordered[-1][1], audio_end), last_text)
    return ordered


def _subtitle_image(base: Path, text: str, dest: Path) -> None:
    image = Image.open(base).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 986, 1920, 1080), fill=f"#{BG}")
    if text:
        points = 22
        font = load_font("sans", points)
        while font.getlength(text) > 1680 and points > 16:
            points -= 1
            font = load_font("sans", points)
        top = 986 + (94 - lh(points)) / 2
        width = font.getlength(text)
        draw.text(((1920 - width) / 2, top), text, font=font, fill=f"#{WHITE}", anchor="lt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(probe.stdout.strip())


async def _synthesize(text: str, dest: Path) -> list[dict]:
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    audio = bytearray()
    boundaries: list[dict] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        else:
            boundaries.append({key: chunk[key] for key in ("type", "offset", "duration", "text")})
    dest.write_bytes(audio)
    return boundaries


async def _ensure_audio(spec: dict, index: int) -> tuple[Path, list[dict]]:
    key = hashlib.sha256(f"{VOICE}|{RATE}|{spec['narration']}".encode()).hexdigest()[:16]
    mp3 = AUDIO / f"slide-{index:02d}-{key}.mp3"
    meta = AUDIO / f"slide-{index:02d}-{key}.json"
    if mp3.exists() and meta.exists():
        return mp3, json.loads(meta.read_text(encoding="utf-8"))
    boundaries = await _synthesize(spec["narration"], mp3)
    meta.write_text(json.dumps(boundaries, ensure_ascii=False), encoding="utf-8")
    return mp3, boundaries


def _encode_segment(
    index: int, cues: list[tuple[float, float, str]], audio: Path, total: float
) -> Path:
    folder = FRAMES / f"slide-{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    base = PNG_DIR / f"slide-{index:02d}.png"
    blank = folder / "blank.png"
    _subtitle_image(base, "", blank)
    frames: list[tuple[Path, float]] = []
    cursor = 0.0
    for order, (start, end, text) in enumerate(cues):
        if start > cursor + 0.05:
            frames.append((blank, start - cursor))
        picture = folder / f"cue-{order:02d}.png"
        _subtitle_image(base, text, picture)
        frames.append((picture, max(end - start, 0.05)))
        cursor = end
    if not frames:
        frames.append((blank, total))
        cursor = total
    elif total > cursor + 0.02:
        frames.append((frames[-1][0], total - cursor))
    script = folder / "frames.txt"
    lines = []
    for path, duration in frames:
        lines.append(f"file '{path}'")
        lines.append(f"duration {duration:.3f}")
    lines.append(f"file '{frames[-1][0]}'")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    segment = SEGMENTS / f"slide-{index:02d}.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(script),
            "-i",
            str(audio),
            "-filter_complex",
            f"[1:a]apad=pad_dur={PAD}[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-r",
            str(FPS),
            "-tune",
            "stillimage",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "160k",
            "-t",
            f"{total:.3f}",
            "-movflags",
            "+faststart",
            str(segment),
        ]
    )
    return segment


def _write_sidecars(rows: list[dict]) -> None:
    srt: list[str] = []
    narration = [
        "# 旁白稿：从注意力到大语言模型",
        "",
        f"配音：edge-tts `{VOICE}`，语速 {RATE}。",
        "",
    ]
    chapters = []
    cursor = 0.0
    cue_index = 1
    for row in rows:
        narration.append(f"## 第 {row['index']} 页 · {row['name']} · {_stamp(cursor)[:-4]}")
        narration.append("")
        narration.append(row["narration"])
        narration.append("")
        chapters.append(
            {
                "index": row["index"],
                "name": row["name"],
                "start": round(cursor, 3),
                "title": row["title"],
            }
        )
        for start, end, text in row["cues"]:
            srt.append(str(cue_index))
            srt.append(f"{_stamp(cursor + start)} --> {_stamp(cursor + end)}")
            srt.append(text)
            srt.append("")
            cue_index += 1
        cursor += row["duration"]
    (HERE / "transformer_gpt2_gpt3.srt").write_text("\n".join(srt), encoding="utf-8")
    (HERE / "narration.md").write_text("\n".join(narration), encoding="utf-8")
    (HERE / "chapters.json").write_text(
        json.dumps(chapters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    meta = [";FFMETADATA1"]
    cursor = 0.0
    for row in rows:
        meta.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={int(cursor * 1000)}",
                f"END={int((cursor + row['duration']) * 1000)}",
                f"title=第{row['index']}页 {row['title']}",
            ]
        )
        cursor += row["duration"]
    (BUILD / "chapters.txt").write_text("\n".join(meta) + "\n", encoding="utf-8")


def _titles() -> list[str]:
    # The first heading-like string is not stored separately; use the function name's
    # spoken slide order and a short label taken from the narration's first clause.
    titles = []
    for spec in build_deck.SLIDES:
        clause = spec["narration"].split("。")[0]
        titles.append(clause[:18])
    return titles


async def build() -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    SEGMENTS.mkdir(parents=True, exist_ok=True)
    titles = _titles()
    rows = []
    for index, spec in enumerate(build_deck.SLIDES, start=1):
        audio, boundaries = await _ensure_audio(spec, index)
        spoken = _duration(audio)
        total = spoken + PAD
        cues = _cues(boundaries, spoken)
        segment = _encode_segment(index, cues, audio, total)
        # Concatenation keeps each file's encoded length, which frame rounding
        # can move off the requested audio-plus-pad duration. Sidecars must use
        # that encoded length or later chapters and subtitles drift early.
        encoded = _duration(segment)
        fitted = [
            (start, min(end, encoded), text) for start, end, text in cues if start < encoded - 0.02
        ]
        print(
            f"encode {index:02d} requested {total:.3f}s encoded {encoded:.3f}s {len(fitted)} cues"
        )
        rows.append(
            {
                "index": index,
                "name": spec["draw"].__name__,
                "title": titles[index - 1],
                "narration": spec["narration"],
                "duration": encoded,
                "cues": fitted,
            }
        )
    _write_sidecars(rows)
    concat = SEGMENTS / "concat.txt"
    lines = []
    for row in rows:
        filename = f"slide-{row['index']:02d}.mp4"
        lines.append(f"file '{SEGMENTS / filename}'")
    concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
    silent = HERE / "transformer_gpt2_gpt3.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            str(BUILD / "joined.mp4"),
        ]
    )
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(BUILD / "joined.mp4"),
            "-i",
            str(BUILD / "chapters.txt"),
            "-map_metadata",
            "1",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(silent),
        ]
    )
    encoded_total = sum(row["duration"] for row in rows)
    final = _duration(silent)
    print(f"sidecar sum {encoded_total:.3f}s video {final:.3f}s")


if __name__ == "__main__":
    asyncio.run(build())
