"""Create Chinese narration, subtitles, chapter markers, and the overview video.

Run from any directory after creating slides.json and build/render/slide-XX.png:
    python video.py --audio-only
    python video.py --video-only

Speech synthesis requires network access. Cached speech and --video-only work offline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import edge_tts
import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
AUDIO = BUILD / "audio"
SAMPLE_RATE = 24000
LEAD = 0.6
TAIL = 0.9


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cache_key(text: str, voice: str, rate: str) -> str:
    return hashlib.sha256(f"v1|{voice}|{rate}|{text}".encode()).hexdigest()[:24]


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as audio:
        return audio.getnframes() / audio.getframerate()


async def synthesize(
    slide: dict[str, Any], index: int, args: argparse.Namespace, semaphore: asyncio.Semaphore
) -> dict[str, Any]:
    """Cache one page of speech with the service's sentence-level timing."""
    text = slide["narration"].strip()
    key = cache_key(text, args.voice, args.rate)
    mp3 = AUDIO / f"{key}.mp3"
    wav = AUDIO / f"{key}.wav"
    metadata = AUDIO / f"{key}.json"
    async with semaphore:
        if not (mp3.exists() and wav.exists() and metadata.exists()):
            if args.video_only:
                raise RuntimeError(f"Slide {index}: speech cache missing; run --audio-only first")
            for attempt in range(4):
                temporary = mp3.with_suffix(".partial.mp3")
                boundaries: list[dict[str, Any]] = []
                try:
                    communicate = edge_tts.Communicate(
                        text,
                        args.voice,
                        rate=args.rate,
                        boundary="SentenceBoundary",
                        connect_timeout=15,
                        receive_timeout=90,
                    )
                    with temporary.open("wb") as stream:
                        async for item in communicate.stream():
                            if item["type"] == "audio":
                                stream.write(item["data"])
                            elif item["type"] == "SentenceBoundary":
                                boundaries.append(
                                    {
                                        "start": item["offset"] / 10_000_000,
                                        "end": (item["offset"] + item["duration"]) / 10_000_000,
                                        "text": item["text"],
                                    }
                                )
                    if not boundaries or temporary.stat().st_size < 1000:
                        raise RuntimeError(
                            "Speech service returned no audio or sentence timestamps"
                        )
                    temporary.replace(mp3)
                    run(
                        [
                            args.ffmpeg,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(mp3),
                            "-ar",
                            str(SAMPLE_RATE),
                            "-ac",
                            "1",
                            "-c:a",
                            "pcm_s16le",
                            str(wav),
                        ]
                    )
                    write_json(metadata, {"boundaries": boundaries, "key": key})
                    break
                except Exception as error:
                    temporary.unlink(missing_ok=True)
                    if attempt == 3:
                        raise RuntimeError(f"Slide {index}: speech synthesis failed") from error
                    print(f"Retry slide {index}, attempt {attempt + 2}: {error}", flush=True)
                    await asyncio.sleep(2 ** (attempt + 1))
        duration = wav_duration(wav)
        print(f"Audio {index:02d}: {duration:6.1f}s  {slide['title']}", flush=True)
        return {"wav": wav, "duration": duration, **read_json(metadata)}


def text_weight(text: str) -> float:
    return sum(1 if "\u2e80" <= char <= "\uffef" else 0.55 for char in text)


def subtitle_chunks(text: str, limit: float = 35) -> list[str]:
    """Split long sentence captions at punctuation, with a hard one-line fallback."""
    parts = re.findall(r"[^，。！？；：,!?;:]+[，。！？；：,!?;:]?", text)
    chunks: list[str] = []
    current = ""
    for part in parts:
        if current and text_weight(current + part) > limit:
            chunks.append(current.strip())
            current = ""
        for char in part:
            if current and text_weight(current + char) > limit:
                chunks.append(current.strip())
                current = ""
            current += char
    if current.strip():
        chunks.append(current.strip())
    return chunks


def srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_int, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"


def ass_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    seconds_int, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{seconds_int:02d}.{centiseconds:02d}"


def chapter_time(seconds: float) -> str:
    seconds_int = round(seconds)
    hours, seconds_int = divmod(seconds_int, 3600)
    minutes, seconds_int = divmod(seconds_int, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"


def create_timeline(
    slides: list[dict[str, Any]], audios: list[dict[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    subtitles: list[tuple[float, float, str]] = []
    chapters: list[dict[str, Any]] = []
    clock = 0.0
    narration_lines = ["# 从注意力机制到 GPT-2 与 GPT-3 逐页讲稿", ""]
    with wave.open(str(BUILD / "narration.wav"), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        for index, (slide, audio) in enumerate(zip(slides, audios, strict=True), 1):
            output.writeframes(b"\x00\x00" * round(LEAD * SAMPLE_RATE))
            with wave.open(str(audio["wav"])) as source:
                output.writeframes(source.readframes(source.getnframes()))
            output.writeframes(b"\x00\x00" * round(TAIL * SAMPLE_RATE))
            page_end = clock + LEAD + audio["duration"] + TAIL
            chapters.append(
                {
                    "slide": index,
                    "title": slide["title"],
                    "section": slide["section"],
                    "start": round(clock, 4),
                    "end": round(page_end, 4),
                    "timestamp": chapter_time(clock),
                    "audio_key": audio["key"],
                }
            )
            boundaries = audio["boundaries"]
            for part_index, boundary in enumerate(boundaries):
                start = max(0.0, boundary["start"])
                end = min(audio["duration"], boundary["end"])
                if part_index + 1 < len(boundaries):
                    end = min(end, boundaries[part_index + 1]["start"])
                if end <= start:
                    continue
                chunks = subtitle_chunks(boundary["text"])
                total = sum(text_weight(chunk) for chunk in chunks)
                cursor = start
                for chunk in chunks:
                    next_cursor = cursor + (end - start) * text_weight(chunk) / total
                    subtitles.append((clock + LEAD + cursor, clock + LEAD + next_cursor, chunk))
                    cursor = next_cursor
            narration_lines.extend(
                [
                    f"## {index:02d} {slide['title']}",
                    "",
                    f"{chapter_time(clock)} · {slide['section']}",
                    "",
                    slide["narration"],
                    "",
                    f"来源：{slide.get('source', '原创教学说明')}",
                    "",
                ]
            )
            clock = page_end

    srt_lines = []
    ass_lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Microsoft YaHei,40,&H00FFFFFF,&H00FFFFFF,&H00101010,&H00000000,"
        "0,0,0,0,100,100,0,0,1,0,0,2,70,70,26,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for index, (start, end, text) in enumerate(subtitles, 1):
        if end - start < 0.02:
            continue
        srt_lines.extend([str(index), f"{srt_time(start)} --> {srt_time(end)}", text, ""])
        escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
        ass_lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{escaped}")
    (HERE / "overview.srt").write_text("\n".join(srt_lines), encoding="utf-8")
    (BUILD / "subtitles.ass").write_text("\n".join(ass_lines) + "\n", encoding="utf-8")
    (HERE / "narration.md").write_text("\n".join(narration_lines), encoding="utf-8")
    write_json(HERE / "chapters.json", chapters)
    manifest = {
        "slide_count": len(slides),
        "duration_seconds": round(clock, 4),
        "duration_hms": chapter_time(clock),
        "voice": args.voice,
        "rate": args.rate,
        "fps": args.fps,
        "width": 1920,
        "height": 1080,
        "subtitle_count": len(subtitles),
        "subtitle_timing": "Speech-service sentence timestamps; proportional sub-sentence splits",
        "chapters": chapters,
    }
    write_json(HERE / "media_manifest.json", manifest)
    return manifest


def escape_metadata(text: str) -> str:
    return re.sub(r"([\\=;#])", r"\\\1", text).replace("\n", " ")


def render_video(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    fonts = BUILD / "fonts"
    fonts.mkdir(exist_ok=True)
    if args.font and args.font.exists():
        shutil.copy2(args.font, fonts / args.font.name)
    elif not list(fonts.glob("*")):
        raise FileNotFoundError("A Chinese font is required: pass --font /path/to/msyh.ttc")
    shot_lines = []
    chapter_lines = [";FFMETADATA1", "title=从注意力机制到 GPT-2 与 GPT-3", "language=zho"]
    for chapter in manifest["chapters"]:
        image = BUILD / "render" / f"slide-{chapter['slide']:02d}.png"
        if not image.exists():
            raise FileNotFoundError(image)
        start = round(chapter["start"] * args.fps)
        end = round(chapter["end"] * args.fps)
        shot_lines.extend(
            [f"file 'render/{image.name}'", f"duration {(end - start) / args.fps:.8f}"]
        )
        chapter_lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={round(chapter['start'] * 1000)}",
                f"END={round(chapter['end'] * 1000)}",
                f"title={chapter['slide']:02d} {escape_metadata(chapter['title'])}",
            ]
        )
    shot_lines.append(f"file 'render/slide-{manifest['slide_count']:02d}.png'")
    (BUILD / "shots.txt").write_text("\n".join(shot_lines) + "\n", encoding="utf-8")
    (BUILD / "chapters.ffmeta").write_text("\n".join(chapter_lines) + "\n", encoding="utf-8")
    command = [
        args.ffmpeg,
        "-hide_banner",
        "-y",
        "-loglevel",
        "warning",
        "-stats",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "build/shots.txt",
        "-i",
        "build/narration.wav",
        "-f",
        "ffmetadata",
        "-i",
        "build/chapters.ffmeta",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "2",
        "-map_chapters",
        "2",
        "-metadata:s:a:0",
        "language=zho",
        "-vf",
        f"fps={args.fps},drawbox=x=0:y=990:w=1920:h=90:color=0x080D16:t=fill,"
        "ass=build/subtitles.ass:fontsdir=build/fonts,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-tune",
        "stillimage",
        "-threads",
        str(args.threads),
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-af",
        "loudnorm=I=-18:TP=-2:LRA=7",
        "-ar",
        "48000",
        "-t",
        str(manifest["duration_seconds"]),
        "-movflags",
        "+faststart",
        "overview.mp4",
    ]
    subprocess.run(command, cwd=HERE, check=True)


async def main_async(args: argparse.Namespace) -> None:
    AUDIO.mkdir(parents=True, exist_ok=True)
    slides = read_json(HERE / "slides.json")
    if args.limit:
        slides = slides[: args.limit]
    for index, slide in enumerate(slides, 1):
        for field in ("title", "section", "narration"):
            if not slide.get(field):
                raise ValueError(f"Slide {index} has no {field}")
    semaphore = asyncio.Semaphore(args.workers)
    audios = await asyncio.gather(
        *(synthesize(slide, index, args, semaphore) for index, slide in enumerate(slides, 1))
    )
    manifest = create_timeline(slides, audios, args)
    print(
        f"Narration ready: {manifest['slide_count']} slides, "
        f"{manifest['duration_hms']}, {manifest['subtitle_count']} captions",
        flush=True,
    )
    if not args.audio_only:
        render_video(manifest, args)
        print(f"Video ready: {HERE / 'overview.mp4'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--audio-only", action="store_true")
    stages.add_argument("--video-only", action="store_true")
    parser.add_argument("--voice", default="zh-CN-YunxiNeural")
    parser.add_argument("--rate", default="+0%")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=22)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--limit", type=int, help="Build only the first N slides for a preview")
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/msyh.ttc"))
    parser.add_argument("--ffmpeg", default=imageio_ffmpeg.get_ffmpeg_exe())
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
