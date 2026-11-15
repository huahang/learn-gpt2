"""Draw one 16:9 slide into both a PPTX shape tree and a 1920x1080 PNG.

The palette, typefaces, and 10 by 5.625 inch page match the causal-attention deck
in presentations/04 (PR #13): ink background #090715, violet #916EF0, Noto Sans
CJK SC for Chinese, Inter for Latin, JetBrains Mono for code.
"""

from __future__ import annotations

import contextlib
import os
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
PNG_DIR = BUILD / "png"

W, H = 1920, 1080
PPI = 192  # 10 in x 5.625 in at 1920x1080, same page as PR #13
MARGIN = 88

BG = "090715"
WHITE = "FFFFFF"
MUTED = "B4B4C0"
DIM = "6E6A86"
PURPLE = "916EF0"
VIOLET = "9A31E8"
DEEP = "6649A5"
CARD = "140C2C"
CARD_EDGE = "2E2358"
BLUE = "82AAFF"
LAVENDER = "E6E1F5"
BLOCKED = "1A1230"

OVERFLOWS: list[str] = []


def px(points: float) -> int:
    return int(round(points * PPI / 72))


def lh(points: int, lines: int = 1) -> float:
    """Line box height. Matches the leading used by Board.text."""
    return px(points) * 1.16 * lines


def inches(pixels: float) -> float:
    return pixels / PPI


def rgb(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


# Keep a trailing mark on the previous line. A line that is only "。" has no
# Han character, so it would otherwise be drawn with Inter and become a missing glyph.
STICKY_PUNCT = set("。，、；：！？》）】」』")


def _first(paths: list[str]) -> str:
    for path in paths:
        if path and Path(path).is_file():
            return path
    raise FileNotFoundError(
        "Missing a font. See README.md for Noto Sans CJK SC, Inter, JetBrains Mono."
    )


def font_paths() -> dict[str, str]:
    inter_dir = os.environ.get("INTER_DIR", "/tmp/fonts/inter/extras/otf")
    return {
        "noto": _first(
            [
                os.environ.get("NOTO_REGULAR", ""),
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]
        ),
        "noto_bold": _first(
            [
                os.environ.get("NOTO_BOLD", ""),
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            ]
        ),
        "inter": _first([os.environ.get("INTER_REGULAR", ""), f"{inter_dir}/Inter-Regular.otf"]),
        "inter_bold": _first(
            [os.environ.get("INTER_SEMIBOLD", ""), f"{inter_dir}/Inter-SemiBold.otf"]
        ),
        "mono": _first(
            [
                os.environ.get("MONO_REGULAR", ""),
                "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
            ]
        ),
        "mono_bold": _first(
            [
                os.environ.get("MONO_BOLD", ""),
                "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
            ]
        ),
    }


PATHS = None


def paths() -> dict[str, str]:
    global PATHS
    if PATHS is None:
        PATHS = font_paths()
    return PATHS


FONT_KIND = {
    "sans": "noto",
    "sans_bold": "noto_bold",
    "latin": "inter",
    "latin_bold": "inter_bold",
    "mono": "mono",
    "mono_bold": "mono_bold",
}


@lru_cache(maxsize=256)
def load_font(kind: str, points: int) -> ImageFont.FreeTypeFont:
    file = paths()[FONT_KIND[kind]]
    index = 2 if file.endswith(".ttc") else 0
    return ImageFont.truetype(file, px(points), index=index)


def _latin_only_punctuation(text: str) -> bool:
    return bool(text) and all(ch in STICKY_PUNCT or ch.isspace() for ch in text)


def face_for(
    text: str, points: int, bold: bool, mono: bool, prefer_cjk: bool = False
) -> ImageFont.FreeTypeFont:
    if prefer_cjk or has_cjk(text) or _latin_only_punctuation(text):
        return load_font("sans_bold" if bold else "sans", points)
    if mono:
        return load_font("mono_bold" if bold else "mono", points)
    return load_font("latin_bold" if bold else "latin", points)


def typeface_name(text: str, mono: bool, prefer_cjk: bool = False) -> str:
    if prefer_cjk or has_cjk(text) or _latin_only_punctuation(text):
        return "Noto Sans CJK SC"
    if mono:
        return "JetBrains Mono"
    return "Inter"


def wrap(text: str, points: int, bold: bool, mono: bool, width: float) -> list[str]:
    prefer_cjk = has_cjk(text)
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            lines.append("")
            continue
        current = ""
        for ch in paragraph:
            trial = current + ch
            font = face_for(trial, points, bold, mono, prefer_cjk=prefer_cjk)
            if current and font.getlength(trial) > width:
                if ch in STICKY_PUNCT:
                    current = trial
                    continue
                split_at = current.rfind(" ")
                tail = current[split_at + 1 :] if split_at > 0 else ""
                if split_at > 0 and tail and tail.isascii() and len(tail) > 1:
                    lines.append(current[:split_at])
                    current = tail + ch
                else:
                    lines.append(current)
                    current = ch
            else:
                current = trial
        lines.append(current)
    return lines


def mix(start: str, end: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    out = []
    for i in range(0, 6, 2):
        a = int(start[i : i + 2], 16)
        b = int(end[i : i + 2], 16)
        out.append(f"{int(a + (b - a) * t):02X}")
    return "".join(out)


def _set_run_font(run, name: str, points: int, bold: bool, color: str) -> None:
    run.font.name = name
    run.font.size = Pt(points)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    r_pr = run._r.get_or_add_rPr()
    for tag in ("latin", "ea", "cs"):
        element = r_pr.find(qn(f"a:{tag}"))
        if element is None:
            element = r_pr.makeelement(qn(f"a:{tag}"))
            r_pr.append(element)
        element.set("typeface", name)


def new_presentation() -> Presentation:
    presentation = Presentation()
    presentation.slide_width = Emu(int(10 * 914400))
    presentation.slide_height = Emu(int(5.625 * 914400))
    presentation.core_properties.title = "从注意力到大语言模型：Transformer、GPT-2 与 GPT-3"
    presentation.core_properties.author = "learn-gpt2"
    presentation.core_properties.subject = "因果自注意力与 KV Cache"
    return presentation


class Board:
    """Pixel coordinates on a 1920x1080 canvas. PPTX uses the same positions at 192 ppi."""

    def __init__(
        self, presentation: Presentation, index: int, total: int, section: str, notes: str
    ):
        self.presentation = presentation
        self.slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = rgb(BG)
        self.image = Image.new("RGB", (W, H), f"#{BG}")
        self.draw = ImageDraw.Draw(self.image)
        self.index = index
        self.total = total
        self.section = section
        self._glow()
        self.slide.notes_slide.notes_text_frame.text = notes

    def _glow(self) -> None:
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        glow = ImageDraw.Draw(overlay)
        glow.ellipse((1280, -260, 2140, 520), fill=(145, 49, 232, 36))
        glow.ellipse((1460, -120, 2060, 360), fill=(145, 110, 240, 28))
        self.image = Image.alpha_composite(self.image.convert("RGBA"), overlay).convert("RGB")
        self.draw = ImageDraw.Draw(self.image)
        # The PNG carries a soft corner glow. The PPTX keeps the flat #090715 field
        # used by the PR #13 deck, so a failed alpha would not paint a solid blob.

    def save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path, "PNG")

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        radius: int = 0,
        line: str | None = None,
    ) -> None:
        outline = f"#{line}" if line else None
        left, top, width, height = (Emu(int(inches(v) * 914400)) for v in (x, y, w, h))
        stroke = 2 if line else 0
        if radius:
            self.draw.rounded_rectangle(
                (x, y, x + w, y + h),
                radius=radius,
                fill=f"#{fill}",
                outline=outline,
                width=stroke,
            )
            shape = self.slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
            )
            with contextlib.suppress(Exception):
                shape.adjustments[0] = min(0.5, radius / max(min(w, h), 1))
        else:
            self.draw.rectangle(
                (x, y, x + w, y + h), fill=f"#{fill}", outline=outline, width=stroke
            )
            shape = self.slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        if line:
            shape.line.color.rgb = rgb(line)
            shape.line.width = Pt(1.25)
        else:
            shape.line.fill.background()

    def text(
        self,
        content: str,
        x: float,
        y: float,
        w: float,
        h: float,
        points: int = 16,
        color: str = WHITE,
        bold: bool = False,
        mono: bool = False,
        align: str = "left",
        valign: str = "top",
        chrome: bool = False,
    ) -> None:
        content = "" if content is None else str(content)
        prefer_cjk = has_cjk(content)
        lines = wrap(content, points, bold, mono, w)
        line_gap = lh(points)
        block = line_gap * max(len(lines), 1)
        if block > h + 2:
            preview = content[:48]
            OVERFLOWS.append(
                f"slide {self.index}: overflow {points}pt h={h:.0f} need={block:.0f} :: {preview!r}"
            )
        if not chrome and y + h > 972:
            OVERFLOWS.append(f"slide {self.index}: below safe area y={y:.0f} :: {content[:40]!r}")
        top = y if valign == "top" else y + max(0, (h - block) / 2)
        align_map = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}
        for i, line in enumerate(lines):
            yy = top + i * line_gap
            font = face_for(line, points, bold, mono, prefer_cjk=prefer_cjk)
            width = font.getlength(line) if line else 0
            xx = x
            if align == "center":
                xx = x + (w - width) / 2
            elif align == "right":
                xx = x + w - width
            if line:
                self.draw.text((xx, yy), line, font=font, fill=f"#{color}", anchor="lt")
            box = self.slide.shapes.add_textbox(
                Emu(int(inches(x) * 914400)),
                Emu(int(inches(yy) * 914400)),
                Emu(int(inches(w) * 914400)),
                Emu(int(inches(line_gap) * 914400)),
            )
            frame = box.text_frame
            frame.word_wrap = False
            frame.auto_size = None
            frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = Emu(0)
            frame.paragraphs[0].alignment = align_map[align]
            frame.paragraphs[0].text = line if line else " "
            _set_run_font(
                frame.paragraphs[0].runs[0],
                typeface_name(line, mono, prefer_cjk=prefer_cjk),
                points,
                bold,
                color,
            )
            with contextlib.suppress(Exception):
                frame._txBody.bodyPr.set("anchor", MSO_ANCHOR.TOP)

    def chrome_bar(self) -> None:
        self.rect(88, 978, 1744, 2, CARD_EDGE)
        self.text(self.section, 88, 996, 980, lh(13), 13, DIM, chrome=True)
        self.text("learn-gpt2", 1100, 996, 732, lh(13), 13, DIM, align="right", chrome=True)
        self.text(
            f"{self.index:02d}  /  {self.total:02d}",
            1588,
            32,
            244,
            lh(13),
            13,
            DIM,
            align="right",
            chrome=True,
        )

    def kicker(self, label: str) -> None:
        self.text(label, MARGIN, 32, 1460, lh(13), 13, PURPLE, bold=True)

    def heading(self, title: str) -> None:
        points = 26 if len(title) > 20 else 30
        self.text(title, MARGIN, 70, 1560, lh(points), points, WHITE, bold=True)

    def lede(self, text: str, y: float = 168) -> float:
        self.text(text, MARGIN, y, 1744, lh(16, 2), 16, MUTED)
        return y + lh(16, 2) + 12

    def card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: str,
        kicker: str | None = None,
        accent: str = PURPLE,
    ) -> None:
        self.rect(x, y, w, h, CARD, radius=16, line=CARD_EDGE)
        self.rect(x, y + 12, 6, h - 24, accent)
        cursor = y + 16
        inner_w = w - 56
        if kicker:
            self.text(kicker, x + 28, cursor, inner_w, lh(12), 12, accent, bold=True)
            cursor += lh(12) + 2
        self.text(title, x + 28, cursor, inner_w, lh(18), 18, WHITE, bold=True)
        cursor += lh(18) + 8
        self.text(body, x + 28, cursor, inner_w, y + h - cursor - 14, 15, MUTED)

    def stat(
        self, x: float, y: float, w: float, h: float, value: str, label: str, accent: str = PURPLE
    ) -> None:
        self.rect(x, y, w, h, CARD, radius=16, line=CARD_EDGE)
        self.text(value, x + 22, y + 16, w - 40, lh(26), 26, accent, bold=True)
        self.text(label, x + 22, y + 16 + lh(26) + 6, w - 40, h - lh(26) - 36, 14, MUTED)

    def source(self, text: str) -> None:
        self.text(text, MARGIN, 888, 1744, lh(12, 2), 12, DIM)
