"""Build an editable PPTX, matching slide PNGs, PDF, and contact sheets.

Run with the independent media environment documented in README.md.
All figures are original teaching diagrams. Paper results are entered in slides.json.
"""

import json
import math
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
RENDER = BUILD / "render"
W, H, SCALE = 1920, 1080, 144
BG, PANEL, INK, MUTED = "0B1220", "142238", "ECF3FF", "A8B9D1"
TEAL, GOLD, PURPLE, RED = "5EEAD4", "FBBF65", "C4A1FF", "FC8797"
FONT_DIR = Path("/mnt/c/Windows/Fonts")
FONT = FONT_DIR / "msyh.ttc"
BOLD = FONT_DIR / "msyhbd.ttc"
FONT_NAME = "Microsoft YaHei"
SOURCES = [
    "https://arxiv.org/pdf/1706.03762",
    "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    "https://arxiv.org/pdf/2005.14165",
    "https://github.com/openai/gpt-2/blob/master/src/model.py",
]
SOURCE_PREFIXES = [("[1]", "Transformer"), ("[2]", "GPT-2"), ("[3]", "GPT-3"), ("[4]",)]
OVERFLOWS: list[str] = []


def color(hex_color: str) -> RGBColor:
    return RGBColor.from_string(hex_color)


@lru_cache(maxsize=128)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def safe_text(text: str) -> str:
    """Keep formula indices explicit in fonts lacking Unicode subscript letters."""
    text = str(text).replace("e⁰·⁷⁰⁷", "exp(0.707)").replace("e⁰", "exp(0)")
    sub = str.maketrans("₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₜₕ₊₋", "0123456789ijkth+-")
    sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹ᵀᴼ⁻", "0123456789TO-")
    text = re.sub(r"[₀₁₂₃₄₅₆₇₈₉ᵢⱼₖₜₕ₊₋]+", lambda m: "_(" + m[0].translate(sub) + ")", text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹ᵀᴼ⁻]+", lambda m: "^" + m[0].translate(sup), text)
    return text.replace("ℓ", "l")


def wrapped(text: str, width: float, size: int, bold: bool = False) -> list[str]:
    f = font(size, bold)
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for ch in paragraph:
            if current and f.getlength(current + ch) > width:
                lines.append(current)
                current = ch
            else:
                current += ch
        lines.append(current)
    return lines


class Board:
    """One set of primitives draws both editable Office objects and raster previews."""

    def __init__(self, prs: Presentation, index: int, data: dict) -> None:
        self.slide = prs.slides.add_slide(prs.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = color(BG)
        self.image = Image.new("RGB", (W, H), f"#{BG}")
        self.draw = ImageDraw.Draw(self.image)
        self.index = index
        self.data = data
        self.accent = (
            GOLD
            if data["section"].startswith(("02", "03"))
            else (PURPLE if data["section"].startswith("04") else TEAL)
        )
        self.source_url = next(
            (
                url
                for prefixes, url in zip(SOURCE_PREFIXES, SOURCES, strict=True)
                if data["source"].startswith(prefixes)
            ),
            None,
        )
        self.slide.notes_slide.notes_text_frame.text = (
            data["narration"] + "\n\n来源：" + data["source"] + "\n" + "\n".join(SOURCES)
        )

    def rect(self, x: float, y: float, w: float, h: float, fill: str = PANEL) -> None:
        self.draw.rectangle((x, y, x + w, y + h), fill=f"#{fill}")
        s = self.slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            Inches(x / SCALE),
            Inches(y / SCALE),
            Inches(w / SCALE),
            Inches(h / SCALE),
        )
        s.fill.solid()
        s.fill.fore_color.rgb = color(fill)
        s.line.fill.background()

    def line(
        self, x1: float, y1: float, x2: float, y2: float, fill: str = MUTED, width: int = 3
    ) -> None:
        from pptx.enum.shapes import MSO_CONNECTOR

        self.draw.line((x1, y1, x2, y2), fill=f"#{fill}", width=width)
        s = self.slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            Inches(x1 / SCALE),
            Inches(y1 / SCALE),
            Inches(x2 / SCALE),
            Inches(y2 / SCALE),
        )
        s.line.color.rgb = color(fill)
        s.line.width = Pt(width / 2)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, fill: str | None = None) -> None:
        c = fill or self.accent
        self.line(x1, y1, x2, y2, c, 4)
        a = math.atan2(y2 - y1, x2 - x1)
        for offset in [-0.5, 0.5]:
            self.line(x2, y2, x2 - 17 * math.cos(a + offset), y2 - 17 * math.sin(a + offset), c, 4)

    def text(
        self,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        size: int = 38,
        fill: str = INK,
        bold: bool = False,
        align: str = "left",
    ) -> None:
        from pptx.enum.text import PP_ALIGN

        text = safe_text(text)
        original = size
        while size > 20:
            lines = wrapped(str(text), w, size, bold)
            if len(lines) * size * 1.3 <= h:
                break
            size -= 1
        lines = wrapped(str(text), w, size, bold)
        if len(lines) * size * 1.3 > h + 2:
            OVERFLOWS.append(f"slide {self.index}, {text[:60]}, box {w}x{h}")
        if size < min(original - 8, 27):
            OVERFLOWS.append(f"small font on slide {self.index}: {size}, {text[:50]}")
        for n, line in enumerate(lines):
            yy = y + n * size * 1.3
            xx = x
            if align == "center":
                xx += (w - font(size, bold).getlength(line)) / 2
            self.draw.text((xx, yy), line, font=font(size, bold), fill=f"#{fill}", anchor="lt")
            box = self.slide.shapes.add_textbox(
                Inches(x / SCALE), Inches(yy / SCALE), Inches(w / SCALE), Inches(size * 1.5 / SCALE)
            )
            tf = box.text_frame
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            tf.word_wrap = False
            p = tf.paragraphs[0]
            p.text = line
            p.font.name, p.font.size, p.font.bold = FONT_NAME, Pt(size / 2), bold
            p.font.color.rgb = color(fill)
            if self.source_url and text == safe_text(self.data["source"]) and p.runs:
                p.runs[0].hyperlink.address = self.source_url
            p.alignment = PP_ALIGN.CENTER if align == "center" else PP_ALIGN.LEFT

    def box(
        self,
        label: str,
        x: float,
        y: float,
        w: float,
        h: float,
        accent: str | None = None,
        size: int = 36,
    ) -> None:
        self.rect(x, y, w, h)
        self.rect(x, y, 5, h, accent or self.accent)
        self.text(label, x + 22, y + 22, w - 44, h - 35, size)

    def bullets(
        self,
        items: list[str],
        x: int = 110,
        y: int = 680,
        w: int = 1690,
        step: int = 60,
        size: int = 32,
    ) -> None:
        for i, item in enumerate(items):
            self.rect(x, y + i * step + 12, 8, 8, self.accent)
            self.text(item, x + 27, y + i * step, w - 27, step - 4, size)

    def chrome(self, count: int) -> None:
        self.text(self.data["section"], 100, 48, 1300, 48, 27, self.accent, True)
        self.text(self.data["title"], 100, 116, 1740, 91, 62, INK, True)
        self.text(self.data["kicker"], 100, 213, 1740, 59, 34, MUTED)
        self.rect(100, 288, 1720, 2, "2B3B53")
        self.rect(100, 868, 5, 65, self.accent)
        self.text(self.data["note"], 122, 875, 1685, 67, 26, MUTED)
        self.text(self.data["source"], 100, 952, 1570, 35, 21, MUTED)
        self.text(f"{self.index:02d} / {count:02d}", 1690, 947, 130, 42, 25, self.accent)
        self.rect(0, 1048, W * self.index / count, 4, self.accent)


def generic_table(b: Board) -> None:
    d = b.data
    headers, rows = d["headers"], d["rows"]
    n = len(headers)
    widths = [1720 / n] * n
    if n == 2:
        widths = [700, 1020]
    row_h = 66 if len(rows) >= 4 else 76
    x0, y0 = 100, 320
    all_rows = [headers, *rows]
    for i, row in enumerate(all_rows):
        x = x0
        for j, cell in enumerate(row):
            b.rect(x, y0 + i * row_h, widths[j] - 3, row_h - 3, "223550" if i == 0 else PANEL)
            b.text(
                cell,
                x + 20,
                y0 + i * row_h + 15,
                widths[j] - 40,
                row_h - 19,
                30 if i == 0 else 32,
                b.accent if i == 0 else INK,
                i == 0,
            )
            x += widths[j]
    yy = max(650, y0 + len(all_rows) * row_h + 22)
    b.bullets(d["bullets"][:3], y=yy, step=max(43, (850 - yy) // 3), size=28)


def generic_flow(b: Board) -> None:
    labels = b.data.get("labels", b.data["bullets"])
    n = len(labels)
    width = (1720 - 45 * (n - 1)) / n
    for i, label in enumerate(labels):
        x = 100 + i * (width + 45)
        b.text(f"0{i + 1}", x + 15, 345, width - 30, 68, 47, b.accent, True)
        b.box(label, x, 432, width, 156, size=33)
        if i < n - 1:
            b.arrow(x + width + 8, 510, x + width + 37, 510)
    b.bullets(b.data["bullets"], y=654, step=60, size=33)


def generic_cards(b: Board) -> None:
    items = b.data["bullets"]
    if len(items) == 4:
        for i, item in enumerate(items):
            x, y = 100 + (i % 2) * 880, 325 + (i // 2) * 252
            b.rect(x, y, 840, 218)
            b.text(f"0{i + 1}", x + 30, y + 25, 120, 62, 44, b.accent, True)
            b.text(item, x + 30, y + 98, 780, 100, 36)
    else:
        for i, item in enumerate(items):
            y = 330 + i * 167
            b.rect(100, y, 1720, 140)
            b.text(f"0{i + 1}", 132, y + 34, 140, 70, 52, b.accent, True)
            b.text(item, 300, y + 36, 1470, 92, 41)


def generic_bars(b: Board) -> None:
    labels, values = b.data["labels"], b.data["values"]
    base_y, chart_x, chart_w = 743, 440, 1250
    for v in range(0, 101, 25):
        xx = chart_x + chart_w * v / 100
        b.line(xx, 340, xx, base_y, "2B3B53", 2)
        b.text(str(v), xx - 15, 755, 100, 40, 24, MUTED)
    for i, (label, val) in enumerate(zip(labels, values, strict=True)):
        yy = 340 + i * 100
        b.text(label, 100, yy + 13, 330, 67, 32)
        b.rect(chart_x, yy, chart_w * val / 100, 65, b.accent)
        b.text(
            f"{val:.1f}",
            min(chart_x + chart_w * val / 100 + 15, 1735),
            yy + 13,
            100,
            50,
            32,
            INK,
            True,
        )
    b.text(b.data["bullets"][0], 100, 811, 1720, 43, 28, MUTED)


def generic_prompt(b: Board) -> None:
    is_gpt3 = b.data["section"].endswith("GPT-3")
    prompts = (
        [
            ("零样本 · 0 条", "判断情感\n\n服务很周到 → ▯"),
            ("单样本 · 1 条", "判断情感\n糟透了 → 负面\n\n服务很周到 → ▯"),
            (
                "少样本 · 3 条",
                "判断情感\n糟透了 → 负面\n很喜欢 → 正面\n太失望 → 负面\n服务很周到 → ▯",
            ),
        ]
        if is_gpt3
        else [
            ("阅读理解", "文章 + 对话历史\n\nQ: 问题\nA: ▯"),
            ("摘要", "一段待概括的文章\n\nTL;DR: ▯"),
            ("翻译", "示例原文 = 译文\n示例原文 = 译文\n\n新的原文 = ▯"),
        ]
    )
    for i, (label, body) in enumerate(prompts):
        x = 100 + i * 585
        b.rect(x, 330, 550, 425)
        b.text(label, x + 30, 358, 490, 64, 34, b.accent, True)
        b.text(body, x + 30, 454, 490, 273, 37)
    b.text("▯ 是等待模型续写的答案；完整示例位于待测输入之前。", 100, 801, 1720, 55, 32, MUTED)


def matrix(b: Board, x: int, y: int, cell: int = 90, numbers: bool = False) -> None:
    tokens = ["我", "爱", "学", "习"]
    b.text("被读取的键 / 值位置 j →", x + cell, y - 70, cell * 5, 45, 28, MUTED)
    for j, token in enumerate(tokens):
        b.text(token, x + (j + 1) * cell, y - 13, cell - 4, 55, 32, MUTED, align="center")
    for i, token in enumerate(tokens):
        b.text(token, x, y + (i + 0.7) * cell, cell - 10, 55, 32, MUTED, align="center")
        for j in range(4):
            xx, yy = x + (j + 1) * cell, y + (i + 0.5) * cell
            b.rect(xx, yy, cell - 4, cell - 4, "1C4D50" if j <= i else "2F2439")
            label = ("0" if j <= i else "−∞") if numbers else ("可见" if j <= i else "屏蔽")
            b.text(
                label,
                xx + 4,
                yy + cell * 0.25,
                cell - 12,
                50,
                29,
                TEAL if j <= i else RED,
                align="center",
            )


def cached_row(b: Board) -> None:
    for i, (heading, expression) in enumerate(b.data["panels"]):
        x = 100 + 585 * i
        b.rect(x, 345, 550, 295)
        b.text(heading, x + 24, 392, 502, 65, 39, b.accent, True)
        b.text(expression, x + 24, 495, 502, 122, 34)
    b.text("新行的输出 ≈ (1.203, 1.593)", 100, 712, 1720, 90, 59, INK, True, "center")


def cached_mask(b: Board) -> None:
    cache_length = 2
    cell_width = 196
    left = 300
    b.text("历史 K/V", left, 322, 2 * cell_width - 5, 60, 35, GOLD, True, "center")
    b.text("本轮 K/V", left + 2 * cell_width, 322, 2 * cell_width - 5, 60, 35, TEAL, True, "center")
    for j in range(4):
        x = left + j * cell_width
        b.text(f"位置 {j}", x, 392, cell_width - 5, 52, 31, MUTED, align="center")
    for i in range(2):
        absolute_position = cache_length + i
        y = 453 + i * 132
        b.text(f"查询 {absolute_position}", 100, y + 40, 184, 50, 33, INK, True)
        for j in range(4):
            x = left + j * cell_width
            visible = j <= absolute_position
            b.rect(x, y, cell_width - 5, 123, "1C4D50" if visible else "2F2439")
            b.text(
                "可见" if visible else "屏蔽",
                x,
                y + 39,
                cell_width - 5,
                53,
                34,
                TEAL if visible else RED,
                True,
                "center",
            )
    b.box("可见条件\n键位置 j ≤ C + i", 1170, 381, 650, 183, size=42)
    b.box("本轮只有一个查询时\n它可以读取全部有效 K/V", 1170, 603, 650, 168, size=40)
    b.text("C = 2，本轮查询 i = 0、1，对应绝对位置 2、3。", 100, 799, 1720, 49, 32)


def special(b: Board) -> bool:
    k = b.data["kind"]
    if k == "cover":
        for i, (year, name, subtitle) in enumerate(
            [
                ("2017", "Transformer", "让位置之间交换信息"),
                ("2019", "GPT-2", "将任务写成文本前缀"),
                ("2020", "GPT-3", "规模与上下文示范"),
            ]
        ):
            x = 100 + i * 590
            b.rect(x, 345, 540, 390)
            b.text(year, x + 30, 377, 470, 90, 67, [TEAL, GOLD, PURPLE][i], True)
            b.text(name, x + 30, 497, 480, 75, 53, INK, True)
            b.text(subtitle, x + 30, 626, 480, 80, 34, MUTED)
        b.text("机制 → 架构 → 实验 · 含 KV Cache 手算和代码对照", 100, 798, 1720, 55, 34)
    elif k == "qkv":
        b.box("输入表示 X\n每个 token 一行", 100, 476, 430, 150, size=42)
        for i, (t, sub) in enumerate(
            [
                ("Q = XWQ", "我需要什么信息"),
                ("K = XWK", "怎样与查询匹配"),
                ("V = XWV", "提供什么内容"),
            ]
        ):
            yy = 331 + 164 * i
            b.arrow(550, 552, 750, yy + 65)
            b.box(t, 775, yy, 420, 130, size=44)
            b.text(sub, 1260, yy + 32, 530, 75, 38, MUTED)
    elif k == "formula":
        b.rect(100, 330, 1720, 180)
        b.text(
            "Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V",
            130,
            385,
            1660,
            88,
            54,
            b.accent,
            True,
            "center",
        )
        b.bullets(b.data["bullets"], y=566, step=67, size=37)
    elif k == "calc":
        for i, (head, body) in enumerate(
            [
                ("1  点积与缩放", "[1, 0] / √2\n≈ [0.707, 0]"),
                ("2  行 softmax", "[e⁰·⁷⁰⁷, e⁰] / 总和\n≈ [0.670, 0.330]"),
                ("3  加权汇总", "0.670 × [2, 0]\n+ 0.330 × [0, 4]"),
            ]
        ):
            x = 100 + i * 585
            b.rect(x, 345, 550, 295)
            b.text(head, x + 25, 377, 500, 70, 37, b.accent, True)
            b.text(body, x + 25, 490, 500, 120, 38)
        b.text("输出 ≈ [1.340, 1.321]", 100, 711, 1720, 90, 59, INK, True, "center")
    elif k == "causal_intro":
        labels = ["我", "爱", "学", "习"]
        for i, token in enumerate(labels):
            x = 160 + i * 420
            b.box(token, x, 372, 240, 140, accent=TEAL if i < 2 else RED, size=60)
            b.text(
                "已知前缀" if i < 2 else "未来，不可读取",
                x - 30,
                538,
                335,
                60,
                32,
                TEAL if i < 2 else RED,
                align="center",
            )
        b.arrow(700, 632, 700, 715)
        b.box("“爱”位置的输出 → 预测下一个 token “学”", 320, 732, 1280, 95, size=38)
        b.text("当前查询位置 i = 2", 500, 615, 620, 60, 33, b.accent)
    elif k == "mask":
        matrix(b, 125, 410, 88, True)
        b.text("查询 i ↓", 102, 306, 340, 52, 28, MUTED)
        b.text("Sᵢⱼ = qᵢ · kⱼ / √dₖ + Mᵢⱼ", 840, 350, 960, 92, 45, b.accent, True)
        b.box("j ≤ i：Mᵢⱼ = 0\nj > i：Mᵢⱼ = −∞", 840, 470, 960, 180, size=43)
        b.text("A = softmax(S)\n每行在可见位置上重新分配概率", 840, 708, 960, 116, 37)
    elif k == "mask_softmax":
        for i, (a, val) in enumerate(
            [
                ("原始分数", "[ 2, 1, 3, 0 ]"),
                ("加因果掩码", "[ 2, 1, −∞, −∞ ]"),
                ("行 softmax", "[ 0.731, 0.269, 0, 0 ]"),
            ]
        ):
            yy = 338 + i * 159
            b.text(a, 120, yy + 28, 410, 70, 41, b.accent, True)
            b.box(val, 570, yy, 1210, 124, size=50)
    elif k == "alignment":
        for row, (label, tokens) in enumerate(
            [("输入 xᵢ", ["〈起始〉", "我", "爱", "学"]), ("标签 xᵢ₊₁", ["我", "爱", "学", "习"])]
        ):
            yy = 370 + row * 290
            b.text(label, 100, yy + 30, 280, 70, 39, b.accent, True)
            for j, t in enumerate(tokens):
                xx = 410 + j * 350
                b.box(t, xx, yy, 290, 130, size=51)
                if row == 0:
                    b.arrow(xx + 145, yy + 145, xx + 145, yy + 265)
                    b.text("预测", xx + 80, yy + 182, 160, 60, 31, MUTED)
    elif k == "train":
        matrix(b, 90, 397, 86)
        b.arrow(600, 584, 775, 584)
        for i in range(4):
            b.box(
                f"位置 {i + 1} 的预测分布 → 交叉熵 L{i + 1}", 820, 330 + 118 * i, 960, 92, size=35
            )
        b.text("整段前向 + 损失汇总 → 一次反向传播", 750, 815, 1050, 45, 31, b.accent)
    elif k == "causalproof":
        for i, t in enumerate(
            ["第 ℓ−1 层位置 j\n依赖 x≤j", "仅读取 j ≤ i\n的键与值", "第 ℓ 层位置 i\n仍只依赖 x≤i"]
        ):
            x = 100 + 590 * i
            b.box(t, x, 380, 540, 210, size=43)
            if i < 2:
                b.arrow(x + 550, 485, x + 580, 485)
        b.box(
            "逐位置 FFN + LayerNorm + 残差\n没有跨位置的额外信息通路", 270, 670, 1380, 155, size=40
        )
    elif k == "decode":
        for i, (ctx, nxt) in enumerate([("我 爱", "学"), ("我 爱 学", "习"), ("我 爱 学 习", "…")]):
            yy = 335 + i * 166
            b.text(f"第 {i + 1} 步", 100, yy + 27, 270, 70, 39, b.accent, True)
            b.box(ctx, 390, yy, 620, 126, size=46)
            b.arrow(1040, yy + 60, 1190, yy + 60)
            b.box(f"选出 {nxt}", 1240, yy, 530, 126, size=44)
    elif k == "kv":
        b.box("本层已缓存\n历史 K/V", 100, 350, 805, 150, size=42)
        b.box("本层的新位置\n计算本轮的 Q、K、V", 1015, 350, 805, 150, size=42)
        b.arrow(500, 516, 720, 587)
        b.arrow(1420, 516, 1200, 587)
        b.box("本轮 Q 读取历史与当前的 K/V", 370, 606, 1180, 115, size=43)
        b.text("将本轮 K/V 写入该层缓存，供下一轮读取。", 100, 775, 1720, 54, 35, MUTED)
    elif k == "kv_row":
        cached_row(b)
    elif k == "kv_mask":
        cached_mask(b)
    elif k == "multihead":
        for i in range(4):
            x = 100 + i * 440
            b.box(f"Head {i + 1}\n独立 Q/K/V\n因果 attention", x, 340, 400, 230, size=37)
            b.arrow(x + 200, 585, 960, 662)
        b.box("Concat(head₁, …, headₕ) → Wᴼ → 输出", 240, 700, 1440, 117, size=46)
    elif k == "position":
        b.box("内容向量\nToken embedding", 130, 365, 650, 180, size=46)
        b.text("+", 870, 405, 180, 100, 65, b.accent, True, "center")
        b.box("位置向量\nPosition encoding", 1140, 365, 650, 180, size=46)
        b.box(
            "2017：sin / cos，不同频率的固定编码\nGPT-2：从位置表中读取可学习向量",
            200,
            645,
            1520,
            180,
            size=42,
        )
    elif k == "arch":
        for x, label, parts in [
            (100, "ENCODER × 6", ["源句 + 位置", "双向自注意力", "逐位置 FFN"]),
            (1050, "DECODER × 6", ["目标前缀 + 位置", "因果自注意力", "交叉注意力", "逐位置 FFN"]),
        ]:
            b.text(label, x, 320, 760, 60, 36, b.accent, True)
            for i, part in enumerate(parts):
                b.box(part, x, 400 + i * 100, 760, 80, size=32)
        b.arrow(870, 600, 1030, 640)
        b.text("K / V", 880, 528, 150, 50, 27, TEAL)
        b.text("Q 来自目标", 1080, 814, 700, 44, 28, MUTED)
        b.text("每个子层均配合残差与归一化", 110, 766, 800, 65, 32, MUTED)
    elif k == "residual":
        for i, (name, expr, extra) in enumerate(
            [
                ("2017 · Post-LN", "LN(x + Sublayer(x))", "先子层 + 残差，再归一化"),
                ("GPT-2 · Pre-LN", "x + Sublayer(LN(x))", "先归一化，再进入子层分支"),
            ]
        ):
            x = 100 + i * 880
            b.rect(x, 355, 840, 300)
            b.text(name, x + 30, 386, 780, 60, 39, b.accent, True)
            b.text(expr, x + 30, 481, 780, 75, 43, INK, True)
            b.text(extra, x + 30, 578, 780, 59, 33, MUTED)
        b.text("FFN：维度扩展 → 非线性 → 维度还原（每个位置独立处理）", 100, 737, 1720, 85, 39)
    else:
        return False
    return True


def main() -> None:
    RENDER.mkdir(parents=True, exist_ok=True)
    data = json.loads((HERE / "slides.json").read_text())
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W / SCALE), Inches(H / SCALE)
    prs.core_properties.title = "注意力机制、GPT-2 与 GPT-3 三篇论文综述"
    prs.core_properties.subject = "因果自注意力、KV Cache、模型架构、训练与评测"
    prs.core_properties.author = "learn-gpt2"
    images = []
    for index, slide in enumerate(data, 1):
        b = Board(prs, index, slide)
        b.chrome(len(data))
        if not special(b):
            kind = slide["kind"]
            {
                "table": generic_table,
                "flow": generic_flow,
                "cards": generic_cards,
                "bars": generic_bars,
                "prompt": generic_prompt,
            }[kind](b)
        b.image.save(RENDER / f"slide-{index:02d}.png")
        images.append(b.image)
    prs.save(HERE / "overview.pptx")
    images[0].save(
        HERE / "overview.pdf", "PDF", resolution=144, save_all=True, append_images=images[1:]
    )
    for batch in range(math.ceil(len(images) / 12)):
        sheet = Image.new("RGB", (1920, 1080), "#24354C")
        draw = ImageDraw.Draw(sheet)
        for j, img in enumerate(images[batch * 12 : (batch + 1) * 12]):
            x, y = (j % 4) * 480, (j // 4) * 360
            sheet.paste(img.resize((470, 264)), (x, y + 38))
            draw.text((x + 12, y + 7), f"{batch * 12 + j + 1:02d}", font=font(23), fill="white")
        sheet.save(BUILD / f"contact-{batch + 1:02d}.jpg", quality=90)
    images[7].save(HERE / "causal_attention_preview.png")
    (BUILD / "layout_check.json").write_text(
        json.dumps(
            {
                "slides": len(data),
                "overflow_errors": OVERFLOWS,
                "render_size": [W, H],
                "editable_text_and_shapes": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    if OVERFLOWS:
        raise RuntimeError("\n".join(OVERFLOWS))
    print(f"Generated {len(data)} slides, editable PPTX, PDF, PNGs; no text overflow.")


if __name__ == "__main__":
    main()
