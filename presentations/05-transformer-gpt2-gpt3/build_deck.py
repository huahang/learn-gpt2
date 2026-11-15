# ruff: noqa: E501
"""Build the lecture deck: editable PPTX plus 1920x1080 PNGs.

Slide text is the picture. Spoken detail lives in the narration, which is also
the PPT speaker notes. Numbers for the mask and the cache come from demo.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import demo
import numpy as np
from engine import (
    BLOCKED,
    BLUE,
    BUILD,
    CARD,
    DEEP,
    DIM,
    LAVENDER,
    MARGIN,
    MUTED,
    OVERFLOWS,
    PNG_DIR,
    PURPLE,
    VIOLET,
    WHITE,
    Board,
    lh,
    mix,
    new_presentation,
)
from PIL import Image

SLIDES: list[dict] = []


def slide(section: str, narration: str):
    text = " ".join(narration.split())

    def decorate(fn):
        SLIDES.append({"section": section, "narration": text, "draw": fn})
        return fn

    return decorate


def boxes(n: int, y: float, height: float, cols: int, gap: float = 18) -> list[tuple]:
    rows = math.ceil(n / cols)
    width = 1744
    cell_w = (width - gap * (cols - 1)) / cols
    cell_h = (height - gap * (rows - 1)) / rows
    found = []
    for i in range(n):
        row, col = divmod(i, cols)
        found.append((MARGIN + col * (cell_w + gap), y + row * (cell_h + gap), cell_w, cell_h))
    return found


def table(board: Board, x, y, w, headers, rows, weights, points: int = 14) -> float:
    row_h = lh(points) + 18
    widths = [w * weight / sum(weights) for weight in weights]
    board.rect(x, y, w, row_h, DEEP, radius=10)
    cursor = x
    for index, header in enumerate(headers):
        board.text(
            header, cursor + 12, y + 8, widths[index] - 20, lh(points), points, WHITE, bold=True
        )
        cursor += widths[index]
    for r, row in enumerate(rows):
        top = y + row_h * (r + 1)
        if r % 2 == 0:
            board.rect(x, top, w, row_h, "100A22")
        cursor = x
        for index, cell in enumerate(row):
            color = WHITE if index == 0 else LAVENDER
            board.text(
                str(cell),
                cursor + 12,
                top + 8,
                widths[index] - 20,
                lh(points),
                points,
                color,
                bold=index == 0,
            )
            cursor += widths[index]
    return y + row_h * (len(rows) + 1)


def weight_matrix(
    board: Board, x: float, y: float, tokens: list[str], weights: np.ndarray, cell: int = 92
) -> None:
    board.text("Q \\ K", x, y, cell, lh(12), 12, DIM, align="center")
    for col, token in enumerate(tokens):
        board.text(
            token, x + (col + 1) * cell, y, cell, lh(14), 14, MUTED, align="center", bold=True
        )
    for row, token in enumerate(tokens):
        top = y + lh(14) + 8 + row * cell
        board.text(token, x, top, cell, cell, 14, MUTED, align="center", valign="middle", bold=True)
        for col in range(len(tokens)):
            value = float(weights[row, col])
            left = x + (col + 1) * cell + 6
            fill = BLOCKED if value < 0.005 else mix("241448", "916EF0", value)
            color = DIM if value < 0.005 else WHITE
            label = "0" if value < 0.005 else f"{value:.2f}"
            board.rect(left, top + 6, cell - 12, cell - 12, fill, radius=10)
            board.text(
                label,
                left,
                top + 6,
                cell - 12,
                cell - 12,
                14,
                color,
                bold=True,
                align="center",
                valign="middle",
                mono=True,
            )


def pattern_matrix(board: Board, x: float, y: float, tokens: list[str], cell: int = 92) -> None:
    board.text("", x, y, cell, lh(12), 12, DIM, align="center")
    for col, token in enumerate(tokens):
        board.text(
            token, x + (col + 1) * cell, y, cell, lh(14), 14, MUTED, align="center", bold=True
        )
    for row, token in enumerate(tokens):
        top = y + lh(14) + 8 + row * cell
        board.text(token, x, top, cell, cell, 14, MUTED, align="center", valign="middle", bold=True)
        for col in range(len(tokens)):
            allowed = col <= row
            left = x + (col + 1) * cell + 6
            board.rect(
                left, top + 6, cell - 12, cell - 12, PURPLE if allowed else BLOCKED, radius=10
            )
            board.text(
                "看" if allowed else "∞",
                left,
                top + 6,
                cell - 12,
                cell - 12,
                14,
                WHITE if allowed else DIM,
                bold=True,
                align="center",
                valign="middle",
            )


def mini_mask(board: Board, x, y, rows: int, cols: int, see, cell: int = 22) -> None:
    gap = 5
    for i in range(rows):
        for j in range(cols):
            board.rect(
                x + j * (cell + gap),
                y + i * (cell + gap),
                cell,
                cell,
                PURPLE if see(i, j) else BLOCKED,
                radius=4,
            )


@slide(
    "开场",
    "大家好。这一讲把三篇改变语言模型路线的论文串起来：2017 年的 Attention Is All You Need，"
    "2019 年的 GPT-2，和 2020 年的 GPT-3。我们会先看 Transformer 为什么出现，再重点把因果自注意力"
    "和 KV Cache 从直觉讲到公式和代码，最后看 GPT-2 的零样本多任务，以及 GPT-3 怎样把规模变成"
    "上下文里的学习能力。",
)
def cover(board: Board) -> None:
    board.text("三篇论文 · 一条主线", 88, 150, 980, lh(16), 16, PURPLE, bold=True)
    board.text("从注意力到\n大语言模型", 88, 200, 1020, lh(44, 2), 44, WHITE, bold=True)
    board.text(
        "Transformer、因果自注意力、KV Cache\n以及 GPT-2 与 GPT-3",
        88,
        210 + lh(44, 2),
        980,
        lh(18, 2),
        18,
        MUTED,
    )
    board.text(
        "面向想入门 LLM 的读者：先建立直觉，再看公式、掩码和本仓库的 gpt2.py。",
        88,
        760,
        980,
        lh(15, 2),
        15,
        LAVENDER,
    )
    pattern_matrix(board, 1120, 210, demo.TOKENS, cell=78)


@slide(
    "开场",
    "全篇分成五段。第一段看注意力要解决什么问题。第二段看 Transformer 的结构和实验。"
    "第三段是重点：因果自注意力，包括掩码矩阵，以及仓库里 gpt2.py 的写法。第四段讲 KV Cache，"
    "为什么能省计算，缓存长什么样，显存和复杂度怎么算。第五段分别进入 GPT-2 和 GPT-3，"
    "看这三篇论文怎样连成一条线。",
)
def roadmap(board: Board) -> None:
    board.kicker("开场")
    board.heading("这一讲怎么走")
    items = [
        ("01", "下一个词", "自回归、错开一位的标签，以及 RNN 为什么不够。", PURPLE),
        ("02", "Transformer", "缩放点积、多头、三种注意力，和翻译实验。", VIOLET),
        ("03", "因果自注意力", "下三角掩码、softmax，以及 gpt2.py 的三种分支。", PURPLE),
        ("04", "KV Cache", "预填充、逐步解码、显存，和有缓存、没缓存的复杂度。", BLUE),
        ("05", "GPT-2 与 GPT-3", "零样本多任务、规模化，以及 few-shot 的能力和边界。", VIOLET),
    ]
    top = 175
    for index, (num, title, body, accent) in enumerate(items):
        y = top + index * 148
        board.rect(88, y, 1744, 136, CARD, radius=16, line="2E2358")
        board.rect(112, y + 36, 72, 64, accent, radius=12)
        board.text(num, 112, y + 36, 72, 64, 18, WHITE, bold=True, align="center", valign="middle")
        board.text(title, 210, y + 28, 1500, lh(20), 20, WHITE, bold=True)
        board.text(body, 210, y + 34 + lh(20), 1560, lh(15), 15, MUTED)


@slide(
    "问题",
    "先把问题说清楚。语言模型做的事情，是根据已经看到的词，预测下一个词。"
    "这句话听起来简单。后面的掩码、缓存，还有 GPT 的训练目标，全都是从这里长出来的。",
)
def section_problem(board: Board) -> None:
    board.text("01", 88, 250, 500, lh(64), 64, PURPLE, bold=True)
    board.text("先预测下一个词", 88, 250 + lh(64) + 8, 1600, lh(36), 36, WHITE, bold=True)
    board.text(
        "注意力、掩码和缓存，都是为了把这件事做得又对、又快。",
        88,
        250 + lh(64) + lh(36) + 24,
        1500,
        lh(18),
        18,
        MUTED,
    )


@slide(
    "问题",
    "拿“今天天气真好”做例子。模型看到“今天”，要为“天气”这个位置出力；看到“今天天气”，要为“真”出力。"
    "训练的时候，整句可以一次送进去，但每个位置的标签是它右边那个词，输入和标签错开一位。"
    "生成的时候后面的词还不存在，只能把刚预测出来的词接回去，再预测下一个。",
)
def next_token(board: Board) -> None:
    board.kicker("01  问题")
    board.heading("输入和标签错开一位")
    tokens = ["今天", "天气", "真", "好"]
    labels = ["天气", "真", "好", "结束"]
    for index, token in enumerate(tokens):
        left = 88 + index * 250
        board.rect(left, 230, 220, 90, CARD, radius=14, line="2E2358")
        board.text(token, left, 230, 220, 90, 22, WHITE, bold=True, align="center", valign="middle")
        board.text("预测", left, 340, 220, lh(13), 13, DIM, align="center")
        board.rect(left, 390, 220, 90, "241448", radius=14)
        board.text(
            labels[index],
            left,
            390,
            220,
            90,
            22,
            PURPLE,
            bold=True,
            align="center",
            valign="middle",
        )
    pairs = boxes(2, 540, 340, 2)
    board.card(
        *pairs[0],
        "训练",
        "整句同时送入，这叫 teacher forcing。每个位置只根据自己和左边的词，预测右边那一个。计算是并行的，依赖仍然是从左到右。",
        "一次前向",
    )
    board.card(
        *pairs[1],
        "生成",
        "未来真的还没写出来。模型输出一个词，把它追加到末尾，再跑下一步。所以生成是自回归的，一步一个 token。",
        "逐步追加",
        BLUE,
    )


@slide(
    "问题",
    "循环网络必须从左到右一步步走。第 n 个词要想用到句首的信息，路径长度就是 n，而且没法按位置并行。"
    "卷积可以并行，但一层只能看一小段，看远了就要堆很多层。自注意力让任意两个位置直接相连，路径长度是常数，"
    "代价是计算量和序列长度的平方成正比。这就是 Transformer 论文表 1 要说的那笔账。",
)
def why_attention(board: Board) -> None:
    board.kicker("01  问题")
    board.heading("为什么要换成注意力")
    board.lede(
        "论文表 1：每层的计算、必须串行的步数，以及信息走完序列的最长路径。n 是长度，d 是宽度。"
    )
    headers = ["层", "每层计算", "串行步数", "最长路径"]
    rows = [
        ["自注意力", "O(n²·d)", "O(1)", "O(1)"],
        ["循环网络", "O(n·d²)", "O(n)", "O(n)"],
        ["卷积", "O(k·n·d^2)", "O(1)", "O(log_k n)"],
    ]
    table(board, 88, 280, 1744, headers, rows, [1.2, 1.3, 1, 1], points=18)
    board.text(
        "n 不太大时，自注意力用平方的计算量，换来全句一步可达、而且可以并行。后面的 KV Cache，就是在生成阶段把这份平方的账再算细一点。",
        88,
        620,
        1744,
        lh(16, 2),
        16,
        MUTED,
    )
    board.source(
        "Vaswani et al., 2017，表 1。受限自注意力把路径放宽到 O(n/r)，论文留作以后的工作。"
    )


@slide(
    "Transformer",
    "2017 年，Vaswani 等人提出 Transformer。机器翻译的主干不再是循环网络。编码器和解码器都只剩下注意力和前馈网络。"
    "论文标题把结论说得很绝对：Attention Is All You Need，注意力就够了。",
)
def section_transformer(board: Board) -> None:
    board.text("02", 88, 250, 500, lh(64), 64, PURPLE, bold=True)
    board.text("Transformer", 88, 250 + lh(64) + 8, 1600, lh(36), 36, WHITE, bold=True)
    board.text(
        "Attention Is All You Need · Vaswani et al., 2017",
        88,
        250 + lh(64) + lh(36) + 24,
        1500,
        lh(18),
        18,
        MUTED,
    )


@slide(
    "Transformer",
    "可以把注意力想成一次软的查表。每个位置准备三份向量。Query 表示我在找什么，Key 表示我拿什么被匹配，"
    "Value 才是我真正交出去的内容。相似度是 Q 和 K 的点积，先除以根号 d_k，再做 softmax，变成一行加起来等于 1 的权重，"
    "最后对 V 求加权和。除以根号 d_k，是为了不让点积随着维度变大，把 softmax 推进几乎没有梯度的区域。",
)
def scaled_dot(board: Board) -> None:
    board.kicker("02  Transformer")
    board.heading("缩放点积注意力")
    board.rect(88, 180, 1744, 150, CARD, radius=16, line="2E2358")
    board.text(
        "softmax( Q K^T / sqrt(d_k) ) V",
        88,
        180,
        1744,
        150,
        28,
        WHITE,
        bold=True,
        align="center",
        valign="middle",
        mono=True,
    )
    found = boxes(3, 360, 300, 3)
    board.card(
        *found[0], "Query", "当前这个位置想查找的内容。它决定分数矩阵的每一行。", "我在找什么"
    )
    board.card(
        *found[1],
        "Key",
        "每个位置用来被匹配的索引。它决定分数矩阵的每一列。",
        "拿什么被匹配",
        VIOLET,
    )
    board.card(
        *found[2],
        "Value",
        "真正被加权平均的内容。权重和为 1，所以输出还在这些向量张成的范围里。",
        "交出的内容",
        BLUE,
    )
    board.source("Vaswani et al., 2017，3.2.1。点积不除以根号 d_k 时，量级大约随 √d_k 增长。")


@slide(
    "Transformer",
    "一组 Q、K、V 只能学到一种匹配方式。多头把向量切成 h 份，每份在自己的小空间里做注意力，再拼回去，过一次线性层。"
    "Base 模型的宽度是 512，8 个头，每个头 64 维。消融里，只用 1 个头，比最好的设置大约差 0.9 个 BLEU。"
    "头太多、每一头太窄，效果也会掉。论文的开发集是 WMT 英德 newstest2013。",
)
def multi_head(board: Board) -> None:
    board.kicker("02  Transformer")
    board.heading("多头：同一次计算，多看几种关系")
    steps = [
        ("输入", "d = 512"),
        ("切成 8 头", "每头 64 维"),
        ("各自做注意力", "互不混合"),
        ("拼回 + 线性层", "再投影到 512"),
    ]
    for index, (title, body) in enumerate(steps):
        left = 88 + index * 450
        board.rect(left, 200, 400, 160, CARD, radius=16, line="2E2358")
        board.text(f"0{index + 1}", left + 24, 218, 340, lh(13), 13, PURPLE, bold=True)
        board.text(title, left + 24, 250, 350, lh(20), 20, WHITE, bold=True)
        board.text(body, left + 24, 250 + lh(20) + 6, 350, lh(15), 15, MUTED)
        if index < 3:
            board.text("→", left + 400, 250, 50, lh(20), 20, PURPLE, align="center")
    found = boxes(2, 410, 460, 2)
    board.card(
        *found[0],
        "Base 的尺寸",
        "6 层，d_model 512，前馈层 2048，8 个头，d_k = d_v = 64，dropout 0.1。参数量大约 6500 万。",
        "表 3",
    )
    board.card(
        *found[1],
        "头数不是越多越好",
        "表 3 的 A 组把总计算量保持不变，只改头数。1 个头比最好设置低大约 0.9 BLEU；32 个头、每头 16 维也更差。",
        "消融",
        VIOLET,
    )
    board.source("Vaswani et al., 2017，3.2.2 与表 3。指标是英德 newstest2013 的开发集 BLEU。")


@slide(
    "Transformer",
    "整张网络里其实有三种注意力。编码器的自注意力可以看完整句输入。解码器里，一种去看编码器的输出，翻译的时候这就是在读原文；"
    "另一种是带掩码的自注意力，生成第 i 个词的时候不能看第 i 个之后的词。每个子层外面都是残差加层归一化，然后再接一个逐位置的前馈网络。"
    "原始论文的前馈网络用 ReLU，中间宽度是模型宽度的 4 倍。",
)
def three_attns(board: Board) -> None:
    board.kicker("02  Transformer")
    board.heading("三种注意力，各看各的")
    found = boxes(3, 190, 520, 3)
    board.card(
        *found[0],
        "编码器自注意力",
        "源句子里的每个位置都可以看整个源句。翻译里，它负责把原文自己先读通。",
        "看全部输入",
    )
    board.card(
        *found[1],
        "解码器看编码器",
        "Query 来自解码器，Key 和 Value 来自编码器。生成译文时，每个词都能回头查阅原文。",
        "交叉注意力",
        VIOLET,
    )
    board.card(
        *found[2],
        "解码器自注意力",
        "必须遮住未来。再叠加“输出嵌入比标签错开一位”，位置 i 的预测就只依赖 i 之前已经知道的词。",
        "因果掩码",
        BLUE,
    )
    board.source(
        "Vaswani et al., 2017，3.1 与 3.2.3。掩码加在 softmax 之前，被挡住的位置不会分到权重。"
    )


@slide(
    "Transformer",
    "注意力本身不分先后，所以要另加位置编码。论文用一组正弦和余弦，也试过可学习的位置嵌入，开发集上几乎一样。"
    "训练用 Adam。学习率先用 4000 步线性升温，再按步数平方根的倒数下降。标签平滑 0.1，dropout 0.1。"
    "Base 模型大约 6500 万参数，8 张 P100 上训练十万步，论文说大约 12 小时。Big 模型 2.13 亿参数，训练了 3.5 天。",
)
def recipe(board: Board) -> None:
    board.kicker("02  Transformer")
    board.heading("位置、前馈网络和训练配方")
    found = boxes(4, 190, 680, 2)
    cards = [
        (
            "位置编码",
            "正弦余弦",
            "和词向量相加。表 3 的 E 行换成可学习位置嵌入，开发集 BLEU 从 25.8 变成 25.7，几乎一样。",
            PURPLE,
        ),
        (
            "前馈网络",
            "逐位置的两层",
            "FFN(x) = max(0, xW1+b1) W2+b2。Base 的中间宽度是 2048，等于 4 乘 512。层与层不共享参数。",
            VIOLET,
        ),
        (
            "优化",
            "Adam + warmup",
            "warmup 4000 步。公式是 d 的 -0.5 次方，再取步数平方根倒数和线性升温里较小的那个。",
            BLUE,
        ),
        (
            "正则",
            "dropout 与平滑",
            "子层输出、嵌入和位置编码之和都做 dropout。标签平滑 ε = 0.1，困惑度会变差，BLEU 会变好。",
            PURPLE,
        ),
    ]
    for box, (title, kicker, body, accent) in zip(found, cards, strict=True):
        board.card(*box, title, body, kicker, accent)
    board.source("Vaswani et al., 2017，3.3 至 5.4，以及表 3。")


@slide(
    "Transformer",
    "实验任务是 WMT 2014 的新闻翻译。英译德，big 模型拿到 28.4 BLEU，比当时包括模型集成在内的最好结果还高出两个 BLEU 以上。"
    "英译法要小心引用。摘要和表 2 写的是 41.8，正文 6.1 节写的是 41.0。两个数字都在同一份 PDF 里，引用时要说清楚位置。"
    "Base 模型更便宜：英德 27.3，英法 38.1，训练成本远低于之前的竞争模型。",
)
def mt_results(board: Board) -> None:
    board.kicker("02  Transformer")
    board.heading("翻译结果：新的单模型，更少的训练")
    headers = ["模型", "英→德", "英→法", "训练"]
    rows = [
        ["Transformer Base", "27.3", "38.1", "8×P100，约 12 小时"],
        ["Transformer Big", "28.4", "41.8 / 41.0", "8×P100，3.5 天"],
        ["此前集成模型", "26.36", "41.29", "成本高一个数量级以上"],
    ]
    table(board, 88, 190, 1744, headers, rows, [1.5, 1, 1.3, 1.8], points=16)
    board.text(
        "测试集是 newstest2014。Big 的英德结果超过此前最好集成两个 BLEU 以上。英法的 41.8 来自摘要和表 2，41.0 来自正文 6.1 节。Big 做英法时 dropout 用了 0.1，而不是表 3 里的 0.3。解码用束搜索，束宽 4，长度惩罚 0.6，并对最后若干个检查点做平均。",
        88,
        520,
        1744,
        lh(16, 4),
        16,
        MUTED,
    )
    board.source("Vaswani et al., 2017，摘要、表 2、第 6.1 节。ConvS2S 集成的英德分数是 26.36。")


@slide(
    "因果自注意力",
    "接下来是这一讲的核心。解码器为什么必须把未来遮住？遮住之后，矩阵是什么形状？softmax 之后权重怎么变？"
    "仓库里的 CausalSelfAttention 又是怎么写的？我们用四个词的例子，把这件事一次算清楚。",
)
def section_causal(board: Board) -> None:
    board.text("03", 88, 250, 500, lh(64), 64, PURPLE, bold=True)
    board.text("因果自注意力", 88, 250 + lh(64) + 8, 1600, lh(36), 36, WHITE, bold=True)
    board.text(
        "每个位置只能看自己和过去。对角线要留下。",
        88,
        250 + lh(64) + lh(36) + 24,
        1500,
        lh(18),
        18,
        MUTED,
    )


@slide(
    "因果自注意力",
    "遮住未来，有三件事情其实是同一条约束。第一，生成是从左到右的，写到第 t 个词时，后面的词还不存在。"
    "第二，训练时虽然整句都在输入里，可一旦允许看未来，模型会直接抄到标签，损失变成零，什么也学不到。"
    "第三，只要每个位置都不依赖未来，旧位置的 Key 和 Value 就不会因为新词而改变。后面的 KV Cache 才站得住。",
)
def three_reasons(board: Board) -> None:
    board.kicker("03  因果自注意力")
    board.heading("为什么不能看未来")
    found = boxes(3, 190, 680, 3)
    board.card(
        *found[0],
        "生成时未来不存在",
        "写到当前位置时，右边还是空的。模型如果在训练里偷看过右边，生成时就没有东西可看，行为会对不上。",
        "1",
    )
    board.card(
        *found[1],
        "训练时不能抄答案",
        "标签就是右边那个词。掩码让每个位置只能用左边和自己的表示，去预测下一个词。",
        "2",
        VIOLET,
    )
    board.card(
        *found[2],
        "旧的 K、V 才能缓存",
        "新词不影响旧位置的键和值。所以预填充时存下来的缓存，解码时仍然有效。",
        "3",
        BLUE,
    )


@slide(
    "因果自注意力",
    "看一个长度为 4 的例子，四个词是今天、天气、真、好。分数矩阵是 4 乘 4。第 i 行是第 i 个 Query，第 j 列是第 j 个 Key。"
    "因果掩码是包含对角线的下三角：列号大于行号的地方，在 softmax 之前填上负无穷。对角线要保留。"
    "当前位置可以读自己的表示，因为它要预测的是下一个词，不是把它自己遮掉。",
)
def mask_shape(board: Board) -> None:
    board.kicker("03  因果自注意力")
    board.heading("掩码是包含对角线的下三角")
    board.text(
        "行是 Query，列是 Key。紫色表示这个位置允许看，包括自己。深色在进 softmax 之前是负无穷。",
        88,
        165,
        760,
        lh(16, 3),
        16,
        MUTED,
    )
    board.rect(88, 340, 760, 250, CARD, radius=16, line="2E2358")
    board.text("允许：j ≤ i，含对角线", 116, 362, 680, lh(16), 16, WHITE, bold=True)
    board.text("位置 i 读这些 Value，\n用来预测下一个词。", 116, 410, 680, lh(16, 2), 16, MUTED)
    board.text("禁止：j > i，标成 ∞", 116, 520, 680, lh(16), 16, WHITE, bold=True)
    board.text("生成时这些词还不存在，\n训练时它们就是答案。", 116, 568, 680, lh(16, 2), 16, MUTED)
    pattern_matrix(board, 980, 175, demo.TOKENS, cell=100)
    board.source("形状是 T×T。本页四个词是教学例子，不是论文里的句子。")


@slide(
    "因果自注意力",
    "负无穷经过 softmax 会变成 0，剩下的位置重新归一化，所以每一行仍然加起来等于 1。被挡住的未来不会分走任何概率。"
    "这里的数字来自一份固定的教学用 Q 和 K，维度是 4，所以缩放因子是 2。以第三行、也就是“真”这个 Query 为例，"
    "四个分数大约是 0.14、0.57、0.87 和负无穷，softmax 之后大约是 0.22、0.33、0.45 和 0。"
    "如果你改掉最后一个词的 Key，前三个位置的输出完全不变。演示脚本里这个差是 0。",
)
def softmax_numbers(board: Board) -> None:
    board.kicker("03  因果自注意力")
    board.heading("负无穷变成 0，每一行仍是 1")
    weights = demo.causal_example()["weights"]
    weight_matrix(board, 860, 160, demo.TOKENS, weights, cell=110)
    board.text("教学网络的权重。d_k = 4，除以 2。", 980, 760, 800, lh(13), 13, DIM)
    board.rect(88, 180, 740, 520, CARD, radius=16, line="2E2358")
    board.text("以“真”这一行为例", 116, 200, 680, lh(16), 16, PURPLE, bold=True)
    lines = [
        ("分数", "0.14   0.57   0.87   -inf"),
        ("归一化之后", "0.22   0.33   0.45    0"),
        ("这一行的和", "1.00"),
    ]
    for index, (name, value) in enumerate(lines):
        top = 270 + index * 120
        board.text(name, 116, top, 680, lh(14), 14, DIM)
        board.text(value, 116, top + lh(14) + 4, 680, lh(16), 16, WHITE, bold=True, mono=True)
    board.source(
        "数字由 causal 演示脚本用固定的 Q、K 算出，保留两位小数。改动未来的 Key 后，前三个位置的最大误差是 0。"
    )


@slide(
    "因果自注意力",
    "训练用 teacher forcing。输入是整句，标签是向右错开一位，一次前向同时得到所有位置的损失。"
    "这和能看未来不是一回事。并行的是矩阵运算，信息依赖仍然是下三角。"
    "生成时未来是真的未知，所以要一个词一个词地追加。也正因为旧的一行不依赖新词，训练时并行算出来的旧位置，"
    "和以后逐步生成时的旧位置，结果是一致的。",
)
def train_vs_gen(board: Board) -> None:
    board.kicker("03  因果自注意力")
    board.heading("并行的是计算，不是偷看")
    found = boxes(2, 190, 680, 2)
    board.card(
        *found[0],
        "训练：一次算出所有位置",
        "四个词同时在输入里。掩码保证“今天”看不到“天气”，“天气”看不到“真”。损失是四个下一词预测的平均。显卡可以把这四行一起算完。",
        "Teacher forcing",
    )
    board.card(
        *found[1],
        "生成：一次只新增一个词",
        "没有标签可以喂回来。采样或贪心选出下一个 token，追加到序列末尾，再算新的一行。旧行不用重算，这就是缓存的入口。",
        "自回归",
        BLUE,
    )


@slide(
    "因果自注意力",
    "本仓库 gpt2.py 的 CausalSelfAttention 在第 92 行到第 122 行。c_attn 把每个 token 从 768 维投影到 2304 维，再切成 q、k、v。"
    "然后 view 成 12 个头，每头 64 维。真正的打分、缩放、掩码和加权，都交给 PyTorch 的 scaled_dot_product_attention。"
    "不传 scale 的时候，默认就是除以根号下每个头的维度。GPT-2 small 里，这个数是 8。",
)
def code_shapes(board: Board) -> None:
    board.kicker("03  因果自注意力")
    board.heading("对照 gpt2.py：形状是怎么变的")
    steps = [
        ("x", "(B, T, C)", "C = 768"),
        ("c_attn", "(B, T, 3C)", "一次投影"),
        ("分头", "(B, nh, T, hs)", "12 × 64"),
        ("打分", "(B, nh, T, L)", "除以 √64 = 8"),
        ("乘 V", "(B, nh, T, hs)", "加权求和"),
        ("c_proj", "(B, T, C)", "把头混合回来"),
    ]
    for index, (name, shape, note) in enumerate(steps):
        col = index % 3
        row = index // 3
        left = 88 + col * 590
        top = 190 + row * 250
        board.rect(left, top, 560, 220, CARD, radius=16, line="2E2358")
        board.text(f"{index + 1:02d}", left + 24, top + 20, 500, lh(13), 13, PURPLE, bold=True)
        board.text(name, left + 24, top + 52, 510, lh(22), 22, WHITE, bold=True)
        board.text(shape, left + 24, top + 52 + lh(22) + 6, 510, lh(16), 16, LAVENDER, mono=True)
        board.text(note, left + 24, top + 150, 510, lh(14), 14, MUTED)
    board.source(
        "src/learn_gpt2/gpt2.py，CausalSelfAttention，第 92–122 行。L 是缓存里的总长度，没有缓存时 L = T。"
    )


@slide(
    "因果自注意力",
    "有了缓存以后，Query 的长度 T 和 Key 的长度 L 不一定相等。代码在第 110 行到第 120 行分了三种情况。"
    "T 等于 L，说明没有更早的缓存，直接用 is_causal，掩码对齐左上角。T 等于 1，新来的这一个词可以看全部历史，不必再遮。"
    "如果一次送进多个新词，前面又已经有缓存，第 i 个 Query 的绝对位置是 L 减 T 再加 i。掩码要用下三角，并把对角线偏移 L 减 T，对齐到右下角。"
    "这时候如果误用 is_causal，会把本该看见的缓存挡住。注释里专门写了这件事。",
)
def three_branches(board: Board) -> None:
    board.kicker("03  因果自注意力")
    board.heading("三种掩码，看 T 和 L")
    found = boxes(3, 185, 500, 3)
    specs = [
        (
            "T = L",
            "没有更早的缓存",
            "is_causal = True\n掩码贴着左上角。",
            lambda i, j: j <= i,
            4,
            4,
        ),
        ("T = 1", "解码一步", "新 token 可以看\n全部已缓存的 Key。", lambda i, j: True, 1, 5),
        (
            "否则",
            "一次来多个新词",
            "tril，对角线偏移\nL − T，贴右下角。",
            lambda i, j: j <= (3 + i),
            2,
            5,
        ),
    ]
    for box, (title, kicker, body, see, rows, cols) in zip(found, specs, strict=True):
        x, y, w, h = box
        board.rect(x, y, w, h, CARD, radius=16, line="2E2358")
        board.text(kicker, x + 20, y + 16, w - 40, lh(12), 12, PURPLE, bold=True)
        board.text(title, x + 20, y + 44, w - 40, lh(20), 20, WHITE, bold=True)
        board.text(body, x + 20, y + 44 + lh(20) + 6, w - 40, lh(14, 2), 14, MUTED)
        mini_mask(board, x + 24, y + 230, rows, cols, see)
    board.text(
        "右图是 T=2、L=5：两个新 Query 坐在绝对位置 3 和 4，只能看到自己和更早的 Key。is_causal 会错把掩码安在左上角。",
        88,
        710,
        1744,
        lh(15, 2),
        15,
        MUTED,
    )
    board.source("gpt2.py 第 110–120 行。第三种分支的 diagonal 是 L − T。")


@slide(
    "KV Cache",
    "因果性带来一个直接的加速机会。旧 token 的 Key 和 Value 不再变化，就不用每次生成都重算。这就是 KV Cache。"
    "接下来看它省的是哪一部分计算，以及要为此付多少显存。",
)
def section_kv(board: Board) -> None:
    board.text("04", 88, 250, 500, lh(64), 64, PURPLE, bold=True)
    board.text("KV Cache", 88, 250 + lh(64) + 8, 1600, lh(36), 36, WHITE, bold=True)
    board.text(
        "旧的键和值是常量。新的词只把自己的一行算出来。",
        88,
        250 + lh(64) + lh(36) + 24,
        1500,
        lh(18),
        18,
        MUTED,
    )


@slide(
    "KV Cache",
    "没有缓存的时候，每生成一个词，都把当前整段前缀再送进模型。所有位置的 Q、K、V 和分数全部重算一遍。"
    "有缓存的时候，提示词先做一次预填充，把每一层的 K 和 V 存下来。之后每一步只把最新的一个 token 送进去，"
    "算出它自己的 q、k、v，把 k 和 v 追加到缓存末尾，再用这个 q 去看全部历史的 K 和 V。"
    "仓库里 LayerKVCache.update 做的就是这段追加，形状注释写的是 batch、头数、最大长度、每头维度。",
)
def cache_idea(board: Board) -> None:
    board.kicker("04  KV Cache")
    board.heading("没有缓存，每一步都在重算过去")
    found = boxes(2, 190, 680, 2)
    board.card(
        *found[0],
        "每步重跑整段前缀",
        "序列从 1 变到 N，第 t 步要把 t 个 token 都送进每一层。token 前向的总次数大约是 N 的平方除以 2。注意力还要为已经算过的行再算一次。",
        "无缓存",
        VIOLET,
    )
    board.card(
        *found[1],
        "预填充一次，解码只算新词",
        "prompt 先把 K、V 填满。之后每个新 token 只做一次投影，再和缓存里的全部 Key 做点积。结果应当和重算最后一行相同。",
        "有缓存",
        BLUE,
    )


@slide(
    "KV Cache",
    "用刚才那四个词做预填充，缓存长度变成 4。下一步来了一个新 token。Query 只有一行，Key 和 Value 有五行。"
    "这一行注意力，应该等于把五个词重算一遍、再取因果矩阵的最后一行。演示脚本里，两条路径的最大绝对误差大约是 5.6 乘以 10 的负 17 次方，"
    "也就是浮点误差。缓存改变的是计算路径，不是数学结果。",
)
def cache_steps(board: Board) -> None:
    board.kicker("04  KV Cache")
    board.heading("预填充，再解码一步")
    stages = [
        ("1", "Prefill", "四个词一次送入。\n每层的 K、V 长度为 4。"),
        ("2", "追加", "新词只算 q、k、v。\n缓存长度变成 5。"),
        ("3", "只算新的一行", "1 个 Query 看 5 个 Key。\n前四行不必重算。"),
    ]
    for index, (num, title, body) in enumerate(stages):
        left = 88 + index * 590
        board.rect(left, 190, 560, 280, CARD, radius=16, line="2E2358")
        board.text(num, left + 24, 210, 80, lh(20), 20, PURPLE, bold=True)
        board.text(title, left + 24, 260, 510, lh(22), 22, WHITE, bold=True)
        board.text(body, left + 24, 260 + lh(22) + 8, 510, lh(16, 2), 16, MUTED)
    board.rect(88, 500, 1744, 280, CARD, radius=16, line="2E2358")
    board.text("和“五个词重算、取最后一行”相比", 116, 524, 1600, lh(16), 16, PURPLE, bold=True)
    board.text("5.6×10⁻¹⁷", 116, 580, 700, lh(36), 36, WHITE, bold=True)
    board.text(
        "这是最大绝对误差。未来的 Key 被改掉时，因果模型前三个位置的输出差是 0。两条性质一起，说明缓存既正确，也不会偷偷用到未来。",
        116,
        700,
        1680,
        lh(16, 2),
        16,
        MUTED,
    )
    board.source(
        "demo.py 的 kv_equivalence 与 future_invariance。教学向量是固定的，不是训练出来的权重。"
    )


@slide(
    "KV Cache",
    "每一层要存两份东西，K 和 V。单个张量的形状是 batch、头数、长度、每头维度。因为头数乘以每头维度等于模型宽度，"
    "字节数可以写成：层数乘 2，乘 batch，乘长度，乘宽度，再乘每个元素占的字节。GPT-2 small、batch 为 1、长度 1024、半精度，正好是 36 MB。"
    "GPT-3 的 1750 亿模型、长度 2048、半精度、batch 为 1，正好是 9 GB。batch 一旦变大，缓存会比权重更早成为麻烦。",
)
def memory(board: Board) -> None:
    board.kicker("04  KV Cache")
    board.heading("缓存的形状，和它占多少显存")
    board.rect(88, 175, 1744, 130, CARD, radius=16, line="2E2358")
    board.text(
        "字节 = 层数 × 2 × B × L × C × 每元素字节",
        88,
        175,
        1744,
        130,
        26,
        WHITE,
        bold=True,
        align="center",
        valign="middle",
        mono=True,
    )
    found = boxes(2, 330, 280, 2)
    board.stat(
        *found[0], "36 MiB", "GPT-2 small · 12 层 · 宽 768\nbatch 1 · 长度 1024 · fp16", PURPLE
    )
    board.stat(
        *found[1], "9 GiB", "GPT-3 175B · 96 层 · 宽 12288\nbatch 1 · 长度 2048 · fp16", BLUE
    )
    board.text(
        "单层的 K 或 V 是 (B, nh, L, hs)。nh 乘 hs 就是 C，所以式子里不再出现头数。batch 64 时，GPT-3 这份缓存会到 576 GiB，这还不算 1750 亿参数本身。",
        88,
        640,
        1744,
        lh(16, 2),
        16,
        MUTED,
    )
    board.source(
        "形状见 gpt2.py 里 LayerKVCache 的注释。175B 的层数和宽度来自 Brown et al., 2020，表 2.1。"
    )


@slide(
    "KV Cache",
    "先看 token 前向次数。从空上下文生成到长度 1024，没有缓存时，各步长度加起来是 52 万 4800 次 token 前向；有缓存时只有 1024 次，大约少 512 倍。"
    "再看一个头要算的分数个数。如果每步都重算整张方阵，总数是 3 亿 5843 万 8400，是有缓存时的 683 倍。"
    "如果无缓存也只算下三角，是 342 倍。请注意：有了缓存，总的注意力仍然随长度的平方增长，并没有变成线性。"
    "因为每一个新词，还是要看全部历史。",
)
def complexity(board: Board) -> None:
    board.kicker("04  KV Cache")
    board.heading("省掉的是重复计算，不是历史本身")
    counts = demo.score_counts(1024)
    headers = ["生成到长度 1024", "分数个数（单头）", "相对有缓存"]
    rows = [
        ["每步重算整张方阵", f"{counts['full']:,}", "683 倍"],
        ["每步只算下三角", f"{counts['triangle']:,}", "342 倍"],
        ["有 KV Cache", f"{counts['cached']:,}", "1 倍"],
    ]
    table(board, 88, 185, 1744, headers, rows, [1.5, 1.3, 1], points=16)
    found = boxes(2, 500, 280, 2)
    board.stat(*found[0], "524,800", "无缓存的 token 前向次数\n1+2+…+1024", VIOLET)
    board.stat(*found[1], "1,024", "有缓存的 token 前向次数\n每步只跑最新的一个词", BLUE)
    board.source(
        "只统计一个头的分数个数，从空上下文生成到 N=1024。投影的节省还要更大：无缓存每步要重算整段的 Q、K、V。"
    )


@slide(
    "KV Cache",
    "如果注意力可以看未来，新词一到，旧位置的输出就会变，缓存立刻作废。掩码把依赖限制在过去和自己，旧的 Key 和 Value 才是常量。"
    "这也是三种分支要对齐绝对位置的原因。缓存让 Query 的长度和 Key 的长度不再相等，掩码不能再假设矩阵的左上角就是第一个词。"
    "位置编码同样要跟着绝对位置走。gpt2.py 里，有缓存时位置下标是从缓存长度开始数的，不是从 0 重新数。",
)
def cache_needs_causal(board: Board) -> None:
    board.kicker("04  KV Cache")
    board.heading("没有因果掩码，缓存就是错的")
    found = boxes(3, 190, 680, 3)
    board.card(
        *found[0],
        "依赖被限制在过去",
        "新 token 可以读旧的 Key，旧的 Query 不会读新的 Key。所以旧行的输出保持不变。",
        "掩码",
    )
    board.card(
        *found[1],
        "绝对位置不能重置",
        "有缓存时，新词的位置编号从缓存长度往后数。学来的绝对位置表，认的是这个编号。",
        "wpe",
        VIOLET,
    )
    board.card(
        *found[2],
        "超出上下文就要丢掉",
        "GPT-2 的位置表只有 1024 项。序列更长时，缓存的位置对不上裁剪后的窗口，generate 会退回重算。",
        "边界",
        BLUE,
    )
    board.source(
        "gpt2.py 的 forward 用 kv_cache.length 作为位置起点；generate 在超过 block_size 时放弃缓存。"
    )


@slide(
    "GPT-2",
    "2019 年的 GPT-2 把解码器单独拿出来，放到大规模网页文本上做下一词预测。论文的主张是：一个足够大的语言模型，"
    "不必为每个任务去更新参数，也能零样本地做很多不同的事。标题就叫：语言模型是无监督的多任务学习者。",
)
def section_gpt2(board: Board) -> None:
    board.text("05", 88, 250, 500, lh(64), 64, PURPLE, bold=True)
    board.text("GPT-2", 88, 250 + lh(64) + 8, 1600, lh(36), 36, WHITE, bold=True)
    board.text(
        "Language Models are Unsupervised Multitask Learners · 2019",
        88,
        250 + lh(64) + lh(36) + 24,
        1600,
        lh(18),
        18,
        MUTED,
    )


@slide(
    "GPT-2",
    "这里的零样本，指的是下游任务上不再做梯度更新。任务靠自然语言本身来说明：给一段上下文，让模型把下文接下去。"
    "它还不是后来 GPT-3 那种把十几个例子塞进提示里的 few-shot 协议。另外，论文说四个模型当时都还没有拟合完 WebText，"
    "验证困惑度仍然会随着多训练而下降。",
)
def gpt2_claim(board: Board) -> None:
    board.kicker("05  GPT-2")
    board.heading("零样本：不更新参数，让文本把任务说出来")
    found = boxes(3, 190, 680, 3)
    board.card(
        *found[0],
        "只有下一词预测",
        "没有为翻译、问答或摘要单独准备损失函数。这些任务都被写成“把一段话接完”。",
        "训练目标",
    )
    board.card(
        *found[1],
        "零样本不是没有提示",
        "提示仍然在指定要做什么。有的评测还会加停用词过滤。这和 GPT-3 的 few-shot 不是同一套协议。",
        "不要混口径",
        VIOLET,
    )
    board.card(
        *found[2],
        "模型仍然欠拟合",
        "四个尺寸的学习率按 WebText 百分之五的留出集手调。论文写明，它们都还没有把 WebText 拟合完。",
        "WebText",
        BLUE,
    )


@slide(
    "GPT-2",
    "训练数据叫 WebText。他们抓取 Reddit 上至少 3 个 karma 的外链，把这种点赞当作一种便宜的人工质量过滤。"
    "去重和清洗之后，略多于 800 万篇文档，一共 40 GB 文本，而且不包含 2017 年 12 月之后的链接。维基百科被整份拿掉，"
    "因为太多测试集都来自那里。分词用字节级 BPE，词表 50257，这样任意 Unicode 字符串都有概率。上下文从 GPT 的 512 加长到 1024。",
)
def webtext(board: Board) -> None:
    board.kicker("05  GPT-2")
    board.heading("WebText：用 karma 滤过一遍的网页")
    found = boxes(4, 190, 420, 4)
    stats = [
        ("≥ 3", "Reddit 外链的 karma\n当作质量启发式", PURPLE),
        ("8M", "去重后的文档数\n来自 4500 万条链接", VIOLET),
        ("40 GB", "文本总量\n不含 2017-12 之后", BLUE),
        ("50257", "字节级 BPE 词表\n上下文 1024", PURPLE),
    ]
    for box, (value, label, accent) in zip(found, stats, strict=True):
        board.stat(*box, value, label, accent)
    board.text(
        "字节级 BPE 的底表只有 256 个字节，因此不会有未知词。他们禁止 BPE 跨字符类别合并，避免把 dog、dog!、dog? 拆成太多几乎重复的词，空格是例外。维基百科被移除，是为了减少和下游测试集的重叠。",
        88,
        640,
        1744,
        lh(16, 3),
        16,
        MUTED,
    )
    board.source("Radford et al., 2019，2.1 与 2.2。文档数是“略多于 800 万”，链接数是 4500 万。")


@slide(
    "GPT-2",
    "模型仍然是只有解码器的 Transformer，但论文 2.3 节写了几处相对原始 Transformer 的改动。层归一化挪到每个子层的输入，"
    "类似预激活残差网络，全部子层之后再加一次层归一化。这就是 Pre-LN。残差路径上的投影，初始化时按 1 除以根号 N 缩小，"
    "用来抵消一层层残差相加造成的方差积累。本仓库只对注意力和 MLP 的 c_proj 做这个缩放，N 取两倍层数，因为每个 Block 有两次残差。"
    "位置编码是可学习的绝对位置表，不是正弦函数。",
)
def gpt2_arch(board: Board) -> None:
    board.kicker("05  GPT-2")
    board.heading("Pre-LN，以及按深度缩小的初始化")
    found = boxes(2, 185, 500, 2)
    board.card(
        *found[0],
        "先归一化，再进子层",
        "原始 Transformer 大致是 x 加子层之后再做 LayerNorm。GPT-2 改成 x 加上子层作用在 LayerNorm(x) 上的结果。本仓库 Block 的两行就是这个形式，最后还有一次 ln_f。",
        "Pre-LN",
    )
    board.card(
        *found[1],
        "残差投影除以根号 N",
        "论文说 N 是残差层的数量。本仓库把注意力和 MLP 两次残差都算进去，所以标准差是 0.02 除以根号下 2 倍层数，并且只缩放名字以 c_proj.weight 结尾的参数。",
        "初始化",
        VIOLET,
    )
    board.text(
        "另外两处容量上的改动：词表扩到 50257，上下文从 512 增到 1024，batch size 写的是 512。激活沿用 GPT 的 GELU；本仓库 MLP 用的是 tanh 近似，和 Hugging Face 的 GPT-2 一致。",
        88,
        710,
        1744,
        lh(15, 2),
        15,
        MUTED,
    )
    board.source(
        "Radford et al., 2019，2.3。代码见 gpt2.py 的 Block.forward 与 _init_weights 后面的 c_proj 缩放。"
    )


@slide(
    "GPT-2",
    "论文训练了四个尺寸，参数量写成 1.17 亿、3.45 亿、7.62 亿和 15.42 亿，层数 12、24、36、48，宽度 768、1024、1280、1600。"
    "公开实现通常把最小的那个计成大约 1.24 亿，本仓库的注释也是这个口径。层数和宽度与论文一致，头数是公开配置里的 12、16、20、25，论文的表 2 没有单列头数。"
    "零样本语言建模上，8 个数据集里有 7 个超过了当时的最好结果。例外是 One Billion Word。那个数据集把句子打乱了，长距离结构没了，GPT-2 反而更差。",
)
def gpt2_sizes(board: Board) -> None:
    board.kicker("05  GPT-2")
    board.heading("四个尺寸，七个数据集上的零样本")
    headers = ["论文参数量", "层数", "宽度", "头数"]
    rows = [
        ["117M", "12", "768", "12"],
        ["345M", "24", "1024", "16"],
        ["762M", "36", "1280", "20"],
        ["1542M", "48", "1600", "25"],
    ]
    table(board, 88, 175, 1100, headers, rows, [1.3, 1, 1, 1], points=15)
    board.rect(1220, 175, 612, 430, CARD, radius=16, line="2E2358")
    board.text("7 / 8", 1244, 200, 560, lh(40), 40, PURPLE, bold=True)
    board.text(
        "零样本语言建模超过当时最好结果的数据集个数。1BW 是例外：1542M 的困惑度 42.16，此前最好是 21.8。句子被打乱后，长上下文帮不上忙。",
        1244,
        320,
        560,
        lh(15, 5),
        15,
        MUTED,
    )
    board.source(
        "表 2、表 3 与 3.1 节。头数来自公开 GPT-2 配置，也是本仓库 GPT2_CONFIGS。1.24 亿是把词嵌入和 bias 算进去的另一种口径。"
    )


@slide(
    "GPT-2",
    "LAMBADA 要预测一个至少需要五十个词的上下文，人才能猜中的句末单词。困惑度从 99.8 降到 8.6，表里最大模型是 8.63。"
    "准确率从 19% 提高到 52.66%。模型的很多错误其实是合法的续写，只是没有停在句末。再加上一个停用词过滤，近似这个约束，准确率到 63.24%。"
    "儿童书测试因为《丛林之书》和训练集重叠，论文改报验证集：普通名词 93.3%，命名实体 89.1%。Winograd 达到 70.70%，比此前最好高 7 个百分点，但只有 273 道题。",
)
def gpt2_downstream(board: Board) -> None:
    board.kicker("05  GPT-2")
    board.heading("长距离、常识，以及这些数字的条件")
    found = boxes(3, 185, 500, 3)
    board.stat(*found[0], "8.6", "LAMBADA 困惑度\n此前 99.8。表 3 为 8.63", PURPLE)
    board.stat(*found[1], "63.24%", "加停用词过滤后的准确率\n不加过滤是 52.66%", VIOLET)
    board.stat(*found[2], "70.70%", "Winograd Schema\n仅 273 题，提升 7 个点", BLUE)
    board.text(
        "儿童书测试报的是没有明显重叠的验证集，不是被《丛林之书》污染的测试集。这些数字都没有做任务微调。读的时候要连同过滤条件、数据划分一起读。",
        88,
        710,
        1744,
        lh(15, 2),
        15,
        MUTED,
    )
    board.source(
        "Radford et al., 2019，3.2 至 3.4，以及表 3。准确率 19% 的对照来自 Dehghani et al., 2018。"
    )


@slide(
    "GPT-3",
    "2020 年的 GPT-3 把同一条路放大到 1750 亿参数。论文想说明的是：模型越大，越能利用上下文里的几个例子，"
    "在不更新任何参数的情况下把任务做出来。这就是 in-context learning，上下文学习。标题换成了：语言模型是少样本学习者。",
)
def section_gpt3(board: Board) -> None:
    board.text("06", 88, 250, 500, lh(64), 64, PURPLE, bold=True)
    board.text("GPT-3", 88, 250 + lh(64) + 8, 1600, lh(36), 36, WHITE, bold=True)
    board.text(
        "Language Models are Few-Shot Learners · Brown et al., 2020",
        88,
        250 + lh(64) + lh(36) + 24,
        1600,
        lh(18),
        18,
        MUTED,
    )


@slide(
    "GPT-3",
    "论文把测试时的用法分成几档，前三档都不做梯度更新。Zero-shot 只给一段自然语言指令，不给例子。"
    "One-shot 在指令后面再给一个完整示范。Few-shot 则尽量多给，通常是十几个到一百个，直到塞满 2048 的上下文。"
    "Fine-tuning 才是用任务数据去更新参数的老办法。Few-shot 往往更强。但论文也说，如果要和人比较，one-shot 甚至 zero-shot 有时才是更公平的对照，"
    "因为人通常也不需要先看一百个例子。",
)
def shot_defs(board: Board) -> None:
    board.kicker("06  GPT-3")
    board.heading("Zero-shot、one-shot、few-shot")
    items = [
        ("0", "Zero-shot", "只有任务说明，\n没有输入输出示范。", PURPLE),
        ("1", "One-shot", "说明之外，\n再给一个完整例子。", VIOLET),
        ("K", "Few-shot", "例子尽量多，\n通常十个到一百个。", BLUE),
        ("FT", "Fine-tune", "用任务数据更新权重。\n这里只作为对照。", DIM),
    ]
    for index, (mark, title, body, accent) in enumerate(items):
        left = 88 + index * 450
        board.rect(left, 200, 420, 460, CARD, radius=16, line="2E2358")
        board.text(mark, left + 24, 230, 370, lh(32), 32, accent, bold=True)
        board.text(title, left + 24, 340, 370, lh(22), 22, WHITE, bold=True)
        board.text(body, left + 24, 340 + lh(22) + 16, 370, lh(16, 3), 16, MUTED)
    board.source("Brown et al., 2020，2 节。上下文窗口 n_ctx = 2048。三种设定都不更新参数。")


@slide(
    "GPT-3",
    "他们训练了 8 个规模，从 1.25 亿到 1750 亿，每个模型都看 3000 亿个 token，上下文都是 2048。"
    "1750 亿的模型有 96 层，宽度 12288，96 个头，每头 128 维。架构沿用 GPT-2 的 Pre-LN、初始化缩放和可逆分词，"
    "另外隔层使用稠密注意力和局部带状的稀疏注意力。数据是过滤后的 Common Crawl、WebText2、两套书和英文维基。"
    "采样不按体积。Common Crawl 占训练混合的百分之六十，但在 3000 亿 token 里大约只看 0.44 遍；维基百科大约看 3.4 遍。",
)
def gpt3_scale(board: Board) -> None:
    board.kicker("06  GPT-3")
    board.heading("1750 亿参数，3000 亿 token")
    board.rect(88, 175, 1744, 150, CARD, radius=16, line="2E2358")
    board.text("175B", 112, 190, 280, lh(28), 28, PURPLE, bold=True)
    board.text(
        "96 层 · 宽 12288 · 96 头 · 每头 128\n8 个模型都训练 300B tokens · 上下文 2048",
        420,
        198,
        1360,
        lh(16, 2),
        16,
        WHITE,
    )
    headers = ["数据", "规模", "采样权重", "看过的遍数"]
    rows = [
        ["Common Crawl（过滤后）", "4100 亿", "60%", "0.44"],
        ["WebText2", "190 亿", "22%", "2.9"],
        ["Books1 / Books2", "120 亿 / 550 亿", "8% / 8%", "1.9 / 0.43"],
        ["Wikipedia", "30 亿", "3%", "3.4"],
    ]
    table(board, 88, 350, 1744, headers, rows, [1.6, 1.2, 1, 1], points=15)
    board.source(
        "Brown et al., 2020，表 2.1 与表 2.2。遍数是训练 3000 亿 token 时的 epochs elapsed。"
    )


@slide(
    "GPT-3",
    "几个数字可以看出 few-shot 和规模各自起了什么作用。LAMBADA 的准确率，zero-shot 是 76.2%，few-shot 是 86.4%，困惑度降到 1.92。"
    "有趣的是 one-shot 反而只有 72.5%，因为填空格式需要好几个例子，模型才认得出来。TriviaQA 从 64.3% 到 71.2%，few-shot 超过了当时的闭卷微调结果。"
    "CoQA 的 F1 从 81.5 到 85.0。两位数加法在 few-shot 下是 100%，五位数加法只剩 9.3%。"
    "1750 亿模型写的大约 200 词新闻，人类辨别准确率 52%，接近瞎猜。SuperGLUE 的 few-shot 平均分 71.8，高于 BERT-Large 的微调 69.0，但仍明显低于微调最高水平 89.0。词义比较 WiC 几乎是随机的 49.4%。",
)
def gpt3_results(board: Board) -> None:
    board.kicker("06  GPT-3")
    board.heading("越大，越会用上下文里的例子")
    headers = ["任务", "Zero-shot", "One-shot", "Few-shot"]
    rows = [
        ["LAMBADA 准确率", "76.2%", "72.5%", "86.4%"],
        ["TriviaQA", "64.3%", "68.0%", "71.2%"],
        ["CoQA F1", "81.5", "84.0", "85.0"],
        ["两位数加法", "76.9%", "99.6%", "100%"],
    ]
    table(board, 88, 170, 1744, headers, rows, [1.4, 1, 1, 1], points=15)
    found = boxes(3, 560, 250, 3)
    board.stat(*found[0], "71.8", "SuperGLUE few-shot\n微调 SOTA 是 89.0", PURPLE)
    board.stat(*found[1], "49.4%", "WiC，几乎等于随机\n比较两个句子里的词", VIOLET)
    board.stat(*found[2], "52%", "人类认出 175B 新闻\n95% 区间 49% 到 54%", BLUE)
    board.source(
        "表 3.2、3.3、3.8、3.9、3.11，以及摘要中的 CoQA。五位数加法 few-shot 是 9.3%。LAMBADA 的 one-shot 低于 zero-shot。"
    )


@slide(
    "GPT-3",
    "论文第 5 节把局限写得很直接。长文本仍然会重复、自相矛盾，写长了就不连贯。比较类的任务，比如 WiC 和 ANLI，few-shot 也几乎不行。"
    "作者认为，这和自回归模型看不到双向信息有关。预训练的样本效率远不如人：测试的时候可以 one-shot，训练的时候却看过远超过一个人一生阅读量的文本。"
    "1750 亿的推理很贵。Few-shot 到底是在临时学会一个新任务，还是在认出训练时见过的任务格式，论文认为仍然不清楚，而且不同任务答案可能不同。"
    "此外还有没滤干净的测试集污染、训练数据里的偏见，以及生成文本被滥用的风险。",
)
def gpt3_limits(board: Board) -> None:
    board.kicker("06  GPT-3")
    board.heading("规模没有取消这些限制")
    found = boxes(4, 185, 680, 2)
    cards = [
        (
            "文本会散",
            "长样本仍会语义重复、自相矛盾，或者出现不承接的句子。论文放出了 500 段未挑选的无条件样本。",
            PURPLE,
        ),
        (
            "比较是弱项",
            "WiC、ANLI，以及一部分阅读理解，few-shot 接近机会水平。自回归模型不方便来回对照两段话。",
            VIOLET,
        ),
        (
            "预训练并不省样本",
            "测试时接近人类的少样本，训练时看的词远多于一个人的一生。模型也不接地：没有视觉和身体经验。",
            BLUE,
        ),
        (
            "贵、难解释、有偏见",
            "推理成本高。预测在新分布上不一定校准。数据里的刻板印象会被复述。污染过滤还有已知的漏洞。",
            PURPLE,
        ),
    ]
    for box, (title, body, accent) in zip(found, cards, strict=True):
        board.card(*box, title, body, accent=accent)
    board.source("Brown et al., 2020，第 5 节，以及第 2.2 节关于污染过滤 bug 的说明。")


@slide(
    "收束",
    "把三篇论文放回一条线上。Transformer 用注意力替换循环，并在解码器里放上因果掩码。GPT-2 只保留解码器，用下一词预测在 WebText 上训练，"
    "发现不更新参数也能做多种任务，同时把 Pre-LN 和初始化缩放定了下来。GPT-3 几乎不改这个配方，把模型和数据放大，于是上下文里的几个例子变成一种新的用法。"
    "因果自注意力保证不能看未来。KV Cache 让这种自回归生成在工程上跑得动。本仓库的 gpt2.py，把这条链上最核心的两段，掩码和缓存，写成了可以逐行对照的代码。",
)
def thread(board: Board) -> None:
    board.kicker("收束")
    board.heading("三篇论文是同一条路的三段")
    items = [
        ("2017", "Transformer", "注意力成为主干。\n解码器必须遮住未来。"),
        ("2019", "GPT-2", "只留解码器。\nPre-LN，零样本多任务。"),
        ("2020", "GPT-3", "同样的配方放大。\n上下文学习随规模变强。"),
    ]
    for index, (year, title, body) in enumerate(items):
        left = 88 + index * 590
        board.rect(left, 190, 560, 420, CARD, radius=16, line="2E2358")
        board.text(year, left + 28, 214, 500, lh(16), 16, PURPLE, bold=True)
        board.text(title, left + 28, 270, 500, lh(26), 26, WHITE, bold=True)
        board.text(body, left + 28, 270 + lh(26) + 16, 500, lh(16, 3), 16, MUTED)
    board.text(
        "本仓库把这条路上的两件具体事情落在 gpt2.py：第 92 到 122 行的因果自注意力，以及第 35 到 89 行的 KV Cache。",
        88,
        640,
        1744,
        lh(16, 2),
        16,
        LAVENDER,
    )


@slide(
    "收束",
    "如果只记四句话。第一，注意力是按相似度对 Value 做加权平均。第二，因果掩码是包含对角线的下三角，在 softmax 之前把未来填成负无穷，每一行的权重仍然和为 1。"
    "第三，KV Cache 省掉的是重复计算。总的注意力代价仍然随长度的平方增长，因为新词还要看全部历史。第四，GPT-2 说明了零样本多任务，GPT-3 说明 few-shot 会随规模变强，"
    "同时也把不连贯、不会比较、很贵、有偏见这些限制暴露得更清楚。下一步可以直接读 gpt2.py 的 forward 和 generate，再跑本目录的演示脚本，改一改未来的 Key，看前面的输出动不动。",
)
def close(board: Board) -> None:
    board.kicker("收束")
    board.heading("四句话")
    lines = [
        ("1", "注意力是对 Value 的加权平均，权重来自缩放后的 Q 与 K 的点积。"),
        ("2", "因果掩码含对角线。未来是负无穷，softmax 之后那一行仍然等于 1。"),
        ("3", "KV Cache 去掉重复计算。看历史这件事本身，仍然是长度的平方。"),
        ("4", "GPT-2 把任务放进文本。GPT-3 让例子随规模变得有用，局限也还在。"),
    ]
    for index, (num, text) in enumerate(lines):
        top = 180 + index * 150
        board.rect(88, top, 1744, 136, CARD, radius=16, line="2E2358")
        board.text(num, 116, top, 80, 136, 26, PURPLE, bold=True, align="center", valign="middle")
        board.text(text, 210, top, 1560, 136, 20, WHITE, valign="middle")
    board.source(
        "演示脚本：presentations/05-transformer-gpt2-gpt3/demo.py。代码：src/learn_gpt2/gpt2.py。"
    )


def render() -> list[Path]:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    presentation = new_presentation()
    total = len(SLIDES)
    paths: list[Path] = []
    OVERFLOWS.clear()
    for index, spec in enumerate(SLIDES, start=1):
        board = Board(presentation, index, total, spec["section"], spec["narration"])
        spec["draw"](board)
        board.chrome_bar()
        path = PNG_DIR / f"slide-{index:02d}.png"
        board.save_png(path)
        paths.append(path)
        print(f"slide {index:02d} {spec['draw'].__name__}")
    if OVERFLOWS:
        print("\n".join(OVERFLOWS))
        raise SystemExit(f"{len(OVERFLOWS)} layout overflows")
    out = Path(__file__).resolve().parent / "transformer_gpt2_gpt3.pptx"
    presentation.save(out)
    _contact(paths)
    _narration()
    print(f"wrote {out} and {len(paths)} pngs")
    return paths


def _contact(paths: list[Path]) -> None:
    columns = 4
    thumb_w, thumb_h = 480, 270
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * thumb_h), "#090715")
    for index, path in enumerate(paths):
        image = Image.open(path).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(image, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    sheet.save(BUILD / "contact.png")


def _narration() -> None:
    chunks = ["# 旁白稿：从注意力到大语言模型", ""]
    for index, spec in enumerate(SLIDES, start=1):
        chunks.append(f"## 第 {index} 页 · {spec['draw'].__name__}")
        chunks.append("")
        chunks.append(spec["narration"])
        chunks.append("")
    path = Path(__file__).resolve().parent / "narration.md"
    path.write_text("\n".join(chunks), encoding="utf-8")


if __name__ == "__main__":
    render()
