"""Draw the deck's figures and the video's animation frames with matplotlib.

    python figures.py

Writes build/figures/<name>.png, build/frames/<anim>/<nn>.png and build/figures/manifest.json
(figure sizes in inches, read by deck.js). Data come from the three papers and from
../01-gpt2-inference/data/{attention,benchmark}.json (real GPT-2 measurements).
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).parent
FIG_DIR = HERE / "build" / "figures"
FRAME_DIR = HERE / "build" / "frames"
DATA_01 = HERE.parent / "01-gpt2-inference" / "data"

INK = "#1B1F3B"
INK2 = "#2D3250"
AMBER = "#F2A541"
TEAL = "#2A9D8F"
CORAL = "#E76F51"
INDIGO = "#5B6CFF"
SLATE = "#8D99AE"
LIGHT = "#F2F4F8"
MID = "#D9DEE8"
WHITE = "#FFFFFF"
DEEP = "#264653"

plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": SLATE,
        "savefig.facecolor": WHITE,
    }
)
MONO = "DejaVu Sans Mono"
DPI = 200
MANIFEST: dict[str, list[float]] = {}

TOKENS = ["我", "喜欢", "吃", "苹果"]
# Scaled scores of the running 4-token example (one head, d_k = 4, so raw scores are 2x these).
SCORES = np.array(
    [
        [2.0, 1.0, 0.5, 0.1],
        [1.0, 2.0, 0.3, 0.8],
        [0.2, 1.5, 1.0, 0.4],
        [0.5, 1.0, 1.2, 2.0],
    ]
)


# ------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------
def canvas(w: float, h: float):
    """A figure of w x h inches whose data coordinates are inches, origin bottom-left."""
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    MANIFEST[name] = [round(float(w), 4), round(float(h), 4)]
    fig.savefig(FIG_DIR / f"{name}.png", dpi=DPI)
    plt.close(fig)


def save_frame(fig, anim: str, k: int) -> None:
    d = FRAME_DIR / anim
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{k:02d}.png", dpi=DPI)


def box(ax, x, y, w, h, fc, ec=None, lw=0.0, r=0.08, alpha=1.0, z=2, ls="-"):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        fc=fc,
        ec=ec or fc,
        lw=lw,
        alpha=alpha,
        zorder=z,
        ls=ls,
    )
    ax.add_patch(p)
    return p


def label(ax, x, y, s, size=12, color=INK, ha="center", va="center", weight="normal", z=6, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, zorder=z, **kw)


def arrow(ax, x1, y1, x2, y2, color=SLATE, lw=1.6, style="-|>", ms=12, z=3, **kw):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=ms,
        color=color,
        lw=lw,
        zorder=z,
        shrinkA=0,
        shrinkB=0,
        **kw,
    )
    ax.add_patch(a)
    return a


def mix(c1: str, c2: str, t: float) -> tuple:
    a = np.array(matplotlib.colors.to_rgb(c1))
    b = np.array(matplotlib.colors.to_rgb(c2))
    return tuple(a + (b - a) * t)


def softmax_rows(m: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    m = m.astype(float).copy()
    if mask is not None:
        m[~mask] = -np.inf
    e = np.exp(m - m.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def style_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color=MID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def grid_matrix(
    ax,
    x0,
    y0,
    cell,
    values,
    colors,
    texts=None,
    rows=None,
    cols=None,
    tsize=11,
    lsize=11,
    tcolor=None,
    border=None,
):
    """Draw a matrix of cells; (x0, y0) is the top-left corner."""
    n, m = values.shape
    for i in range(n):
        for j in range(m):
            x, y = x0 + j * cell, y0 - (i + 1) * cell
            ax.add_patch(Rectangle((x, y), cell, cell, fc=colors[i][j], ec=WHITE, lw=1.5, zorder=2))
            if texts is not None and texts[i][j]:
                c = tcolor[i][j] if tcolor is not None else INK
                label(ax, x + cell / 2, y + cell / 2, texts[i][j], size=tsize, color=c, z=4)
    if rows:
        for i, r in enumerate(rows):
            label(ax, x0 - 0.1, y0 - (i + 0.5) * cell, r, size=lsize, ha="right", color=INK2)
    if cols:
        for j, c in enumerate(cols):
            label(ax, x0 + (j + 0.5) * cell, y0 + 0.14, c, size=lsize, va="bottom", color=INK2)
    if border is not None:
        i0, j0, i1, j1, col = border
        ax.add_patch(
            Rectangle(
                (x0 + j0 * cell, y0 - (i1 + 1) * cell),
                (j1 - j0 + 1) * cell,
                (i1 - i0 + 1) * cell,
                fc="none",
                ec=col,
                lw=2.6,
                zorder=5,
            )
        )


def heat(v: float, hi: str = TEAL) -> tuple:
    return mix(WHITE, hi, 0.12 + 0.88 * float(np.clip(v, 0, 1)))


# ------------------------------------------------------------------------------------------
# Part 0: language modeling
# ------------------------------------------------------------------------------------------
def fig_lm_next() -> None:
    cands = ["苹果", "火锅", "面条", "饭", "西瓜", "辣的"]
    probs = [0.31, 0.18, 0.12, 0.10, 0.07, 0.05]
    fig = plt.figure(figsize=(5.4, 3.9), dpi=DPI)
    ax = fig.add_axes((0.2, 0.13, 0.74, 0.62))
    y = np.arange(len(cands))[::-1]
    ax.barh(y, probs, color=[TEAL] + [MID] * (len(cands) - 1), height=0.62, zorder=2)
    for yi, p in zip(y, probs, strict=True):
        ax.text(p + 0.006, yi, f"{p:.2f}", va="center", fontsize=11, color=INK2)
    ax.set_yticks(y, cands, fontsize=13)
    ax.set_xlim(0, 0.38)
    ax.set_xlabel("p( 下一个 token | 我 喜欢 吃 )", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    fig.text(0.05, 0.93, "输入：", fontsize=12, color=INK2, va="center")
    xs = 0.19
    for t in ["我", "喜欢", "吃", "？"]:
        w = 0.07 + 0.045 * len(t)
        fc = AMBER if t == "？" else LIGHT
        fig.patches.append(
            FancyBboxPatch(
                (xs, 0.875),
                w,
                0.11,
                boxstyle="round,pad=0,rounding_size=0.02",
                transform=fig.transFigure,
                fc=fc,
                ec=MID,
                lw=1,
            )
        )
        fig.text(xs + w / 2, 0.93, t, fontsize=13, ha="center", va="center", color=INK)
        xs += w + 0.025
    fig.text(0.95, 0.80, "（示意数据）", fontsize=9, color=SLATE, ha="right")
    save(fig, "lm_next")


# ------------------------------------------------------------------------------------------
# Part 1: attention
# ------------------------------------------------------------------------------------------
def fig_rnn_vs_attn() -> None:
    w, h = 6.2, 4.7
    fig, ax = canvas(w, h)
    n, x0, dx = 6, 0.75, 0.95
    # RNN chain
    label(ax, 0.25, 4.42, "循环网络：逐步传递，O(n) 步", size=13, ha="left", weight="bold")
    for i in range(n):
        x = x0 + i * dx
        box(ax, x - 0.3, 3.35, 0.6, 0.5, INDIGO, r=0.08)
        label(ax, x, 3.6, f"h{i + 1}", size=12, color=WHITE, weight="bold")
        box(ax, x - 0.26, 2.55, 0.52, 0.42, LIGHT, ec=MID, lw=1)
        label(ax, x, 2.76, f"x{i + 1}", size=11, color=INK2)
        arrow(ax, x, 2.97, x, 3.35, color=SLATE, lw=1.2, ms=9)
        if i < n - 1:
            arrow(ax, x + 0.3, 3.6, x + dx - 0.3, 3.6, color=CORAL, lw=2.0, ms=11)
    label(ax, w - 0.2, 4.08, "必须等上一步算完 →", size=10, color=CORAL, ha="right")
    ax.plot([0.25, w - 0.25], [2.3, 2.3], color=MID, lw=1)
    # Self-attention
    label(ax, 0.25, 2.02, "自注意力：任意两点一步直达，O(1)", size=13, ha="left", weight="bold")
    ys = 0.55
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = x0 + i * dx, x0 + j * dx
            hl = i == 0 and j == n - 1
            a = FancyArrowPatch(
                (xi, ys + 0.25),
                (xj, ys + 0.25),
                connectionstyle=f"arc3,rad=-{0.28 + 0.02 * (j - i)}",
                arrowstyle="-",
                color=TEAL if hl else mix(WHITE, TEAL, 0.45),
                lw=2.4 if hl else 1.0,
                zorder=2,
            )
            ax.add_patch(a)
    for i in range(n):
        x = x0 + i * dx
        box(ax, x - 0.26, ys, 0.52, 0.42, TEAL, r=0.08, z=4)
        label(ax, x, ys + 0.21, f"x{i + 1}", size=11, color=WHITE, weight="bold", z=5)
    label(ax, w - 0.2, 0.2, "所有位置同时计算", size=10, color=TEAL, ha="right")
    save(fig, "rnn_vs_attn")


def fig_qkv() -> None:
    w, h = 6.0, 4.3
    fig, ax = canvas(w, h)
    # token vector
    box(ax, 0.25, 1.45, 1.25, 1.1, LIGHT, ec=MID, lw=1)
    label(ax, 0.875, 2.2, "喜欢", size=16, weight="bold")
    label(ax, 0.875, 1.75, "向量 x", size=11, color=INK2)
    rows = [
        (3.3, "W", "Q", INDIGO, "查询 q", "我想找什么？"),
        (2.0, "W", "K", TEAL, "键 k", "我能提供什么？"),
        (0.7, "W", "V", AMBER, "值 v", "被选中时给出的内容"),
    ]
    for y, base, sup, col, name, desc in rows:
        arrow(ax, 1.5, 2.0, 2.25, y + 0.3, color=SLATE, lw=1.4, ms=10)
        box(ax, 2.3, y, 0.85, 0.6, mix(WHITE, col, 0.25), ec=col, lw=1.4)
        label(ax, 2.725, y + 0.3, f"${base}^{sup}$", size=15, color=INK)
        arrow(ax, 3.15, y + 0.3, 3.55, y + 0.3, color=col, lw=1.6, ms=10)
        for k in range(5):
            ax.add_patch(
                Rectangle(
                    (3.6 + k * 0.18, y + 0.12),
                    0.18,
                    0.36,
                    fc=mix(WHITE, col, 0.35 + 0.13 * k),
                    ec=WHITE,
                    lw=1,
                    zorder=3,
                )
            )
        label(ax, 4.65, y + 0.42, name, size=13, ha="left", weight="bold", color=col)
        label(ax, 4.65, y + 0.12, desc, size=10, ha="left", color=INK2)
    label(ax, 0.25, 4.12, "同一个向量，乘以三个可学习矩阵", size=12, ha="left", color=INK2)
    save(fig, "qkv")


def _sdpa_stage(k: int):
    """Frame k of the scaled dot-product attention walk-through (0 = overview)."""
    w, h = 7.6, 5.4
    fig, ax = canvas(w, h)
    raw = SCORES * 2.0
    weights = softmax_rows(SCORES)
    v = np.array([[1.0, 0.0, 0.2], [0.0, 1.0, 0.4], [0.5, 0.5, 1.0], [0.9, 0.1, 0.0]])
    out = weights @ v
    panels = [
        ("① 打分  S = Q·Kᵀ", raw, lambda x: heat(x / 4.0, INDIGO), "{:.1f}", 4),
        ("② 缩放  S / √d_k（d_k = 4）", SCORES, lambda x: heat(x / 2.0, INDIGO), "{:.1f}", 4),
        ("③ softmax（每行和为 1）", weights, lambda x: heat(x / 0.6, TEAL), "{:.2f}", 4),
        ("④ 加权求和  O = A·V", out, lambda x: heat(x, AMBER), "{:.2f}", 3),
    ]
    pos = [(0.35, 5.25), (4.05, 5.25), (0.35, 2.85), (4.05, 2.85)]
    for idx, ((title, mat, cfun, fmt, ncol), (px, py)) in enumerate(zip(panels, pos, strict=True)):
        active = k == 0 or idx + 1 <= k
        current = idx + 1 == k
        fade = 0.0 if active else 0.82
        if current:
            box(ax, px - 0.2, py - 2.2, 3.55, 2.3, mix(WHITE, AMBER, 0.12), ec=AMBER, lw=2, z=1)
        label(
            ax,
            px,
            py - 0.05,
            title,
            size=12,
            ha="left",
            va="top",
            weight="bold",
            color=mix(INK, WHITE, fade),
        )
        cell = 0.36
        x0 = px + 0.75
        y0 = py - 0.55
        colors = [[mix(cfun(mat[i, j]), WHITE, fade) for j in range(ncol)] for i in range(4)]
        texts = [[fmt.format(mat[i, j]) for j in range(ncol)] for i in range(4)]
        tcol = [[mix(INK, WHITE, fade)] * ncol for _ in range(4)]
        cols = ["我", "喜欢", "吃", "苹果"] if ncol == 4 else ["d1", "d2", "d3"]
        grid_matrix(
            ax,
            x0,
            y0,
            cell,
            mat[:, :ncol],
            colors,
            texts,
            rows=TOKENS,
            cols=cols,
            tsize=9,
            lsize=9,
            tcolor=tcol,
        )
    label(
        ax,
        w / 2,
        0.3,
        r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(QK^{\top}/\sqrt{d_k}\right)V$",
        size=15,
        color=INK,
    )
    return fig


def fig_sdpa() -> None:
    for k in range(5):
        fig = _sdpa_stage(k)
        save_frame(fig, "sdpa", k)
        if k == 4:
            save(fig, "sdpa")
        else:
            plt.close(fig)


def fig_scaling() -> None:
    rng = np.random.default_rng(7)
    dk, n = 64, 10
    q = rng.standard_normal(dk)
    keys = rng.standard_normal((n, dk))
    s = keys @ q
    p_raw = np.exp(s - s.max())
    p_raw /= p_raw.sum()
    s2 = s / math.sqrt(dk)
    p_sc = np.exp(s2 - s2.max())
    p_sc /= p_sc.sum()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.9), dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.9, bottom=0.08, hspace=0.75)
    for ax, p, col, ttl in [
        (axes[0], p_raw, CORAL, f"不缩放：q·k 的标准差 ≈ {s.std():.1f}"),
        (axes[1], p_sc, TEAL, f"除以 √d_k：标准差 ≈ {s2.std():.1f}"),
    ]:
        x = np.arange(n)
        ax.bar(x, p, color=col, width=0.62, zorder=2)
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x, [f"k{i + 1}" for i in range(n)], fontsize=9)
        ax.set_yticks([0, 0.5, 1.0])
        style_axes(ax)
        ax.set_title(ttl, fontsize=12, loc="left", color=INK)
        ax.text(
            n - 0.5,
            0.98,
            f"最大权重 {p.max():.2f}",
            ha="right",
            va="top",
            fontsize=10,
            color=col,
            weight="bold",
        )
    note = f"d_k = {dk}，随机 q 与 {n} 个 k 的 softmax 权重"
    fig.text(0.1, 0.965, note, fontsize=10, color=SLATE)
    save(fig, "scaling")


def fig_multihead() -> None:
    w, h = 7.0, 4.4
    fig, ax = canvas(w, h)
    box(ax, 0.2, 1.75, 1.0, 0.9, LIGHT, ec=MID, lw=1)
    label(ax, 0.7, 2.35, "X", size=16, weight="bold")
    label(ax, 0.7, 1.98, "512 维", size=10, color=INK2)
    heads = 8
    for i in range(heads):
        y = 3.85 - i * 0.47
        arrow(ax, 1.2, 2.2, 1.65, y + 0.17, color=MID, lw=1.0, ms=7)
        col = [INDIGO, TEAL, AMBER, CORAL][i % 4]
        box(ax, 1.7, y, 2.3, 0.36, mix(WHITE, col, 0.22), ec=col, lw=1.1, r=0.06)
        label(ax, 1.82, y + 0.18, f"head {i + 1}", size=9.5, ha="left", color=INK, family=MONO)
        label(ax, 3.92, y + 0.18, "Attention · 64 维", size=9, ha="right", color=INK2)
        arrow(ax, 4.0, y + 0.18, 4.45, 2.2, color=MID, lw=1.0, ms=7)
    box(ax, 4.5, 1.6, 1.0, 1.2, INK2, r=0.08)
    label(ax, 5.0, 2.36, "Concat", size=11, color=WHITE, weight="bold")
    label(ax, 5.0, 2.0, "8×64", size=10, color=MID)
    arrow(ax, 5.5, 2.2, 5.8, 2.2, color=SLATE, lw=1.4, ms=10)
    box(ax, 5.85, 1.85, 0.95, 0.7, TEAL, r=0.08)
    label(ax, 6.325, 2.2, "$W^O$", size=14, color=WHITE)
    label(ax, 6.325, 1.5, "512 维", size=10, color=INK2)
    note = "每个头：独立的 $W_i^Q, W_i^K, W_i^V$，关注不同子空间"
    label(ax, 1.7, 0.18, note, size=10, ha="left", color=SLATE)
    save(fig, "multihead")


def fig_transformer() -> None:
    w, h = 6.0, 5.4
    fig, ax = canvas(w, h)

    def stack(x, y0, layers, title, nlab):
        bw = 2.3
        top = y0 + sum(hh for _, _, hh in layers) + 0.12 * (len(layers) + 1)
        box(ax, x - 0.15, y0 - 0.1, bw + 0.3, top - y0 + 0.1, LIGHT, ec=MID, lw=1.2, r=0.12, z=1)
        y = y0 + 0.08
        for name, col, hh in layers:
            box(ax, x, y, bw, hh, col, r=0.06)
            label(ax, x + bw / 2, y + hh / 2, name, size=10.5, color=WHITE, weight="bold")
            y += hh + 0.12
        label(ax, x + bw / 2, top + 0.15, f"{title}  {nlab}", size=12, weight="bold")
        return top

    ffn = ("前馈网络 FFN", DEEP, 0.5)
    enc_top = stack(0.75, 1.2, [("多头自注意力", TEAL, 0.5), ffn], "编码器", "×6")
    dec_layers = [("带掩码自注意力", CORAL, 0.5), ("编码器-解码器注意力", AMBER, 0.5), ffn]
    dec_top = stack(3.45, 1.2, dec_layers, "解码器", "×6")
    arrow(ax, 3.05, 1.85, 3.45, 2.1, color=AMBER, lw=2.2, ms=12)
    label(ax, 3.25, 1.62, "K, V", size=9, color=AMBER)
    for x, txt in [(0.75, "源句嵌入 + 位置编码"), (3.45, "目标嵌入（右移）+ 位置编码")]:
        box(ax, x, 0.35, 2.3, 0.45, mix(WHITE, SLATE, 0.3), r=0.06)
        label(ax, x + 1.15, 0.575, txt, size=9.5, color=INK)
        arrow(ax, x + 1.15, 0.8, x + 1.15, 1.1, color=SLATE, lw=1.3, ms=9)
    box(ax, 3.45, dec_top + 0.45, 2.3, 0.45, INK2, r=0.06)
    label(ax, 4.6, dec_top + 0.675, "Linear + Softmax", size=10, color=WHITE)
    arrow(ax, 4.6, dec_top + 0.25, 4.6, dec_top + 0.45, color=SLATE, lw=1.3, ms=9)
    label(ax, 0.6, h - 0.25, "残差连接 + LayerNorm 包裹每个子层", size=9.5, ha="left", color=SLATE)
    _ = enc_top
    save(fig, "transformer")


def fig_posenc() -> None:
    pos, d = 64, 128
    pe = np.zeros((pos, d))
    p = np.arange(pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    pe[:, 0::2] = np.sin(p / 10000 ** (i / d))
    pe[:, 1::2] = np.cos(p / 10000 ** (i / d))
    fig = plt.figure(figsize=(5.6, 3.9), dpi=DPI)
    ax = fig.add_axes((0.13, 0.15, 0.72, 0.75))
    im = ax.imshow(pe, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xlabel("维度 i", fontsize=11)
    ax.set_ylabel("位置 pos", fontsize=11)
    ax.tick_params(labelsize=9)
    cax = fig.add_axes((0.87, 0.15, 0.03, 0.75))
    fig.colorbar(im, cax=cax).ax.tick_params(labelsize=8)
    ax.set_title("正弦位置编码（d_model = 128）", fontsize=11, loc="left")
    save(fig, "posenc")


# ------------------------------------------------------------------------------------------
# Part 2: causal self-attention
# ------------------------------------------------------------------------------------------
def _causal_frame(k: int):
    w, h = 7.0, 5.0
    fig, ax = canvas(w, h)
    weights = softmax_rows(SCORES, np.tril(np.ones((4, 4), dtype=bool)))
    stages = [
        "① 缩放后的打分矩阵 S",
        "② 加上因果掩码：对角线以上 = −∞",
        "③ 逐行 softmax",
        "③ 逐行 softmax",
        "③ 逐行 softmax",
        "③ 逐行 softmax",
        "④ 结果：下三角的注意力权重",
    ]
    label(ax, 0.3, 4.72, stages[k], size=14, ha="left", weight="bold")
    cell = 0.82
    x0, y0 = 1.55, 4.0
    done = max(0, k - 1)  # rows already through softmax
    colors, texts, tcol = [], [], []
    for i in range(4):
        rc, rt, rtc = [], [], []
        for j in range(4):
            masked = j > i and k >= 1
            if i < done and k >= 2:
                if j > i:
                    rc.append(LIGHT)
                    rt.append("0")
                    rtc.append(SLATE)
                else:
                    rc.append(heat(weights[i, j] / 0.8, TEAL))
                    rt.append(f"{weights[i, j]:.2f}")
                    rtc.append(INK)
            elif masked:
                rc.append(mix(WHITE, SLATE, 0.55))
                rt.append("−∞")
                rtc.append(WHITE)
            else:
                rc.append(heat(SCORES[i, j] / 2.2, INDIGO))
                rt.append(f"{SCORES[i, j]:.1f}")
                rtc.append(INK)
        colors.append(rc)
        texts.append(rt)
        tcol.append(rtc)
    border = None
    if 2 <= k <= 5:
        border = (k - 2, 0, k - 2, 3, AMBER)
    grid_matrix(
        ax,
        x0,
        y0,
        cell,
        SCORES,
        colors,
        texts,
        rows=TOKENS,
        cols=TOKENS,
        tsize=14,
        lsize=13,
        tcolor=tcol,
        border=border,
    )
    label(ax, x0 - 0.95, y0 + 0.45, "查询 ↓  键 →", size=10, color=SLATE, ha="left")
    # side notes
    nx = x0 + 4 * cell + 0.3
    notes = {
        0: ["每行：一个查询", "每列：一个键", "数值越大越关注"],
        1: ["后面的位置 = 未来", "softmax 前设为 −∞", "e^(−∞) = 0"],
        2: ["“我”只能看自己", "权重 = 1.00"],
        3: ["“喜欢”：自己 0.73", "“我” 0.27"],
        4: ["“吃”看不到“苹果”", "一半以上给“喜欢”"],
        5: ["“苹果”看全部四个", "每行和仍为 1"],
        6: ["原始分 0.1 → 权重 0", "未来信息完全被挡住", "位置 i 只依赖 x≤i"],
    }[k]
    for t, s in enumerate(notes):
        label(ax, nx, y0 - 0.45 - 0.5 * t, s, size=12, ha="left", color=INK2)
    if k == 6:
        ax.add_patch(
            Rectangle((x0 + 3 * cell, y0 - cell), cell, cell, fc="none", ec=CORAL, lw=3, zorder=6)
        )
        arrow(ax, x0 + 3.5 * cell, y0 - cell - 0.05, nx - 0.1, y0 - 0.5, color=CORAL, lw=1.5, ms=10)
    if k == 6:
        note = "softmax( S + M )：权重矩阵是下三角"
        label(ax, w / 2, 0.3, note, size=12, color=TEAL, weight="bold")
    elif k >= 1:
        label(ax, w / 2, 0.3, "M[i, j] = 0（j ≤ i），−∞（j > i）", size=12, color=INK2)
    return fig


def fig_causal() -> None:
    for k in range(7):
        fig = _causal_frame(k)
        save_frame(fig, "causal", k)
        if k == 6:
            save(fig, "causal")
        else:
            plt.close(fig)


def fig_causal_layers() -> None:
    w, h = 6.4, 4.6
    fig, ax = canvas(w, h)
    n, layers = 6, 3
    x0, dx = 0.9, 0.95
    ys = [0.6, 1.75, 2.9, 3.95]
    focus = 2  # 0-based position whose dependency cone we show
    for li in range(layers):
        y_lo, y_hi = ys[li], ys[li + 1]
        for i in range(n):
            for j in range(i + 1):
                in_cone = i <= focus
                hl = i == focus
                dim = mix(WHITE, TEAL, 0.4) if in_cone else mix(WHITE, SLATE, 0.35)
                col = TEAL if hl else dim
                ax.plot(
                    [x0 + j * dx, x0 + i * dx],
                    [y_lo + 0.18, y_hi - 0.18],
                    color=col,
                    lw=1.8 if hl else 0.8,
                    zorder=1,
                )
    for li, y in enumerate(ys):
        for i in range(n):
            x = x0 + i * dx
            in_cone = i <= focus
            fc = (TEAL if i == focus else mix(WHITE, TEAL, 0.55)) if in_cone else LIGHT
            ec = TEAL if in_cone else MID
            ax.add_patch(plt.Circle((x, y), 0.18, fc=fc, ec=ec, lw=1.2, zorder=3))
            if li == 0:
                label(ax, x, y - 0.42, f"x{i + 1}", size=11, color=INK if in_cone else SLATE)
        name = "输入" if li == 0 else f"第 {li} 层"
        label(ax, 0.15, y, name, size=10, ha="left", color=INK2)
    label(
        ax,
        x0 + focus * dx,
        ys[-1] + 0.33,
        "k₃, v₃ 只取决于 x₁…x₃",
        size=11,
        color=TEAL,
        weight="bold",
    )
    label(ax, x0 + 4.5 * dx, ys[-1] + 0.33, "x₄ 之后的 token 影响不到它", size=10, color=SLATE)
    save(fig, "causal_layers")


def fig_real_attention() -> None:
    data = json.loads((DATA_01 / "attention.json").read_text())
    toks = [t.replace(" ", "·") for t in data["tokens"]]
    maps = np.array(data["maps"])  # (layer, head, T, T)
    picks = pick_heads(maps)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.93, bottom=0.1, wspace=0.35, hspace=0.5)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("t", [WHITE, TEAL])
    for ax, (layer, head, desc) in zip(axes.flat, picks, strict=True):
        m = maps[layer, head]
        ax.imshow(m, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(toks)), toks, rotation=60, fontsize=7, family=MONO)
        ax.set_yticks(range(len(toks)), toks, fontsize=7, family=MONO)
        ax.set_title(f"第 {layer} 层 · 头 {head}：{desc}", fontsize=10, loc="left")
        for s in ax.spines.values():
            s.set_color(MID)
    save(fig, "real_attention")


def pick_heads(maps: np.ndarray) -> list[tuple[int, int, str]]:
    """Pick four illustrative heads: previous-token, first-token, and two spread-out ones."""
    # maps: (layer, head, query, key)
    prev = np.diagonal(maps, offset=-1, axis1=2, axis2=3).mean(axis=-1)  # attention to i - 1
    first = maps[:, :, 2:, 0].mean(axis=-1)
    rows = maps[:, :, 3:]
    ent = -np.mean(np.sum(rows * np.log(rows + 1e-9), axis=-1), axis=-1)
    lp, hp = np.unravel_index(np.argmax(prev), prev.shape)
    first[0] = 0  # prefer a deeper layer for the first-token sink
    lf, hf = np.unravel_index(np.argmax(first), first.shape)
    order = np.dstack(np.unravel_index(np.argsort(-ent, axis=None), ent.shape))[0]
    spread = [(int(a), int(b)) for a, b in order if (a, b) not in {(lp, hp), (lf, hf)}][:2]
    return [
        (int(lp), int(hp), "看前一个 token"),
        (int(lf), int(hf), "集中在第一个 token"),
        (spread[0][0], spread[0][1], "分散关注"),
        (spread[1][0], spread[1][1], "分散关注"),
    ]


# ------------------------------------------------------------------------------------------
# Part 3: GPT-2
# ------------------------------------------------------------------------------------------
def fig_gpt2_arch() -> None:
    w, h = 4.3, 5.5
    fig, ax = canvas(w, h)
    cx, bw = 2.15, 2.6
    x = cx - bw / 2

    def blk(y, hh, txt, col, tc=WHITE, size=10.5):
        box(ax, x, y, bw, hh, col, r=0.06)
        label(ax, cx, y + hh / 2, txt, size=size, color=tc, weight="bold")

    blk(0.15, 0.42, "token id  →  wte + wpe", mix(WHITE, SLATE, 0.35), tc=INK)
    arrow(ax, cx, 0.57, cx, 0.82, color=SLATE, lw=1.3, ms=9)
    box(ax, x - 0.22, 0.85, bw + 0.44, 2.7, LIGHT, ec=MID, lw=1.2, r=0.12, z=1)
    label(ax, x - 0.28, 2.2, "N×", size=13, ha="right", color=INK2, weight="bold")
    blk(1.0, 0.36, "ln_1", AMBER, tc=INK)
    blk(1.5, 0.5, "因果自注意力", CORAL)
    blk(2.25, 0.36, "ln_2", AMBER, tc=INK)
    blk(2.75, 0.5, "MLP（4×，GELU）", DEEP)
    for y in (1.36, 2.0, 2.61):
        arrow(ax, cx, y, cx, y + 0.14, color=SLATE, lw=1.1, ms=7)
    # residual paths
    for yb, yt in ((0.92, 2.13), (2.17, 3.35)):
        ax.plot([x + bw + 0.1, x + bw + 0.1], [yb, yt], color=TEAL, lw=1.6)
        ax.plot([x + bw - 0.05, x + bw + 0.1], [yb, yb], color=TEAL, lw=1.6)
        arrow(ax, x + bw + 0.1, yt, x + bw - 0.05, yt, color=TEAL, lw=1.6, ms=8)
        label(ax, x + bw + 0.17, yt, "+", size=12, color=TEAL, ha="left", weight="bold")
    arrow(ax, cx, 3.55, cx, 4.12, color=SLATE, lw=1.3, ms=9)
    blk(4.15, 0.36, "ln_f（新增）", AMBER, tc=INK)
    arrow(ax, cx, 4.51, cx, 4.68, color=SLATE, lw=1.3, ms=9)
    blk(4.7, 0.42, "lm_head → logits", INK2)
    label(ax, cx + 0.15, 0.7, "Pre-LN：先归一化再进子层", size=8.5, ha="left", color=SLATE)
    save(fig, "gpt2_arch")


def fig_gpt2_lm() -> None:
    params = np.array([117, 345, 762, 1542]) * 1e6
    ppl = [35.13, 15.60, 10.87, 8.63]
    acc = [45.99, 55.48, 60.12, 63.24]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.9), dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.17, wspace=0.35)
    ax = axes[0]
    ax.plot(params, ppl, "-o", color=TEAL, lw=2.2, ms=7, zorder=3)
    ax.axhline(99.8, color=CORAL, ls="--", lw=1.4)
    ax.text(1.3e8, 80, "此前最佳 99.8", color=CORAL, fontsize=10)
    for p, v in zip(params, ppl, strict=True):
        ax.text(p, v * 1.18, f"{v:g}", ha="center", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5, 150)
    ax.set_title("LAMBADA 困惑度（越低越好）", fontsize=11, loc="left")
    ax = axes[1]
    ax.plot(params, acc, "-o", color=INDIGO, lw=2.2, ms=7, zorder=3)
    ax.axhline(59.23, color=CORAL, ls="--", lw=1.4)
    ax.text(1.25e8, 60.3, "此前最佳 59.23", color=CORAL, fontsize=10)
    for p, v in zip(params, acc, strict=True):
        ax.text(p, v - 2.6, f"{v:g}", ha="center", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(40, 68)
    ax.set_title("LAMBADA 准确率 %", fontsize=11, loc="left")
    for ax in axes:
        style_axes(ax)
        ax.set_xticks(params, ["117M", "345M", "762M", "1.5B"], fontsize=9)
        ax.minorticks_off()
        ax.set_xlabel("参数量", fontsize=10)
    fig.text(0.1, 0.94, "零样本，不做任何微调（GPT-2 论文表 3）", fontsize=10, color=SLATE)
    save(fig, "gpt2_lm")


# ------------------------------------------------------------------------------------------
# Part 4: GPT-3
# ------------------------------------------------------------------------------------------
def fig_gpt3_data() -> None:
    names = ["Common Crawl", "WebText2", "Books1", "Books2", "Wikipedia"]
    tokens = [410, 19, 12, 55, 3]
    weight = [60, 22, 8, 8, 3]
    epochs = [0.44, 2.9, 1.9, 0.43, 3.4]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.7), dpi=DPI, sharey=True)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.84, bottom=0.12, wspace=0.12)
    y = np.arange(len(names))[::-1]
    axes[0].barh(y, tokens, color=SLATE, height=0.6, zorder=2)
    for yi, t in zip(y, tokens, strict=True):
        axes[0].text(t + 6, yi, f"{t}B", va="center", fontsize=9, color=INK2)
    axes[0].set_xlim(0, 520)
    axes[0].set_title("token 数（十亿）", fontsize=11, loc="left")
    axes[1].barh(y, weight, color=TEAL, height=0.6, zorder=2)
    for yi, wt, ep in zip(y, weight, epochs, strict=True):
        axes[1].text(wt + 1, yi, f"{wt}%  ·  {ep} 轮", va="center", fontsize=9, color=INK2)
    axes[1].set_xlim(0, 100)
    axes[1].set_title("训练采样权重 · 经过轮数", fontsize=11, loc="left")
    axes[0].set_yticks(y, names, fontsize=10)
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    fig.text(0.02, 0.95, "共训练 3000 亿 token（GPT-3 论文表 2.2）", fontsize=10, color=SLATE)
    save(fig, "gpt3_data")


def fig_gpt3_results() -> None:
    tasks = ["LAMBADA\n准确率", "TriviaQA\n准确率", "CoQA\nF1", "法→英\nBLEU", "德→英\nBLEU"]
    zero = [76.2, 64.3, 81.5, 21.2, 27.2]
    one = [72.5, 68.0, 84.0, 33.7, 30.4]
    few = [86.4, 71.2, 85.0, 39.2, 40.6]
    sota = [68.0, 68.0, 90.7, 35.0, 40.2]
    fig = plt.figure(figsize=(7.6, 4.6), dpi=DPI)
    ax = fig.add_axes((0.08, 0.17, 0.9, 0.68))
    x = np.arange(len(tasks))
    bw = 0.24
    series = [(-bw, zero, MID, "零样本"), (0, one, SLATE, "单样本"), (bw, few, TEAL, "少样本")]
    for off, vals, col, name in series:
        ax.bar(x + off, vals, bw * 0.92, color=col, label=name, zorder=2)
        for xi, v in zip(x + off, vals, strict=True):
            ax.text(xi, v + 1.2, f"{v:g}", ha="center", fontsize=7.5, color=INK2)
    for xi, s in zip(x, sota, strict=True):
        ax.plot([xi - 1.6 * bw, xi + 1.6 * bw], [s, s], color=CORAL, ls="--", lw=1.5, zorder=3)
    ax.plot([], [], color=CORAL, ls="--", label="此前最佳")
    ax.set_xticks(x, tasks, fontsize=10)
    ax.set_ylim(0, 100)
    style_axes(ax)
    ax.legend(ncol=4, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    note = "TriviaQA 的“此前最佳”为开放域 RAG；CoQA 为微调 SOTA；翻译为有监督 SOTA"
    fig.text(0.08, 0.02, note, fontsize=8, color=SLATE)
    save(fig, "gpt3_results")


def fig_gpt3_arith() -> None:
    tasks = ["2D+", "2D−", "3D+", "3D−", "4D+", "4D−", "5D+", "5D−", "2D×", "1DC"]
    zero = [76.9, 58.0, 34.2, 48.3, 4.0, 7.5, 0.7, 0.8, 19.8, 9.8]
    few = [100.0, 98.9, 80.4, 94.2, 25.5, 26.8, 9.3, 9.9, 29.2, 21.3]
    fig = plt.figure(figsize=(6.0, 3.3), dpi=DPI)
    ax = fig.add_axes((0.1, 0.15, 0.88, 0.66))
    x = np.arange(len(tasks))
    ax.bar(x - 0.2, zero, 0.38, color=MID, label="零样本", zorder=2)
    ax.bar(x + 0.2, few, 0.38, color=TEAL, label="少样本", zorder=2)
    ax.set_xticks(x, tasks, fontsize=9)
    ax.set_ylim(0, 110)
    style_axes(ax)
    ax.legend(ncol=2, fontsize=9, frameon=False, loc="upper right")
    ax.set_title("GPT-3 175B 算术准确率 %（表 3.9）", fontsize=11, loc="left", pad=12)
    save(fig, "gpt3_arith")


def fig_gpt3_scaling() -> None:
    names = ["Small", "Medium", "Large", "XL", "2.7B", "6.7B", "13B", "175B"]
    params = np.array([125e6, 356e6, 760e6, 1.32e9, 2.65e9, 6.66e9, 12.85e9, 174.6e9])
    pfdays = np.array([2.60, 7.42, 15.8, 27.5, 55.2, 139, 268, 3640])
    detect = [76, 61, 68, 62, 62, 60, 55, 52]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.9), dpi=DPI)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.16, wspace=0.22)
    ax = axes[0]
    ax.plot(params, pfdays, "-o", color=INDIGO, lw=2, ms=6, zorder=3)
    for p, c, n in zip(params, pfdays, names, strict=True):
        ax.text(p, c * 1.55, n, ha="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1, 2e4)
    ax.set_xlabel("参数量", fontsize=10)
    ax.set_title("训练计算量（PF/s-天，附录表 D.1）", fontsize=11, loc="left")
    ax.text(1.5e8, 3e3, "175B：3.14×10²³ FLOPs\n≈ 3640 PF/s-天", fontsize=10, color=INDIGO)
    ax = axes[1]
    ax.plot(params, detect, "-o", color=CORAL, lw=2, ms=6, zorder=3)
    ax.axhline(50, color=SLATE, ls="--", lw=1.2)
    ax.text(1.3e8, 51.5, "随机猜测 50%", fontsize=9.5, color=SLATE)
    ax.axhline(86, color=AMBER, ls=":", lw=1.4)
    ax.text(1.3e8, 82.5, "故意做差的对照模型 86%", fontsize=9.5, color=AMBER)
    for p, d in zip(params, detect, strict=True):
        ax.text(p, d + 2, f"{d}%", ha="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(40, 92)
    ax.set_xlabel("参数量", fontsize=10)
    ax.set_title("人类识别“模型写的新闻”的准确率（表 3.11）", fontsize=11, loc="left")
    for ax in axes:
        style_axes(ax)
        ax.tick_params(labelsize=9)
    save(fig, "gpt3_scaling")


# ------------------------------------------------------------------------------------------
# Part 5: KV cache
# ------------------------------------------------------------------------------------------
def _recompute_frame(k: int, n: int = 6):
    w, h = 6.6, 4.9
    fig, ax = canvas(w, h)
    cell = 0.5
    x0, y0 = 0.95, 4.35
    label(ax, x0 + n * cell / 2, y0 + 0.28, "本步送入模型的位置", size=11, color=INK2)
    for j in range(n):
        label(ax, x0 + (j + 0.5) * cell, y0 + 0.02, f"{j + 1}", size=10, color=SLATE, va="bottom")
    for i in range(n):
        y = y0 - (i + 1) * cell - 0.06 * i
        label(ax, x0 - 0.15, y + cell / 2, f"第 {i + 1} 步", size=11, ha="right", color=INK2)
        if i < k:
            for j in range(i + 1):
                fc = TEAL if j == i else mix(WHITE, SLATE, 0.45)
                sq = Rectangle((x0 + j * cell, y), cell, cell, fc=fc, ec=WHITE, lw=2, zorder=2)
                ax.add_patch(sq)
            text = f"{i + 1} 次前向"
            label(ax, x0 + n * cell + 0.25, y + cell / 2, text, size=11, ha="left", color=INK)
        else:
            for j in range(i + 1):
                sq = Rectangle((x0 + j * cell, y), cell, cell, fc=LIGHT, ec=WHITE, lw=2, zorder=1)
                ax.add_patch(sq)
    # legend
    ax.add_patch(Rectangle((0.3, 0.42), 0.28, 0.28, fc=mix(WHITE, SLATE, 0.45), ec="none"))
    label(ax, 0.66, 0.56, "重复计算的旧 token", size=10, ha="left", color=INK2)
    ax.add_patch(Rectangle((2.65, 0.42), 0.28, 0.28, fc=TEAL, ec="none"))
    label(ax, 3.01, 0.56, "真正需要的新位置", size=10, ha="left", color=INK2)
    if k >= n + 1:
        box(ax, 5.0, 1.25, 1.5, 1.45, mix(WHITE, CORAL, 0.12), ec=CORAL, lw=1.5)
        label(ax, 5.75, 2.35, "1+2+…+6", size=12, color=INK)
        label(ax, 5.75, 1.95, "= 21 次", size=14, color=CORAL, weight="bold")
        label(ax, 5.75, 1.52, "有用的只有 6 次", size=10, color=INK2)
        note = "生成 N 个 token：N(N+1)/2 ≈ N²/2 次 token 前向"
        label(ax, w / 2, 0.12, note, size=11, color=CORAL, weight="bold")
    return fig


def fig_recompute() -> None:
    for k in range(8):
        fig = _recompute_frame(k)
        save_frame(fig, "recompute", k)
        if k == 7:
            save(fig, "recompute")
        else:
            plt.close(fig)


PROMPT = ["我", "喜欢", "吃", "苹果"]
GEN = ["和", "香蕉", "。"]


def _kv_frame(k: int):
    """k = 0: before prefill; 1: after prefill; 2..4: decode steps 1..3; 5: summary."""
    w, h = 12.0, 4.5
    fig, ax = canvas(w, h)
    max_len = 8
    n_cached = 0 if k == 0 else min(4 + (k - 1), 7)
    step = k - 1  # decode step number when k >= 2
    if k == 5:
        n_cached = 7
    # phase title
    titles = {
        0: "提示 4 个 token，缓存为空",
        1: "预填充：整段提示一次送入，写入 4 个位置",
        2: "解码第 1 步：只送 1 个 token（位置 4），缓存 5 个位置",
        3: "解码第 2 步：只送 1 个 token（位置 5），缓存 6 个位置",
        4: "解码第 3 步：只送 1 个 token（位置 6），缓存 7 个位置",
        5: "旧 token 的 K、V 只写入一次，之后每步只读",
    }
    title_color = CORAL if k >= 2 else INK
    label(ax, 0.3, 4.2, titles[k], size=14, ha="left", weight="bold", color=title_color)
    # input tokens this step
    label(ax, 0.3, 3.55, "本步输入", size=11, ha="left", color=INK2)
    if k <= 1:
        feed = [(t, i) for i, t in enumerate(PROMPT)]
    elif k <= 4:
        feed = [(GEN[step - 1], 3 + step)]
    else:
        feed = []
        text = "之后每步也只送\n1 个新 token"
        label(ax, 0.3, 3.05, text, size=11, ha="left", color=SLATE, linespacing=1.6)
    for idx, (t, p) in enumerate(feed):
        x = 0.3 + idx * 0.62
        fc = TEAL if k <= 1 else CORAL
        box(ax, x, 2.75, 0.56, 0.6, fc, r=0.06)
        label(ax, x + 0.28, 3.12, t, size=12, color=WHITE, weight="bold")
        label(ax, x + 0.28, 2.88, f"pos {p}", size=7.5, color=WHITE)
    # model box
    box(ax, 3.0, 1.15, 2.3, 2.3, INK2, r=0.1)
    label(ax, 4.15, 2.95, "GPT 各层", size=13, color=WHITE, weight="bold")
    label(ax, 4.15, 2.5, "算 q、k、v", size=11, color=MID)
    nfed = len(feed)
    fed_text = f"穿过网络：{nfed} 个 token" if nfed else ""
    label(ax, 4.15, 1.6, fed_text, size=10.5, color=AMBER, weight="bold")
    if nfed:
        arrow(ax, 0.3 + nfed * 0.62 + 0.05, 3.05, 3.0, 2.6, color=SLATE, lw=1.5, ms=11)
    # cache grid
    cx0, cell = 6.35, 0.62
    label(ax, cx0, 3.62, "KV Cache（每层一份）", size=11, ha="left", color=INK2)
    words = PROMPT + GEN
    for row, (name, col) in enumerate([("K", TEAL), ("V", AMBER)]):
        y = 2.75 - row * 0.85
        label(ax, cx0 - 0.15, y + cell / 2, name, size=14, ha="right", weight="bold", color=col)
        for j in range(max_len):
            filled = j < n_cached
            fresh = (k == 1 and j < 4) or (2 <= k <= 4 and j == n_cached - 1)
            fc = mix(WHITE, col, 0.85 if fresh else 0.45) if filled else LIGHT
            ec = CORAL if fresh and k >= 2 else WHITE
            lw = 2.4 if fresh and k >= 2 else 1.5
            ax.add_patch(Rectangle((cx0 + j * cell, y), cell, cell, fc=fc, ec=ec, lw=lw, zorder=2))
            if filled:
                text = f"{name.lower()}{j}"
                label(ax, cx0 + (j + 0.5) * cell, y + cell / 2, text, size=10, color=INK, z=4)
    for j in range(max_len):
        label(ax, cx0 + (j + 0.5) * cell, 3.43, str(j), size=9, color=SLATE)
        if j < n_cached:
            label(ax, cx0 + (j + 0.5) * cell, 1.72, words[j], size=10, color=INK2)
    if nfed:
        arrow(ax, 5.3, 2.3, cx0 - 0.45, 2.3, color=SLATE, lw=1.5, ms=11)
        label(ax, 5.82, 2.5, "追加 k, v", size=9.5, color=INK2)
    # attention of the new query over the cache
    if 2 <= k <= 4:
        qx, qy = cx0 + (n_cached - 1 + 0.5) * cell, 0.35
        box(ax, qx - 0.3, qy - 0.2, 0.6, 0.4, CORAL, r=0.06, z=5)
        label(ax, qx, qy, "q", size=12, color=WHITE, weight="bold", z=6)
        for j in range(n_cached):
            a = FancyArrowPatch(
                (qx, qy + 0.2),
                (cx0 + (j + 0.5) * cell, 1.55),
                arrowstyle="-",
                color=mix(WHITE, CORAL, 0.55),
                lw=1.0,
                zorder=1,
                connectionstyle="arc3,rad=0.0",
            )
            ax.add_patch(a)
        msg = f"新 token 的 q 读取全部 {n_cached} 个 k、v（无需掩码）"
        label(ax, qx - 0.45, qy, msg, size=10, ha="right", color=CORAL)
    if k == 1:
        note = "普通因果掩码，并行计算 4 个位置，并预测出第一个新 token"
        label(ax, cx0, 0.4, note, size=10.5, ha="left", color=TEAL)
    if k == 5:
        note = "预填充 1 次 + 每步 1 个 token：总前向 ≈ N 次"
        label(ax, cx0, 0.4, note, size=11, ha="left", color=TEAL, weight="bold")
    return fig


def fig_kvdecode() -> None:
    for k in range(6):
        fig = _kv_frame(k)
        save_frame(fig, "kvdecode", k)
        if k == 5:
            save(fig, "kvdecode")
        else:
            plt.close(fig)


def fig_kv_masks() -> None:
    w, h = 5.0, 2.55
    fig, ax = canvas(w, h)
    cases = [
        ("T = L = 4\n普通因果掩码", 4, 4),
        ("T = 1, L = 5\n无需掩码", 1, 5),
        ("T = 3, L = 6\ntril(L−T)", 3, 6),
    ]
    x = 0.25
    cell = 0.27
    for title, T, L in cases:
        y0 = 1.9
        for i in range(T):
            for j in range(L):
                ok = j <= (L - T) + i
                fc = TEAL if ok else LIGHT
                if T < L and j < L - T and ok:
                    fc = mix(WHITE, TEAL, 0.55)
                xy = (x + j * cell, y0 - (i + 1) * cell)
                ax.add_patch(Rectangle(xy, cell, cell, fc=fc, ec=WHITE, lw=1.2))
        label(ax, x, 2.25, title, size=11, ha="left", va="center", color=INK)
        x += L * cell + 0.35
    note = "行：本次输入的 T 个查询\n列：缓存 + 新 token 共 L 个键；浅色 = 缓存部分"
    label(ax, 0.25, 0.32, note, size=10, ha="left", color=SLATE, linespacing=1.5)
    save(fig, "kv_masks")


def fig_kv_bench() -> None:
    b = json.loads((DATA_01 / "benchmark.json").read_text())
    runs = {r["model"]: r for r in b["runs"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.3), dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.15, wspace=0.3)
    panels = [(axes[0], "gpt2", "gpt2（124M）"), (axes[1], "gpt2-xl", "gpt2-xl（1.5B）")]
    for ax, name, label_ in panels:
        r = runs[name]
        steps = np.arange(1, r["n_steps"] + 1)
        ax.plot(steps, r["uncached_step_ms"], color=CORAL, lw=1.2, label="无缓存")
        ax.plot(steps, r["cached_step_ms"], color=TEAL, lw=1.6, label="KV Cache")
        style_axes(ax)
        ax.set_xlabel("生成到第几个 token", fontsize=10)
        times = f"{r['uncached_total_s']:.1f}s → {r['cached_total_s']:.1f}s"
        ax.set_title(f"{label_}：{times}（{r['speedup']}×）", fontsize=10.5, loc="left")
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    axes[0].set_ylabel("单步耗时（毫秒）", fontsize=10)
    note = f"实测：{b['device']}，提示 8 token，贪心生成 1000 token，输出完全一致"
    fig.text(0.1, 0.94, note, fontsize=9, color=SLATE)
    save(fig, "kv_bench")


def fig_kv_memory() -> None:
    items = [
        ("GPT-2 small\nfp32 · L=1024", 2 * 12 * 1024 * 768 * 4, TEAL),
        ("gpt2-xl\nfp16 · L=1024", 2 * 48 * 1024 * 1600 * 2, INDIGO),
        ("GPT-3 175B\nfp16 · L=2048", 2 * 96 * 2048 * 12288 * 2, CORAL),
    ]
    fig = plt.figure(figsize=(5.4, 3.7), dpi=DPI)
    ax = fig.add_axes((0.3, 0.1, 0.62, 0.76))
    y = np.arange(len(items))[::-1]
    vals = [v / 2**20 for _, v, _ in items]
    ax.barh(y, vals, color=[c for _, _, c in items], height=0.55, zorder=2)
    for yi, v in zip(y, vals, strict=True):
        s = f"{v:.0f} MiB" if v < 1024 else f"{v / 1024:.1f} GiB"
        ax.text(v * 1.15, yi, s, va="center", fontsize=12.5, color=INK, weight="bold")
    ax.set_xscale("log")
    ax.set_xlim(10, 2e5)
    ax.set_yticks(y, [n for n, _, _ in items], fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("单条满长度序列的 KV Cache（对数轴）", fontsize=12, loc="left")
    save(fig, "kv_memory")


def main() -> None:
    fig_lm_next()
    fig_rnn_vs_attn()
    fig_qkv()
    fig_sdpa()
    fig_scaling()
    fig_multihead()
    fig_transformer()
    fig_posenc()
    fig_causal()
    fig_causal_layers()
    fig_real_attention()
    fig_gpt2_arch()
    fig_gpt2_lm()
    fig_gpt3_data()
    fig_gpt3_results()
    fig_gpt3_arith()
    fig_gpt3_scaling()
    fig_recompute()
    fig_kvdecode()
    fig_kv_masks()
    fig_kv_bench()
    fig_kv_memory()
    (FIG_DIR / "manifest.json").write_text(json.dumps(MANIFEST, indent=1))
    print(f"wrote {len(MANIFEST)} figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
