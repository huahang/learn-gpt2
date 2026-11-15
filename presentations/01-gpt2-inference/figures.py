"""Draw the deck's diagrams and the video's animation frames with matplotlib.

python figures.py            # slide figures into build/figures/
python figures.py --frames   # also animation frames into build/frames/
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = Path(__file__).parent
FIG_DIR = HERE / "build" / "figures"
FRAME_DIR = HERE / "build" / "frames"

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

plt.rcParams.update(
    {
        "font.family": ["Noto Sans CJK SC", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": False,
        "text.color": INK,
        "savefig.facecolor": WHITE,
    }
)
MONO = "DejaVu Sans Mono"


def new_canvas(w: float, h: float, xlim: float, ylim: float):
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=200)
    plt.close(fig)


def box(ax, x, y, w, h, fc, ec=None, lw=0, r=0.08, alpha=1.0, z=2, ls="-"):
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
        linestyle=ls,
    )
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=SLATE, lw=2.0, style="-|>", ls="-", z=1, ms=14):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            mutation_scale=ms,
            color=color,
            lw=lw,
            linestyle=ls,
            zorder=z,
            shrinkA=0,
            shrinkB=0,
        )
    )


def text(ax, x, y, s, size=13, color=INK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5, **kw)


# --------------------------------------------------------------------------------------------
# Slide figures
# --------------------------------------------------------------------------------------------


def fig_qkv() -> None:
    fig, ax = new_canvas(7.6, 4.3, 15.2, 8.6)
    # x: (T, 768)
    box(ax, 0.3, 2.4, 1.6, 3.8, LIGHT, MID, 1.5)
    text(ax, 1.1, 4.3, "x", 20, weight="bold")
    text(ax, 1.1, 1.9, "(B, T, 768)", 11, SLATE, family=MONO)
    arrow(ax, 2.1, 4.3, 3.6, 4.3, INK2)
    text(ax, 2.85, 5.0, "c_attn", 11, INK2, family=MONO)
    text(ax, 2.85, 3.6, "768→2304", 10, SLATE, family=MONO)
    # qkv: (T, 2304) split into three
    for i, (name, c) in enumerate((("q", INDIGO), ("k", TEAL), ("v", AMBER))):
        box(ax, 3.8 + i * 1.25, 2.4, 1.15, 3.8, c, r=0.06)
        text(ax, 4.375 + i * 1.25, 4.3, name, 20, WHITE, "bold")
        text(ax, 4.375 + i * 1.25, 2.05, "768", 10, SLATE, family=MONO)
    text(ax, 5.6, 6.65, ".split(768, dim=2)", 11, INK2, family=MONO)
    arrow(ax, 7.75, 4.3, 9.0, 4.3, INK2)
    text(ax, 8.4, 5.0, "view +", 10, INK2, family=MONO)
    text(ax, 8.4, 3.6, "transpose", 10, INK2, family=MONO)
    # heads: 12 stacked cards per tensor
    for i, (name, c) in enumerate((("q", INDIGO), ("k", TEAL), ("v", AMBER))):
        x0, y0 = 9.3 + i * 1.95, 2.2
        for j in range(5, -1, -1):
            box(
                ax,
                x0 + j * 0.12,
                y0 + j * 0.12,
                1.2,
                3.6,
                c,
                WHITE,
                1.2,
                r=0.05,
                alpha=0.55 + 0.075 * (5 - j),
            )
        text(ax, x0 + 0.6, y0 + 1.8, name, 18, WHITE, "bold")
    text(ax, 12.1, 1.55, "(B, n_head=12, T, hs=64)", 11, SLATE, family=MONO)
    text(ax, 12.1, 7.35, "12 个头，每头 64 维", 12, INK2)
    save(fig, "qkv.png")


def fig_attention() -> None:
    data = json.loads((HERE / "data" / "attention.json").read_text())
    tokens = [t.replace(" ", "␣") for t in data["tokens"]]
    m = data["maps"][0][7]
    T = len(tokens)
    fig = plt.figure(figsize=(5.6, 5.4), dpi=200)
    ax = fig.add_axes((0.2, 0.03, 0.75, 0.72))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("t", [WHITE, TEAL, INK2])
    for i in range(T):
        for j in range(T):
            if j > i:
                ax.add_patch(Rectangle((j, i), 1, 1, fc=LIGHT, ec=WHITE, lw=1.5))
                text(ax, j + 0.5, i + 0.5, "×", 10, MID)
            else:
                p = m[i][j]
                ax.add_patch(Rectangle((j, i), 1, 1, fc=cmap(min(1.0, p * 1.4)), ec=WHITE, lw=1.5))
                text(ax, j + 0.5, i + 0.5, f"{p:.2f}", 8.5, WHITE if p > 0.35 else INK)
    ax.set_xlim(0, T)
    ax.set_ylim(T, 0)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(T)], tokens, rotation=45, ha="right", fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(T)], tokens, fontsize=10)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Key：被看的位置 j", fontsize=11, color=SLATE)
    ax.set_ylabel("Query：正在计算的位置 i", fontsize=11, color=SLATE)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
    save(fig, "attention.png")


def draw_grid(ax, x0, y0, n_steps, prompt_len, mode, upto=None, cell=0.62, labels=True, fs=10):
    """Rows are decoding steps, columns positions; colour shows what each step computes."""
    upto = n_steps if upto is None else upto
    n_pos = prompt_len + n_steps - 1
    for s in range(n_steps):
        y = y0 - s * cell
        t = prompt_len + s  # tokens visible at this step
        if labels:
            text(ax, x0 - 0.35, y + cell / 2, f"第{s + 1}步", fs, SLATE, ha="right")
        for p in range(n_pos):
            x = x0 + p * cell
            if s >= upto or p >= t:
                ax.add_patch(Rectangle((x, y), cell, cell, fc=WHITE, ec=LIGHT, lw=1))
                continue
            first = s == 0 and mode == "cached"  # prefill computes the whole prompt once
            if mode == "uncached":
                fc = AMBER if p == t - 1 else (CORAL if s > 0 else AMBER)
            else:
                fc = AMBER if (p == t - 1 or first) else WHITE
            ec = TEAL if fc == WHITE else WHITE
            ax.add_patch(Rectangle((x, y), cell, cell, fc=fc, ec=ec, lw=1.6 if fc == WHITE else 1))
    if labels:
        for p in range(n_pos):
            text(ax, x0 + (p + 0.5) * cell, y0 + cell + 0.25, str(p), fs - 1, SLATE)


def grid_count(n_steps, prompt_len, mode, upto):
    if mode == "uncached":
        return sum(prompt_len + s for s in range(upto))
    return prompt_len + max(0, upto - 1) if upto else 0


def fig_nocache_grid() -> None:
    fig, ax = new_canvas(6.6, 4.8, 13.2, 9.6)
    draw_grid(ax, 1.6, 7.9, 10, 3, "uncached", cell=0.75)
    text(ax, 1.6 + 6.0 * 0.75, 9.25, "位置", 10, SLATE)
    text(ax, 12.15, 9.25, "本步计算", 10, SLATE)
    for s in range(10):
        text(ax, 12.15, 7.9 - s * 0.75 + 0.375, f"{3 + s} 个", 10.5, CORAL if s else AMBER, "bold")
    for i, (c, label) in enumerate(((AMBER, "本步新 token"), (CORAL, "重复计算的旧 token"))):
        box(ax, 1.6 + i * 4.6, 0.15, 0.45, 0.45, c, r=0.05)
        text(ax, 2.25 + i * 4.6, 0.375, label, 11, INK, ha="left")
    save(fig, "nocache_grid.png")


def fig_dependency() -> None:
    fig, ax = new_canvas(6.6, 4.8, 13.2, 9.6)
    cols, rows = 6, 4
    xs = [2.6 + i * 1.95 for i in range(cols)]
    ys = [1.5 + j * 1.95 for j in range(rows)]
    for j, name in enumerate(["Embedding", "Block 1", "Block 2", "Block 3"]):
        text(ax, 1.95, ys[j], name, 10, SLATE, ha="right")
    # Each node reads every node at positions <= i in the layer below (causal attention).
    for j in range(1, rows):
        for i in range(cols):
            for k in range(i + 1):
                new = i == cols - 1
                arrow(
                    ax,
                    xs[k],
                    ys[j - 1] + 0.3,
                    xs[i],
                    ys[j] - 0.3,
                    AMBER if new else MID,
                    1.4 if new else 0.8,
                    "-",
                    z=1,
                )
    for j in range(rows):
        for i in range(cols):
            c = AMBER if i == cols - 1 else TEAL
            ax.add_patch(
                matplotlib.patches.Circle((xs[i], ys[j]), 0.3, fc=c, ec=WHITE, lw=2, zorder=3)
            )
    for i in range(cols):
        new = i == cols - 1
        text(ax, xs[i], 0.55, f"位置 {i}", 10, AMBER if new else SLATE, "bold" if new else "normal")
    # The dependency that does not exist: the new token never feeds an older position.
    ax.add_patch(
        FancyArrowPatch(
            (xs[5], ys[3] + 0.38),
            (xs[3], ys[3] + 0.38),
            connectionstyle="arc3,rad=0.55",
            arrowstyle="-|>",
            mutation_scale=14,
            color=CORAL,
            lw=2.2,
            linestyle="--",
            zorder=4,
        )
    )
    text(ax, xs[4], ys[3] + 1.12, "✗", 20, CORAL, "bold")
    text(ax, 7.4, 9.25, "新 token 只读旧位置的 K、V，绝不会反过来改变它们", 11.5, INK, "bold")
    save(fig, "dependency.png")


def matrix(
    ax, x, y, rows, cols, cw, ch, fc, last_row=None, last_col=None, label=None, ec=WHITE, fs=12
):
    for r in range(rows):
        for c in range(cols):
            col = fc
            if last_row is not None and r == rows - 1:
                col = last_row
            if last_col is not None and c == cols - 1:
                col = last_col
            ax.add_patch(
                Rectangle((x + c * cw, y - (r + 1) * ch), cw, ch, fc=col, ec=ec, lw=1.2, zorder=2)
            )
    if label:
        text(ax, x + cols * cw / 2, y + 0.35, label, fs, INK, "bold")


def fig_kv_phases() -> None:
    fig, ax = new_canvas(11.6, 3.9, 23.2, 7.8)
    # Prefill panel
    text(ax, 5.4, 7.3, "Prefill：一次并行处理整段 prompt", 14, INK, "bold")
    toks = ["Hello", ",", "␣I", "'m", "␣a"]
    for i, t in enumerate(toks):
        box(ax, 0.4, 5.6 - i * 0.95, 1.6, 0.75, LIGHT, MID, 1.2, r=0.15)
        text(ax, 1.2, 5.975 - i * 0.95, t, 11)
    arrow(ax, 2.3, 3.8, 3.7, 3.8, INK2)
    text(ax, 3.0, 4.35, "12 层前向", 10, SLATE)
    matrix(ax, 4.1, 6.4, 5, 5, 0.55, 0.95, TEAL, label="K 缓存")
    matrix(ax, 7.6, 6.4, 5, 5, 0.55, 0.95, TEAL, label="V 缓存")
    text(ax, 5.4, 0.9, "每层、每个头都存一份  (B, n_head, L, hs)", 11, SLATE)
    ax.plot([11.3, 11.3], [0.6, 7.5], color=MID, lw=1.5)
    # Decode panel
    text(ax, 17.3, 7.3, "Decode：每步只算 1 个新 token", 14, INK, "bold")
    box(ax, 12.0, 3.4, 1.6, 0.75, AMBER, r=0.15)
    text(ax, 12.8, 3.775, "␣language", 10, WHITE, "bold")
    arrow(ax, 13.8, 3.8, 15.0, 3.8, INK2)
    text(ax, 14.4, 4.35, "只算 1 行", 10, SLATE)
    matrix(ax, 15.3, 6.4, 6, 5, 0.55, 0.82, TEAL, last_row=AMBER, label="K 缓存 +1 行")
    matrix(ax, 18.8, 6.4, 6, 5, 0.55, 0.82, TEAL, last_row=AMBER, label="V 缓存 +1 行")
    text(ax, 17.3, 0.9, "新 token 的 q 与全部缓存的 K、V 做注意力", 11, SLATE)
    save(fig, "kv_phases.png")


def fig_decode_step() -> None:
    fig, ax = new_canvas(11.6, 3.5, 23.2, 7.0)
    cw, ch = 0.42, 0.42
    y = 5.4
    # q_t (1 x d)
    matrix(ax, 0.3, y - 1.4, 1, 6, cw, ch, AMBER, label="q  (1×hs)")
    text(ax, 3.3, y - 1.6, "×", 20, INK2, "bold")
    # K^T (d x t)
    matrix(ax, 3.8, y, 6, 7, cw, ch, TEAL, last_col=AMBER, label="Kᵀ  (hs×t)")
    text(ax, 7.1, y - 1.6, "=", 20, INK2, "bold")
    matrix(ax, 7.6, y - 1.4, 1, 7, cw, ch, INDIGO, label="分数 (1×t)")
    arrow(ax, 10.7, y - 1.6, 11.9, y - 1.6, INK2)
    text(ax, 11.3, y - 0.95, "softmax", 10, SLATE, family=MONO)
    matrix(ax, 12.2, y - 1.4, 1, 7, cw, ch, "#8E97FF", label="权重 (1×t)")
    text(ax, 15.5, y - 1.6, "×", 20, INK2, "bold")
    matrix(ax, 16.0, y + 0.4, 7, 6, cw, ch, TEAL, last_row=AMBER, label="V  (t×hs)")
    text(ax, 18.9, y - 1.6, "=", 20, INK2, "bold")
    matrix(ax, 19.4, y - 1.4, 1, 6, cw, ch, AMBER, label="输出 (1×hs)")
    text(
        ax,
        11.6,
        0.75,
        "只有 1 个 query：每步注意力 ∝ t·d，而不是 t²·d；琥珀色是本步新增的 k、v",
        12,
        INK2,
    )
    save(fig, "decode_step.png")


def fig_buffer() -> None:
    fig, ax = new_canvas(6.4, 3.6, 12.8, 7.2)
    n, filled, new = 14, 7, 2
    cw = 0.76
    x0, y0 = 0.6, 2.4
    for i in range(n):
        if i < filled:
            fc, ec, ls = TEAL, WHITE, "-"
        elif i < filled + new:
            fc, ec, ls = AMBER, WHITE, "-"
        else:
            fc, ec, ls = WHITE, SLATE, "--"
        ax.add_patch(Rectangle((x0 + i * cw, y0), cw, 1.4, fc=fc, ec=ec, lw=1.2, ls=ls, zorder=2))

    def tick(i, label, color, up=True):
        x = x0 + i * cw
        ax.plot([x, x], [y0 - 0.15, y0 + 1.55], color=color, lw=2.2, zorder=3)
        text(ax, x, y0 - 0.6 if up else y0 + 2.0, label, 10.5, color, "bold", family=MONO)

    tick(filled, "length", TEAL)
    tick(filled + new, "end", AMBER, up=False)
    tick(n, "max_len", SLATE)
    text(ax, x0 + filled * cw / 2, y0 + 0.7, "已缓存", 12, WHITE, "bold")
    text(ax, x0 + (filled + new / 2) * cw, y0 + 0.7, "新", 12, WHITE, "bold")
    text(ax, x0 + (filled + new + (n - filled - new) / 2) * cw, y0 + 0.7, "预留空位", 11, SLATE)
    text(ax, 6.4, 6.4, "self.k[:, :, length:end] = k", 13, INK2, family=MONO)
    text(ax, 6.4, 0.75, "return self.k[:, :, :end], self.v[:, :, :end]", 11.5, SLATE, family=MONO)
    save(fig, "buffer.png")


def mask_panel(ax, x0, y0, T, L, cell, title, sub, highlight=True):
    for i in range(T):
        for j in range(L):
            allowed = j <= (L - T) + i
            fc = TEAL if allowed else LIGHT
            ax.add_patch(
                Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, fc=fc, ec=WHITE, lw=1.5)
            )
    # mark the new keys (the T newest positions)
    ax.add_patch(
        Rectangle(
            (x0 + (L - T) * cell, y0 - T * cell),
            T * cell,
            T * cell,
            fill=False,
            ec=AMBER,
            lw=2.4,
            zorder=4,
        )
    )
    text(ax, x0 + L * cell / 2, y0 + 0.85, title, 13, INK, "bold")
    font = {"family": MONO} if sub.isascii() else {}
    text(ax, x0 + L * cell / 2, y0 + 0.35, sub, 10.5, SLATE, **font)


def fig_masks() -> None:
    fig, ax = new_canvas(11.6, 3.6, 23.2, 7.2)
    c = 0.62
    mask_panel(ax, 0.6, 5.6, 6, 6, c, "① 预填充 T = L", "is_causal=True")
    mask_panel(ax, 8.0, 5.6, 1, 8, c, "② 解码一步 T = 1", "不需要掩码")
    mask_panel(ax, 15.6, 5.6, 3, 9, c, "③ 带缓存的多 token  T < L", "tril(diagonal=L-T)")
    text(ax, 8.0 + 4 * c, 3.9, "新 token 可以看全部缓存", 11, SLATE)
    text(ax, 15.6 + 4.5 * c, 3.1, "第 i 个新 token 位于 L-T+i", 11, SLATE)
    text(ax, 15.6 + 4.5 * c, 2.5, "左上角对齐的 is_causal 会出错", 11, CORAL)
    for i, (col, label) in enumerate(((TEAL, "可以看"), (LIGHT, "被掩盖"))):
        ax.add_patch(Rectangle((0.6 + i * 3.2, 0.5), 0.5, 0.5, fc=col, ec=MID, lw=1))
        text(ax, 1.3 + i * 3.2, 0.75, label, 11, INK, ha="left")
    ax.add_patch(Rectangle((7.0, 0.5), 0.5, 0.5, fill=False, ec=AMBER, lw=2.4))
    text(ax, 7.7, 0.75, "本次输入的新 token", 11, INK, ha="left")
    save(fig, "masks.png")


def chip(ax, x, y, label, fc, w=1.15, h=0.8, fs=10.5, tc=WHITE):
    box(ax, x, y, w, h, fc, r=0.18)
    text(ax, x + w / 2, y + h / 2, label, fs, tc, "bold")


def fig_positions() -> None:
    fig, ax = new_canvas(11.6, 3.6, 23.2, 7.2)
    text(ax, 0.3, 6.75, "正常情况：新 token 接在缓存后面", 15, INK, "bold", ha="left")
    for i in range(6):
        chip(ax, 0.3 + i * 1.32, 5.1, f"pos {i}", TEAL, w=1.2, h=0.9, fs=12)
    for i in range(2):
        chip(ax, 0.3 + (6 + i) * 1.32, 5.1, f"pos {6 + i}", AMBER, w=1.2, h=0.9, fs=12)
    text(ax, 0.3, 4.35, "start = kv_cache.length = 6", 13, INK2, family=MONO, ha="left")
    text(ax, 0.3, 3.45, "pos = arange(start, start + T)", 13, INK2, family=MONO, ha="left")
    text(ax, 0.3, 2.55, "位置编号连续，旧 K、V 依然有效", 13, TEAL, "bold", ha="left")
    ax.plot([11.3, 11.3], [0.4, 7.0], color=MID, lw=1.5)
    text(ax, 11.8, 6.75, "超过 block_size：窗口滑动，位置全部改变", 15, INK, "bold", ha="left")
    toks = "ABCDEFGH"
    for i in range(8):
        chip(ax, 11.8 + i * 1.4, 5.1, toks[i], TEAL, w=1.25, h=0.9, fs=13)
        text(ax, 11.8 + i * 1.4 + 0.625, 4.7, f"pos {i}", 10.5, SLATE)
    for i in range(8):
        chip(
            ax,
            11.8 + i * 1.4,
            2.75,
            toks[i + 1] if i < 7 else "I",
            CORAL if i < 7 else AMBER,
            w=1.25,
            h=0.9,
            fs=13,
        )
        text(ax, 11.8 + i * 1.4 + 0.625, 2.35, f"pos {i}", 10.5, SLATE)
    arrow(ax, 17.3, 4.35, 17.3, 3.75, INK2)
    text(ax, 17.6, 4.05, "裁掉最早的 A", 12, INK2, ha="left")
    text(ax, 17.3, 1.45, "B 原来在 pos 1，现在变成 pos 0：缓存里的 K、V 用的是旧位置", 12, CORAL)
    text(ax, 17.3, 0.75, "→ generate() 丢弃缓存，回到完整重算", 12, INK2, "bold")
    save(fig, "positions.png")


# --------------------------------------------------------------------------------------------
# Video animation frames (1920 x 1080)
# --------------------------------------------------------------------------------------------


def frame_canvas():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.axis("off")
    return fig, ax


def frames_compare(n_steps=10, prompt_len=3) -> int:
    out = FRAME_DIR / "compare"
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for upto in range(0, n_steps + 1):
        fig, ax = frame_canvas()
        text(
            ax,
            0.9,
            9.9,
            "动画：有无 KV Cache，每一步要算多少个 token？",
            30,
            INK,
            "bold",
            ha="left",
        )
        for mode, x0, title, color in (
            ("uncached", 1.6, "不使用缓存：每步重算整个前缀", CORAL),
            ("cached", 11.2, "使用 KV Cache：每步只算新 token", TEAL),
        ):
            text(ax, x0 + 3.6, 8.75, title, 20, color, "bold")
            draw_grid(ax, x0, 7.55, n_steps, prompt_len, mode, upto=upto, cell=0.58, fs=16)
            total = grid_count(n_steps, prompt_len, mode, upto)
            text(ax, x0 + 3.6, 1.05, f"累计计算 token 数：{total}", 22, color, "bold")
        for i, (c, label) in enumerate(
            ((AMBER, "本步计算"), (CORAL, "重复计算"), (WHITE, "从缓存读取 K、V"))
        ):
            ax.add_patch(
                Rectangle(
                    (1.6 + i * 3.4, 0.2), 0.35, 0.35, fc=c, ec=TEAL if c == WHITE else c, lw=1.6
                )
            )
            text(ax, 2.1 + i * 3.4, 0.375, label, 14, INK, ha="left")
        fig.savefig(out / f"{k:03d}.png", dpi=100)
        plt.close(fig)
        k += 1
    return k


def frames_decode(n_steps=6) -> int:
    out = FRAME_DIR / "decode"
    out.mkdir(parents=True, exist_ok=True)
    words = ["Hello", ",", "␣I", "'m", "␣a", "␣language", "␣model", ",", "␣not", "␣a"]
    for s in range(n_steps + 1):
        fig, ax = frame_canvas()
        text(ax, 0.9, 9.9, "动画：解码时 KV Cache 一行一行地增长", 30, INK, "bold", ha="left")
        L = 4 + s
        for i in range(L):
            fc = AMBER if (i == L - 1 and s > 0) else TEAL if i < L else LIGHT
            chip(ax, 0.9 + i * 1.75, 8.1, words[i], fc, w=1.6, h=0.85, fs=16)
        cw, ch = 0.5, 0.5
        matrix(
            ax,
            2.0,
            6.6,
            L,
            8,
            cw,
            ch,
            TEAL,
            last_row=AMBER if s > 0 else None,
            label="K 缓存",
            fs=22,
        )
        matrix(
            ax,
            7.4,
            6.6,
            L,
            8,
            cw,
            ch,
            TEAL,
            last_row=AMBER if s > 0 else None,
            label="V 缓存",
            fs=22,
        )
        text(ax, 4.0, 6.6 - L * ch - 0.5, f"{L} 行", 22, SLATE)
        text(ax, 9.4, 6.6 - L * ch - 0.5, f"{L} 行", 22, SLATE)
        x = 12.4
        lines = [
            ("本步输入", "1 个 token" if s > 0 else f"{L} 个 token（prefill）"),
            ("新算的 k、v", "1 行" if s > 0 else f"{L} 行"),
            ("注意力", f"1 × {L} 个分数" if s > 0 else f"{L} × {L} 个分数"),
            ("线性层计算", "1 个 token" if s > 0 else f"{L} 个 token"),
        ]
        for i, (a, b) in enumerate(lines):
            text(ax, x, 6.2 - i * 1.05, a, 24, SLATE, ha="left")
            text(ax, x + 2.6, 6.2 - i * 1.05, b, 24, AMBER if s > 0 else TEAL, "bold", ha="left")
        text(ax, x, 1.6, f"第 {s} 步" if s > 0 else "Prefill", 36, INK, "bold", ha="left")
        fig.savefig(out / f"{s:03d}.png", dpi=100)
        plt.close(fig)
    return n_steps + 1


def main() -> None:
    fig_qkv()
    fig_attention()
    fig_nocache_grid()
    fig_dependency()
    fig_kv_phases()
    fig_decode_step()
    fig_buffer()
    fig_masks()
    fig_positions()
    print("figures:", sorted(p.name for p in FIG_DIR.glob("*.png")))
    if "--frames" in sys.argv:
        print("compare frames:", frames_compare(), "decode frames:", frames_decode())


if __name__ == "__main__":
    main()
