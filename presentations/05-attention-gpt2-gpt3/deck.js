// Build attention_gpt2_gpt3.pptx: Transformer, causal self-attention, GPT-2, GPT-3 and the KV cache.
//
//   NODE_PATH=<dir with pptxgenjs>/node_modules node deck.js
//
// Reads narration.json (speaker notes), build/deck_data.json (prepare.py) and
// build/figures/*.png + manifest.json (figures.py). Writes build/anim_boxes.json: where each
// animated figure sits on its slide, so video.py can composite the animation frames there.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const BUILD = path.join(HERE, "build");
const NARR = JSON.parse(fs.readFileSync(path.join(HERE, "narration.json"), "utf8"));
const DATA = JSON.parse(fs.readFileSync(path.join(BUILD, "deck_data.json"), "utf8"));
const SIZES = JSON.parse(fs.readFileSync(path.join(BUILD, "figures", "manifest.json"), "utf8"));
const OUT = path.join(HERE, "attention_gpt2_gpt3.pptx");

const INK = "1B1F3B", WHITE = "FFFFFF", INK2 = "2D3250", LIGHT = "F2F4F8", MID = "D9DEE8";
const TEAL = "2A9D8F", AMBER = "F2A541", CORAL = "E76F51", INDIGO = "5B6CFF", SLATE = "8D99AE";
const DEEP = "264653";
const FONT = "Microsoft YaHei", CODE_FONT = "Consolas";
const DECK_TITLE = "注意力、GPT-2 与 GPT-3";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = DECK_TITLE;
pres.author = "learn-gpt2";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

const W = 13.333, M = 0.6, CW = W - 2 * M; // content width 12.133

// ------------------------------------------------------------------------------------------
// Layouts
// ------------------------------------------------------------------------------------------
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.6, w: CW, h: 1.3, fontSize: 46, bold: true, color: WHITE, valign: "bottom", align: "left", margin: 0, fontFace: FONT }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.1, w: CW, h: 0.8, fontSize: 22, color: SLATE, valign: "top", margin: 0, fontFace: FONT }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "num", type: "body", x: M, y: 1.45, w: 4, h: 1.4, fontSize: 80, bold: true, color: AMBER, valign: "bottom", margin: 0, fontFace: FONT }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.95, w: CW, h: 1.1, fontSize: 44, bold: true, color: WHITE, valign: "middle", align: "left", margin: 0, fontFace: FONT }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.15, w: CW, h: 0.9, fontSize: 20, color: SLATE, valign: "top", margin: 0, fontFace: FONT }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.38, w: CW, h: 0.85, fontSize: 30, bold: true, color: INK, valign: "middle", align: "left", margin: 0, fontFace: FONT }, text: "" } },
    { text: { text: DECK_TITLE, options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: SLATE, margin: 0, fontFace: FONT } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: SLATE, align: "right", fontFace: FONT },
});

// ------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------
let slideCount = 0;
const animBoxes = {};
function slideOf(masterName, id) {
  const entry = NARR[slideCount];
  if (!entry || entry.id !== id) throw new Error(`slide ${slideCount + 1}: expected ${entry && entry.id}, got ${id}`);
  slideCount += 1;
  const s = pres.addSlide({ masterName });
  s.addNotes(entry.text);
  return s;
}
function title(s, text) {
  s.addText(text, { placeholder: "title" });
}
function txt(s, text, o) {
  s.addText(text, { isTextBox: true, margin: 0, fontSize: 16, color: INK, valign: "top", fontFace: FONT, ...o });
}
function rect(s, x, y, w, h, fill, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: o.r === undefined ? 0.12 : o.r,
    fill: { color: fill, ...(o.fillT !== undefined ? { transparency: o.fillT } : {}) },
    line: o.line ? { color: o.line, width: o.lineW || 1 } : { type: "none" },
  });
}
function line(s, x1, y1, x2, y2, color = SLATE, o = {}) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color, width: o.width || 2, ...(o.arrow ? { endArrowType: "triangle" } : {}), ...(o.dash ? { dashType: o.dash } : {}) },
  });
}
function circleNum(s, x, y, d, label, fill, color = WHITE, size = 16) {
  s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" } });
  txt(s, label, { x, y, w: d, h: d, fontSize: size, bold: true, color, align: "center", valign: "middle" });
}
// Width of a string in inches: CJK glyphs are 1 em, the rest about 0.56 em.
function estW(text, pt) {
  let em = 0;
  for (const ch of text) em += /[⺀-￯]/.test(ch) ? 1 : 0.56;
  return (em * pt) / 72;
}
function chip(s, x, y, text, fill, o = {}) {
  const pt = o.pt || 16;
  const w = o.w || Math.max(0.7, estW(text, pt) + 0.4);
  const h = o.h || 0.6;
  rect(s, x, y, w, h, fill, { line: o.line, fillT: o.fillT });
  txt(s, text, { x, y, w, h, fontSize: pt, bold: o.bold !== false, color: o.color || WHITE, align: "center", valign: "middle", fontFace: o.mono ? CODE_FONT : FONT });
  return w;
}
function card(s, x, y, w, h, head, body, accent, o = {}) {
  rect(s, x, y, w, h, o.fill || LIGHT, { fillT: o.fillT });
  if (o.bar !== false) s.addShape(pres.shapes.RECTANGLE, { x, y: y + 0.12, w: 0.08, h: h - 0.24, fill: { color: accent }, line: { type: "none" } });
  const pad = 0.25;
  let tx = x + pad + 0.05;
  if (o.num) {
    circleNum(s, tx, y + pad - 0.02, 0.46, o.num, accent, WHITE, 14);
    tx += 0.62;
  }
  const headH = o.headH || 0.46;
  txt(s, head, { x: tx, y: y + pad - 0.02, w: x + w - pad - tx, h: headH, fontSize: o.headPt || 17, bold: true, color: o.headColor || INK, valign: "middle" });
  if (body) {
    txt(s, body, {
      x: o.num ? tx : x + pad + 0.05, y: y + pad + headH + 0.06, w: x + w - pad - (o.num ? tx : x + pad + 0.05),
      h: h - pad * 2 - headH - 0.06, fontSize: o.bodyPt || 14, color: o.bodyColor || INK2, paraSpaceAfter: 4,
    });
  }
}
function stat(s, x, y, w, big, label, color, o = {}) {
  txt(s, big, { x, y, w, h: o.bigH || 0.8, fontSize: o.bigPt || 38, bold: true, color, valign: "bottom" });
  txt(s, label, { x, y: y + (o.bigH || 0.8) + 0.05, w, h: o.labelH || 0.6, fontSize: o.labelPt || 13, color: INK2 });
}
// Place a figure inside a box, keeping its aspect ratio. align: "c" | "l" | "r" | "t".
function figure(s, name, x, y, maxW, maxH, o = {}) {
  const [fw, fh] = SIZES[name];
  const k = Math.min(maxW / fw, maxH / fh);
  const w = fw * k, h = fh * k;
  let fx = x + (maxW - w) / 2, fy = y + (maxH - h) / 2;
  if (o.alignX === "l") fx = x;
  if (o.alignX === "r") fx = x + maxW - w;
  if (o.alignY === "t") fy = y;
  s.addImage({ path: path.join(BUILD, "figures", `${name}.png`), x: fx, y: fy, w, h, altText: o.alt || name });
  if (o.anim) animBoxes[slideCount] = { name: o.anim, x: fx, y: fy, w, h };
  return { x: fx, y: fy, w, h };
}
function bullets(s, items, o) {
  const runs = [];
  items.forEach((it, i) => {
    const parts = Array.isArray(it) ? it : [{ text: it }];
    parts.forEach((p, j) => {
      runs.push({
        text: p.text,
        options: {
          // pptxgenjs 会在每个带 bullet 的 run 处另起段落, 所以只给首个 run 加
          ...(j === 0 ? { bullet: { indent: 18 } } : {}),
          ...(p.bold ? { bold: true } : {}),
          ...(p.color ? { color: p.color } : {}),
          ...(p.mono ? { fontFace: CODE_FONT } : {}),
          breakLine: j === parts.length - 1 && i < items.length - 1,
        },
      });
    });
  });
  txt(s, runs, { paraSpaceAfter: 8, ...o });
}
function codePanel(s, key, x, y, w, h, pt = 12) {
  rect(s, x, y, w, h, INK);
  const runs = [];
  const lines = DATA.code[key];
  lines.forEach((ln, i) => {
    const merged = [];
    for (const r of ln) {
      const last = merged[merged.length - 1];
      if (last && last.color === r.color) last.text += r.text;
      else merged.push({ ...r });
    }
    if (merged.length === 0) merged.push({ text: " ", color: "E6E9F2" });
    merged.forEach((r, j) => {
      runs.push({ text: r.text, options: { color: r.color, fontFace: CODE_FONT, fontSize: pt, breakLine: j === merged.length - 1 && i < lines.length - 1 } });
    });
  });
  s.addText(runs, { isTextBox: true, x: x + 0.22, y: y + 0.18, w: w - 0.44, h: h - 0.36, margin: 0, valign: "top", fontFace: CODE_FONT, fontSize: pt, lineSpacingMultiple: 1.1 });
}
function table(s, head, rows, o) {
  const hdr = head.map((t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: INK } } }));
  const body = rows.map((r, i) =>
    r.map((t) => {
      const hl = o.highlight && o.highlight.includes(i);
      return { text: String(t), options: { fill: { color: hl ? "FCEBD2" : i % 2 ? LIGHT : WHITE }, bold: hl, color: INK } };
    }),
  );
  s.addTable([hdr, ...body], {
    fontFace: FONT, fontSize: 14, align: "center", valign: "middle",
    border: { type: "solid", pt: 1, color: MID }, ...o, highlight: undefined,
  });
}
function banner(s, x, y, w, h, runs, fill = AMBER, o = {}) {
  rect(s, x, y, w, h, fill, { fillT: o.fillT === undefined ? 82 : o.fillT });
  txt(s, runs, { x: x + 0.3, y, w: w - 0.6, h, fontSize: o.pt || 18, valign: "middle", color: INK, align: o.align || "left" });
}
function sectionSlide(id, num, head, sub) {
  const s = slideOf("SECTION", id);
  s.addText(num, { placeholder: "num" });
  s.addText(head, { placeholder: "title" });
  s.addText(sub, { placeholder: "body" });
  for (let i = 0; i < 6; i++) rect(s, M + i * 1.05, 6.1, 0.9, 0.45, i === 5 ? AMBER : TEAL, { fillT: i === 5 ? 0 : 55 });
  return s;
}

// ------------------------------------------------------------------------------------------
// Slides
// ------------------------------------------------------------------------------------------
// 1. Cover
{
  const s = slideOf("DARK_TITLE", "cover");
  title(s, "注意力、GPT-2 与 GPT-3");
  s.addText("三篇论文精读 · 重点展开：因果自注意力与 KV Cache", { placeholder: "body" });
  const papers = [
    ["2017", "Attention Is All You Need", TEAL],
    ["2019", "Language Models are Unsupervised Multitask Learners", INDIGO],
    ["2020", "Language Models are Few-Shot Learners", AMBER],
  ];
  papers.forEach(([y, t, c], i) => {
    const yy = 4.45 + i * 0.68;
    chip(s, M, yy, y, c, { w: 1.1, h: 0.52, pt: 16, color: c === AMBER ? INK : WHITE });
    txt(s, t, { x: M + 1.3, y: yy, w: 9, h: 0.52, fontSize: 17, color: "E6E9F2", valign: "middle" });
  });
}

// 2. Roadmap
{
  const s = slideOf("CONTENT", "roadmap");
  title(s, "一条主线：三篇论文，三个递进的问题");
  const cols = [
    ["2017", "Transformer", "序列建模能不能完全不用循环网络？", "只用注意力就够了", TEAL],
    ["2019", "GPT-2", "只做语言建模，能否不经专门训练就完成各种任务？", "可以：零样本提示", INDIGO],
    ["2020", "GPT-3", "把规模再放大一百多倍，会发生什么？", "上下文学习：给几个例子就会做新任务", AMBER],
  ];
  const w = 3.75, gap = (CW - 3 * w) / 2;
  cols.forEach(([yr, name, q, a, c], i) => {
    const x = M + i * (w + gap), y = 1.5;
    rect(s, x, y, w, 3.85, LIGHT);
    txt(s, yr, { x: x + 0.3, y: y + 0.25, w: 2, h: 0.7, fontSize: 36, bold: true, color: c, valign: "middle" });
    txt(s, name, { x: x + 0.3, y: y + 0.95, w: w - 0.6, h: 0.5, fontSize: 21, bold: true, valign: "middle" });
    txt(s, q, { x: x + 0.3, y: y + 1.55, w: w - 0.6, h: 1.0, fontSize: 15, color: INK2 });
    rect(s, x + 0.3, y + 2.65, w - 0.6, 0.95, c, { fillT: 80 });
    txt(s, a, { x: x + 0.45, y: y + 2.65, w: w - 0.9, h: 0.95, fontSize: 15, bold: true, valign: "middle" });
    if (i < 2) line(s, x + w + 0.08, y + 1.9, x + w + gap - 0.08, y + 1.9, SLATE, { arrow: true, width: 2.5 });
  });
  banner(s, M, 5.65, CW, 0.95, [
    { text: "贯穿始终的两个技术事实：", options: { bold: true } },
    { text: "因果自注意力 → 训练可以并行；   KV Cache → 推理变得高效" },
  ]);
}

// 3. LM basics
{
  const s = slideOf("CONTENT", "lm_basics");
  title(s, "语言模型在做什么：预测下一个 token");
  rect(s, M, 1.5, 6.6, 1.2, LIGHT);
  txt(s, "p(x₁, …, xₙ) = ∏ₜ p(xₜ | x₁, …, xₜ₋₁)", { x: M + 0.3, y: 1.5, w: 6.0, h: 0.75, fontSize: 22, bold: true, valign: "middle" });
  txt(s, "链式法则：每个 token 只以它前面的 token 为条件", { x: M + 0.3, y: 2.2, w: 6.0, h: 0.4, fontSize: 14, color: INK2 });
  bullets(s, [
    [{ text: "训练：", bold: true }, { text: "最小化每个位置的负对数似然 −Σ log p(xₜ | x<ₜ)" }],
    [{ text: "生成：", bold: true }, { text: "预测分布 → 选一个 → 拼到末尾 → 再预测（自回归）" }],
  ], { x: M, y: 2.95, w: 6.6, h: 1.4, fontSize: 15 });
  card(s, M, 4.5, 3.2, 2.15, "训练：整段已知", "希望一次前向算出所有位置的预测\n→ 靠因果自注意力", TEAL);
  card(s, M + 3.4, 4.5, 3.2, 2.15, "推理：未来不存在", "只能一个一个生成\n→ 靠 KV Cache 加速", CORAL);
  figure(s, "lm_next", 7.55, 1.5, 5.18, 4.9);
}

// 4. Section: attention
sectionSlide("sec_attention", "01", "注意力机制", "Attention Is All You Need（Vaswani 等，2017）");

// 5. RNN problem
{
  const s = slideOf("CONTENT", "rnn_problem");
  title(s, "循环网络的两个瓶颈");
  txt(s, "hₜ = f(hₜ₋₁, xₜ)", { x: M, y: 1.5, w: 5.6, h: 0.6, fontSize: 24, bold: true, valign: "middle" });
  txt(s, "LSTM 等循环网络按时间步逐个计算", { x: M, y: 2.1, w: 5.6, h: 0.4, fontSize: 14, color: INK2 });
  card(s, M, 2.7, 5.6, 1.35, "① 无法并行", "必须等上一步算完才能算下一步，序列越长训练越慢", CORAL, { bodyPt: 14 });
  card(s, M, 4.2, 5.6, 1.35, "② 长程依赖难学", "第 1 个词的信息传到第 n 个词要经过 n 步，信号不断衰减", CORAL, { bodyPt: 14 });
  banner(s, M, 5.75, 5.6, 0.9, [{ text: "论文主张：", options: { bold: true } }, { text: "抛弃循环和卷积，只用注意力" }], TEAL, { pt: 17 });
  figure(s, "rnn_vs_attn", 6.6, 1.45, 6.13, 5.25);
}

// 6. QKV
{
  const s = slideOf("CONTENT", "qkv");
  title(s, "注意力 = 用查询去匹配键，取回值的加权和");
  txt(s, "把一个查询和一组键值对映射成输出：输出是值的加权和，权重由查询与每个键的匹配程度决定。", { x: M, y: 1.5, w: 5.9, h: 0.9, fontSize: 15, color: INK2 });
  const rows = [
    ["查询 Q", "脑子里的问题", INDIGO],
    ["键 K", "每本书书脊上的标签", TEAL],
    ["值 V", "书里的内容", AMBER],
  ];
  rows.forEach(([a, b, c], i) => {
    const y = 2.55 + i * 0.85;
    chip(s, M, y, a, c, { w: 1.6, h: 0.65, pt: 17, color: c === AMBER ? INK : WHITE });
    txt(s, b, { x: M + 1.85, y, w: 4.0, h: 0.65, fontSize: 17, valign: "middle" });
  });
  txt(s, "图书馆的比喻", { x: M, y: 2.15, w: 4, h: 0.35, fontSize: 13, color: SLATE });
  banner(s, M, 5.3, 5.9, 1.3, [
    { text: "自注意力：", options: { bold: true, breakLine: false } },
    { text: "Q、K、V 都来自同一个序列。每个位置既能提问，也能被别人查询。" },
  ], INDIGO, { pt: 16 });
  figure(s, "qkv", 6.85, 1.5, 5.88, 5.1);
}

// 7. SDPA (animated)
{
  const s = slideOf("CONTENT", "sdpa");
  title(s, "缩放点积注意力：四步看懂核心公式");
  rect(s, M, 1.5, 4.3, 1.15, INK);
  txt(s, "softmax(QKᵀ / √d_k) · V", { x: M, y: 1.5, w: 4.3, h: 1.15, fontSize: 21, bold: true, color: WHITE, align: "center", valign: "middle", fontFace: CODE_FONT });
  const steps = [
    ["打分", "Q·Kᵀ：第 i 行第 j 列 = 位置 i 对 j 的关注度"],
    ["缩放", "除以 √d_k（例子里 d_k = 4，即除以 2）"],
    ["归一化", "每行 softmax，权重加起来等于 1"],
    ["加权求和", "用权重对 V 加权平均，得到新向量"],
  ];
  steps.forEach(([h, b], i) => {
    const y = 2.85 + i * 0.86;
    circleNum(s, M, y + 0.05, 0.5, String(i + 1), [INDIGO, INDIGO, TEAL, AMBER][i], WHITE, 15);
    txt(s, h, { x: M + 0.68, y, w: 3.6, h: 0.38, fontSize: 16, bold: true, valign: "middle" });
    txt(s, b, { x: M + 0.68, y: y + 0.38, w: 3.6, h: 0.45, fontSize: 12, color: INK2 });
  });
  txt(s, "两次矩阵乘法 + 一次 softmax，GPU 友好", { x: M, y: 6.38, w: 4.3, h: 0.4, fontSize: 13, color: TEAL, bold: true });
  figure(s, "sdpa", 5.1, 1.4, 7.63, 5.45, { anim: "sdpa" });
}

// 8. Scaling
{
  const s = slideOf("CONTENT", "scaling");
  title(s, "为什么要除以 √d_k");
  txt(s, "假设 q、k 的分量独立，均值 0、方差 1：", { x: M, y: 1.5, w: 5.8, h: 0.45, fontSize: 15, color: INK2 });
  rect(s, M, 2.0, 5.8, 1.0, LIGHT);
  txt(s, "q·k = Σᵢ qᵢkᵢ   →   均值 0，方差 d_k", { x: M + 0.25, y: 2.0, w: 5.4, h: 1.0, fontSize: 20, bold: true, valign: "middle" });
  bullets(s, [
    [{ text: "d_k = 64 时，点积标准差约为 8" }],
    [{ text: "不缩放：", bold: true, color: CORAL }, { text: "差距过大，softmax 接近 one-hot，落入梯度极小的饱和区" }],
    [{ text: "除以 √d_k：", bold: true, color: TEAL }, { text: "方差回到 1，权重柔和，梯度正常流动" }],
  ], { x: M, y: 3.2, w: 5.8, h: 2.1, fontSize: 15 });
  banner(s, M, 5.5, 5.8, 1.1, "缩放后点积注意力与加性注意力效果相当，却能用高度优化的矩阵乘法实现，快得多", TEAL, { pt: 15 });
  figure(s, "scaling", 6.65, 1.45, 6.08, 5.3);
}

// 9. Multi-head
{
  const s = slideOf("CONTENT", "multihead");
  title(s, "多头注意力：在多个子空间里各看各的");
  rect(s, M, 1.5, 5.5, 1.35, LIGHT);
  txt(s, [
    { text: "MultiHead = Concat(head₁, …, headₕ) · Wᴼ", options: { breakLine: true } },
    { text: "headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)" },
  ], { x: M + 0.25, y: 1.5, w: 5.1, h: 1.35, fontSize: 15, bold: true, valign: "middle", paraSpaceAfter: 6 });
  const st = [["8", "个头 h", INDIGO], ["512", "模型维度", TEAL], ["64", "每头维度", AMBER]];
  st.forEach(([b, l, c], i) => stat(s, M + i * 1.85, 3.0, 1.7, b, l, c, { bigPt: 34, bigH: 0.7, labelH: 0.4 }));
  bullets(s, [
    "每个头维度缩小，总计算量与单头全维度相当",
    "不同的头各司其职：相邻词、句法依存、指代……",
    [{ text: "消融：单头比最佳配置低 " }, { text: "0.9 BLEU", bold: true, color: CORAL }, { text: "；头太多也会变差" }],
  ], { x: M, y: 4.3, w: 5.5, h: 2.3, fontSize: 15 });
  figure(s, "multihead", 6.35, 1.5, 6.38, 5.1);
}

// 10. Architecture
{
  const s = slideOf("CONTENT", "architecture");
  title(s, "Transformer 结构与注意力的三种用法");
  figure(s, "transformer", M, 1.4, 5.9, 5.35);
  const uses = [
    ["编码器自注意力", "每个位置看到源句的全部位置", TEAL],
    ["编码器-解码器注意力", "Q 来自解码器，K、V 来自编码器输出：译文的每个位置查看整句原文", AMBER],
    ["解码器自注意力（带掩码）", "每个位置只能看到自己和之前的位置", CORAL],
  ];
  uses.forEach(([h, b, c], i) => card(s, 6.75, 1.5 + i * 1.38, 5.98, 1.22, h, b, c, { num: String(i + 1), bodyPt: 13.5, headH: 0.42 }));
  banner(s, 6.75, 5.75, 5.98, 0.9, [{ text: "GPT 系列只保留第 3 种 → 仅解码器架构", options: { bold: true } }], CORAL, { pt: 17 });
}

// 11. Components
{
  const s = slideOf("CONTENT", "components");
  title(s, "前馈网络、权重共享与位置编码");
  card(s, M, 1.5, 6.4, 1.45, "逐位置前馈网络", "两层线性 + ReLU，内层 2048 维。注意力负责位置之间交换信息，前馈网络负责每个位置内部加工", TEAL, { num: "1", bodyPt: 13.5 });
  card(s, M, 3.1, 6.4, 1.1, "权重共享", "输入嵌入、输出嵌入、softmax 前的线性层共用一个矩阵", INDIGO, { num: "2", bodyPt: 13.5 });
  card(s, M, 4.35, 6.4, 2.3, "位置编码", "注意力对顺序不敏感，必须显式注入位置：\nPE(pos, 2i) = sin(pos / 10000^(2i/d))\nPE(pos, 2i+1) = cos(pos / 10000^(2i/d))\n可学习的位置嵌入效果几乎一样；GPT-2/3 用的正是它", AMBER, { num: "3", bodyPt: 13.5 });
  figure(s, "posenc", 7.25, 1.6, 5.48, 4.9);
}

// 12. Complexity
{
  const s = slideOf("CONTENT", "complexity");
  title(s, "为什么用自注意力：三个角度的比较（论文表 1）");
  table(s, ["层类型", "每层计算量", "顺序操作数", "最大路径长度"], [
    ["自注意力", "O(n²·d)", "O(1)", "O(1)"],
    ["循环层", "O(n·d²)", "O(n)", "O(n)"],
    ["卷积层", "O(k·n·d²)", "O(1)", "O(logₖ n)"],
    ["受限自注意力（窗口 r）", "O(r·n·d)", "O(1)", "O(n/r)"],
  ], { x: M, y: 1.55, w: CW, colW: [3.4, 2.9, 2.9, 2.933], rowH: 0.62, fontSize: 17, highlight: [0] });
  const notes = [
    ["顺序操作数", "决定能否并行：自注意力是常数", TEAL],
    ["最大路径长度", "决定长程依赖好不好学：自注意力一步直达", INDIGO],
    ["计算量", "n < d 时自注意力比循环层更省", AMBER],
  ];
  notes.forEach(([h, b, c], i) => card(s, M + i * 4.1, 4.85, 3.93, 1.0, h, b, c, { bodyPt: 13, headH: 0.36, headPt: 15 }));
  banner(s, M, 6.0, CW, 0.7, [{ text: "代价是 n²：", options: { bold: true, color: CORAL } }, { text: "序列很长时注意力矩阵成为瓶颈 —— 讲推理时它会以另一种形式再次出现" }], CORAL, { pt: 15 });
}

// 13. Transformer results
{
  const s = slideOf("CONTENT", "transformer_results");
  title(s, "实验结果：更好，也便宜得多（WMT 2014，论文表 2）");
  table(s, ["模型", "英→德 BLEU", "英→法 BLEU", "训练 FLOPs（英→德）"], [
    ["GNMT + RL 集成", "26.30", "41.16", "1.8 × 10²⁰"],
    ["ConvS2S 集成", "26.36", "41.29", "7.7 × 10¹⁹"],
    ["Transformer（base）", "27.3", "38.1", "3.3 × 10¹⁸"],
    ["Transformer（big）", "28.4", "41.8", "2.3 × 10¹⁹"],
  ], { x: M, y: 1.55, w: 7.6, colW: [2.6, 1.6, 1.6, 1.8], rowH: 0.62, fontSize: 15, highlight: [3] });
  const st = [["12 小时", "base：8 块 P100", TEAL], ["3.5 天", "big：8 块 P100", INDIGO]];
  st.forEach(([b, l, c], i) => stat(s, 8.75 + i * 2.05, 1.5, 1.95, b, l, c, { bigPt: 30, bigH: 0.7 }));
  card(s, 8.75, 3.0, 3.98, 2.0, "训练细节", "Adam；学习率线性预热 4000 步，再按步数平方根倒数衰减；dropout 0.1、标签平滑 0.1", AMBER, { bodyPt: 13 });
  card(s, M, 4.95, 7.6, 1.7, "结论", "big 模型英德比此前最好（含集成）高出 2 BLEU 以上，英法为新的单模型最佳；同一架构用于英语句法分析也表现很好 → 通用序列建模架构", TEAL, { bodyPt: 14 });
  card(s, 8.75, 5.15, 3.98, 1.5, "h = 8，d_model = 512", "d_ff = 2048；big 模型约 2.13 亿参数", INDIGO, { bodyPt: 13, headPt: 15 });
}

// 14. Section: causal
sectionSlide("sec_causal", "02", "因果自注意力", "论文里只有一句话，却是整个 GPT 系列的地基");

// 15. Teacher forcing
{
  const s = slideOf("CONTENT", "teacher_forcing");
  title(s, "问题：训练时整段文本已知，怎样防止“偷看答案”");
  const inp = ["我", "喜欢", "吃", "苹果"], tgt = ["喜欢", "吃", "苹果", "。"];
  const x0 = 2.1, cw = 1.45, gap = 0.3;
  txt(s, "输入（真实历史）", { x: M, y: 1.75, w: 1.5, h: 0.6, fontSize: 13, color: INK2, valign: "middle" });
  txt(s, "要预测的目标", { x: M, y: 3.35, w: 1.5, h: 0.6, fontSize: 13, color: INK2, valign: "middle" });
  inp.forEach((t, i) => {
    const x = x0 + i * (cw + gap);
    chip(s, x, 1.75, t, TEAL, { w: cw, h: 0.6, pt: 17 });
    line(s, x + cw / 2, 2.4, x + cw / 2, 3.3, SLATE, { arrow: true, width: 1.75 });
    txt(s, `位置 ${i + 1}`, { x: x + cw / 2 + 0.08, y: 2.6, w: 1.0, h: 0.4, fontSize: 11, color: SLATE });
    chip(s, x, 3.35, tgt[i], AMBER, { w: cw, h: 0.6, pt: 17, color: INK });
  });
  txt(s, "一次前向 → T 个预测（teacher forcing）", { x: x0, y: 4.05, w: 7, h: 0.4, fontSize: 14, color: TEAL, bold: true });
  card(s, 9.4, 1.6, 3.33, 2.85, "普通自注意力的漏洞", "位置 2 预测第 3 个 token 时，若能直接看到第 3 个 token，只要学会“抄答案”：训练损失很低，推理时毫无用处", CORAL, { bodyPt: 13.5 });
  banner(s, M, 4.85, CW, 1.6, [
    { text: "必须保证的规则：", options: { bold: true, breakLine: true } },
    { text: "位置 i 的输出，只能依赖第 1 到第 i 个 token", options: { fontSize: 24, bold: true, color: CORAL } },
  ], CORAL, { pt: 17 });
}

// 16. Causal mask (animated)
{
  const s = slideOf("CONTENT", "causal_mask");
  title(s, "做法：softmax 之前加上因果掩码");
  rect(s, M, 1.5, 5.0, 1.0, INK);
  txt(s, "A = softmax(S + M)", { x: M, y: 1.5, w: 5.0, h: 1.0, fontSize: 24, bold: true, color: WHITE, align: "center", valign: "middle", fontFace: CODE_FONT });
  rect(s, M, 2.7, 5.0, 1.0, LIGHT);
  txt(s, "M[i, j] = 0（j ≤ i）；−∞（j > i）", { x: M + 0.2, y: 2.7, w: 4.6, h: 1.0, fontSize: 19, bold: true, valign: "middle" });
  bullets(s, [
    "S：缩放后的打分矩阵，行 = 查询，列 = 键",
    [{ text: "e^(−∞) = 0：", bold: true }, { text: "被遮住的格子权重恰好为 0" }],
    "每行剩下的权重仍然加起来等于 1",
    [{ text: "“我”对“苹果”原有 0.1 分，也被抹成 0", color: CORAL, bold: true }],
  ], { x: M, y: 3.95, w: 5.0, h: 2.2, fontSize: 15 });
  txt(s, "结果：下三角的注意力权重矩阵", { x: M, y: 6.2, w: 5.0, h: 0.45, fontSize: 15, color: TEAL, bold: true });
  figure(s, "causal", 5.85, 1.4, 6.88, 5.45, { anim: "causal" });
}

// 17. Causality across layers
{
  const s = slideOf("CONTENT", "causal_layers");
  title(s, "关键性质：因果性逐层保持");
  card(s, M, 1.5, 5.9, 1.3, "逐位置操作（不跨位置）", "层归一化、QKV 投影、前馈网络、残差相加", TEAL, { bodyPt: 14 });
  card(s, M, 2.95, 5.9, 1.3, "跨位置操作：只有因果自注意力", "位置 i 只读取不超过 i 的位置", CORAL, { bodyPt: 14 });
  banner(s, M, 4.4, 5.9, 0.85, [{ text: "归纳法 ⇒ 任意层位置 i 的 h、k、v 只取决于 x₁…xᵢ", options: { bold: true } }], INDIGO, { pt: 15 });
  card(s, M, 5.4, 2.87, 1.3, "训练", "一次前向，T 个损失同时算", TEAL, { bodyPt: 13, headPt: 16, headH: 0.38 });
  card(s, M + 3.03, 5.4, 2.87, 1.3, "推理", "旧位置的 k、v 不变 → KV Cache", AMBER, { bodyPt: 13, headPt: 16, headH: 0.38 });
  figure(s, "causal_layers", 6.75, 1.5, 5.98, 5.2);
}

// 18. Real attention
{
  const s = slideOf("CONTENT", "real_attention");
  title(s, "真实的 GPT-2：注意力全部落在下三角");
  figure(s, "real_attention", M, 1.4, 7.4, 5.4, { alignX: "l" });
  const notes = [
    ["右上三角全空白", "因果掩码的效果", SLATE],
    ["左上：看前一个 token", "最常见的局部模式", TEAL],
    ["右上：集中在第一个 token", "第一个位置像存放多余注意力的地方", INDIGO],
    ["下面两个：分散关注", "把注意力分给更多语义相关的位置", AMBER],
  ];
  notes.forEach(([h, b, c], i) => card(s, 8.25, 1.5 + i * 1.22, 4.48, 1.08, h, b, c, { bodyPt: 12.5, headPt: 15, headH: 0.36 }));
  txt(s, "数据：本仓库 01 期，GPT-2 small，提示 “Hello, I'm a language model,”", { x: 8.25, y: 6.4, w: 4.48, h: 0.45, fontSize: 11, color: SLATE });
}

// 19. Causal code
{
  const s = slideOf("CONTENT", "causal_code");
  title(s, "代码：is_causal=True 背后就是那个下三角");
  txt(s, "src/learn_gpt2/gpt2.py", { x: M, y: 1.35, w: 6, h: 0.35, fontSize: 13, color: SLATE, fontFace: CODE_FONT });
  codePanel(s, "attn", M, 1.75, 6.55, 4.1, 13);
  txt(s, "手写等价实现", { x: 7.35, y: 1.35, w: 5, h: 0.35, fontSize: 13, color: SLATE });
  codePanel(s, "manual", 7.35, 1.75, 5.38, 4.1, 13);
  banner(s, M, 6.05, CW, 0.65, [
    { text: "c_attn 一次算出 Q、K、V → 切成多头 → 带掩码的 softmax(QKᵀ/√d)·V → 拼回 → c_proj", options: { fontFace: FONT } },
  ], TEAL, { pt: 15 });
}

// 20. Section: GPT-2
sectionSlide("sec_gpt2", "03", "GPT-2", "Language Models are Unsupervised Multitask Learners（Radford 等，2019）");

// 21. GPT-2 idea
{
  const s = slideOf("CONTENT", "gpt2_idea");
  title(s, "核心想法：任务本身也可以写成文本");
  card(s, M, 1.5, 5.9, 1.55, "当时的主流：预训练 + 有监督微调", "每个任务都要标注数据、再训练一遍；得到一个个狭窄的专家，换个分布就脆弱", SLATE, { bodyPt: 13.5 });
  rect(s, 6.75, 1.5, 5.98, 1.55, LIGHT);
  txt(s, [
    { text: "单个任务：p(output | input)", options: { breakLine: true } },
    { text: "通用系统：p(output | input, task)", options: { color: INDIGO } },
  ], { x: 7.0, y: 1.5, w: 5.5, h: 1.55, fontSize: 18, bold: true, valign: "middle", fontFace: CODE_FONT, paraSpaceAfter: 6 });
  txt(s, "把任务、输入、输出都写成一个文字序列：", { x: M, y: 3.3, w: 6, h: 0.4, fontSize: 15, color: INK2 });
  const ex = [
    "(translate to french, english text, french text)",
    "(answer the question, document, question, answer)",
  ];
  ex.forEach((t, i) => {
    rect(s, M, 3.8 + i * 0.8, 5.9, 0.65, INK);
    txt(s, t, { x: M + 0.2, y: 3.8 + i * 0.8, w: 5.5, h: 0.65, fontSize: 13, color: "C3E88D", valign: "middle", fontFace: CODE_FONT });
  });
  txt(s, "足够强的语言模型，在预测下一个词的过程中，就可能顺带学会这些任务", { x: M, y: 5.5, w: 5.9, h: 0.9, fontSize: 15, bold: true, color: TEAL });
  card(s, 6.75, 3.3, 5.98, 3.35, "网页里自然出现的翻译示范（WebText，论文表 1）", "“I'm not the cleverest man in the world, but like they say in French: Je ne suis pas un imbecile [I'm not a fool].”\n\n为了预测好这样的文本，模型就必须学会翻译", AMBER, { bodyPt: 15, headPt: 15 });
}

// 22. WebText & BPE
{
  const s = slideOf("CONTENT", "webtext_bpe");
  title(s, "数据与输入表示：WebText 和字节级 BPE");
  rect(s, M, 1.5, 5.9, 5.15, LIGHT);
  txt(s, "WebText：用人类投票做质量过滤", { x: M + 0.3, y: 1.7, w: 5.3, h: 0.5, fontSize: 18, bold: true });
  txt(s, "抓取 Reddit 上获得至少 3 个赞（karma）的外链", { x: M + 0.3, y: 2.25, w: 5.3, h: 0.45, fontSize: 14, color: INK2 });
  const st = [["4500 万", "外链", INDIGO], ["800 万", "篇文档", TEAL], ["40 GB", "文本", AMBER]];
  st.forEach(([b, l, c], i) => stat(s, M + 0.3 + i * 1.8, 2.85, 1.7, b, l, c, { bigPt: 28, bigH: 0.65, labelH: 0.4 }));
  txt(s, "删除了所有维基百科文档，避免与评测集重叠", { x: M + 0.3, y: 4.25, w: 5.3, h: 0.45, fontSize: 14, color: INK2 });
  txt(s, "对比：Common Crawl 体量巨大，但质量很差", { x: M + 0.3, y: 4.8, w: 5.3, h: 0.45, fontSize: 14, color: SLATE });
  rect(s, 6.75, 1.5, 5.98, 5.15, LIGHT);
  txt(s, "字节级 BPE：能表示任意字符串", { x: 7.05, y: 1.7, w: 5.4, h: 0.5, fontSize: 18, bold: true });
  table(s, ["做法", "基础词表"], [
    ["Unicode 字符 BPE", "> 13 万"],
    ["UTF-8 字节 BPE", "256"],
  ], { x: 7.05, y: 2.35, w: 5.38, colW: [3.2, 2.18], rowH: 0.5, fontSize: 14, highlight: [1] });
  txt(s, "朴素字节 BPE 会学出 dog. / dog! 这类冗余 token → 禁止跨字符类别合并", { x: 7.05, y: 4.0, w: 5.38, h: 0.85, fontSize: 14, color: INK2 });
  stat(s, 7.05, 4.85, 5.38, "50257", "最终词表大小，GPT-3 原样沿用", CORAL, { bigPt: 34, bigH: 0.75 });
}

// 23. GPT-2 architecture
{
  const s = slideOf("CONTENT", "gpt2_arch");
  title(s, "模型：仅解码器 Transformer + 几处关键修改");
  figure(s, "gpt2_arch", M, 1.4, 3.6, 5.4, { alignX: "l" });
  const mods = [
    ["Pre-LN", "LN 移到子块输入端，深层更稳定"],
    ["最后再加 ln_f", "最后一个块之后追加一层归一化"],
    ["残差初始化 × 1/√N", "N 为残差层数，抵消方差随深度累积"],
    ["上下文 1024", "从 512 扩大到 1024，批大小 512"],
  ];
  mods.forEach(([h, b], i) => card(s, 4.45, 1.5 + i * 1.0, 4.1, 0.88, h, b, [AMBER, AMBER, TEAL, INDIGO][i], { num: String(i + 1), bodyPt: 11.5, headPt: 14.5, headH: 0.3 }));
  table(s, ["参数量", "层数", "d_model"], [
    ["117M", "12", "768"],
    ["345M", "24", "1024"],
    ["762M", "36", "1280"],
    ["1542M", "48", "1600"],
  ], { x: 8.85, y: 1.5, w: 3.88, colW: [1.5, 1.09, 1.29], rowH: 0.52, fontSize: 14, highlight: [3] });
  txt(s, "最小 ≈ GPT-1；最大约 15 亿参数，才是真正的 GPT-2", { x: 8.85, y: 4.2, w: 3.88, h: 0.8, fontSize: 13, color: INK2 });
  banner(s, 4.45, 5.65, 8.28, 1.0, "代码对照：Block.forward 中 x = x + attn(ln_1(x))；x = x + mlp(ln_2(x))", TEAL, { pt: 14 });
}

// 24. GPT-2 LM results
{
  const s = slideOf("CONTENT", "gpt2_lm");
  title(s, "零样本语言建模：8 个数据集中 7 个刷新纪录");
  figure(s, "gpt2_lm", M, 1.45, 8.0, 5.0, { alignX: "l", alignY: "t" });
  stat(s, 9.0, 1.5, 3.73, "99.8 → 8.63", "LAMBADA 困惑度（考察长程依赖）", CORAL, { bigPt: 30, bigH: 0.7 });
  stat(s, 9.0, 2.85, 3.73, "59 → 63", "LAMBADA 准确率 %", INDIGO, { bigPt: 30, bigH: 0.7 });
  card(s, 9.0, 4.2, 3.73, 1.15, "唯一没赢：One Billion Word", "句子被打乱，破坏了长程结构", SLATE, { bodyPt: 12.5, headPt: 14, headH: 0.34 });
  card(s, 9.0, 5.5, 3.73, 1.15, "仍然欠拟合 WebText", "训练、测试一起提升：继续放大还有空间", TEAL, { bodyPt: 12.5, headPt: 14, headH: 0.34 });
}

// 25. GPT-2 tasks
{
  const s = slideOf("CONTENT", "gpt2_tasks");
  title(s, "下游任务：任务完全靠输入格式来暗示");
  const tasks = [
    ["阅读理解", "文档 + 对话历史 + \"A:\"", "CoQA 55 F1", "追平或超过 4 个有监督基线中的 3 个（它们用了 12.7 万条标注）", TEAL],
    ["摘要", "文章 + \"TL;DR:\"", "去掉提示 −6.4", "末尾几个字符就切换到摘要模式：提示工程的雏形", INDIGO],
    ["翻译", "english sentence = french sentence …", "法→英 11.5 BLEU", "训练集里只有约 10 MB 法语", AMBER],
  ];
  const w = 3.9, gap = (CW - 3 * w) / 2;
  tasks.forEach(([h, prompt, big, b, c], i) => {
    const x = M + i * (w + gap), y = 1.5;
    rect(s, x, y, w, 4.3, LIGHT);
    txt(s, h, { x: x + 0.3, y: y + 0.2, w: w - 0.6, h: 0.5, fontSize: 19, bold: true, color: c === AMBER ? "B9771A" : c });
    rect(s, x + 0.3, y + 0.85, w - 0.6, 0.85, INK);
    txt(s, prompt, { x: x + 0.45, y: y + 0.85, w: w - 0.9, h: 0.85, fontSize: 12.5, color: "C3E88D", valign: "middle", fontFace: CODE_FONT });
    txt(s, big, { x: x + 0.3, y: y + 1.9, w: w - 0.6, h: 0.75, fontSize: 25, bold: true, color: INK, valign: "middle" });
    txt(s, b, { x: x + 0.3, y: y + 2.7, w: w - 0.6, h: 1.4, fontSize: 13.5, color: INK2 });
  });
  banner(s, M, 6.0, CW, 0.65, "客观地说：这些零样本结果离实用还很远（摘要只比随机抽三句话略好）", SLATE, { pt: 15 });
}

// 26. Memorization & summary
{
  const s = slideOf("CONTENT", "gpt2_memo");
  title(s, "是背下了测试集吗？以及 GPT-2 的结论");
  txt(s, "8-gram 布隆过滤器检查评测集与 WebText 的重叠", { x: M, y: 1.5, w: 6, h: 0.45, fontSize: 15, color: INK2 });
  const st = [["3.2%", "与 WebText 平均重叠（1%～6%）", INDIGO], ["5.9%", "与它们自己的训练集平均重叠", AMBER], ["8.6 → 8.7", "剔除重叠后 LAMBADA 困惑度", TEAL]];
  st.forEach(([b, l, c], i) => stat(s, M, 2.05 + i * 1.3, 5.8, b, l, c, { bigPt: 32, bigH: 0.7, labelH: 0.4 }));
  banner(s, M, 5.95, 5.8, 0.7, "重叠带来少量但一致的好处，不足以解释 GPT-2 的表现", TEAL, { pt: 14 });
  rect(s, 6.85, 1.5, 5.88, 5.15, INK);
  txt(s, "GPT-2 小结", { x: 7.15, y: 1.75, w: 5.3, h: 0.5, fontSize: 20, bold: true, color: AMBER });
  txt(s, [
    { text: "只在大规模、多样化文本上做语言建模的模型，会自发学到许多任务的能力。", options: { breakLine: true } },
    { text: "可以用自然语言提示把这些能力调出来。", options: { breakLine: true } },
    { text: "但表现离实用还远。" },
  ], { x: 7.15, y: 2.4, w: 5.3, h: 2.6, fontSize: 17, color: "E6E9F2", paraSpaceAfter: 12 });
  txt(s, "那么，如果继续放大呢？", { x: 7.15, y: 5.4, w: 5.3, h: 0.7, fontSize: 24, bold: true, color: WHITE, valign: "middle" });
}

// 27. Section: GPT-3
sectionSlide("sec_gpt3", "04", "GPT-3", "Language Models are Few-Shot Learners（Brown 等，2020）");

// 28. GPT-3 motivation
{
  const s = slideOf("CONTENT", "gpt3_motivation");
  title(s, "动机：摆脱“每个任务都要微调”");
  const probs = [
    ["标注成本", "每个新任务都要成千上万条标注数据"],
    ["虚假相关", "模型越大、微调分布越窄，越容易学到捷径，高分可能夸大真实能力"],
    ["不像人类", "人学新任务往往只需一句说明或几个例子"],
  ];
  probs.forEach(([h, b], i) => card(s, M, 1.5 + i * 1.32, 5.4, 1.18, h, b, CORAL, { num: String(i + 1), bodyPt: 13, headPt: 15.5, headH: 0.36 }));
  banner(s, M, 5.55, 5.4, 1.1, [{ text: "上下文学习：", options: { bold: true } }, { text: "完全发生在一次前向计算里，不做任何梯度更新" }], TEAL, { pt: 15 });
  // meta-learning diagram (after the paper's figure 1.1)
  const x0 = 6.4, y0 = 1.5, w0 = 6.33, h0 = 5.15;
  rect(s, x0, y0, w0, h0, LIGHT, { line: MID });
  txt(s, "外循环：预训练（梯度下降）学到大量技能与模式", { x: x0 + 0.3, y: y0 + 0.15, w: w0 - 0.6, h: 0.5, fontSize: 15, bold: true, color: INDIGO, valign: "middle" });
  const seqs = [
    ["序列 1：算术", "5 + 8 = 13\n7 + 2 = 9\n1 + 0 = 1\n3 + 4 = 7"],
    ["序列 2：纠错", "gaot => goat\nsakne => snake\nbrid => bird\nfsih => fish"],
    ["序列 3：翻译", "thanks => merci\nhello => bonjour\nmint => menthe\nwall => mur"],
  ];
  const sw = 1.83, sg = (w0 - 0.6 - 3 * sw) / 2;
  seqs.forEach(([h, b], i) => {
    const x = x0 + 0.3 + i * (sw + sg), y = y0 + 0.85;
    rect(s, x, y, sw, 3.2, WHITE, { line: AMBER, lineW: 1.5 });
    txt(s, h, { x: x + 0.12, y: y + 0.12, w: sw - 0.24, h: 0.4, fontSize: 12.5, bold: true, color: "B9771A" });
    txt(s, b, { x: x + 0.12, y: y + 0.6, w: sw - 0.24, h: 2.4, fontSize: 12, fontFace: CODE_FONT, color: INK, paraSpaceAfter: 4 });
  });
  txt(s, "内循环：在每条序列内部，根据上文识别任务（上下文学习）", { x: x0 + 0.3, y: y0 + 4.2, w: w0 - 0.6, h: 0.7, fontSize: 14, bold: true, color: "B9771A", valign: "middle" });
}

// 29. GPT-3 settings
{
  const s = slideOf("CONTENT", "gpt3_settings");
  title(s, "四种设定：微调、零样本、单样本、少样本");
  const panels = [
    ["微调", "梯度更新", "sea otter => loutre de mer   ← 更新\npeppermint => menthe poivrée ← 更新\n…（大量标注样本）\ncheese =>", SLATE],
    ["零样本", "不更新参数", "Translate English to French:\ncheese =>", TEAL],
    ["单样本", "不更新参数", "Translate English to French:\nsea otter => loutre de mer\ncheese =>", INDIGO],
    ["少样本", "不更新参数", "Translate English to French:\nsea otter => loutre de mer\npeppermint => menthe poivrée\nplush girafe => girafe peluche\ncheese =>", AMBER],
  ];
  const w = 5.95, h = 2.15;
  panels.forEach(([name, tag, prompt, c], i) => {
    const x = M + (i % 2) * (w + 0.233), y = 1.45 + Math.floor(i / 2) * (h + 0.2);
    rect(s, x, y, w, h, LIGHT);
    chip(s, x + 0.2, y + 0.18, name, c, { h: 0.45, pt: 15, color: c === AMBER ? INK : WHITE });
    txt(s, tag, { x: x + 1.6, y: y + 0.18, w: 2.5, h: 0.45, fontSize: 13, color: i === 0 ? CORAL : TEAL, bold: true, valign: "middle" });
    rect(s, x + 0.2, y + 0.75, w - 0.4, h - 0.9, INK, { r: 0.08 });
    txt(s, prompt, { x: x + 0.35, y: y + 0.8, w: w - 0.7, h: h - 1.0, fontSize: 12.5, color: "E6E9F2", fontFace: CODE_FONT, valign: "middle" });
  });
  banner(s, M, 6.15, CW, 0.6, "示范数 K 通常 10～100，受 2048 上下文限制；模型做的仍只是预测下一个 token —— 提示越长，推理要处理的 token 越多", AMBER, { pt: 13.5 });
}

// 30. GPT-3 models
{
  const s = slideOf("CONTENT", "gpt3_models");
  title(s, "8 个规模，结构与 GPT-2 相同（论文表 2.1）");
  table(s, ["模型", "参数量", "层数", "d_model", "头数", "d_head", "批大小", "学习率"], [
    ["Small", "125M", "12", "768", "12", "64", "0.5M", "6.0e-4"],
    ["Medium", "350M", "24", "1024", "16", "64", "0.5M", "3.0e-4"],
    ["Large", "760M", "24", "1536", "16", "96", "0.5M", "2.5e-4"],
    ["XL", "1.3B", "24", "2048", "24", "128", "1M", "2.0e-4"],
    ["2.7B", "2.7B", "32", "2560", "32", "80", "1M", "1.6e-4"],
    ["6.7B", "6.7B", "32", "4096", "32", "128", "2M", "1.2e-4"],
    ["13B", "13.0B", "40", "5140", "40", "128", "2M", "1.0e-4"],
    ["GPT-3", "175.0B", "96", "12288", "96", "128", "3.2M", "0.6e-4"],
  ], { x: M, y: 1.45, w: 8.4, colW: [1.25, 1.15, 0.85, 1.1, 0.85, 1.0, 1.1, 1.1], rowH: 0.5, fontSize: 13.5, highlight: [7] });
  card(s, 9.25, 1.45, 3.48, 2.1, "与 GPT-2 相同", "初始化、Pre-LN、分词器一样；唯一区别：各层交替使用稠密与局部带状稀疏注意力", TEAL, { bodyPt: 12.5, headPt: 15, headH: 0.36 });
  stat(s, 9.25, 3.7, 3.48, "3000 亿", "每个模型都训练这么多 token", INDIGO, { bigPt: 30, bigH: 0.7, labelH: 0.4 });
  stat(s, 9.25, 4.9, 3.48, "2048", "上下文长度 n_ctx", AMBER, { bigPt: 30, bigH: 0.7, labelH: 0.4 });
  txt(s, "批大小单位为 token。规律：模型越大，批越大，学习率越小", { x: M, y: 6.2, w: 8.4, h: 0.45, fontSize: 13, color: SLATE });
}

// 31. GPT-3 data
{
  const s = slideOf("CONTENT", "gpt3_data");
  title(s, "数据：质量比数量更重要");
  figure(s, "gpt3_data", M, 1.45, 7.0, 4.3, { alignX: "l", alignY: "t" });
  const steps = [
    ["按相似度过滤 Common Crawl", "以高质量语料为参照"],
    ["文档级模糊去重", "减少冗余，保证验证集可信"],
    ["混入高质量语料", "WebText2、Books1/2、英文维基百科"],
  ];
  steps.forEach(([h, b], i) => card(s, 7.9, 1.5 + i * 1.12, 4.83, 1.0, h, b, TEAL, { num: String(i + 1), bodyPt: 12.5, headPt: 14.5, headH: 0.32 }));
  banner(s, M, 5.95, 7.0, 0.7, "采样不按大小：维基百科看 3.4 遍，Common Crawl 不到半遍", AMBER, { pt: 14.5 });
  stat(s, 7.9, 5.0, 4.83, "3.14 × 10²³", "最大模型的训练 FLOPs（微软 V100 集群）", CORAL, { bigPt: 30, bigH: 0.7, labelH: 0.5 });
}

// 32. GPT-3 results 1
{
  const s = slideOf("CONTENT", "gpt3_results1");
  title(s, "结果（一）：少样本逼近甚至超过微调");
  const pts = [
    ["LAMBADA 86.4%", "少样本比此前最好高 18 个百分点；示例里的填空格式让模型领会“只填一个词”", CORAL],
    ["TriviaQA 71.2%", "闭卷问答，超过同样闭卷设定下微调的 T5", INDIGO],
    ["CoQA 85 F1", "少样本阅读理解，接近人类水平", TEAL],
    ["翻译", "译入英语超过此前无监督机器翻译；译出英语较弱（分词器为英语设计）", AMBER],
  ];
  pts.forEach(([h, b, c], i) => card(s, M, 1.45 + i * 1.32, 4.6, 1.2, h, b, c, { bodyPt: 12, headPt: 16, headH: 0.36 }));
  figure(s, "gpt3_results", 5.45, 1.45, 7.28, 5.25);
}

// 33. GPT-3 results 2
{
  const s = slideOf("CONTENT", "gpt3_results2");
  title(s, "结果（二）：算术、新闻生成，以及短板");
  figure(s, "gpt3_arith", M, 1.45, 6.6, 3.65, { alignX: "l", alignY: "t" });
  txt(s, "D = 位数，+ / − / × 为运算，1DC = 一位数复合运算", { x: M, y: 5.15, w: 6.6, h: 0.4, fontSize: 12, color: SLATE });
  banner(s, M, 5.65, 6.6, 1.0, "训练数据中三位数加法题不到 1%，错误多是忘记进位 → 确实在计算", TEAL, { pt: 14 });
  const st = [
    ["71.8", "SuperGLUE 少样本（每任务 32 例），超过微调 BERT-Large 的 69.0", INDIGO],
    ["52%", "人类识别最大模型写的约 200 词新闻，几乎等于瞎猜", CORAL],
    ["49.4%", "短板：WiC 词义比较 ≈ 随机；ANLI 也接近随机", SLATE],
  ];
  st.forEach(([b, l, c], i) => stat(s, 7.6, 1.45 + i * 1.75, 5.13, b, l, c, { bigPt: 34, bigH: 0.72, labelH: 0.8 }));
}

// 34. GPT-3 scaling
{
  const s = slideOf("CONTENT", "gpt3_scaling");
  title(s, "规模的规律");
  figure(s, "gpt3_scaling", M, 1.4, CW, 3.95, { alignY: "t" });
  const pts = [
    ["平滑幂律", "验证损失随计算量幂律下降，趋势又延伸了两个数量级", INDIGO],
    ["差距随规模扩大", "零/单/少样本的差距随模型变大而扩大：大模型是更好的元学习者", TEAL],
    ["能力涌现", "多位数算术等能力，小模型几乎为零，最大模型才明显出现", CORAL],
  ];
  pts.forEach(([h, b, c], i) => card(s, M + i * 4.1, 5.45, 3.93, 1.25, h, b, c, { bodyPt: 12.5, headPt: 15, headH: 0.34 }));
}

// 35. GPT-3 limits
{
  const s = slideOf("CONTENT", "gpt3_limits");
  title(s, "局限与风险：论文自己说了什么");
  const cols = [
    ["数据污染", "过滤重叠的代码有 bug，重训成本太高\n事后为每个基准构造干净子集评估\n受影响较大的结果加星号标注", AMBER],
    ["能力局限", "长文本重复、前后矛盾\n常识物理较弱（奶酪放进冰箱会融化吗？）\n比较类任务弱：与单向因果结构有关\n预训练样本效率低\n少样本是在学新任务，还是在识别见过的任务？", CORAL],
    ["成本与社会影响", "推理昂贵、难以解释\n滥用风险\n性别、种族、宗教方面的偏见\n能耗问题", INDIGO],
  ];
  const w = 3.9, gap = (CW - 3 * w) / 2;
  cols.forEach(([h, b, c], i) => {
    const x = M + i * (w + gap);
    rect(s, x, 1.5, w, 5.15, LIGHT);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.5, w, h: 0.1, fill: { color: c }, line: { type: "none" } });
    txt(s, h, { x: x + 0.3, y: 1.8, w: w - 0.6, h: 0.5, fontSize: 19, bold: true });
    bullets(s, b.split("\n"), { x: x + 0.3, y: 2.45, w: w - 0.6, h: 4.0, fontSize: 14, color: INK2 });
  });
}

// 36. Section: KV cache
sectionSlide("sec_kv", "05", "KV Cache", "三篇论文都没写，却是让 GPT 真正可用的工程关键 —— 它直接来自因果性");

// 37. Recompute (animated)
{
  const s = slideOf("CONTENT", "recompute");
  title(s, "不用缓存：每一步都把整段序列重算一遍");
  bullets(s, [
    "每一轮把整段已有序列送进模型",
    "每一层重算所有 token 的 Q、K、V、注意力和前馈网络",
    [{ text: "最后只用最后一个位置的 logits", bold: true }],
  ], { x: M, y: 1.5, w: 5.6, h: 1.9, fontSize: 15 });
  rect(s, M, 3.55, 5.6, 1.6, LIGHT);
  txt(s, [
    { text: "生成 N 个 token 的 token 前向次数", options: { fontSize: 14, color: INK2, breakLine: true } },
    { text: "1 + 2 + … + N = N(N+1)/2 ≈ N²/2", options: { fontSize: 22, bold: true, color: CORAL } },
  ], { x: M + 0.25, y: 3.55, w: 5.1, h: 1.6, valign: "middle", paraSpaceAfter: 8 });
  stat(s, M, 5.25, 2.7, "524,800", "N = 1024 时的总次数", CORAL, { bigPt: 28, bigH: 0.65, labelH: 0.4 });
  stat(s, M + 2.9, 5.25, 2.7, "1,024", "真正有用的次数", TEAL, { bigPt: 28, bigH: 0.65, labelH: 0.4 });
  figure(s, "recompute", 6.45, 1.45, 6.28, 5.3, { anim: "recompute" });
}

// 38. KV insight
{
  const s = slideOf("CONTENT", "kv_insight");
  title(s, "核心洞察：旧位置的 K、V 永远不会变");
  banner(s, M, 1.5, CW, 1.0, [
    { text: "因果掩码 ⇒ 任意层、位置 j 的 kⱼ、vⱼ 只取决于 x₁…xⱼ ⇒ ", options: {} },
    { text: "算过一次，就可以一直复用", options: { bold: true, color: CORAL } },
  ], INDIGO, { pt: 18 });
  const cards = [
    ["存什么", "每一层算过的全部 K 和 V（每层一份缓存）", TEAL],
    ["每步做什么", "只为新 token 计算 q、k、v；把 k、v 追加到缓存末尾；用 q 和整个缓存做注意力", INDIGO],
    ["为什么不缓存 Q", "每步只用最新 token 的查询；旧 token 的 q 只在它自己最新时用过一次", AMBER],
    ["为什么不需要掩码", "新 token 排在序列最后，本来就能看到所有历史位置", CORAL],
  ];
  cards.forEach(([h, b, c], i) => card(s, M + (i % 2) * 6.17, 2.75 + Math.floor(i / 2) * 1.62, 5.96, 1.45, h, b, c, { num: String(i + 1), bodyPt: 14, headPt: 16, headH: 0.38 }));
  rect(s, M, 6.05, CW, 0.65, INK);
  txt(s, "oₜ = Σⱼ₌₁…ₜ softmax(qₜ·kⱼ / √d) · vⱼ     ← 只有 qₜ 是新的，kⱼ、vⱼ 全部来自缓存", { x: M + 0.3, y: 6.05, w: CW - 0.6, h: 0.65, fontSize: 16, color: WHITE, valign: "middle" });
}

// 39. KV decode (animated)
{
  const s = slideOf("CONTENT", "kv_decode");
  title(s, "动画：预填充 + 逐步解码");
  figure(s, "kvdecode", M, 1.35, CW, 4.6, { anim: "kvdecode", alignY: "t" });
  card(s, M, 6.0, 5.95, 0.75, "预填充：整段提示一次送入，并行计算，只做一次", null, TEAL, { headPt: 14.5, headH: 0.3 });
  card(s, 6.783, 6.0, 5.95, 0.75, "逐步解码：每步只有 1 个 token 穿过网络", null, CORAL, { headPt: 14.5, headH: 0.3 });
}

// 40. KV code
{
  const s = slideOf("CONTENT", "kv_code");
  title(s, "代码：三个要点和一个容易出错的地方");
  txt(s, "src/learn_gpt2/gpt2.py（节选）", { x: M, y: 1.3, w: 6, h: 0.35, fontSize: 13, color: SLATE, fontFace: CODE_FONT });
  codePanel(s, "kvloop", M, 1.7, 7.1, 5.0, 12);
  card(s, 7.95, 1.45, 4.78, 2.25, "掩码的三种情形", "T = L：没有缓存，普通因果掩码\nT = 1：单个新 token，不需要掩码\n1 < T < L：不能用 is_causal（它对齐左上角），要把下三角右移 L − T 格", CORAL, { bodyPt: 12.5, headPt: 15, headH: 0.34 });
  figure(s, "kv_masks", 7.95, 3.85, 4.78, 2.85);
}

// 41. KV benchmark
{
  const s = slideOf("CONTENT", "kv_benchmark");
  title(s, "收益：省掉的是最重的矩阵乘法");
  table(s, ["", "不用缓存", "KV Cache"], [
    ["token 前向次数", "≈ N²/2", "≈ N"],
    ["每步新算", "整段序列", "1 个 token"],
  ], { x: M, y: 1.5, w: 5.3, colW: [1.9, 1.7, 1.7], rowH: 0.55, fontSize: 14 });
  txt(s, "省掉：QKV 投影、输出投影、前馈网络。注意力仍要读全部历史 K、V，累计仍是 N² 量级，但便宜得多", { x: M, y: 3.35, w: 5.3, h: 1.0, fontSize: 13.5, color: INK2 });
  stat(s, M, 4.4, 2.55, "8.19×", "gpt2：19.1 s → 2.3 s", TEAL, { bigPt: 34, bigH: 0.75, labelH: 0.4 });
  stat(s, M + 2.75, 4.4, 2.55, "12.41×", "gpt2-xl：256 s → 20.6 s", CORAL, { bigPt: 34, bigH: 0.75, labelH: 0.4 });
  txt(s, "贪心生成 1000 个 token，两种方式输出完全一致；模型越大加速越明显", { x: M, y: 5.75, w: 5.3, h: 0.8, fontSize: 13, color: SLATE });
  figure(s, "kv_bench", 6.1, 1.45, 6.63, 5.25);
}

// 42. KV memory
{
  const s = slideOf("CONTENT", "kv_memory");
  title(s, "代价：显存");
  rect(s, M, 1.5, CW, 0.95, INK);
  txt(s, "缓存字节 = 2 × 层数 × 批大小 × 序列长度 × d_model × 每个数的字节数", { x: M + 0.3, y: 1.5, w: CW - 0.6, h: 0.95, fontSize: 19, bold: true, color: WHITE, valign: "middle" });
  table(s, ["模型", "层数", "d_model", "精度", "长度", "单条序列"], [
    ["GPT-2 small", "12", "768", "fp32", "1024", "72 MiB"],
    ["gpt2-xl", "48", "1600", "fp16", "1024", "300 MiB"],
    ["GPT-3 175B", "96", "12288", "fp16", "2048", "9 GiB"],
  ], { x: M, y: 2.75, w: 6.6, colW: [1.6, 0.8, 1.1, 0.9, 0.95, 1.25], rowH: 0.55, fontSize: 13.5, highlight: [2] });
  txt(s, "GPT-3 每个 token 就要 4.5 MiB；同时服务几十个用户，缓存总量很快超过模型权重本身", { x: M, y: 5.05, w: 6.6, h: 0.8, fontSize: 14, color: INK2 });
  banner(s, M, 5.9, 6.6, 0.75, [{ text: "用显存换计算：", options: { bold: true } }, { text: "显存容量与带宽成了新瓶颈" }], CORAL, { pt: 15 });
  figure(s, "kv_memory", 7.45, 2.75, 5.28, 3.9);
}

// 43. Beyond
{
  const s = slideOf("CONTENT", "kv_beyond");
  title(s, "之后的改进方向（超出三篇论文的范围）");
  const cards = [
    ["减少要缓存的头", "MQA：所有查询头共享一组 K、V\nGQA：每组查询头共享一组 K、V\n缓存缩小好几倍", TEAL],
    ["更好地管理显存", "PagedAttention：像操作系统分页一样管理缓存\n减少碎片，提高并发", INDIGO],
    ["限制长度或精度", "滑动窗口注意力：只保留最近若干 token\nKV 量化：用更低精度存缓存", AMBER],
    ["改变位置编码", "RoPE：把位置写进 q、k 的旋转\n截断缓存时不必整体失效", CORAL],
  ];
  cards.forEach(([h, b, c], i) => card(s, M + (i % 2) * 6.17, 1.5 + Math.floor(i / 2) * 2.32, 5.96, 2.12, h, b, c, { num: String(i + 1), bodyPt: 14, headPt: 17, headH: 0.4 }));
  banner(s, M, 6.15, CW, 0.55, "它们都建立在因果自注意力与 KV Cache 之上", SLATE, { pt: 14, align: "center" });
}

// 44. Summary
{
  const s = slideOf("CONTENT", "summary");
  title(s, "总结");
  const items = [
    ["注意力", "位置之间的信息交换 = 矩阵乘法：softmax(QKᵀ/√d_k)·V，完全并行，任意两点一步直达", TEAL],
    ["因果掩码", "softmax 前把上三角设为 −∞：位置 i 只依赖前 i 个 token，且在所有层成立", CORAL],
    ["GPT-2", "去掉编码器，只做语言建模：预测下一个 token 本身就是多任务学习", INDIGO],
    ["GPT-3", "放大到 1750 亿参数：上下文学习涌现，写几个示例就能做新任务", AMBER],
    ["KV Cache", "因果性在推理时的红利：缓存每层 K、V，每步只算一个新 token，用显存换计算", DEEP],
  ];
  items.forEach(([h, b, c], i) => {
    const y = 1.45 + i * 0.93;
    rect(s, M, y, CW, 0.8, LIGHT);
    circleNum(s, M + 0.18, y + 0.15, 0.5, String(i + 1), c, WHITE, 15);
    txt(s, h, { x: M + 0.85, y, w: 1.9, h: 0.8, fontSize: 17, bold: true, valign: "middle" });
    txt(s, b, { x: M + 2.75, y, w: CW - 3.0, h: 0.8, fontSize: 14.5, color: INK2, valign: "middle" });
  });
  txt(s, "从 Transformer 到 GPT-3，结构几乎没变；变的是数据、规模，以及我们使用模型的方式。", { x: M, y: 6.2, w: CW, h: 0.5, fontSize: 15, bold: true, color: TEAL, align: "center", valign: "middle" });
}

// 45. End
{
  const s = slideOf("DARK_TITLE", "end");
  title(s, "感谢收看");
  s.addText("综述、幻灯片和讲稿：presentations/05-attention-gpt2-gpt3/", { placeholder: "body" });
  txt(s, "代码对照：src/learn_gpt2/gpt2.py", { x: M, y: 4.0, w: CW, h: 0.5, fontSize: 18, color: "E6E9F2", fontFace: CODE_FONT });
  const refs = [
    "Vaswani et al. Attention Is All You Need. NeurIPS 2017. arXiv:1706.03762",
    "Radford et al. Language Models are Unsupervised Multitask Learners. OpenAI 2019",
    "Brown et al. Language Models are Few-Shot Learners. NeurIPS 2020. arXiv:2005.14165",
  ];
  refs.forEach((r, i) => txt(s, r, { x: M, y: 5.2 + i * 0.42, w: CW, h: 0.4, fontSize: 13, color: SLATE }));
}

if (slideCount !== NARR.length) throw new Error(`built ${slideCount} slides, narration has ${NARR.length}`);
fs.writeFileSync(path.join(BUILD, "anim_boxes.json"), JSON.stringify(animBoxes, null, 1));
// pptxgenjs 给段落里的每个 run 都写一个 <a:pPr>, 渲染器取最后一个, 于是多 run 的条目丢了项目符号.
// 写出后只保留每个段落的第一个 <a:pPr>.
const JSZip = require("jszip");
const PPR = /<a:pPr\b[^>]*?(?:\/>|>[\s\S]*?<\/a:pPr>)/g;
pres
  .write({ outputType: "nodebuffer" })
  .then((buf) => JSZip.loadAsync(buf))
  .then(async (zip) => {
    for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
      const xml = await zip.file(name).async("string");
      const fixed = xml.replace(/<a:p>[\s\S]*?<\/a:p>/g, (p) => {
        let seen = false;
        return p.replace(PPR, (m) => (seen ? "" : ((seen = true), m)));
      });
      zip.file(name, fixed);
    }
    fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
    console.log(`wrote ${OUT} (${slideCount} slides)`);
  });
