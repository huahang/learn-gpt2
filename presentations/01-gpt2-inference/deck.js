// Build gpt2_inference.pptx: GPT-2 inference and the KV cache, explained against gpt2.py.
//
//   NODE_PATH=<dir with pptxgenjs>/node_modules node deck.js <path to pptx skill apply_theme.js>
//
// Reads build/deck_data.json (prepare.py) and build/figures/*.png (figures.py).
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.argv[2]);

const HERE = __dirname;
const DATA = JSON.parse(fs.readFileSync(path.join(HERE, "build", "deck_data.json"), "utf8"));
const FIG = (name) => path.join(HERE, "build", "figures", name);
const OUT = path.join(HERE, "gpt2_inference.pptx");

const HEX = {
  ink: "1B1F3B", white: "FFFFFF", ink2: "2D3250", light: "F2F4F8",
  teal: "2A9D8F", amber: "F2A541", coral: "E76F51", indigo: "5B6CFF", slate: "8D99AE", deep: "264653",
  mid: "D9DEE8",
};
const THEME = {
  name: "GPT-2 Ink",
  headFontFace: "Microsoft YaHei",
  bodyFontFace: "Microsoft YaHei",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.ink2, lt2: HEX.light,
    accent1: HEX.teal, accent2: HEX.amber, accent3: HEX.coral, accent4: HEX.indigo,
    accent5: HEX.slate, accent6: HEX.deep, hlink: HEX.indigo, folHlink: HEX.slate,
  },
};
const CODE_FONT = "Consolas";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "GPT-2 推理：原理与实现";
pres.author = "learn-gpt2";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const INK = C.text1, WHITE = C.background1, INK2 = C.text2, LIGHT = C.background2;
const TEAL = C.accent1, AMBER = C.accent2, CORAL = C.accent3, INDIGO = C.accent4, SLATE = C.accent5;

const W = 13.333, M = 0.6, CW = W - 2 * M; // content width 12.133

// ------------------------------------------------------------------------------------------
// Layouts
// ------------------------------------------------------------------------------------------
pres.defineSlideMaster({
  title: "DARK_TITLE",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 1.7, w: CW, h: 1.3, fontSize: 48, bold: true, color: WHITE, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.15, w: CW, h: 0.8, fontSize: 22, color: C.accent5, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "DARK_CONTENT",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: CW, h: 0.85, fontSize: 36, bold: true, color: WHITE, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION",
  background: { color: INK },
  objects: [
    { placeholder: { options: { name: "num", type: "body", x: M, y: 1.55, w: 4, h: 1.4, fontSize: 80, bold: true, color: AMBER, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 3.05, w: CW, h: 1.1, fontSize: 44, bold: true, color: WHITE, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.2, w: CW, h: 0.7, fontSize: 20, color: SLATE, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.38, w: CW, h: 0.85, fontSize: 32, bold: true, color: INK, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "GPT-2 推理：原理与实现", options: { x: M, y: 7.0, w: 6, h: 0.3, fontSize: 10, color: SLATE, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontSize: 10, color: SLATE, align: "right" },
});

// ------------------------------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------------------------------
let currentSection = null;
function section(title) {
  pres.addSection({ title });
  currentSection = title;
}
function slideOf(masterName, id) {
  const s = pres.addSlide({ masterName, sectionTitle: currentSection });
  if (id) s.addNotes(DATA.narration[id]);
  return s;
}
function title(s, text) {
  s.addText(text, { placeholder: "title" });
}
function txt(s, text, o) {
  s.addText(text, { isTextBox: true, margin: 0, fontSize: 16, color: INK, valign: "top", ...o });
}
function rect(s, x, y, w, h, fill, o = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill, ...(o.fillT !== undefined ? { transparency: o.fillT } : {}) }, line: o.line ? { color: o.line, width: o.lineW || 1 } : { type: "none" }, ...(o.name ? { objectName: o.name } : {}) });
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
// Width of a string in inches: CJK glyphs are 1 em, the rest about 0.55 em.
function estW(text, pt) {
  let em = 0;
  for (const ch of text) em += /[⺀-￯]/.test(ch) ? 1 : 0.56;
  return (em * pt) / 72;
}
function chip(s, x, y, text, fill, o = {}) {
  const pt = o.pt || 16;
  const w = o.w || Math.max(0.7, estW(text, pt) + 0.36);
  const h = o.h || 0.62;
  rect(s, x, y, w, h, fill, { line: o.line });
  txt(s, text, { x, y, w, h, fontSize: pt, bold: true, color: o.color || WHITE, align: "center", valign: "middle" });
  return w;
}
function codePanel(s, key, x, y, w, h, pt = 12) {
  rect(s, x, y, w, h, INK, { name: `code-${key}` });
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
  s.addText(runs, { isTextBox: true, x: x + 0.22, y: y + 0.16, w: w - 0.44, h: h - 0.32, margin: 0, valign: "top", fontFace: CODE_FONT, fontSize: pt, objectName: `code-text-${key}` });
}
function card(s, x, y, w, h, head, body, accent, o = {}) {
  rect(s, x, y, w, h, o.fill || LIGHT, { fillT: o.fillT });
  const pad = 0.22;
  let tx = x + pad;
  if (o.num) {
    circleNum(s, x + pad, y + pad, 0.5, o.num, accent, WHITE, 15);
    tx = x + pad + 0.68;
  }
  txt(s, head, { x: tx, y: y + pad, w: x + w - pad - tx, h: 0.5, fontSize: o.headPt || 18, bold: true, color: o.headColor || INK, valign: "middle" });
  if (body) txt(s, body, { x: tx, y: y + pad + 0.58, w: x + w - pad - tx, h: h - pad * 2 - 0.58, fontSize: o.bodyPt || 15, color: o.bodyColor || INK2 });
}
function stat(s, x, y, w, big, label, color, o = {}) {
  txt(s, big, { x, y, w, h: o.bigH || 0.9, fontSize: o.bigPt || 44, bold: true, color, valign: "bottom" });
  txt(s, label, { x, y: y + (o.bigH || 0.9) + 0.06, w, h: o.labelH || 0.6, fontSize: o.labelPt || 14, color: INK2 });
}
function image(s, name, x, y, w, h, alt) {
  s.addImage({ path: FIG(name), x, y, w, h, altText: alt });
}
const chartText = {
  catAxisLabelColor: HEX.slate, valAxisLabelColor: HEX.slate,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  titleFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, legendFontSize: 12,
  titleColor: HEX.ink, titleFontSize: 14, legendColor: HEX.ink2,
  valGridLine: { color: HEX.mid, size: 0.5 }, catGridLine: { style: "none" },
  catAxisLineColor: HEX.mid, valAxisLineShow: false,
};

// ------------------------------------------------------------------------------------------
// Slides
// ------------------------------------------------------------------------------------------
const dist = DATA.distribution;
const lat = DATA.latency;
const runs = DATA.runs;
const flops = DATA.flops;

// 1. Title -----------------------------------------------------------------------------------
section("开场");
{
  const s = slideOf("DARK_TITLE", "s01_title");
  title(s, "GPT-2 推理：原理与实现");
  s.addText("逐行读懂 src/learn_gpt2/gpt2.py，看清 KV Cache 为什么能加速", { placeholder: "body" });
  let x = M;
  const toks = dist.token_strs.map((t) => t.trim() || "␣");
  toks.forEach((t) => { x += chip(s, x, 4.75, t, TEAL, { pt: 16 }) + 0.14; });
  line(s, x + 0.05, 5.06, x + 0.6, 5.06, AMBER, { arrow: true, width: 2.5 });
  chip(s, x + 0.75, 4.75, dist.greedy.trim(), AMBER, { color: INK });
  txt(s, "每次只预测下一个 token —— 本讲的全部故事都从这里开始", { x: M, y: 5.6, w: 9, h: 0.4, fontSize: 14, color: SLATE });
}

// 2. Agenda ------------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s02_agenda");
  title(s, "这一讲的路线");
  const items = [
    ["01", "自回归生成", "模型如何一个 token 接一个 token 地写出文本", TEAL],
    ["02", "GPT-2 前向计算", "Embedding、注意力、MLP，以及每一步的张量形状", INDIGO],
    ["03", "朴素生成的瓶颈", "为什么不加缓存时，越往后写越慢", CORAL],
    ["04", "KV Cache", "原理、gpt2.py 中的实现、计算量与实测加速", AMBER],
  ];
  const w = 2.8, gap = (CW - 4 * w) / 3;
  items.forEach(([n, h, b, c], i) => {
    const x = M + i * (w + gap), y = 1.6;
    rect(s, x, y, w, 3.7, LIGHT);
    txt(s, n, { x: x + 0.3, y: y + 0.35, w: 2, h: 1.0, fontSize: 54, bold: true, color: c, valign: "middle" });
    txt(s, h, { x: x + 0.3, y: y + 1.55, w: w - 0.6, h: 0.6, fontSize: 20, bold: true, color: INK, valign: "middle" });
    txt(s, b, { x: x + 0.3, y: y + 2.3, w: w - 0.6, h: 1.2, fontSize: 15, color: INK2 });
  });
  rect(s, M, 5.6, CW, 1.05, AMBER, { fillT: 82 });
  txt(s, [
    { text: "主线问题：", options: { bold: true, color: INK } },
    { text: "为什么加上 KV Cache，同样的模型生成文本能快一个数量级？", options: { color: INK } },
  ], { x: M + 0.3, y: 5.6, w: CW - 0.6, h: 1.05, fontSize: 20, valign: "middle" });
}

// 3. Section 01 --------------------------------------------------------------------------------
function sectionSlide(id, num, head, sub) {
  const s = slideOf("SECTION", id);
  s.addText(num, { placeholder: "num" });
  s.addText(head, { placeholder: "title" });
  s.addText(sub, { placeholder: "body" });
  // token-chip motif along the bottom
  for (let i = 0; i < 6; i++) rect(s, M + i * 1.05, 6.1, 0.9, 0.45, i === 5 ? AMBER : TEAL, { fillT: i === 5 ? 0 : 55 });
  return s;
}
section("01 自回归生成");
sectionSlide("s03_sec1", "01", "自回归生成", "语言模型如何一步一步写出文本");

// 4. Next-token prediction -----------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s04_next_token");
  title(s, "语言模型只做一件事：预测下一个 token");
  rect(s, M, 1.5, 6.0, 0.66, LIGHT);
  txt(s, `"${dist.prompt}"`, { x: M + 0.25, y: 1.5, w: 5.6, h: 0.66, fontSize: 18, valign: "middle" });
  line(s, M + 3.0, 2.22, M + 3.0, 2.68, SLATE, { arrow: true });
  txt(s, "tiktoken 分词", { x: M + 3.2, y: 2.25, w: 2.5, h: 0.4, fontSize: 13, color: SLATE, valign: "middle" });
  const cw = 1.38, cg = (6.0 - 4 * cw) / 3;
  dist.token_strs.forEach((t, i) => {
    const x = M + (i % 4) * (cw + cg), y = 2.78 + Math.floor(i / 4) * 1.05;
    rect(s, x, y, cw, 0.92, TEAL);
    txt(s, [
      { text: t.trim() || "␣", options: { fontSize: 16, bold: true, color: WHITE, breakLine: true } },
      { text: String(dist.token_ids[i]), options: { fontSize: 12, color: WHITE } },
    ], { x, y: y + 0.06, w: cw, h: 0.8, align: "center", valign: "middle" });
  });
  line(s, M + 3.0, 4.82, M + 3.0, 5.25, SLATE, { arrow: true });
  rect(s, M, 5.32, 6.0, 0.9, INK);
  txt(s, [
    { text: "GPT-2", options: { fontSize: 20, bold: true, color: WHITE, breakLine: true } },
    { text: "12 层 Transformer → 每个位置输出 50257 维 logits", options: { fontSize: 13, color: C.accent5 } },
  ], { x: M + 0.25, y: 5.32, w: 5.6, h: 0.9, valign: "middle" });
  line(s, M + 6.1, 5.77, M + 6.6, 5.77, SLATE, { arrow: true });
  const top = dist.top10["1.0"].slice(0, 6);
  s.addChart(pres.charts.BAR, [{ name: "概率", labels: top.map((d) => `"${d.token.trim()}"`), values: top.map((d) => d.prob) }], {
    x: 7.3, y: 1.45, w: 5.43, h: 4.95, barDir: "bar", catAxisOrientation: "maxMin",
    chartColors: [HEX.teal], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0%",
    dataLabelColor: HEX.ink, dataLabelFontSize: 12, valAxisHidden: true, valAxisMaxVal: 0.2,
    showLegend: false, showTitle: true, title: "下一个 token 的概率（前 6 名）",
    barGapWidthPct: 45, ...chartText, catAxisLabelFontSize: 14, catAxisLabelColor: HEX.ink,
  });
  txt(s, "输出是整个词表上的概率分布，生成时从中挑出一个 token", { x: 7.3, y: 6.45, w: 5.43, h: 0.4, fontSize: 13, color: SLATE });
}

// 5. Generation loop ---------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s05_loop");
  title(s, "生成 = 反复调用模型的循环");
  const steps = [
    ["前向计算", "logits = model(idx)    (B, T, 50257)"],
    ["只取最后一个位置", "logits[:, -1, :]    (B, 50257)"],
    ["采样下一个 token", "贪心 / 温度 / top-k"],
    ["拼接到序列末尾", "idx = cat(idx, next)，T → T + 1"],
  ];
  const bx = 1.35, bw = 4.95, bh = 0.92, gap = 0.3;
  steps.forEach(([h, b], i) => {
    const y = 1.55 + i * (bh + gap);
    rect(s, bx, y, bw, bh, LIGHT);
    circleNum(s, bx + 0.2, y + 0.21, 0.5, String(i + 1), TEAL);
    txt(s, h, { x: bx + 0.88, y: y + 0.1, w: bw - 1.0, h: 0.4, fontSize: 17, bold: true, valign: "middle" });
    txt(s, b, { x: bx + 0.88, y: y + 0.5, w: bw - 1.0, h: 0.34, fontSize: 13, color: INK2, fontFace: CODE_FONT, valign: "middle" });
    if (i < 3) line(s, bx + bw / 2, y + bh + 0.03, bx + bw / 2, y + bh + gap - 0.03, SLATE, { arrow: true });
  });
  const yTop = 1.55 + bh / 2, yBot = 1.55 + 3 * (bh + gap) + bh / 2;
  line(s, bx, yBot, 0.8, yBot, AMBER, { width: 2.5 });
  line(s, 0.8, yBot, 0.8, yTop, AMBER, { width: 2.5 });
  line(s, 0.8, yTop, bx - 0.02, yTop, AMBER, { width: 2.5, arrow: true });
  txt(s, "↻ 重复 max_new_tokens 次", { x: bx, y: 6.25, w: bw, h: 0.4, fontSize: 14, color: AMBER, bold: true, align: "center", valign: "middle" });
  codePanel(s, "loop", 6.85, 1.55, 5.88, 1.75, 13);
  rect(s, 6.85, 3.6, 5.88, 3.05, CORAL, { fillT: 88 });
  txt(s, "每生成 1 个 token", { x: 7.15, y: 3.9, w: 5.3, h: 0.5, fontSize: 18, color: INK2 });
  txt(s, "= 跑 1 次完整前向", { x: 7.15, y: 4.45, w: 5.3, h: 0.9, fontSize: 32, bold: true, color: CORAL, valign: "middle" });
  txt(s, "而输入序列每一步都变长一个 token，所以前向一次比一次贵", { x: 7.15, y: 5.55, w: 5.3, h: 0.8, fontSize: 15, color: INK2 });
}

// 6. Section 02 --------------------------------------------------------------------------------
section("02 GPT-2 前向计算");
sectionSlide("s06_sec2", "02", "GPT-2 的前向计算", "从 token 编号到 logits，张量形状一路跟踪");

// 7. Configs -----------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s07_configs");
  title(s, "四种尺寸，一套结构");
  const head = ["型号", "层数 n_layer", "头数 n_head", "宽度 n_embd", "参数量"].map((t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: INK } } }));
  const rows = [
    ["gpt2", "12", "12", "768", "124M"],
    ["gpt2-medium", "24", "16", "1024", "355M"],
    ["gpt2-large", "36", "20", "1280", "774M"],
    ["gpt2-xl", "48", "25", "1600", "1558M"],
  ].map((r, i) => r.map((t) => ({ text: t, options: { fill: { color: i === 0 ? C.accent1 : i % 2 ? LIGHT : WHITE, ...(i === 0 ? { transparency: 82 } : {}) }, bold: i === 0, color: INK } })));
  s.addTable([head, ...rows], {
    x: M, y: 1.6, w: 7.6, colW: [1.9, 1.45, 1.4, 1.45, 1.4], rowH: 0.6, fontSize: 16, align: "center", valign: "middle",
    border: { type: "solid", pt: 1, color: HEX.mid },
  });
  txt(s, "结构完全相同，只是层数、头数和宽度不同。本讲以第一行的 gpt2（124M）为例。", { x: M, y: 4.85, w: 7.6, h: 0.8, fontSize: 15, color: INK2 });
  const stats = [["50257", "词表大小 vocab_size", TEAL], ["1024", "最大上下文 block_size（学出来的位置编码）", AMBER], ["64", "每个头的维度 n_embd ÷ n_head（四种尺寸都一样）", INDIGO]];
  stats.forEach(([b, l, c], i) => stat(s, 8.75, 1.45 + i * 1.75, 3.98, b, l, c, { bigPt: 44, bigH: 0.85, labelH: 0.7 }));
}

// 8. Architecture ------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s08_arch");
  title(s, "整体结构：Embedding → 12 个 Block → lm_head");
  const x = M, w = 5.4;
  const arrowDown = (y1, y2) => line(s, x + w / 2, y1, x + w / 2, y2, SLATE, { arrow: true, width: 1.75 });
  rect(s, x, 1.45, w, 0.55, LIGHT);
  txt(s, "idx：token 编号    (B, T)", { x: x + 0.25, y: 1.45, w: w - 0.5, h: 0.55, fontSize: 15, valign: "middle" });
  arrowDown(2.03, 2.25);
  rect(s, x, 2.3, 2.45, 0.85, TEAL);
  txt(s, [{ text: "wte 词嵌入", options: { bold: true, breakLine: true } }, { text: "50257 × 768", options: { fontSize: 12 } }], { x, y: 2.3, w: 2.45, h: 0.85, fontSize: 15, color: WHITE, align: "center", valign: "middle" });
  txt(s, "+", { x: x + 2.45, y: 2.3, w: 0.5, h: 0.85, fontSize: 26, bold: true, color: INK2, align: "center", valign: "middle" });
  rect(s, x + 2.95, 2.3, 2.45, 0.85, INDIGO);
  txt(s, [{ text: "wpe 位置嵌入", options: { bold: true, breakLine: true } }, { text: "1024 × 768", options: { fontSize: 12 } }], { x: x + 2.95, y: 2.3, w: 2.45, h: 0.85, fontSize: 15, color: WHITE, align: "center", valign: "middle" });
  arrowDown(3.18, 3.4);
  txt(s, "(B, T, 768)", { x: x + w / 2 + 0.15, y: 3.15, w: 2, h: 0.3, fontSize: 11, color: SLATE, fontFace: CODE_FONT, valign: "middle" });
  rect(s, x, 3.45, w, 1.25, INK);
  txt(s, [{ text: "Transformer Block × 12", options: { fontSize: 20, bold: true, breakLine: true } }, { text: "LN → 因果自注意力 → 残差 → LN → MLP → 残差", options: { fontSize: 13, color: C.accent5 } }], { x: x + 0.25, y: 3.45, w: w - 0.5, h: 1.25, color: WHITE, valign: "middle" });
  arrowDown(4.73, 4.95);
  rect(s, x, 5.0, w, 0.5, LIGHT);
  txt(s, "ln_f：最后一次 LayerNorm", { x: x + 0.25, y: 5.0, w: w - 0.5, h: 0.5, fontSize: 15, valign: "middle" });
  arrowDown(5.53, 5.75);
  rect(s, x, 5.8, w, 0.5, AMBER);
  txt(s, "lm_head：768 → 50257（与 wte 共享权重）", { x: x + 0.25, y: 5.8, w: w - 0.5, h: 0.5, fontSize: 15, bold: true, color: INK, valign: "middle" });
  txt(s, "→ logits  (B, T, 50257)", { x: x, y: 6.38, w: w, h: 0.4, fontSize: 14, color: INK2, fontFace: CODE_FONT, align: "center", valign: "middle" });
  codePanel(s, "forward", 6.33, 1.45, 6.4, 2.6, 12);
  rect(s, 6.33, 4.35, 6.4, 1.6, INDIGO, { fillT: 88 });
  txt(s, [
    { text: "位置编码 = 按位置编号查表", options: { bold: true, fontSize: 17, color: INK, breakLine: true } },
    { text: "pos 只取决于 token 在序列中的编号，讲 KV Cache 时会用到", options: { fontSize: 14, color: INK2 } },
  ], { x: 6.63, y: 4.35, w: 5.9, h: 1.6, valign: "middle" });
}

// 9. Block ---------------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s09_block");
  title(s, "Transformer Block：Pre-LN + 残差");
  const cx = 3.55, bw = 3.7, bx = cx - bw / 2;
  txt(s, "x   (B, T, 768)", { x: bx, y: 1.45, w: bw, h: 0.4, fontSize: 14, fontFace: CODE_FONT, align: "center", valign: "middle", color: INK2 });
  const boxes = [
    [1.98, 0.5, LIGHT, INK, "ln_1：LayerNorm"],
    [2.68, 0.72, TEAL, WHITE, "因果自注意力"],
    [4.15, 0.5, LIGHT, INK, "ln_2：LayerNorm"],
    [4.85, 0.72, AMBER, INK, "MLP：768 → 3072 → GELU → 768"],
  ];
  boxes.forEach(([y, h, f, c, t]) => {
    rect(s, bx, y, bw, h, f);
    txt(s, t, { x: bx, y, w: bw, h, fontSize: 15, bold: true, color: c, align: "center", valign: "middle" });
  });
  const plus = (y) => {
    s.addShape(pres.shapes.OVAL, { x: cx - 0.22, y, w: 0.44, h: 0.44, fill: { color: WHITE }, line: { color: INK2, width: 1.5 } });
    txt(s, "+", { x: cx - 0.22, y: y - 0.02, w: 0.44, h: 0.44, fontSize: 20, bold: true, color: INK2, align: "center", valign: "middle" });
  };
  plus(3.55); plus(5.72);
  const a = (y1, y2) => line(s, cx, y1, cx, y2, SLATE, { arrow: true, width: 1.5 });
  a(1.86, 1.96); a(2.5, 2.66); a(3.42, 3.53); a(4.0, 4.13); a(4.67, 4.83); a(5.59, 5.7); a(6.17, 6.45);
  // residual bypasses
  line(s, cx, 1.9, 1.15, 1.9, INK2, { width: 1.5 }); line(s, 1.15, 1.9, 1.15, 3.77, INK2, { width: 1.5 }); line(s, 1.15, 3.77, cx - 0.24, 3.77, INK2, { width: 1.5, arrow: true });
  line(s, cx, 4.06, 1.15, 4.06, INK2, { width: 1.5 }); line(s, 1.15, 4.06, 1.15, 5.94, INK2, { width: 1.5 }); line(s, 1.15, 5.94, cx - 0.24, 5.94, INK2, { width: 1.5, arrow: true });
  txt(s, "残差", { x: 0.3, y: 2.6, w: 0.75, h: 0.4, fontSize: 12, color: INK2, align: "center" });
  txt(s, "残差", { x: 0.3, y: 4.8, w: 0.75, h: 0.4, fontSize: 12, color: INK2, align: "center" });
  txt(s, "输出 (B, T, 768)", { x: cx + 0.15, y: 6.4, w: 2.5, h: 0.35, fontSize: 12, color: SLATE, valign: "middle" });
  codePanel(s, "block", 6.6, 1.45, 6.13, 2.25, 13);
  rect(s, 6.6, 4.0, 6.13, 2.45, TEAL, { fillT: 85 });
  txt(s, "关键事实", { x: 6.9, y: 4.2, w: 5.5, h: 0.45, fontSize: 18, bold: true, color: TEAL, valign: "middle" });
  txt(s, [
    { text: "LayerNorm、MLP 和所有 Linear 层都逐位置独立计算。", options: { breakLine: true } },
    { text: "在不同位置之间传递信息的，只有注意力这一处。", options: { bold: true } },
  ], { x: 6.9, y: 4.75, w: 5.55, h: 1.5, fontSize: 17, color: INK, paraSpaceAfter: 6 });
}

// 10. QKV --------------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s10_qkv");
  title(s, "因果自注意力 ①：Q、K、V 从哪里来");
  image(s, "qkv.png", M, 1.35, 7.25, 4.1, "x 经过 c_attn 得到 2304 维，切成 q、k、v，再拆成 12 个 64 维的头");
  const items = [["q", "Query：我在找什么", INDIGO], ["k", "Key：我能提供什么线索", TEAL], ["v", "Value：我实际携带的内容", AMBER]];
  items.forEach(([l, t, c], i) => {
    const y = 1.6 + i * 1.22;
    circleNum(s, 8.35, y, 0.62, l, c, WHITE, 20);
    txt(s, t, { x: 9.15, y, w: 3.58, h: 0.62, fontSize: 17, bold: true, valign: "middle" });
  });
  txt(s, "每个头独立做一次注意力，最后拼回 768 维，经 c_proj 输出", { x: 8.35, y: 5.0, w: 4.38, h: 0.6, fontSize: 13, color: INK2 });
  codePanel(s, "qkv", M, 5.6, CW, 1.25, 11);
}

// 11. Causal mask --------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s11_mask");
  title(s, "因果自注意力 ②：只看自己和过去");
  image(s, "attention.png", M, 1.35, 5.55, 5.35, "真实 GPT-2 第 0 层第 7 个头的注意力权重，右上角被因果掩码遮住");
  rect(s, 6.55, 1.45, 6.18, 1.45, LIGHT);
  txt(s, [
    { text: "y = softmax( Q·K", options: {} }, { text: "T", options: { superscript: true } },
    { text: " / √64 + M ) · V", options: { breakLine: true } },
    { text: "M：对角线以上为 −∞，softmax 后权重为 0", options: { fontSize: 14, color: INK2 } },
  ], { x: 6.8, y: 1.45, w: 5.7, h: 1.45, fontSize: 22, bold: false, color: INK, valign: "middle", fontFace: "Cambria" });
  const bullets = [
    "第 i 行：位置 i 对 0…i 的注意力权重，和为 1",
    "右上角被遮住：未来的 token 不会影响过去的位置",
    "例：model 这一行最关注 language（权重 0.42）",
  ];
  txt(s, bullets.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < bullets.length - 1 } })), { x: 6.55, y: 3.15, w: 6.18, h: 2.1, fontSize: 16, color: INK, paraSpaceAfter: 10 });
  rect(s, 6.55, 5.6, 6.18, 0.75, INK);
  txt(s, [
    { text: "F.scaled_dot_product_attention", options: { color: "82AAFF" } },
    { text: "(q, k, v, is_causal=", options: { color: "E6E9F2" } },
    { text: "True", options: { color: "C792EA" } },
    { text: ")", options: { color: "E6E9F2" } },
  ], { x: 6.75, y: 5.6, w: 5.9, h: 0.75, fontFace: CODE_FONT, fontSize: 12.5, valign: "middle" });
}

// 12. Sampling ------------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s12_sampling");
  title(s, "从 logits 到下一个 token：三种选法");
  const t1 = dist.top10["1.0"].slice(0, 8), t07 = dist.top10["0.7"].slice(0, 8);
  const labels = t1.map((d) => `"${d.token.trim()}"`);
  s.addChart(pres.charts.BAR, [
    { name: "temperature = 1.0", labels, values: t1.map((d) => d.prob) },
    { name: "temperature = 0.7", labels, values: t07.map((d) => d.prob) },
  ], {
    x: M, y: 1.45, w: 7.0, h: 5.2, barDir: "col", barGrouping: "clustered", chartColors: [HEX.slate, HEX.indigo],
    valAxisLabelFormatCode: "0%", valAxisMaxVal: 0.3, valAxisMinVal: 0, showLegend: true, legendPos: "t",
    showTitle: true, title: "前 8 个候选的概率：温度越低，分布越尖", barGapWidthPct: 60, ...chartText,
    catAxisLabelColor: HEX.ink, catAxisLabelFontSize: 13,
  });
  const cards = [
    ["贪心解码", "temperature = 0：直接取 argmax，结果确定", TEAL],
    ["温度 temperature", "logits ÷ T：T < 1 更保守，T > 1 更随机", INDIGO],
    ["Top-k 采样", "只留概率最高的 k 个，其余设为 −∞，再用 multinomial 抽样", AMBER],
  ];
  cards.forEach(([h, b, c], i) => card(s, 8.0, 1.5 + i * 1.72, 4.73, 1.5, h, b, c, { num: String(i + 1), headPt: 17, bodyPt: 14 }));
}

// 13. Section 03 --------------------------------------------------------------------------------
section("03 朴素生成的瓶颈");
sectionSlide("s13_sec3", "03", "朴素生成的瓶颈", "为什么不加缓存时，越往后写越慢");

// 14. No cache ---------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s14_nocache");
  title(s, "不加缓存：每一步都重算整个前缀");
  image(s, "nocache_grid.png", M, 1.35, 7.3, 5.31, "每一步都把全部旧 token 重新计算一遍，红色是重复计算");
  const pts = [
    "第 t 步：t 个 token 全部重跑 12 层",
    "线性层累计 ≈ N²/2 个 token",
    "注意力每步 t × t 个分数，累计 ≈ N³/3",
  ];
  txt(s, pts.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < pts.length - 1 } })), { x: 8.2, y: 1.5, w: 4.53, h: 1.9, fontSize: 16, paraSpaceAfter: 10 });
  txt(s, "8 个 token 的 prompt，生成 1000 个 token：", { x: 8.2, y: 3.55, w: 4.53, h: 0.4, fontSize: 14, color: INK2 });
  stat(s, 8.2, 3.95, 4.53, "507,500", "次 token 前向（不加缓存）", CORAL, { bigPt: 40, bigH: 0.75, labelH: 0.4 });
  stat(s, 8.2, 5.25, 4.53, "1,007", "次就够了（每个 token 只算一次）", TEAL, { bigPt: 40, bigH: 0.75, labelH: 0.4 });
}

// 15. Observation ------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s15_observation");
  title(s, "关键观察：旧 token 的 K、V 不会再变");
  image(s, "dependency.png", M, 1.35, 7.3, 5.31, "每个位置只依赖自己和之前的位置；新 token 不会影响旧位置");
  const reasons = [["因果掩码", "位置 i 只看位置 0…i"], ["逐位置计算", "LayerNorm、MLP、c_attn 只看本位置"], ["绝对位置编码", "旧 token 的位置编号不会变"]];
  reasons.forEach(([h, b], i) => card(s, 8.2, 1.45 + i * 1.2, 4.53, 1.05, h, null, TEAL, { num: String(i + 1), headPt: 17 }));
  reasons.forEach(([, b], i) => txt(s, b, { x: 9.1, y: 1.45 + i * 1.2 + 0.58, w: 3.5, h: 0.4, fontSize: 13, color: INK2 }));
  rect(s, 8.2, 5.15, 4.53, 1.5, TEAL);
  txt(s, [
    { text: "⇒ 旧位置的 k、v 永远不变", options: { bold: true, fontSize: 18, breakLine: true } },
    { text: "算一次、存下来、以后直接读", options: { breakLine: true } },
    { text: "这就是 KV Cache", options: { bold: true } },
  ], { x: 8.45, y: 5.15, w: 4.1, h: 1.5, fontSize: 15, color: WHITE, valign: "middle", paraSpaceAfter: 2 });
}

// 16. Section 04 --------------------------------------------------------------------------------
section("04 KV Cache");
sectionSlide("s16_sec4", "04", "KV Cache", "原理、gpt2.py 中的实现、计算量与实测加速");

// 17. KV cache phases ---------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s17_kvcache");
  title(s, "KV Cache：用显存换计算");
  image(s, "kv_phases.png", M, 1.35, CW, 4.08, "Prefill 一次写入整段 prompt 的 K、V；Decode 每步只追加一行");
  rect(s, M, 5.65, CW, 1.1, INDIGO, { fillT: 88 });
  txt(s, [
    { text: "为什么只缓存 K、V，不缓存 Q？  ", options: { bold: true, color: INK } },
    { text: "旧位置的 q 只在算它自己的输出时用过一次；旧位置的 k、v 却是每个新 token 都要读的。", options: { color: INK2 } },
  ], { x: M + 0.3, y: 5.65, w: CW - 0.6, h: 1.1, fontSize: 16, valign: "middle" });
}

// 18. Decode step -------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s18_decode");
  title(s, "解码一步：一个 query 对全部缓存");
  image(s, "decode_step.png", M, 1.3, CW, 3.66, "新 token 的 q 与缓存的 K 相乘得到分数，softmax 后与 V 相乘");
  const head = ["第 t 步的计算", "不加缓存", "使用 KV Cache"].map((t, i) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: i === 1 ? CORAL : i === 2 ? TEAL : INK } } }));
  const rows = [["输入的 token 数", "t", "1"], ["线性层与 lm_head", "∝ t", "常数"], ["注意力分数", "t × t", "1 × t"]]
    .map((r) => r.map((t, i) => ({ text: t, options: { bold: i === 2, color: INK, fill: { color: i === 0 ? LIGHT : WHITE } } })));
  s.addTable([head, ...rows], { x: 1.6, y: 5.08, w: 10.13, colW: [3.6, 3.27, 3.26], rowH: 0.4, fontSize: 14, align: "center", valign: "middle", border: { type: "solid", pt: 1, color: HEX.mid } });
}

// 19. LayerKVCache ------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s19_impl_cache");
  title(s, "实现 ①：LayerKVCache 与 KVCache");
  image(s, "buffer.png", M, 1.35, 5.6, 3.15, "预分配 max_len 的缓冲区：已缓存部分、本次写入部分、预留空位");
  const pts = [
    "预分配 max_len，免去每步 torch.cat 拷贝整个缓存",
    "首次 update 时按 batch、dtype、device 惰性分配",
    "越界或不匹配立即报错，绝不悄悄算错",
    "KVCache = 每层一个 LayerKVCache",
  ];
  txt(s, pts.map((b, i) => ({ text: b, options: { bullet: true, breakLine: i < pts.length - 1 } })), { x: M, y: 4.7, w: 5.6, h: 2.1, fontSize: 15, paraSpaceAfter: 8 });
  codePanel(s, "cache", 6.5, 1.45, 6.23, 5.25, 13);
}

// 20. Three attention cases ----------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s20_impl_masks");
  title(s, "实现 ②：带缓存的注意力，三种情形");
  image(s, "masks.png", (W - 9.6) / 2, 1.3, 9.6, 2.98, "三种掩码：预填充的下三角、解码一步的整行、带缓存多 token 的平移下三角");
  codePanel(s, "masks", M, 4.4, CW, 2.4, 12);
}

// 21. Positions and generate --------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s21_impl_pos");
  title(s, "实现 ③：位置编号与 generate()");
  image(s, "positions.png", M, 1.3, CW, 3.76, "新 token 的位置接在缓存后面；超过 block_size 时窗口滑动，位置全部改变");
  codePanel(s, "pos", M, 5.15, 4.75, 1.68, 11);
  codePanel(s, "generate", M + 4.95, 5.15, CW - 4.95, 1.68, 11);
}

// 22. FLOPs ------------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s22_flops");
  title(s, "为什么快 ①：计算量从 N³ 级降到 N² 级");
  const head = ["", "不加缓存", "KV Cache"].map((t, i) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: i === 1 ? CORAL : i === 2 ? TEAL : INK } } }));
  const rows = [["每步线性层", "∝ t·d²", "∝ d²"], ["每步注意力", "∝ t²·d", "∝ t·d"], ["N 步总计", "∝ N²d² + N³d", "∝ Nd² + N²d"]]
    .map((r) => r.map((t, i) => ({ text: t, options: { color: INK, bold: i === 0, fill: { color: i === 0 ? LIGHT : WHITE } } })));
  s.addTable([head, ...rows], { x: M, y: 1.55, w: 6.3, colW: [1.7, 2.3, 2.3], rowH: 0.55, fontSize: 16, align: "center", valign: "middle", border: { type: "solid", pt: 1, color: HEX.mid } });
  txt(s, "t：当前长度    d：隐藏宽度 768    N：生成的 token 数", { x: M, y: 3.85, w: 6.3, h: 0.35, fontSize: 12, color: SLATE });
  txt(s, "gpt2 生成 1000 个 token 的总浮点运算", { x: M, y: 4.4, w: 6.3, h: 0.4, fontSize: 14, color: INK2 });
  txt(s, [
    { text: `${flops.total_uncached_tflops}`, options: { color: CORAL } },
    { text: " → ", options: { color: SLATE } },
    { text: `${flops.total_cached_tflops}`, options: { color: TEAL } },
    { text: " 万亿次", options: { color: INK2, fontSize: 20, bold: false } },
  ], { x: M, y: 4.8, w: 6.3, h: 0.9, fontSize: 44, bold: true, valign: "middle" });
  txt(s, `少约 ${flops.ratio} 倍`, { x: M, y: 5.8, w: 6.3, h: 0.6, fontSize: 24, bold: true, color: AMBER });
  s.addChart(pres.charts.LINE, [
    { name: "不加缓存", labels: flops.labels, values: flops.uncached },
    { name: "KV Cache", labels: flops.labels, values: flops.cached },
  ], {
    x: 7.15, y: 1.4, w: 5.58, h: 5.3, chartColors: [HEX.coral, HEX.teal], lineSize: 2.5, lineDataSymbol: "none",
    showLegend: true, legendPos: "t", showTitle: true, title: "每一步的计算量（GFLOPs，gpt2 估算）",
    catAxisLabelFrequency: 20, showCatAxisTitle: true, catAxisTitle: "生成第几个 token", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.slate,
    valAxisLabelFormatCode: "#,##0", ...chartText,
  });
}

// 23. Measured ---------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s23_measured");
  title(s, "为什么快 ②：在 GPU 上实测");
  s.addChart(pres.charts.LINE, [
    { name: "不加缓存", labels: lat.labels, values: lat.gpt2_uncached },
    { name: "KV Cache", labels: lat.labels, values: lat.gpt2_cached },
  ], {
    x: M, y: 1.4, w: 7.9, h: 4.95, chartColors: [HEX.coral, HEX.teal], lineSize: 2.5, lineDataSymbol: "none",
    showLegend: true, legendPos: "t", showTitle: true, title: "gpt2 每生成一个 token 的耗时（毫秒）",
    catAxisLabelFrequency: 20, showCatAxisTitle: true, catAxisTitle: "生成第几个 token", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.slate,
    ...chartText,
  });
  const g = runs["gpt2"], xl = runs["gpt2-xl"];
  stat(s, 8.95, 1.45, 3.78, `${g.speedup.toFixed(1)}×`, `gpt2：${g.uncached_total_s.toFixed(1)} s → ${g.cached_total_s.toFixed(1)} s`, TEAL, { bigPt: 48 });
  stat(s, 8.95, 3.05, 3.78, `${xl.speedup.toFixed(1)}×`, `gpt2-xl：${xl.uncached_total_s.toFixed(0)} s → ${xl.cached_total_s.toFixed(1)} s`, TEAL, { bigPt: 48 });
  stat(s, 8.95, 4.65, 3.78, "✓ 完全一致", "两种方式生成的 1000 个 token 逐个相同", AMBER, { bigPt: 32 });
  txt(s, `${DATA.device} · fp32 · batch 1 · 8 个 token 的 prompt，贪心生成 1000 个 token · 每步取 3 次运行的中位数`, { x: M, y: 6.45, w: CW, h: 0.35, fontSize: 11, color: SLATE });
}

// 24. Bandwidth ---------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s24_bandwidth");
  title(s, "为什么实测只快 8~12 倍，而不是 516 倍？");
  stat(s, M, 1.4, 6.2, "≈ 0.5 GB", "gpt2 fp32 的全部权重：每生成一个 token，都要从显存完整读一遍", INDIGO, { bigPt: 44, labelH: 0.7 });
  const pts = [
    ["瓶颈从计算变成显存带宽", "只算 1 个 token 时，GPU 算力大多闲置；耗时 ≈ 读权重 + 启动内核"],
    ["所以缓存后每步耗时几乎恒定", "gpt2 约 2~3 ms，gpt2-xl 约 18~23 ms"],
    ["模型越大、序列越长，加速越多", "计算占比越高，省下的越多：gpt2 8.2×，gpt2-xl 12.4×"],
  ];
  pts.forEach(([h, b], i) => card(s, M, 3.2 + i * 1.2, 6.2, 1.05, h, null, INDIGO, { num: String(i + 1), headPt: 16 }));
  pts.forEach(([, b], i) => txt(s, b, { x: M + 0.9, y: 3.2 + i * 1.2 + 0.58, w: 5.15, h: 0.4, fontSize: 13, color: INK2 }));
  s.addChart(pres.charts.LINE, [
    { name: "不加缓存", labels: lat.labels, values: lat.xl_uncached },
    { name: "KV Cache", labels: lat.labels, values: lat.xl_cached },
  ], {
    x: 7.15, y: 1.4, w: 5.58, h: 5.3, chartColors: [HEX.coral, HEX.teal], lineSize: 2.5, lineDataSymbol: "none",
    showLegend: true, legendPos: "t", showTitle: true, title: "gpt2-xl 每生成一个 token 的耗时（毫秒）",
    catAxisLabelFrequency: 20, showCatAxisTitle: true, catAxisTitle: "生成第几个 token", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.slate,
    ...chartText,
  });
}

// 25. Memory -----------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s25_memory");
  title(s, "代价：KV Cache 占用显存");
  rect(s, M, 1.45, CW, 1.25, INK);
  txt(s, [
    { text: "每个 token 的 KV Cache  =  2 × n_layer × n_embd × 每个数的字节数", options: { fontSize: 22, bold: true, color: WHITE, breakLine: true } },
    { text: "2 = K 和 V；n_embd = n_head × hs；每一层都要存", options: { fontSize: 14, color: C.accent5 } },
  ], { x: M + 0.3, y: 1.45, w: CW - 0.6, h: 1.25, valign: "middle" });
  const head = ["型号", "层数", "宽度", "每个 token", "1024 个 token"].map((t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: INK2 } } }));
  const rows = [["gpt2", "12", "768", "72 KB", "72 MB"], ["gpt2-medium", "24", "1024", "192 KB", "192 MB"], ["gpt2-large", "36", "1280", "360 KB", "360 MB"], ["gpt2-xl", "48", "1600", "600 KB", "600 MB"]]
    .map((r, i) => r.map((t, j) => ({ text: t, options: { color: INK, bold: j >= 3, fill: { color: i % 2 ? LIGHT : WHITE } } })));
  s.addTable([head, ...rows], { x: M, y: 3.0, w: 7.6, colW: [1.9, 1.1, 1.2, 1.6, 1.8], rowH: 0.52, fontSize: 15, align: "center", valign: "middle", border: { type: "solid", pt: 1, color: HEX.mid } });
  txt(s, "batch = 1、fp32；fp16 / bf16 减半；占用与 batch、序列长度成正比", { x: M, y: 5.75, w: 7.6, h: 0.4, fontSize: 13, color: SLATE });
  stat(s, 8.75, 2.95, 3.98, "600 MB", "gpt2-xl 一条 1024 token 序列的 KV Cache（fp32）", CORAL, { bigPt: 48, labelH: 0.7 });
  txt(s, "更大的模型、更长的上下文里，KV Cache 常常是显存的大头 —— GQA、MQA 等技术就是为此而生", { x: 8.75, y: 4.85, w: 3.98, h: 1.3, fontSize: 14, color: INK2 });
}

// 26. Tests ------------------------------------------------------------------------------------
{
  const s = slideOf("CONTENT", "s26_tests");
  title(s, "正确性：缓存不能改变结果");
  const tests = [
    ["test_cached_forward_matches_full_forward", "按 8、1、1、6、4 个 token 分块喂入缓存，每块 logits 都与一次性前向一致"],
    ["test_greedy_generate_matches_uncached", "加不加缓存，贪心生成逐 token 相同；包括超过 block_size 后的回退路径"],
    ["test_sampled_generate_matches_uncached", "同一随机种子下，top-k 采样结果完全相同"],
    ["test_cached_forward_matches_hf", "与 Hugging Face 官方实现 + DynamicCache 的 logits 对齐"],
  ];
  const w = (CW - 0.3) / 2;
  tests.forEach(([n, d], i) => {
    const x = M + (i % 2) * (w + 0.3), y = 1.5 + Math.floor(i / 2) * 2.15;
    rect(s, x, y, w, 1.9, LIGHT);
    circleNum(s, x + 0.25, y + 0.25, 0.55, "✓", TEAL, WHITE, 18);
    txt(s, n, { x: x + 1.0, y: y + 0.25, w: w - 1.2, h: 0.55, fontSize: 13, bold: true, fontFace: CODE_FONT, valign: "middle" });
    txt(s, d, { x: x + 1.0, y: y + 0.9, w: w - 1.2, h: 0.85, fontSize: 15, color: INK2 });
  });
  rect(s, M, 5.95, CW, 0.75, INK);
  txt(s, [
    { text: "$ uv run --extra cuda pytest tests/test_gpt2.py", options: { color: "E6E9F2" } },
    { text: "      18 passed", options: { color: "C3E88D", bold: true } },
  ], { x: M + 0.3, y: 5.95, w: CW - 0.6, h: 0.75, fontFace: CODE_FONT, fontSize: 14, valign: "middle" });
}

// 27. Summary ----------------------------------------------------------------------------------
section("总结");
{
  const s = slideOf("DARK_CONTENT", "s27_summary");
  title(s, "总结");
  const items = [
    "GPT-2 推理是自回归循环：每步一次前向，取最后位置的 logits 采样",
    "因果掩码 + 逐位置计算 + 绝对位置编码 ⇒ 旧 token 的 K、V 永远不变",
    "KV Cache 只为新 token 计算：每步 O(t²) → O(t)，总量 N³ 级 → N² 级",
    "代价是显存；解码受带宽限制，实测 gpt2 快 8.2×，gpt2-xl 快 12.4×",
  ];
  items.forEach((t, i) => {
    const y = 1.65 + i * 1.22;
    circleNum(s, M, y, 0.7, String(i + 1), AMBER, INK, 22);
    txt(s, t, { x: M + 1.0, y, w: CW - 1.0, h: 0.7, fontSize: 20, color: WHITE, valign: "middle" });
  });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
