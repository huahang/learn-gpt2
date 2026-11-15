"""Build a fully offline, typeset HTML edition of review.md.

Requirements: Python ``markdown`` and Node.js. Install the pinned math renderer once::

    npm install --prefix build/review_deps --no-save --no-package-lock katex@0.16.22
    python build_review.py
    python build_review.py --pdf  # Optional: installed Chrome/Edge and PyMuPDF

The output embeds all KaTeX CSS and font assets. No network requests or browser-side
JavaScript are needed to read it. Math rendering errors fail the build.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
MATH_PATTERN = re.compile(r"\$\$(.*?)\$\$|(?<!\\)\$([^\n$]+?)(?<!\\)\$", re.DOTALL)

CSS = """
:root { color-scheme:light; --ink:#15273d; --muted:#53677d; --accent:#136d80;
  --line:#d7e1e8; --paper:#fff; --soft:#f0f6f8; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:24px; }
body { margin:0; background:#edf2f6; color:var(--ink);
  font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif;
  font-size:17px; line-height:1.94; -webkit-font-smoothing:antialiased; }
a { color:#006b84; text-decoration-thickness:1px; text-underline-offset:3px; }
a:hover { color:#a34c10; }
.layout { display:grid; grid-template-columns:280px minmax(0,1040px);
  justify-content:center; gap:28px; padding:32px 24px 64px; }
aside { position:sticky; top:24px; align-self:start; max-height:calc(100vh - 48px);
  overflow:auto; padding:22px 20px; background:#f8fbfd; border:1px solid var(--line);
  border-radius:12px; font-size:13px; line-height:1.65; }
aside .toc-label { font-size:17px; font-weight:700; margin-bottom:14px; }
.toc ul { list-style:none; padding:0; margin:0; }
.toc>ul>li>ul { padding:0; }
.toc ul ul ul { padding-left:13px; margin:5px 0 12px; border-left:2px solid #dae6ee; }
.toc li { margin:5px 0; }
.toc a { text-decoration:none; color:#3c566d; display:block; }
.toc>ul>li>a { display:none; }
.toc>ul>li>ul>li>a { color:#163e53; font-weight:700; }
article { min-width:0; padding:42px 58px 52px; background:var(--paper);
  border:1px solid var(--line); border-radius:14px; box-shadow:0 8px 30px #12263c08; }
.kicker { color:var(--accent); font-size:13px; font-weight:700; letter-spacing:1.5px; }
h1 { font-size:35px; line-height:1.4; letter-spacing:-.6px; margin:12px 0 18px;
  color:#102d45; }
h2 { font-size:25px; line-height:1.45; margin:54px 0 20px; padding-top:14px;
  border-top:2px solid #dbe6ed; color:#12445a; }
h3 { font-size:20px; line-height:1.55; margin:32px 0 13px; color:#17495a; }
p { margin:14px 0; text-align:justify; overflow-wrap:break-word; }
strong { font-weight:700; color:#123d51; }
ul,ol { padding-left:1.7em; }
li { margin:8px 0; }
.meta { padding:10px 15px; color:#486274; background:#eef6f8;
  border-left:4px solid #178399; border-radius:0 6px 6px 0; font-size:14px; }
.table-scroll { overflow-x:auto; margin:22px 0; }
table { border-collapse:collapse; width:100%; font-size:14px; line-height:1.7;
  margin:0; text-align:left; }
th { background:#e8f1f5; color:#1c4358; font-weight:700; }
th,td { border:1px solid #d2dee6; padding:10px 12px; vertical-align:top; }
tr:nth-child(even) td { background:#f7fafc; }
code { font-family:'Cascadia Mono',Consolas,'Noto Sans Mono',monospace; font-size:.86em;
  background:#eef3f6; padding:2px 4px; border-radius:3px; }
pre { white-space:pre; overflow-x:auto; border-radius:9px; background:#142d43;
  color:#eef5fb; padding:20px 22px; font-size:14px; line-height:1.65; }
pre code { background:transparent; color:inherit; padding:0; font-size:inherit; }
.math-block { margin:22px 0; padding:9px 12px; overflow-x:auto;
  border-left:3px solid #d4e6ec; background:#f8fbfd; border-radius:0 7px 7px 0; }
.katex-display { margin:.7em 0 !important; }
.katex { font-size:1.07em !important; }
.math-block .katex { font-size:1.12em !important; }
footer { margin-top:40px; border-top:1px solid var(--line); padding-top:16px;
  font-size:13px; color:var(--muted); }
@media(max-width:1080px) { .layout { grid-template-columns:minmax(0,900px); }
  aside { position:static; max-height:320px; }
  .toc>ul>li>ul { column-count:2; column-gap:28px; }
  .toc>ul>li>ul>li { break-inside:avoid; }
  article { padding:32px 36px; } }
@media(max-width:620px) { body { font-size:16px; } .layout { padding:12px; gap:14px; }
  article { padding:24px 20px; } h1 { font-size:27px; } h2 { font-size:23px; }
  h3 { font-size:19px; } .toc>ul>li>ul { column-count:1; }
  th,td { min-width:100px; padding:8px; } }
@media print { @page { size:A4; margin:17mm 17mm 18mm; }
  html { scroll-behavior:auto; } body { background:white; font-size:10.5pt; line-height:1.7; }
  .layout { display:block; padding:0; } aside { display:none; }
  article { padding:0; border:0; border-radius:0; box-shadow:none; }
  h1 { font-size:23pt; } h2 { font-size:16pt; margin-top:26pt; padding-top:9pt; }
  h3 { font-size:12.5pt; margin-top:17pt; }
  h1,h2,h3 { break-after:avoid; } p { orphans:3; widows:3; }
  .meta { font-size:9pt; } .kicker { font-size:9pt; }
  table { font-size:8.7pt; } th,td { padding:5pt 6pt; }
  thead { display:table-header-group; } tr { break-inside:avoid; }
  pre,.math-block { break-inside:avoid; } pre { font-size:8.2pt; }
  .table-scroll { overflow:visible; } .math-block { overflow:visible; }
  a { color:inherit; } footer { font-size:8pt; }
  * { -webkit-print-color-adjust:exact; print-color-adjust:exact; }
}
"""

NODE_RENDERER = """
const fs = require('fs');
const katex = require(process.argv[1]);
const equations = JSON.parse(fs.readFileSync(0, 'utf8'));
const result = equations.map((item, index) => {
  try {
    return katex.renderToString(item.tex, {
      displayMode: item.display, output:'htmlAndMathml',
      throwOnError:true, strict:'warn', trust:false
    });
  } catch (err) { throw new Error(`Equation ${index+1}: ${item.tex}\\n${err.message}`); }
});
process.stdout.write(JSON.stringify(result));
"""


def embed_fonts(katex_dir: Path) -> str:
    """Inline all font variants so the HTML remains a single portable file."""
    css_path = katex_dir / "dist" / "katex.min.css"
    css = css_path.read_text(encoding="utf-8")

    def embed(match: re.Match[str]) -> str:
        relative = match.group(1).strip("\"'")
        path = css_path.parent / relative
        mime = {".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf"}[path.suffix]
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'url("data:{mime};base64,{encoded}")'

    return re.sub(r"url\(([^)]+)\)", embed, css)


def typeset_math(source: str, katex_dir: Path, node: str) -> tuple[str, int, int]:
    """Replace math with static HTML/MathML before Markdown processing."""
    equations: list[dict[str, str | bool]] = []

    def collect(match: re.Match[str]) -> str:
        display = match.group(1) is not None
        tex = (match.group(1) if display else match.group(2)).strip()
        index = len(equations)
        equations.append({"tex": tex, "display": display})
        placeholder = f"MATHPLACEHOLDER{index:04d}END"
        return f"\n\n{placeholder}\n\n" if display else placeholder

    prepared = MATH_PATTERN.sub(collect, source)
    process = subprocess.run(
        [node, "-e", NODE_RENDERER, str(katex_dir)],
        input=json.dumps(equations, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=True,
    )
    rendered: list[str] = json.loads(process.stdout)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"])
    body = md.convert(prepared)
    for index, (item, equation) in enumerate(zip(equations, rendered, strict=True)):
        placeholder = f"MATHPLACEHOLDER{index:04d}END"
        if item["display"]:
            body = body.replace(
                f"<p>{placeholder}</p>", f'<div class="math-block">{equation}</div>'
            )
        else:
            body = body.replace(placeholder, equation)
    body = body.replace("<table>", '<div class="table-scroll"><table>')
    body = body.replace("</table>", "</table></div>")
    display_count = sum(bool(item["display"]) for item in equations)
    return md.toc + "\x00" + body, len(equations), display_count


def export_pdf(html_path: Path, browser: str | None) -> Path:
    """Print with a local Chromium browser, including WSL-to-Windows support."""
    if browser is None:
        candidates = [
            shutil.which("chromium"),
            shutil.which("google-chrome"),
            "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
            "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
        ]
        browser = next((item for item in candidates if item and Path(item).exists()), None)
    if browser is None:
        raise FileNotFoundError("No Chrome/Chromium/Edge found; pass --browser PATH.")
    html_path = html_path.resolve()
    pdf_path = html_path.with_suffix(".pdf")
    profile = HERE / "build/review_browser"
    wsl_windows = browser.lower().endswith(".exe") and sys.platform != "win32"

    def browser_path(path: Path) -> str:
        if wsl_windows:
            return subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip()
        return str(path)

    url = (
        "file:" + browser_path(html_path).replace(chr(92), "/")
        if wsl_windows
        else html_path.as_uri()
    )
    subprocess.run(
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--no-pdf-header-footer",
            "--no-first-run",
            "--allow-file-access-from-files",
            "--user-data-dir=" + browser_path(profile),
            "--print-to-pdf=" + browser_path(pdf_path),
            url,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    import pymupdf

    doc = pymupdf.open(pdf_path)
    for index, page in enumerate(doc):
        page.insert_textbox(
            pymupdf.Rect(0, page.rect.height - 31, page.rect.width, page.rect.height - 15),
            f"{index + 1} / {len(doc)}",
            fontsize=8,
            fontname="helv",
            align=1,
            color=(0.35, 0.42, 0.49),
        )
    numbered = pdf_path.with_suffix(".numbered.pdf")
    doc.save(numbered)
    page_count = len(doc)
    doc.close()
    numbered.replace(pdf_path)
    print(f"Printed {pdf_path}: {page_count} pages with page numbers.")
    return pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "review.md")
    parser.add_argument("--output", type=Path, default=HERE / "review.html")
    parser.add_argument(
        "--katex-dir", type=Path, default=HERE / "build/review_deps/node_modules/katex"
    )
    parser.add_argument("--node", default="node")
    parser.add_argument("--pdf", action="store_true", help="Also print review.pdf")
    parser.add_argument("--browser", help="Chrome/Chromium/Edge executable for --pdf")
    args = parser.parse_args()
    if not args.katex_dir.is_dir():
        parser.error("KaTeX not found. Run the npm install command in the module docstring.")
    source = args.input.read_text(encoding="utf-8")
    result, equation_count, display_count = typeset_math(source, args.katex_dir, args.node)
    toc, body = result.split("\x00", 1)
    title = re.search(r"^# (.+)$", source, re.MULTILINE).group(1)
    body = body.replace(
        "</h1>",
        '</h1><div class="meta">三篇论文综述 · 因果自注意力与 KV Cache · '
        "包含手算例子、实验口径与自测练习</div>",
        1,
    )
    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>"
        + embed_fonts(args.katex_dir)
        + CSS
        + '</style></head><body><div class="layout"><aside aria-label="目录">'
        '<div class="toc-label">目录</div>'
        + toc
        + '</aside><article><div class="kicker">TRANSFORMER · GPT-2 · GPT-3</div>'
        + body
        + "<footer>离线阅读版：公式、样式与数学字体已嵌入本文件。"
        "正文以 2017、2019、2020 年三篇论文为研究范围；来源与版本见参考文献。</footer>"
        "</article></div></body></html>"
    )
    if "MATHPLACEHOLDER" in document or 'class="katex-error"' in document:
        raise ValueError("Unresolved math placeholder or math rendering error")
    if re.search(r"(?:src|url)\s*[=(]\s*[\"']?https?://", document):
        raise ValueError("An external rendering dependency remained in the HTML")
    args.output.write_text(document, encoding="utf-8")
    print(
        f"Built {args.output}: {equation_count} equations "
        f"({display_count} display), {len(document):,} characters; fully offline."
    )
    if args.pdf:
        export_pdf(args.output, args.browser)


if __name__ == "__main__":
    main()
