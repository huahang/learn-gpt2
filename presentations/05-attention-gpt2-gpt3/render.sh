#!/usr/bin/env bash
# Render attention_gpt2_gpt3.pptx to build/render/slide-NN.png (1920x1080) with LibreOffice.
#   SOFFICE=<path to soffice> ./render.sh    (default: soffice on PATH)
set -euo pipefail
cd "$(dirname "$0")"
SOFFICE="${SOFFICE:-soffice}"
mkdir -p build/render
# tools/fonts.conf maps the deck's Windows fonts to Noto Sans CJK SC / DejaVu Sans Mono.
FONTCONFIG_FILE="$PWD/tools/fonts.conf" "$SOFFICE" -env:UserInstallation=file:///tmp/lo_profile_05 \
    --headless --convert-to pdf --outdir build/render attention_gpt2_gpt3.pptx >/dev/null
rm -f build/render/slide-*.png
# Exactly 1920 x 1080; -r 144 rounds up to 1921 px and leaves a white last column.
pdftoppm -png -scale-to-x 1920 -scale-to-y 1080 build/render/attention_gpt2_gpt3.pdf build/render/slide
ls build/render/slide-*.png | wc -l
