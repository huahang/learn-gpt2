#!/usr/bin/env bash
# Render gpt2_inference.pptx to build/render/slide-NN.png (1920x1080) with LibreOffice in Docker.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/render
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work learn-gpt2-media \
    soffice -env:UserInstallation=file:///tmp/lo_profile --headless --convert-to pdf --outdir build/render gpt2_inference.pptx >/dev/null
rm -f build/render/slide-*.png
# Exactly 1920 x 1080; -r 144 rounds up to 1921 px and leaves a white last column.
pdftoppm -png -scale-to-x 1920 -scale-to-y 1080 build/render/gpt2_inference.pdf build/render/slide
ls build/render/slide-*.png | wc -l
