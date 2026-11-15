# LibreOffice (render .pptx to PDF) and ffmpeg (assemble the video), with CJK fonts.
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-impress libreoffice-core ffmpeg fontconfig \
        fonts-noto-cjk fonts-liberation fonts-liberation2 fonts-dejavu-core \
        fonts-crosextra-carlito fonts-crosextra-caladea \
    && rm -rf /var/lib/apt/lists/*
COPY fonts.conf /etc/fonts/local.conf
RUN fc-cache -f
