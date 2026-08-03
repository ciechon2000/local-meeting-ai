FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYANNOTE_METRICS_ENABLED=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    TOKENIZERS_PARALLELISM=false \
    MODEL_CACHE=/data/models \
    TMP_DIR=/data/tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      ffmpeg \
      git \
      curl \
      ca-certificates \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Wersje zgodne z wymaganiami WhisperX 3.8.6 i CUDA 12.8.
RUN python3 -m pip install \
      torch==2.8.0 \
      torchvision==0.23.0 \
      torchaudio==2.8.0 \
      --index-url https://download.pytorch.org/whl/cu128

WORKDIR /app
COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

COPY app ./app

RUN mkdir -p /data/models /data/tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "120"]
