# ------------------------------------------------------------
# Whisper Worker – CUDA 11.8 (estable en Vast)
# - faster-whisper con calidad alta (float16)
# - healthcheck GPU dentro del contenedor
# - warmup para descargar large-v3 antes de procesar
# ------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Defaults (se pueden sobreescribir al ejecutar docker)
ENV WHISPER_MODEL=large-v3
ENV WHISPER_LANG=es
ENV WHISPER_BEAM=5
ENV WHISPER_COMPUTE=float16
ENV MODELS_DIR=/models

# Base system
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg \
    git \
    openssh-client \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Python deps
# Nota: faster-whisper usa CTranslate2 por debajo. Lo importante es que el runtime CUDA exista (lo aporta la base nvidia/cuda)
RUN pip install --no-cache-dir \
    faster-whisper \
    soundfile \
    numpy \
    tqdm

# App dirs
WORKDIR /app
RUN mkdir -p /app/downloads /app/output /app/locks /models

# Scripts (los creas en el repo y se copian aquí)
COPY healthcheck_gpu.py /app/healthcheck_gpu.py
COPY warmup_model.py /app/warmup_model.py

# Por defecto no hacemos nada. El capataz lanzará comandos concretos.
CMD ["bash"]
