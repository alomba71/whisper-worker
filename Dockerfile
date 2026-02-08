# ------------------------------------------------------------
# Whisper Worker – CUDA 11.8 (compatible y estable)
# ------------------------------------------------------------
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Madrid
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Sistema base
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    git \
    openssh-client \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# ------------------------------------------------------------
# Dependencias Python
# Usamos faster-whisper (más estable que whisper+torch)
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    faster-whisper \
    soundfile \
    numpy \
    tqdm

# ------------------------------------------------------------
# Directorios de trabajo
# ------------------------------------------------------------
WORKDIR /app
RUN mkdir -p /app/downloads /app/output /app/locks

# ------------------------------------------------------------
# Comando por defecto (lo sobreescribirá el worker)
# ------------------------------------------------------------
CMD ["bash"]
