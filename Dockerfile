FROM nvidia/cuda:12.1.0-base-ubuntu22.04

LABEL maintainer="Teknofest_Team"
LABEL description="FTR - 5G ve Yapay Zeka ile Akilli Yol Guvenligi"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/data/input /app/data/output /app/models /app/logs

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY models/ /app/models/
COPY src/ /app/src/
COPY main.py .
COPY README.md .

CMD ["python3", "main.py"]
