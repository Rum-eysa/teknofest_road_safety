FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Sistem meta bilgileri
LABEL maintainer="Teknofest_Team"
LABEL description="Teknofest Akilli Yol Guvenlig Yarismas - Road Safety AI"

# Sistem paketlerini güncelle ve gerekli baðýmlýlýklarý kur
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Çalýþma dizinini ayarla
WORKDIR /app

# Gerekli klasör yapýlarýný oluþtur
RUN mkdir -p /app/data/input \
    && mkdir -p /app/data/output \
    && mkdir -p /app/models \
    && mkdir -p /app/src \
    && mkdir -p /app/logs

# requirements.txt kopyala ve baðýmlýlýklarý kur (build aþamasýnda)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Kaynak kodlarýný kopyala
COPY src/ /app/src/
COPY main.py .

# Konteyner açýlýrken çalýþacak komut
CMD ["python3", "main.py"]
