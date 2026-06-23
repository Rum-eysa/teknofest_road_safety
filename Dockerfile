FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Sistem meta bilgileri
LABEL maintainer="Teknofest_Team"
LABEL description="Teknofest Akilli Yol Guvenlig Yarismas - Road Safety AI"

# Sistem paketlerini guncelle ve gerekli bagimliliklari kur
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

# Calisma dizinini ayarla
WORKDIR /app

# Gerekli klasor yapilarini olustur
RUN mkdir -p /app/data/input \
    && mkdir -p /app/data/output \
    && mkdir -p /app/models \
    && mkdir -p /app/src

# requirements.txt kopyala ve bagimliliklari kur (build asamasinda)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model agirliklarini models/ klasorunden kopyala
COPY models/ /app/models/

# Moduler kaynak kodlari kopyala
COPY src/ /app/src/
COPY main.py .
COPY README.md .

# Konteyner acilirken calisacak komut
CMD ["python3", "main.py"]
