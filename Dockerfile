# =============================================================================
# Dockerfile — Cikarim (Inference) Pipeline
# -----------------------------------------------------------------------------
# NEDEN TAM OLARAK "nvidia/cuda:12.1.0-base-ubuntu22.04" (FTR'de ACIKCA istenen imaj)?
#   FTR sartnamesi, degerlendirme ortaminin BU imaj uzerinde kurulu olacagini
#   varsayar; baska bir taban imaj (orn. pytorch/pytorch) kullanmak, komite
#   degerlendirme altyapisindaki SURUM/UYUMLULUK beklentisini BOZABILIR.
#   Egitim Dockerfile'indan (pytorch/pytorch taban) FARKLI olarak, BURADA
#   torch + bagimliliklari BIZ ACIKCA kuruyoruz; cunku "base" imaj SADECE
#   CUDA runtime icerir, cuDNN/torch ICERMEZ.
# =============================================================================
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

LABEL maintainer="Teknofest_Team"
LABEL description="Teknofest Akilli Yol Guvenligi Yarismasi - Cikarim (Inference) Pipeline"

# Sistem paketleri: Python calisma zamani + OpenCV/ffmpeg video isleme bagimliliklari.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Calisma zamaninda kullanilacak klasor iskeleti.
RUN mkdir -p /app/data/input \
    && mkdir -p /app/data/output \
    && mkdir -p /app/models \
    && mkdir -p /app/src

# NEDEN torch, requirements.txt'TEN ONCE ve OZEL bir index-url ile kuruluyor?
#   "base" CUDA imajinda torch'un CUDA 12.1 ile UYUMLU (binary-compatible)
#   derlenmis surumunun kurulmasi gerekir; pip'in VARSAYILAN PyPI index'i
#   genellikle CPU-only torch dagitir. Resmi PyTorch CUDA 12.1 wheel
#   index'inden ACIKCA kurum yaparak bu riski ORTADAN KALDIRIYORUZ.
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Model agirliklari (best_model_a.pt, best_model_b.pt), egitim ciktilarindan
# bu klasore yerlestirilir (egitim Docker/Colab ortamlarinin CIKTISI).
COPY models/ /app/models/

# Moduler kaynak kodlar ve konfigurasyon.
COPY src/ /app/src/
COPY config.yaml .
COPY main.py .
COPY README.md .

# NEDEN ENTRYPOINT/CMD ayriminda CMD bos parametre listesi ile birakildi?
#   main.py, TUM yapilandirmayi config.yaml'dan okur (FTR Madde E geregi
#   ortam degiskeni/CLI bayragiyla DAVRANIS DEGISTIRME yoktur); bu yuzden
#   ekstra CLI parametresine GEREK YOKTUR — konteyner HER zaman AYNI
#   komutla, AYNI sekilde calisir.
CMD ["python3", "main.py"]
