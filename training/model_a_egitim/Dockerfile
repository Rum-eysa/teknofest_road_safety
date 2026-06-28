# =============================================================================
# Dockerfile — Model A (Arac Bilgisi) Egitim Ortami
# -----------------------------------------------------------------------------
# NEDEN pytorch/pytorch CUDA imaji taban alindi (cuda:12.1.0-base degil)?
#   Egitim (training) asamasi, cikarimdan (inference) farkli olarak; CUDA
#   Toolkit + cuDNN + torch derleme baglantilarinin TAM ve BIRBIRIYLE UYUMLU
#   olmasini gerektirir. "base" CUDA imajinda bu kütüphaneler eksiktir ve
#   pip ile torch kurulumu surucu/CUDA versiyon uyumsuzluguna acik kapi
#   birakir. Resmi pytorch/pytorch imaji, torch+CUDA+cuDNN ucluyu ONCEDEN
#   TEST EDILMIS sekilde saglayarak ekibin 5 saatlik kisitli surede ortam
#   kurulumuyla degil, dogrudan egitimle ugrasmasini saglar.
# =============================================================================
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

LABEL maintainer="Teknofest_Team"
LABEL description="Model A (Arac Bilgisi) - YOLO Egitim Ortami - Teknofest 2026"

# Sistem bagimliliklari: OpenCV'nin calismasi icin gerekli paylasilan kutuphaneler
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Once SADECE requirements.txt kopyalanir.
# NEDEN? Docker katman onbellegi (layer cache) sayesinde, kod (train.py)
# degistiginde bagimliliklar YENIDEN KURULMAZ; sadece kod degisikligi
# iceren katman yeniden olusturulur. Bu, 5 kisilik ekibin sik sik
# kod guncelleyip yeniden build alacagi bir senaryoda buyuk zaman kazandirir.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Egitim kodu, konfigurasyon ve veri seti sablonu konteynera kopyalanir.
COPY config.yaml .
COPY data/ ./data/
COPY train.py .

# NOT: Gercek veri seti (Roboflow export'u) imaj icine GOMULMEZ; boyutu
# nedeniyle docker-compose.yml uzerinden VOLUME olarak mount edilir.
# Bu, hem imaj boyutunu kucuk tutar hem de veri seti guncellendiginde
# imajin yeniden build edilmesini GEREKSIZ kilar.

ENTRYPOINT ["python", "train.py"]
CMD ["--config", "config.yaml"]
