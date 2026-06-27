# TEKNOFEST Yol Güvenliği AI - Mimari Tasarım Raporu
## Senior Level Architecture & Design Decisions

---

## 1. GENEL MİMARİ YAKLAŞIM

### 1.1 İki Ayrı Model Stratejisi (Neden Tek Model Değil?)

**Karar:** Model A (Araç Bilgisi) ve Model B (Yol Güvenliği) olarak iki bağımsız YOLO modeli.

**Neden Bu Yaklaşım?**
```
┌─────────────────────────────────────────────────────────────────┐
│                    Gradient Interference Problemi                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Model A (Araç Bilgisi)            Model B (Yol Güvenliği)       │
│ ├─ Uzak nesneler (2-50m)         ├─ Yakın plan (0.5-3m)        │
│ ├─ Geniş açı perspektif          ├─ Sürücü odaklı/kabin içi     │
│ ├─ Genel kotur/boyut             ├─ İnce hareketler/yüz ifadeleri
│ ├─ Sabit ışık koşulları          └─ Dinamik/değişken ışık       │
│                                                                   │
│ Problemi: Tek model feature extractor'da çelişkili gradientler  │
│ → Arkadaki katmanlar iki farklı görsel semantiği öğrenmek       │
│   için savaşır (conflicting objectives)                          │
│ → Her modelin doğruluğu %3-7 düşer                              │
│                                                                   │
│ Çözüm: Ayrı modeller → Bağımsız feature extraction              │
│ → Her model kendi domain'ine optimize olabilir                  │
│ → Ekip paralel eğitim yapabilir (5 kişi 2 takıma bölünür)      │
│ → Donanım bölüşümü (Colab + lokal GPU)                          │
└─────────────────────────────────────────────────────────────────┘
```

**Teknik Gerekçe:**
- **Backbone Uyumsuzluğu:** ResNet/EfficientNet feature pyramidları farklı semantic seviyelere optimized
- **Data Augmentation Çatışması:** Araç için rotation/crop yararlı, sürücü için distorting
- **Batch Normalization:** Her domain için farklı feature distribution → BN istatistikleri çelişir
- **Loss Function Balancing:** 7 araç sınıfı vs 18 eylem sınıfı farklı weight allocation gerekli

---

## 2. YOLO MİMARİ SEÇİMİ

### 2.1 Neden YOLOv8 (Modern Version)?
```
Karşılaştırma Tablosu:
┌──────────┬────────────┬──────────┬────────────┬─────────────┐
│ Metrik   │ YOLOv8m    │ YOLOv5l  │ YOLOv10n   │ Faster RCNN │
├──────────┼────────────┼──────────┼────────────┼─────────────┤
│ mAP@50   │ 0.52       │ 0.48     │ 0.51       │ 0.58        │
│ Çıkarım  │ 5.2ms      │ 7.1ms    │ 3.8ms      │ 45ms        │
│ FLOPs    │ 16B        │ 20B      │ 12B        │ 130B        │
│ Boyut    │ 25MB       │ 45MB     │ 8MB        │ 200MB       │
└──────────┴────────────┴──────────┴────────────┴─────────────┘

Seçim: YOLOv8m (medium)
• mAP ve speed arasında optimal dengeleme
• Docker imaj boyutu: 8GB limit → YOLOv8m uygun
• 10 dakika timeout içinde 5+ video işlenebilir
```

### 2.2 Model Seçim Stratejisi
- **Model A:** YOLOv8m (Araç tespiti + plaka localization)
- **Model B:** YOLOv8n (Sürücü/kabin tespiti - daha hafif, hızlı)
  - Model B 18 sınıf olduğu için daha karmaşık, ama smaller model yeterli
  - Nested bounding boxes nadiren çakışır (sürücü vs arka koltuk zaten spatially separated)

---

## 3. VERİ İŞLEME VE AUGMENTASYON STRATEJİSİ

### 3.1 Mosaic & MixUp Stratejisi

```python
# Mosaic Augmentation (YOLOv8 default)
# Neden?
# ├─ Küçük nesneleri büyütür (plaka önemli!)
# ├─ Context kötüye kullanım engeller (model kenarlardan öğrenmesini zorlar)
# ├─ Batch normalization gürültülü mini-batch istatistikleri oluşturur
# └─ → Model multi-scale detection robustness gelişir

# MixUp (YOLOv8 optional)
# Neden?
# ├─ İki image'ı λ*img1 + (1-λ)*img2 şeklinde blend
# ├─ Smooth decision boundary oluşturur
# ├─ Adversarial robustness artar
# └─ → Over-confidence azalır, better calibrated confidence scores

# Ama yol güvenliği modeli için MixUp dikkatli kullan:
# - Sürücü yüzü blend edilirse sınıf bilgisi kaybolabilir
# → conf: "mixup_prob": 0.3  # 0.5 yerine daha düşük
```

### 3.2 Domain-Specific Augmentation

**Model A (Araç):**
```yaml
augmentations:
  - type: "hsv_augment"
    h_gain: 0.015  # Işık değişimlerine robust
    s_gain: 0.7    # Renk tahmini için
    v_gain: 0.4
  - type: "perspective"
    degrees: 10    # Araç açı değişiklikleri
  - type: "scale"
    factor: [0.5, 1.5]  # Uzak/yakın nesneler
```

**Model B (Sürücü/Kabin):**
```yaml
augmentations:
  - type: "rotate"
    max_degrees: 15  # Sürücü başını çevirme
  - type: "shear"
    max_shear: 8  # Kamera açısı değişimi
  - type: "mosaic"  # Daha agresif (yüzü temizle)
    mosaic_prob: 0.9
```

---

## 4. TRAINING STRATEJISI: 5 SAATLIK BÜTÇE

### 4.1 Zaman Bütçesi Planlaması

```
Toplam: 300 dakika = 18,000 saniye

Model A Eğitim: 120 dakika (2 saat)
├─ Epoch sayısı: 100 (config.yaml)
├─ Early stopping patience: 15 (15 epoch improvement yok → dur)
├─ Batch size: 32 (GPU RAM optimize)
└─ Expected epochs before stop: ~85

Model B Eğitim: 90 dakika (1.5 saat)
├─ Epoch sayısı: 150
├─ Early stopping patience: 15
├─ Batch size: 24 (daha fazla sınıf, daha az RAM)
└─ Expected epochs before stop: ~120

Buffer: 90 dakika
├─ Data loading time: ~10 min
├─ Model validation overhead: ~15 min
├─ Final checkpoint saving: ~5 min
├─ Inference script testing: ~60 min
└─ Safety margin: ~0 min (çok sıkı!)

KRITIK: Zamanı aşan epoch'u kesinlikle kaydet!
```

### 4.2 Early Stopping Implementation

```python
# Neden Early Stopping?
# ├─ Test loss 15 epoch boyunca iyileşmezse training durdu
# ├─ Overfitting engellenir (zaman ve RAM tasarrufu)
# ├─ Best model otomatik load edilir
# └─ → Kalan zaman diğer optimizasyonlara harcanabilir

# Patience=15 seçiminin nedeni:
# ├─ Çok düşük (5): Stochasticity nedeniyle erken durabilir
# ├─ Çok yüksek (30): Zaman kaybı ve overfitting riski
# └─ 15: İstatistiksel olarak güvenli ve zaman-efficient
```

### 4.3 AMP (Automatic Mixed Precision) Optimizasyonu

```python
# Neden AMP kullanılır?
# ├─ float16 hesaplamalar 2x hızlı (Tesla T4'de)
# ├─ float32 backprop duyarlılığı korunur
# ├─ Memory usage %40 azalır → daha büyük batch size
# ├─ Model convergence hemen hemen aynı
# └─ → Toplam training süresi %35-45 kısalır

# YOLO (PyTorch) otomatik AMP destekler:
# trainer.train(device='0', epochs=100, amp=True)
```

---

## 5. RENK TAHMİNİ STRATEJİSİ

### 5.1 HSV Color Space vs RGB

```python
# Neden HSV kullanılır?
# ├─ H (Hue): İştah bakmayan renk → 0-180° (OpenCV)
# ├─ S (Saturation): Renk doygunluğu → sağlamlık
# ├─ V (Value): Parlaklık → ışık koşullarından bağımsız
#
# Problem: Doğrudan HSV histogram matching yetersiz
# → Çıkmazlı renkler: kirmizi, turuncu, kahverengi ayrımı zor
#
# Çözüm: Hybrid approach
# ├─ 1. Adım: Model B plakasını crop et
# ├─ 2. Adım: Dominant HSV color range tespit et
# ├─ 3. Adım: 9 reference renk templates ile Bhattacharyya mesafesi
# ├─ 4. Adım: Küçük CNN head ile fine-tuning (optional)
# └─ → Confidence: histogram match score
```

### 5.2 9 Standart Renk Paleti

```python
COLOR_TEMPLATES = {
    "beyaz": {"h_range": (0, 30), "s_range": (0, 50), "v_range": (200, 255)},
    "siyah": {"h_range": (0, 255), "s_range": (0, 255), "v_range": (0, 50)},
    "gri": {"h_range": (0, 255), "s_range": (0, 50), "v_range": (80, 180)},
    "kirmizi": {"h_range": (0, 10), "s_range": (100, 255), "v_range": (100, 255)},
    # ... (diğerleri)
}

# Bhattacharyya Distance (Histogram Karşılaştırma)
# → Tüm 9 renk için histogram distance hesapla
# → Minimum distance seçilen renk olur
# → Confidence = (max_dist - min_dist) / max_dist
```

---

## 6. PLAKA OCR VE REGEX VALİDASYON

### 6.1 OCR Pipeline

```
┌──────────────────────────────────┐
│  1. Bounding Box Crop (Model A)  │
├──────────────────────────────────┤
│ Plaka ROI: x1,y1,x2,y2 → crop   │
│                                  │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  2. Ön İşlem (Preprocessing)     │
├──────────────────────────────────┤
│ • Histogram Equalization         │
│ • Contrast Stretching (CLAHE)    │
│ • Denoise (bilateral filter)     │
│ → OCR accuracy +15-20%           │
│                                  │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  3. OCR Engine: EasyOCR          │
├──────────────────────────────────┤
│ Seçim: EasyOCR                   │
│ • PaddleOCR'dan %2-3 daha iyi    │
│ • Türkçe char support           │
│ • GPU accelerated                │
│ • Confidence score per char      │
│                                  │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  4. Regex Validation & Cleanup   │
├──────────────────────────────────┤
│ Pattern:                         │
│ (0[1-9]|[1-7][0-9]|8[01])       │
│   ( (\s?[a-zA-Z]\s?)(\d{4,5})   │
│   | (\s?[a-zA-Z]{2}\s?)(\d{3,4})│
│   | (\s?[a-zA-Z]{3}\s?)(\d{2,3}))│
│                                  │
│ • Boşlukları temizle            │
│ • Harfleri UPPERCASE tut        │
│ • Türkçe char varsa → drop      │
│                                  │
└──────────────────────────────────┘
            ↓
┌──────────────────────────────────┐
│  5. Confidence Calculation       │
├──────────────────────────────────┤
│ score = min(ocr_char_confs)      │
│ → En zayıf character belirler    │
│                                  │
└──────────────────────────────────┘
```

---

## 7. SLALOM TESPİTİ: TRAJECTORY ANALYSIS

### 7.1 Neden Heuristic Yaklaşım?

```
Problem: Slalom için eğitim veri seti yok
Çözüm: Geometry + Signal Processing
```

### 7.2 Algoritma

```python
"""
Slalom = Aracın sol-sağ periyodik salınımı

1. OBJECT TRACKING (Centroid)
   ├─ Frame-by-frame: bbox center (x_t, y_t) izle
   ├─ Kalman filter ile smooth trajectory
   └─ Outlier handling (sudden jumps)

2. FREQUENCY ANALYSIS
   ├─ X ekseni time-series: [x_0, x_1, x_2, ...]
   ├─ FFT → dominant frequency tespit et
   ├─ Threshold: 0.2 Hz < f < 2.0 Hz (araç hızına göre)
   │  • Normal lane change: ~0.5-1.5 Hz
   │  • Slalom: 1.0-2.5 Hz (hızlı sol-sağ)
   └─ Amplitude: std(x) > 30 pixels (en az)

3. DETECTION LOGIC
   if dominant_freq in [1.0, 2.5] and std(x) > threshold:
       label = "slalom"
       confidence = (peak_power / total_power)  # 0-1
   else:
       label = "normal"
       confidence = 1.0

4. TEMPORAL CONTINUITY
   ├─ Min 3 consecutive frames: düz bir şekilde işaretlenmeli
   └─ Boşluklar fill edilebilir (1 frame interpolation)
"""
```

### 7.3 Implementasyon Detayları

```python
# Kalman Filter kullanım sebebi:
# ├─ YOLO bbox'leri jittery (frame-to-frame noise)
# ├─ Gerçek trajectory smooth
# ├─ Kalman: E[x_t | observations] → smooth estimate
# └─ → Daha güvenilir frequency analysis

# FFT pencere fonksiyonu: Hanning
# ├─ Spectral leakage azaltır
# ├─ Side lobe level: -43 dB
# └─ → Neighboring frequencies temiz

# Confidence score mantığı:
# confidence = power_peak / sum(all_freqs)
# ├─ Periyodiklik ne kadar güçlü?
# └─ Spectral "spikiness" → slalom certainty
```

---

## 8. INFERENCE PIPELINE TASARIMI

### 8.1 Execution Flow

```
                    ┌─────────────────────┐
                    │  Docker Container   │
                    │  (nvidia/cuda:12.1) │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   main.py (Entry)   │
                    │ ├─ Input validation │
                    │ └─ Error handling   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  Video Reader (OpenCV)      │
                    │ ├─ FPS, frame_count tespit │
                    │ ├─ Frame read loop          │
                    │ └─ Timestamp calculation    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │  Inference Loop (Per Frame)         │
                    │                                      │
                    │  ┌─────────────────────────────┐   │
                    │  │ Model A: Vehicle Detection  │   │
                    │  │ ├─ Forward pass              │   │
                    │  │ ├─ NMS post-processing       │   │
                    │  │ └─ bbox, confidence → next   │   │
                    │  └────────────────┬──────────────┘   │
                    │                   │                  │
                    │  ┌────────────────▼──────────────┐   │
                    │  │ Detections: Vehicle Found?   │   │
                    │  │ YES                           │   │
                    │  │ ├─ Crop vehicle ROI           │   │
                    │  │ ├─ Color Detection (HSV)      │   │
                    │  │ ├─ Plate Localization        │   │
                    │  │ ├─ Crop plate → OCR + Regex  │   │
                    │  │ └─ Consolidate arac_bilgisi  │   │
                    │  └────────────────┬──────────────┘   │
                    │                   │                  │
                    │  ┌────────────────▼──────────────┐   │
                    │  │ Model B: Safety Detection      │   │
                    │  │ ├─ Forward pass (18 sınıf)    │   │
                    │  │ ├─ Each detection → list item │   │
                    │  │ └─ time, category, label, conf│   │
                    │  └────────────────┬──────────────┘   │
                    │                   │                  │
                    │  ┌────────────────▼──────────────┐   │
                    │  │ Trajectory Tracking (Slalom)  │   │
                    │  │ ├─ Model A bbox center store │   │
                    │  │ ├─ Kalman smooth trajectory   │   │
                    │  │ ├─ FFT frequency analysis     │   │
                    │  │ └─ Slalom detection check     │   │
                    │  └────────────────┬──────────────┘   │
                    │                   │                  │
                    │  ┌────────────────▼──────────────┐   │
                    │  │ Post-Processing               │   │
                    │  │ ├─ Merge consecutive detections│  │
                    │  │ ├─ Temporal smoothing         │   │
                    │  │ └─ Format standardization     │   │
                    │  └────────────────┬──────────────┘   │
                    │                   │                  │
                    └───────────────────┼──────────────────┘
                                        │
                    ┌───────────────────▼─────────────────┐
                    │  JSON Output Generation             │
                    │ ├─ Schema validation                │
                    │ ├─ UTF-8 encoding                   │
                    │ ├─ Turkish char cleanup             │
                    │ └─ Pretty printing (indent=2)       │
                    └───────────────────┬─────────────────┘
                                        │
                    ┌───────────────────▼─────────────────┐
                    │  /app/data/output/results.json       │
                    └─────────────────────────────────────┘
```

### 8.2 Asenkron vs Sıralı Çalışma

```python
# Yapı: SEQUENTIAL (asenkron değil)
# Neden?
# ├─ Model A → Model B bağımlılık yok
# ├─ Ama video frame stream lineer (sıralı)
# ├─ GPU memory: max 1 model at a time optimal
# ├─ Async queue management → complex error handling
# └─ → Sıralı: simple, deterministic, easy debug

# Optimizasyon: Overlap Computation & I/O
# for frame in video:
#   ├─ Frame read (CPU)
#   ├─ Preprocess (CPU parallel)
#   ├─ Model A forward (GPU)
#   ├─ Model B forward (GPU)  ← Pipelined execution
#   ├─ Post-processing (CPU)
#   └─ Next frame async read (I/O thread)
```

---

## 9. MODÜLER KODLAMASI VE CONFIGURATION

### 9.1 config.yaml Yapısı (Model A Eğitim)

```yaml
# Neden config.yaml?
# ├─ 5 kişi 5 farklı kombinasyon deneyebilsin
# ├─ Hyperparameter tracking (MLOps best practice)
# ├─ Reproducibility: git history + config → full experiment replay
# ├─ DRY principle: hardcoded values yok
# └─ → Collaboration efficient ve scientific

# config.yaml ile paramete override:
# python train.py --config config_experiment_1.yaml
# python train.py --config config_experiment_2.yaml
```

### 9.2 Modüler Dosya Yapısı

```
src/
├── predict.py         # Main inference orchestrator
│                      # Neden ayrı file?
│                      # ├─ Testable (mock models, fixtures)
│                      # ├─ Reusable (batch processing için)
│                      # └─ Clean separation of concerns
│
├── color_detector.py   # HSV + histogram matching
│                       # Neden ayrı?
│                       # └─ Color logic independent module
│
├── ocr_handler.py      # EasyOCR + regex validation
│                       # Neden ayrı?
│                       # ├─ OCR library abstraction
│                       # ├─ Easy swap (EasyOCR→PaddleOCR)
│                       # └─ Unit testing focused
│
├── trajectory_analyzer.py  # Kalman + FFT slalom detection
│                           # Neden ayrı?
│                           # ├─ Signal processing isolated
│                           # ├─ Numpy-heavy, PyTorch-free
│                           # └─ Can be optimized independently
│
└── utils.py            # Common utilities
                        # ├─ Preprocessing
                        # ├─ JSON formatting
                        # ├─ Error handling
                        # └─ Logging
```

---

## 10. ORTAM MANIPÜLASYONU YASAĞININ İMPLEMENTASYONU

### 10.1 YAP: Universal Code

```python
# ✅ DOĞRU: Tüm ortamlarda aynı davranış
def load_video(video_path):
    """Video dosyasını oku. Environment check yok."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    return cap
```

### 10.2 YAPMA: Environment-Specific Code

```python
# ❌ YANLIŞ: Environment manipulation (şartname ihlali)
def load_video(video_path):
    """NEVER DO THIS!"""
    
    # ❌ Problem 1: Environment variable check
    if os.getenv("EVAL_MODE") == "1":
        skip_validation = True
    
    # ❌ Problem 2: Hostname detection
    if "eval-server" in socket.gethostname():
        enable_debug = False
    
    # ❌ Problem 3: File existence trick
    try:
        open("/etc/hostname", "r")
        in_docker = True
    except:
        in_docker = False
    
    # ❌ Problem 4: Try-except based behavior
    try:
        import special_eval_lib
        use_special_mode = True
    except:
        use_special_mode = False
    
    # ❌ Problem 5: IP address check
    if ipaddress.ip_address(socket.gethostbyname(socket.gethostname())) in ipaddress.ip_network("10.0.0.0/8"):
        eval_mode_detected = True
```

**Şartname Kuralı:** Kod, **her ortamda tamamen aynı** şekilde davranmalıdır. Jüri, kodları statik analiz ve runtime monitoring ile kontrol edecek. Herhangi bir environment-dependent behavior bulunursa **disqualification**.

---

## 11. DOCKER OPTIMIZASYON STRATEJİSİ

### 11.1 Multi-Stage Build

```dockerfile
# Neden multi-stage?
# ├─ Final image'dan compile/build tools çıkar
# ├─ pip cache temizle
# ├─ 8GB limit kesindir
# └─ → %40-50 boyut azalması

# Stage 1: Builder
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as builder
RUN apt-get update && apt-get install -y python3-pip build-essential

# Stage 2: Runtime (minimal)
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
COPY --from=builder /usr/local/lib/python3.* /usr/local/lib/
```

### 11.2 Layer Caching Optimization

```dockerfile
# GOOD: Frequently changing layer last
COPY requirements.txt .           # Layer 1
RUN pip install -r requirements.txt   # Layer 2 (cached)

COPY src/ /app/src/              # Layer 3 (code changes often)
COPY main.py .                   # Layer 4

# BAD: Cache invalidation
COPY . .                         # Layer 1 (any file change → rebuild all)
RUN pip install -r requirements.txt  # Layer 2 (every rebuild)
```

### 11.3 Shared Memory (SHM) Management

```python
# Neden DataLoader num_workers için SHM önemli?
# ├─ Docker: default 2GB /dev/shm
# ├─ num_workers=4 + pin_memory=True
# ├─ Each worker: ~500MB shared memory gereksinim
# └─ 4 * 500 = 2000MB → SHM full!

# Çözüm:
# docker run ... --shm-size=4gb \
#   --ipc=host (recommended)

# Code tarafından:
# num_workers=2  # SHM limit'e uygun
# pin_memory=True (but not pin all tensors)
```

---

## 12. TESTING VE VALIDATION STRATEJİSİ

### 12.1 Unit Tests (Önemli Bileşenler)

```python
# test_ocr_handler.py
def test_plaka_regex_validation():
    """Regex'in all valid plate formats'ı accept etmesi"""
    valid_plates = [
        "34ABC123",      # Standard
        "01AB1234",      # Min province
        "81AB1234",      # Max province
        "01A1234",       # 1-letter format
        "01AB123",       # Min 3-digit format
    ]
    for plate in valid_plates:
        assert is_valid_plate(plate), f"Should accept {plate}"

def test_plaka_regex_rejection():
    """Invalid formats reject edilmesi"""
    invalid = ["82AB123", "34XYZ123", "34A1234"]
    for plate in invalid:
        assert not is_valid_plate(plate), f"Should reject {plate}"

# test_color_detector.py
def test_color_similarity():
    """HSV template matching accuracy"""
    red_img = np.full((100, 100, 3), (0, 255, 255), dtype=np.uint8)  # HSV red
    detected = detect_color(red_img)
    assert detected == "kirmizi"

# test_trajectory.py
def test_slalom_detection():
    """Periyodik motion tespit etmesi"""
    sinusoidal_trajectory = np.sin(np.linspace(0, 4*np.pi, 100)) * 50 + 320
    is_slalom, conf = detect_slalom(sinusoidal_trajectory)
    assert is_slalom and conf > 0.7
```

### 12.2 Integration Tests

```python
# test_inference_end_to_end.py
def test_full_pipeline():
    """Dummy video ile full pipeline"""
    dummy_video = create_dummy_video("test_input.mp4")
    output = run_inference(dummy_video, model_a_weights, model_b_weights)
    
    # Schema validation
    assert "arac_bilgisi" in output
    assert "tespitler" in output
    assert isinstance(output["arac_bilgisi"]["confidence_score"], float)
    assert 0.0 <= output["arac_bilgisi"]["confidence_score"] <= 1.0
```

---

## 13. PERFORMANCE BENCHMARKING

### 13.1 Beklenen Çıkarım Hızı

```
Donanım: Tesla T4 + 16GB RAM + 4 vCPU

┌─────────────────────────────────┬──────────┬───────────┐
│ Operasyon                       │ ms/frame │ FPS       │
├─────────────────────────────────┼──────────┼───────────┤
│ Video read (I/O)                │ 8        │ 125 FPS   │
│ Preprocess (resize, normalize)  │ 12       │ 83 FPS    │
│ Model A inference (YOLOv8m)     │ 25       │ 40 FPS    │
│ Color detection (HSV)           │ 8        │ 125 FPS   │
│ OCR (EasyOCR, small plate)      │ 35       │ 28.6 FPS  │
│ Model B inference (YOLOv8n)     │ 18       │ 55.5 FPS  │
│ Trajectory tracking (Kalman)    │ 5        │ 200 FPS   │
│ JSON formatting & write         │ 10       │ 100 FPS   │
├─────────────────────────────────┼──────────┼───────────┤
│ TOTAL per frame                 │ 121 ms   │ 8.3 FPS   │
└─────────────────────────────────┴──────────┴───────────┘

1920x1080 video, 30 FPS:
├─ Frame count in 10 min: 30 * 60 * 10 = 18,000
├─ Processing time: 18,000 * 121ms = 2,178 sec ≈ 36 min
├─ 10 min timeout → ~5 min video (150 frames) max
└─ Video segmentation optimization gerekebilir
```

### 13.2 Optimization Noktaları

```python
# 1. Model parallelization (if applicable)
# Model A ve B sıralı değil, frame queue kullanılabilir
# ├─ Thread-1: Model A inferences
# ├─ Thread-2: Model B inferences
# └─ Main: I/O coordination

# 2. Batch inference (multi-frame)
# ├─ Buffer 4 frames → batch process
# ├─ → Model A throughput 4x
# ├─ ⚠️ Ama temporal dependencies care
# └─ Slalom detection için problematic

# 3. Model quantization
# ├─ int8 quantization: 4x memory, 2-3x speed
# ├─ Accuracy loss: ~1-2%
# ├─ Risky: inference accuracy critical
# └─ Not recommended for this task

# 4. TensorRT optimization
# ├─ YOLOv8 → TensorRT conversion
# ├─ Speed: +40-60%
# ├─ Memory: -20%
# └─ Feasible!
```

---

## 14. TÜRKÇE KARAKTER YÖNETİMİ (KRİTİK)

### 14.1 Allowed Characters

```
✅ Allowed:
- İngilizce harfler: a-z, A-Z
- Rakamlar: 0-9
- Semboller: _

❌ NOT Allowed:
- Türkçe: ç, ğ, ı, ö, ş, ü, Ç, Ğ, İ, Ö, Ş, Ü
- Unicode: emoji, special symbols

Mapping (if needed):
├─ ç → c
├─ ğ → g
├─ ı → i
├─ ö → o
├─ ş → s
└─ ü → u
```

### 14.2 JSON Encoding

```python
# ✅ DOĞRU:
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    # ensure_ascii=False: Türkçe chars directly write
    # ama bu case'de Türkçe char yok anyway

# JSON internal check:
for key, value in output.items():
    if isinstance(value, str):
        # Strip any Turkish chars
        value = value.encode("ascii", "ignore").decode("ascii")

# Plaka örneği:
# ✅ "34ABC123" (doğru - tüm uppercase)
# ❌ "34abc123" (yanlış - lowercase)
# ❌ "34 ABC 123" (yanlış - boşluk)
# ❌ "34Abc123" (yanlış - mixed case)
```

---

## 15. CONTAINERIZATION BEST PRACTICES

### 15.1 Security & Isolation

```dockerfile
# ✅ BEST PRACTICE:
# 1. Non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# 2. Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python3 -c "import torch; print(torch.cuda.is_available())"

# 3. Immutable image
RUN chmod -R 555 /app/weights

# 4. Resource limits
# memory: 14GB (16GB - overhead)
# cpus: 3.5 (4 cores - 0.5 overhead)
```

### 15.2 Artifact Management

```dockerfile
# ✅ Model weights inside Docker
COPY weights/best_a.pt /app/weights/
COPY weights/best_b.pt /app/weights/

# ⚠️ Eğer weights external:
# docker run -v /path/to/weights:/app/weights:ro ...

# Git LFS usage (if in repo):
# .gitattributes:
# *.pt filter=lfs diff=lfs merge=lfs -text
```

---

## SUMMARY: Neden Bu Mimari?

| Karar | Neden | Trade-off |
|-------|-------|-----------|
| 2 Model | Gradient interference | Complexity +1x |
| YOLOv8m | Speed-accuracy balance | mAP marginal -2% vs larger |
| HSV+CNN renk | Domain-specific | Extra inference 8ms |
| EasyOCR | Türkçe + GPU support | Başka solutions'dan +15ms |
| Sequential inference | Deterministic, debugging | Parallelization benefit -20% |
| Config.yaml | Team collaboration | One more file |
| No env checks | Specification compliance | Can't debug eval env |

**Netice:** Production-grade, compliant, scalable, team-friendly architecture.

---

*Rapor Tarihi: 2026-06-27*
*Senior AI Architect*
