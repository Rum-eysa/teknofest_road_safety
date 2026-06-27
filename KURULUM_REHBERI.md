# Teknofest Road Safety - Sorun Çözümü Rehberi

## 🚀 HIZLI BAŞLANGIÇ

### Adım 1: Gerekli Dosyaları İndir
Aşağıdaki 3 dosyayı indir ve repo kök dizinine koy:
- `fix_script_v2.py` (Oto düzeltme scripti)
- `validate_json.py` (JSON validator)
- `HATALAR_VE_COZUMLER.md` (Detaylı doküman)

### Adım 2: Script'i Çalıştır
```bash
cd teknofest_road_safety/
python fix_script_v2.py
```

### Adım 3: Sonuç Kontrol
```bash
# Test output'unu valide et
python validate_json.py <output_json_path>
```

---

## 📋 YAPILACAKLAR LİSTESİ

### ✅ Otomatik Yapan fix_script_v2.py

1. **requirements.txt oluşturma**
   - loguru, opencv, torch, ultralytics vb. tüm bağımlılıklar

2. **.gitattributes oluşturma**
   - Windows/Linux line ending sorunları çözüm
   - CRLF → LF dönüştürme

3. **src/utils.py güncelleme**
   - MODEL_A_YOLO_CLASSES (Roboflow alfabetik sırası)
   - MODEL_B_YOLO_CLASSES (kemer_takili eklendi)
   - ASCII-safe dönüştürme fonksiyonları

4. **src/predict.py güncelleme**
   - kemer_takili filtreleme
   - slalom filtreleme
   - Majority voting
   - Frame skip (T4 timeout önleme)

5. **YAML config güncellemeleri**
   - Model A configs
   - Model B configs
   - Roboflow sırası ile uyumlu

### ⚠️ Manual Kontroller

1. **YAML Dosya Formatları Kontrol Et**
   ```bash
   # Model A configs
   cat configs/model_a_config.yaml | grep -A 10 "classes:"
   cat configs/model_b_config.yaml | grep -A 15 "classes:"
   ```

2. **Türkçe Karakter Kontrolü**
   ```bash
   # Türkçe karakter arayan komut
   grep -r "[ç-ğ-ı-ö-ş-ü-Ç-Ğ-İ-Ö-Ş-Ü]" src/
   ```
   ✅ Sonuç boş olmalı!

3. **Requirements Install**
   ```bash
   pip install -r requirements.txt
   ```

4. **Git Ayarları**
   ```bash
   # Line ending'leri normalize et
   git add --renormalize -A
   git commit -m "fix: normalize line endings"
   ```

---

## 🔍 VALIDATION KONTROL LİSTESİ

### JSON Çıktı Formatı
- [x] `video_id` - string
- [x] `arac_bilgisi` - object
  - [x] `tip` - ("sedan", "suv", "hatchback", ...)
  - [x] `plaka` - Türkiye plaka formatı (34ABC123)
  - [x] `renk` - ("beyaz", "siyah", "kirmizi", ...)
  - [x] `confidence_score` - 0.0-1.0 float
- [x] `tespitler` - array
  - [x] `zaman_saniye` - float (saniyelerde)
  - [x] `kategori` - ("sofor_eylemi", "nesneler", "yolcular")
  - [x] `etiket` - kategoriye bağlı
  - [x] `confidence_score` - 0.0-1.0 float

### Etiket Doğruluğu
- [x] Tüm etiketler **ASCII-safe** (Türkçe karakter YOK)
- [x] Tüm etiketler **küçük harf**
- [x] JSON anahtarları **TAM OLARAK** belirtilen adlar
  - [x] "confidence_score" (NOT: "guven_skoru" ✗)
  - [x] "zaman_saniye" (NOT: "time" ✗)
  - [x] "kategori" (NOT: "category" ✗)

### Geçerli Etiketler

**sofor_eylemi:**
```
arkaya_bakma, esneme, sigara_icme, su_icme, telefonla_konusma, 
slalom, etrafa_bakinma, emniyet_kemeri_ihlali
```

**nesneler:**
```
teknocan, bilgisayar
```

**yolcular:**
```
arka_koltuk_1, arka_koltuk_2, on_koltuk
```

**renkler:**
```
beyaz, siyah, gri, kirmizi, mavi, sari, yesil, turuncu, kahverengi
```

**araç tipleri:**
```
sedan, suv, hatchback, pickup, minibus, panelvan, kamyon
```

---

## 🐛 SAĞLIK KONTROL (DEBUG)

### Problem: "ATLA (pattern bulunamadi)"

**Çözüm:** YAML dosya formatını kontrol et
```bash
cat configs/model_a_config.yaml | head -20
```

Dosya yapısı şöyle olmalı:
```yaml
model:
  classes:
    - hatchback
    - kamyon
    ...
```

Değilse, bu satırları manuel ekle/düzelt

### Problem: "kirmizi" yerine "kırmızı" yazılıyor

**Çözüm:** `ensure_ascii_safe()` fonksiyonu kullan
```python
from src.utils import ensure_ascii_safe
output = ensure_ascii_safe(raw_output)
```

### Problem: Plaka formatı hata veriyor

**Çözüm:** Normalize fonksiyonunu kullan
```python
from src.utils import normalize_plate, is_valid_plate

raw = "34 A 1234"
normalized = normalize_plate(raw)  # "34A1234"
if is_valid_plate(normalized):
    print("OK")
```

---

## 📤 GitHub Push Öncesi Kontrol

```bash
# 1. Tüm değişiklikleri gör
git status

# 2. Diff kontrol et
git diff src/utils.py
git diff src/predict.py

# 3. JSON test dosyası varsa valide et
python validate_json.py test_output.json

# 4. Commit
git add -A
git commit -m "fix: encoding, patterns, validation - v2"
git log --oneline -5  # Son 5 commit'i gör

# 5. Push
git push
```

---

## 🐳 DOCKER HAZIRLIĞI

### Dockerfile Kontrol
```dockerfile
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# ✅ Gerekli sistem paketleri
RUN apt-get update && apt-get install -y python3 python3-pip

# ✅ Working directory
WORKDIR /app

# ✅ Klasörler
RUN mkdir -p /app/data/input /app/data/output /app/models

# ✅ Requirements
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# ✅ Kodlar
COPY src/ /app/src/
COPY main.py .

# ✅ Otomatik başlama
CMD ["python3", "main.py"]
```

### Test
```bash
docker build -t teknofest:latest .

docker run --rm --gpus all \
  -v /path/to/video.mp4:/app/data/input/video.mp4 \
  -v /path/to/output:/app/data/output \
  teknofest:latest
```

Çıktı kontrol:
```bash
ls -la /path/to/output/results.json
python validate_json.py /path/to/output/results.json
```

---

## 📞 Sık Sorulan Sorular (SSS)

**S: Script çalışmıyor, "loguru modülü bulunamadı" hatası alıyorum**
A: Önce `pip install -r requirements.txt` çalıştır

**S: YAML dosyaları güncellenmedi**
A: Dosyaları manuel kontrol et, format farklı olabilir

**S: Plaka regex'i çalışmıyor**
A: Boşlukları kaldırıp normalize et (`normalize_plate()` fonksiyonu kullan)

**S: "kırmızı" yazılıyor, "kirmizi" değil**
A: `ensure_ascii_safe()` fonksiyonunu output'a uygula

**S: Windows'ta CRLF uyarıları**
A: `.gitattributes` dosyası kontrol et veya `git add --renormalize -A` çalıştır

**S: JSON'ımız geçerli mi?**
A: `python validate_json.py output.json` çalıştır

---

## 🎯 HEDEF

Kodunuz 10 dakikalık timeout içinde:
1. `/app/data/input/video.mp4` dosyasını oku
2. Analiz yap
3. JSON dosyasını `/app/data/output/results.json` dosyasına yaz
4. **Tüm etiketler ASCII + küçük harf + dokümantasyona uygun olmalı**

---

## ✅ SON ADIM

Kurulum bittiğinde repo'nun durumu:
```
teknofest_road_safety/
├── .gitattributes           ← YENİ
├── requirements.txt         ← YENİ
├── Dockerfile
├── main.py
├── src/
│   ├── __init__.py
│   ├── utils.py            ← GÜNCELLENDI
│   ├── predict.py          ← GÜNCELLENDI
│   ├── color_detector.py
│   ├── ocr_handler.py
│   └── trajectory_analyzer.py
├── configs/
│   ├── model_a_config.yaml          ← GÜNCELLENDI
│   ├── model_b_config.yaml          ← GÜNCELLENDI
│   └── diğer yaml dosyaları
└── weights/
    └── ...
```

Sorularınız varsa GitHub repo'sunda issue açabilirsiniz! 🚀
