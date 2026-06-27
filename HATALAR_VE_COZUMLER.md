# Teknofest Road Safety - Hata Analizi ve Çözümler

## 🔴 Tespit Edilen Hatalar

### 1. **Character Encoding Sorunu**
```
'k' is not recognized as an internal or external command
```
**Sebep:** Windows'ta Python script çalışırken karakter kodlama sorunu (UTF-8 vs ANSI)

**Çözüm:**
- Tüm Python dosyaları UTF-8 BOM olmadan kaydedilmeli
- Script başında encoding tanımla:
```python
# -*- coding: utf-8 -*-
import sys
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
```

---

### 2. **YAML Pattern Matching Başarısız**
```
ATLA (pattern bulunamadi): configs/model_a_config.yaml
```

**Sebep:** Regex pattern, YAML dosyalarındaki class tanımlarıyla eşleşmiyor.

**Mevcut Pattern:**
```python
pattern_a = r'  classes:\n(?:    - "[^"]+"\n)+'
```

**Sorun:** 
- Bazı YAML dosyalarında satır başı boşluk sayısı farklı olabilir
- Türkçe karakterler (ç, ğ, ı, ö, ş, ü) regex ile sorun çıkarabilir
- Yorum satırları (#) dahil edilmemesi

**Yeni Pattern (Daha Robust):**
```python
pattern_a = r'classes:\s*\n([\s\S]*?)(?=\n[a-z_]+:|$)'
```

---

### 3. **Modül Import Hatası**
```
[UYARI] Import dogrulamasi calismadi: No module named 'loguru'
```

**Sebep:** `loguru` kütüphanesi kurulu değil.

**Çözüm:** requirements.txt oluştur ve tüm bağımlılıkları ekle

---

### 4. **Git CRLF/LF Uyarıları**
```
warning: in the working copy of 'configs/model_a_config.yaml', 
CRLF will be replaced by LF
```

**Sebep:** Windows (CRLF) ve Linux (LF) satır sonu farklılığı

**Çözüm:** `.gitattributes` dosyası oluştur

---

### 5. **Türkçe Karakter Kısıtlaması**

Dokümantasyon (Sayfa 2) açıkça belirtmektedir:
> "Tüm kategori adları ve sınıf etiketleri Türkçe karakter içermeyen (ASCII-safe) ve küçük harfli standart metinler olmalıdır."

**Mevcut Kodda Sorun:**
```python
VALID_COLORS = {
    "beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"
}
```
✅ Burada doğru (ASCII-safe)

**Ama docstring'lerde sorun olabilir:**
```python
# "Türkçe karakter" yerine "Turkce karakter" kullanılmalı
```

---

## ✅ Yapılması Gereken Düzeltmeler

### Priority 1: HEMEN YAPILMALI

#### Dosya 1: `.gitattributes` (Yeni Dosya)
```
* text=auto
*.py text eol=lf
*.yaml text eol=lf
*.json text eol=lf
*.txt text eol=lf
```

#### Dosya 2: `requirements.txt` (Yeni Dosya)
```
loguru==0.7.0
opencv-python==4.8.0.74
numpy==1.24.3
torch==2.0.1
ultralytics==8.0.195
```

#### Dosya 3: `fix_script.py` (Geliştirilmiş Versiyon)
Aşağıda tam kod verilmiş...

---

### Priority 2: Windows Uyumluluğu

Python scriptler Windows'ta çalıştırırken:
1. **Dosya Yolu:** `\` yerine `os.path.join()` kullan
2. **Encoding:** Tüm file open() işlemlerinde `encoding='utf-8'` belirt
3. **Subprocess:** `shell=True` yerine `shell=False` kullan (Windows güvenlik)

---

## 📋 Kurulum Adımları

```bash
# 1. Repository'yi klonla
git clone https://github.com/Rum-eysa/teknofest_road_safety.git
cd teknofest_road_safety

# 2. .gitattributes ekle
echo '* text=auto
*.py text eol=lf
*.yaml text eol=lf' > .gitattributes

# 3. requirements.txt oluştur
pip install -r requirements.txt

# 4. Geliştirilmiş fix script'i çalıştır
python fix_script_v2.py

# 5. Git line endings'i düzelt
git add --renormalize -A
git commit -m "fix: normalize line endings"
```

---

## 🔍 Validation Checklist

Dokümantasyondaki (Sayfa 7) kontrol listesi:

- [ ] Dockerfile proje en üst dizininde
- [ ] Base image: `nvidia/cuda:12.1.0-base-ubuntu22.04`
- [ ] GPU CUDA yapılandırması var
- [ ] Input: `/app/data/input/video.mp4`
- [ ] Output: `/app/data/output/results.json`
- [ ] **TÜM ETIKETLER ASCII ve KÜÇÜK HARF**
- [ ] Docker run otomatik başlıyor

---

## ⚠️ En Önemli Kurallar (Dokümantasyon s.3-4)

1. **JSON Anahtarları SABIT:**
   ```json
   {
     "video_id": "",
     "arac_bilgisi": {
       "tip": "",
       "plaka": "",
       "renk": "",
       "confidence_score": 0.0
     },
     "tespitler": [
       {
         "zaman_saniye": 0.0,
         "kategori": "sofor_eylemi",
         "etiket": "",
         "confidence_score": 0.0
       }
     ]
   }
   ```

2. **Geçerli Kategori Değerleri:**
   - `sofor_eylemi` (NOT: sofor_eylemi ✓, sürücü_eylemi ✗)
   - `nesneler`
   - `yolcular`

3. **Geçerli Etiketler:**
   - **sofor_eylemi:** arkaya_bakma, esneme, sigara_icme, su_icme, telefonla_konusma, slalom, etrafa_bakinma, emniyet_kemeri_ihlali
   - **nesneler:** teknocan, bilgisayar
   - **yolcular:** arka_koltuk_1, arka_koltuk_2, on_koltuk

4. **Araç Renkleri:**
   beyaz, siyah, gri, kirmizi, mavi, sari, yesil, turuncu, kahverengi
   (kırmızı ✗ → kirmizi ✓)

5. **Araç Tipleri:**
   sedan, suv, hatchback, pickup, minibus, panelvan, kamyon

---

## 🛠️ Debug İçin Faydalı Kodlar

```python
# 1. Character encoding test
import json

test_data = {
    "kategori": "sofor_eylemi",  # ASCII-safe
    "etiket": "telefonla_konusma",  # ASCII-safe
}

# UTF-8 olmaması için kontrol
try:
    json.dumps(test_data, ensure_ascii=True)
    print("✓ ASCII-safe")
except:
    print("✗ ASCII olmayan karakter var!")

# 2. YAML Dosya Kontrolü
import re

yaml_content = open('config.yaml', 'r', encoding='utf-8').read()
if re.search(r'classes:', yaml_content):
    print("✓ Classes section bulundu")
else:
    print("✗ Classes section bulunamadı")

# 3. Plaka Regex Test
PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"((\s?[a-zA-Z]\s?)(\d{4,5})|(\s?[a-zA-Z]{2}\s?)(\d{3,4})|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$"
)

plates = ["34ABC123", "34 A 5678", "06AAA123"]
for p in plates:
    if PLATE_REGEX.match(p.upper()):
        print(f"✓ {p} valid")
    else:
        print(f"✗ {p} invalid")
```

---

## 📞 GitHub'a Push Öncesi

```bash
# 1. Local'de test et
python fix_script_v2.py

# 2. JSON output'ı valide et
python -m json.tool configs/test_output.json

# 3. Tüm etiketleri ASCII olup olmadığını kontrol et
grep -r "[ç-ğ-ı-ö-ş-ü-Ç-Ğ-İ-Ö-Ş-Ü]" src/

# 4. Commit et
git add -A
git commit -m "fix: encoding, patterns, requirements"
git push
```
