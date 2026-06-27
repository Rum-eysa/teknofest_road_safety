# ⚡ HEMEN YAPILACAK İŞLER - Adım Adım

## 🎯 KISACASı

Script başarılı çalıştı **ANCAK** YAML dosyaları manuel düzeltme gerekli.
2 adımla bitir:

```bash
# Adım 1: YAML Dosyaları Düzelt
python fix_yaml.py

# Adım 2: Git'e Push Et
git add -A
git commit -m "fix: YAML classes format - manual fix"
git push
```

---

## 📥 Yapması Gerekenler (DETAYLI)

### **1. GitHub Repo'nuzda 3 YENI DOSYA KOPYALAYIN**

Aşağıdaki dosyaları indiriniz ve `teknofest_road_safety/` kökünde koyunuz:

1. **fix_yaml.py** ← YAML düzeltme scripti
2. **validate_json.py** ← JSON kontrol aracı (zaten var)
3. **example_output.json** ← Test örneği

### **2. YAML Dosyalarını Düzelt (OTOMATİK)**

Terminal/PowerShell'de:

```bash
cd teknofest_road_safety/
python fix_yaml.py
```

**Beklenen Çıktı:**
```
======================================================================
  Teknofest Road Safety - YAML Classes Otomatik Düzeltme
======================================================================

[1/2] Model A Config Dosyaları Düzeltiliyor...

✅ DÜZELTILDI: configs/model_a_config.yaml
✅ DÜZELTILDI: configs/model_a_config_local.yaml
✅ DÜZELTILDI: configs/config_exp_aggressive_aug.yaml
✅ DÜZELTILDI: configs/config_exp_combined.yaml

[2/2] Model B Config Dosyaları Düzeltiliyor...

✅ DÜZELTILDI: configs/model_b_config.yaml
✅ DÜZELTILDI: configs/model_b_config_local.yaml

======================================================================
ÖZET
======================================================================
Model A: 4/4 düzeltildi
Model B: 2/2 düzeltildi

✅ TÜM DOSYALAR BAŞARIYLA DÜZELTILDI!
```

### **3. JSON Validator ile Test Et**

```bash
# Örnek JSON'u valide et
python validate_json.py example_output.json
```

**Beklenen Çıktı:**
```
======================================================================
Kontrol ediliyor: example_output.json
======================================================================

--- Üst Seviye Alanlar ---
✅ video_id: video_001.mp4

--- Araç Bilgisi Kontrolü ---
✅ tip geçerli: sedan
✅ Plaka formatı geçerli: 34ABC123 -> 34ABC123
✅ renk geçerli: beyaz
✅ confidence_score geçerli: 0.94

--- Tespitler Kontrol Ediliyor ---

  [0] Kontrol ediliyor...
✅ tespitler[0].zaman_saniye geçerli: 14.5s
✅ tespitler[0].kategori geçerli: sofor_eylemi
✅ tespitler[0].etiket geçerli: telefonla_konusma
✅ tespitler[0].confidence_score geçerli: 0.89

  [1] Kontrol ediliyor...
...

======================================================================
ÖZET RAPOR
======================================================================

✅ Başarılı: 26
❌ Hata: 0
⚠️  Uyarı: 0

======================================================================
✅ TÜM KONTROLLER GEÇTI - JSON Yarışma Standartlarına Uygun!
```

### **4. Git'e Push Et**

```bash
# Durum kontrol et
git status

# Tüm değişiklikleri stage et
git add -A

# Commit et
git commit -m "fix: YAML classes format + validation - Complete"

# Push et
git push

# Kontrol et (5 saniye bekle)
git log --oneline -3
```

---

## ✅ Kontrol Listesi

- [ ] `fix_yaml.py` indirildi ve `teknofest_road_safety/` klasörüne kopyalandı
- [ ] `python fix_yaml.py` başarıyla çalıştı (tüm ✅ görmek gerekir)
- [ ] `python validate_json.py example_output.json` başarıyla çalıştı
- [ ] `git add -A` yapıldı
- [ ] `git commit` yapıldı
- [ ] `git push` yapıldı
- [ ] GitHub'da değişiklikleri görmek için repo sayfasını yenile

---

## 🎉 İşlemi Bitirince

Repo'nuzda şu dosyalar olmalı:

```
teknofest_road_safety/
├── .gitattributes              ✅ (Script oluşturdu)
├── requirements.txt            ✅ (Script oluşturdu)
├── fix_yaml.py                 ✅ (Yeni dosya)
├── validate_json.py            ✅ (Yeni dosya)
├── example_output.json         ✅ (Örnek)
├── Dockerfile
├── main.py
├── src/
│   ├── utils.py               ✅ (Güncellenmiş)
│   ├── predict.py             ✅ (Güncellenmiş)
│   └── ...
├── configs/
│   ├── model_a_config.yaml    ✅ (Güncellenmiş)
│   ├── model_b_config.yaml    ✅ (Güncellenmiş)
│   └── diğer yaml dosyaları   ✅ (Güncellenmiş)
└── ...
```

---

## 🆘 Hata Alırsanız

### **Hata: "No such file or directory: configs/model_a_config.yaml"**
**Çözüm:** Repo'nun kök dizininden (`cd teknofest_road_safety/`) script'i çalıştırın

### **Hata: "ModuleNotFoundError: No module named 'loguru'"**
**Çözüm:** `pip install -r requirements.txt` çalıştırın

### **Hata: "❌ HATA: ... (format eşleşmiyor)"**
**Çözüm:** `YAML_MANUAL_FIX.md` dosyasını açıp SEÇENEKler bölümüne gidin

### **GitHub'a Push edilmiyor**
**Çözüm:**
```bash
# SSH anahtarı kontrol et
git config user.name
git config user.email

# Yoksa ayarla
git config --global user.name "İsminiz"
git config --global user.email "email@example.com"

# Tekrar push et
git push
```

---

## 🚀 Bitirince Yapacağınız Şeyler

### 1. **Docker Image'ı Oluştur**
```bash
docker build -t teknofest:latest .
```

### 2. **Requirements'ları İndir**
```bash
pip install -r requirements.txt
```

### 3. **Model Ağırlıklarını Yerleştir**
```bash
# Eğer model ağırlıkları varsa
cp models/best_model.pt models/best_a.pt
```

### 4. **Test Et** (Eğer test videosu varsa)
```bash
docker run --rm --gpus all \
  -v /path/to/video.mp4:/app/data/input/video.mp4 \
  -v ./output:/app/data/output \
  teknofest:latest

# Sonuç kontrol
python validate_json.py output/results.json
```

---

## 📞 Son Kontrol

Komut satırında test et:

```bash
# 1. Repository'de misin?
pwd  # (Windows: cd)
# Çıktı: .../teknofest_road_safety

# 2. fix_yaml.py var mı?
ls fix_yaml.py  # Windows: dir fix_yaml.py

# 3. Çalıştır
python fix_yaml.py

# 4. Git push
git push

# 5. GitHub'da kontrol et (browser'da açarak)
https://github.com/Rum-eysa/teknofest_road_safety
```

---

**Başarılar! Sorunlar varsa sorabilirsiniz.** 🚀
