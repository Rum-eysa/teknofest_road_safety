# YAML Dosya Format Problemi - Çözüm Rehberi

## ✅ Başarılı Olanlar
- ✅ requirements.txt oluşturuldu
- ✅ .gitattributes oluşturuldu  
- ✅ src/utils.py güncellendi (MODEL_A_YOLO_CLASSES vb.)
- ✅ src/predict.py güncellendi (majority voting, frame skip)
- ✅ 9 dosya değişikliği (Git diff)

## ⚠️ Manual Yapılması Gereken: YAML Dosyaları

Script, YAML dosyalarının formatı farklı olduğu için pattern eşleştirilemedi.
Bunun sebebi: **Dosyalar 'classes:' içeriyor ama format biraz farklı**

### Adım 1: YAML Dosya Formatını Kontrol Et

```bash
# Windows PowerShell'de aç
cd teknofest_road_safety
type configs/model_a_config.yaml
```

Çıktı şöyle görünebilir:

```yaml
# YOLOv8 detection model configuration
model:
  type: detection
  classes:
    - hatchback
    - kamyon
    - minibus
    # ... vs
```

VEYA:

```yaml
classes:
  - hatchback
  - kamyon
  - minibus
```

---

## Adım 2: Manual Düzeltme

### YAML'de classes kısmını ŞÖYLE DÜZELTİN:

#### Model A Configs (model_a_config.yaml, model_a_config_local.yaml, config_exp_aggressive_aug.yaml, config_exp_combined.yaml):

**Eski Format:**
```yaml
  classes:
    - "hatchback"
    - "kamyon"
    # ... diğer eski sınıflar
```

**Yeni Format (ŞUNU YAZIN):**
```yaml
  classes:
    - hatchback
    - kamyon
    - minibus
    - panelvan
    - pickup
    - plaka
    - sedan
    - suv
```

#### Model B Configs (model_b_config.yaml, model_b_config_local.yaml):

**Yeni Format (ŞUNU YAZIN):**
```yaml
  classes:
    - arka_koltuk_1
    - arka_koltuk_2
    - arkaya_bakma
    - bilgisayar
    - emniyet_kemeri_ihlali
    - esneme
    - etrafa_bakinma
    - kemer_takili
    - on_koltuk
    - sigara_icme
    - su_icme
    - teknocan
    - telefonla_konusma
```

---

## Adım 3: Doğru Şekilde Düzelt

### **SEÇENEK A: Windows'ta Notepad++'de Düzelt** (Kolay)

1. Notepad++ aç
2. `configs/model_a_config.yaml` dosyasını aç
3. `classes:` satırını bul (Ctrl+F)
4. Bulduğun satırdan sonraki tüm `-` satırlarını sil
5. Yukarıdaki "Yeni Format" kısmını yapıştır
6. Kaydet (Ctrl+S)
7. Diğer dosyalar için tekrarla

### **SEÇENEK B: PowerShell Script ile Otomatik** (Hızlı)

Aşağıdaki PowerShell scriptini çalıştır:

```powershell
# YAML Düzeltme Script

# Model A Classes
$model_a_classes = @"
  classes:
    - hatchback
    - kamyon
    - minibus
    - panelvan
    - pickup
    - plaka
    - sedan
    - suv
"@

# Model B Classes
$model_b_classes = @"
  classes:
    - arka_koltuk_1
    - arka_koltuk_2
    - arkaya_bakma
    - bilgisayar
    - emniyet_kemeri_ihlali
    - esneme
    - etrafa_bakinma
    - kemer_takili
    - on_koltuk
    - sigara_icme
    - su_icme
    - teknocan
    - telefonla_konusma
"@

# Model A Dosyaları
foreach ($file in @("configs/model_a_config.yaml", "configs/model_a_config_local.yaml", "configs/config_exp_aggressive_aug.yaml", "configs/config_exp_combined.yaml")) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        # classes: öncesi ve sonrası ayır
        if ($content -match '(.*?)(  classes:.*?)(\n[a-z_]+:|$)') {
            $before = $matches[1]
            $after = $matches[3]
            $new_content = $before + $model_a_classes + $after
            Set-Content -Path $file -Value $new_content -Encoding UTF8
            Write-Host "✅ Düzeltildi: $file"
        }
    }
}

# Model B Dosyaları
foreach ($file in @("configs/model_b_config.yaml", "configs/model_b_config_local.yaml")) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        if ($content -match '(.*?)(  classes:.*?)(\n[a-z_]+:|$)') {
            $before = $matches[1]
            $after = $matches[3]
            $new_content = $before + $model_b_classes + $after
            Set-Content -Path $file -Value $new_content -Encoding UTF8
            Write-Host "✅ Düzeltildi: $file"
        }
    }
}

Write-Host "✅ TÜMLERI TAMAMLANDI!"
```

Çalıştırma:
```powershell
# PowerShell'de aç ve yapıştır, Enter'e bas
```

### **SEÇENEK C: Python Script ile Otomatik** (En Güvenli)

Aşağıdaki Python kodunu `fix_yaml.py` olarak kaydet:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re

MODEL_A_CLASSES = """  classes:
    - hatchback
    - kamyon
    - minibus
    - panelvan
    - pickup
    - plaka
    - sedan
    - suv"""

MODEL_B_CLASSES = """  classes:
    - arka_koltuk_1
    - arka_koltuk_2
    - arkaya_bakma
    - bilgisayar
    - emniyet_kemeri_ihlali
    - esneme
    - etrafa_bakinma
    - kemer_takili
    - on_koltuk
    - sigara_icme
    - su_icme
    - teknocan
    - telefonla_konusma"""

# Model A Dosyaları
files_a = [
    'configs/model_a_config.yaml',
    'configs/model_a_config_local.yaml',
    'configs/config_exp_aggressive_aug.yaml',
    'configs/config_exp_combined.yaml',
]

for fpath in files_a:
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # classes: ile başlayan bloğu bul ve değiştir
        content = re.sub(
            r'  classes:\n(?:    - [^\n]*\n)+',
            MODEL_A_CLASSES + '\n',
            content
        )
        
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Düzeltildi: {fpath}")

# Model B Dosyaları
files_b = [
    'configs/model_b_config.yaml',
    'configs/model_b_config_local.yaml',
]

for fpath in files_b:
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = re.sub(
            r'  classes:\n(?:    - [^\n]*\n)+',
            MODEL_B_CLASSES + '\n',
            content
        )
        
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Düzeltildi: {fpath}")

print("\n✅ TÜMLERI TAMAMLANDI!")
```

Çalıştırma:
```bash
python fix_yaml.py
```

---

## Adım 4: Doğrulama

Düzeltme sonrasında kontrol et:

```bash
# Model A config kontrolü
type configs/model_a_config.yaml | findstr /A:2 "classes:"

# Model B config kontrolü
type configs/model_b_config.yaml | findstr /A:15 "classes:"
```

Çıktı şöyle olmalı:
```
  classes:
    - hatchback
    - kamyon
    ...
```

---

## Adım 5: validate_json.py Doğru Kullanımı

Script'i test etmek için:

```bash
# 1. Örnek JSON dosyası oluştur
echo {\"video_id\": \"test.mp4\", \"arac_bilgisi\": {\"tip\": \"sedan\", \"plaka\": \"34ABC123\", \"renk\": \"beyaz\", \"confidence_score\": 0.95}, \"tespitler\": [{\"zaman_saniye\": 5.0, \"kategori\": \"sofor_eylemi\", \"etiket\": \"telefonla_konusma\", \"confidence_score\": 0.89}]} > test_output.json

# 2. Valide et
python validate_json.py test_output.json
```

Doğru çıktı:
```
======================================================================
Kontrol ediliyor: test_output.json
======================================================================

--- Üst Seviye Alanlar ---
✅ video_id: test.mp4
...
✅ TÜM KONTROLLER GEÇTI - JSON Yarışma Standartlarına Uygun!
```

---

## Adım 6: Git Commit

Düzeltme sonrasında:

```bash
git add -A
git commit -m "fix: YAML classes format - manual fix"
git push
```

---

## 📋 ÖZETİN ÖZETİ

| Dosya | Durum | Yapılması Gereken |
|-------|-------|-------------------|
| requirements.txt | ✅ Tamam | - |
| .gitattributes | ✅ Tamam | - |
| src/utils.py | ✅ Tamam | - |
| src/predict.py | ✅ Tamam | - |
| YAML Configs | ⚠️ Format | SEÇENEK A/B/C'den birini yapın |

---

## Sorun Giderme

### Problem: "YAML dosyası UTF-8 değil" hatası
**Çözüm:** Dosyayı Notepad++'de açıp "Encoding → UTF-8 (without BOM)" yap

### Problem: Düzeltme sonrasında "syntax error"
**Çözüm:** YAML'de indentation (tab/space) karışmış olabilir. Tüm satırları 4 space ile gir (tab kullanma)

### Problem: Hala pattern eşleşmiyor
**Çözüm:** YAML dosyasını ilk 50 satırı kopyalayıp bana gönder, format kontrol edebilirim

---

**Sorunları çözdüğünüzde repo'ya push edebilirsiniz!** ✅
