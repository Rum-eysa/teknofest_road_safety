# Cikarim (Inference) Pipeline — Teknofest 2026 Akilli Yol Guvenligi

FTR sartnamesindeki ornek mimariye (main.py, src/predict.py, src/utils.py)
sadik kalinarak hazirlanmis, Model A (Arac Bilgisi) ve Model B (Yol Guvenligi)
modellerini AYNI video uzerinde calistirip, TEK bir konsolide `results.json`
ureten production-grade cikarim konteyneri.

## Klasor Yapisi
```
cikarim_pipeline/
├── Dockerfile               # nvidia/cuda:12.1.0-base-ubuntu22.04
├── docker-compose.yml
├── config.yaml               # Yollar, esikler, sinif listeleri (FTR sabitleri)
├── requirements.txt
├── main.py                   # Giris noktasi (orkestrasyon)
├── models/                   # best_model_a.pt, best_model_b.pt buraya yerlestirilir
└── src/
    ├── predict.py             # Model A + Model B birlikte calistirma
    ├── utils.py                # Loglama, video bilgisi, ASCII-safe, JSON sema
    ├── color_utils.py          # HSV tabanli renk tahmini
    ├── ocr_utils.py            # EasyOCR + plaka regex normalizasyonu
    └── tracking.py             # Slalom heuristik tespiti (trajektori analizi)
```

## Calistirma
```bash
# 1) Egitilmis agirliklari yerlestirin
cp /egitimden/gelen/best_model_a.pt models/
cp /egitimden/gelen/best_model_b.pt models/

# 2) Test videosunu yerlestirin
cp /test/videosu.mp4 data/input/video.mp4

# 3) Build + calistir
docker compose build
docker compose run --rm cikarim

# 4) Sonuc
cat data/output/results.json
```

## Cikti Semasi (results.json)
```json
{
  "video_id": "video.mp4",
  "arac_bilgisi": {
    "tip": "sedan",
    "plaka": "34ABC123",
    "renk": "beyaz",
    "confidence_score": 0.87
  },
  "tespitler": [
    {
      "zaman_saniye": 5.2,
      "kategori": "sofor_eylemi",
      "etiket": "telefonla_konusma",
      "confidence_score": 0.89
    },
    {
      "zaman_saniye": 12.8,
      "kategori": "nesneler",
      "etiket": "teknocan",
      "confidence_score": 0.92
    },
    {
      "zaman_saniye": 18.4,
      "kategori": "arac_hareketi",
      "etiket": "slalom",
      "confidence_score": 0.75
    }
  ]
}
```

## Onemli Tasarim Kararlari (Neden?)
| Karar | Neden |
|---|---|
| `plaka` HARICINDE her sey ASCII-safe + kucuk harf | FTR Madde 3.4/3.6 |
| Model A & B `ThreadPoolExecutor` ile eszamanli | FTR Madde 3.3 — ayni karede asenkron/ardisik calisma |
| Slalom = heuristik trajektori analizi (ML degil) | FTR Madde C — sabit veri seti yok |
| Renk tahmini = HSV (CNN degil) | FTR Madde B — 5 saatlik egitim butcesini renk icin harcamamak |
| `arac_bilgisi.confidence_score` = tek ortak deger | FTR Madde 3.5 |
| Ortam tespiti (env/hostname/IP) YOK | FTR Madde E — her ortamda BIREBIR ayni davranis |
| Taban imaj: `nvidia/cuda:12.1.0-base-ubuntu22.04` | FTR'de ACIKCA istenen imaj |
