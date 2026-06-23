# Teknofest 2026 - Akıllı Yol Güvenliği Yarışması

5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması FTR aşaması için hazırlanmış Docker projesi.

## Proje Yapısı

```
./
├── Dockerfile
├── README.md
├── requirements.txt
├── main.py
├── models/
│   └── best_model.pt
└── src/
    ├── predict.py
    └── utils.py
```

## Kullanım

Model ağırlıklarını `models/best_model.pt` yoluna yerleştirin, ardından aşağıdaki komutları çalıştırın:

```bash
docker build -t teknofest/road_safety:latest .
docker run --rm --gpus all \
  -v /path/to/video.mp4:/app/data/input/video.mp4 \
  -v ./output:/app/data/output \
  teknofest/road_safety:latest
```

## Giriş / Çıkış Yolları

| Amaç | Yol |
|------|-----|
| Girdi videosu | `/app/data/input/video.mp4` |
| Çıktı JSON | `/app/data/output/results.json` |
| Model ağırlıkları | `/app/models/best_model.pt` |

## Çıktı Formatı

Program, yarışma şablonuna uygun `results.json` dosyası üretir. JSON anahtarları ve etiket değerleri ASCII karakterli ve küçük harflidir.

## Sonuç

Çıktı dosyası: `output/results.json`
