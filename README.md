# Teknofest 2026 - Akıllı Yol Güvenliği Yarışması

5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması için hazırlanmış Docker projesi.

## Kullanım

```bash
docker build -t rumicim/road_safety:latest .
docker run --rm --gpus all \
  -v /path/to/video.mp4:/app/data/input/video.mp4 \
  -v ./output:/app/data/output \
  teknofest/road_safety:latest
```

## Sonuç

Çıktı dosyası: `output/results.json`
