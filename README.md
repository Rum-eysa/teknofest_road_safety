# Teknofest 2026 - Akıllı Yol Güvenliği Yarışması

5G ve Yapay Zekâ ile Akıllı Yol Güvenliği Yarışması FTR aşaması için hazırlanmış Docker projesi.

## Proje Yapısı

```
./
├── Dockerfile              # Cikarim (inference) imaji
├── Dockerfile.train        # Model A/B egitim imaji
├── docker-compose.yml      # Model A hiperparametre denemeleri
├── README.md
├── requirements.txt
├── main.py                 # Cikarim giris noktasi
├── train.py                # Model A/B egitim giris noktasi
├── configs/
│   ├── model_a_config.yaml
│   ├── model_a_config_local.yaml
│   ├── model_b_config.yaml
│   ├── model_b_config_local.yaml
│   ├── config_exp_aggressive_aug.yaml
│   └── config_exp_combined.yaml
├── data/
│   ├── arac_bilgisi/       # Model A YOLO verisi
│   └── input/tespitler/    # Model B YOLO verisi
├── docs/
│   ├── PROJECT_EXPLANATION.md
│   └── arac_govde.png
├── models/
│   ├── best_a.pt
│   └── best_b.pt
└── src/
    ├── predict.py
    ├── color_detector.py
    ├── ocr_handler.py
    ├── trajectory_analyzer.py
    ├── utils.py
    └── training/
        ├── config_loader.py
        ├── dataset_yaml.py
        ├── train_yolo.py
        ├── train_model_a.py
        └── train_model_b.py
```

## Kullanım

Model ağırlıklarını `models/best_a.pt` ve `models/best_b.pt` yoluna yerleştirin, ardından aşağıdaki komutları çalıştırın:

```bash
docker build -t rumicim/road_safety:latest .
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
| Model ağırlıkları | `/app/models/best_a.pt`, `/app/models/best_b.pt` |

## Çıktı Formatı

Program, yarışma şablonuna uygun `results.json` dosyası üretir. JSON anahtarları ve etiket değerleri ASCII karakterli ve küçük harflidir.

## Sonuç

Çıktı dosyası: `output/results.json`

## Model A Eğitimi (arac_bilgisi)

Model A; araç tipi (7 sınıf) ve plaka ROI tespiti için YOLOv8 kullanır. Renk sınıflandırması ayrı bir adımda yapılabilir.

### Veri seti yapısı (YOLO format)

```
data/arac_bilgisi/
├── images/
│   ├── train/
│   ├── val/
│   └── test/        # opsiyonel
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml        # yoksa otomatik uretilir
```

Sınıflar: `hatchback`, `pickup`, `sedan`, `suv`, `minibus`, `panelvan`, `kamyon`, `plaka`

### Yerel eğitim

```bash
pip install -r requirements.txt
python train.py --config configs/model_a_config_local.yaml
```

Hiperparametre denemeleri için `configs/model_a_config.yaml` dosyasını kopyalayıp (`model_a_config_exp2.yaml` gibi) değerleri değiştirin.

### Docker ile eğitim

```bash
docker build -f Dockerfile.train -t road_safety_train:latest .
docker run --rm --gpus all \
  -v /path/to/arac_bilgisi:/data/arac_bilgisi \
  -v ./output:/output \
  -e TRAIN_CONFIG=/app/configs/model_a_config.yaml \
  road_safety_train:latest
```

Eğitim sonrası `output/model_a/experiment/weights/best.pt` dosyasını `models/best_a.pt` olarak kopyalayın.

### Docker Compose ile deneyler (Model A)

```bash
docker compose up baseline --build
docker compose up exp_high_lr exp_combined --build
docker compose logs -f baseline
```

### Yarışma çıktı standartları

`arac_bilgisi` alanları FTR dokümantasyonuna uygun olmalıdır:

| Alan | Değerler |
|------|----------|
| tip | sedan, suv, hatchback, pickup, minibus, panelvan, kamyon |
| plaka | Türkiye plaka regex (örn. `34ABC123`) |
| renk | beyaz, siyah, gri, kirmizi, mavi, sari, yesil, turuncu, kahverengi |
| confidence_score | 0.0 – 1.0 |

## Model B Eğitimi (tespitler)

Model B; sürücü eylemleri, kabin nesneleri ve yolcu tespiti için YOLOv8n kullanır. Çıktı JSON'daki `tespitler` alanına karşılık gelir.

### Veri seti yapısı (YOLO format)

```
data/input/tespitler/
├── images/train, val, test
├── labels/train, val, test
└── data.yaml
```

Sınıflar (FTR uyumlu 13 etiket):

| Kategori | Etiketler |
|----------|-----------|
| sofor_eylemi | arkaya_bakma, esneme, sigara_icme, su_icme, telefonla_konusma, slalom, etrafa_bakinma, emniyet_kemeri_ihlali |
| nesneler | teknocan, bilgisayar |
| yolcular | arka_koltuk_1, arka_koltuk_2, on_koltuk |

Roboflow'dan ek sınıf eklerseniz `classes` listesine aynı sırada ekleyin.

### Yerel eğitim

```bash
python train.py --config configs/model_b_config_local.yaml
```

### Docker ile eğitim

```bash
docker build -f Dockerfile.train -t road_safety_train:latest .
docker run --rm --gpus all \
  -v /path/to/tespitler:/data/tespitler \
  -v ./output:/output \
  -e TRAIN_CONFIG=/app/configs/model_b_config.yaml \
  road_safety_train:latest
```

Eğitim sonrası `output/model_b/experiment/weights/best.pt` dosyasını `models/best_b.pt` olarak kopyalayın.

Detayli mimari aciklama: `docs/PROJECT_EXPLANATION.md`
