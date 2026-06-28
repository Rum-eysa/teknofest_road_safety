# Model A — Arac Bilgisi Egitim Ortami (Docker)

YOLO tabanli, **Roboflow** uzerinden etiketlenmis ~7.500 gorsellik "arac_bilgisi"
veri setiyle (7 arac tipi + plaka bolgesi = 8 sinif) egitim yapan, **parametrik**
ve **tekrarlanabilir** Docker ortami.

## Klasor Yapisi
```
model_a_egitim/
├── Dockerfile
├── docker-compose.yml
├── config.yaml          # Tum hiperparametreler buradan okunur
├── requirements.txt
├── train.py              # Parametrik egitim scripti
└── data/
    └── data.yaml          # Roboflow export sablonu (kendi export'unuzla guncelleyin)
```

## Hizli Baslangic
```bash
# 1) Roboflow'dan indirdiginiz veri setini ./datasets/arac_bilgisi altina yerlestirin
# 2) data/data.yaml icindeki yollari kendi export klasor adlarinizla eslestirin
# 3) Imaji olusturun ve varsayilan config ile egitimi baslatin
docker compose build
docker compose run --rm train
```

## Ekip Icin Farkli Hiperparametre Denemeleri
Her ekip uyesi, AYNI imaji farkli komut satiri argumanlariyla calistirarak
kendi denemesini yurutebilir (config.yaml degismeden, sadece o calismaya
ozel override edilir):

```bash
docker compose run --rm train --batch_size 32 --lr 0.001 --optimizer AdamW --run_name deneme_ali
docker compose run --rm train --batch_size 8  --mosaic 0.5 --mixup 0.0 --run_name deneme_zeynep
docker compose run --rm train --epochs 150 --optimizer SGD --lr 0.02 --run_name deneme_mehmet
```

Sonuclar `./runs/model_a/<run_name>/weights/best.pt` altinda biriken,
KARSILASTIRILABILIR ve ISIMLENDIRILMIS klasorlerde tutulur.

## Onemli Tasarim Kararlari (Neden?)
| Karar | Neden |
|---|---|
| `patience=15` (early stopping) | FTR sarti; gereksiz epoch'larla zaman butcesini tuketmemek |
| `amp: true` | FTR sarti; FP16/FP32 karisik hassasiyetle hiz/VRAM optimizasyonu |
| `time_budget.max_seconds=17100` | 5 saatlik FTR siniri - 15 dk guvenlik payi = 4sa45dk |
| Ortam tespiti YOK | FTR Madde E; kod her ortamda BIREBIR ayni davranir |
| Veri seti volume olarak mount | Imaj boyutu kucuk kalsin, veri guncellemesi rebuild gerektirmesin |
