"""
train.py — Model A (Arac Bilgisi) Egitim Scripti
=============================================================================
NEDEN bu script bu sekilde tasarlandi?

1) PARAMETRIK YAPI (config.yaml + CLI override)
   5 kisilik ekip aynı senaryoyu farkli hiperparametrelerle (batch, lr, optimizer,
   augmentation) paralel olarak deneyebilsin diye HICBIR hiperparametre kod
   icine sabit (hardcoded) yazilmamistir. Once config.yaml okunur, ardindan
   CLI argumanlari verilmisse bunlar config'in uzerine yazilir (override).
   Bu sayede her ekip uyesi kendi terminalinde:
       python train.py --batch_size 32 --lr 0.001 --optimizer AdamW --run_name deneme_ali
   gibi komutlarla, AYNI KOD ile FARKLI deneyler yurutebilir.

2) ZAMAN BUTCESI GUVENLIGI (FTR Madde D)
   Yarismada kesin ve esnemeyen bir 5 saatlik egitim suresi limiti vardir.
   Bu yuzden epoch sayisi yerine GERCEK GECEN SURE referans alinir. Egitim
   baslangic zamanindan itibaren 4 saat 45 dakika (guvenlik payi: 15 dk)
   gectiginde, ozel bir callback egitimi GUVENLI sekilde durdurur. Ultralytics
   kutuphanesi her epoch sonunda metrigi iyilesen agirligi otomatik olarak
   best.pt'ye yazdigi icin, bu noktaya kadar elde edilen EN IYI agirlik
   HER ZAMAN korunmus olur — ani kesintide veri kaybi riski yoktur.

3) EARLY STOPPING (patience=15) + AMP
   FTR sartinin acikca istedigi iki optimizasyon: sabit epoch sayisi yerine
   "patience=15" ile dogrulama metrigi iyilesmedigi surece egitim erken
   sonlandirilir; AMP (Automatic Mixed Precision) ile GPU bellegi ve egitim
   hizi FP16/FP32 karisik hassasiyetle optimize edilir.

4) DEGERLENDIRME ORTAMI MANIPULASYONU YOKTUR (FTR Madde E)
   Bu script icinde ortam degiskeni / hostname / IP / dosya varligi kontrolu
   ile "hangi ortamdayim?" tespiti yapan HICBIR if/else veya try/except yapisi
   YOKTUR. Script, lokal makinede de, Docker konteynerinde de, Colab'da da
   BIREBIR AYNI sekilde calisir ve davranir.
=============================================================================
"""

import os
import sys
import time
import argparse
import copy

import yaml
from loguru import logger
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# Ozel istisna sinifi: zaman butcesi asildiginda firlatilir.
# NEDEN ozel bir Exception sinifi?
#   Genel bir "except Exception" yerine ozel bir sinif kullanmak, zaman butcesi
#   asimini DIGER olasi hatalardan (veri bozuklugu, CUDA hatasi vb.) AYIRT
#   edebilmemizi saglar. Bu, "ortam tespiti" amacli bir kontrol DEGILDIR;
#   sadece sure asimini, programatik hatalardan ayrıstıran standart bir
#   yazilim mühendisligi pratigidir.
# -----------------------------------------------------------------------------
class ZamanButcesiAsildiHatasi(Exception):
    """Egitim suresi, FTR'de tanimlanan guvenlik esigini astiginda firlatilir."""
    pass


def konfigurasyonu_yukle(config_yolu: str) -> dict:
    """config.yaml dosyasini okuyup Python sozlugune cevirir.

    NEDEN ayri bir fonksiyon?
        Konfigurasyon okuma mantigi, test edilebilirlik ve tek-sorumluluk
        ilkesi (Single Responsibility Principle) geregi ana egitim akisindan
        ayristirilmistir.
    """
    if not os.path.exists(config_yolu):
        raise FileNotFoundError(f"Konfigurasyon dosyasi bulunamadi: {config_yolu}")

    with open(config_yolu, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cli_argumanlarini_tanimla() -> argparse.Namespace:
    """Ekip uyelerinin hiperparametre denemeleri icin CLI override argumanlarini tanimlar.

    NEDEN default=None?
        default=None birakilarak, kullanici bir parametreyi ELLE VERMEDIGI
        surece config.yaml'daki degerin DEGISTIRILMEDEN kullanilmasi saglanir.
        Bu, "kismi override" (sadece batch_size'i degistirip digerlerini
        config'den birakma) senaryosunu dogal olarak destekler.
    """
    parser = argparse.ArgumentParser(
        description="Model A (Arac Bilgisi) YOLO Egitim Scripti — Teknofest 2026"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Konfigurasyon dosyasi yolu")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None,
                         help="Learning rate (config.train.learning_rate'i override eder)")
    parser.add_argument("--optimizer", type=str, default=None, choices=["SGD", "AdamW"])
    parser.add_argument("--epochs", type=int, default=None,
                         help="Ust sinir epoch sayisi (gercek durdurma early_stopping/zaman ile olur)")
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)
    parser.add_argument("--run_name", type=str, default=None,
                         help="Bu denemeye ozel benzersiz isim (runs/model_a/<run_name>)")
    parser.add_argument("--device", type=str, default=None,
                         help="GPU index ('0') veya 'cpu'")
    return parser.parse_args()


def konfigurasyona_cli_override_uygula(config: dict, args: argparse.Namespace) -> dict:
    """CLI'dan gelen ve None olmayan her degeri config sozlugune islerler.

    NEDEN deepcopy?
        Orijinal config nesnesi (loglama/raporlama icin) bozulmadan, override
        edilmis yeni bir kopya uzerinde calismak; "neyin degistirildigi"ni
        net bicimde izleyebilmek icindir.
    """
    yeni_config = copy.deepcopy(config)

    if args.batch_size is not None:
        yeni_config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        yeni_config["train"]["learning_rate"] = args.lr
    if args.optimizer is not None:
        yeni_config["train"]["optimizer"] = args.optimizer
    if args.epochs is not None:
        yeni_config["train"]["epochs"] = args.epochs
    if args.mosaic is not None:
        yeni_config["augmentation"]["mosaic"] = args.mosaic
    if args.mixup is not None:
        yeni_config["augmentation"]["mixup"] = args.mixup
    if args.run_name is not None:
        yeni_config["output"]["run_name"] = args.run_name
    if args.device is not None:
        yeni_config["train"]["device"] = args.device

    return yeni_config


def zaman_kontrol_callback_olustur(egitim_baslangic_zamani: float, max_sure_sn: float):
    """Ultralytics 'on_train_epoch_end' olayina baglanacak callback'i uretir.

    NEDEN closure (ic fonksiyon) deseni?
        egitim_baslangic_zamani ve max_sure_sn degerlerini global degisken
        kullanmadan, callback'in kendi kapsamina (closure) tasimak; fonksiyonu
        yan etkisiz (side-effect-free) ve test edilebilir kilar.
    """

    def _callback(trainer):
        gecen_sure_sn = time.time() - egitim_baslangic_zamani
        kalan_sn = max_sure_sn - gecen_sure_sn
        logger.info(
            f"[Zaman Kontrolu] Epoch {trainer.epoch + 1} tamamlandi | "
            f"Gecen sure: {gecen_sure_sn/60:.1f} dk | Kalan butce: {kalan_sn/60:.1f} dk"
        )
        if gecen_sure_sn >= max_sure_sn:
            raise ZamanButcesiAsildiHatasi(
                f"FTR zaman butcesi ({max_sure_sn/3600:.2f} saat) asildi. "
                f"Egitim, o ana kadar kaydedilmis best.pt korunarak guvenli sekilde durduruluyor."
            )

    return _callback


def egitimi_baslat(config: dict) -> None:
    """Asil egitim akisini yurutur: model yukleme, callback baglama, train() cagirma."""

    egitim_baslangic_zamani = time.time()
    max_sure_sn = config["time_budget"]["max_seconds"]

    logger.info("=" * 78)
    logger.info("MODEL A (ARAC BILGISI) — YOLO EGITIMI BASLIYOR")
    logger.info("=" * 78)
    logger.info(f"Taban model       : {config['model']['architecture']}")
    logger.info(f"Batch size        : {config['train']['batch_size']}")
    logger.info(f"Learning rate     : {config['train']['learning_rate']}")
    logger.info(f"Optimizer         : {config['train']['optimizer']}")
    logger.info(f"AMP aktif mi      : {config['train']['amp']}")
    logger.info(f"Early stop patience: {config['train']['patience']}")
    logger.info(f"Zaman butcesi     : {max_sure_sn/3600:.2f} saat")
    logger.info("=" * 78)

    # NEDEN onceden egitilmis (pretrained) agirlik?
    #   Roboflow setindeki ~7500 gorsel, sifirdan (from-scratch) egitim icin
    #   sinirlidir. COCO uzerinde onceden egitilmis genel nesne tespiti
    #   ozellikleri (kenarlar, doku, govde formlari) transfer ogrenme (transfer
    #   learning) ile aktarilarak, 5 saatlik kisitli surede daha hizli ve
    #   kararli bir yakinsama saglanir.
    model = YOLO(config["model"]["architecture"])

    zaman_callback = zaman_kontrol_callback_olustur(egitim_baslangic_zamani, max_sure_sn)
    model.add_callback("on_train_epoch_end", zaman_callback)

    try:
        model.train(
            data=config["dataset"]["data_yaml"],
            epochs=config["train"]["epochs"],          # ust sinir; gercek durdurma asagidaki mekanizmalarla
            patience=config["train"]["patience"],       # FTR sarti: early stopping
            batch=config["train"]["batch_size"],
            lr0=config["train"]["learning_rate"],
            optimizer=config["train"]["optimizer"],
            amp=config["train"]["amp"],                 # FTR sarti: Automatic Mixed Precision
            imgsz=config["model"]["img_size"],
            workers=config["train"]["workers"],
            device=config["train"]["device"],
            seed=config["train"]["seed"],
            # --- Veri artirma (augmentation) parametreleri ---
            mosaic=config["augmentation"]["mosaic"],
            mixup=config["augmentation"]["mixup"],
            hsv_h=config["augmentation"]["hsv_h"],
            hsv_s=config["augmentation"]["hsv_s"],
            hsv_v=config["augmentation"]["hsv_v"],
            degrees=config["augmentation"]["degrees"],
            translate=config["augmentation"]["translate"],
            scale=config["augmentation"]["scale"],
            fliplr=config["augmentation"]["fliplr"],
            # --- Cikti / loglama ---
            project=config["output"]["project_dir"],
            name=config["output"]["run_name"],
            exist_ok=True,
            verbose=True,
        )
        logger.info("Egitim, early_stopping veya max epoch ile DOGAL olarak tamamlandi.")

    except ZamanButcesiAsildiHatasi as e:
        # NEDEN bu noktada exit(1) DEGIL, normal akista devam?
        #   Zaman butcesi asimi bir HATA degil, BEKLENEN ve TASARLANMIS bir
        #   durumdur. best.pt zaten diskte guvende oldugu icin programi
        #   basarisiz (exit code != 0) olarak sonlandirmak YANLIS olur.
        logger.warning(str(e))

    toplam_sure_dk = (time.time() - egitim_baslangic_zamani) / 60
    en_iyi_agirlik_yolu = os.path.join(
        config["output"]["project_dir"], config["output"]["run_name"], "weights", "best.pt"
    )
    logger.info("=" * 78)
    logger.info(f"EGITIM SONLANDI — Toplam sure: {toplam_sure_dk:.1f} dakika")
    logger.info(f"En iyi agirlik (best.pt) yolu: {en_iyi_agirlik_yolu}")
    logger.info("=" * 78)


def main():
    logger.remove()
    logger.add(sys.stdout, format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | {level: <8} | {message}", level="INFO")

    args = cli_argumanlarini_tanimla()
    taban_config = konfigurasyonu_yukle(args.config)
    calisma_config = konfigurasyona_cli_override_uygula(taban_config, args)

    egitimi_baslat(calisma_config)


if __name__ == "__main__":
    main()
