"""
main.py — Cikarim Pipeline Giris Noktasi
=============================================================================
FTR sartnamesinde verilen ornek mimariye (main.py, src/predict.py,
src/utils.py) sadik kalinarak hazirlanmistir. Bu dosya, SADECE orkestrasyon
yapar: konfigurasyonu okur, cikarimi baslatir, ciktiyi diske yazar.
TUM is mantigi (model yukleme, OCR, renk tahmini, slalom) src/ altindaki
modullere DELEGE edilmistir (bkz. src/predict.py, src/ocr_utils.py,
src/color_utils.py, src/tracking.py, src/utils.py).

NEDEN bu dosyada DEGERLENDIRME ORTAMI TESPITI YOKTUR (FTR Madde E)?
    Asagidaki kodda hostname/IP/ortam degiskeni/dosya varligi kontrolune
    dayanan TEK BIR if/else veya try/except YOKTUR. Tek try/except blogu,
    cikarim SIRASINDA olusabilecek GERCEK calisma zamani hatalarini
    (corrupt video, eksik agirlik dosyasi vb.) yakalamak icindir — bu,
    standart hata yonetimi pratigidir, ortam tespiti DEGILDIR.
=============================================================================
"""

import os
import sys
import json
import traceback

import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import cikarimi_calistir
from src.utils import loglamayi_kur


def konfigurasyonu_yukle(config_yolu: str = "config.yaml") -> dict:
    with open(config_yolu, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    logger = loglamayi_kur()
    config = konfigurasyonu_yukle()

    girdi_video = config["yollar"]["girdi_video"]
    cikti_json = config["yollar"]["cikti_json"]

    logger.info("=" * 78)
    logger.info("TEKNOFEST 2026 — AKILLI YOL GUVENLIGI — CIKARIM (INFERENCE) PIPELINE'I")
    logger.info("=" * 78)
    logger.info(f"Girdi videosu      : {girdi_video}")
    logger.info(f"Cikti JSON         : {cikti_json}")
    logger.info(f"Model A agirligi   : {config['yollar']['model_a_agirlik']}")
    logger.info(f"Model B agirligi   : {config['yollar']['model_b_agirlik']}")
    logger.info("=" * 78)

    if not os.path.exists(girdi_video):
        logger.error(f"HATA: Girdi videosu bulunamadi -> {girdi_video}")
        sys.exit(1)

    try:
        cikti_verisi = cikarimi_calistir(girdi_video, config, logger)

        cikti_dizini = os.path.dirname(cikti_json)
        os.makedirs(cikti_dizini, exist_ok=True)

        # NEDEN ensure_ascii=True (varsayilan)?
        #   FTR Madde 3.6: "Turkce karakter kullanimindan KESINLIKLE
        #   kacinilmalidir." JSON degerleri ZATEN ascii_guvenli_kucuk_harf
        #   ile temizlenmis olsa da, ensure_ascii=True IKINCI bir guvenlik
        #   katmani olarak, olasi kacak (escape edilmemis) bir Unicode
        #   karakterin dosyaya yazilmasini ENGELLER.
        with open(cikti_json, "w", encoding="utf-8") as f:
            json.dump(cikti_verisi, f, ensure_ascii=True, indent=2)

        logger.info(f"Islem basarili! Cikti kaydedildi: {cikti_json}")
        logger.info("=" * 78)

    except Exception as e:
        logger.error(f"HATA: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
