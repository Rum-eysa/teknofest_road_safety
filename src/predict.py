"""
src/predict.py — Cikarim (Inference) Orkestrasyonu
=============================================================================
Bu modul, FTR Madde 3 (CIKTI 3) altinda istenen tum cikarim akisini
yonetir: video okuma, Model A (Arac Bilgisi) ve Model B (Yol Guvenligi)
cikarimlarini BIRLIKTE calistirma, plaka OCR'i, renk tahmini, slalom
tespiti ve son olarak konsolide JSON semasinin olusturulmasi.

NEDEN bu kadar fonksiyon TEK dosyada degil, kucuk parcalara bolundu?
    Her bir alt-gorev (OCR, renk, takip) kendi modulunde (ocr_utils,
    color_utils, tracking) yer alir; bu dosya SADECE bu modulleri
    ORKESTRE EDER. Bu ayrim, yarisma komitesinin her bir bilesimi BAGIMSIZ
    olarak inceleyebilmesini ve ekip uyelerinin farkli modullerde AYNI ANDA
    (cakismadan) calisabilmesini saglar.
=============================================================================
"""

import os
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import cv2

from src.utils import video_bilgisini_oku, cikti_semasini_olustur
from src.color_utils import arac_rengini_tahmin_et
from src.ocr_utils import plaka_bolgesini_oku
from src.tracking import SlalomDedektoru


def modelleri_yukle(model_a_yolu: str, model_b_yolu: str, logger):
    """Model A (Arac Bilgisi) ve Model B (Yol Guvenligi) agirliklarini yukler.

    NEDEN her iki model BASLANGICTA (video isleme dongusunden ONCE) yuklenir?
        Model yukleme (agirlik dosyasini diskten okuma + GPU'ya tasima),
        her kare icin TEKRARLANIRSA performansi feci sekilde dusurur. Modeller
        BIR KEZ yuklenip, video boyunca TEKRAR KULLANILIR (bellekte tutulur).
    """
    from ultralytics import YOLO

    if not os.path.exists(model_a_yolu):
        raise FileNotFoundError(f"Model A agirligi bulunamadi: {model_a_yolu}")
    if not os.path.exists(model_b_yolu):
        raise FileNotFoundError(f"Model B agirligi bulunamadi: {model_b_yolu}")

    logger.info(f"Model A yukleniyor (Arac Bilgisi): {model_a_yolu}")
    model_a = YOLO(model_a_yolu)

    logger.info(f"Model B yukleniyor (Yol Guvenligi): {model_b_yolu}")
    model_b = YOLO(model_b_yolu)

    return model_a, model_b


def _karede_model_calistir(model, kare, guven_esigi: float):
    """Tek bir YOLO modelini tek bir karede calistirir ve ham sonucu doner.

    NEDEN ayri, kucuk bir fonksiyon?
        ThreadPoolExecutor.submit() bu fonksiyonu Model A ve Model B icin
        PARALEL olarak cagirir (bkz. _frame_cikarimi_yap). Fonksiyonun
        yalitilmis (isolated) olmasi, thread-safety acisindan ONEMLIDIR:
        her cagri kendi yerel degiskenleriyle calisir, PAYLASILAN durum
        (shared state) YOKTUR.
    """
    sonuc = model.predict(kare, conf=guven_esigi, verbose=False)
    return sonuc[0] if sonuc else None


def _frame_cikarimi_yap(model_a, model_b, kare, config: dict) -> tuple:
    """Model A VE Model B'yi AYNI kare uzerinde ESZAMANLI (concurrent) calistirir.

    NEDEN ThreadPoolExecutor (multiprocessing DEGIL)?
        Her iki model de AYNI GPU belleginde (CUDA context) yasar; ayri
        PROCESS'lere bolmek (multiprocessing) modelleri HER PROCESS'TE
        TEKRAR yuklemeyi gerektirir ve GPU bellek cakismasi riski yaratir.
        Thread tabanli yaklasimda ise PyTorch'un CUDA kernel cagrilari
        GIL'i (Global Interpreter Lock) BIRAKTIGI icin, iki modelin GPU
        uzerindeki hesaplamalari GERCEKTEN orusebilir (overlap); bu da
        Madde 3.3'teki "asenkron/ardisik calistirma" sartini, ek surec
        karmasikligi olmadan karsilar.

    NEDEN bu fonksiyon hata durumunda istisna FIRLATIYOR (yutmuyor)?
        Bir karede cikarim hatasi, sessizce yutulup video isleme devam
        ederse, yanlislikla EKSIK/HATALI bir JSON ciktisi uretilebilir.
        Hata, main.py'deki TEK merkezi try/except blogunda ele alinir.
    """
    with ThreadPoolExecutor(max_workers=2) as havuz:
        gelecek_a = havuz.submit(
            _karede_model_calistir, model_a, kare, config["cikarim"]["guven_esigi_model_a"]
        )
        gelecek_b = havuz.submit(
            _karede_model_calistir, model_b, kare, config["cikarim"]["guven_esigi_model_b"]
        )
        sonuc_a = gelecek_a.result()
        sonuc_b = gelecek_b.result()

    return sonuc_a, sonuc_b


def _model_a_sonucunu_isle(sonuc_a, kare, config: dict) -> dict:
    """Model A ciktisindan EN YUKSEK guvenli arac kutusunu, plaka crop'unu ve renk tahminini cikarir.

    NEDEN "EN YUKSEK guvenli TEK arac" varsayimi yapildi?
        Yarisma senaryosu (slalom parkuru, plaka okuma istasyonu), TEK bir
        TEST aracinin degerlendirildigi bir kurguya dayanir. Sahnede
        BIRDEN FAZLA arac olsa da, FTR ciktisi TEK bir 'arac_bilgisi'
        alani ister; bu yuzden HER karede EN YUKSEK guven skoruna sahip
        arac, "ilgilenilen arac" (ego-vehicle) olarak secilir. Coklu-arac
        takibi (multi-object tracking + ID atama) gerektiren bir senaryoya
        genisletmek icin, bu fonksiyon ve SlalomDedektoru'nun arac ID'sine
        gore COKLU ornek tutacak sekilde uyarlanmasi yeterlidir.
    """
    bos_sonuc = {"arac_kutusu": None, "arac_tipi": "", "arac_tipi_guven": 0.0,
                 "merkez_x": None, "plaka_metni": "", "plaka_guven": 0.0,
                 "renk": "", "renk_guven": 0.0}

    if sonuc_a is None or sonuc_a.boxes is None or len(sonuc_a.boxes) == 0:
        return bos_sonuc

    sinif_isimleri = sonuc_a.names
    plaka_sinif_adi = config["cikarim"]["plaka_sinif_adi"]

    en_iyi_arac_kutusu = None
    en_iyi_arac_guven = -1.0
    plaka_kutusu = None
    plaka_guven_skoru = 0.0

    for kutu in sonuc_a.boxes:
        sinif_id = int(kutu.cls[0])
        sinif_adi = sinif_isimleri.get(sinif_id, "")
        guven = float(kutu.conf[0])

        if sinif_adi == plaka_sinif_adi:
            if guven > plaka_guven_skoru:
                plaka_kutusu = kutu
                plaka_guven_skoru = guven
        else:
            if guven > en_iyi_arac_guven:
                en_iyi_arac_kutusu = kutu
                en_iyi_arac_guven = guven

    sonuc = dict(bos_sonuc)

    if en_iyi_arac_kutusu is not None:
        x1, y1, x2, y2 = map(int, en_iyi_arac_kutusu.xyxy[0])
        arac_crop = kare[max(0, y1):y2, max(0, x1):x2]

        sonuc["arac_tipi"] = sinif_isimleri.get(int(en_iyi_arac_kutusu.cls[0]), "")
        sonuc["arac_tipi_guven"] = en_iyi_arac_guven
        sonuc["merkez_x"] = (x1 + x2) / 2.0

        renk, renk_guven = arac_rengini_tahmin_et(arac_crop)
        sonuc["renk"] = renk
        sonuc["renk_guven"] = renk_guven

    if plaka_kutusu is not None and plaka_guven_skoru >= config["cikarim"]["plaka_ocr_min_guven"]:
        px1, py1, px2, py2 = map(int, plaka_kutusu.xyxy[0])
        plaka_crop = kare[max(0, py1):py2, max(0, px1):px2]
        plaka_metni, ocr_guven = plaka_bolgesini_oku(plaka_crop)
        sonuc["plaka_metni"] = plaka_metni
        # NEDEN tespit guveni VE OCR guveni CARPILIYOR (toplanmiyor)?
        #   Plaka bilgisinin GUVENILIR sayilmasi icin HEM bolgenin DOGRU
        #   tespit edilmis OLMASI HEM DE metnin DOGRU okunmus OLMASI
        #   gerekir; bu iki BAGIMSIZ olasiligin BIRLESIK guveni, CARPIM
        #   ile (olasilik teorisindeki bagimsiz olaylar gibi) ifade edilir.
        sonuc["plaka_guven"] = round(plaka_guven_skoru * ocr_guven, 4)

    return sonuc


def _model_b_sonucunu_isle(sonuc_b, zaman_saniye: float, config: dict) -> list:
    """Model B ciktisindaki HER tespiti, FTR'nin tespit semasina (kategori/etiket) esler."""
    tespitler = []
    if sonuc_b is None or sonuc_b.boxes is None or len(sonuc_b.boxes) == 0:
        return tespitler

    sinif_isimleri = sonuc_b.names
    siniflar = config["siniflar"]

    for kutu in sonuc_b.boxes:
        sinif_adi = sinif_isimleri.get(int(kutu.cls[0]), "")
        guven = float(kutu.conf[0])

        if sinif_adi in siniflar["sofor_eylemi"]:
            kategori = "sofor_eylemi"
        elif sinif_adi in siniflar["nesneler"]:
            kategori = "nesneler"
        elif sinif_adi in siniflar["yolcular"]:
            kategori = "yolcular"
        else:
            continue  # Taninmayan/ilgisiz sinif -> atla

        tespitler.append({
            "zaman_saniye": zaman_saniye,
            "kategori": kategori,
            "etiket": sinif_adi,
            "confidence_score": guven,
        })

    return tespitler


def cikarimi_calistir(video_yolu: str, config: dict, logger) -> dict:
    """Tum video uzerinde kare-kare cikarim yapar ve FTR semasina uygun sonucu doner."""
    video_bilgisi = video_bilgisini_oku(video_yolu)
    fps = video_bilgisi["fps"]
    logger.info(f"   - FPS: {fps:.2f} | Cozunurluk: {video_bilgisi['genislik']}x{video_bilgisi['yukseklik']} "
                f"| Kare sayisi: {video_bilgisi['kare_sayisi']}")

    model_a, model_b = modelleri_yukle(
        config["yollar"]["model_a_agirlik"], config["yollar"]["model_b_agirlik"], logger
    )

    slalom_dedektoru = SlalomDedektoru(
        pencere_kare_sayisi=config["slalom"]["pencere_kare_sayisi"],
        min_yon_degisimi=config["slalom"]["min_yon_degisimi"],
        min_genlik_piksel=config["slalom"]["min_genlik_piksel"],
    )
    slalom_devam_ediyor = False  # Debounce: ayni slalom hareketini TEKRAR TEKRAR raporlamamak icin

    # Video boyunca biriken istatistikler (tum video icin TEK bir arac_bilgisi uretmek uzere)
    arac_tipi_sayaci = Counter()
    renk_sayaci = Counter()
    plaka_sayaci = Counter()
    arac_guven_listesi = []
    plaka_guven_haritasi = {}  # plaka_metni -> en yuksek guven

    tum_tespitler = []
    kare_atlama = max(1, config["cikarim"]["kare_atlama"])

    cap = cv2.VideoCapture(video_yolu)
    if not cap.isOpened():
        raise ValueError(f"Video acilamadi: {video_yolu}")

    kare_indeksi = 0
    while True:
        basarili, kare = cap.read()
        if not basarili:
            break

        if kare_indeksi % kare_atlama != 0:
            kare_indeksi += 1
            continue

        # FTR Madde 3.2: zaman_saniye, FPS uzerinden float olarak hesaplanir.
        zaman_saniye = kare_indeksi / fps if fps > 0 else 0.0

        sonuc_a, sonuc_b = _frame_cikarimi_yap(model_a, model_b, kare, config)

        # --- Model A: arac tipi / renk / plaka ---
        a_ozet = _model_a_sonucunu_isle(sonuc_a, kare, config)
        if a_ozet["arac_tipi"]:
            arac_tipi_sayaci[a_ozet["arac_tipi"]] += 1
            arac_guven_listesi.append(a_ozet["arac_tipi_guven"])
        if a_ozet["renk"]:
            renk_sayaci[a_ozet["renk"]] += 1
            arac_guven_listesi.append(a_ozet["renk_guven"])
        if a_ozet["plaka_metni"]:
            plaka_sayaci[a_ozet["plaka_metni"]] += 1
            mevcut_en_iyi = plaka_guven_haritasi.get(a_ozet["plaka_metni"], 0.0)
            plaka_guven_haritasi[a_ozet["plaka_metni"]] = max(mevcut_en_iyi, a_ozet["plaka_guven"])

        # --- Slalom: arac merkezinin X eksenindeki zamansal salinimi ---
        if a_ozet["merkez_x"] is not None:
            slalom_su_an_var = slalom_dedektoru.guncelle(zaman_saniye, a_ozet["merkez_x"])
            if slalom_su_an_var and not slalom_devam_ediyor:
                # FTR Madde C: slalom, sabit veri seti OLMADAN, trajektori
                # analiziyle uretilen DINAMIK bir tespittir; bu yuzden
                # "arac_hareketi" kategorisinde, diger model tabanli
                # tespitlerle AYNI semaya uyacak sekilde raporlanir.
                tum_tespitler.append({
                    "zaman_saniye": zaman_saniye,
                    "kategori": "arac_hareketi",
                    "etiket": "slalom",
                    # NEDEN sabit 0.75 guven?
                    #   Bu bir SINIFLANDIRMA modeli ciktisi olmadigi icin
                    #   olasiliksal bir guven skoru DOGAL OLARAK YOKTUR;
                    #   heuristigin esik kosullarini (genlik + yon degisim
                    #   sayisi) ayni anda KARSILAMASI, ORTA-YUKSEK sabit bir
                    #   guven degeriyle temsil edilir.
                    "confidence_score": 0.75,
                })
            slalom_devam_ediyor = slalom_su_an_var

        # --- Model B: surucu davranisi / nesneler / yolcular ---
        tum_tespitler.extend(_model_b_sonucunu_isle(sonuc_b, zaman_saniye, config))

        if kare_indeksi % 100 == 0:
            logger.info(f"   - Islenen kare: {kare_indeksi}/{video_bilgisi['kare_sayisi']}")

        kare_indeksi += 1

    cap.release()
    logger.info("Video analiz tamamlandi, sonuclar konsolide ediliyor.")

    # --- Video boyunca biriken istatistiklerden TEK bir arac_bilgisi cikar ---
    en_sik_arac_tipi = arac_tipi_sayaci.most_common(1)[0][0] if arac_tipi_sayaci else ""
    en_sik_renk = renk_sayaci.most_common(1)[0][0] if renk_sayaci else ""
    en_sik_plaka = plaka_sayaci.most_common(1)[0][0] if plaka_sayaci else ""
    plaka_guven = plaka_guven_haritasi.get(en_sik_plaka, 0.0)

    # FTR Madde 3.5: "Arac ozellikleri icin TEK BIR ortak confidence."
    # Bu, video boyunca toplanan TUM arac-ozelligi guven degerlerinin
    # (tip + renk, + varsa plaka) ORTALAMASI olarak hesaplanir.
    tum_guvenler = list(arac_guven_listesi)
    if plaka_guven > 0:
        tum_guvenler.append(plaka_guven)
    ortak_arac_guveni = sum(tum_guvenler) / len(tum_guvenler) if tum_guvenler else 0.0

    arac_bilgisi = {
        "tip": en_sik_arac_tipi,
        "plaka": en_sik_plaka,
        "renk": en_sik_renk,
        "confidence_score": ortak_arac_guveni,
    }

    video_dosya_adi = os.path.basename(video_yolu)
    return cikti_semasini_olustur(video_dosya_adi, arac_bilgisi, tum_tespitler)
