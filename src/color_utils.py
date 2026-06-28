"""
src/color_utils.py — HSV Renk Uzayi Analizi ile Arac Rengi Tahmini
=============================================================================
FTR Madde B: "Renk tahmini, kirpilan arac goruntusu uzerinden HSV renk uzayi
analizi VEYA hafif bir CNN sinifllandirma kafasi ile yapilacaktir."

NEDEN HSV (RGB degil)?
    RGB uzayinda renk benzerligi, parlaklik (isik siddeti) degisimlerinden
    COK fazla etkilenir (orn. golgedeki beyaz bir arac, RGB'de griye yakin
    gorunebilir). HSV ayriminda Hue (renk tonu) bilesini, aydinlatmadan
    BAGIMSIZ kalir; bu da gun/golge/farkli kamera senaryolarinda daha
    KARARLI bir siniflandirma saglar.

NEDEN hafif CNN DEGIL, HSV histogram tercih edildi?
    5 saatlik egitim butcesi, ZATEN iki ana model (A ve B) icin paylasilmis
    durumdadir. Renk tahmini icin UCUNCU bir model egitmek, butceyi gereksiz
    yere boler. HSV tabanli kural-temelli (rule-based) yaklasim, EGITIM
    GEREKTIRMEDIGI icin zaman butcesi disinda kalir ve 9 sabit renk sinifi
    icin yeterli ayirt edicilige sahiptir.
=============================================================================
"""

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# 9 renk icin HSV esik araliklari.
# NEDEN H, S, V ayrimi bu sekilde yapildi?
#   Beyaz/siyah/gri (akromatik renkler) Hue'dan BAGIMSIZDIR; bunlar SADECE
#   Saturation (doygunluk) ve Value (parlaklik) ile ayirt edilir. Kromatik
#   renkler (kirmizi, mavi, sari, yesil, turuncu, kahverengi) ise ONCELIKLE
#   Hue acisina gore siniflandirilir. Bu iki kademeli mantik, OpenCV'nin
#   standart HSV temsiline (H: 0-179, S/V: 0-255) dayanir.
# -----------------------------------------------------------------------------
_AKROMATIK_DOYGUNLUK_ESIGI = 40   # Bu degerin ALTINDAKI doygunluk -> akromatik (beyaz/siyah/gri)
_SIYAH_PARLAKLIK_ESIGI = 60       # Akromatik VE bu degerin ALTINDA parlaklik -> siyah
_BEYAZ_PARLAKLIK_ESIGI = 180      # Akromatik VE bu degerin UZERINDE parlaklik -> beyaz
                                   # Arada kalan akromatik bolge -> gri

# Hue (0-179) araliklari -> kromatik renk adi. Kirmizi, HSV cember
# baslangic/bitisinde (0 ve 179 civari) iki parcali oldugu icin AYRI
# kontrol edilir.
_KROMATIK_HUE_ARALIKLARI = [
    ("kirmizi", (0, 8)),
    ("turuncu", (9, 20)),
    ("sari", (21, 35)),
    ("yesil", (36, 85)),
    ("mavi", (86, 130)),
    # 131-159 araligi -> mor/pembe tonlari; FTR'nin 9 renginde KARSILIGI
    # olmadigi icin en yakin kategori olan "kirmizi"ye (manyenta-kirmizi
    # gecisi) atfedilir; kahverengi asagida doygunluk/parlaklik ile ayrica
    # ele alinir.
    ("kirmizi", (160, 179)),
]


def _kirpilmis_goruntuyu_hazirla(arac_crop_bgr: np.ndarray) -> np.ndarray:
    """Kenar/golge gurultusunu azaltmak icin goruntunun MERKEZ bolgesini alir.

    NEDEN merkez crop?
        Bounding box kenarlarinda arac govdesi DISINDAKI piksel (yol, gokyuzu,
        baska arac parcasi) sizmasi olasidir. Goruntunun orta %60'lik
        bolgesini almak, bu "kenar gurultusunu" buyuk olcude eler.
    """
    yukseklik, genislik = arac_crop_bgr.shape[:2]
    y0, y1 = int(yukseklik * 0.2), int(yukseklik * 0.8)
    x0, x1 = int(genislik * 0.2), int(genislik * 0.8)
    merkez = arac_crop_bgr[y0:y1, x0:x1]
    return merkez if merkez.size > 0 else arac_crop_bgr


def arac_rengini_tahmin_et(arac_crop_bgr: np.ndarray) -> tuple:
    """Kirpilan arac goruntusunden HSV histogram analiziyle renk ve guven skoru dondurur.

    Donus: (renk_adi: str, guven_skoru: float [0.0-1.0])
    """
    if arac_crop_bgr is None or arac_crop_bgr.size == 0:
        return "", 0.0

    merkez_bolge = _kirpilmis_goruntuyu_hazirla(arac_crop_bgr)
    hsv = cv2.cvtColor(merkez_bolge, cv2.COLOR_BGR2HSV)

    h_kanali = hsv[:, :, 0].astype(np.float32)
    s_kanali = hsv[:, :, 1].astype(np.float32)
    v_kanali = hsv[:, :, 2].astype(np.float32)

    ortalama_s = float(np.median(s_kanali))   # NEDEN medyan (ortalama DEGIL)?
    ortalama_v = float(np.median(v_kanali))   # Cam/parlak yansima gibi UC DEGERLERE (outlier) karsi daha dayanikli

    # --- Adim 1: Akromatik mi (beyaz/siyah/gri), kromatik mi? ---
    if ortalama_s < _AKROMATIK_DOYGUNLUK_ESIGI:
        if ortalama_v < _SIYAH_PARLAKLIK_ESIGI:
            tahmini_renk = "siyah"
        elif ortalama_v > _BEYAZ_PARLAKLIK_ESIGI:
            tahmini_renk = "beyaz"
        else:
            tahmini_renk = "gri"

        # Guven skoru: esik degerinden ne kadar UZAKTA oldugumuza bagli
        # (esige cok yakin pikseller -> dusuk guven).
        esik_uzakligi = min(abs(ortalama_s - _AKROMATIK_DOYGUNLUK_ESIGI), 40) / 40
        guven = float(np.clip(0.55 + 0.4 * esik_uzakligi, 0.0, 0.95))
        return tahmini_renk, guven

    # --- Adim 2: Kromatik renk -> Hue histogramindaki EN SIK deger (mod) ---
    histogram, _ = np.histogram(h_kanali, bins=180, range=(0, 180))
    en_sik_hue = int(np.argmax(histogram))

    # NEDEN kahverengi burada ozel ele alinir?
    #   Kahverengi, HSV'de TURUNCU ile AYNI Hue araliginda yer alir; onu
    #   turuncudan ayiran TEMEL fark DUSUK parlaklik VE orta-dusuk doygunluktur.
    if 5 <= en_sik_hue <= 25 and ortalama_v < 120:
        return "kahverengi", 0.65

    tahmini_renk = ""
    for renk_adi, (alt, ust) in _KROMATIK_HUE_ARALIKLARI:
        if alt <= en_sik_hue <= ust:
            tahmini_renk = renk_adi
            break

    if not tahmini_renk:
        return "", 0.0

    # Guven skoru: histogramda en sik Hue degerinin TOPLAM piksellere
    # oranina (baskinlik) dayanir -> ne kadar baskin, o kadar guvenilir.
    toplam_piksel = float(histogram.sum()) or 1.0
    baskinlik_orani = float(histogram[en_sik_hue]) / toplam_piksel
    guven = float(np.clip(0.5 + baskinlik_orani * 5, 0.0, 0.95))

    return tahmini_renk, guven
