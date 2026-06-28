"""
src/ocr_utils.py — Plaka OCR ve Regex Normalizasyonu
=============================================================================
FTR Madde B: Model A tarafindan tespit edilen plaka bolgesi crop edilip
EasyOCR/PaddleOCR ile metne donusturulur; cikti, sartnamede verilen regex'e
uygun olacak sekilde BOSLUKLAR TEMIZLENEREK normalize edilir.

NEDEN EasyOCR (PaddleOCR yerine varsayilan secim)?
    EasyOCR, PyTorch tabanli oldugu icin Model A/B'nin ZATEN kullandigi
    PyTorch/Ultralytics ortamiyla AYNI bagimlilik agacini paylasir; bu da
    Docker imajinda EK bir derin ogrenme framework'u (PaddlePaddle) kurma
    ihtiyacini ORTADAN KALDIRIR ve imaj boyutunu kucuk tutar. PaddleOCR'a
    gecis istenirse, bu modulun PUBLIC arayuzu (oku_plaka) AYNI kalacagi
    icin cagiran kod (predict.py) DEGISMEDEN calismaya devam eder.
=============================================================================
"""

import re
import numpy as np

# -----------------------------------------------------------------------------
# Turkiye plaka regex'i — FTR'de verilen orijinal ifadenin, BOSLUKLARI
# ONCEDEN TEMIZLENMIS bir girdi uzerinde calisacak sekilde SADELESTIRILMIS
# halidir. Orijinal regex, OCR ciktisindaki olasi bosluklari (orn. "34 ABC 123")
# kapsayacak sekilde yazilmisti; biz ayni mantigi, "once boslugu temizle,
# SONRA regex uygula" iki adimli yaklasimla daha OKUNABILIR ve test edilebilir
# hale getiriyoruz (sonuc kume olarak BIREBIR aynidir).
#
# Yapi: <Il Kodu (01-81)><Harf Grubu><Rakam Grubu>
#   1 harf  -> 4-5 rakam   (orn. 34A1234)
#   2 harf  -> 3-4 rakam   (orn. 34AB123)
#   3 harf  -> 2-3 rakam   (orn. 34ABC12)
# -----------------------------------------------------------------------------
PLAKA_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"(?:([A-Z])(\d{4,5})|([A-Z]{2})(\d{3,4})|([A-Z]{3})(\d{2,3}))$"
)

_easyocr_okuyucu = None  # Tembel (lazy) singleton — model bir kez yuklenir


def _ocr_okuyucusunu_getir():
    """EasyOCR Reader nesnesini TEMBEL (lazy) sekilde yukler ve yeniden kullanir.

    NEDEN tembel yukleme?
        EasyOCR'in agirliklarinin ilk yuklenmesi (~birkac saniye) maliyetlidir.
        Bu fonksiyon, modulu import eden HER cagrida degil, SADECE gercekten
        bir plaka okunacagi an modeli yukler ve sonraki cagrilarda
        TEKRAR KULLANIR (singleton deseni).
    """
    global _easyocr_okuyucu
    if _easyocr_okuyucu is None:
        import easyocr
        # NEDEN gpu=True default?
        #   Inference konteyneri GPU destekli calisacagi icin (Dockerfile'da
        #   nvidia/cuda taban imaji) OCR'i da GPU'da calistirmak, CPU'ya
        #   gore onemli bir hiz kazanci saglar; ayrica Model A/B ZATEN
        #   GPU'da oldugu icin ek bellek transferi maliyeti yoktur.
        _easyocr_okuyucu = easyocr.Reader(["en"], gpu=True)
    return _easyocr_okuyucu


def plakayi_normalize_et(ham_metin: str) -> str:
    """OCR'dan gelen ham metni FTR regex'ine uygun BUYUK HARF plakaya cevirir.

    Adimlar:
        1) Tum bosluklar ve OCR'da sik karisan noktalama isaretleri temizlenir.
        2) Metin BUYUK HARFE cevrilir (FTR Madde 3.4: plaka buyuk harf kalmali).
        3) PLAKA_REGEX'e uyuyor mu kontrol edilir; uymuyorsa BOS STRING donulur
           (yarisma JSON semasinda gecersiz/eksik plaka icin beklenen davranis).
    """
    if not ham_metin:
        return ""

    # NEDEN sadece harf/rakam birakiliyor (bosluk, tire, nokta vb. atiliyor)?
    #   FTR acikca "aradaki bosluklar temizlenerek normalize edilecektir" der;
    #   OCR ciktisinda sik gorulen "-", ".", "_" gibi gurultu karakterleri de
    #   AYNI mantikla temizlenmelidir, aksi halde gecerli bir plaka regex'i
    #   YANLISLIKLA reddedebilir.
    temizlenmis = re.sub(r"[^A-Za-z0-9]", "", ham_metin).upper()

    if PLAKA_REGEX.match(temizlenmis):
        return temizlenmis
    return ""


def plaka_bolgesini_oku(plaka_crop_bgr: np.ndarray) -> tuple:
    """Kirpilan plaka bolgesini OCR'dan gecirir, normalize eder ve guven skoruyla dondurur.

    Donus: (normalize_plaka: str, guven_skoru: float [0.0-1.0])
    """
    if plaka_crop_bgr is None or plaka_crop_bgr.size == 0:
        return "", 0.0

    okuyucu = _ocr_okuyucusunu_getir()
    # detail=1 -> [(bbox, metin, guven), ...] formatinda sonuc dondurur;
    # bu sayede HER bir metin parcasinin OCR guven skoruna erisebiliriz.
    sonuclar = okuyucu.readtext(plaka_crop_bgr, detail=1)

    if not sonuclar:
        return "", 0.0

    # NEDEN tum parcalar BIRLESTIRILIYOR?
    #   EasyOCR, "34" ve "ABC123" gibi plakayi BIRDEN FAZLA metin parcasina
    #   bolebilir (harf/rakam gruplari arasindaki bosluk nedeniyle). Plaka
    #   bolgesi ZATEN Model A tarafindan tek bir nesne olarak crop edildigi
    #   icin, TUM parcalarin soldan saga birlestirilmesi dogru plakayi verir.
    sonuclar.sort(key=lambda kayit: kayit[0][0][0])  # bbox sol-ust x koordinatina gore sirala
    birlesik_metin = "".join(metin for (_bbox, metin, _guven) in sonuclar)
    ortalama_guven = float(np.mean([guven for (_bbox, _metin, guven) in sonuclar]))

    normalize_plaka = plakayi_normalize_et(birlesik_metin)

    # Regex'e UYMAYAN bir okuma, dusuk guvenilirlikte sayilir; FTR'nin
    # istedigi "gecerli plaka formati" disinda hicbir deger JSON'a yazilmamali.
    if not normalize_plaka:
        return "", 0.0

    return normalize_plaka, round(ortalama_guven, 4)
