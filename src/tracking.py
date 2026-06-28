"""
src/tracking.py — Slalom Dinamik Aksiyonu Tespiti (Trajektori Tabanli)
=============================================================================
FTR Madde C: "Slalom tespiti icin sabit bir veri seti yoktur. Bu durum,
arac tespiti modelinden donen bounding box'larin merkez noktalarinin
ZAMANSAL ANALIZI ile heuristik/geometrik olarak cozulecektir."

NEDEN bu bir "model" DEGIL, geometrik/heuristik bir algoritmadir?
    Slalom, bir GORUNTU OZELLIGI degil bir HAREKET ORUNTUSUDUR (pattern).
    Tek bir karede slalom YAPILDIGINI gormek imkansizdir; sadece ARDISIK
    karelerdeki arac merkezinin X eksenindeki PERIYODIK salinimi bu davranisi
    ortaya cikarir. Bu yuzden EGITIM GEREKTIRMEYEN, deterministik bir
    sinyal isleme (signal processing) yaklasimi -- ardisik yon degisimi
    (zero-crossing) sayimi -- kullanilir.
=============================================================================
"""

from collections import deque


class SlalomDedektoru:
    """Bir aracin X eksenindeki merkez konum gecmisini tutar ve periyodik
    salinim (slalom) oruntusunu tespit eder.

    NEDEN sinif (class) olarak tasarlandi (fonksiyon degil)?
        Slalom tespiti, KARELER ARASI DURUM (state) gerektirir — tek bir
        karenin bilgisi yeterli degildir. Sinif, bu durumu (konum gecmisi)
        DOGAL ve kapsulleyici (encapsulated) bir sekilde tasir; video
        boyunca HER arac icin bagimsiz bir SlalomDedektoru ornegi tutulabilir.
    """

    def __init__(self, pencere_kare_sayisi: int = 30, min_yon_degisimi: int = 3,
                 min_genlik_piksel: float = 15.0):
        """
        Args:
            pencere_kare_sayisi: Analiz icin tutulan son N karenin merkez
                konum gecmisi (kayan pencere / sliding window).
            min_yon_degisimi: Pencere icinde, slalom olarak sayilmasi icin
                gereken MINIMUM yon degisimi (zero-crossing) sayisi.
            min_genlik_piksel: Gurultu (titreme/jitter) ile GERCEK salinimi
                ayirt etmek icin gereken minimum X ekseni hareket genligi.
        """
        self._gecmis = deque(maxlen=pencere_kare_sayisi)
        self._min_yon_degisimi = min_yon_degisimi
        self._min_genlik_piksel = min_genlik_piksel

    def guncelle(self, zaman_saniye: float, merkez_x: float) -> bool:
        """Yeni bir kare gozlemini ekler ve GUNCEL pencerede slalom var mi doner.

        NEDEN her guncellemede TUM pencere yeniden degerlendiriliyor
        (artimli/incremental hesap degil)?
            Pencere boyutu (varsayilan 30 kare, ~1-2 saniyelik video) kucuk
            oldugu icin O(N) yeniden hesaplama maliyeti ihmal edilebilir
            duzeydedir; KOD SADELIGI ve dogrulugu, mikro-optimizasyondan
            DAHA ONEMLIDIR.
        """
        self._gecmis.append((zaman_saniye, merkez_x))
        return self._slalom_oruntusu_var_mi()

    def _slalom_oruntusu_var_mi(self) -> bool:
        if len(self._gecmis) < 5:
            return False  # Guvenilir bir periyot tespiti icin yetersiz veri

        x_degerleri = [x for (_t, x) in self._gecmis]

        genlik = max(x_degerleri) - min(x_degerleri)
        if genlik < self._min_genlik_piksel:
            # Hareket genligi gurultu (titreme) seviyesinde -> slalom DEGIL
            return False

        # Ardisik kareler arasindaki konum FARKLARININ (delta) isaret
        # degisimi (zero-crossing) sayisi -> periyodik sag-sol salinimin
        # GUVENILIR bir gostergesidir (duz cizgide veya tek yonlu donuste
        # isaret degisimi OLMAZ ya da cok azdir).
        farklar = [x_degerleri[i + 1] - x_degerleri[i] for i in range(len(x_degerleri) - 1)]
        yon_degisim_sayisi = sum(
            1 for i in range(len(farklar) - 1)
            if farklar[i] != 0 and farklar[i + 1] != 0
            and (farklar[i] > 0) != (farklar[i + 1] > 0)
        )

        return yon_degisim_sayisi >= self._min_yon_degisimi
