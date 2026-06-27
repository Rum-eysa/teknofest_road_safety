# -*- coding: utf-8 -*-
"""
Teknofest Road Safety - JSON Output Validator
Çıktı dosyalarının yarışma standartlarına uygunluğunu kontrol eder.
"""

import json
import re
import sys
from pathlib import Path


# Yarışma standartları (Dokümantasyon s.2)
VALID_CATEGORIES = {"sofor_eylemi", "nesneler", "yolcular"}

VALID_LABELS = {
    "sofor_eylemi": {
        "arkaya_bakma", "esneme", "sigara_icme", "su_icme", 
        "telefonla_konusma", "slalom", "etrafa_bakinma", "emniyet_kemeri_ihlali"
    },
    "nesneler": {"teknocan", "bilgisayar"},
    "yolcular": {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}
}

VALID_VEHICLE_TYPES = {"sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"}
VALID_COLORS = {"beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"}

PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"((\s?[a-zA-Z]\s?)(\d{4,5})|(\s?[a-zA-Z]{2}\s?)(\d{3,4})|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$"
)

# Türkçe karakterler (OLMAMASI GEREKEN)
TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")


class Validator:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.errors = []
        self.warnings = []
        self.info = []
    
    def _log(self, level, message):
        if level == "ERROR":
            self.errors.append(message)
            if self.verbose:
                print(f"❌ {message}")
        elif level == "WARNING":
            self.warnings.append(message)
            if self.verbose:
                print(f"⚠️  {message}")
        elif level == "INFO":
            self.info.append(message)
            if self.verbose:
                print(f"ℹ️  {message}")
        elif level == "OK":
            if self.verbose:
                print(f"✅ {message}")
    
    def check_turkish_chars(self, text, field_name):
        """Türkçe karakter kontrolü"""
        if any(c in TURKISH_CHARS for c in str(text)):
            self._log("ERROR", f"{field_name} Türkçe karakter içeriyor: {text}")
            return False
        return True
    
    def check_ascii_safe(self, text, field_name):
        """ASCII güvenliği kontrolü"""
        try:
            text.encode('ascii')
            return True
        except UnicodeEncodeError:
            self._log("ERROR", f"{field_name} ASCII-safe değil: {text}")
            return False
    
    def check_plate(self, plate):
        """Plaka formatı kontrolü"""
        if not plate:
            return True
        
        # Normalize et (boşlukları kaldır)
        normalized = re.sub(r"\s+", "", plate.strip().upper())
        
        if PLATE_REGEX.match(normalized):
            self._log("OK", f"Plaka formatı geçerli: {plate} -> {normalized}")
            return True
        else:
            self._log("ERROR", f"Plaka formatı geçersiz: {plate}")
            return False
    
    def validate_arac_bilgisi(self, arac_bilgisi):
        """Araç bilgisi validasyonu"""
        print("\n--- Araç Bilgisi Kontrolü ---")
        
        if not isinstance(arac_bilgisi, dict):
            self._log("ERROR", "arac_bilgisi dict olmalı")
            return False
        
        all_valid = True
        
        # tip kontrolü
        if "tip" not in arac_bilgisi:
            self._log("ERROR", "arac_bilgisi.tip eksik")
            all_valid = False
        else:
            tip = arac_bilgisi["tip"]
            if tip and tip not in VALID_VEHICLE_TYPES:
                self._log("ERROR", f"tip geçersiz: {tip} (geçerli: {VALID_VEHICLE_TYPES})")
                all_valid = False
            else:
                self._log("OK", f"tip geçerli: {tip}")
            if not self.check_ascii_safe(tip, "tip"):
                all_valid = False
        
        # plaka kontrolü
        if "plaka" not in arac_bilgisi:
            self._log("ERROR", "arac_bilgisi.plaka eksik")
            all_valid = False
        else:
            if not self.check_plate(arac_bilgisi["plaka"]):
                all_valid = False
        
        # renk kontrolü
        if "renk" not in arac_bilgisi:
            self._log("ERROR", "arac_bilgisi.renk eksik")
            all_valid = False
        else:
            renk = arac_bilgisi["renk"]
            if renk and renk not in VALID_COLORS:
                self._log("ERROR", f"renk geçersiz: {renk} (geçerli: {VALID_COLORS})")
                all_valid = False
            else:
                self._log("OK", f"renk geçerli: {renk}")
            if renk and not self.check_ascii_safe(renk, "renk"):
                all_valid = False
        
        # confidence_score kontrolü
        if "confidence_score" not in arac_bilgisi:
            self._log("ERROR", "arac_bilgisi.confidence_score eksik")
            all_valid = False
        else:
            score = arac_bilgisi["confidence_score"]
            try:
                score_float = float(score)
                if 0.0 <= score_float <= 1.0:
                    self._log("OK", f"confidence_score geçerli: {score_float}")
                else:
                    self._log("ERROR", f"confidence_score 0-1 aralığı dışında: {score_float}")
                    all_valid = False
            except (ValueError, TypeError):
                self._log("ERROR", f"confidence_score sayı olmalı: {score}")
                all_valid = False
        
        return all_valid
    
    def validate_tespitler(self, tespitler):
        """Tespit listesi validasyonu"""
        print("\n--- Tespitler Kontrol Ediliyor ---")
        
        if not isinstance(tespitler, list):
            self._log("ERROR", "tespitler list olmalı")
            return False
        
        all_valid = True
        
        if len(tespitler) == 0:
            self._log("WARNING", "tespitler listesi boş")
            return True
        
        for idx, tespit in enumerate(tespitler):
            print(f"\n  [{idx}] Kontrol ediliyor...")
            
            if not isinstance(tespit, dict):
                self._log("ERROR", f"tespitler[{idx}] dict olmalı")
                all_valid = False
                continue
            
            # zaman_saniye kontrolü
            if "zaman_saniye" not in tespit:
                self._log("ERROR", f"tespitler[{idx}].zaman_saniye eksik")
                all_valid = False
            else:
                try:
                    zaman = float(tespit["zaman_saniye"])
                    if zaman >= 0:
                        self._log("OK", f"tespitler[{idx}].zaman_saniye geçerli: {zaman}s")
                    else:
                        self._log("ERROR", f"tespitler[{idx}].zaman_saniye negatif olmamalı: {zaman}")
                        all_valid = False
                except (ValueError, TypeError):
                    self._log("ERROR", f"tespitler[{idx}].zaman_saniye sayı olmalı: {tespit['zaman_saniye']}")
                    all_valid = False
            
            # kategori kontrolü
            if "kategori" not in tespit:
                self._log("ERROR", f"tespitler[{idx}].kategori eksik")
                all_valid = False
            else:
                kategori = tespit["kategori"]
                if kategori not in VALID_CATEGORIES:
                    self._log("ERROR", f"tespitler[{idx}].kategori geçersiz: {kategori}")
                    all_valid = False
                else:
                    self._log("OK", f"tespitler[{idx}].kategori geçerli: {kategori}")
            
            # etiket kontrolü
            if "etiket" not in tespit:
                self._log("ERROR", f"tespitler[{idx}].etiket eksik")
                all_valid = False
            else:
                etiket = tespit["etiket"]
                kategori = tespit.get("kategori", "")
                
                if kategori in VALID_LABELS:
                    if etiket not in VALID_LABELS[kategori]:
                        self._log("ERROR", 
                            f"tespitler[{idx}].etiket geçersiz: {etiket} "
                            f"(geçerli: {VALID_LABELS[kategori]})")
                        all_valid = False
                    else:
                        self._log("OK", f"tespitler[{idx}].etiket geçerli: {etiket}")
                
                if not self.check_ascii_safe(etiket, f"tespitler[{idx}].etiket"):
                    all_valid = False
            
            # confidence_score kontrolü
            if "confidence_score" not in tespit:
                self._log("ERROR", f"tespitler[{idx}].confidence_score eksik")
                all_valid = False
            else:
                score = tespit["confidence_score"]
                try:
                    score_float = float(score)
                    if 0.0 <= score_float <= 1.0:
                        self._log("OK", f"tespitler[{idx}].confidence_score geçerli: {score_float}")
                    else:
                        self._log("ERROR", f"tespitler[{idx}].confidence_score 0-1 aralığı dışında: {score_float}")
                        all_valid = False
                except (ValueError, TypeError):
                    self._log("ERROR", f"tespitler[{idx}].confidence_score sayı olmalı: {score}")
                    all_valid = False
        
        return all_valid
    
    def validate_json(self, json_file):
        """Tam JSON dosyası validasyonu"""
        print("="*70)
        print(f"Kontrol ediliyor: {json_file}")
        print("="*70)
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            self._log("ERROR", f"Dosya bulunamadı: {json_file}")
            return False
        except json.JSONDecodeError as e:
            self._log("ERROR", f"JSON format hatası: {e}")
            return False
        
        all_valid = True
        
        # Gerekli alanlar
        print("\n--- Üst Seviye Alanlar ---")
        if "video_id" not in data:
            self._log("ERROR", "video_id eksik")
            all_valid = False
        else:
            self._log("OK", f"video_id: {data['video_id']}")
        
        if "arac_bilgisi" not in data:
            self._log("ERROR", "arac_bilgisi eksik")
            all_valid = False
        else:
            if not self.validate_arac_bilgisi(data["arac_bilgisi"]):
                all_valid = False
        
        if "tespitler" not in data:
            self._log("ERROR", "tespitler eksik")
            all_valid = False
        else:
            if not self.validate_tespitler(data["tespitler"]):
                all_valid = False
        
        return all_valid
    
    def report(self):
        """Özet rapor"""
        print("\n" + "="*70)
        print("ÖZET RAPOR")
        print("="*70)
        
        print(f"\n✅ Başarılı: {len(self.info)}")
        print(f"❌ Hata: {len(self.errors)}")
        print(f"⚠️  Uyarı: {len(self.warnings)}")
        
        if self.errors:
            print("\n--- HATALAR ---")
            for err in self.errors:
                print(f"  • {err}")
        
        if self.warnings:
            print("\n--- UYARILAR ---")
            for warn in self.warnings:
                print(f"  • {warn}")
        
        print("\n" + "="*70)
        
        if len(self.errors) == 0:
            print("✅ TÜM KONTROLLER GEÇTI - JSON Yarışma Standartlarına Uygun!")
            return True
        else:
            print("❌ HATA VAR - Lütfen Yukarıdaki Hatalar Düzeltilsin")
            return False


def main():
    if len(sys.argv) < 2:
        print("Kullanım: python validate_json.py <results.json>")
        print("\nÖrnek: python validate_json.py output/results.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    validator = Validator(verbose=True)
    success = validator.validate_json(json_file)
    validator.report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
