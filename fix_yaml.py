#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teknofest Road Safety - YAML Classes Otomatik Düzeltme
Model A ve Model B config dosyalarındaki classes bloğunu düzeltir.
"""

import os
import re
import sys
from pathlib import Path


MODEL_A_CLASSES = """  classes:
    - hatchback
    - kamyon
    - minibus
    - panelvan
    - pickup
    - plaka
    - sedan
    - suv"""

MODEL_B_CLASSES = """  classes:
    - arka_koltuk_1
    - arka_koltuk_2
    - arkaya_bakma
    - bilgisayar
    - emniyet_kemeri_ihlali
    - esneme
    - etrafa_bakinma
    - kemer_takili
    - on_koltuk
    - sigara_icme
    - su_icme
    - teknocan
    - telefonla_konusma"""


def fix_yaml_file(file_path, new_classes, file_type="model_a"):
    """
    YAML dosyasının classes bloğunu düzelt
    """
    fpath = Path(file_path)
    
    if not fpath.exists():
        print(f"⚠️  DOSYA BULUNAMADI: {file_path}")
        return False
    
    try:
        # Dosyayı oku
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # classes: ile başlayan bloğu bul
        # Pattern: "  classes:" ve sonra "    - ..." satırları
        pattern = r'  classes:\s*\n(?:    - [^\n]*\n?)+'
        
        # Eğer pattern bulunamazsa, farklı bir format deneme
        if not re.search(pattern, content):
            # Başka bir pattern: "classes:" sadece boşluksuz
            pattern = r'classes:\s*\n(?:  - [^\n]*\n?)+'
            if not re.search(pattern, content):
                # Son deneme: yaml list formatı
                pattern = r'  classes:\s*\n((?:\s*-\s+[^\n]*\n)*)'
        
        # Değiştir
        new_content = re.sub(
            pattern,
            new_classes + '\n',
            content,
            count=1
        )
        
        # Eğer hiç değişmemişse, format tamamen farklı demek
        if new_content == content:
            print(f"❌ HATA: {file_path} (format eşleşmiyor, manuel kontrol gerekli)")
            print(f"   Dosyanın ilk 30 satırını kontrol et:")
            print("   " + "\n   ".join(content.split('\n')[:30]))
            return False
        
        # Dosyaya yaz
        with open(fpath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(new_content)
        
        print(f"✅ DÜZELTILDI: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ HATA: {file_path} - {str(e)}")
        return False


def main():
    print("\n" + "="*70)
    print("  Teknofest Road Safety - YAML Classes Otomatik Düzeltme")
    print("="*70 + "\n")
    
    # Model A Dosyaları
    files_a = [
        'configs/model_a_config.yaml',
        'configs/model_a_config_local.yaml',
        'configs/config_exp_aggressive_aug.yaml',
        'configs/config_exp_combined.yaml',
    ]
    
    print("[1/2] Model A Config Dosyaları Düzeltiliyor...\n")
    success_a = 0
    for fpath in files_a:
        if fix_yaml_file(fpath, MODEL_A_CLASSES, "model_a"):
            success_a += 1
    
    # Model B Dosyaları
    files_b = [
        'configs/model_b_config.yaml',
        'configs/model_b_config_local.yaml',
    ]
    
    print("\n[2/2] Model B Config Dosyaları Düzeltiliyor...\n")
    success_b = 0
    for fpath in files_b:
        if fix_yaml_file(fpath, MODEL_B_CLASSES, "model_b"):
            success_b += 1
    
    # Özet
    print("\n" + "="*70)
    print("ÖZET")
    print("="*70)
    print(f"Model A: {success_a}/{len(files_a)} düzeltildi")
    print(f"Model B: {success_b}/{len(files_b)} düzeltildi")
    
    if success_a == len(files_a) and success_b == len(files_b):
        print("\n✅ TÜM DOSYALAR BAŞARIYLA DÜZELTILDI!")
        print("\nSonraki adım:")
        print("  git add -A")
        print("  git commit -m 'fix: YAML classes format'")
        print("  git push")
        return 0
    else:
        print("\n⚠️  BAZΙ DOSYALARDA SORUN VAR")
        print("   Yukarıdaki hata mesajlarını kontrol edin")
        print("   Manuel olarak YAML_MANUAL_FIX.md rehberine bakın")
        return 1


if __name__ == "__main__":
    sys.exit(main())
