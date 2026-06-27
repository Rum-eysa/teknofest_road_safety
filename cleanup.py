#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL: Repo'yu tamamen temizle ve GitHub'a push et
"""
import os
import subprocess

def run(cmd):
    print(f"$ {cmd}")
    os.system(cmd)

print("\n" + "="*60)
print("FINAL CLEANUP - Repository Temizleme")
print("="*60 + "\n")

# Silinmiş dosyaları git'ten kaldır
print("🗑️  Silinmiş dosyalar git'ten kaldırılıyor...\n")
run('git add -A')

print("\n✅ Commit yapılıyor...\n")
run('git commit -m "cleanup: final cleanup"')

print("\n📤 GitHub'a gönderiliyor...\n")
run('git push')

print("\n" + "="*60)
print("✅ BITTI! Repo temiz ve güncel")
print("="*60)
print("\nKontrol et: https://github.com/Rum-eysa/teknofest_road_safety")