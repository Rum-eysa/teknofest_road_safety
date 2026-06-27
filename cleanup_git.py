# -*- coding: utf-8 -*-
"""
Teknofest - Git Cleanup Script
Gereksiz dosyaları git'ten kaldırır ve .gitignore oluşturur
"""

import os
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Komut çalıştır"""
    if description:
        print(f"\n{description}")
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"❌ Hata: {result.stderr}")
    return result.returncode == 0

def main():
    print("\n" + "="*70)
    print("  Teknofest - Git Cleanup (Gereksiz Dosyaları Kaldırma)")
    print("="*70 + "\n")
    
    # Kaldırılması gereken dosyalar (git tracked olarak)
    files_to_remove = [
        'bool',
        'dict', 
        'str',
        'src/__pycache__/utils.cpython-311.pyc',
        'fix_and_prepare.bat',
        'fix_pipeline.py',
    ]
    
    # .gitignore oluştur (olmadıysa)
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
models/
weights/
output/
runs/
logs/
*.mp4
*.avi
*.mov

# Temp files
*.tmp
*.bak
*.log
bool
dict
str
"""
    
    print("[1/4] .gitignore oluşturuluyor...")
    try:
        with open('.gitignore', 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("✅ .gitignore oluşturuldu\n")
    except Exception as e:
        print(f"❌ .gitignore oluşturulamadı: {e}\n")
    
    print("[2/4] Git history'den gereksiz dosyalar kaldırılıyor...\n")
    
    success = 0
    for file in files_to_remove:
        if os.path.exists(file) or file in ['src/__pycache__/utils.cpython-311.pyc']:
            # Git'ten kaldır (history'den değil, sadece staging'den)
            cmd = f'git rm --cached "{file}"'
            if run_command(cmd, f"  🗑️  Kaldırılıyor: {file}"):
                success += 1
        else:
            print(f"  ℹ️  Zaten yok: {file}")
    
    print(f"\n✅ {success}/{len(files_to_remove)} dosya kaldırıldı")
    
    # Gerekli dosyalar kontrol et
    print("\n[3/4] Gerekli dosyalar kontrol ediliyor...\n")
    
    required_files = [
        'requirements.txt',
        '.gitattributes',
        'fix_script_v2.py',
        'fix_yaml.py',
        'validate_json.py',
        'example_output.json',
        'KURULUM_REHBERI.md',
        'HATALAR_VE_COZUMLER.md',
        'HEMEN_YAPILACAK.md',
        'YAML_MANUAL_FIX.md',
        'src/utils.py',
        'src/predict.py',
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - EKSIK!")
            all_present = False
    
    if not all_present:
        print("\n⚠️  Bazı dosyalar eksik!")
    
    print("\n[4/4] Final Git İşlemleri...\n")
    
    # Status
    print("📊 Git Status:")
    run_command('git status --short', "")
    
    print("\n" + "="*70)
    print("YAPILACAK İŞLEM:")
    print("="*70)
    print("\nAşağıdaki komutları çalıştırın:\n")
    
    print("1️⃣  Değişiklikleri stage et:")
    print("   git add -A\n")
    
    print("2️⃣  Commit et:")
    print('   git commit -m "cleanup: remove unnecessary files"\n')
    
    print("3️⃣  Push et:")
    print("   git push\n")
    
    print("="*70)
    print("\nOtomatik yapılsın mı? (y/n): ", end='')
    
    # Otomatik olarak yapma seçeneği
    auto = input().lower() == 'y'
    
    if auto:
        print("\n⏳ İşlemler yapılıyor...\n")
        
        if run_command('git add -A', "git add -A"):
            print("✅ Dosyalar staged\n")
        
        if run_command('git commit -m "cleanup: remove unnecessary files"', 
                      "git commit"):
            print("✅ Commit oluşturuldu\n")
        
        if run_command('git push', "git push"):
            print("✅ Push yapıldı\n")
            print("="*70)
            print("✅ TEMİZLİK TAMAMLANDI!")
            print("="*70)
        else:
            print("❌ Push başarısız")
    else:
        print("\nℹ️  Yukarıdaki komutları manual olarak çalıştırın")


if __name__ == "__main__":
    main()
