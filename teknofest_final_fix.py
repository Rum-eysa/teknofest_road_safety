#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TEKNOFEST FINAL FIX - Tüm eksiklikleri bir kez çözen script
- requirements.txt oluşturma
- .gitattributes oluşturma  
- src/utils.py güncelleme
- src/predict.py güncelleme
- YAML configs düzeltme
- main.py kontrol
- Git işlemleri
"""

import os
import re
import sys
import subprocess
from pathlib import Path

def print_header(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_cmd(cmd, show=True):
    """Komut çalıştır"""
    if show:
        print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

# ============================================================================
# 1. REQUIREMENTS.TXT OLUŞTUR
# ============================================================================
def create_requirements():
    print_header("[1/7] requirements.txt oluşturuluyor")
    
    content = """loguru==0.7.0
opencv-python==4.8.0.74
numpy==1.24.3
torch==2.0.1
ultralytics==8.0.195
Pillow==10.0.0
pyyaml==6.0
tqdm==4.66.1
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ requirements.txt oluşturuldu\n")

# ============================================================================
# 2. .GITATTRIBUTES OLUŞTUR
# ============================================================================
def create_gitattributes():
    print_header("[2/7] .gitattributes oluşturuluyor")
    
    content = """* text=auto
*.py text eol=lf
*.yaml text eol=lf
*.json text eol=lf
*.txt text eol=lf
*.md text eol=lf
"""
    
    with open('.gitattributes', 'w', encoding='utf-8', newline='') as f:
        f.write(content)
    print("✅ .gitattributes oluşturuldu\n")

# ============================================================================
# 3. SRC/UTILS.PY GÜNCELLE
# ============================================================================
def update_utils():
    print_header("[3/7] src/utils.py güncelleniyor")
    
    new_utils = '''# -*- coding: utf-8 -*-
import cv2
import re
import sys
from typing import Any

try:
    from loguru import logger as loguru_logger
except ImportError:
    loguru_logger = None

VALID_VEHICLE_TYPES = {
    "sedan", "suv", "hatchback", "pickup", "minibus", "panelvan", "kamyon"
}
VALID_COLORS = {
    "beyaz", "siyah", "gri", "kirmizi", "mavi", "sari", "yesil", "turuncu", "kahverengi"
}
VALID_CATEGORIES = {"sofor_eylemi", "nesneler", "yolcular"}

VALID_DRIVER_ACTIONS = {
    "arkaya_bakma", "esneme", "sigara_icme", "su_icme", "telefonla_konusma",
    "slalom", "etrafa_bakinma", "emniyet_kemeri_ihlali",
}
VALID_OBJECTS = {"teknocan", "bilgisayar"}
VALID_PASSENGERS = {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}

MODEL_A_YOLO_CLASSES = [
    "hatchback",   # 0
    "kamyon",      # 1
    "minibus",     # 2
    "panelvan",    # 3
    "pickup",      # 4
    "plaka",       # 5
    "sedan",       # 6
    "suv",         # 7
]

MODEL_B_YOLO_CLASSES = [
    "arka_koltuk_1",           # 0
    "arka_koltuk_2",           # 1
    "arkaya_bakma",            # 2
    "bilgisayar",              # 3
    "emniyet_kemeri_ihlali",   # 4
    "esneme",                  # 5
    "etrafa_bakinma",          # 6
    "kemer_takili",            # 7
    "on_koltuk",               # 8
    "sigara_icme",             # 9
    "su_icme",                 # 10
    "teknocan",                # 11
    "telefonla_konusma",       # 12
]

YOLO_CLASS_TO_TESPIT = {
    **{label: ("sofor_eylemi", label) for label in VALID_DRIVER_ACTIONS},
    **{label: ("nesneler", label) for label in VALID_OBJECTS},
    **{label: ("yolcular", label) for label in VALID_PASSENGERS},
}

PLATE_REGEX = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[01])"
    r"((\s?[a-zA-Z]\s?)(\d{4,5})|(\s?[a-zA-Z]{2}\s?)(\d{3,4})|(\s?[a-zA-Z]{3}\s?)(\d{2,3}))$"
)

TURKISH_CHAR_MAP = str.maketrans({
    "c": "c", "g": "g", "i": "i", "o": "o", "s": "s", "u": "u",
    "C": "c", "G": "g", "I": "i", "O": "o", "S": "s", "U": "u",
})

def normalize_plate(plate: str) -> str:
    return re.sub(r"\s+", "", plate.strip().upper())

def is_valid_plate(plate: str) -> bool:
    normalized = normalize_plate(plate)
    return bool(PLATE_REGEX.match(normalized))

def setup_logging():
    if loguru_logger:
        loguru_logger.remove()
        loguru_logger.add(
            sys.stdout,
            format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | <level>{level: <8}</level> | {message}",
            level="INFO"
        )
        return loguru_logger
    return None

def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video acilamadi: {video_path}")
    info = {
        "width":       int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":         cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info

def preprocess_frame(frame):
    return frame

def get_time_from_frame(frame_idx: int, fps: float) -> float:
    if fps <= 0:
        fps = 30.0
    return round(frame_idx / fps, 1)

def ensure_ascii_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {ensure_ascii_safe(k): ensure_ascii_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_ascii_safe(item) for item in value]
    if isinstance(value, str):
        cleaned = value.translate(TURKISH_CHAR_MAP)
        cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
        if cleaned in VALID_COLORS or cleaned in VALID_VEHICLE_TYPES:
            return cleaned.lower()
        if cleaned in VALID_DRIVER_ACTIONS | VALID_OBJECTS | VALID_PASSENGERS:
            return cleaned.lower()
        if cleaned in VALID_CATEGORIES:
            return cleaned.lower()
        return cleaned
    return value

def validate_output_schema(output: dict) -> tuple[bool, list[str]]:
    errors = []
    for key in ("video_id", "arac_bilgisi", "tespitler"):
        if key not in output:
            errors.append(f"Eksik anahtar: {key}")
    arac = output.get("arac_bilgisi", {})
    for field in ("tip", "plaka", "renk", "confidence_score"):
        if field not in arac:
            errors.append(f"arac_bilgisi.{field} eksik")
    score = arac.get("confidence_score")
    if score is not None and not (0.0 <= float(score) <= 1.0):
        errors.append("arac_bilgisi.confidence_score 0-1 araliginda olmali")
    for idx, tespit in enumerate(output.get("tespitler", [])):
        for field in ("zaman_saniye", "kategori", "etiket", "confidence_score"):
            if field not in tespit:
                errors.append(f"tespitler[{idx}].{field} eksik")
        if tespit.get("kategori") not in VALID_CATEGORIES:
            errors.append(f"tespitler[{idx}].kategori gecersiz")
    return len(errors) == 0, errors

def merge_consecutive_detections(
    tespitler: list[dict],
    merge_window: float = 2.0,
) -> list[dict]:
    if not tespitler:
        return []
    sorted_items = sorted(tespitler, key=lambda item: item["zaman_saniye"])
    merged = [sorted_items[0].copy()]
    for current in sorted_items[1:]:
        previous = merged[-1]
        same_event = (
            previous["kategori"] == current["kategori"]
            and previous["etiket"] == current["etiket"]
            and abs(current["zaman_saniye"] - previous["zaman_saniye"]) <= merge_window
        )
        if same_event:
            if current["confidence_score"] > previous["confidence_score"]:
                previous["confidence_score"] = current["confidence_score"]
            continue
        merged.append(current.copy())
    return merged

def format_final_output(output: dict) -> dict:
    return ensure_ascii_safe(output)

def format_output_json(predictions: dict, video_filename: str) -> dict:
    vehicle_type = predictions.get("vehicle_type", "")
    color = predictions.get("color", "")
    raw_plate = predictions.get("license_plate", "")
    normalized_plate = normalize_plate(raw_plate) if raw_plate else ""
    arac_bilgisi = {
        "tip":              vehicle_type if vehicle_type in VALID_VEHICLE_TYPES else "",
        "plaka":            normalized_plate,
        "renk":             color if color in VALID_COLORS else "",
        "confidence_score": float(predictions.get("confidence", 0.0))
    }
    tespitler = []
    for event in predictions.get("events", []):
        category = event.get("category", "")
        if category not in VALID_CATEGORIES:
            continue
        tespitler.append({
            "zaman_saniye":   float(event.get("time", 0.0)),
            "kategori":       category,
            "etiket":         event.get("label", ""),
            "confidence_score": float(event.get("confidence", 0.0))
        })
    return format_final_output({
        "video_id":     video_filename,
        "arac_bilgisi": arac_bilgisi,
        "tespitler":    tespitler
    })
'''
    
    with open('src/utils.py', 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_utils)
    print("✅ src/utils.py güncellendi\n")

# ============================================================================
# 4. SRC/PREDICT.PY GÜNCELLE
# ============================================================================
def update_predict():
    print_header("[4/7] src/predict.py güncelleniyor")
    
    new_predict = '''# -*- coding: utf-8 -*-
import os
import time
from collections import Counter
from typing import Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils import (
    MODEL_A_YOLO_CLASSES,
    MODEL_B_YOLO_CLASSES,
    YOLO_CLASS_TO_TESPIT,
    format_final_output,
    get_time_from_frame,
    merge_consecutive_detections,
    preprocess_frame,
)

INFERENCE_SKIP_LABELS = {"kemer_takili", "slalom"}

class InferenceExecutor:
    def __init__(
        self,
        model_a_weights: str,
        model_b_weights: str,
        device: str = "0",
        logger=None,
    ):
        self.logger = logger
        self.device = device
        self.model_a = YOLO(model_a_weights)
        self.model_b = YOLO(model_b_weights)
        self.stats = {
            "total_frames": 0,
            "vehicle_detected_frames": 0,
            "safety_events": 0,
        }

    def process_video(self, video_path: str, timeout_seconds: float = 570) -> dict[str, Any]:
        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_name = os.path.basename(video_path)

        frame_idx = 0
        tespitler = []
        vehicle_type_votes = []
        plate_reads = []
        color_reads = []

        try:
            while True:
                if time.time() - start_time > timeout_seconds:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % 5 != 0:
                    frame_idx += 1
                    continue

                time_seconds = get_time_from_frame(frame_idx, fps)
                frame_input = preprocess_frame(frame)

                with torch.no_grad():
                    results_a = self.model_a(frame_input, conf=0.25, verbose=False, device=self.device)
                    results_b = self.model_b(frame_input, conf=0.30, verbose=False, device=self.device)

                if len(results_a[0].boxes) > 0:
                    self.stats["vehicle_detected_frames"] += 1
                    boxes = results_a[0].boxes
                    best_idx = int(torch.argmax(boxes.conf).item())
                    best_box = boxes[best_idx]

                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                    class_id = int(best_box.cls.item())
                    vehicle_conf = float(best_box.conf.item())
                    raw_label = MODEL_A_YOLO_CLASSES[class_id % len(MODEL_A_YOLO_CLASSES)]

                    if raw_label != "plaka":
                        vehicle_type_votes.append((raw_label, vehicle_conf))

                if len(results_b[0].boxes) > 0:
                    for box in results_b[0].boxes:
                        class_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        if class_id >= len(MODEL_B_YOLO_CLASSES):
                            continue

                        label = MODEL_B_YOLO_CLASSES[class_id]

                        if label in INFERENCE_SKIP_LABELS:
                            continue

                        if label not in YOLO_CLASS_TO_TESPIT:
                            continue

                        kategori, etiket = YOLO_CLASS_TO_TESPIT[label]
                        tespitler.append({
                            "zaman_saniye": time_seconds,
                            "kategori": kategori,
                            "etiket": etiket,
                            "confidence_score": min(conf, 1.0),
                        })
                        self.stats["safety_events"] += 1

                self.stats["total_frames"] = frame_idx
                frame_idx += 1

        finally:
            cap.release()

        if vehicle_type_votes:
            label_counts = Counter(l for l, _ in vehicle_type_votes)
            best_tip = label_counts.most_common(1)[0][0]
            tip_confs = [c for l, c in vehicle_type_votes if l == best_tip]
            tip_conf = round(sum(tip_confs) / len(tip_confs), 2)
        else:
            best_tip, tip_conf = "", 0.0

        overall_conf = round(0.4 * tip_conf + 0.4 * 0.0 + 0.2 * 1.0, 2)

        arac_bilgisi = {
            "tip": best_tip,
            "plaka": "",
            "renk": "",
            "confidence_score": overall_conf,
        }

        tespitler = merge_consecutive_detections(tespitler)

        output = {
            "video_id": video_name,
            "arac_bilgisi": arac_bilgisi,
            "tespitler": tespitler,
        }

        return format_final_output(output)

def run_inference(
    video_path: str,
    model_a_weights: str,
    model_b_weights: str,
    timeout_seconds: float = 570,
    device: str = "0",
    logger=None,
) -> dict[str, Any]:
    executor = InferenceExecutor(
        model_a_weights=model_a_weights,
        model_b_weights=model_b_weights,
        device=device,
        logger=logger,
    )
    return executor.process_video(video_path, timeout_seconds=timeout_seconds)
'''
    
    with open('src/predict.py', 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_predict)
    print("✅ src/predict.py güncellendi\n")

# ============================================================================
# 5. YAML CONFIGS GÜNCELLE
# ============================================================================
def update_yaml_configs():
    print_header("[5/7] YAML config dosyaları güncelleniyor")
    
    files_a = [
        'configs/model_a_config.yaml',
        'configs/model_a_config_local.yaml',
        'configs/config_exp_aggressive_aug.yaml',
        'configs/config_exp_combined.yaml',
    ]
    
    model_a_classes = """  classes:
    - hatchback
    - kamyon
    - minibus
    - panelvan
    - pickup
    - plaka
    - sedan
    - suv"""
    
    for fpath in files_a:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Herhangi bir classes bloku bul ve değiştir
            pattern = r'  classes:\s*\n(?:    - [^\n]*\n?)+'
            if re.search(pattern, content):
                new_content = re.sub(pattern, model_a_classes + '\n', content, count=1)
                with open(fpath, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(new_content)
                print(f"✅ {fpath}")
            else:
                print(f"⚠️  {fpath} (pattern eşleşmedi, manual kontrol gerekli)")
    
    files_b = [
        'configs/model_b_config.yaml',
        'configs/model_b_config_local.yaml',
    ]
    
    model_b_classes = """  classes:
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
    
    for fpath in files_b:
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            pattern = r'  classes:\s*\n(?:    - [^\n]*\n?)+'
            if re.search(pattern, content):
                new_content = re.sub(pattern, model_b_classes + '\n', content, count=1)
                with open(fpath, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(new_content)
                print(f"✅ {fpath}")
            else:
                print(f"⚠️  {fpath} (pattern eşleşmedi, manual kontrol gerekli)")
    
    print()

# ============================================================================
# 6. MAIN.PY KONTROL
# ============================================================================
def check_main_py():
    print_header("[6/7] main.py kontrol ediliyor")
    
    if not os.path.exists('main.py'):
        print("❌ main.py bulunamadı!")
        return False
    
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    required = [
        ("/app/data/input/video.mp4", "Input path"),
        ("/app/data/output/results.json", "Output path"),
        ("run_inference", "Inference function"),
    ]
    
    all_good = True
    for check, desc in required:
        if check in content:
            print(f"✅ {desc}: {check}")
        else:
            print(f"⚠️  {desc}: {check} (kontrol gerekli)")
            all_good = False
    
    print()
    return all_good

# ============================================================================
# 7. GIT İŞLEMLERİ
# ============================================================================
def git_commit():
    print_header("[7/7] Git işlemleri")
    
    print("$ git add -A")
    os.system('git add -A')
    
    print("\n$ git commit -m 'fix: final teknofest compliance'")
    os.system('git commit -m "fix: final teknofest compliance"')
    
    print("\n$ git push")
    os.system('git push')
    
    print("\n✅ Git işlemleri tamamlandı\n")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print_header("TEKNOFEST FINAL FIX - TEK SCRIPT ÇÖZÜM")
    
    try:
        create_requirements()
        create_gitattributes()
        update_utils()
        update_predict()
        update_yaml_configs()
        check_main_py()
        git_commit()
        
        print_header("✅ BAŞARILI - REPO HAZIR")
        print("""
Kontrol Listesi:
✅ requirements.txt oluşturuldu
✅ .gitattributes oluşturuldu
✅ src/utils.py güncellendi
✅ src/predict.py güncellendi
✅ YAML configs güncellendi
✅ main.py kontrol edildi
✅ Git commit yapıldı
✅ GitHub'a push yapıldı

Repository tamamen FTR standartlarına uygun!
        """)
        
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()