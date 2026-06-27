import os
import re
import sys
import subprocess
from collections import Counter

def check_directory():
    print("\n============================================================")
    print("  Teknofest Road Safety - Otomatik Kod Duzeltme Scripti")
    print("============================================================\n")
    
    if not os.path.exists(os.path.join("src", "predict.py")):
        print("[HATA] Bu scripti repo kok dizininde calistirin.")
        print("       Simdi bulundugunuz yer yanlis olmali.")
        print("       Dogru dizin: teknofest_road_safety/")
        sys.exit(1)

def update_utils():
    print("[1/6] src/utils.py guncelleniyor...")
    print("       - MODEL_A_YOLO_CLASSES: Roboflow alfabetik sirasi")
    print("       - MODEL_B_YOLO_CLASSES: kemer_takili eklendi, slalom kaldirildi")
    print("       - VALID_DRIVER_ACTIONS: slalom eklendi (trajectory'den geliyor)")

    new_utils = '''import cv2
import re
import sys
from typing import Any

from loguru import logger as loguru_logger

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

# -------------------------------------------------------------------
# Model A: Roboflow alfabetik export sirasi (Roboflow data.yaml ile eslesir)
# hatchback=0, kamyon=1, minibus=2, panelvan=3, pickup=4,
# plaka=5, sedan=6, suv=7
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# Model B: Roboflow alfabetik export sirasi
# kemer_takili: modelin ogrenecegi ama JSON'a YAZILMAYACAK class
# Neden var? emniyet_kemeri_ihlali ile karsilastirmali ogrenmesi icin
# -------------------------------------------------------------------
MODEL_B_YOLO_CLASSES = [
    "arka_koltuk_1",           # 0
    "arka_koltuk_2",           # 1
    "arkaya_bakma",            # 2
    "bilgisayar",              # 3
    "emniyet_kemeri_ihlali",   # 4
    "esneme",                  # 5
    "etrafa_bakinma",          # 6
    "kemer_takili",            # 7  -- JSON'a yazilmaz, sadece egitimde kontrast saglar
    "on_koltuk",               # 8
    "sigara_icme",             # 9
    "su_icme",                 # 10
    "teknocan",                # 11
    "telefonla_konusma",       # 12
]

# -------------------------------------------------------------------
# Inference sirasinda hangi label hangi FTR kategori/etiketine gider
# kemer_takili burada YOK -- predict.py'de filtreleniyor
# -------------------------------------------------------------------
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
    loguru_logger.remove()
    loguru_logger.add(
        sys.stdout,
        format="<level>{time:YYYY-MM-DD HH:mm:ss}</level> | <level>{level: <8}</level> | {message}",
        level="INFO"
    )
    return loguru_logger


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
    with open('src/utils.py', 'w', encoding='utf-8') as f:
        f.write(new_utils)
    print('[OK] src/utils.py guncellendi\n')

def update_predict():
    print("[2/6] src/predict.py guncelleniyor...")
    print("       - vehicle_type==plaka oldugunda None ata (sedan degil)")
    print("       - kemer_takili filtreleme eklendi")
    print("       - majority voting ile arac_bilgisi (tek kare degil tum video)")

    new_predict = '''import os
import time
from collections import Counter
from typing import Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.color_detector import detect_color
from src.ocr_handler import extract_and_validate_plate
from src.trajectory_analyzer import TrajectoryAnalyzer
from src.utils import (
    MODEL_A_YOLO_CLASSES,
    MODEL_B_YOLO_CLASSES,
    YOLO_CLASS_TO_TESPIT,
    format_final_output,
    get_time_from_frame,
    merge_consecutive_detections,
    preprocess_frame,
)

# JSON'a yazilmayacak class'lar:
# kemer_takili: egitimde kontrast icin var, FTR ciktisinda yok
# slalom: trajectory_analyzer'dan geliyor, model B'den gelmemeli
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

        if logger:
            logger.info(f"Model A yuklendi: {model_a_weights}")
            logger.info(f"Model B yuklendi: {model_b_weights}")

        self.trajectory_analyzer = TrajectoryAnalyzer(window_size=30, logger=logger)
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
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_path)

        frame_idx = 0
        tespitler = []

        # Majority voting icin biriktirme listeleri
        vehicle_type_votes = []   # [(tip, conf), ...]
        plate_reads        = []   # [(plaka, conf), ...]
        color_reads        = []   # [renk, ...]

        try:
            while True:
                if time.time() - start_time > timeout_seconds:
                    if self.logger:
                        self.logger.warning("Inference timeout ulasildi")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Her 5. kareyi isle -- T4'te 10 dakika timeout icin kritik
                if frame_idx % 5 != 0:
                    frame_idx += 1
                    continue

                time_seconds = get_time_from_frame(frame_idx, fps)
                frame_input = preprocess_frame(frame)

                with torch.no_grad():
                    results_a = self.model_a(frame_input, conf=0.25, verbose=False, device=self.device)
                    results_b = self.model_b(frame_input, conf=0.30, verbose=False, device=self.device)

                # ── MODEL A: Arac tipi, plaka, renk ─────────────────────
                if len(results_a[0].boxes) > 0:
                    self.stats["vehicle_detected_frames"] += 1
                    boxes    = results_a[0].boxes
                    best_idx = int(torch.argmax(boxes.conf).item())
                    best_box = boxes[best_idx]

                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                    class_id    = int(best_box.cls.item())
                    vehicle_conf = float(best_box.conf.item())
                    raw_label   = MODEL_A_YOLO_CLASSES[class_id % len(MODEL_A_YOLO_CLASSES)]

                    # plaka class'i arac tipi degil -- atla, sadece OCR icin kullan
                    if raw_label != "plaka":
                        vehicle_type_votes.append((raw_label, vehicle_conf))

                    vehicle_roi = frame[y1:y2, x1:x2]

                    # Renk tespiti
                    color = detect_color(vehicle_roi)
                    if color:
                        color_reads.append(color)

                    # Plaka OCR
                    plate = extract_and_validate_plate(
                        frame, vehicle_roi, results_a[0], logger=self.logger
                    )
                    if plate:
                        plate_reads.append((plate, vehicle_conf))

                    # Slalom icin arac merkezi guncelle
                    vehicle_center_x = (x1 + x2) // 2
                    self.trajectory_analyzer.update(vehicle_center_x)

                # ── MODEL B: Surucu eylemi, yolcu, nesne ────────────────
                if len(results_b[0].boxes) > 0:
                    for box in results_b[0].boxes:
                        class_id = int(box.cls.item())
                        conf     = float(box.conf.item())
                        if class_id >= len(MODEL_B_YOLO_CLASSES):
                            continue

                        label = MODEL_B_YOLO_CLASSES[class_id]

                        if label in INFERENCE_SKIP_LABELS:
                            continue

                        if label not in YOLO_CLASS_TO_TESPIT:
                            continue

                        kategori, etiket = YOLO_CLASS_TO_TESPIT[label]
                        tespitler.append({
                            "zaman_saniye":   time_seconds,
                            "kategori":       kategori,
                            "etiket":         etiket,
                            "confidence_score": min(conf, 1.0),
                        })
                        self.stats["safety_events"] += 1

                # ── SLALOM: Konum bazli trajectory analizi ───────────────
                if frame_idx > 20:
                    is_slalom, slalom_conf = self.trajectory_analyzer.detect_slalom()
                    if is_slalom:
                        tespitler.append({
                            "zaman_saniye":   time_seconds,
                            "kategori":       "sofor_eylemi",
                            "etiket":         "slalom",
                            "confidence_score": slalom_conf,
                        })

                self.stats["total_frames"] = frame_idx
                frame_idx += 1

        finally:
            cap.release()

        # ── MAJORITY VOTING: Tum video icin en guvenilir arac bilgisi ───
        if vehicle_type_votes:
            label_counts = Counter(l for l, _ in vehicle_type_votes)
            best_tip = label_counts.most_common(1)[0][0]
            tip_confs = [c for l, c in vehicle_type_votes if l == best_tip]
            tip_conf  = round(sum(tip_confs) / len(tip_confs), 2)
        else:
            best_tip, tip_conf = "", 0.0

        if plate_reads:
            best_plaka, plaka_conf = max(plate_reads, key=lambda x: x[1])
        else:
            best_plaka, plaka_conf = "", 0.0

        if color_reads:
            best_renk = Counter(color_reads).most_common(1)[0][0]
        else:
            best_renk = ""

        overall_conf = round(0.4 * tip_conf + 0.4 * plaka_conf + 0.2 * 1.0, 2)

        arac_bilgisi = {
            "tip":              best_tip,
            "plaka":            best_plaka,
            "renk":             best_renk,
            "confidence_score": overall_conf,
        }

        tespitler = merge_consecutive_detections(tespitler)

        output = {
            "video_id":     video_name,
            "arac_bilgisi": arac_bilgisi,
            "tespitler":    tespitler,
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
    with open('src/predict.py', 'w', encoding='utf-8') as f:
        f.write(new_predict)
    print('[OK] src/predict.py guncellendi\n')

def update_yaml_configs():
    print("[3/6] Model A konfigürasyonları güncelleniyor...")
    files_a = [
        'configs/model_a_config.yaml',
        'configs/model_a_config_local.yaml',
        'configs/config_exp_aggressive_aug.yaml',
        'configs/config_exp_combined.yaml',
    ]
    new_classes_a = '''  classes:
    - "hatchback"   # 0 - Roboflow alfabetik sirasi
    - "kamyon"      # 1
    - "minibus"     # 2
    - "panelvan"    # 3
    - "pickup"      # 4
    - "plaka"       # 5
    - "sedan"       # 6
    - "suv"         # 7
'''
    pattern_a = r'  classes:\n(?:    - "[^"]+"\n)+'
    for fpath in files_a:
        _modify_yaml(fpath, pattern_a, new_classes_a)

    print("\n[4/6] Model B konfigürasyonları güncelleniyor...")
    files_b = [
        'configs/model_b_config.yaml',
        'configs/model_b_config_local.yaml',
    ]
    new_classes_b = '''  classes:
    - "arka_koltuk_1"           # 0 - Roboflow alfabetik sirasi
    - "arka_koltuk_2"           # 1
    - "arkaya_bakma"            # 2
    - "bilgisayar"              # 3
    - "emniyet_kemeri_ihlali"   # 4
    - "esneme"                  # 5
    - "etrafa_bakinma"          # 6
    - "kemer_takili"            # 7 -- JSON'a yazilmaz, egitimde kontrast saglar
    - "on_koltuk"               # 8
    - "sigara_icme"             # 9
    - "su_icme"                 # 10
    - "teknocan"                # 11
    - "telefonla_konusma"       # 12
'''
    pattern_b = r'  classes:\n(?:    - "[^"]+"\s*(?:#[^\n]*)?\n)+'
    for fpath in files_b:
        _modify_yaml(fpath, pattern_b, new_classes_b)

def _modify_yaml(fpath, pattern, replacement):
    if not os.path.exists(fpath):
        print(f"DOSYA YOK (atla): {fpath}")
        return
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"OK: {fpath}")
    else:
        print(f"ATLA (pattern bulunamadi): {fpath}")

def verify_changes():
    print("\n[5/6] Dogrulama: Degisiklikler kontrol ediliyor...")
    try:
        from src.utils import MODEL_A_YOLO_CLASSES, MODEL_B_YOLO_CLASSES
        from src.predict import INFERENCE_SKIP_LABELS
        print('MODEL_A_YOLO_CLASSES:')
        for i, c in enumerate(MODEL_A_YOLO_CLASSES):
            print(f'  {i}: {c}')
        print('\nMODEL_B_YOLO_CLASSES:')
        for i, c in enumerate(MODEL_B_YOLO_CLASSES):
            skip = ' <-- JSON yazilmaz' if c in INFERENCE_SKIP_LABELS else ''
            print(f'  {i}: {c}{skip}')
    except Exception as e:
        print(f"[UYARI] Import dogrulamasi calismadi: {e}")
        print("         (Gerekli kütüphaneler yerelde kurulu olmayabilir. Docker içinde sorunsuz çalışacaktır.)")

def git_status():
    print("\n[6/6] Git durumu...")
    try:
        subprocess.run(["git", "diff", "--stat"], check=True)
        print("\nDegisiklikleri commit etmek ister misiniz?")
        print('  git add -A')
        print('  git commit -m "fix: class sirasi, kemer_takili filtresi, majority voting"')
        print('  git push')
    except Exception:
        print("[UYARI] Git bulunamadi veya repo initialize edilmedi.")

def show_summary():
    print("\n============================================================")
    print(" TAMAMLANDI - Yapilan degisiklikler:")
    print("============================================================\n")
    print(" src/utils.py\n    + MODEL_A_YOLO_CLASSES: Roboflow alfabetik sirasi\n    + MODEL_B_YOLO_CLASSES: kemer_takili=7 eklendi\n    + VALID_DRIVER_ACTIONS: slalom eklendi\n")
    print(" src/predict.py\n    + vehicle_type==plaka oldugunda Atlanıyor (sedan degil)\n    + kemer_takili ve slalom filtreleme: INFERENCE_SKIP_LABELS\n    + Majority voting: tum video en guvenilir arac_bilgisi\n    + Frame skip=5: T4 timeout onleme\n")
    print(" Konfigürasyon Dosyaları\n    + model_a ve model_b YAML dosyaları Roboflow export sırasıyla senkronize edildi.\n")
    print("============================================================")
    print(" SONRAKI ADIMLAR:")
    print("============================================================\n")
    print(" 1. Roboflow'dan veri setlerini YOLOv8 formatinda export et")
    print(" 2. data.yaml icindeki 'names:' sirasiyla config sınıflarını doğrula")
    print(" 3. Docker build: docker compose build baseline")
    print(" 4. Egitimi baslat: docker compose up -d baseline")
    print(" 5. Ağırlıkları kopyala: copy output\\experiment_baseline\\weights\\best.pt models\\best_a.pt")

if __name__ == "__main__":
    try:
        check_directory()
        update_utils()
        update_predict()
        update_yaml_configs()
        verify_changes()
        git_status()
        show_summary()
    except Exception as e:
        print(f"\n❌ [HATA] Bir sorun olustu: {e}")
        sys.exit(1)