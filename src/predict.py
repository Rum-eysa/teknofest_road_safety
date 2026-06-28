# -*- coding: utf-8 -*-
"""
src/predict.py — Ana Inference Pipeline

Her video frame'inde iki model paralel çalışır:
  Model A → araç tipi + plaka bbox (+ OCR) + renk (HSV)
  Model B → sürücü eylemi / yolcu / nesne
  Trajectory → slalom (konum bazlı, model gerektirmez)

Tüm sonuçlar tek bir JSON'da birleşir.
"""
import os
import time
from collections import Counter
from typing import Any

import cv2
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

# JSON'a yazılmayacak labellar:
#   kemer_takili → eğitimde kontrast sağlar, FTR'de tanımsız
#   slalom       → trajectory'den geliyor, Model B'den gelmemeli
INFERENCE_SKIP_LABELS = {"kemer_takili", "slalom"}

# Slalom cooldown: aynı slalomu kaç saniyede bir raporla
SLALOM_COOLDOWN_SEC = 5.0


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

        self.stats = {
            "total_frames": 0,
            "vehicle_detected_frames": 0,
            "safety_events": 0,
        }

    def process_video(
        self, video_path: str, timeout_seconds: float = 570
    ) -> dict[str, Any]:

        start_time = time.time()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Video acilamadi: {video_path}")

        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        video_name = os.path.basename(video_path)

        if self.logger:
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.logger.info(f"Video: {video_name} | FPS:{fps:.1f} | Kare:{total_f}")

        frame_idx  = 0
        tespitler  = []

        # Majority voting için biriktirme
        vehicle_type_votes = []   # [(tip, conf), ...]
        plate_reads        = []   # [(plaka_str, conf), ...]
        color_reads        = []   # [renk_str, ...]

        # Slalom takibi
        trajectory   = TrajectoryAnalyzer(window_size=30, logger=self.logger)
        last_slalom_t = -SLALOM_COOLDOWN_SEC  # cooldown başlangıcı

        try:
            while True:
                # Timeout kontrolü
                if time.time() - start_time > timeout_seconds:
                    if self.logger:
                        self.logger.warning("Inference timeout ulasildi")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Her 5. kareyi işle — T4'te 10dk timeout için kritik
                # 30fps → efektif 6fps, tespit için yeterli
                if frame_idx % 5 != 0:
                    frame_idx += 1
                    continue

                time_seconds = get_time_from_frame(frame_idx, fps)
                frame_input  = preprocess_frame(frame)

                with torch.no_grad():
                    results_a = self.model_a(
                        frame_input, conf=0.25, verbose=False, device=self.device
                    )
                    results_b = self.model_b(
                        frame_input, conf=0.30, verbose=False, device=self.device
                    )

                # ── MODEL A: Araç tipi + Plaka + Renk ───────────────────
                if len(results_a[0].boxes) > 0:
                    self.stats["vehicle_detected_frames"] += 1
                    boxes    = results_a[0].boxes
                    best_idx = int(torch.argmax(boxes.conf).item())
                    best_box = boxes[best_idx]

                    x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
                    class_id     = int(best_box.cls.item())
                    vehicle_conf = float(best_box.conf.item())
                    raw_label    = MODEL_A_YOLO_CLASSES[class_id % len(MODEL_A_YOLO_CLASSES)]

                    # "plaka" class'ı araç tipi değil — sadece OCR için kullan
                    if raw_label != "plaka":
                        vehicle_type_votes.append((raw_label, vehicle_conf))

                    # Araç bölgesini crop et
                    h_frame, w_frame = frame.shape[:2]
                    x1c = max(0, x1); y1c = max(0, y1)
                    x2c = min(w_frame, x2); y2c = min(h_frame, y2)
                    vehicle_roi = frame[y1c:y2c, x1c:x2c]

                    # Renk tespiti (HSV histogram)
                    color = detect_color(vehicle_roi)
                    if color:
                        color_reads.append(color)

                    # Plaka OCR (Model A'nın "plaka" class bbox'ını kullanır)
                    plate = extract_and_validate_plate(
                        frame, vehicle_roi, results_a[0], logger=self.logger
                    )
                    if plate:
                        plate_reads.append((plate, vehicle_conf))

                    # Slalom için araç merkezini güncelle
                    center_x = (x1 + x2) / 2
                    trajectory.update(center_x)

                # ── MODEL B: Sürücü eylemi / Yolcu / Nesne ──────────────
                if len(results_b[0].boxes) > 0:
                    for box in results_b[0].boxes:
                        class_id = int(box.cls.item())
                        conf     = float(box.conf.item())

                        if class_id >= len(MODEL_B_YOLO_CLASSES):
                            continue

                        label = MODEL_B_YOLO_CLASSES[class_id]

                        # kemer_takili ve slalom JSON'a yazılmaz
                        if label in INFERENCE_SKIP_LABELS:
                            continue

                        if label not in YOLO_CLASS_TO_TESPIT:
                            continue

                        kategori, etiket = YOLO_CLASS_TO_TESPIT[label]
                        tespitler.append({
                            "zaman_saniye":   time_seconds,
                            "kategori":       kategori,
                            "etiket":         etiket,
                            "confidence_score": round(min(conf, 1.0), 2),
                        })
                        self.stats["safety_events"] += 1

                # ── SLALOM: Trajectory analizi ───────────────────────────
                is_slalom, slalom_conf = trajectory.detect_slalom()
                if is_slalom and (time_seconds - last_slalom_t) >= SLALOM_COOLDOWN_SEC:
                    tespitler.append({
                        "zaman_saniye":   time_seconds,
                        "kategori":       "sofor_eylemi",
                        "etiket":         "slalom",
                        "confidence_score": round(slalom_conf, 2),
                    })
                    last_slalom_t = time_seconds
                    self.stats["safety_events"] += 1

                self.stats["total_frames"] = frame_idx
                frame_idx += 1

        finally:
            cap.release()

        elapsed = time.time() - start_time
        if self.logger:
            self.logger.info(
                f"Video islendi | Sure:{elapsed:.1f}s | "
                f"Islem:{self.stats['total_frames']} kare | "
                f"Tespit:{self.stats['safety_events']} olay"
            )

        # ── MAJORITY VOTING: Tüm video için en güvenilir arac_bilgisi ───
        # Araç tipi: en çok görülen kazanır
        if vehicle_type_votes:
            label_counts = Counter(l for l, _ in vehicle_type_votes)
            best_tip     = label_counts.most_common(1)[0][0]
            tip_confs    = [c for l, c in vehicle_type_votes if l == best_tip]
            tip_conf     = round(sum(tip_confs) / len(tip_confs), 2)
        else:
            best_tip, tip_conf = "", 0.0

        # Plaka: en yüksek confidence'lı okuma kazanır
        if plate_reads:
            best_plaka, plaka_conf = max(plate_reads, key=lambda x: x[1])
            plaka_conf = round(plaka_conf, 2)
        else:
            best_plaka, plaka_conf = "", 0.0

        # Renk: en çok görülen kazanır
        best_renk = Counter(color_reads).most_common(1)[0][0] if color_reads else ""

        # Genel confidence: tip + plaka ağırlıklı
        overall_conf = round(0.4 * tip_conf + 0.4 * plaka_conf + 0.2 * 1.0, 2)

        arac_bilgisi = {
            "tip":              best_tip,
            "plaka":            best_plaka,
            "renk":             best_renk,
            "confidence_score": overall_conf,
        }

        if self.logger:
            self.logger.info(
                f"Arac bilgisi → tip:{best_tip} | "
                f"plaka:{best_plaka} | renk:{best_renk} | "
                f"conf:{overall_conf}"
            )

        tespitler = merge_consecutive_detections(tespitler)

        return format_final_output({
            "video_id":     video_name,
            "arac_bilgisi": arac_bilgisi,
            "tespitler":    tespitler,
        })


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