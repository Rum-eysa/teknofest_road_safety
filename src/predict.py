import os
import time
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
        arac_bilgisi = None
        tespitler = []

        try:
            while True:
                if time.time() - start_time > timeout_seconds:
                    if self.logger:
                        self.logger.warning("Inference timeout ulasildi")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

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
                    vehicle_type = MODEL_A_YOLO_CLASSES[class_id % len(MODEL_A_YOLO_CLASSES)]

                    if vehicle_type == "plaka":
                        vehicle_type = "sedan"

                    vehicle_roi = frame[y1:y2, x1:x2]
                    color = detect_color(vehicle_roi)
                    plate = extract_and_validate_plate(
                        frame, vehicle_roi, results_a[0], logger=self.logger
                    )

                    vehicle_center_x = (x1 + x2) // 2
                    self.trajectory_analyzer.update(vehicle_center_x)

                    if arac_bilgisi is None:
                        arac_bilgisi = {
                            "tip": vehicle_type,
                            "plaka": plate,
                            "renk": color,
                            "confidence_score": min(vehicle_conf, 1.0),
                        }
                    elif vehicle_conf > arac_bilgisi["confidence_score"]:
                        arac_bilgisi["confidence_score"] = min(vehicle_conf, 1.0)
                        if plate:
                            arac_bilgisi["plaka"] = plate
                        if color:
                            arac_bilgisi["renk"] = color
                        if vehicle_type:
                            arac_bilgisi["tip"] = vehicle_type

                if len(results_b[0].boxes) > 0:
                    for box in results_b[0].boxes:
                        class_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        if class_id >= len(MODEL_B_YOLO_CLASSES):
                            continue

                        label = MODEL_B_YOLO_CLASSES[class_id]
                        if label == "slalom":
                            continue

                        kategori, etiket = YOLO_CLASS_TO_TESPIT[label]
                        tespitler.append({
                            "zaman_saniye": time_seconds,
                            "kategori": kategori,
                            "etiket": etiket,
                            "confidence_score": min(conf, 1.0),
                        })
                        self.stats["safety_events"] += 1

                if frame_idx % 5 == 0 and frame_idx > 20:
                    is_slalom, slalom_conf = self.trajectory_analyzer.detect_slalom()
                    if is_slalom:
                        tespitler.append({
                            "zaman_saniye": time_seconds,
                            "kategori": "sofor_eylemi",
                            "etiket": "slalom",
                            "confidence_score": slalom_conf,
                        })

                self.stats["total_frames"] = frame_idx
                frame_idx += 1

        finally:
            cap.release()

        tespitler = merge_consecutive_detections(tespitler)

        output = {
            "video_id": video_name,
            "arac_bilgisi": arac_bilgisi or {
                "tip": "",
                "plaka": "",
                "renk": "",
                "confidence_score": 0.0,
            },
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
