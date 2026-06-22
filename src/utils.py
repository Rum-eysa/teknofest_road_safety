import cv2
import os
import sys
from loguru import logger as loguru_logger

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
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info

def format_output_json(predictions: dict, video_filename: str) -> dict:
    arac_bilgisi = {
        "tip": predictions.get("vehicle_type", "bilinmeyen"),
        "plaka": predictions.get("license_plate", ""),
        "renk": predictions.get("color", "bilinmeyen"),
        "confidence_score": float(predictions.get("confidence", 0.0))
    }
    
    tespitler = []
    for event in predictions.get("events", []):
        tespit = {
            "zaman_saniye": float(event.get("time", 0.0)),
            "kategori": event.get("category", "sofor_eylemi"),
            "etiket": event.get("label", ""),
            "confidence_score": float(event.get("confidence", 0.0))
        }
        tespitler.append(tespit)
    
    return {
        "video_id": video_filename,
        "arac_bilgisi": arac_bilgisi,
        "tespitler": tespitler
    }