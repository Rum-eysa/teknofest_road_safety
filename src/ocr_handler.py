import re
from typing import Optional

import cv2
import numpy as np

from src.utils import is_valid_plate, normalize_plate

_OCR_READER = None


def _get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _OCR_READER


def _preprocess_plate_crop(plate_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def extract_and_validate_plate(
    frame: np.ndarray,
    vehicle_roi: np.ndarray,
    yolo_result,
    logger=None,
) -> str:
    plate_crop = _find_plate_crop(frame, yolo_result)
    if plate_crop is None or plate_crop.size == 0:
        return ""

    try:
        reader = _get_ocr_reader()
        processed = _preprocess_plate_crop(plate_crop)
        results = reader.readtext(processed, detail=1, paragraph=False)
    except Exception as exc:
        if logger:
            logger.warning(f"OCR basarisiz: {exc}")
        return ""

    candidates = []
    for _bbox, text, conf in results:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text)
        normalized = normalize_plate(cleaned)
        if normalized and is_valid_plate(normalized):
            candidates.append((normalized, float(conf)))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def _find_plate_crop(frame: np.ndarray, yolo_result) -> Optional[np.ndarray]:
    if yolo_result is None or yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return None

    names = yolo_result.names
    plate_indices = [
        idx for idx, cls_id in enumerate(yolo_result.boxes.cls.tolist())
        if names.get(int(cls_id), "") == "plaka"
    ]

    if plate_indices:
        idx = plate_indices[0]
        x1, y1, x2, y2 = map(int, yolo_result.boxes.xyxy[idx].tolist())
        return frame[y1:y2, x1:x2]

    boxes = yolo_result.boxes
    best_idx = int(boxes.conf.argmax().item())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
    h, w = frame.shape[:2]
    plate_y1 = min(y2, h - 1)
    plate_y2 = min(y2 + max(20, (y2 - y1) // 4), h)
    plate_x1 = max(0, x1)
    plate_x2 = min(w, x2)
    return frame[plate_y1:plate_y2, plate_x1:plate_x2]
