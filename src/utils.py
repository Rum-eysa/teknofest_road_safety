import cv2
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
     "etrafa_bakinma", "emniyet_kemeri_ihlali",
}
VALID_OBJECTS = {"teknocan", "bilgisayar"}
VALID_PASSENGERS = {"arka_koltuk_1", "arka_koltuk_2", "on_koltuk"}

MODEL_A_YOLO_CLASSES = [
    "hatchback", "pickup", "sedan", "suv", "minibus", "panelvan", "kamyon", "plaka"
]

MODEL_B_YOLO_CLASSES = [
    "arkaya_bakma", "esneme", "sigara_icme", "su_icme", "telefonla_konusma",
     "etrafa_bakinma", "emniyet_kemeri_ihlali",
    "teknocan", "bilgisayar",
    "arka_koltuk_1", "arka_koltuk_2", "on_koltuk",
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
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "Ç": "c", "Ğ": "g", "İ": "i", "Ö": "o", "Ş": "s", "Ü": "u",
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
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
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
    merge_window: float = 1.0,
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
        "tip": vehicle_type if vehicle_type in VALID_VEHICLE_TYPES else "",
        "plaka": normalized_plate,
        "renk": color if color in VALID_COLORS else "",
        "confidence_score": float(predictions.get("confidence", 0.0))
    }

    tespitler = []
    for event in predictions.get("events", []):
        category = event.get("category", "")
        if category not in VALID_CATEGORIES:
            continue

        tespitler.append({
            "zaman_saniye": float(event.get("time", 0.0)),
            "kategori": category,
            "etiket": event.get("label", ""),
            "confidence_score": float(event.get("confidence", 0.0))
        })

    return format_final_output({
        "video_id": video_filename,
        "arac_bilgisi": arac_bilgisi,
        "tespitler": tespitler
    })
