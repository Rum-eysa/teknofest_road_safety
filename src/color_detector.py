import cv2
import numpy as np

from src.utils import VALID_COLORS

COLOR_TEMPLATES = {
    "beyaz": {"h_range": (0, 180), "s_range": (0, 50), "v_range": (200, 255)},
    "siyah": {"h_range": (0, 180), "s_range": (0, 255), "v_range": (0, 50)},
    "gri": {"h_range": (0, 180), "s_range": (0, 50), "v_range": (80, 180)},
    "kirmizi": {"h_range": (0, 10), "s_range": (100, 255), "v_range": (100, 255)},
    "mavi": {"h_range": (100, 130), "s_range": (100, 255), "v_range": (100, 255)},
    "sari": {"h_range": (20, 35), "s_range": (100, 255), "v_range": (100, 255)},
    "yesil": {"h_range": (35, 85), "s_range": (100, 255), "v_range": (100, 255)},
    "turuncu": {"h_range": (10, 20), "s_range": (100, 255), "v_range": (100, 255)},
    "kahverengi": {"h_range": (10, 20), "s_range": (100, 255), "v_range": (50, 150)},
}


def detect_color(vehicle_roi) -> str:
    if vehicle_roi is None or vehicle_roi.size == 0:
        return ""

    hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    median_h = float(np.median(h))
    median_s = float(np.median(s))
    median_v = float(np.median(v))

    best_color = ""
    best_score = -1.0

    for color_name, ranges in COLOR_TEMPLATES.items():
        h_min, h_max = ranges["h_range"]
        s_min, s_max = ranges["s_range"]
        v_min, v_max = ranges["v_range"]

        h_ok = h_min <= median_h <= h_max
        if color_name == "kirmizi" and median_h > 170:
            h_ok = True
        s_ok = s_min <= median_s <= s_max
        v_ok = v_min <= median_v <= v_max
        score = int(h_ok) + int(s_ok) + int(v_ok)

        if score > best_score:
            best_score = score
            best_color = color_name

    return best_color if best_color in VALID_COLORS else ""
