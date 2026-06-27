import cv2
import numpy as np

def detect_color(vehicle_roi) -> str:
    if vehicle_roi is None or vehicle_roi.size == 0:
        return ""

    # Gövde ortasını al: lastik ve cam hariç tutulur
    h, w = vehicle_roi.shape[:2]
    roi = vehicle_roi[int(h*0.15):int(h*0.85), int(w*0.1):int(w*0.9)]
    if roi.size == 0:
        return ""

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    total = hsv.shape[0] * hsv.shape[1]

    COLOR_RANGES = {
        "kirmizi":    [((0,50,50),(10,255,255)), ((160,50,50),(179,255,255))],
        "turuncu":    [((10,100,100),(25,255,255))],
        "sari":       [((25,80,80),(35,255,255))],
        "yesil":      [((35,50,50),(85,255,255))],
        "mavi":       [((85,50,50),(130,255,255))],
        "beyaz":      [((0,0,180),(179,40,255))],
        "siyah":      [((0,0,0),(179,255,50))],
        "gri":        [((0,0,51),(179,50,179))],
        "kahverengi": [((5,50,30),(20,200,160))],
    }

    scores = {}
    for color, ranges in COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
        scores[color] = cv2.countNonZero(mask) / total

    best = max(scores, key=scores.get)
    return best if scores[best] > 0.05 else "gri"
