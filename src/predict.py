import cv2
import os
from src.utils import format_output_json, get_video_info

def run_inference(video_path: str, weights_path: str, logger) -> dict:
    logger.info(f"[Inference] Video analiz ediliyor: {video_path}")
    logger.info(f"[Inference] Model agirligi: {weights_path}")

    try:
        video_info = get_video_info(video_path)
        logger.info(f"   - FPS: {video_info['fps']}")
        logger.info(f"   - Cozunurluk: {video_info['width']}x{video_info['height']}")
        logger.info(f"   - Frame sayisi: {video_info['frame_count']}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Video acilamadi: {video_path}")

        # TODO: weights_path uzerinden model yukleme ve frame bazli cikarim burada yapilacak
        if not os.path.exists(weights_path):
            logger.warning(f"Model dosyasi bulunamadi: {weights_path}")

        predictions = {
            "vehicle_type": "sedan",
            "license_plate": "34ABC123",
            "color": "beyaz",
            "confidence": 0.85,
            "events": [
                {"time": 5.2, "category": "sofor_eylemi", "label": "telefonla_konusma", "confidence": 0.89},
                {"time": 12.8, "category": "nesneler", "label": "teknocan", "confidence": 0.92},
            ]
        }

        cap.release()
        logger.info("Video analiz tamamlandi")

        video_filename = os.path.basename(video_path)
        return format_output_json(predictions, video_filename)

    except Exception as e:
        logger.error(f"Hata: {str(e)}")
        raise
