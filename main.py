import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import run_inference
from src.utils import setup_logging

def main():
    logger = setup_logging()

    input_path = "/app/data/input/video.mp4"
    output_path = "/app/data/output/results.json"
    weights_path = "/app/models/best_model.pt"

    logger.info("=" * 70)
    logger.info("Teknofest Akilli Yol Guvenligi - AI Inference Islemi Basladi")
    logger.info("=" * 70)

    if not os.path.exists(input_path):
        logger.error(f"HATA: Girdi videosu bulunamadi -> {input_path}")
        sys.exit(1)

    logger.info(f"Video: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Model: {weights_path}")

    try:
        logger.info("Video analiz islemi basliyor...")
        output_data = run_inference(input_path, weights_path, logger)

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Islem basarili! Cikti kaydedildi: {output_path}")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"HATA: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
