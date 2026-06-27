#!/usr/bin/env python3
"""
Teknofest Yol Guvenligi - inference entry point.

Input : /app/data/input/video.mp4
Output: /app/data/output/results.json
Models: /app/models/best_a.pt, /app/models/best_b.pt
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import run_inference
from src.utils import ensure_ascii_safe, setup_logging, validate_output_schema

INPUT_VIDEO_PATH = "/app/data/input/video.mp4"
OUTPUT_JSON_PATH = "/app/data/output/results.json"
MODEL_A_WEIGHTS = "/app/models/best_a.pt"
MODEL_B_WEIGHTS = "/app/models/best_b.pt"
MAX_EXECUTION_TIME_SECONDS = 10 * 60


def main():
    logger = setup_logging()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("Teknofest Yol Guvenligi - Inference Pipeline")
    logger.info("=" * 70)

    if not os.path.exists(INPUT_VIDEO_PATH):
        logger.error(f"Input video bulunamadi: {INPUT_VIDEO_PATH}")
        sys.exit(1)

    if not os.path.exists(MODEL_A_WEIGHTS):
        logger.error(f"Model A agirligi bulunamadi: {MODEL_A_WEIGHTS}")
        sys.exit(1)

    if not os.path.exists(MODEL_B_WEIGHTS):
        logger.error(f"Model B agirligi bulunamadi: {MODEL_B_WEIGHTS}")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    try:
        output_data = run_inference(
            video_path=INPUT_VIDEO_PATH,
            model_a_weights=MODEL_A_WEIGHTS,
            model_b_weights=MODEL_B_WEIGHTS,
            timeout_seconds=MAX_EXECUTION_TIME_SECONDS - 30,
            logger=logger,
        )

        is_valid, errors = validate_output_schema(output_data)
        if not is_valid:
            logger.error(f"Output schema hatasi: {errors}")
            sys.exit(1)

        output_data = ensure_ascii_safe(output_data)

        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - start_time
        logger.info(f"Inference tamamlandi ({elapsed:.1f}s)")
        logger.info(f"Cikti: {OUTPUT_JSON_PATH}")
        logger.info(f"Tespit sayisi: {len(output_data.get('tespitler', []))}")

        if elapsed > MAX_EXECUTION_TIME_SECONDS:
            logger.error("10 dakika timeout asildi")
            sys.exit(1)

    except TimeoutError as exc:
        logger.error(f"Timeout: {exc}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"Inference hatasi: {exc}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
