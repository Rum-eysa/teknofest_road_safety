#!/usr/bin/env python3
"""
Model A / Model B YOLOv8 egitim giris noktasi.

Ornek:
    python train.py --config configs/model_a_config.yaml
    python train.py --config configs/model_b_config.yaml --lr 0.02 --batch-size 48
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.training.config_loader import load_config
from src.training.train_yolo import TrainingTimeoutError, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model A (arac_bilgisi) veya Model B (tespitler) YOLOv8 egitimi"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Egitim config YAML dosyasi",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Epoch override")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--device", type=str, default=None, help="GPU device override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": args.device,
    }

    try:
        load_config(args.config)
        train_from_config(args.config, overrides=overrides)
    except TrainingTimeoutError as exc:
        print(f"UYARI: {exc}")
        sys.exit(2)
    except Exception as exc:
        print(f"HATA: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
