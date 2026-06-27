import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config dosyasi bulunamadi: {config_path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Gecersiz config formati: {config_path}")

    _validate_config(config)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    required_sections = ("dataset", "model", "training", "hardware", "output")
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config'de '{section}' bolumu eksik")

    classes = config["dataset"].get("classes", [])
    if not classes:
        raise ValueError("dataset.classes bos olamaz")

    splits = (
        config["dataset"].get("train_split", 0),
        config["dataset"].get("val_split", 0),
        config["dataset"].get("test_split", 0),
    )
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError("train/val/test split oranlari toplami 1.0 olmali")


def resolve_dataset_path(config: dict[str, Any]) -> Path:
    dataset_path = Path(config["dataset"]["path"])
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    return dataset_path


def resolve_output_dir(config: dict[str, Any]) -> Path:
    project = Path(config["output"]["project"])
    if not project.is_absolute():
        project = Path.cwd() / project
    return project / config["output"]["name"]


def apply_config_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    if overrides.get("epochs") is not None:
        config["training"]["epochs"] = overrides["epochs"]
    if overrides.get("batch_size") is not None:
        config["training"]["batch_size"] = overrides["batch_size"]
    if overrides.get("lr") is not None:
        config["training"]["lr0"] = overrides["lr"]
    if overrides.get("device") is not None:
        config["hardware"]["device"] = overrides["device"]
    return config
