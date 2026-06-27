from pathlib import Path

import yaml


def ensure_data_yaml(dataset_path: Path, classes: list[str]) -> Path:
    data_yaml_path = dataset_path / "data.yaml"

    if data_yaml_path.exists():
        return data_yaml_path

    _validate_yolo_layout(dataset_path)
    _write_data_yaml(data_yaml_path, dataset_path, classes)
    return data_yaml_path


def _validate_yolo_layout(dataset_path: Path) -> None:
    required_dirs = [
        dataset_path / "images" / "train",
        dataset_path / "images" / "val",
        dataset_path / "labels" / "train",
        dataset_path / "labels" / "val",
    ]
    missing = [str(d) for d in required_dirs if not d.exists()]
    if missing:
        raise FileNotFoundError(
            "YOLO veri seti yapisi eksik. Asagidaki klasorler bulunamadi:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\n\nBeklenen yapi:\n"
            "  arac_bilgisi/\n"
            "    images/train, images/val, images/test\n"
            "    labels/train, labels/val, labels/test\n"
            "    data.yaml (opsiyonel, otomatik uretilir)"
        )


def _write_data_yaml(data_yaml_path: Path, dataset_path: Path, classes: list[str]) -> None:
    content = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (dataset_path / "images" / "test").exists() else "images/val",
        "names": {idx: name for idx, name in enumerate(classes)},
        "nc": len(classes),
    }

    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
