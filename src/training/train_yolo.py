import signal
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.training.config_loader import load_config, resolve_dataset_path, resolve_output_dir
from src.training.dataset_yaml import ensure_data_yaml


class TrainingTimeoutError(TimeoutError):
    pass


def _infer_model_label(config_path: str, config: dict[str, Any]) -> str:
    meta = config.get("meta", {})
    if meta.get("description"):
        return meta["description"]

    name = Path(config_path).stem.lower()
    if "model_b" in name or "tespit" in name:
        return "Model B (tespitler)"
    if "model_a" in name or "arac" in name:
        return "Model A (arac_bilgisi)"
    return "YOLOv8 Egitimi"


def _build_train_kwargs(config: dict[str, Any], data_yaml: Path) -> tuple[str, dict[str, Any]]:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    train_cfg = config["training"]
    infer_cfg = config["inference"]
    hw_cfg = config["hardware"]
    out_cfg = config["output"]
    aug_cfg = dataset_cfg.get("augmentation", {})
    hsv_cfg = aug_cfg.get("hsv_augment", {})
    geo_cfg = aug_cfg.get("geometric", {})
    mosaic_cfg = aug_cfg.get("mosaic", {})
    mixup_cfg = aug_cfg.get("mixup", {})

    architecture = model_cfg["architecture"]
    if model_cfg.get("pretrained", True):
        model_name = f"{architecture}.pt"
    else:
        model_name = f"{architecture}.yaml"

    scale = geo_cfg.get("scale", 0.5)
    if isinstance(scale, list):
        scale = float(sum(scale) / len(scale))

    kwargs = {
        "data": str(data_yaml),
        "epochs": train_cfg["epochs"],
        "patience": train_cfg["patience"],
        "batch": train_cfg["batch_size"],
        "imgsz": train_cfg.get("imgsz", 640),
        "device": hw_cfg["device"],
        "project": out_cfg["project"],
        "name": out_cfg["name"],
        "exist_ok": True,
        "pretrained": model_cfg.get("pretrained", True),
        "optimizer": train_cfg["optimizer"],
        "lr0": train_cfg["lr0"],
        "lrf": train_cfg["lrf"],
        "momentum": train_cfg["momentum"],
        "weight_decay": train_cfg["weight_decay"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "warmup_momentum": train_cfg["warmup_momentum"],
        "amp": hw_cfg.get("amp", True),
        "workers": hw_cfg.get("workers", 4),
        "cache": hw_cfg.get("cache_images", False),
        "save": True,
        "save_period": infer_cfg.get("save_period", -1),
        "verbose": out_cfg.get("verbose", True),
        "single_cls": model_cfg.get("single_cls", False),
        "conf": infer_cfg.get("nms_conf_threshold", 0.25),
        "iou": infer_cfg.get("nms_iou_threshold", 0.45),
        "max_det": infer_cfg.get("max_det", 10),
        "hsv_h": hsv_cfg.get("h_gain", 0.015),
        "hsv_s": hsv_cfg.get("s_gain", 0.7),
        "hsv_v": hsv_cfg.get("v_gain", 0.4),
        "degrees": geo_cfg.get("degrees", 0.0),
        "translate": geo_cfg.get("translate", 0.1),
        "scale": scale,
        "perspective": geo_cfg.get("perspective", 0.0),
        "flipud": geo_cfg.get("flipud", 0.0),
        "fliplr": geo_cfg.get("fliplr", 0.5),
        "mosaic": 1.0 if mosaic_cfg.get("enabled", True) else 0.0,
        "mixup": mixup_cfg.get("prob", 0.0) if mixup_cfg.get("enabled", False) else 0.0,
        "plots": config.get("debug", {}).get("visualize_training", False),
    }

    accumulation = train_cfg.get("gradient_accumulation", 1)
    if accumulation and accumulation > 1:
        kwargs["accumulate"] = accumulation

    return model_name, kwargs


def train_from_config(config_path: str, overrides: dict[str, Any] | None = None) -> Path:
    config = load_config(config_path)
    if overrides:
        from src.training.config_loader import apply_config_overrides
        config = apply_config_overrides(config, overrides)
    dataset_path = resolve_dataset_path(config)
    classes = config["dataset"]["classes"]
    label = _infer_model_label(config_path, config)

    data_yaml = ensure_data_yaml(dataset_path, classes)
    model_name, train_kwargs = _build_train_kwargs(config, data_yaml)

    output_dir = resolve_output_dir(config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"{label} - YOLOv8 Egitimi Baslatiliyor")
    print("=" * 70)
    print(f"Config      : {config_path}")
    print(f"Dataset     : {dataset_path}")
    print(f"Data YAML   : {data_yaml}")
    print(f"Model       : {model_name}")
    print(f"Sinif sayisi: {len(classes)}")
    print(f"Siniflar    : {', '.join(classes)}")
    print(f"Cikti       : {output_dir}")
    print("=" * 70)

    model = YOLO(model_name)
    max_seconds = config["hardware"].get("max_training_time_seconds")

    if max_seconds:
        _train_with_timeout(model, train_kwargs, max_seconds)
    else:
        model.train(**train_kwargs)

    best_weights = output_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = output_dir / "weights" / "last.pt"

    print(f"\nEgitim tamamlandi. Agirliklar: {best_weights}")
    return best_weights


def _train_with_timeout(model: YOLO, train_kwargs: dict[str, Any], max_seconds: int) -> None:
    start = time.time()
    timed_out = False

    def _handle_timeout(signum, frame):
        nonlocal timed_out
        timed_out = True
        raise TrainingTimeoutError(
            f"Egitim suresi asildi ({max_seconds} saniye / {max_seconds / 3600:.1f} saat)"
        )

    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(max_seconds)

    try:
        model.train(**train_kwargs)
    finally:
        if use_alarm:
            signal.alarm(0)

    elapsed = time.time() - start
    print(f"Toplam egitim suresi: {elapsed / 60:.1f} dakika")

    if timed_out:
        raise TrainingTimeoutError("Egitim zaman asimina ugradi")
