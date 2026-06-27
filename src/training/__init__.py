__all__ = ["train_from_config", "train_model_a", "train_model_b"]


def __getattr__(name: str):
    if name == "train_from_config":
        from src.training.train_yolo import train_from_config
        return train_from_config
    if name == "train_model_a":
        from src.training.train_model_a import train_model_a
        return train_model_a
    if name == "train_model_b":
        from src.training.train_model_b import train_model_b
        return train_model_b
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
