from src.training.train_yolo import TrainingTimeoutError, train_from_config


def train_model_a(config_path: str):
    return train_from_config(config_path)


__all__ = ["TrainingTimeoutError", "train_model_a"]
