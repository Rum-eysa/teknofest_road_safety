from collections import deque

import numpy as np


class TrajectoryAnalyzer:
    def __init__(self, window_size: int = 30, logger=None):
        self.window_size = window_size
        self.logger = logger
        self.positions = deque(maxlen=window_size)

    def update(self, x_position: float) -> None:
        self.positions.append(float(x_position))

    def detect_slalom(self, min_std: float = 25.0) -> tuple[bool, float]:
        if len(self.positions) < 10:
            return False, 0.0

        series = np.array(self.positions, dtype=np.float32)
        centered = series - np.mean(series)
        std_x = float(np.std(centered))
        if std_x < min_std:
            return False, 0.0

        fft = np.fft.rfft(centered)
        power = np.abs(fft) ** 2
        if power.sum() <= 0:
            return False, 0.0

        peak_power = float(power[1:].max()) if len(power) > 1 else 0.0
        confidence = min(peak_power / float(power.sum()), 1.0)

        if confidence >= 0.35 and std_x >= min_std:
            return True, confidence
        return False, confidence


def detect_slalom(positions: list[float]) -> tuple[bool, float]:
    analyzer = TrajectoryAnalyzer(window_size=len(positions))
    for pos in positions:
        analyzer.update(pos)
    return analyzer.detect_slalom()
