#!/usr/bin/env python3

import numpy as np
from pyod.models.kde import KDE
from numpy.lib.stride_tricks import sliding_window_view

class KDE_AD:
    def __init__(self, window_size: int = 100, contamination: float = 0.05, bandwidth: float = 1, algorithm: str = "auto", leaf_size: int = 30):
        self.window_size = window_size
        self.kde = KDE(
            contamination=contamination,
            bandwidth=bandwidth,
            algorithm=algorithm,
            leaf_size=leaf_size,
        )

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        self.kde.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        return np.r_[self.kde.decision_scores_[:self.window_size//2 - 1], self.kde.decision_scores_, self.kde.decision_scores_[-self.window_size//2:]]