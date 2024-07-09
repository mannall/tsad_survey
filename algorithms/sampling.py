#!/usr/bin/env python3

import numpy as np
from pyod.models.sampling import Sampling
from numpy.lib.stride_tricks import sliding_window_view

class Sampling_AD:
    def __init__(self, window_size: int = 64, contamination: float = 0.05, subset_size: float = 0.1):
        self.window_size = window_size
        self.sampling = Sampling(
            contamination=contamination,
            subset_size=subset_size,
        )

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        self.sampling.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        if self.window_size != 1:
            return np.r_[self.sampling.decision_scores_[:self.window_size//2 - 1], self.sampling.decision_scores_, self.sampling.decision_scores_[-self.window_size//2:]]
        return self.sampling.decision_scores_