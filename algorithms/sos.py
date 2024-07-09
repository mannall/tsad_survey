#!/usr/bin/env python3

import numpy as np
from pyod.models.sos import SOS
from numpy.lib.stride_tricks import sliding_window_view

class SOS_AD:
    def __init__(self, window_size: int = 64, contamination: float = 0.05, perplexity: int = 32):
        self.window_size = window_size
        self.sos = SOS(
            contamination=contamination,
            perplexity=perplexity,
        )

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        self.sos.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        if self.window_size != 1:
            return np.r_[self.sos.decision_scores_[:self.window_size//2 - 1], self.sos.decision_scores_, self.sos.decision_scores_[-self.window_size//2:]]
        return self.sos.decision_scores_