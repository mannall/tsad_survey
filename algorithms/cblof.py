#!/usr/bin/env python3

import numpy as np
from pyod.models.cblof import CBLOF
from numpy.lib.stride_tricks import sliding_window_view

class CBLOF_AD():
    def __init__(self, window_size: int = 50, contamination: float = 0.05, n_clusters: int = 10, alpha: float = 0.9, beta: float = 5):
        self.window_size = window_size
        self.cblof = CBLOF(
            contamination=contamination,
            n_clusters=n_clusters,
            alpha=alpha,
            beta=beta,
        )
    
    def score(self, test_ts: np.ndarray) -> np.ndarray:
        # transforms the timeseries into a (window_size, len(data) - window_size) matrix
        # each row @ index j contains the timeseries samples from times j to j+window_size
        self.cblof.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        # centre anomaly scores, and mirror scores around edges
        return np.r_[self.cblof.decision_scores_[:self.window_size//2 - 1], self.cblof.decision_scores_, self.cblof.decision_scores_[-self.window_size//2:]]