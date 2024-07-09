#!/usr/bin/env python3

import numpy as np
from sklearn.ensemble import IsolationForest
from numpy.lib.stride_tricks import sliding_window_view

class IForest_AD():
    def __init__(self, window_size: int = 64, n_trees: int = 100, max_samples: str = "auto", contamination: float = 0.05):
        self.window_size = window_size
        # n_trees really has no effect on the classifier performance >100
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.contamination = contamination
        self.iforest = IsolationForest(
            n_estimators=self.n_trees,
            max_samples=self.max_samples,
            contamination=self.contamination,
        )

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        self.iforest.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        scores = -self.iforest.decision_function(sliding_window_view(test_ts, window_shape=self.window_size))
        if self.window_size != 1:
            return np.r_[scores[:self.window_size//2 - 1], scores, scores[-self.window_size//2:]]
        else:
            return scores