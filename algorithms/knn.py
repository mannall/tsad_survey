#!/usr/bin/env python3

import numpy as np
from pyod.models.knn import KNN
from numpy.lib.stride_tricks import sliding_window_view

class KNN_AD:
    def __init__(self, window_size: int = 100, contamination: float = 0.05, n_neighbors: int = 65, leaf_size: int = 30):
        self.window_size = window_size
        self.clf = KNN(
            contamination=contamination,
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
        )

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        self.clf.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        return np.r_[self.clf.decision_scores_[:self.window_size//2 - 1], self.clf.decision_scores_, self.clf.decision_scores_[-self.window_size//2:]]