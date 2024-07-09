#!/usr/bin/env python3

import numpy as np
from pyod.models.lof import LOF
from numpy.lib.stride_tricks import sliding_window_view

class LOF_AD():
    def __init__(self, window_size: int = 50, contamination: float = 0.05, n_neighbors: int = 50, leaf_size: int = 30):
        # determines the context window. optimal value dependent on the period of the timeseries - ideally less than 10% of input length
        self.window_size = window_size
        # https://pyod.readthedocs.io/en/latest/_modules/pyod/models/lof.html
        self.clf = LOF(
            # the estimated proportion of outliers in input
            contamination = contamination,
            # how many neighbors to consider when determining outlier score. higher values = less spiky scores(?) 
            n_neighbors = n_neighbors,
            # doesn't impact results, only the speed
            leaf_size = leaf_size,
        )
    
    def score(self, test_ts: np.ndarray) -> np.ndarray:
        # transforms the timeseries into a (window_size, len(data) - window_size) matrix
        # each row @ index j contains the timeseries samples from times j to j+window_size
        self.clf.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        # centre anomaly scores, and mirror scores around edges
        return np.r_[self.clf.decision_scores_[:self.window_size//2 - 1], self.clf.decision_scores_, self.clf.decision_scores_[-self.window_size//2:]]