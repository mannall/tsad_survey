#!/usr/bin/env python3

import numpy as np
from sklearn.cluster import KMeans
from numpy.lib.stride_tricks import sliding_window_view
from .utils import reverse_windowing

class KMeansAD:
    def __init__(self, window_size: int = 32, k: int = 65):
        self.window_size = window_size
        self.k = k
        self.model = KMeans(n_clusters=k)

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        windowed_ts = sliding_window_view(test_ts, self.window_size)
        self.model.fit(windowed_ts)
        clusters = self.model.predict(windowed_ts)
        diffs = np.linalg.norm(windowed_ts - self.model.cluster_centers_[clusters], axis=1)
        return reverse_windowing(diffs, self.window_size)