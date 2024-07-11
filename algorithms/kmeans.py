#!/usr/bin/env python3

import numpy as np

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, OutlierMixin
from numpy.lib.stride_tricks import sliding_window_view

from .utils import reverse_windowing

class KMeans_AD(BaseEstimator, OutlierMixin):
    def __init__(self, k: int = 65):
        self.k = k
        self.model = KMeans(n_clusters=k)

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.fit(X)
        clusters = self.model.predict(X)
        diffs = np.linalg.norm(X - self.model.cluster_centers_[clusters], axis=1)
        return reverse_windowing(diffs, X.shape[1])