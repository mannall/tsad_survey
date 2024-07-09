#!/usr/bin/env python3

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Simple_AD:
    def score(self, test_ts: np.ndarray) -> np.ndarray:
        window_size = int(0.01*len(test_ts))
        scores = np.std(sliding_window_view(np.diff(test_ts, 2, append=test_ts[-1]), window_shape=window_size), axis=-1)
        return np.r_[scores[:window_size//2], scores, scores[-window_size//2:]]