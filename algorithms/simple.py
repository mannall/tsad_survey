#!/usr/bin/env python3

import numpy as np
from .utils import reverse_windowing
from numpy.lib.stride_tricks import sliding_window_view

class SimpleAD:
    def score(self, test_ts, window_size: int = 64) -> np.ndarray:
        windowed_ts = sliding_window_view(np.diff(test_ts, 2),  window_size)
        return reverse_windowing(np.std(windowed_ts, axis=1), window_size+2)