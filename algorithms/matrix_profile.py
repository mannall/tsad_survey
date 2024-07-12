#!/usr/bin/env python3

import stumpy
import numpy as np

from .utils import reverse_windowing

class MatrixProfileAD():
    def __init__(self, m: int = 100):
        self.m = m

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        # initializing the first stumpy.stump() object will take a long time (~20s)
        mp = stumpy.stump(test_ts, self.m)
        return reverse_windowing(mp[:, 0], self.m)