#!/usr/bin/env python3

import stumpy
import numpy as np

class Matrix_Profile_AD():
    def __init__(self, m: int = 100, percentage: float = 0.1):
        self.m = m
        # percentage of pairs used when computing matrix profile.
        # matrix profile's can often be very well approximated using ~5-10% of the data
        # this parameter increases computation time linearly, but values >0.1 seem very unadvantageous.
        self.percentage = percentage

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        # initializing the first stumpy.scrump() object will take a long time (~20s)
        approx_mp = stumpy.scrump(test_ts, self.m, percentage=self.percentage)
        approx_mp.update()
        scores = approx_mp.P_
        return np.r_[scores[:self.m//2 - 1], scores, scores[-self.m//2:]]