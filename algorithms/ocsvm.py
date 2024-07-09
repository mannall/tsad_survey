#!/usr/bin/env python3

import numpy as np
from sklearn.svm import OneClassSVM
from numpy.lib.stride_tricks import sliding_window_view

class OCSVM_AD:
    def __init__(self, window_size: int = 4) -> None:
        super().__init__()
        self.window_size = window_size
        self.ocsvm = OneClassSVM(
            kernel="rbf",   
            tol=1e-3,
            nu=0.5,
        )
        
    def score(self, test_ts):
        self.ocsvm.fit(sliding_window_view(test_ts, window_shape=self.window_size))
        scores = -self.ocsvm.decision_function(sliding_window_view(test_ts, window_shape=self.window_size))
        return np.r_[scores[:self.window_size//2 - 1], scores, scores[-self.window_size//2:]]