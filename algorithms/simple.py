#!/usr/bin/env python3

import numpy as np
from .utils import reverse_windowing

class Simple_AD:
    def score(self, X: np.ndarray) -> np.ndarray:
        return reverse_windowing(np.std(np.diff(X, 2, axis=1), axis=1), X.shape[1])