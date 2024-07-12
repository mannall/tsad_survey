import numpy as np
from math import isclose
from typing import Callable

def normalise(scores: np.ndarray) -> np.ndarray:
    ptp = np.ptp(scores)
    if isclose(ptp, 0, abs_tol=1e-4):
        return np.zeros(len(scores))
    return (scores - np.min(scores))/ptp

def reverse_windowing(scores: np.ndarray, window_size: int) -> np.ndarray:
    # compute begin and end indices of windows
    begins = np.array([list(range(scores.shape[0]))])
    ends = begins + window_size

    # prepare target array
    unwindowed_length = (scores.shape[0] - 1) + window_size
    mapped = np.full(unwindowed_length, fill_value=np.nan)

    # only iterate over window intersections
    indices = np.unique(np.r_[begins, ends])
    for i, j in zip(indices[:-1], indices[1:]):
        window_indices = np.flatnonzero((begins <= i) & (j-1 < ends))
        mapped[i:j] = np.nanmean(scores[window_indices])

    # replace untouched indices with 0 (especially for the padding at the end)
    np.nan_to_num(mapped, copy=False)
    return mapped