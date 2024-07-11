#!/usr/bin/env python3

import numpy as np
import pywt as wt

from typing import List
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.covariance import EmpiricalCovariance

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

def pad_series(test_ts: np.ndarray) -> np.ndarray:
    n = len(test_ts)
    exp = np.ceil(np.log2(n))
    m = int(np.power(2, exp))
    return wt.pad(test_ts, (0, m - n), "periodic")

def multilevel_dwt(test_ts: np.ndarray, wavelet: str = "haar", mode: str = "periodic", level_from: int = 0, level_to=None):
    if level_to is None:
        level_to = int(np.log2(len(test_ts)))
    ls_ = []
    as_ = []
    ds_ = []
    a = test_ts
    for i in range(level_to):
        a, d = wt.dwt(a, wavelet, mode)
        if i + 1 >= level_from:
            ls_.append(i + 1)
            as_.append(a)
            ds_.append(d)
    return np.array(ls_), as_, ds_

def reverse_windowing(test_ts: np.ndarray, window_length: int, full_length: int) -> np.ndarray:
    mapped = np.full(shape=(full_length, window_length), fill_value=0)
    mapped[:len(test_ts), 0] = test_ts

    for w in range(1, window_length):
        mapped[:, w] = np.roll(mapped[:, 0], w)

    return np.sum(mapped, axis=1)

def combine_alternating(xs, ys):
    for x, y in zip(xs, ys):
        yield x
        yield y

class DWT_MLEAD_AD:
    def __init__(self, start_level = 3, quantile_epsilon = 0.05):
        self.time_aware = True
        self.requires_training = False

        self.start_level: int = start_level
        self.quantile_epsilon: float = quantile_epsilon

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        self.n = len(test_ts)
        self.test_ts = pad_series(test_ts)
        self.m = len(self.test_ts)
        self.max_level = int(np.log2(self.m))

        self.window_sizes = np.array(
            [max(2, self.max_level - l - self.start_level + 1) for l in range(self.max_level)])

        levels, approx_coefs, detail_coefs = multilevel_dwt(self.test_ts,
            wavelet="haar",
            mode="periodic",
            level_from=self.start_level,
            # skip last level, because we cannot slide a window of size 2 over it (too small)
            level_to=self.max_level - 1,)
        
        coef_anomaly_counts = []
        for x, level in zip(combine_alternating(detail_coefs, approx_coefs), levels.repeat(2, axis=0)):
            window_size = self.window_sizes[level]
            x_view = sliding_window_view(x, window_size)

            p = self._estimate_gaussian_likelihoods(level, x_view)
            a = self._mark_anomalous_windows(p)
            xa = reverse_windowing(a, window_length=window_size, full_length=len(x))
            coef_anomaly_counts.append(xa)

        return self._push_anomaly_counts_down_to_points(coef_anomaly_counts)
    
    def _estimate_gaussian_likelihoods(self, level: float, x_view: np.ndarray) -> np.ndarray:
        # fit gaussion distribution with mean and covariance
        e_cov_est = EmpiricalCovariance(assume_centered=False)
        e_cov_est.fit(x_view)

        # compute log likelihood for each window x in x_view
        p = np.empty(shape=len(x_view))
        for i, window in enumerate(x_view):
            p[i] = e_cov_est.score(window.reshape(1, -1))

        # coefficients for time indices >n are excluded from anomaly point score
        p[int(p.size*(self.n / self.m))::] = 0
        return p

    def _mark_anomalous_windows(self, p: np.ndarray):
        z_eps = np.percentile(p, self.quantile_epsilon * 100)
        return p < z_eps

    def _push_anomaly_counts_down_to_points(self, coef_anomaly_counts: List[np.ndarray]) -> np.ndarray:
        # sum up anomaly counters of detail coefs (orig. D^l) and approx coefs (orig. C^l)
        anomaly_counts = coef_anomaly_counts[0::2]
        anomaly_counts += coef_anomaly_counts[1::2]

        # extrapolate anomaly counts to the original series' points
        counter = np.zeros(self.m)
        for ac in anomaly_counts:
            counter += ac.repeat(self.m // len(ac), axis=0)
        # delete event counters with count < 2
        counter[counter < 2] = 0
        return counter[:self.n]