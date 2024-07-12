#!/usr/bin/env python3

import numpy as np
from .utils import reverse_windowing
from numpy.lib.stride_tricks import sliding_window_view

from pyod.models.base import BaseDetector
from pyod.models.kde import KDE
from pyod.models.sampling import Sampling
from pyod.models.mcd import MCD
from pyod.models.sos import SOS
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cof import COF
from pyod.models.sod import SOD

from typing import Dict, Any, Type

class PyODAD:
    def __init__(self, model: Type[BaseDetector], window_size: int = 64, **kwargs):
        self.window_size = window_size
        self.model = model(**kwargs)

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        self.model.fit(sliding_window_view(test_ts, self.window_size))
        return reverse_windowing(self.model.decision_scores_, self.window_size)


class AutoEncoderAD(PyODAD):
    def __init__(self, window_size, contamination: float, epoch_num: int = 15, lr: float = 5e-4, dropout_rate: float = 0.1, verbose: bool = False):
        super().__init__(AutoEncoder, window_size=window_size, contamination=contamination, epoch_num=epoch_num, lr=lr, dropout_rate=dropout_rate, verbose=verbose)


class CBLOFAD(PyODAD):
    def __init__(self, window_size, contamination: float, n_clusters: int = 10, alpha: float = 0.9, beta: float = 5):
        super().__init__(CBLOF, window_size=window_size, contamination=contamination, n_clusters=n_clusters, alpha=alpha, beta=beta)


class COFAD(PyODAD):
    def __init__(self, window_size, contamination: float, n_neighbors: int = 20):
        super().__init__(COF, window_size=window_size, contamination=contamination, n_neighbors=n_neighbors)


class IForestAD(PyODAD):
    def __init__(self, window_size, contamination: float, n_estimators: int = 100, max_samples: str = "auto"):
        super().__init__(IForest, window_size=window_size, contamination=contamination, n_estimators=n_estimators, max_samples=max_samples)


class KDEAD(PyODAD):
    def __init__(self, window_size, contamination: float, bandwidth: int = 1, algorithm: str = "auto", leaf_size: int = 30):
        super().__init__(KDE, window_size=window_size, contamination=contamination, bandwidth=bandwidth, algorithm=algorithm, leaf_size=leaf_size)


class KNNAD(PyODAD):
    def __init__(self, window_size, contamination: float, n_neighbors: int = 65, leaf_size: int = 30):
        super().__init__(KNN, window_size=window_size, contamination=contamination, n_neighbors=n_neighbors, leaf_size=leaf_size)


class LOFAD(PyODAD):
    def __init__(self, window_size, contamination: float, n_neighbors: int = 50, leaf_size: int = 30):
        super().__init__(LOF, window_size=window_size, contamination=contamination, n_neighbors=n_neighbors, leaf_size=leaf_size)


class MCDAD(PyODAD):
    def __init__(self, window_size, contamination: float):
        super().__init__(MCD, window_size=window_size, contamination=contamination)


class SamplingAD(PyODAD):
    def __init__(self, window_size, contamination: float, subset_size: float = 0.2):
        super().__init__(Sampling, window_size=window_size, contamination=contamination, subset_size=subset_size)


class SODAD(PyODAD):
    def __init__(self, window_size, contamination: float, n_neighbors: int = 20, ref_set: int = 10, alpha: float = 0.8):
        super().__init__(SOD, window_size=window_size, contamination=contamination, n_neighbors=n_neighbors, ref_set=ref_set, alpha=alpha)


class SOSAD(PyODAD):
    def __init__(self, window_size, contamination: float, perplexity: int = 32):
        super().__init__(SOS, window_size=window_size, contamination=contamination, perplexity=perplexity)