#!/usr/bin/env python3

import numpy as np
from .utils import reverse_windowing

from pyod.models.base import BaseDetector
from pyod.models.kde import KDE
from pyod.models.ocsvm import OCSVM
from pyod.models.sampling import Sampling
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
    def __init__(self, model: Type[BaseDetector], **kwargs):
        self.model = model(**kwargs)

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.fit(X)
        return reverse_windowing(self.model.decision_scores_, X.shape[1])

def pyod_algs(contamination: float) -> Dict[str, PyODAD]:
    models: Dict[str, Dict[str, Any]] = {
        "AutoEncoder": {"model": AutoEncoder, "epoch_num": 15, "lr": 5e-4, "dropout_rate": 0.1, "verbose": False},
        "CBLOF": {"model": CBLOF, "n_clusters": 10, "alpha": 0.9, "beta": 5},
        "COF": {"model": COF, "n_neighbors": 20},
        "IForest": {"model": IForest, "n_estimators": 100, "max_samples": "auto"},
        "KDE": {"model": KDE, "bandwidth": 1, "algorithm": "auto", "leaf_size": 30},
        "KNN": {"model": KNN, "n_neighbors": 65, "leaf_size": 30},
        "LOF": {"model": LOF, "n_neighbors": 50, "leaf_size": 30},
        "OCSVM": {"model": OCSVM, "kernel": "rbf", "tol": 1e-3, "nu": 0.5},
        "Sampling": {"model": Sampling, "subset_size": 0.2},
        "SOD": {"model": SOD, "n_neighbors": 20, "ref_set": 10, "alpha": 0.8},
        "SOS": {"model": SOS, "perplexity": 32},
    }
    return {name: PyODAD(contamination=contamination, **models[name]) for name in models}