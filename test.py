import datetime
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib.pyplot as plt

from algorithms.dwt_mlead import DWT_MLEAD_AD
from algorithms.simple import Simple_AD
from algorithms.kmeans import KMeans_AD
from algorithms.matrix_profile import Matrix_Profile_AD

from algorithms.rforest import RForest_AD

from algorithms.pyod import pyod_algs

from algorithms.utils import normalise

def main():
    test_path = Path("./data/raw/test/example.test.csv")
    train_path = Path("./data/raw/test/example.train.csv")

    train_ts = np.genfromtxt(train_path)
    test_ts, test_labels = np.split(np.genfromtxt(test_path, delimiter=','), 2, axis=1)

    test_ts = test_ts.flatten()
    train_ts = train_ts.flatten()

    # current_time = datetime.datetime.now().replace(microsecond=0).isoformat()
    # output_dir = Path(f"./results/{current_time}")
    # output_dir.mkdir(parents=False, exist_ok=False)

    # models_dir = output_dir / "models"
    # scores_dir = output_dir / "scores"

    # models_dir.mkdir(parents=False, exist_ok=False)
    # scores_dir.mkdir(parents=False, exist_ok=False)

    contamination = 0.02
    window_shape = 32

    scores = {}
    algs = {
        "KMeans": KMeans_AD(), 
        "Simple": Simple_AD(),
        "DWT_MLEAD": DWT_MLEAD_AD(),
        "Matrix_Profile": Matrix_Profile_AD(),
    } | pyod_algs(contamination)

    X = sliding_window_view(test_ts, window_shape)
    for alg_name, alg in algs.items():
        if hasattr(alg, 'time_aware'):
            test_scores = alg.score(test_ts)
        else:
            test_scores = alg.score(X)
        
        assert len(test_scores) == len(test_ts)
        assert not np.isnan(test_scores).any()
        scores[alg_name] = normalise(test_scores)

    fig, axs = plt.subplots(len(scores)+1, 1, layout='constrained')
    axs[0].plot(test_ts)
    for i, (alg_name, alg_scores) in enumerate(scores.items()):
        axs[i+1].set_title(alg_name)
        axs[i+1].plot(alg_scores)

    plt.show()

if __name__ == "__main__":
    main()