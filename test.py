import os
import datetime
import shutil

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from algorithms.dwt_mlead import DWTMLEADAD
from algorithms.simple import SimpleAD
from algorithms.kmeans import KMeansAD
from algorithms.matrix_profile import MatrixProfileAD
from algorithms.pyod import AutoEncoderAD, CBLOFAD, COFAD, IForestAD, KDEAD, KNNAD, LOFAD, SamplingAD, SODAD, SOSAD
from algorithms.machine_learning import MLPAD, LSTMAD, TransformerAD
from algorithms.trees import RForestAD, GradientBoostAD
from algorithms.donut import DonutAD
from algorithms.omni_anomaly import OmniAnomalyAD
from algorithms.deepant import DeepAnTCNNAD

from algorithms.utils import normalise

def make_output_directories():
    output_dir = Path(f"./results/")
    output_dir.mkdir(parents=False, exist_ok=True)

    models_dir = output_dir / "models"
    scores_dir = output_dir / "scores"

    models_dir.mkdir(parents=False, exist_ok=True)
    scores_dir.mkdir(parents=False, exist_ok=True)

    return models_dir, scores_dir

def plot_scores(test_ts: np.ndarray, scores: dict):
    fig, axs = plt.subplots(5, 5, figsize=(20, 20), layout='constrained')

    ax_ts = fig.add_subplot(511)
    ax_ts.plot(test_ts)
    ax_ts.set_title('Test Time Series')

    for ax in axs[0]:
        ax.remove()

    # Plot algorithm scores
    for (alg_name, alg_scores), ax in zip(scores.items(), axs[1:].flatten()):
        ax.plot(alg_scores)
        ax.set_title(alg_name)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def main():
    test_path = Path("./data/raw/test/example.test.csv")
    train_path = Path("./data/raw/test/example.train.csv")

    train_ts = np.genfromtxt(train_path)
    test_ts, test_labels = np.split(np.genfromtxt(test_path, delimiter=','), 2, axis=1)
    test_ts, train_ts = test_ts.flatten(), train_ts.flatten()

    models_dir, scores_dir = make_output_directories()

    contamination = 0.01
    window_size = 32

    scores = {}
    unsupervised_algs = [
        # DWTMLEADAD(), 
        # SimpleAD(), 
        # KMeansAD(window_size=window_size), 
        # MatrixProfileAD(m=window_size), 
        # AutoEncoderAD(window_size, contamination), 
        # CBLOFAD(window_size, contamination), 
        # COFAD(window_size, contamination), 
        # IForestAD(window_size, contamination), 
        # KDEAD(window_size, contamination), 
        # KNNAD(window_size, contamination), 
        # LOFAD(window_size, contamination), 
        # SamplingAD(window_size, contamination), 
        # SODAD(window_size, contamination), 
        # SOSAD(window_size, contamination), 
    ]
    semi_supervised_algs = [
        # MLPAD(window_size=window_size), 
        # LSTMAD(window_size=window_size, hidden_size=64, num_layers=2), 
        # TransformerAD(window_size=window_size, hidden_size=64, hidden_feedforward=128, num_heads=4, num_layers=2), 
        # RForestAD(window_size=window_size), 
        # GradientBoostAD(window_size=window_size)
        # DonutAD(),
        # OmniAnomalyAD(),
        DeepAnTCNNAD(window_size=window_size)
    ]

    scores = {}
    for alg in semi_supervised_algs:
        alg.train_model(train_ts)
        alg.save(models_dir / type(alg).__name__)
    for alg in (unsupervised_algs + semi_supervised_algs):
        if alg in semi_supervised_algs:
            alg.load(models_dir / type(alg).__name__)
        test_scores = alg.score(test_ts)

        print(len(test_scores))
        # assert len(test_scores) == len(test_ts)
        scores[type(alg).__name__] = normalise(test_scores)

    plot_scores(test_ts, scores)

if __name__ == "__main__":
    main()