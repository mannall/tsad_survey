import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from algorithms.dwt_mlead import DWT_MLEAD_AD
from algorithms.simple import Simple_AD
from algorithms.kmeans import KMeans_AD
from algorithms.knn import KNN_AD
from algorithms.lof import LOF_AD
from algorithms.ocsvm import OCSVM_AD
from algorithms.iforest import IForest_AD
from algorithms.matrix_profile import Matrix_Profile_AD
from algorithms.sos import SOS_AD
from algorithms.sampling import Sampling_AD
from algorithms.kde import KDE_AD
from algorithms.cblof import CBLOF_AD

def main():
    test_path = Path("./data/test/example.test.csv")
    train_path = Path("./data/test/example.train.csv")

    train_ts = np.genfromtxt(train_path)
    test_ts, test_labels = np.split(np.genfromtxt(test_path, delimiter=','), 2, axis=1)

    scores = []
    algs = [
        DWT_MLEAD_AD(), 
        Simple_AD(), 
        KMeans_AD(), 
        KNN_AD(), 
        LOF_AD(), 
        # OCSVM_AD(), 
        # CBLOF_AD(),
        # IForest_AD(), 
        # Matrix_Profile_AD(), 
        SOS_AD(), 
        Sampling_AD(), 
        KDE_AD()
    ]
    for alg in algs:
        if test_ts.ndim > 1: test_ts = test_ts.flatten()
        test_scores = alg.score(test_ts)
        assert len(test_scores) == len(test_ts)
        scores.append(test_scores)
    
    fig, axs = plt.subplots(len(scores)+1, 1, layout='constrained')
    
    axs[0].plot(test_ts)
    for i, score in enumerate(scores):
        axs[i+1].plot(score)

    axs[0].set_title("Test Anomalous Data")
    for i, ax in enumerate(axs[1::]):
        ax.set_title(type(algs[i]).__name__)

    plt.show()

if __name__ == "__main__":
    main()