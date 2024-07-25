# convert data to parquet
# generate index

import numba
import numpy as np
import pandas as pd

from math import ceil

from pathlib import Path
from typing import Optional, Union
from enum import Enum

from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.tsa.stattools import pacf

# from aeon.distances import euclidean_pairwise_distance
from sklearn.metrics import pairwise_distances
from fast_sbd import fast_sbd_matrix


def ds_index(base_path: Path = Path("./data/raw")) -> dict[str, list]:
    return {ds_path.name: ts_pairs(ds_path) for ds_path in sorted(base_path.iterdir())}


def ts_pairs(ds_path: Path) -> list[tuple[Path, Optional[Path]]]:
    pairs = []
    for test_file in list(ds_path.glob('*test*')):
        train_file = test_file.with_name(test_file.name.replace('test', 'train'))
        if train_file.exists(): pairs.append((test_file, train_file))
        else: pairs.append((test_file, None))
    return pairs


def validate_test(test_df: pd.DataFrame) -> int:
    if test_df.isnull().any().any(): return 1
    if test_df['is_anomaly'].sum() == 0: return 2
    return 0


def validate_train(train_df: pd.DataFrame) -> int:
    if train_df.isnull().any().any(): return 3
    if train_df['is_anomaly'].sum() != 0: return 4
    return 0


def load_pair(test_path: Path, train_path: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    test_df = pd.read_csv(test_path, names=['value', 'is_anomaly'])
    train_df = pd.read_csv(train_path, names=['value', 'is_anomaly']) if train_path else None
    is_filtered = max(validate_test(test_df), validate_train(train_df) if train_path else 0)
    return test_df, train_df, is_filtered


def relative_contrast(x: np.ndarray, w: int) -> Union[float, np.float64]:
    stride = max(ceil(x.size//2500), 1)
    X = sliding_window_view(x, w)[::stride, :]
    # dist_matrix = pairwise_distances(X, n_jobs=4, metric='cosine')
    dist_matrix = fast_sbd_matrix(X)
    if np.count_nonzero(np.isnan(dist_matrix))/(dist_matrix.size) < 0.5:
        # remove diagonal entries (all zeros)
        dist_matrix = dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)].reshape(dist_matrix.shape[0],-1)
        d_means = np.nanmean(dist_matrix, axis=0)
        d_mins = np.nanmin(dist_matrix, axis=0)
        r_c = np.mean(d_means)/np.mean(d_mins)
        return r_c
    return np.nan


def main():
    index = ds_index()
    for ds, ts_paths in index.items():
        for (test_path, train_path) in ts_paths:
            test_df, train_df, is_filtered = load_pair(test_path, train_path)

            # print(ds, is_filtered, test_path, train_path)
            if is_filtered: continue

            test_values, test_labels = test_df['value'].values, test_df['is_anomaly'].values
            # if train_path:
            #     train_values, train_labels = train_df['value'].values, train_df['is_anomaly'].values

            #     ts_len = min(len(test_values), len(train_values))

            #     test_values = test_values[:ts_len]
            #     train_values = train_values[:ts_len]

            print(ds, test_path.name, test_values.shape[0], relative_contrast(test_values, 64))

            # print(ds, is_filtered, euclidean_distance(test_values, train_values), dtw_distance(test_values, train_values), sbd_distance(test_values, train_values))

            # compute euclidean distance between each pair of test, train

main()

# ds_name, ts_name, is_filtered, test_path, train_path, stats...