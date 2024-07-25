import numpy as np
from numba import jit, prange


@jit(nopython=True)
def _univariate_sbd_distance(x: np.ndarray, y: np.ndarray) -> float:
    # a = correlate(x, y, method="auto")
    a = np.correlate(x, y, mode="full")
    b = np.linalg.norm(x) * np.linalg.norm(y)
    # b = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return np.abs(1.0 - np.max(a / b))


@jit(nopython=True, parallel=True)
def _standardize(X: np.ndarray) -> np.ndarray:
    for i in prange(X.shape[0]):
        mean = np.mean(X[i, :])
        std = np.std(X[i, :])
        X[i, :] = (X[i, :] - mean)/std
    return X


@jit(nopython=True, parallel=True)
def fast_sbd_matrix(X: np.ndarray) -> np.ndarray:
    n_cases = X.shape[0]
    distances = np.zeros((n_cases, n_cases))

    # (n_cases, n_timepoints)
    X = _standardize(X.astype(np.float64))

    for i in prange(n_cases):
        for j in range(i + 1, n_cases):
            distances[i, j] = _univariate_sbd_distance(X[i], X[j])
            distances[j, i] = distances[i, j]

    return distances


def _test_fast_sbd_matrix():
    from aeon.distances import sbd_pairwise_distance
    X = np.random.uniform(size=(100, 100))
    assert np.isclose(fast_sbd_matrix(X) - sbd_pairwise_distance(X), np.zeros((100, 100))).all().all()


if __name__ == "__main__":
    _test_fast_sbd_matrix()