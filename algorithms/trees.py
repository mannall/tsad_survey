import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from typing import Union
from pathlib import Path
from joblib import load, dump

class RForestAD:
    def __init__(self, window_size: int = 32, n_diff: int = 1, n_trees: int = 100, max_features: Union[str, int] = "sqrt"):
        self.window_size = window_size
        # make time series stationary by differencing
        self.n_diff = n_diff
        self.n_trees = n_trees
        # can also be log2 or 1.0
        self.max_features = max_features

        self.random_forest = RandomForestRegressor(
            n_estimators=self.n_trees,
            max_features=self.max_features,
        )

    def _transform_ts(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windowed_ts = sliding_window_view(np.diff(ts, n=self.n_diff), self.window_size+1)
        return windowed_ts[:, :self.window_size], np.ravel(windowed_ts[:, self.window_size:])

    def score(self, test_ts: np.ndarray) -> np.ndarray:
        X_test, y_test = self._transform_ts(test_ts)
        y_hat = self.random_forest.predict(X_test)
        scores = (y_test - y_hat)**2
        return np.r_[scores[:self.window_size+self.n_diff], scores]

    def train_model(self, train_ts: np.ndarray):
        X_train, y_train = self._transform_ts(train_ts)
        self.random_forest.fit(X_train, y_train)

    def load(self, model_path: Path):
        self.random_forest = load(model_path)

    def save(self, model_path: Path):
        dump(self.random_forest, model_path)


class GradientBoostAD:
    def __init__(self, window_size: int = 32, n_diff: int = 1, learning_rate: float = 0.05, max_iter: int = 100, min_samples_leaf: int = 15):
        self.window_size = window_size
        self.n_diff = n_diff
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_samples_leaf = min_samples_leaf
        self.gradient_boost = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
        )
    
    def _transform_ts(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        windowed_ts = sliding_window_view(np.diff(ts, n=self.n_diff), self.window_size+1)
        return windowed_ts[:, :self.window_size], np.ravel(windowed_ts[:, self.window_size:])
    
    def score(self, test_ts: np.ndarray) -> np.ndarray:
        X_test, y_test = self._transform_ts(test_ts)
        y_hat = self.gradient_boost.predict(X_test)
        scores = (y_test - y_hat)**2
        return np.r_[scores[:self.window_size+self.n_diff], scores]

    def train_model(self, train_ts: np.ndarray):
        X_train, y_train = self._transform_ts(train_ts)
        self.gradient_boost.fit(X_train, y_train)
    
    def load(self, model_path: Path):
        self.gradient_boost = load(model_path)

    def save(self, model_path: Path):
        dump(self.gradient_boost, model_path)
