import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_distances

from typing import Optional, Union
from pathlib import Path
from joblib import dump, load

class SlidingWindowProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, window_size: int, standardize: bool = False):
        self.window_size = window_size
        if standardize:
            self.scaler = StandardScaler()
        else:
            self.scaler = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> 'SlidingWindowProcessor':
        if self.scaler:
            self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        if self.scaler:
            X = self.scaler.transform(X)
        X = X.reshape(-1)
        # the last window would have no target to predict, e.g. for n=10: [[1, 2] -> 3, ..., [8, 9] -> 10, [9, 10] -> ?]
        new_X = sliding_window_view(X, window_shape=(self.window_size))[:-1]
        new_y = np.roll(X, -self.window_size)[:-self.window_size]
        return new_X, new_y

    def transform_y(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            X = self.scaler.transform(X)
        return np.roll(X, -self.window_size)[:-self.window_size]

    def inverse_transform_y(self, y: np.ndarray, skip_inverse_scaling: bool = False) -> np.ndarray:
        result = np.full(shape=self.window_size+len(y), fill_value=np.nan)
        result[-len(y):] = y
        if not skip_inverse_scaling and self.scaler:
            result = self.scaler.inverse_transform(result)
        return result


class RForest_AD(BaseEstimator, RegressorMixin):
    def __init__(self, n_trees: int = 100, window_size: int = 50, max_features: Union[str, int] = "sqrt"):
        self.n_trees = n_trees
        self.window_size = window_size
        # can also be log2 or 1.0
        self.max_features = max_features
        self.standardize = False
        self.preprocessor = SlidingWindowProcessor(self.window_size, self.standardize)
        self.clf = RandomForestRegressor(
            n_estimators=self.n_trees,
            max_features=self.max_features,
        )

    def load(self, model_path: Path):
        self.clf = load(model_path)

    def save(self, model_path: Path):
        dump(self.clf, model_path)

    def predict(self, data: np.ndarray) -> np.ndarray:
        X, _ = self.preprocessor.transform(data)
        y_hat = self._predict_internal(X)
        return self.preprocessor.inverse_transform_y(y_hat)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def score(self, data: np.ndarray) -> np.ndarray:
        X, y = self.preprocessor.transform(data)
        y_hat = self._predict_internal(X)
        scores = paired_distances(y.reshape(-1, 1), y_hat.reshape(-1, 1)).reshape(-1)
        return self.preprocessor.inverse_transform_y(scores, skip_inverse_scaling=True)

    def train_model(self, data: np.ndarray, model_path: Path):
        X, y = self.preprocessor.fit_transform(data)
        self.clf.fit(X, y)
        self.save(model_path)

