"""Общие компоненты предобработки для обучения и прогнозирования."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Ограничивает признаки процентилями, вычисленными во время fit."""

    def __init__(self, lower_percentile=1.0, upper_percentile=99.0):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile

    def fit(self, X, y=None):
        X_checked = check_array(X, ensure_2d=True, dtype=float)
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError("Процентили должны удовлетворять 0 <= lower < upper <= 100")

        self.lower_bounds_ = np.percentile(X_checked, self.lower_percentile, axis=0)
        self.upper_bounds_ = np.percentile(X_checked, self.upper_percentile, axis=0)
        self.n_features_in_ = X_checked.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, ("lower_bounds_", "upper_bounds_"))
        X_checked = check_array(X, ensure_2d=True, dtype=float)
        return np.clip(X_checked, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)
