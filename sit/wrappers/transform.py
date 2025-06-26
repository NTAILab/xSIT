from typing import Literal
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class TransformRegressorWrapper(RegressorMixin, BaseEstimator):
    def __init__(self, model: BaseEstimator, feature_transform, target_transform):
        self.model = model
        self.feature_transform = feature_transform
        self.target_transform = target_transform

    def fit(self, X, y, group_sizes):
        X = self.feature_transform.fit_transform(X)
        if y.ndim == 1:
            y = self.target_transform.fit_transform(y.reshape((-1, 1))).flatten()
        else:
            y = self.target_transform.fit_transform(y)
        self.model.fit(X, y, group_sizes)
        return self

    def predict(self, X, group_sizes):
        X = self.feature_transform.transform(X)
        preds = self.model.predict(X, group_sizes)
        if preds.ndim == 1:
            return self.target_transform.inverse_transform(preds.reshape((-1, 1))).flatten()
        else:
            return self.target_transform.inverse_transform(preds)
