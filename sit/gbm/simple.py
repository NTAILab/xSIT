import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_random_state
from ..tree.any_all import AnyAllSetInputTree


class SimpleSetGBM(RegressorMixin, BaseEstimator):
    def __init__(self, n_estimators: int = 100,
                 lr: float = 0.1,
                 max_depth: int = 3,
                 min_samples_leaf: int = 1,
                 extra_rand: bool = False,
                 random_state: int | None = None,
                 tree_class=AnyAllSetInputTree):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.extra_rand = extra_rand
        self.random_state = random_state
        self.tree_class = tree_class

    def fit(self, X, y, group_sizes):
        rng = check_random_state(self.random_state)
        seeds = rng.randint(0, np.iinfo(np.int32).max, size=self.n_estimators)
        self.estimators_ = []
        self.mean_ = y.mean(axis=0)
        cumulative_prediction = 0.0 + self.mean_

        for i in range(self.n_estimators):
            model = self.tree_class(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                extra_rand=self.extra_rand,
                random_state=seeds[i],
            )
            model.fit(X, y - cumulative_prediction, group_sizes)
            self.estimators_.append(model)
            if i < self.n_estimators - 1:
                preds = model.predict(X, group_sizes)
                cumulative_prediction += preds * self.lr
        return self

    def predict(self, X, group_sizes):
        cumulative_prediction = 0.0 + self.mean_
        for i, model in enumerate(self.estimators_):
            preds = model.predict(X, group_sizes)
            cumulative_prediction += preds * self.lr
        return cumulative_prediction
