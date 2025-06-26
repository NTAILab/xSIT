import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from ..mil.data import MILData
from .embedder import GradientSetInputTreeEmbedder


class GradSITRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, base_reg=None, use_agg_features: bool = False, **params):
        if base_reg is None:
            base_reg = GradientBoostingRegressor()
        self.base_reg = base_reg
        self.use_agg_features = use_agg_features
        self.params = params

    def _enrich_features(self, emb: np.ndarray, data: MILData):
        if not self.use_agg_features:
            return emb
        AGG_FUNCTIONS = (np.min, np.max, np.mean)
        group_X = np.stack([
            np.concatenate([
                fn(data.X[data.shifts[i]:data.shifts[i + 1]], axis=0)
                for fn in AGG_FUNCTIONS
            ])
            for i in range(len(data.group_sizes))
        ], axis=0)
        return group_X

    def fit(self, X, y, group_sizes):
        data = MILData(X, y, group_sizes)
        self.embedder = GradientSetInputTreeEmbedder(self.params)
        emb_train = self.embedder.fit_transform(X, y, group_sizes)

        self.bag_gbm = clone(self.base_reg)

        self.bag_gbm.fit(self._enrich_features(emb_train, data), y)
        return self

    def predict(self, X, group_sizes):
        data = MILData(X, None, group_sizes)
        return self.bag_gbm.predict(
            self._enrich_features(
                self.embedder.transform(X, group_sizes),
                data
            )
        )


class GradSetForestRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_estimators: int = 10,
                 max_samples: float = 1.0,
                 max_features: float = 1.0,
                 bootstrap: bool = True,
                 bootstrap_features: bool = True,
                 random_state: int | None = None,
                 **params):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.params = params
        self.random_state = random_state

    def fit(self, X, y, group_sizes):
        data = MILData(X, y, group_sizes)
        rng = check_random_state(self.random_state)
        seeds = rng.randint(0, np.iinfo(np.int32).max, size=self.n_estimators)
        fake_bagging = BaggingRegressor(
            DummyRegressor(),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            bootstrap_features=self.bootstrap_features,
            random_state=rng.randint(0, np.iinfo(np.int32).max),
        )
        fake_bagging.fit(X[:len(y)], y)

        self.estimators_ = []
        self.estimators_features_ = fake_bagging.estimators_features_
        for i in range(self.n_estimators):
            model = GradSITRegressor(
                **self.params,
                random_state=seeds[i],
            )
            cur = data[fake_bagging.estimators_samples_[i]]
            model.fit(cur.X[:, self.estimators_features_[i]], cur.y, cur.group_sizes)
            self.estimators_.append(model)
        return self

    def predict(self, X, group_sizes):
        predictions = 0.0
        for i, model in enumerate(self.estimators_):
            preds = model.predict(X[:, self.estimators_features_[i]], group_sizes)
            predictions += preds
        return predictions / len(self.estimators_)
