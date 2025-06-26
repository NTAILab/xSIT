import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyRegressor
from sklearn.utils.validation import check_random_state
from sklearn.ensemble import BaggingRegressor, GradientBoostingClassifier
from ..mil.data import MILData
from .embedder import GradientSetInputTreeEmbedder


class GradSITClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, base_clf=None, use_agg_features: bool = False, **params):
        if 'loss_fn' not in params:
            params['loss_fn'] = 'bce'
        if base_clf is None:
            base_clf = GradientBoostingClassifier()
        self.base_clf = base_clf
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

        self.bag_gbm = clone(self.base_clf)

        self.bag_gbm.fit(self._enrich_features(emb_train, data), y)
        return self

    def predict_proba(self, X, group_sizes):
        data = MILData(X, None, group_sizes)
        return self.bag_gbm.predict_proba(
            self._enrich_features(
                self.embedder.transform(X, group_sizes),
                data
            )
        )

    def predict(self, X, group_sizes):
        data = MILData(X, None, group_sizes)
        return self.bag_gbm.predict(
            self._enrich_features(
                self.embedder.transform(X, group_sizes),
                data
            )
        )


class GradSetForestClassifier(ClassifierMixin, BaseEstimator):
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
            model = GradSITClassifier(
                **self.params,
                random_state=seeds[i],
            )
            cur = data[fake_bagging.estimators_samples_[i]]
            model.fit(cur.X[:, self.estimators_features_[i]], cur.y, cur.group_sizes)
            self.estimators_.append(model)
        return self

    def predict_proba(self, X, group_sizes):
        cumulative_prediction = 0.0
        for i, model in enumerate(self.estimators_):
            preds = model.predict_proba(X[:, self.estimators_features_[i]], group_sizes)
            cumulative_prediction += preds
        cumulative_prediction /= len(self.estimators_)
        return cumulative_prediction

    def predict(self, X, group_sizes):
        return np.argmax(self.predict_proba(X, group_sizes), axis=1)
