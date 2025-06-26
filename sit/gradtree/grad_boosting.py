import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_random_state
from gradient_growing_trees.tree import GradientGrowingTreeRegressor
from .set_tree_nn import SetTreeNN
from ..mil.data import MILData


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


class GradBoostingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, **params):
        if 'loss_fn' not in params:
            params['loss_fn'] = 'bce'
        self.params = params

    def fit(self, X, y, group_sizes):
        mil_train = MILData(X, y, group_sizes)
        params = self.params

        self.stnn = SetTreeNN(
            base_estimator=GradientGrowingTreeRegressor(
                lam_2=params['lam_2'],
                lr=params['lr'],
                splitter=params['splitter'],
                max_depth=params['max_depth'],
                random_state=1,
            ),
            n_estimators=params['n_estimators'],
            lam_2=params['lam_2'],
            lr=params['lr'],
            tree_loss_on_sample_ids=False,
            n_update_iterations=params['n_update_iterations'],
        ).set_embedding_size(params['embedding_size'])\
         .set_nn_lr(params['nn_lr'])\
         .set_nn_num_heads(params['nn_num_heads'])\
         .set_nn_steps(params['nn_steps'])\
         .set_dropout(params['dropout'])
        if 'loss_fn' in params:
            self.stnn.set_loss_fn(params['loss_fn'])

        self.stnn.enable_postiter_nn = False
        self.stnn.fit(
            mil_train.X,
            mil_train.y.reshape((-1, 1)) if mil_train.y.ndim == 1 else mil_train.y,
            X_nn=mil_train.group_ids.reshape((-1, 1)),
            # eval_XyXnn=(mil_test.X, mil_test.y.reshape((-1, 1)), mil_test.group_ids.reshape((-1, 1)))
        )
        return self

    def predict_proba(self, X, group_sizes):
        mil_train = MILData(X, None, group_sizes)
        proba = sigmoid(self.stnn.predict(X=mil_train.X, X_nn=mil_train.group_ids.reshape((-1, 1))).numpy())
        return np.concatenate([1.0 - proba, proba], axis=1)

    def predict(self, X, group_sizes):
        return np.argmax(self.predict_proba(X, group_sizes), axis=1)


class GradBoostingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, group_sizes):
        mil_train = MILData(X, y, group_sizes)
        params = self.params

        self.stnn = SetTreeNN(
            base_estimator=GradientGrowingTreeRegressor(
                lam_2=params['lam_2'],
                lr=params['lr'],
                splitter=params['splitter'],
                max_depth=params['max_depth'],
                random_state=1,
            ),
            n_estimators=params['n_estimators'],
            lam_2=params['lam_2'],
            lr=params['lr'],
            tree_loss_on_sample_ids=False,
            n_update_iterations=params['n_update_iterations'],
        ).set_embedding_size(params['embedding_size'])\
         .set_nn_lr(params['nn_lr'])\
         .set_nn_num_heads(params['nn_num_heads'])\
         .set_nn_steps(params['nn_steps'])\
         .set_dropout(params['dropout'])
        if 'loss_fn' in params:
            self.stnn.set_loss_fn(params['loss_fn'])

        self.stnn.enable_postiter_nn = False
        self.stnn.fit(
            mil_train.X,
            mil_train.y.reshape((-1, 1)) if mil_train.y.ndim == 1 else mil_train.y,
            X_nn=mil_train.group_ids.reshape((-1, 1)),
            # eval_XyXnn=(mil_test.X, mil_test.y.reshape((-1, 1)), mil_test.group_ids.reshape((-1, 1)))
        )
        return self

    def predict(self, X, group_sizes):
        mil_test = MILData(X, None, group_sizes)
        return self.stnn.predict(X=mil_test.X, X_nn=mil_test.group_ids.reshape((-1, 1))).numpy()
