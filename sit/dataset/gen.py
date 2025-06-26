import numpy as np
from typing import Literal
from sklearn.utils.validation import check_random_state
from functools import partial
from abc import ABC, abstractmethod
from typing import Callable
from ..mil.data import MILData


class BaseDataGen(ABC):
    @abstractmethod
    def make(self, X: np.ndarray, y: np.ndarray) -> MILData:
        pass


class FeaturesAsInstancesDataGen(BaseDataGen):
    def __init__(self, index_features: Literal['one_hot', 'int'] = 'one_hot'):
        self.index_features = index_features

    def convert_X_to_bags(self, X: np.ndarray) -> MILData:
        group_sizes = np.full(X.shape[0], X.shape[1])  # each bag of size == n_features
        if self.index_features == 'one_hot':
            index_features = np.eye(X.shape[1])
        elif self.index_features == 'int':
            index_features = np.arange(X.shape[1]).reshape((-1, 1))
        else:
            raise ValueError(f'Wrong {self.index_features=!r}')

        X_bags = np.concatenate([
            X.reshape((-1, 1)),
            np.tile(index_features, (X.shape[0], 1))
        ], axis=1)
        return MILData(X_bags, None, group_sizes)

    def make(self, X: np.ndarray, y: np.ndarray) -> MILData:
        result = self.convert_X_to_bags(X)
        result.y = y
        return result


class CountPositiveInstancesDataGen(BaseDataGen):
    def __init__(self, bag_size: int = 3,
                 clamp_label_max: int = np.inf,
                 random_state: int | None = None):
        self.bag_size = bag_size
        self.clamp_label_max = clamp_label_max
        self.random_state = random_state

    def make(self, X: np.ndarray, y: np.ndarray) -> MILData:
        rng = check_random_state(self.random_state)
        group_sizes = np.full(X.shape[0] // self.bag_size, self.bag_size, dtype=np.uint64)
        n_instances = self.bag_size * (X.shape[0] // self.bag_size)
        permutation = rng.choice(X.shape[0], size=n_instances, replace=False)
        instance_X = X[permutation]
        instance_y = y[permutation]
        y_bags = np.clip(  # count of positive elements in bag
            instance_y.reshape((-1, self.bag_size)).sum(axis=1),
            a_min=0,
            a_max=self.clamp_label_max,
        )
        data = MILData(instance_X, y_bags, group_sizes)
        return data


class SetFunctionsDataGen(BaseDataGen):
    FUNCTIONS = {
        'min': partial(np.min, axis=1),
        'max': partial(np.max, axis=1),
        'mean': partial(np.mean, axis=1),
        'sum': partial(np.sum, axis=1),
    }

    def __init__(self, bag_size: int = 3,
                 fn: Literal['min', 'max', 'mean', 'sum'] | Callable = 'mean',
                 random_state: int | None = None):
        self.bag_size = bag_size
        self.fn = fn
        self.random_state = random_state

    def get_aggregation_fn(self) -> Callable:
        if callable(self.fn):
            return self.fn

        return self.FUNCTIONS[self.fn]

    def make(self, X: np.ndarray, y: np.ndarray) -> MILData:
        rng = check_random_state(self.random_state)
        group_sizes = np.full(X.shape[0] // self.bag_size, self.bag_size, dtype=np.uint64)
        n_instances = self.bag_size * (X.shape[0] // self.bag_size)
        permutation = rng.choice(X.shape[0], size=n_instances, replace=False)
        instance_X = X[permutation]
        instance_y = y[permutation]
        agg_fn = self.get_aggregation_fn()
        y_bags = agg_fn(instance_y.reshape((-1, self.bag_size)))
        data = MILData(instance_X, y_bags, group_sizes)
        return data
