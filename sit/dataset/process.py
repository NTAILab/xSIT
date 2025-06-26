import numpy as np
from sklearn.utils.validation import check_random_state
from abc import ABC, abstractmethod
from ..mil.data import MILData


class BaseDataProcess(ABC):
    @abstractmethod
    def process(self, data: MILData) -> MILData:
        pass


class BalanceSubsampleDataProcess(BaseDataProcess):
    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def process(self, data: MILData) -> MILData:
        rng = check_random_state(self.random_state)
        unique_y, n_unique = np.unique(data.y, return_counts=True)
        n_to_extract = np.min(n_unique)
        ids = np.concatenate([
            rng.choice(np.argwhere(data.y == cur_y).ravel(), size=n_to_extract)
            for cur_y in unique_y
        ])
        return data[ids]


class ThinOutGroupsDataProcess(BaseDataProcess):
    def __init__(self, subset_size: int | float = 0.8,
                 random_state: int | None = None):
        self.subset_size = subset_size
        self.random_state = random_state

    def clean_empty_groups(self, data: MILData) -> MILData:
        group_mask = (data.group_sizes != 0)
        return MILData(
            data.X,
            data.y[group_mask] if data.y is not None else None,
            data.group_sizes[group_mask]
        )

    def apply_instance_mask(self, data: MILData, mask: np.ndarray) -> MILData:
        active_elements_num = np.insert(np.cumsum(mask), 0, 0)
        return self.clean_empty_groups(
            MILData(
                data.X[mask],
                data.y,
                group_sizes=active_elements_num[data.shifts[1:]] - active_elements_num[data.shifts[:-1]],
            )
        )

    def prepare_instance_mask(self, data: MILData, rng: np.random.RandomState) -> np.ndarray:
        mask = np.zeros(data.X.shape[0], dtype=bool)
        subset_size = (
            self.subset_size if isinstance(self.subset_size, int)
            else round(self.subset_size * mask.shape[0])
        )
        mask[rng.choice(data.X.shape[0], size=subset_size, replace=False)] = True
        return mask

    def process(self, data: MILData) -> MILData:
        rng = check_random_state(self.random_state)
        mask = self.prepare_instance_mask(data, rng)
        return self.apply_instance_mask(data, mask)
