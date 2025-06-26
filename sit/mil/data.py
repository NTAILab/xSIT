import numpy as np
from dataclasses import dataclass, field


@dataclass
class MILData:
    X: np.ndarray
    y: np.ndarray | None
    group_sizes: np.ndarray
    group_ids: np.ndarray | None = field(default=None)
    shifts: np.ndarray | None = field(default=None)

    def __post_init__(self):
        if self.group_ids is None:
            self.group_ids = np.array([
                g_id
                for g_id, gs in enumerate(self.group_sizes)
                for _ in range(gs)
            ])
        if self.shifts is None:
            self.shifts = np.insert(np.cumsum(self.group_sizes), 0, 0)

    @classmethod
    def from_bags(cls, bags, y=None) -> 'MILData':
        return cls(
            X=np.concatenate([
                np.asarray(b)
                for b in bags
            ], axis=0),
            y=y,
            group_sizes=np.array([len(b) for b in bags])
        )

    def union(self, other: 'MILData') -> 'MILData':
        return MILData(
            X=np.concatenate([self.X, other.X], axis=0),
            y=np.concatenate([self.y, other.y], axis=0),
            group_sizes=np.concatenate([self.group_sizes, other.group_sizes], axis=0),
        )

    def __getitem__(self, subset_ids) -> 'MILData':
        assert not str(subset_ids.dtype) == 'bool', 'Masks are not supported'
        group_sizes = self.group_sizes[subset_ids]
        assert self.shifts is not None
        return MILData(
            X=np.concatenate(
                [
                    self.X[self.shifts[i]:self.shifts[i + 1]]
                    for i in subset_ids
                ],
                axis=0
            ),
            y=(self.y[subset_ids] if self.y is not None else self.y),
            group_sizes=group_sizes,
        )

    def __len__(self) -> int:
        return len(self.group_sizes)
