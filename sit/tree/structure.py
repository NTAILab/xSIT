import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, List
from enum import Enum


@dataclass
class TreeData:
    X: np.ndarray
    y: np.ndarray | None
    group_sizes: np.ndarray
    group_ids: np.ndarray = field(init=False)
    shifts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.group_ids = np.array([
            g_id
            for g_id, gs in enumerate(self.group_sizes)
            for _ in range(gs)
        ])
        self.shifts = np.insert(np.cumsum(self.group_sizes), 0, 0)


@dataclass
class SplitInfo:
    loss: float
    feature: int
    threshold: float
    mode: Literal['any', 'all']


@dataclass
class FractionSplitInfo:
    loss: float
    feature: int
    threshold: float
    mode: Literal['any', 'all']
    fraction_threshold: float


@dataclass
class FractionActiveSplitInfo:
    loss: float
    feature: int
    threshold: float
    mode: Literal['any', 'all']
    fraction_threshold: float
    activation_threshold: int


@dataclass
class Node:
    idx: int
    value: float | None = None
    split: SplitInfo | FractionSplitInfo | FractionActiveSplitInfo | None = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None


@dataclass
class TmpSITLeaf:
    left: int
    right: int
    parent: Node | None = None
    is_left: bool = True
    depth: int = 0


class Mode(Enum):
    WRONG = -1
    ANY = 1
    ALL = 2


@dataclass
class Tree:
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    left: np.ndarray
    right: np.ndarray
    mode: np.ndarray
