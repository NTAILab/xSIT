from itertools import product
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_random_state
from dataclasses import dataclass, field

from numba import njit

from typing import Literal, Optional, List
from enum import Enum

from .preprocessing import FeaturePreprocessingT, make_preprocessing_ft
from .structure import *


@njit
def find_best_split_for_feature_fraction_active_le(x, sort_ids, group_ids, group_y, raw_point_indices, q,
                                        group_sizes,
                                        total_sum, total_count,
                                        init_k, init_max_k,
                                        fraction_threshold,
                                        activation_threshold,
                                        point_activations):
    left_y_sum = 0.0
    left_y_count = 0

    best_loss = np.inf
    best_t = 0

    k = 0
    # max_k = len(sort_ids)
    # k = init_k
    max_k = init_max_k
    while k < max_k:
        # skip the same elements
        t = k
        while t < len(sort_ids) - 1 and x[sort_ids[t]] == x[sort_ids[t + 1]]:
            t += 1
        # now x elements with indices sort_ids[k], ..., sort_ids[t] are the same
        # perform a step:
        for i in range(k, t + 1):
            idx = raw_point_indices[sort_ids[i]]
            group = group_ids[idx]
            if q[group] == -1:  # -1 is a special label for already counted groups
                continue
            # TODO: check
            # NOTE: new
            if point_activations[sort_ids[i]] < activation_threshold:
                continue

            q[group] += 1
            current_fraction = float(q[group]) / group_sizes[group]
            if current_fraction >= fraction_threshold:
                q[group] = -1  # set the special label
                left_y_sum += group_y[group]
                left_y_count += 1

        right_y_sum = total_sum - left_y_sum
        right_y_count = total_count - left_y_count

        if k >= init_k and left_y_count > 0 and right_y_count > 0:
            cur_loss = -left_y_sum * left_y_sum / left_y_count - right_y_sum * right_y_sum / right_y_count
            # print(k, t, cur_loss, left_y_sum / left_y_count, right_y_sum / right_y_count)
            if cur_loss <= best_loss:
                best_loss = cur_loss
                best_t = t

        k = t + 1

    return best_t, best_loss


# @njit
# def find_best_split_for_feature_fraction_active_gt(x, sort_ids, group_ids, group_y, raw_point_indices, q,
#                                            group_sizes,
#                                            total_sum, total_count,
#                                            init_k, init_min_k,
#                                            fraction_threshold):
#     # TODO: check this implementation
#     right_y_sum = 0.0
#     right_y_count = 0

#     best_loss = np.inf
#     best_t = 0

#     k = len(sort_ids) - 1
#     # min_k = 0
#     # k = init_k
#     min_k = init_min_k
#     while k >= min_k:
#         # skip the same elements
#         t = k
#         while t > 0 and x[sort_ids[t]] == x[sort_ids[t - 1]]:
#             t -= 1
#         # now x elements with indices sort_ids[k], ..., sort_ids[t] are the same
#         # perform a step:
#         for i in range(k, t - 1, -1):
#             idx = raw_point_indices[sort_ids[i]]
#             group = group_ids[idx]
#             if q[group] == -1:  # -1 is a special label for already counted groups
#                 continue
#             q[group] += 1
#             current_fraction = float(q[group]) / group_sizes[group]
#             if current_fraction >= fraction_threshold:
#                 q[group] = -1  # set the special label
#                 right_y_sum += group_y[group]
#                 right_y_count += 1

#         left_y_sum = total_sum - right_y_sum
#         left_y_count = total_count - right_y_count

#         if k <= init_k and left_y_count > 0 and right_y_count > 0:
#             cur_loss = -left_y_sum * left_y_sum / left_y_count - right_y_sum * right_y_sum / right_y_count
#             # print(k, t, cur_loss, left_y_sum / left_y_count, right_y_sum / right_y_count)
#             if cur_loss <= best_loss:
#                 best_loss = cur_loss
#                 best_t = t - 1

#         k = t - 1

#     return best_t, best_loss


class ThresholdMode(Enum):
    WRONG = -1
    ANY = 1
    ALL = 2


@dataclass
class FractionActiveTree:
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    left: np.ndarray
    right: np.ndarray
    mode: np.ndarray
    fraction_threshold: np.ndarray
    activation_threshold: np.ndarray


class FractionActiveTreeBuilder:
    def __init__(self, max_depth: int | None = None, min_samples_leaf: int = 2, extra_rand: bool = False,
                 fraction_grid_size: int = 10,
                 max_activations: int = 1,
                 random_state: np.random.RandomState | None = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.extra_rand = extra_rand
        self.fraction_grid_size = fraction_grid_size
        self.max_activations = max_activations
        self.rng = check_random_state(random_state)

    def feature_iterator(self, data: TreeData):
        # TODO: feature subsampling, etc.
        n_features = data.X.shape[1]
        for i in range(n_features):
            yield i

    def find_best_split(self, data: TreeData, point_indices: np.ndarray, point_activations_count: np.ndarray,
                       fraction_grid_size: int,
                       max_activations: int):
        assert data.y is not None
        n_groups = data.y.shape[0]
        # q_buffer = np.zeros(n_groups, dtype=np.uint8)
        q_buffer = np.zeros(n_groups, dtype=np.int64)
        current_X = data.X[point_indices]
        current_point_activations = point_activations_count[point_indices]

        # calculate total_sum, total_count
        current_group_ids = np.unique(data.group_ids[point_indices])  # TODO: remove `unique`
        total_sum = data.y[current_group_ids].sum()
        total_count = len(current_group_ids)

        best_loss = np.inf
        best_feature = 0
        best_threshold = 0.0
        best_mode = 'any'
        best_fraction_threshold = -1.0
        best_activation_threshold = 0

        fraction_threshold_grid = np.linspace(0.0, 1.0, fraction_grid_size)
        activation_threshold_grid = list(range(max_activations + 1))
        # activation_threshold_grid = [0, 1]

        for feature_id in self.feature_iterator(data):
            sort_ids = np.argsort(current_X[:, feature_id])
            # forward traversal
            for fraction_threshold, activation_threshold in product(fraction_threshold_grid, activation_threshold_grid):
                q_buffer.fill(0)
                init_k = 0
                init_max_k = len(sort_ids)
                if self.extra_rand:
                    init_k = self.rng.randint(0, len(sort_ids))
                    init_max_k = init_k + 1

                split_any_le, loss_any_le = find_best_split_for_feature_fraction_active_le(
                    data.X[:, feature_id],
                    sort_ids,
                    data.group_ids,
                    data.y,
                    point_indices,  # non-sorted (!)
                    q_buffer,
                    data.group_sizes,
                    total_sum,
                    total_count,
                    init_k,
                    init_max_k,
                    fraction_threshold,
                    activation_threshold,
                    current_point_activations,
                )
                if loss_any_le <= best_loss:
                    best_loss = loss_any_le
                    best_feature = feature_id
                    # print('sort_ids:', sort_ids, 'point_ids:', point_indices)
                    # best_threshold = data.X[point_indices[sort_ids[split_any_le]], feature_id]
                    best_threshold = current_X[sort_ids[split_any_le], feature_id]
                    best_mode = 'any'
                    best_fraction_threshold = fraction_threshold
                    best_activation_threshold = activation_threshold

            continue  # TODO: remove
            # reverse traversal
            for fraction_threshold in fraction_threshold_grid:
                q_buffer.fill(0)
                init_k = len(sort_ids) - 1
                init_min_k = 0
                if self.extra_rand and len(sort_ids) > 1:
                    init_k = self.rng.randint(1, len(sort_ids))
                    init_min_k = init_k - 1

                split_any_gt, loss_any_gt = find_best_split_for_feature_fraction_gt(
                    data.X[:, feature_id],
                    sort_ids,
                    data.group_ids,
                    data.y,
                    point_indices,  # non-sorted (!)
                    q_buffer,
                    data.group_sizes,
                    total_sum,
                    total_count,
                    init_k,
                    init_min_k,
                    fraction_threshold,
                )
                split_all_le, loss_all_le = split_any_gt, loss_any_gt
                if loss_all_le <= best_loss:  # TODO: change to `<` (prefer "any")
                    best_loss = loss_all_le
                    best_feature = feature_id
                    # print('   split_all_le:', split_all_le)
                    # best_threshold = data.X[point_indices[sort_ids[split_all_le]], feature_id]
                    best_threshold = current_X[sort_ids[split_all_le], feature_id]
                    best_mode = 'all'
                    best_fraction_threshold = fraction_threshold
        # print(best_loss)
        return FractionActiveSplitInfo(
            loss=best_loss,
            feature=best_feature,
            threshold=best_threshold,
            mode=best_mode,
            fraction_threshold=best_fraction_threshold,
            activation_threshold=best_activation_threshold,
        )

    def reorder_indices(self, point_indices: np.ndarray, left: int, right: int,
                        data: TreeData, split: FractionActiveSplitInfo,
                        point_activations_count: np.ndarray) -> int:
        current_x = data.X[point_indices[left:right]][:, split.feature]
        if split.mode == 'any':
            # TODO: make faster implementation
            # groups_left = set([
            #     data.group_ids[i]
            #     for i in point_indices[left:right]
            #     if data.X[i, split.feature] <= split.threshold
            # ])
            mask = (data.X[point_indices[left:right], split.feature] <= split.threshold)
            mask &= (point_activations_count[point_indices[left:right]] >= split.activation_threshold)  # NOTE: new
            active_points = point_indices[left:right][mask]
            active_groups, active_group_counts = np.unique(data.group_ids[active_points], return_counts=True)
            active_group_sizes = data.group_sizes[active_groups]
            active_group_fractions = active_group_counts.astype(np.float32) / active_group_sizes
            groups_left = set(active_groups[active_group_fractions >= split.fraction_threshold])
            # update point activations count
            # NOTE: new
            mask_of_active = np.zeros_like(mask)
            mask_of_active[np.isin(data.group_ids[point_indices[left:right]], groups_left)] = True
            mask_of_active &= mask
            point_activations_count[point_indices[left:right]] += 1
        elif split.mode == 'all':
            raise NotImplementedError()
            # TODO: check
            groups_left = set([
                data.group_ids[i]
                for i in point_indices[left:right]
            ]).difference(set([
                data.group_ids[i]
                for i in point_indices[left:right]
                if data.X[i, split.feature] > split.threshold
            ]))
        else:
            raise ValueError(f'Unknown {split.mode=!r}')
        # TODO: make faster implementation
        ids_left = list(filter(lambda i: data.group_ids[i] in groups_left, point_indices[left:right]))
        ids_right = list(filter(lambda i: data.group_ids[i] not in groups_left, point_indices[left:right]))
        point_indices[left:right] = np.array(ids_left + ids_right)
        pos = left + len(ids_left)
        return pos

    def make_node(self):
        node = Node(idx=len(self.nodes), split=None)
        self.nodes.append(node)
        return node

    def calc_value(self, data, point_indices):
        current_group_ids = np.unique(data.group_ids[point_indices])  # TODO: without unique
        return data.y[current_group_ids].mean(axis=0)

    def convert_to_tree(self, nodes: List[Node]) -> FractionActiveTree:
        n_nodes = len(nodes)
        tree = FractionActiveTree(
            feature=np.full(n_nodes, -1, dtype=np.int64),
            threshold=np.zeros(n_nodes, dtype=np.float64),
            value=np.zeros(n_nodes, dtype=np.float64),
            left=np.full(n_nodes, -1, dtype=np.int64),
            right=np.full(n_nodes, -1, dtype=np.int64),
            mode=np.full(n_nodes, ThresholdMode.WRONG.value, dtype=np.int64),
            fraction_threshold=np.zeros(n_nodes, dtype=np.float64),
            activation_threshold=np.zeros(n_nodes, dtype=np.uint8),
        )
        for node in nodes:
            if node.split:
                tree.feature[node.idx] = node.split.feature
                tree.threshold[node.idx] = node.split.threshold
                tree.mode[node.idx] = ThresholdMode[node.split.mode.upper()].value
                tree.fraction_threshold[node.idx] = node.split.fraction_threshold
                tree.activation_threshold[node.idx] = node.split.activation_threshold
            tree.value[node.idx] = node.value
            if node.left is not None and node.right is not None:
                tree.left[node.idx] = node.left.idx
                tree.right[node.idx] = node.right.idx
            elif node.left is None and node.right is None:
                tree.left[node.idx] = -1
                tree.right[node.idx] = -1
            else:
                raise RuntimeError(f'{node.left=!r}, {node.right=!r}')
        return tree

    def build(self, X, y, group_sizes) -> FractionActiveTree:
        data = TreeData(X=X, y=y, group_sizes=group_sizes)
        point_activations_count = np.full(X.shape[0], 0, dtype=np.uint8)
        # indices of bags corresponding to nodes
        # bag_indices = np.arange(y.shape[0], dtype=np.int64)
        # indices of points (feature vectors) corresponding to nodes
        point_indices = np.arange(X.shape[0], dtype=np.int64)
        # initially we have one node, and all the bags correspond to it
        self.nodes = []
        current_leaves = [TmpSITLeaf(left=0, right=len(point_indices))]
        while len(current_leaves) > 0:
            leaf = current_leaves.pop()
            node = self.make_node()
            if leaf.parent is not None:
                if leaf.is_left:
                    leaf.parent.left = node
                else:
                    leaf.parent.right = node
            assert leaf.left < leaf.right, f'Wrong order: {leaf.left=} >= {leaf.right=}'

            node.value = self.calc_value(data, point_indices[leaf.left:leaf.right])
            if leaf.right - leaf.left < self.min_samples_leaf:
                continue
            if self.max_depth is not None and leaf.depth >= self.max_depth:
                continue

            # print('left, right:', leaf.left, leaf.right)
            split = self.find_best_split(
                data,
                point_indices[leaf.left:leaf.right],
                point_activations_count=point_activations_count,
                fraction_grid_size=self.fraction_grid_size,
                max_activations=min(leaf.depth, self.max_activations),
            )
            split_pos = self.reorder_indices(
                point_indices, leaf.left, leaf.right, data, split,
                point_activations_count=point_activations_count,
            )
            # print(' ', split)
            # print('  split_pos:', split_pos)
            if split_pos == leaf.left or split_pos == leaf.right:
                continue
            node.split = split
            current_leaves.append(TmpSITLeaf(
                left=leaf.left, right=split_pos, parent=node,
                is_left=True,
                depth=leaf.depth + 1,
            ))
            current_leaves.append(TmpSITLeaf(
                left=split_pos, right=leaf.right, parent=node,
                is_left=False,
                depth=leaf.depth + 1,
            ))

        return self.convert_to_tree(self.nodes)


class FractionActiveSetInputTree(BaseEstimator):
    def __init__(self, max_depth: int | None = None, min_samples_leaf: int = 2, extra_rand: bool = False,
                 fraction_grid_size: int = 10,
                 max_activations: int = 1,
                 ft: FeaturePreprocessingT = 'concat_negative',
                 random_state: np.random.RandomState | None = None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.extra_rand = extra_rand
        self.fraction_grid_size = fraction_grid_size
        self.max_activations = max_activations
        self.ft = ft
        self.random_state = random_state

    def fit(self, X, y, group_sizes):
        """
        Args:
            X: Features of size (n_instances, n_features).
            y: labels of size (n_samples).
            group_sizes: Group sizes of shape (n_samples).

        """
        self.ft_func_ = make_preprocessing_ft(self.ft)
        if y.ndim == 2:
            y = y[:, np.newaxis]
        builder = FractionActiveTreeBuilder(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            extra_rand=self.extra_rand,
            fraction_grid_size=self.fraction_grid_size,
            max_activations=self.max_activations,
            random_state=self.random_state,
        )
        self.tree_ = builder.build(self.ft_func_(X), y, group_sizes)
        return self

    def _predict_one_group(self, group_x, buffer: np.ndarray):
        # point_activations_count = np.full(group_x.shape[0], 0, dtype=np.uint8)
        point_activations_count = buffer[:group_x.shape[0]]
        point_activations_count.fill(0)
        node_id = 0
        while self.tree_.left[node_id] != self.tree_.right[node_id]:
            feature = self.tree_.feature[node_id]
            threshold = self.tree_.threshold[node_id]
            mode = self.tree_.mode[node_id]
            activation_threshold = self.tree_.activation_threshold[node_id]
            point_predicate = (group_x[:, feature] <= threshold)
            if mode == ThresholdMode.ANY.value:
                # is_left = np.any(group_x[:, feature] <= threshold)
                is_left = (
                    np.mean(
                        # NOTE: new
                        point_predicate & (point_activations_count >= activation_threshold)
                    ) >= self.tree_.fraction_threshold[node_id]
                )
            elif mode == ThresholdMode.ALL.value:
                raise NotImplementedError()
                is_left = np.all(point_predicate)
            else:
                raise NotImplementedError()
            if is_left:
                # NOTE: new
                point_activations_count[point_predicate] += 1
                node_id = self.tree_.left[node_id]
            else:
                node_id = self.tree_.right[node_id]
        return self.tree_.value[node_id]

    def predict(self, X, group_sizes):
        data = TreeData(X=self.ft_func_(X), y=None, group_sizes=group_sizes)

        # NOTE: new
        max_group_size = np.max(group_sizes)
        buffer = np.zeros(max_group_size, dtype=np.uint8)

        result = np.zeros(group_sizes.shape[0], dtype=np.float64)
        for group in range(group_sizes.shape[0]):
            result[group] = self._predict_one_group(data.X[data.group_ids == group], buffer)
        return result
