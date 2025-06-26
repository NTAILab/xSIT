import torch
from ..mil.data import MILData
from gradient_growing_trees.tree_nn import TreeNN
from gradient_growing_trees.tree import BatchArbitraryLoss
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone, defaultdict
from sklearn.metrics import r2_score
from abc import ABCMeta, abstractmethod


class AttentionAggregationNN(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, out_features: int = 1):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = torch.nn.Parameter(torch.ones((1, 1, embed_dim), dtype=torch.float32, requires_grad=True))
        self.linear = torch.nn.Linear(embed_dim, out_features)
        self.group_ids = None

    def _recompute_group_cache(self, group_ids):
        if group_ids is self.group_ids:
            return
        # print('Recomputing cache')
        self.group_ids = group_ids

        group_ids = group_ids.ravel().to(torch.long)
        # These operations can be run once in advance
        unique_group_ids, group_sizes = torch.unique(group_ids, return_counts=True)
        self.n_groups = len(unique_group_ids)
        self.max_group_size = torch.max(group_sizes)

        self.kp_mask = torch.zeros(self.n_groups, self.max_group_size, dtype=torch.bool)  # this mask can also be prefilled
        for gid, gs in zip(unique_group_ids, group_sizes):
            # embs[gid, :gs] = tree_preds[group_ids == gid]
            self.kp_mask[gid, gs:] = True
        self.emplacement_ids = tuple(torch.argwhere(~self.kp_mask).T)
        self.instance_sorter = torch.argsort(group_ids)

    def forward(self, tree_preds, group_ids):
        self._recompute_group_cache(group_ids)
        embed_dim = tree_preds.shape[1]

        # Embeddings tensor should be constructed each time
        embs = torch.zeros(self.n_groups, self.max_group_size, embed_dim, dtype=tree_preds.dtype)
        embs[self.emplacement_ids] = tree_preds[self.instance_sorter]

        # print(self.query.shape, embs.shape, kp_mask.shape)
        group_embeddings, *_ = self.attention(
            self.query.expand(embs.shape[0], 1, embs.shape[2]),  # expand the batch dimension
            embs,
            embs,
            key_padding_mask=self.kp_mask,
            is_causal=False,
        )
        group_embeddings = group_embeddings.squeeze(1)
        return self.linear(group_embeddings)


class SetTreeNN(TreeNN):
    def __post_init__(self):
        self.history = defaultdict(list)
        self.enable_postiter_nn = False
        self.nn_lr = 1.e-4
        self.nn_steps = 1
        self.nn_num_heads = 4
        self.dropout = 0.0
        self.random_state = 1
        self.loss_fn = 'se'
        torch.manual_seed(self.random_state)
        self.metrics = {
            'r2': r2_score,
        }
        self.make_nn = lambda: (
            AttentionAggregationNN(
                embed_dim=self.embedding_size,
                num_heads=self.nn_num_heads,
                dropout=self.dropout,
                out_features=self.n_outputs_,
            )
        )

    def set_embedding_size(self, embedding_size: int):
        self.embedding_size = embedding_size
        return self

    def set_nn_lr(self, nn_lr: float):
        self.nn_lr = nn_lr
        return self

    def set_nn_steps(self, nn_steps: int):
        self.nn_steps = nn_steps
        return self

    def set_nn_num_heads(self, nn_num_heads: int):
        self.nn_num_heads = nn_num_heads
        return self

    def set_dropout(self, dropout: float):
        self.dropout = dropout
        return self

    def set_loss_fn(self, loss_fn: str):
        self.loss_fn = loss_fn
        return self

    def set_make_nn(self, make_nn):
        self.make_nn = make_nn

    def _postiter_nn(self, X_torch, y_torch, cumulative_predictions,
                     eval_X_nn=None,
                     eval_y=None,
                     eval_cumulative_predictions=None):
        if not self.enable_postiter_nn:
            return
        with torch.inference_mode():
            preds = self._predict_nn(X_torch, cumulative_predictions)
            self.history['loss/train'].append(
                self.__loss_fn(X_torch, y_torch, preds).item()
            )
            for name, metric_fn in self.metrics.items():
                self.history[name + '/train'].append(
                    metric_fn(y_torch.numpy(), preds.numpy())
                )
            if eval_cumulative_predictions is not None:
                assert eval_y is not None
                eval_preds = self._predict_nn(eval_X_nn, eval_cumulative_predictions)
                self.history['loss/val'].append(
                    self.__loss_fn(eval_X_nn, eval_y, eval_preds).item()
                )
                for name, metric_fn in self.metrics.items():
                    self.history[name + '/val'].append(
                        metric_fn(eval_y.numpy(), eval_preds.numpy())
                    )

    def _pretrain_nn(self, X_nn_torch, y_torch):
        self.n_outputs_ = y_torch.shape[1]
        self.nn_ = self.make_nn().to(torch.float64)
        self.optim_ = torch.optim.AdamW(self.nn_.parameters(), lr=self.nn_lr)

    def _predict_nn(self, cur_X_torch, cur_trees_predictions_torch):
        return self.nn_(cur_trees_predictions_torch, group_ids=cur_X_torch)

    def __loss_fn(self, cur_X_torch, cur_y_torch, nn_preds):
        # group_ids = cur_X_torch
        if callable(self.loss_fn):
            return self.loss_fn(nn_preds, cur_y_torch)
        elif self.loss_fn.lower() == 'se':
            return (cur_y_torch - nn_preds).pow(2).sum()
        elif self.loss_fn.lower() == 'bce':
            return torch.nn.functional.binary_cross_entropy_with_logits(
                nn_preds,
                cur_y_torch,
                reduction='sum'
            )
        else:
            raise ValueError(f'Wrong {self.loss_fn=!r}')

    def _post_update_nn(self, X_nn_torch, y_torch, sample_ids_torch, cumulative_predictions):
        for _ in range(self.nn_steps):
            self.optim_.zero_grad()
            nn_preds = self._predict_nn(X_nn_torch, cumulative_predictions)
            loss = self.__loss_fn(X_nn_torch, y_torch, nn_preds)
            loss.backward()
            self.optim_.step()

    def _calc_sample_grads(self, cur_X_torch, cur_y_torch,
                           cur_trees_predictions_torch,
                           cur_sample_predictions):
        nn_preds = self._predict_nn(cur_X_torch, cur_trees_predictions_torch)
        loss = self.__loss_fn(cur_X_torch, cur_y_torch, nn_preds)
        grads, = torch.autograd.grad(loss, cur_sample_predictions)
        return grads
