# xSIT

This repository contains reference implementation of Decision Trees able to process sets of instances as inputs
(like in Multiple Instance Learning, but without assumptions on how bag-level label is obtained).

Main models:

- Tree-PA – trees with predicate aggregation at each node (each decision is based on the whole bag)
- Tree-Frac – extended predicate aggregation
- Grad-Tree – gradient-based tree with attention for instance-level embeddings aggregation
- Grad-SIT – gradient-based tree with an aggregation model (arbitrary, like GBM, RF, etc.) applied to a bag-level embedding
- Grad-SIT-Forest – ensemble of Grad-SIT trees

## Installation

1. Install `gradient_growing_trees`.

Unfortunately, the `gradient_growing_trees` package is incompatible with isolated PIP builds
(because depends on Cython+NumPy build),
and therefore should be installed manually:

```
pip install setuptools numpy==2.2.0 scikit-learn==1.5.2 Cython==3.0.11
pip install git+https://github.com/NTAILab/gradient_growing_trees.git --no-build-isolation
```

2. Clone this repository and install `sit` (this package) in development mode.

```
pip install -e .
```

## Package structure

- `sit.mil.data` – The MIL (Set-Input) dataset structure `MILData`
- `sit.tree.{any_all, fraction, fraction_active}` – Trees with predicate aggregation, that processes whole bags at once
- `sit.gradtree.grad_boosting` – Gradient-based trees with Attention aggregation
- `sit.gradtree.embedder` – Bag embedder based on gradient-based trees (trims Attention)
- `sit.gradtree.{classifier, regressor}` – Grad-SIT models, based on bag embedder.

## Usage example

**The detailed examples are provided in [notebooks/](notebooks/) directory.**

A short Grad-SIT usage example:

```python
mil_data_train: MILData
mil_data_test: MILData

model = GradSITRegressor(
    lam_2=0.001,  # regularization
    lr=1.0,       # learning rate of a single decision tree
    max_depth=7,
    splitter='random',
    n_update_iterations=1,  # number of node value updates
    embedding_size=32,      # size of dense embeddings contained in leaves
    nn_lr=1.e-4,            # neural network learning rate (relative to sum loss, not mean)
    nn_num_heads=8,         # number of attention heads
    nn_steps=1,             # number of neural network update steps at one tree construction interation
    dropout=0.0,            # neural network dropout
)
model.fit(mil_data_train.X, mil_data_train.y, mil_data_train.group_sizes)
predictions = model.predict(mil_data_test.X, mil_data_test.group_sizes)
```
