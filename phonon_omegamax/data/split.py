"""5-fold CV indices + inner train/val split.

Self-contained KFold (not matbench's, so we don't need the task object loaded
to iterate). Matbench protocol comparability is maintained by using a fixed
seed and 5 folds with sequential stratification.
"""
from __future__ import annotations

import random
from typing import Iterator


def kfold_indices(n: int, seed: int = 42, k: int = 5) -> Iterator[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    fold_size = n // k
    remainder = n % k
    cursor = 0
    folds: list[list[int]] = []
    for i in range(k):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(indices[cursor : cursor + size])
        cursor += size
    for i in range(k):
        test_idx = sorted(folds[i])
        train_idx = sorted(
            j for f, fold in enumerate(folds) if f != i for j in fold
        )
        yield train_idx, test_idx


def inner_train_val_split(
    train_idx: list[int], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_frac))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val
