import numpy as np

from phonon_omegamax.data.split import inner_train_val_split, kfold_indices


def test_kfold_indices_partition_full_set():
    folds = list(kfold_indices(n=100, seed=42))
    assert len(folds) == 5
    all_test = []
    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == 100
        all_test.extend(test_idx)
    # Every index appears in exactly one test fold.
    assert sorted(all_test) == list(range(100))


def test_kfold_indices_are_deterministic():
    a = list(kfold_indices(n=50, seed=42))
    b = list(kfold_indices(n=50, seed=42))
    for (ta, te_a), (tb, te_b) in zip(a, b):
        assert te_a == te_b
        assert ta == tb


def test_inner_train_val_split_proportional():
    rng = np.arange(100)
    train, val = inner_train_val_split(rng.tolist(), val_frac=0.2, seed=0)
    assert len(val) == 20
    assert len(train) == 80
    assert set(train) | set(val) == set(rng.tolist())
    assert set(train) & set(val) == set()
