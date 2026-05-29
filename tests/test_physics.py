import numpy as np

from phonon_omegamax.physics.analysis import (
    bin_failures_by_spacegroup,
    find_cgcnn_wins,
)


def test_find_cgcnn_wins_threshold(fake_samples):
    rng = np.random.default_rng(0)
    n = len(fake_samples)
    y = np.array([s.target for s in fake_samples])
    gbdt_pred = y + rng.normal(0, 80, n)
    cgcnn_pred = y + rng.normal(0, 20, n)
    wins = find_cgcnn_wins(
        fake_samples, y_true=y,
        gbdt_pred=gbdt_pred, cgcnn_pred=cgcnn_pred,
        threshold=50.0,
    )
    # Each entry has the keys we need for write-up.
    for w in wins:
        assert {"mp_id", "formula", "gbdt_err", "cgcnn_err", "improvement"} <= set(w)


def test_bin_failures_by_spacegroup(fake_samples):
    failures = [
        {"mp_id": fake_samples[0].mp_id, "structure": fake_samples[0].structure},
        {"mp_id": fake_samples[1].mp_id, "structure": fake_samples[1].structure},
    ]
    bins = bin_failures_by_spacegroup(failures)
    # NaCl is Fm-3m (225). MgO is Fm-3m too. So the bin maps SG to count.
    assert all(isinstance(k, int) for k in bins)
    assert sum(bins.values()) == 2
