import numpy as np

from phonon_omegamax.eval.metrics import mae, r2, residual_quantiles


def test_mae_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert mae(y_true, y_pred) == 0.5


def test_r2_unit_on_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == 1.0


def test_r2_zero_on_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, y.mean())
    assert abs(r2(y, pred)) < 1e-9


def test_residual_quantiles():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    pred = y + rng.normal(scale=0.5, size=1000)
    q = residual_quantiles(y, pred)
    assert set(q.keys()) == {"p05", "p50", "p95", "mean"}
    assert q["p05"] < q["p50"] < q["p95"]
