import numpy as np

from phonon_omegamax.models.gbdt import GBDTRegressor


def test_gbdt_fits_and_predicts_synthetic():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 8))
    y = 3 * X[:, 0] + 2 * X[:, 1] ** 2 - X[:, 3] + rng.normal(0, 0.1, n)

    clf = GBDTRegressor(random_state=0).fit(X, y)
    pred = clf.predict(X)
    mae = np.abs(pred - y).mean()
    assert mae < 1.0  # comfortably better than predicting the mean


def test_gbdt_returns_self_for_chaining():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=50) + 100
    clf = GBDTRegressor(random_state=0)
    result = clf.fit(X, y)
    assert result is clf
