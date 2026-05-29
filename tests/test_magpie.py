import numpy as np

from phonon_omegamax.features.magpie import featurize_samples


def test_magpie_shape(fake_samples):
    X = featurize_samples(fake_samples)
    assert X.shape == (12, 132)
    assert X.dtype == np.float64
    assert np.isfinite(X).all()


def test_magpie_cache_round_trip(tmp_path, fake_samples):
    cache = tmp_path / "magpie.npy"
    X1 = featurize_samples(fake_samples, cache_path=cache)
    assert cache.exists()
    X2 = featurize_samples(fake_samples, cache_path=cache)
    assert np.allclose(X1, X2)
