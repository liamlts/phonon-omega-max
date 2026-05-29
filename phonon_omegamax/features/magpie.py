"""Magpie 132-dim composition fingerprint via matminer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..sample import Sample


def _featurizer():
    from matminer.featurizers.composition import ElementProperty

    return ElementProperty.from_preset("magpie")


def featurize_samples(
    samples: list[Sample], cache_path: Path | None = None
) -> np.ndarray:
    if cache_path is not None and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(samples):
            return cached

    feat = _featurizer()
    compositions = [s.structure.composition for s in samples]
    X = np.asarray(feat.featurize_many(compositions), dtype=np.float64)
    # matminer can emit NaN on edge cases (e.g. element not in lookup table);
    # impute with column mean rather than dropping rows so downstream code
    # doesn't have to track which rows survived.
    col_mean = np.nanmean(X, axis=0)
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.take(col_mean, np.where(mask)[1])

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X
