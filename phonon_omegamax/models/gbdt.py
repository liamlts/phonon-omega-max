"""sklearn HistGradientBoostingRegressor wrapper.

We use sklearn rather than lightgbm because lightgbm and torch share a
libomp symbol on macOS that deadlocks on dual-load. The xanes-oxstate
project documented this; we pre-emptively avoid it here.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GBDTRegressor:
    def __init__(self, random_state: int = 0) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=63,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDTRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
