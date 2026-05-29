"""Regression metrics: MAE, R², residual quantiles."""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.abs(np.asarray(y_true) - np.asarray(y_pred)).mean())


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 0.0 if ss_res == 0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def residual_quantiles(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    r = np.asarray(y_pred) - np.asarray(y_true)
    return {
        "mean": float(r.mean()),
        "p05": float(np.quantile(r, 0.05)),
        "p50": float(np.quantile(r, 0.50)),
        "p95": float(np.quantile(r, 0.95)),
    }
