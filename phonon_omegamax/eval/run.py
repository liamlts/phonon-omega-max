"""End-to-end CV runner: load → featurize → train both models → write per-fold metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.load import load_phonons
from ..data.split import inner_train_val_split, kfold_indices
from ..features.magpie import featurize_samples
from ..models.dataset import StructureDataset
from ..models.ensemble import load_ensemble_predictions, train_fold_ensemble
from ..models.gbdt import GBDTRegressor
from ..sample import Sample
from .metrics import mae, r2, residual_quantiles


def run_gbdt_cv(
    samples: list[Sample],
    cache_dir: Path,
    metrics_dir: Path,
    seed: int = 42,
) -> dict:
    cache_dir = Path(cache_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    X = featurize_samples(samples, cache_path=cache_dir / "magpie.npy")
    y = np.array([s.target for s in samples], dtype=np.float64)

    fold_metrics: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(samples), seed=seed)):
        model = GBDTRegressor(random_state=seed + fold).fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        fold_metrics.append({
            "fold": fold,
            "mae": mae(y[test_idx], pred),
            "r2": r2(y[test_idx], pred),
            "residuals": residual_quantiles(y[test_idx], pred),
            "test_idx": list(map(int, test_idx)),
            "predictions": pred.tolist(),
        })
    summary = _summarize(fold_metrics, "gbdt")
    (metrics_dir / "gbdt_folds.json").write_text(json.dumps(fold_metrics, indent=2))
    return summary


def run_cgcnn_cv(
    samples: list[Sample],
    cache_dir: Path,
    ckpt_dir: Path,
    metrics_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    epochs: int = 100,
    seed: int = 42,
) -> dict:
    cache_dir = Path(cache_dir)
    ckpt_dir = Path(ckpt_dir)
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    y = np.array([s.target for s in samples], dtype=np.float64)
    graphs_dir = cache_dir / "graphs"

    fold_metrics: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(samples), seed=seed)):
        inner_train, inner_val = inner_train_val_split(train_idx, val_frac=0.2, seed=seed)
        train_ds = StructureDataset([samples[i] for i in inner_train], cache_dir=graphs_dir)
        val_ds = StructureDataset([samples[i] for i in inner_val], cache_dir=graphs_dir)
        test_ds = StructureDataset([samples[i] for i in test_idx], cache_dir=graphs_dir)

        train_fold_ensemble(
            train_ds, val_ds,
            ckpt_dir=ckpt_dir, fold=fold, seeds=seeds, epochs=epochs,
        )
        pred = load_ensemble_predictions(ckpt_dir, fold=fold, dataset=test_ds)
        fold_metrics.append({
            "fold": fold,
            "mae": mae(y[test_idx], pred),
            "r2": r2(y[test_idx], pred),
            "residuals": residual_quantiles(y[test_idx], pred),
            "test_idx": list(map(int, test_idx)),
            "predictions": pred.tolist(),
        })
    summary = _summarize(fold_metrics, "cgcnn")
    (metrics_dir / "cgcnn_folds.json").write_text(json.dumps(fold_metrics, indent=2))
    return summary


def _summarize(fold_metrics: list[dict], name: str) -> dict:
    maes = [f["mae"] for f in fold_metrics]
    r2s = [f["r2"] for f in fold_metrics]
    return {
        "model": name,
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "n_folds": len(fold_metrics),
    }
