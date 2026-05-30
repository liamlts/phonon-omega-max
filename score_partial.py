"""One-off: score the partial CGCNN checkpoints (folds 0-2 only).

Fold 0: 3 seeds. Fold 1: 3 seeds. Fold 2: 1 seed (seed 0).
Folds 3-4: no checkpoints, skipped.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch  # noqa: F401

from phonon_omegamax.data.load import load_phonons
from phonon_omegamax.data.split import kfold_indices
from phonon_omegamax.eval.metrics import mae, r2, residual_quantiles
from phonon_omegamax.models.dataset import StructureDataset
from phonon_omegamax.models.ensemble import load_ensemble_predictions

ROOT = Path(__file__).parent
CKPT = ROOT / "checkpoints"
CACHE = ROOT / "data" / "cache"
METRICS = ROOT / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

samples = load_phonons(cache_path=CACHE / "phonons.parquet")
y = np.array([s.target for s in samples], dtype=np.float64)
graphs_dir = CACHE / "graphs"

SCORED_FOLDS = [0, 1, 2]  # only folds we have checkpoints for
fold_metrics: list[dict] = []
for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(samples), seed=42)):
    if fold not in SCORED_FOLDS:
        continue
    test_ds = StructureDataset([samples[i] for i in test_idx], cache_dir=graphs_dir)
    pred = load_ensemble_predictions(CKPT, fold=fold, dataset=test_ds)
    n_seeds = len(sorted(CKPT.glob(f"fold{fold}_seed*.pt")))
    m = mae(y[test_idx], pred)
    r = r2(y[test_idx], pred)
    print(f"fold {fold}: n_seeds={n_seeds}  MAE={m:.2f}  R²={r:.3f}")
    fold_metrics.append({
        "fold": fold,
        "n_seeds": n_seeds,
        "mae": m,
        "r2": r,
        "residuals": residual_quantiles(y[test_idx], pred),
        "test_idx": list(map(int, test_idx)),
        "predictions": pred.tolist(),
    })

maes = [f["mae"] for f in fold_metrics]
r2s = [f["r2"] for f in fold_metrics]
summary = {
    "model": "cgcnn",
    "mae_mean": float(np.mean(maes)),
    "mae_std": float(np.std(maes)),
    "r2_mean": float(np.mean(r2s)),
    "r2_std": float(np.std(r2s)),
    "n_folds": len(fold_metrics),
    "note": "partial: folds 0-1 ensembled over 3 seeds, fold 2 single-seed, folds 3-4 not run",
}
(METRICS / "cgcnn_folds.json").write_text(json.dumps(fold_metrics, indent=2))
(METRICS / "cgcnn_summary.json").write_text(json.dumps(summary, indent=2))
print()
print(f"CGCNN (partial) MAE = {summary['mae_mean']:.1f} ± {summary['mae_std']:.1f} cm⁻¹")
print(f"            R²      = {summary['r2_mean']:.3f} ± {summary['r2_std']:.3f}")
print(f"            n_folds = {summary['n_folds']}")
