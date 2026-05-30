"""Generate figures/headline.png — GBDT 5-fold parity + residual KDE.

CGCNN is excluded from the headline because the partial 3-fold run covers a
different (smaller) sample set than GBDT's full 5-fold sweep, and overlaying
them would be misleading. The leaderboard table still records both numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).parent
folds = json.loads((ROOT / "metrics" / "gbdt_folds.json").read_text())

# Concatenate all 5 test folds (covers every sample exactly once)
y_true_all: list[float] = []
y_pred_all: list[float] = []

from phonon_omegamax.data.load import load_phonons
samples = load_phonons(cache_path=ROOT / "data/cache/phonons.parquet")
y_full = np.array([s.target for s in samples], dtype=np.float64)

for f in folds:
    idx = f["test_idx"]
    pred = f["predictions"]
    y_true_all.extend(y_full[idx].tolist())
    y_pred_all.extend(pred)

y_true = np.asarray(y_true_all)
y_pred = np.asarray(y_pred_all)

fig, (ax_par, ax_res) = plt.subplots(1, 2, figsize=(11, 4.5))

lim_hi = float(max(y_true.max(), y_pred.max()) * 1.05)
ax_par.fill_between(
    [0, lim_hi], [-50, lim_hi - 50], [50, lim_hi + 50],
    color="grey", alpha=0.1, label="±50 cm⁻¹",
)
ax_par.plot([0, lim_hi], [0, lim_hi], "k-", linewidth=0.8, zorder=1)
ax_par.scatter(y_true, y_pred, s=8, alpha=0.55, color="C0", zorder=2,
               label=f"Magpie + GBDT (N={len(y_true)})")
ax_par.set_xlim(0, lim_hi)
ax_par.set_ylim(0, lim_hi)
ax_par.set_xlabel("True ω_max (cm⁻¹)")
ax_par.set_ylabel("Predicted ω_max (cm⁻¹)")
ax_par.set_title("Parity — 5-fold CV")
ax_par.legend(loc="upper left", fontsize=9)

r = y_pred - y_true
span = np.linspace(r.min(), r.max(), 400)
ax_res.plot(span, gaussian_kde(r)(span), color="C0", lw=1.5)
ax_res.axvline(0, color="k", linestyle="--", linewidth=0.8)
ax_res.set_xlabel("Residual (predicted − true, cm⁻¹)")
ax_res.set_ylabel("Density")
mae = np.abs(r).mean()
ax_res.set_title(f"Residuals — mean |r| = {mae:.1f} cm⁻¹")

fig.tight_layout()
out = ROOT / "figures" / "headline.png"
fig.savefig(out, dpi=200)
print(f"wrote {out}")
