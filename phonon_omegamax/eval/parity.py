"""Headline 2-panel figure: parity plot + residual KDE."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def headline_figure(
    y_true: np.ndarray,
    gbdt_pred: np.ndarray,
    cgcnn_pred: np.ndarray,
    out_path: Path | None = None,
):
    y_true = np.asarray(y_true)
    gbdt_pred = np.asarray(gbdt_pred)
    cgcnn_pred = np.asarray(cgcnn_pred)

    fig, (ax_par, ax_res) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Parity ---
    lim_lo = 0.0
    lim_hi = float(max(y_true.max(), gbdt_pred.max(), cgcnn_pred.max()) * 1.05)
    ax_par.fill_between(
        [lim_lo, lim_hi], [lim_lo - 50, lim_hi - 50], [lim_lo + 50, lim_hi + 50],
        color="grey", alpha=0.1, label="±50 cm⁻¹",
    )
    ax_par.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k-", linewidth=0.8, zorder=1)
    ax_par.scatter(y_true, gbdt_pred, s=8, alpha=0.5, color="grey",
                   label="Magpie + GBDT", zorder=2)
    ax_par.scatter(y_true, cgcnn_pred, s=8, alpha=0.6, color="C0",
                   label="CGCNN", zorder=3)
    ax_par.set_xlim(lim_lo, lim_hi)
    ax_par.set_ylim(lim_lo, lim_hi)
    ax_par.set_xlabel("True ω_max (cm⁻¹)")
    ax_par.set_ylabel("Predicted ω_max (cm⁻¹)")
    ax_par.set_title("Parity")
    ax_par.legend(loc="upper left", fontsize=9)

    # --- Residual KDE ---
    r_gbdt = gbdt_pred - y_true
    r_cgcnn = cgcnn_pred - y_true
    span = np.linspace(min(r_gbdt.min(), r_cgcnn.min()),
                       max(r_gbdt.max(), r_cgcnn.max()), 400)
    ax_res.plot(span, gaussian_kde(r_gbdt)(span), color="grey", label="Magpie + GBDT")
    ax_res.plot(span, gaussian_kde(r_cgcnn)(span), color="C0", label="CGCNN")
    ax_res.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax_res.set_xlabel("Residual (predicted − true, cm⁻¹)")
    ax_res.set_ylabel("Density")
    ax_res.set_title("Residual distribution")
    ax_res.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig
