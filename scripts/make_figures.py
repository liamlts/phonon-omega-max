"""Aggregate per-fold JSONs → headline figure + leaderboard.md."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phonon_omegamax.eval.leaderboard import build_leaderboard
from phonon_omegamax.eval.parity import headline_figure


def main() -> None:
    metrics_dir = Path("metrics")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    gbdt_summary = _maybe_load(metrics_dir / "gbdt_summary.json")
    cgcnn_summary = _maybe_load(metrics_dir / "cgcnn_summary.json")
    table = build_leaderboard(gbdt_summary, cgcnn_summary)
    (fig_dir / "leaderboard.md").write_text(table + "\n")
    print(f"wrote {fig_dir / 'leaderboard.md'}")

    gbdt_folds = _maybe_load(metrics_dir / "gbdt_folds.json")
    cgcnn_folds = _maybe_load(metrics_dir / "cgcnn_folds.json")
    if gbdt_folds and cgcnn_folds:
        y_true, gbdt_pred, cgcnn_pred = _gather_oof(gbdt_folds, cgcnn_folds)
        headline_figure(
            y_true=y_true, gbdt_pred=gbdt_pred, cgcnn_pred=cgcnn_pred,
            out_path=fig_dir / "headline.png",
        )
        headline_figure(
            y_true=y_true, gbdt_pred=gbdt_pred, cgcnn_pred=cgcnn_pred,
            out_path=fig_dir / "headline.pdf",
        )
        print(f"wrote {fig_dir / 'headline.png'} and headline.pdf")
    else:
        print("skip headline figure: missing per-fold JSONs")


def _maybe_load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _gather_oof(gbdt_folds: list[dict], cgcnn_folds: list[dict]):
    # Reassemble out-of-fold predictions by their test_idx.
    from phonon_omegamax.data.load import load_phonons
    samples = load_phonons()
    n = len(samples)
    y_true = np.array([s.target for s in samples], dtype=np.float64)
    gbdt_pred = np.full(n, np.nan)
    cgcnn_pred = np.full(n, np.nan)
    for f in gbdt_folds:
        gbdt_pred[f["test_idx"]] = f["predictions"]
    for f in cgcnn_folds:
        cgcnn_pred[f["test_idx"]] = f["predictions"]
    mask = np.isfinite(gbdt_pred) & np.isfinite(cgcnn_pred)
    return y_true[mask], gbdt_pred[mask], cgcnn_pred[mask]


if __name__ == "__main__":
    main()
