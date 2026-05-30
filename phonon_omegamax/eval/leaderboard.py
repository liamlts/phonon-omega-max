"""Render a markdown leaderboard table comparing our results to published baselines."""
from __future__ import annotations

PUBLISHED = [
    ("Roost (published)", 75.6),
    ("CGCNN (published)", 57.8),
    ("MEGNet (published)", 49.9),
    ("ALIGNN (published)", 29.5),
]


def build_leaderboard(
    ours_gbdt: dict | None,
    ours_cgcnn: dict | None,
) -> str:
    lines = [
        "| Model | MAE (cm⁻¹) | R² | Δ vs Magpie |",
        "|---|---|---|---|",
    ]
    gbdt_mae = ours_gbdt["mae_mean"] if ours_gbdt else None
    if ours_gbdt is not None:
        lines.append(
            f"| Magpie + GBDT (ours) | "
            f"{ours_gbdt['mae_mean']:.1f} ± {ours_gbdt['mae_std']:.1f} | "
            f"{ours_gbdt['r2_mean']:.3f} | — |"
        )
    if ours_cgcnn is not None:
        delta = ""
        if gbdt_mae is not None:
            delta = f"{gbdt_mae - ours_cgcnn['mae_mean']:+.1f}"
        lines.append(
            f"| CGCNN (ours) | "
            f"{ours_cgcnn['mae_mean']:.1f} ± {ours_cgcnn['mae_std']:.1f} | "
            f"{ours_cgcnn['r2_mean']:.3f} | {delta} |"
        )
    lines.append("| --- | --- | --- | --- |")
    for name, mae_val in PUBLISHED:
        lines.append(f"| {name} | {mae_val} | — | — |")
    return "\n".join(lines)
