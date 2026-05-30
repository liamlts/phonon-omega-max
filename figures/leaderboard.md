# Phonon ω_max prediction — Leaderboard

| Model | MAE (cm⁻¹) | R² | Δ vs Magpie |
|---|---|---|---|
| Magpie + GBDT (ours) | 70.1 ± 10.3 | 0.910 | — |
| CGCNN (ours) | 94.7 ± 12.4 | 0.799 | -24.6 |
| --- | --- | --- | --- |
| Roost (published) | 75.6 | — | — |
| CGCNN (published) | 57.8 | — | — |
| MEGNet (published) | 49.9 | — | — |
| ALIGNN (published) | 29.5 | — | — |

_Note: CGCNN is a partial result over 3 of 5 CV folds — partial: folds 0-1 ensembled over 3 seeds, fold 2 single-seed, folds 3-4 not run_
