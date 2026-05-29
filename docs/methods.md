# Methods

## Phase A: Magpie + GBDT

Composition fingerprint via matminer's `ElementProperty.from_preset("magpie")`
(132 element-property statistics: mean/min/max/std of atomic mass,
electronegativity, valence electron count, etc., across the formula).
NaN values are mean-imputed per column.

Model: sklearn `HistGradientBoostingRegressor`, 500 trees, learning
rate 0.05, 63 leaves, early stopping on an internal 20% validation
holdout. One model per outer CV fold, fixed seed.

Why sklearn rather than LightGBM: the sibling `xanes-oxstate` project
documented a torch + lightgbm libomp dual-load deadlock on macOS. We
preempt it here.

## Phase B: CGCNN

Graph construction: for each crystal, atoms become nodes with 92-dim
one-hot element identity. Edges connect atom pairs within 8 Å (PBC-aware),
with a 41-dim Gaussian-basis distance feature (σ = 0.2 Å, centered
uniformly in [0, 8] Å).

Model: hand-implemented CGCNN (Xie & Grossman, 2018). Three gated
convolution layers with hidden width 64, then global mean pooling and
a two-layer MLP head (64 → 128 → 1). 80,065 parameters.

Training: Adam (lr 1e-3, cosine decay to 1e-5), batch size 64, Huber
loss (δ = 10 cm⁻¹), 200 epochs max with early stopping (patience 30)
on validation MAE. Five-seed ensemble per outer CV fold; ensemble
prediction is the mean of the five raw outputs.

## Splits

5-fold outer CV (test indices fixed by our deterministic KFold).
Inner 80/20 train/val split inside each outer training set, used for
GBDT early stopping and CGCNN best-checkpoint selection. Test indices
are never touched during model selection.

## Metrics

- MAE (primary), cm⁻¹.
- R² (coefficient of determination).
- Residual quantiles P05/P50/P95 for fat-tail diagnostics.
- All reported as mean ± std across the 5 outer folds.
