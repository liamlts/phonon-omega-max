# phonon-omega-max — Design Spec

**Date:** 2026-05-29
**Status:** Draft, pre-implementation
**Author:** Liam Schmidt

---

## 1. Scope, deliverable, success criteria

### Pitch

A Matbench-style regression of ω_max (last-peak phonon frequency, cm⁻¹)
for 1,265 inorganic crystals, comparing a composition-only Magpie + GBDT
baseline against a structure-based CGCNN, with honest uncertainty bands
and a leaderboard-positioning writeup.

### Portfolio context

Project #2 in a four-project physics-ML portfolio series. Sibling
repo `xanes-oxstate` (project #1) shipped at mean CNN accuracy 0.702
across 8 transition metals.

### Deliverables (priority order)

1. GitHub repo + README with one-command training, inline leaderboard
   table positioning our results against published Matbench baselines.
2. One end-to-end notebook (`notebooks/walkthrough.ipynb`).
3. Headline figure: 2-panel parity plot + residual distribution,
   Magpie+GBDT vs. CGCNN.
4. Physics finding: 1–2 paragraphs identifying material classes where
   the structure-based CGCNN beats the composition-only GBDT — the
   "structure helped" story.
5. *Optional*: HuggingFace model card or small Gradio demo.

### Scope (locked)

- **Data:** Matbench `phonons` task (1,265 inorganic crystals).
- **Target:** ω_max, single scalar in cm⁻¹.
- **Phase A:** composition-only Magpie features → sklearn
  `HistGradientBoostingRegressor`.
- **Phase B:** structure-based CGCNN trained on the same 5-fold CV
  splits.
- **Comparison protocol:** identical Matbench outer test folds for
  both models; published baselines included in the leaderboard
  table for context.

### Out of scope (explicit)

- e3nn / NequIP / MACE / any equivariant net (deferred to README
  "future work").
- Full DOS curve prediction (the broader Phase B option discussed at
  brainstorm time; this project predicts the scalar ω_max only).
- Materials Project phonon endpoint (sticking with Matbench for
  leaderboard story).
- Hyperparameter search, active learning, transfer learning,
  multi-task setups.

### Success criteria

- Magpie + GBDT: mean MAE ≤ 80 cm⁻¹ across the 5 Matbench folds
  (published Roost is at 75.6).
- CGCNN reproduction: mean MAE ≤ 65 cm⁻¹ (published CGCNN is 57.8).
- CGCNN strictly beats Magpie + GBDT by ≥ 10 cm⁻¹ MAE — the
  headline ablation.
- Mean residual ∈ [−5, 5] cm⁻¹ for both models (unbiased).
- ≥ 3 physics-finding bullets in `docs/physics_findings.md`, each
  grounded in concrete material examples.
- `pip install -e . && make all` reproduces both panels of the
  headline figure in < 6 h on a laptop CPU.

### Timeline

~5–7 days part-time:
- Day 1: data fetch + EDA + train/test split.
- Day 2: Magpie + GBDT baseline.
- Day 3–5: CGCNN training and tuning.
- Day 6: parity plot + physics-finding writeup.
- Day 7: notebook + README polish.

---

## 2. Architecture

### Repo layout

```
phonon-omega-max/
├── README.md
├── pyproject.toml
├── Makefile                       # make {data, train-gbdt, train-cgcnn, eval, figures, all}
├── .gitignore
│
├── phonon_omegamax/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py                # matbench → list[Sample]
│   │   ├── split.py               # Matbench 5-fold CV
│   │   └── sample.py              # Sample dataclass
│   ├── features/
│   │   ├── __init__.py
│   │   ├── magpie.py              # composition → 132-dim Magpie vector
│   │   └── graph.py               # pymatgen.Structure → CGCNN graph
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gbdt.py                # HistGradientBoostingRegressor wrapper
│   │   ├── cgcnn.py               # CGCNN model + Dataset
│   │   └── train.py               # generic fit/predict/cv runner
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py             # MAE, R², residuals
│   │   ├── parity.py              # parity-plot figure
│   │   └── leaderboard.py         # table assembly
│   └── cli.py                     # python -m phonon_omegamax.cli {...}
│
├── notebooks/
│   └── walkthrough.ipynb
├── data/                          # gitignored
├── checkpoints/                   # gitignored
├── figures/                       # versioned
├── configs/
│   ├── gbdt.yaml
│   └── cgcnn.yaml
└── tests/
    ├── conftest.py
    ├── fixtures/
    ├── test_sample.py
    ├── test_magpie.py
    ├── test_graph.py
    ├── test_gbdt.py
    ├── test_cgcnn.py
    ├── test_metrics.py
    └── test_smoke.py
```

### Data flow

```
matbench.MatbenchBenchmark.phonons
     ↓
load.py → list[Sample]   (structure, target, mp_id)
     ↓
       ┌───────────────┴───────────────┐
       ↓                               ↓
features/magpie.py            features/graph.py
   132-dim X                  Data objects (atoms, edges)
       ↓                               ↓
models/gbdt.py                  models/cgcnn.py
   (5-fold CV)                  (5-fold CV w/ ensemble seeds)
       ↓                               ↓
       └───────────────┬───────────────┘
                       ↓
                 eval/metrics.py
                 eval/parity.py
                 eval/leaderboard.py
```

### Key interface contracts

1. **`Sample` dataclass** — single shared input record.
   Fields: `mp_id: str, structure: pymatgen.Structure, target: float`.
2. **`Featurizer` protocol** —
   `__call__(samples: list[Sample]) -> np.ndarray | list[Data]`.
3. **`Regressor` protocol** —
   `.fit(features, targets)`, `.predict(features) -> np.ndarray`.
   Both GBDT and CGCNN implement this.
4. **Matbench fold contract** —
   `split.matbench_folds() -> Iterator[tuple[list[int], list[int]]]`
   yielding (train_idx, test_idx) for each of 5 folds.

### CGCNN model architecture

Hand-implemented per Xie & Grossman 2018, sized to be CPU-trainable:

```
Atom embedding: 92-dim one-hot → 64-dim learned
Edge embedding: 41-dim Gaussian basis on distances (0–8 Å) → 64-dim
3 ConvLayers (gated, 64 → 64)
Pooling: mean over atoms in unit cell
Head: Linear(64 → 128) → ReLU → Dropout(0.2) → Linear(128 → 1)
```

~50k params. Forward pass on a batch of 64 structures: <1 s on CPU.

### Dependencies

- `matbench` — the benchmark library (bundles the data)
- `matminer` — Magpie features
- `pymatgen` — structure handling
- `torch` + `torch-geometric` — GNN training
- `scikit-learn`, `numpy`, `pandas`, `pyarrow`, `matplotlib`, `pyyaml`
- dev: `pytest`, `pytest-mock`

### macOS gotcha

`xanes-oxstate` discovered that on macOS, `lightgbm` co-loaded with
`torch` causes a libomp dual-load deadlock. We're not using `lightgbm`
here (sklearn HistGradientBoosting only), so the issue doesn't apply.
We still set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE` in the Makefile for belt-and-suspenders.

---

## 3. Data

### Source

Matbench `phonons` task, accessed via the `matbench` package.
Data is bundled via figshare and downloaded on first use — no API
key needed.

- **1,265 materials** after Matbench's standard filtering
- **Target**: ω_max in cm⁻¹ (last DOS peak frequency)
- **Input**: `pymatgen.core.Structure` per material
- **License**: CC-BY 4.0 (Materials Project)

### Loading

`data/load.py` exposes:

```python
def load_phonons() -> list[Sample]:
    """Pull matbench_phonons; return canonical Sample list."""
```

Internally calls
`matbench.bench.MatbenchBenchmark().matbench_phonons.load()`,
iterates the dataframe, builds one `Sample(mp_id, structure, target)`
per row. Cached to `data/cache/phonons.parquet` after first load.

### Cleaning

Matbench is pre-cleaned. The only added checks:
1. `len(structure) >= 1` (defensive).
2. `target` is finite and positive (defensive).
3. Log and drop failures with `mp_id` and reason.

Expected drops: 0. The check exists so an upstream-change regression
fails loudly.

### Splits (Matbench protocol)

Matbench dictates the exact splits for leaderboard comparability.
We use them verbatim:

```python
def matbench_folds() -> Iterator[tuple[list[int], list[int]]]:
    """Yields (train_idx, test_idx) for each of 5 outer folds."""
```

For hyperparameter tuning we further split each train fold 80/20
train/val internally. Test indices come from Matbench and are
never touched during tuning.

Final reported metric: **mean MAE across 5 outer folds**, ± std.
This is the leaderboard convention.

### Caching

- `data/cache/phonons.parquet` — cleaned samples (gitignored).
- `data/cache/magpie.npy` — pre-computed Magpie matrix (~1265 × 132).
- `data/cache/graphs/` — pre-built CGCNN graphs as `.pt` files.

All three regenerated by `make data`.

### Dataset card

`docs/data_card.md` covers:
- Source, license, Matbench version tag.
- Distribution of ω_max (histogram, percentiles).
- Element coverage.
- Known limitations: derived scalar target (not full curve); DFT not
  experimental.

---

## 4. Methods

### 4.1 Phase A: Magpie + GBDT

**Features** (`features/magpie.py`):

```python
from matminer.featurizers.composition import ElementProperty
featurizer = ElementProperty.from_preset("magpie")  # 132 features
X = featurizer.featurize_many(
    [s.structure.composition for s in samples]
)
```

132-dim element-property statistics (mean/min/max/std of atomic
mass, electronegativity, etc.). Cached as `data/cache/magpie.npy`.

**Model** (`models/gbdt.py`):

```python
from sklearn.ensemble import HistGradientBoostingRegressor

model = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
    early_stopping=True, validation_fraction=0.2,
    random_state=0,
)
```

Hyperparameters in `configs/gbdt.yaml`. Each fold trains in <30 s on CPU.

**Per-fold protocol**:
1. Train on `train_idx` (with internal 80/20 train/val for early stop).
2. Predict on `test_idx`.
3. Record MAE, R², predictions.

Report mean ± std over the 5 outer folds plus the per-fold prediction array.

### 4.2 Phase B: CGCNN

**Graph construction** (`features/graph.py`):

For each Sample:
1. Atom features: 92-dim one-hot of element identity.
2. Edges: neighbor list within 8 Å via
   `pymatgen.core.Structure.get_neighbor_list`.
3. Edge features: 41-dim Gaussian basis on distance, centered
   uniformly in `[0, 8]` Å with σ = 0.2.
4. Returned as `torch_geometric.data.Data`.

Cached as `data/cache/graphs/<mp_id>.pt`.

**CGCNN architecture** (`models/cgcnn.py`):

Hand-implemented gated convolution from Xie & Grossman 2018:

```python
class CGCNNConv(nn.Module):
    """m_ij = σ(W1·[h_i, h_j, e_ij]) ⊗ tanh(W2·[h_i, h_j, e_ij])"""

class CGCNN(nn.Module):
    def __init__(self, n_atom_feats=92, n_edge_feats=41,
                 hidden=64, n_conv=3, n_dense=128, dropout=0.2):
        # atom emb → 3 conv layers → mean pool → MLP → scalar
```

~50k params.

**Training**:
- 200 epochs max, early stop on val MAE, patience 30.
- Adam, lr 1e-3, cosine decay to 1e-5.
- Batch size 64.
- Loss: smooth-L1 (Huber, δ = 10 cm⁻¹) — robust to ω_max outliers.

5 seeds × 5 folds = 25 runs. Total ~5 h on CPU. Checkpoint
best-val per (fold, seed). For prediction: ensemble the 5 seeds
within each fold (average raw outputs).

### 4.3 Comparison protocol

Both models evaluated on identical Matbench test folds for each of
the 5 outer folds. Direct A/B test, no contamination possible.

`eval/leaderboard.py` produces the markdown table:

```
Model                | MAE (cm⁻¹) | R²    | Δ vs Magpie
---------------------|------------|-------|-------------
Magpie + GBDT (ours) |  X.X ± X.X | 0.XXX | -
CGCNN (ours)         |  X.X ± X.X | 0.XXX | +X.X
---
Roost (published)    |       75.6 | -     | -
CGCNN (published)    |       57.8 | -     | -
MEGNet (published)   |       49.9 | -     | -
ALIGNN (published)   |       29.5 | -     | -
```

Published numbers come from the Matbench v0.1 leaderboard, frozen.

### 4.4 Physics-finding pass

After both models trained, take test predictions across all 5 folds.
Find materials where CGCNN beats GBDT by > 50 cm⁻¹ absolute error.
Look for patterns:

- Are they framework / layered / 1D structures?
  Compute Robocrystallographer-style descriptors via
  `pymatgen.analysis.dimensionality` and bin.
- Are they polymorphs of common compositions?
  Group by reduced formula; check if same-composition different-
  structure pairs sit on different sides of the gap.
- Are they high-symmetry vs low-symmetry?
  Bin by space group.

2–4 sentences per pattern in `docs/physics_findings.md`. The story:
"structure helps where composition is degenerate." Concrete
material examples make it credible.

### 4.5 Explicitly not in scope

- Hyperparameter search (GBDT defaults; CGCNN matches published).
- Equivariant nets.
- Uncertainty beyond fold std (no MC dropout, no quantile regression).
- Test-time augmentation.

---

## 5. Evaluation & deliverables

### 5.1 Metrics

- **MAE** (primary), cm⁻¹, on the test fold per outer CV iteration.
- **R²** on the test fold.
- **Per-fold std** — robustness across the 5 splits.
- **Residual quantiles** (P05, P50, P95) — fat-tail check.

All reported in `metrics/summary.json` plus per-fold breakdowns in
`metrics/fold_{0..4}.json`.

### 5.2 Headline figure

Single figure, two panels:

**Panel A — Parity plot** (predicted vs. true ω_max). All 1,265
materials shown (each prediction from its outer-CV-held-out fold).
Two scatters overlaid:
- Magpie + GBDT in grey
- CGCNN ensemble in blue
- y = x diagonal line, ± 50 cm⁻¹ shaded band

**Panel B — Residual distribution** (Δ = predicted − true).
KDE per model. Vertical dashed line at 0; ± MAE band shaded.

PDF + PNG, 300 DPI, caption-ready.

### 5.3 Leaderboard table

Markdown table in README and `figures/leaderboard.md`. Our two rows
plus the four published baselines, frozen.

### 5.4 Pass/fail checklist

- [ ] Magpie + GBDT mean MAE ≤ 80 cm⁻¹ across 5 folds.
- [ ] CGCNN mean MAE ≤ 65 cm⁻¹.
- [ ] CGCNN MAE strictly lower than Magpie + GBDT MAE.
- [ ] ≥ 3 physics-finding bullets in `docs/physics_findings.md`.
- [ ] Mean residual ∈ [−5, 5] cm⁻¹ for both models.
- [ ] `pip install -e . && make all` reproduces both panels in < 6 h.
- [ ] Data card, methods doc, leaderboard table all populated.

### 5.5 Repository deliverables

```
README.md                      ← portfolio writeup, 800–1500 words
notebooks/walkthrough.ipynb    ← single-fold end-to-end demo
figures/headline.{pdf,png}     ← 2-panel parity + residual
figures/leaderboard.md         ← standalone leaderboard table
docs/data_card.md
docs/methods.md
docs/physics_findings.md
configs/gbdt.yaml, configs/cgcnn.yaml
metrics/summary.json
metrics/fold_{0..4}.json
```

README sections:
1. One-line pitch + headline figure inline.
2. Leaderboard table.
3. How to reproduce (`make all`).
4. Short methods overview pointing at `docs/methods.md`.
5. Physics findings as bullets pointing at `docs/physics_findings.md`.
6. Limitations + future work (equivariant nets, full DOS prediction).

---

## 6. Open questions for implementation phase

These are flagged for resolution during implementation, not in this spec:

1. **CGCNN training wall-clock vs spec target.** 5 seeds × 5 folds at
   ~10 min each is ~5 h. If wall-clock starts running long during
   implementation, dropping to 3 seeds × 5 folds (~3 h) is an
   acceptable degradation. The leaderboard story doesn't depend on
   ensemble size.
2. **Matbench release tag.** Pin to a specific Matbench version once
   we confirm install works; record in data card.
3. **Robocrystallographer descriptor library.** `pymatgen.analysis.
   dimensionality` provides dimensionality classification. If it's
   too coarse for the physics-finding pass, fall back to manual
   space-group binning.
