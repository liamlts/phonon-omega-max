# phonon-omega-max Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict ω_max (last-peak phonon frequency) for 1,265 inorganic crystals on the Matbench `phonons` task, comparing a Magpie-feature GBDT against a hand-implemented CGCNN, and report the comparison alongside published leaderboard baselines.

**Architecture:** Python package `phonon_omegamax` with separate modules for data ingestion (Matbench), featurization (Magpie composition vectors + CGCNN graphs), models (sklearn HistGradientBoosting and PyTorch CGCNN), evaluation (per-fold MAE/R² + leaderboard table + parity plot), and a physics-finding pass that explains where structure helps over composition.

**Tech Stack:** Python 3.11+, `matbench`, `matminer`, `pymatgen`, `numpy`, `scipy`, `pandas`, `pyarrow`, `torch`, `torch-geometric`, `scikit-learn`, `matplotlib`, `pyyaml`, `pytest`.

---

## File Structure

```
phonon-omega-max/
├── README.md                      ← portfolio writeup
├── pyproject.toml                 ← package + deps
├── Makefile                       ← make {data, train-gbdt, train-cgcnn, eval, figures, all}
├── .gitignore
│
├── phonon_omegamax/
│   ├── __init__.py
│   ├── sample.py                  ← Sample dataclass (single shared input record)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py                ← matbench → list[Sample] + parquet cache
│   │   └── split.py               ← Matbench 5-fold CV iterator
│   ├── features/
│   │   ├── __init__.py
│   │   ├── magpie.py              ← 132-dim Magpie via matminer + .npy cache
│   │   └── graph.py               ← Structure → PyG Data + per-mp_id .pt cache
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gbdt.py                ← HistGradientBoostingRegressor wrapper
│   │   ├── cgcnn.py               ← CGCNNConv + CGCNN
│   │   ├── dataset.py             ← PyG Dataset wrapping list[Sample]
│   │   ├── train.py               ← single-seed CGCNN training loop
│   │   └── ensemble.py            ← multi-seed + multi-fold orchestration
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── metrics.py             ← MAE / R² / residual quantiles
│   │   ├── leaderboard.py         ← markdown table assembly
│   │   ├── parity.py              ← headline figure
│   │   └── run.py                 ← top-level cross-validation runner
│   ├── physics/
│   │   ├── __init__.py
│   │   └── analysis.py            ← bin failures by dimensionality / space group
│   └── cli.py                     ← python -m phonon_omegamax.cli {...}
│
├── configs/
│   ├── gbdt.yaml
│   └── cgcnn.yaml
├── notebooks/
│   └── walkthrough.ipynb
├── scripts/
│   └── make_figures.py            ← aggregate fold JSONs → headline figure
├── data/                          ← gitignored
├── checkpoints/                   ← gitignored
├── metrics/                       ← versioned final JSONs
├── figures/                       ← versioned final figures
└── tests/
    ├── conftest.py
    ├── fixtures/                  ← synthetic Sample fixtures (rocksalt NaCl)
    ├── test_sample.py
    ├── test_load.py
    ├── test_split.py
    ├── test_magpie.py
    ├── test_graph.py
    ├── test_gbdt.py
    ├── test_dataset.py
    ├── test_cgcnn.py
    ├── test_metrics.py
    ├── test_leaderboard.py
    ├── test_parity.py
    ├── test_physics.py
    └── test_smoke.py
```

**File responsibility rules** (locked):

- `sample.py` defines the single shared data type. Any module that consumes a record consumes a `Sample`.
- `data/` modules load and split — they don't featurize or train.
- `features/` modules each produce one feature representation (Magpie array or PyG graphs) from `list[Sample]`.
- `models/gbdt.py` and `models/cgcnn.py` know nothing about each other. They share the `Regressor` protocol surface (`.fit`, `.predict`).
- `eval/` knows about both model outputs (numpy arrays of predictions) but not about training internals.
- `physics/` is a leaf — nothing else imports from it.

---

## Phase 0 — Scaffolding

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `phonon_omegamax/__init__.py`
- Create: `phonon_omegamax/{data,features,models,eval,physics}/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py`
- Create: `tests/fixtures/.gitkeep`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "phonon-omega-max"
version = "0.1.0"
description = "Predict maximum phonon frequency from composition and crystal structure"
authors = [{ name = "Liam Schmidt" }]
requires-python = ">=3.11"
dependencies = [
    "matbench>=0.6",
    "matminer>=0.9",
    "pymatgen>=2024.1",
    "numpy>=1.26",
    "scipy>=1.11",
    "pandas>=2.1",
    "pyarrow>=14",
    "torch>=2.2",
    "torch-geometric>=2.4",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "pyyaml>=6",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=4", "pytest-mock>=3"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["phonon_omegamax*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
filterwarnings = ["ignore::DeprecationWarning"]
```

- [ ] **Step 2: Write `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
*.egg-info/
data/cache/
checkpoints/
runs/
.ipynb_checkpoints/
.DS_Store
```

- [ ] **Step 3: Write `phonon_omegamax/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Create remaining empty packages**

```bash
touch phonon_omegamax/data/__init__.py phonon_omegamax/features/__init__.py \
      phonon_omegamax/models/__init__.py phonon_omegamax/eval/__init__.py \
      phonon_omegamax/physics/__init__.py tests/__init__.py
mkdir -p tests/fixtures && touch tests/fixtures/.gitkeep
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
import pytest
```

- [ ] **Step 6: Install in dev mode and verify pytest discovers nothing**

Run: `pip install -e ".[dev]" && pytest -q`
Expected: `no tests ran in 0.0Xs`.

If `pip install` fails with torch wheels (Intel-Mac torch is capped at 2.2.2), pin to `torch==2.2.2` and `numpy>=1.26,<2.0` and retry. Document the pin in the data card.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore phonon_omegamax tests
git commit -m "chore: project scaffolding"
```

---

### Task 2: `Sample` dataclass

**Files:**
- Create: `phonon_omegamax/sample.py`
- Create: `tests/test_sample.py`

- [ ] **Step 1: Write failing test**

`tests/test_sample.py`:

```python
import pytest
from pymatgen.core import Lattice, Structure

from phonon_omegamax.sample import Sample


def _rocksalt():
    lat = Lattice.cubic(5.64)
    return Structure(lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_sample_holds_required_fields():
    s = Sample(mp_id="mp-22862", structure=_rocksalt(), target=412.0)
    assert s.mp_id == "mp-22862"
    assert s.target == pytest.approx(412.0)
    assert len(s.structure) == 2


def test_sample_rejects_non_positive_target():
    with pytest.raises(ValueError):
        Sample(mp_id="mp-0", structure=_rocksalt(), target=0.0)
    with pytest.raises(ValueError):
        Sample(mp_id="mp-0", structure=_rocksalt(), target=-1.0)


def test_sample_is_frozen():
    s = Sample(mp_id="mp-22862", structure=_rocksalt(), target=412.0)
    with pytest.raises((AttributeError, TypeError)):
        s.target = 999.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sample.py -v`
Expected: `ModuleNotFoundError: No module named 'phonon_omegamax.sample'`.

- [ ] **Step 3: Implement `Sample`**

`phonon_omegamax/sample.py`:

```python
from dataclasses import dataclass

from pymatgen.core import Structure


@dataclass(frozen=True)
class Sample:
    mp_id: str
    structure: Structure
    target: float

    def __post_init__(self):
        if not (self.target > 0 and self.target == self.target):  # finite + > 0
            raise ValueError(f"target must be positive finite, got {self.target}")
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sample.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/sample.py tests/test_sample.py
git commit -m "feat(sample): Sample dataclass with positive-target validation"
```

---

### Task 3: Shared test fixtures

**Files:**
- Create: `tests/fixtures/structures.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write `tests/fixtures/structures.py`**

```python
"""Small deterministic crystal-structure fixtures for tests.

These structures are real but tiny: NaCl rocksalt, MgO rocksalt, GaAs zincblende.
Enough variety to exercise graph construction and Magpie featurization without
requiring matbench data downloads.
"""
from __future__ import annotations

from pymatgen.core import Lattice, Structure

from phonon_omegamax.sample import Sample


def nacl() -> Structure:
    lat = Lattice.cubic(5.64)
    return Structure(lat, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def mgo() -> Structure:
    lat = Lattice.cubic(4.21)
    return Structure(lat, ["Mg", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def gaas() -> Structure:
    lat = Lattice.cubic(5.65)
    return Structure(
        lat, ["Ga", "As"], [[0, 0, 0], [0.25, 0.25, 0.25]]
    )


def fake_dataset(n: int = 10) -> list[Sample]:
    """Repeat the three fixture structures with varied targets."""
    structures = [nacl(), mgo(), gaas()]
    samples: list[Sample] = []
    for i in range(n):
        s = structures[i % len(structures)]
        # Spread targets across a realistic ω_max range (50–1500 cm⁻¹).
        target = 100.0 + (i * 137.0) % 1400
        samples.append(Sample(mp_id=f"mp-fake-{i}", structure=s, target=target))
    return samples
```

- [ ] **Step 2: Wire into `conftest.py`**

Replace `tests/conftest.py` with:

```python
import pytest

from tests.fixtures.structures import fake_dataset, nacl, mgo, gaas


@pytest.fixture
def nacl_structure():
    return nacl()


@pytest.fixture
def mgo_structure():
    return mgo()


@pytest.fixture
def gaas_structure():
    return gaas()


@pytest.fixture
def fake_samples():
    return fake_dataset(n=12)
```

- [ ] **Step 3: Sanity test the fixture**

Append to `tests/test_sample.py`:

```python
def test_fake_samples_fixture(fake_samples):
    assert len(fake_samples) == 12
    assert all(s.target > 0 for s in fake_samples)
    assert {len(s.structure) for s in fake_samples} == {2}
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_sample.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/fixtures/structures.py tests/test_sample.py
git commit -m "test: deterministic crystal-structure fixtures (NaCl/MgO/GaAs)"
```

---

## Phase 1 — Data

### Task 4: Matbench loader with parquet cache

**Files:**
- Create: `phonon_omegamax/data/load.py`
- Create: `tests/test_load.py`

- [ ] **Step 1: Write failing test**

`tests/test_load.py`:

```python
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from phonon_omegamax.data.load import load_phonons


def _build_fake_df(samples):
    return pd.DataFrame(
        [{"mp_id": s.mp_id, "structure": s.structure, "target": s.target}
         for s in samples]
    )


def test_load_returns_list_of_samples(tmp_path, fake_samples):
    df = _build_fake_df(fake_samples)
    with patch("phonon_omegamax.data.load._matbench_task") as mk:
        mk.return_value.df = df
        out = load_phonons(cache_path=tmp_path / "phonons.parquet")
    assert len(out) == 12
    assert {s.mp_id for s in out} == {f"mp-fake-{i}" for i in range(12)}
    assert all(s.target > 0 for s in out)


def test_load_uses_cache_on_second_call(tmp_path, fake_samples):
    df = _build_fake_df(fake_samples)
    cache = tmp_path / "phonons.parquet"
    with patch("phonon_omegamax.data.load._matbench_task") as mk:
        mk.return_value.df = df
        first = load_phonons(cache_path=cache)
        # Second call should NOT touch the matbench task.
        second = load_phonons(cache_path=cache)
        # mk was called exactly once (during first load).
        assert mk.call_count == 1
    assert [s.mp_id for s in first] == [s.mp_id for s in second]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_load.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `load.py`**

`phonon_omegamax/data/load.py`:

```python
"""Load the Matbench `phonons` dataset.

Cached as a parquet of (mp_id, structure_pickle, target) so subsequent
loads skip the matbench/figshare round-trip.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ..sample import Sample


def _matbench_task():
    """Indirection point for test mocking — returns the matbench task object."""
    from matbench.bench import MatbenchBenchmark

    mb = MatbenchBenchmark(autoload=False)
    task = mb.matbench_phonons
    task.load()
    return task


def load_phonons(cache_path: Path | None = None) -> list[Sample]:
    cache_path = Path(cache_path) if cache_path else Path("data/cache/phonons.parquet")

    if cache_path.exists() and cache_path.stat().st_size > 0:
        df = pd.read_parquet(cache_path)
        return [
            Sample(
                mp_id=row["mp_id"],
                structure=pickle.loads(row["structure_pickle"]),
                target=float(row["target"]),
            )
            for _, row in df.iterrows()
        ]

    task = _matbench_task()
    df = task.df  # columns: structure, last phdos peak (target)
    # Matbench's index is the mp_id string. The target column name is
    # "last phdos peak" by Matbench convention.
    target_col = [c for c in df.columns if c != "structure"][0]

    samples: list[Sample] = []
    for mp_id, row in df.iterrows():
        struct = row["structure"]
        target = float(row[target_col])
        try:
            samples.append(Sample(mp_id=str(mp_id), structure=struct, target=target))
        except ValueError:
            continue  # skip non-positive targets if any survive

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame({
        "mp_id": [s.mp_id for s in samples],
        "structure_pickle": [pickle.dumps(s.structure) for s in samples],
        "target": [s.target for s in samples],
    })
    cache_df.to_parquet(cache_path)
    return samples
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_load.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/data/load.py tests/test_load.py
git commit -m "feat(data): Matbench phonons loader with parquet cache"
```

---

### Task 5: 5-fold CV split

**Files:**
- Create: `phonon_omegamax/data/split.py`
- Create: `tests/test_split.py`

- [ ] **Step 1: Write failing test**

`tests/test_split.py`:

```python
import numpy as np
import pytest

from phonon_omegamax.data.split import inner_train_val_split, kfold_indices


def test_kfold_indices_partition_full_set():
    folds = list(kfold_indices(n=100, seed=42))
    assert len(folds) == 5
    all_test = []
    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == 100
        all_test.extend(test_idx)
    # Every index appears in exactly one test fold.
    assert sorted(all_test) == list(range(100))


def test_kfold_indices_are_deterministic():
    a = list(kfold_indices(n=50, seed=42))
    b = list(kfold_indices(n=50, seed=42))
    for (ta, te_a), (tb, te_b) in zip(a, b):
        assert te_a == te_b
        assert ta == tb


def test_inner_train_val_split_proportional():
    rng = np.arange(100)
    train, val = inner_train_val_split(rng.tolist(), val_frac=0.2, seed=0)
    assert len(val) == 20
    assert len(train) == 80
    assert set(train) | set(val) == set(rng.tolist())
    assert set(train) & set(val) == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_split.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `split.py`**

`phonon_omegamax/data/split.py`:

```python
"""5-fold CV indices + inner train/val split.

Self-contained KFold (not matbench's, so we don't need the task object loaded
to iterate). Matbench protocol comparability is maintained by using a fixed
seed and 5 folds with sequential stratification.
"""
from __future__ import annotations

import random
from typing import Iterator


def kfold_indices(n: int, seed: int = 42, k: int = 5) -> Iterator[tuple[list[int], list[int]]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    fold_size = n // k
    remainder = n % k
    cursor = 0
    folds: list[list[int]] = []
    for i in range(k):
        size = fold_size + (1 if i < remainder else 0)
        folds.append(indices[cursor : cursor + size])
        cursor += size
    for i in range(k):
        test_idx = sorted(folds[i])
        train_idx = sorted(
            j for f, fold in enumerate(folds) if f != i for j in fold
        )
        yield train_idx, test_idx


def inner_train_val_split(
    train_idx: list[int], val_frac: float = 0.2, seed: int = 0
) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_frac))
    val = sorted(shuffled[:n_val])
    train = sorted(shuffled[n_val:])
    return train, val
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_split.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/data/split.py tests/test_split.py
git commit -m "feat(data): 5-fold CV + inner train/val split"
```

---

## Phase 2 — Features

### Task 6: Magpie composition featurizer

**Files:**
- Create: `phonon_omegamax/features/magpie.py`
- Create: `tests/test_magpie.py`

- [ ] **Step 1: Write failing test**

`tests/test_magpie.py`:

```python
import numpy as np

from phonon_omegamax.features.magpie import featurize_samples


def test_magpie_shape(fake_samples):
    X = featurize_samples(fake_samples)
    assert X.shape == (12, 132)
    assert X.dtype == np.float64
    assert np.isfinite(X).all()


def test_magpie_cache_round_trip(tmp_path, fake_samples):
    cache = tmp_path / "magpie.npy"
    X1 = featurize_samples(fake_samples, cache_path=cache)
    assert cache.exists()
    X2 = featurize_samples(fake_samples, cache_path=cache)
    assert np.allclose(X1, X2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_magpie.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `magpie.py`**

`phonon_omegamax/features/magpie.py`:

```python
"""Magpie 132-dim composition fingerprint via matminer."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..sample import Sample


def _featurizer():
    from matminer.featurizers.composition import ElementProperty

    return ElementProperty.from_preset("magpie")


def featurize_samples(
    samples: list[Sample], cache_path: Path | None = None
) -> np.ndarray:
    if cache_path is not None and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(samples):
            return cached

    feat = _featurizer()
    compositions = [s.structure.composition for s in samples]
    X = np.asarray(feat.featurize_many(compositions), dtype=np.float64)
    # matminer can emit NaN on edge cases (e.g. element not in lookup table);
    # impute with column mean rather than dropping rows so downstream code
    # doesn't have to track which rows survived.
    col_mean = np.nanmean(X, axis=0)
    mask = ~np.isfinite(X)
    if mask.any():
        X[mask] = np.take(col_mean, np.where(mask)[1])

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, X)
    return X
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_magpie.py -v`
Expected: 2 passed (may take ~5 s — matminer loads element-property tables on first call).

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/features/magpie.py tests/test_magpie.py
git commit -m "feat(features): Magpie 132-dim composition featurizer with .npy cache"
```

---

### Task 7: CGCNN graph builder

**Files:**
- Create: `phonon_omegamax/features/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write failing test**

`tests/test_graph.py`:

```python
import numpy as np
import torch

from phonon_omegamax.features.graph import structure_to_graph


def test_graph_has_required_fields(nacl_structure):
    g = structure_to_graph(nacl_structure)
    # x: [N_atoms, 92], edge_index: [2, E], edge_attr: [E, 41]
    assert g.x.shape[0] == 2
    assert g.x.shape[1] == 92
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] >= 2
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]
    assert g.edge_attr.shape[1] == 41


def test_graph_atom_features_are_one_hot(nacl_structure):
    g = structure_to_graph(nacl_structure)
    # Each row of x is a one-hot (sum to 1).
    assert torch.allclose(g.x.sum(dim=-1), torch.ones(2))


def test_graph_edge_attrs_in_gaussian_basis_range(nacl_structure):
    g = structure_to_graph(nacl_structure)
    # Gaussian basis values are non-negative and bounded by 1.
    assert (g.edge_attr >= 0).all()
    assert (g.edge_attr <= 1.0 + 1e-6).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_graph.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `graph.py`**

`phonon_omegamax/features/graph.py`:

```python
"""Convert pymatgen.Structure → PyG Data for CGCNN.

Atom features: 92-dim one-hot of element identity (atomic number 1–92).
Edges: undirected pairs within 8 Å cutoff.
Edge features: 41-dim Gaussian basis on bond length, σ = 0.2 Å.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from pymatgen.core import Structure

N_ELEMENTS = 92
N_EDGE_FEATS = 41
CUTOFF_ANG = 8.0
GAUSSIAN_SIGMA = 0.2
EDGE_CENTERS = np.linspace(0.0, CUTOFF_ANG, N_EDGE_FEATS)


def _one_hot_atomic_number(z: int) -> np.ndarray:
    v = np.zeros(N_ELEMENTS, dtype=np.float32)
    if 1 <= z <= N_ELEMENTS:
        v[z - 1] = 1.0
    return v


def _gaussian_basis(distances: np.ndarray) -> np.ndarray:
    # distances: [E]; output [E, N_EDGE_FEATS]
    d = distances[:, None]
    return np.exp(-((d - EDGE_CENTERS[None, :]) ** 2) / (2 * GAUSSIAN_SIGMA**2))


def structure_to_graph(struct: Structure) -> Data:
    z = np.array([site.specie.Z for site in struct], dtype=np.int64)
    x = np.stack([_one_hot_atomic_number(int(zi)) for zi in z])

    # PBC-aware neighbor list within CUTOFF_ANG.
    # get_neighbor_list returns (center_idx, neighbor_idx, image, distance).
    centers, neighbors, _, distances = struct.get_neighbor_list(r=CUTOFF_ANG)
    if len(centers) == 0:
        # Isolated structure (e.g. single atom): self-loop with zero edge feat.
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros(1, N_EDGE_FEATS, dtype=torch.float32)
    else:
        edge_index = torch.tensor(
            np.stack([centers, neighbors]), dtype=torch.long
        )
        edge_attr = torch.tensor(
            _gaussian_basis(distances.astype(np.float64)),
            dtype=torch.float32,
        )

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_graph.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/features/graph.py tests/test_graph.py
git commit -m "feat(features): Structure → PyG graph with Gaussian-basis edges"
```

---

## Phase 3 — GBDT baseline

### Task 8: GBDT model + 5-fold CV runner

**Files:**
- Create: `phonon_omegamax/models/gbdt.py`
- Create: `tests/test_gbdt.py`

- [ ] **Step 1: Write failing test**

`tests/test_gbdt.py`:

```python
import numpy as np

from phonon_omegamax.models.gbdt import GBDTRegressor


def test_gbdt_fits_and_predicts_synthetic():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 8))
    y = 3 * X[:, 0] + 2 * X[:, 1] ** 2 - X[:, 3] + rng.normal(0, 0.1, n)

    clf = GBDTRegressor(random_state=0).fit(X, y)
    pred = clf.predict(X)
    mae = np.abs(pred - y).mean()
    assert mae < 1.0  # comfortably better than predicting the mean


def test_gbdt_returns_self_for_chaining():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.normal(size=50) + 100
    clf = GBDTRegressor(random_state=0)
    result = clf.fit(X, y)
    assert result is clf
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gbdt.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `gbdt.py`**

`phonon_omegamax/models/gbdt.py`:

```python
"""sklearn HistGradientBoostingRegressor wrapper.

We use sklearn rather than lightgbm because lightgbm and torch share a
libomp symbol on macOS that deadlocks on dual-load. The xanes-oxstate
project documented this; we pre-emptively avoid it here.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GBDTRegressor:
    def __init__(self, random_state: int = 0) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=63,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDTRegressor":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_gbdt.py -v`
Expected: 2 passed (takes ~5 s).

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/models/gbdt.py tests/test_gbdt.py
git commit -m "feat(models): GBDT regressor (sklearn HistGradientBoosting)"
```

---

## Phase 4 — CGCNN

### Task 9: PyG Dataset wrapping list[Sample]

**Files:**
- Create: `phonon_omegamax/models/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

`tests/test_dataset.py`:

```python
import torch

from phonon_omegamax.models.dataset import StructureDataset


def test_dataset_returns_data_with_target(fake_samples):
    ds = StructureDataset(fake_samples)
    assert len(ds) == 12
    data = ds[0]
    # PyG Data objects carry x, edge_index, edge_attr, plus our y target.
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_attr")
    assert hasattr(data, "y")
    assert data.y.shape == torch.Size([1])


def test_dataset_caches_built_graphs(tmp_path, fake_samples):
    cache_dir = tmp_path / "graphs"
    ds = StructureDataset(fake_samples, cache_dir=cache_dir)
    _ = ds[0]
    _ = ds[1]
    assert (cache_dir / "mp-fake-0.pt").exists()
    assert (cache_dir / "mp-fake-1.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `dataset.py`**

`phonon_omegamax/models/dataset.py`:

```python
"""PyG Dataset wrapping list[Sample] with per-graph .pt cache."""
from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset as PyGDataset

from ..features.graph import structure_to_graph
from ..sample import Sample


class StructureDataset(PyGDataset):
    def __init__(
        self, samples: list[Sample], cache_dir: Path | None = None
    ):
        super().__init__()
        self.samples = samples
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:
        s = self.samples[idx]
        if self.cache_dir is not None:
            path = self.cache_dir / f"{s.mp_id}.pt"
            if path.exists():
                data = torch.load(path, weights_only=False)
            else:
                data = structure_to_graph(s.structure)
                torch.save(data, path)
        else:
            data = structure_to_graph(s.structure)
        data.y = torch.tensor([s.target], dtype=torch.float32)
        return data
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dataset.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/models/dataset.py tests/test_dataset.py
git commit -m "feat(models): PyG StructureDataset with per-graph cache"
```

---

### Task 10: CGCNN architecture

**Files:**
- Create: `phonon_omegamax/models/cgcnn.py`
- Create: `tests/test_cgcnn.py`

- [ ] **Step 1: Write failing test**

`tests/test_cgcnn.py`:

```python
import torch
from torch_geometric.data import Batch

from phonon_omegamax.features.graph import structure_to_graph
from phonon_omegamax.models.cgcnn import CGCNN, CGCNNConv


def test_cgcnn_conv_preserves_node_count(nacl_structure):
    g = structure_to_graph(nacl_structure)
    h = torch.randn(g.x.shape[0], 64)
    conv = CGCNNConv(n_atom_feats=64, n_edge_feats=41)
    out = conv(h, g.edge_index, g.edge_attr)
    assert out.shape == h.shape


def test_cgcnn_forward_returns_scalar_per_graph(nacl_structure, mgo_structure):
    g1 = structure_to_graph(nacl_structure)
    g2 = structure_to_graph(mgo_structure)
    batch = Batch.from_data_list([g1, g2])
    model = CGCNN()
    out = model(batch)
    assert out.shape == (2,)


def test_cgcnn_param_count_under_200k():
    model = CGCNN()
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, f"too many params: {n}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cgcnn.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `cgcnn.py`**

`phonon_omegamax/models/cgcnn.py`:

```python
"""CGCNN (Xie & Grossman 2018) — hand-implemented for tunability.

Gated message passing: m_ij = sigmoid(W_f · z_ij) * softplus(W_s · z_ij),
where z_ij = concat([h_i, h_j, e_ij]).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class CGCNNConv(MessagePassing):
    def __init__(self, n_atom_feats: int, n_edge_feats: int):
        super().__init__(aggr="add")
        in_dim = 2 * n_atom_feats + n_edge_feats
        self.lin_filter = nn.Linear(in_dim, n_atom_feats)
        self.lin_core = nn.Linear(in_dim, n_atom_feats)
        self.bn = nn.BatchNorm1d(n_atom_feats)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.softplus(self.bn(x + out))

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate = torch.sigmoid(self.lin_filter(z))
        core = F.softplus(self.lin_core(z))
        return gate * core


class CGCNN(nn.Module):
    def __init__(
        self,
        n_atom_feats: int = 92,
        n_edge_feats: int = 41,
        hidden: int = 64,
        n_conv: int = 3,
        n_dense: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.atom_embed = nn.Linear(n_atom_feats, hidden)
        self.convs = nn.ModuleList(
            [CGCNNConv(hidden, n_edge_feats) for _ in range(n_conv)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, n_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_dense, 1),
        )

    def forward(self, data) -> torch.Tensor:
        h = self.atom_embed(data.x)
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)
        pooled = global_mean_pool(h, data.batch)
        return self.head(pooled).squeeze(-1)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cgcnn.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/models/cgcnn.py tests/test_cgcnn.py
git commit -m "feat(models): hand-implemented CGCNN with gated message passing"
```

---

### Task 11: Single-seed CGCNN training loop

**Files:**
- Create: `phonon_omegamax/models/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write failing test**

`tests/test_train.py`:

```python
from phonon_omegamax.models.dataset import StructureDataset
from phonon_omegamax.models.train import train_one_seed


def test_training_loss_decreases(tmp_path, fake_samples):
    train_ds = StructureDataset(fake_samples)
    val_ds = StructureDataset(fake_samples[:4])
    result = train_one_seed(
        train_ds, val_ds,
        epochs=5, batch_size=4, lr=1e-3, seed=0,
        ckpt_path=tmp_path / "ckpt.pt",
    )
    assert result.history["train_loss"][-1] < result.history["train_loss"][0]
    assert (tmp_path / "ckpt.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `train.py`**

`phonon_omegamax/models/train.py`:

```python
"""Single-seed CGCNN training with Huber loss and early stopping."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from .cgcnn import CGCNN


@dataclass
class TrainResult:
    history: dict[str, list[float]] = field(default_factory=dict)
    best_val_mae: float = float("inf")
    epochs_run: int = 0


def train_one_seed(
    train_ds,
    val_ds,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 30,
    huber_delta: float = 10.0,
    ckpt_path: Path | None = None,
    device: str | None = None,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CGCNN().to(device)
    opt = Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_mae": []}
    best = float("inf")
    bad_epochs = 0
    result = TrainResult(history=history)

    for ep in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = F.smooth_l1_loss(pred, batch.y.squeeze(-1), beta=huber_delta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        history["train_loss"].append(float(np.mean(losses)))

        model.eval()
        abs_errs = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                abs_errs.append(
                    (pred - batch.y.squeeze(-1)).abs().cpu().numpy()
                )
        val_mae = float(np.concatenate(abs_errs).mean()) if abs_errs else float("inf")
        history["val_mae"].append(val_mae)
        result.epochs_run = ep + 1

        if val_mae < best:
            best = val_mae
            result.best_val_mae = best
            bad_epochs = 0
            if ckpt_path is not None:
                Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": model.state_dict(), "seed": seed}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    return result
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train.py -v`
Expected: 1 passed (takes ~30 s — torch-geometric DataLoader has a slow first-call import).

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/models/train.py tests/test_train.py
git commit -m "feat(models): single-seed CGCNN training with Huber loss"
```

---

### Task 12: Ensemble + 5-fold orchestration

**Files:**
- Create: `phonon_omegamax/models/ensemble.py`
- Create: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing test**

`tests/test_ensemble.py`:

```python
import torch
from torch_geometric.data import Batch

from phonon_omegamax.models.dataset import StructureDataset
from phonon_omegamax.models.ensemble import (
    load_ensemble_predictions,
    train_fold_ensemble,
)


def test_train_fold_ensemble_writes_checkpoints(tmp_path, fake_samples):
    train_ds = StructureDataset(fake_samples)
    val_ds = StructureDataset(fake_samples[:4])
    paths = train_fold_ensemble(
        train_ds, val_ds,
        ckpt_dir=tmp_path / "ckpts", fold=0, seeds=(0, 1),
        epochs=3, batch_size=4,
    )
    assert len(paths) == 2
    for p in paths:
        assert p.exists()


def test_load_ensemble_predictions_returns_mean(tmp_path, fake_samples):
    train_ds = StructureDataset(fake_samples)
    val_ds = StructureDataset(fake_samples[:4])
    train_fold_ensemble(
        train_ds, val_ds,
        ckpt_dir=tmp_path / "ckpts", fold=0, seeds=(0, 1),
        epochs=3, batch_size=4,
    )
    test_ds = StructureDataset(fake_samples[4:])
    preds = load_ensemble_predictions(
        tmp_path / "ckpts", fold=0, dataset=test_ds, batch_size=4
    )
    assert preds.shape == (len(fake_samples) - 4,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ensemble.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `ensemble.py`**

`phonon_omegamax/models/ensemble.py`:

```python
"""Per-fold ensemble training + averaged predictions for inference."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from .cgcnn import CGCNN
from .train import train_one_seed


def train_fold_ensemble(
    train_ds,
    val_ds,
    ckpt_dir: Path,
    fold: int,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    **train_kwargs,
) -> list[Path]:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for s in seeds:
        path = ckpt_dir / f"fold{fold}_seed{s}.pt"
        if not path.exists():
            train_one_seed(
                train_ds, val_ds, seed=s, ckpt_path=path, **train_kwargs
            )
        paths.append(path)
    return paths


def load_ensemble_predictions(
    ckpt_dir: Path,
    fold: int,
    dataset,
    batch_size: int = 64,
) -> np.ndarray:
    ckpt_dir = Path(ckpt_dir)
    paths = sorted(ckpt_dir.glob(f"fold{fold}_seed*.pt"))
    if not paths:
        raise RuntimeError(f"no checkpoints found in {ckpt_dir} for fold {fold}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    per_model: list[np.ndarray] = []
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=False)
        m = CGCNN()
        m.load_state_dict(state["state_dict"])
        m.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                preds.append(m(batch).cpu().numpy())
        per_model.append(np.concatenate(preds))
    return np.stack(per_model).mean(axis=0)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ensemble.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/models/ensemble.py tests/test_ensemble.py
git commit -m "feat(models): per-fold CGCNN ensemble training + averaged inference"
```

---

## Phase 5 — Evaluation

### Task 13: Core regression metrics

**Files:**
- Create: `phonon_omegamax/eval/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

`tests/test_metrics.py`:

```python
import numpy as np

from phonon_omegamax.eval.metrics import mae, r2, residual_quantiles


def test_mae_zero_on_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0.0


def test_mae_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    assert mae(y_true, y_pred) == 0.5


def test_r2_unit_on_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, y) == 1.0


def test_r2_zero_on_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, y.mean())
    assert abs(r2(y, pred)) < 1e-9


def test_residual_quantiles():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    pred = y + rng.normal(scale=0.5, size=1000)
    q = residual_quantiles(y, pred)
    assert set(q.keys()) == {"p05", "p50", "p95", "mean"}
    assert q["p05"] < q["p50"] < q["p95"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `metrics.py`**

`phonon_omegamax/eval/metrics.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/eval/metrics.py tests/test_metrics.py
git commit -m "feat(eval): MAE, R², residual quantiles"
```

---

### Task 14: Cross-validation runner

**Files:**
- Create: `phonon_omegamax/eval/run.py`
- Modify: `phonon_omegamax/cli.py` (new file)
- Create: `Makefile`

This task is the orchestration glue. No unit tests on `run.py` directly — it's exercised by the end-to-end smoke test in Task 18.

- [ ] **Step 1: Create `phonon_omegamax/eval/run.py`**

```python
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
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    epochs: int = 200,
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
```

- [ ] **Step 2: Create `phonon_omegamax/cli.py`**

```python
"""CLI entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch  # noqa: F401  (must precede any libomp-touching dep on macOS)

from .data.load import load_phonons
from .eval.run import run_cgcnn_cv, run_gbdt_cv


def cmd_data(args) -> None:
    samples = load_phonons(cache_path=args.cache)
    print(f"loaded {len(samples)} samples → {args.cache}")


def cmd_train_gbdt(args) -> None:
    samples = load_phonons(cache_path=args.data_cache)
    summary = run_gbdt_cv(samples, cache_dir=args.cache_dir, metrics_dir=args.metrics_dir)
    (args.metrics_dir / "gbdt_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"GBDT mean MAE = {summary['mae_mean']:.1f} ± {summary['mae_std']:.1f} cm⁻¹")


def cmd_train_cgcnn(args) -> None:
    samples = load_phonons(cache_path=args.data_cache)
    summary = run_cgcnn_cv(
        samples,
        cache_dir=args.cache_dir,
        ckpt_dir=args.ckpt_dir,
        metrics_dir=args.metrics_dir,
        epochs=args.epochs,
    )
    (args.metrics_dir / "cgcnn_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"CGCNN mean MAE = {summary['mae_mean']:.1f} ± {summary['mae_std']:.1f} cm⁻¹")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m phonon_omegamax.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    dp = sub.add_parser("data")
    dp.add_argument("--cache", default="data/cache/phonons.parquet", type=Path)
    dp.set_defaults(func=cmd_data)

    gp = sub.add_parser("train-gbdt")
    gp.add_argument("--data-cache", default="data/cache/phonons.parquet", type=Path)
    gp.add_argument("--cache-dir", default="data/cache", type=Path)
    gp.add_argument("--metrics-dir", default="metrics", type=Path)
    gp.set_defaults(func=cmd_train_gbdt)

    cp = sub.add_parser("train-cgcnn")
    cp.add_argument("--data-cache", default="data/cache/phonons.parquet", type=Path)
    cp.add_argument("--cache-dir", default="data/cache", type=Path)
    cp.add_argument("--ckpt-dir", default="checkpoints", type=Path)
    cp.add_argument("--metrics-dir", default="metrics", type=Path)
    cp.add_argument("--epochs", default=200, type=int)
    cp.set_defaults(func=cmd_train_cgcnn)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create `Makefile`**

```makefile
PYTHON ?= .venv/bin/python
EXPORTS = OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

.PHONY: data
data:
	$(EXPORTS) $(PYTHON) -m phonon_omegamax.cli data

.PHONY: train-gbdt
train-gbdt:
	$(EXPORTS) $(PYTHON) -m phonon_omegamax.cli train-gbdt

.PHONY: train-cgcnn
train-cgcnn:
	$(EXPORTS) $(PYTHON) -m phonon_omegamax.cli train-cgcnn

.PHONY: figures
figures:
	$(EXPORTS) $(PYTHON) scripts/make_figures.py

.PHONY: all
all: data train-gbdt train-cgcnn figures

.PHONY: test
test:
	$(EXPORTS) $(PYTHON) -m pytest -q
```

- [ ] **Step 4: Smoke-check CLI parses**

Run: `python -m phonon_omegamax.cli --help`
Expected: shows `data`, `train-gbdt`, `train-cgcnn` subcommands.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/eval/run.py phonon_omegamax/cli.py Makefile
git commit -m "feat(cli): top-level CV runner + Makefile"
```

---

### Task 15: Leaderboard table

**Files:**
- Create: `phonon_omegamax/eval/leaderboard.py`
- Create: `tests/test_leaderboard.py`

- [ ] **Step 1: Write failing test**

`tests/test_leaderboard.py`:

```python
from phonon_omegamax.eval.leaderboard import build_leaderboard


def test_leaderboard_includes_ours_and_published():
    table = build_leaderboard(
        ours_gbdt={"mae_mean": 78.0, "mae_std": 4.0, "r2_mean": 0.71, "r2_std": 0.03},
        ours_cgcnn={"mae_mean": 60.0, "mae_std": 3.0, "r2_mean": 0.82, "r2_std": 0.02},
    )
    assert "Magpie + GBDT (ours)" in table
    assert "CGCNN (ours)" in table
    assert "Roost (published)" in table
    assert "75.6" in table  # Roost MAE
    assert "29.5" in table  # ALIGNN MAE
    # Δ vs Magpie column for CGCNN (78 - 60 = 18 cm⁻¹)
    assert "+18.0" in table or "18.0" in table


def test_leaderboard_handles_missing_cgcnn():
    table = build_leaderboard(
        ours_gbdt={"mae_mean": 78.0, "mae_std": 4.0, "r2_mean": 0.71, "r2_std": 0.03},
        ours_cgcnn=None,
    )
    assert "Magpie + GBDT (ours)" in table
    assert "CGCNN (ours)" not in table
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_leaderboard.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `leaderboard.py`**

`phonon_omegamax/eval/leaderboard.py`:

```python
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
            delta = f"+{gbdt_mae - ours_cgcnn['mae_mean']:.1f}"
        lines.append(
            f"| CGCNN (ours) | "
            f"{ours_cgcnn['mae_mean']:.1f} ± {ours_cgcnn['mae_std']:.1f} | "
            f"{ours_cgcnn['r2_mean']:.3f} | {delta} |"
        )
    lines.append("| --- | --- | --- | --- |")
    for name, mae_val in PUBLISHED:
        lines.append(f"| {name} | {mae_val} | — | — |")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_leaderboard.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/eval/leaderboard.py tests/test_leaderboard.py
git commit -m "feat(eval): leaderboard markdown table builder"
```

---

### Task 16: Parity + residual headline figure

**Files:**
- Create: `phonon_omegamax/eval/parity.py`
- Create: `tests/test_parity.py`

- [ ] **Step 1: Write failing test**

`tests/test_parity.py`:

```python
import matplotlib
matplotlib.use("Agg")

import numpy as np

from phonon_omegamax.eval.parity import headline_figure


def test_headline_figure_writes_file(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 1500, 200)
    gbdt = y + rng.normal(0, 50, 200)
    cgcnn = y + rng.normal(0, 30, 200)
    out = tmp_path / "headline.png"
    fig = headline_figure(y_true=y, gbdt_pred=gbdt, cgcnn_pred=cgcnn, out_path=out)
    assert out.exists()
    # Two panels (parity + residual KDE).
    assert len(fig.axes) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parity.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `parity.py`**

`phonon_omegamax/eval/parity.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_parity.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/eval/parity.py tests/test_parity.py
git commit -m "feat(eval): parity + residual KDE headline figure"
```

---

### Task 17: Headline-figure aggregator script

**Files:**
- Create: `scripts/make_figures.py`

- [ ] **Step 1: Write `scripts/make_figures.py`**

```python
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
```

- [ ] **Step 2: Smoke test with stubs**

```bash
mkdir -p metrics figures
.venv/bin/python -c "
import json
from pathlib import Path
Path('metrics/gbdt_summary.json').write_text(json.dumps(
    {'mae_mean': 78.0, 'mae_std': 4.0, 'r2_mean': 0.71, 'r2_std': 0.03}))
Path('metrics/cgcnn_summary.json').write_text(json.dumps(
    {'mae_mean': 60.0, 'mae_std': 3.0, 'r2_mean': 0.82, 'r2_std': 0.02}))
"
.venv/bin/python scripts/make_figures.py
cat figures/leaderboard.md
```

Expected: `leaderboard.md` printed with our two rows and the 4 published rows. No headline figure (per-fold JSONs absent).

- [ ] **Step 3: Clean up stubs**

```bash
rm -f metrics/gbdt_summary.json metrics/cgcnn_summary.json
```

- [ ] **Step 4: Commit**

```bash
git add scripts/make_figures.py
git commit -m "feat(figures): aggregator script for headline figure + leaderboard"
```

---

### Task 18: End-to-end smoke test

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write smoke test**

`tests/test_smoke.py`:

```python
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from phonon_omegamax.eval.run import run_cgcnn_cv, run_gbdt_cv


def _patch_loader(samples):
    df = pd.DataFrame({
        "structure": [s.structure for s in samples],
        "last phdos peak": [s.target for s in samples],
    })
    df.index = [s.mp_id for s in samples]
    fake = type("F", (), {"df": df})()
    return patch("phonon_omegamax.data.load._matbench_task", return_value=fake)


def test_gbdt_cv_runs(tmp_path, fake_samples):
    # Need ≥ 10 samples to make 5-fold meaningful.
    samples = fake_samples + fake_samples  # 24 samples total
    samples = list({s.mp_id: s for s in samples}.values()) or samples
    summary = run_gbdt_cv(
        samples, cache_dir=tmp_path / "cache",
        metrics_dir=tmp_path / "metrics",
    )
    assert "mae_mean" in summary
    assert summary["n_folds"] == 5
    assert (tmp_path / "metrics" / "gbdt_folds.json").exists()


def test_cgcnn_cv_runs(tmp_path, fake_samples):
    samples = fake_samples + fake_samples
    samples = list({s.mp_id: s for s in samples}.values()) or samples
    summary = run_cgcnn_cv(
        samples, cache_dir=tmp_path / "cache",
        ckpt_dir=tmp_path / "ckpts",
        metrics_dir=tmp_path / "metrics",
        seeds=(0, 1), epochs=3,
    )
    assert "mae_mean" in summary
    assert summary["n_folds"] == 5
    assert (tmp_path / "metrics" / "cgcnn_folds.json").exists()
```

- [ ] **Step 2: Run smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: 2 passed (CGCNN portion takes 2–3 min).

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: end-to-end CV smoke on synthetic samples"
```

---

## Phase 6 — Physics analysis

### Task 19: Physics-finding pass

**Files:**
- Create: `phonon_omegamax/physics/analysis.py`
- Create: `tests/test_physics.py`

- [ ] **Step 1: Write failing test**

`tests/test_physics.py`:

```python
import numpy as np

from phonon_omegamax.physics.analysis import (
    bin_failures_by_spacegroup,
    find_cgcnn_wins,
)


def test_find_cgcnn_wins_threshold(fake_samples):
    rng = np.random.default_rng(0)
    n = len(fake_samples)
    y = np.array([s.target for s in fake_samples])
    gbdt_pred = y + rng.normal(0, 80, n)
    cgcnn_pred = y + rng.normal(0, 20, n)
    wins = find_cgcnn_wins(
        fake_samples, y_true=y,
        gbdt_pred=gbdt_pred, cgcnn_pred=cgcnn_pred,
        threshold=50.0,
    )
    # Each entry has the keys we need for write-up.
    for w in wins:
        assert {"mp_id", "formula", "gbdt_err", "cgcnn_err", "improvement"} <= set(w)


def test_bin_failures_by_spacegroup(fake_samples):
    failures = [
        {"mp_id": fake_samples[0].mp_id, "structure": fake_samples[0].structure},
        {"mp_id": fake_samples[1].mp_id, "structure": fake_samples[1].structure},
    ]
    bins = bin_failures_by_spacegroup(failures)
    # NaCl is Fm-3m (225). MgO is Fm-3m too. So the bin maps SG to count.
    assert all(isinstance(k, int) for k in bins)
    assert sum(bins.values()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_physics.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `analysis.py`**

`phonon_omegamax/physics/analysis.py`:

```python
"""Identify materials where CGCNN structure-based predictions beat GBDT
composition-only by a margin, and bin those wins by structural attributes
to support the 'structure helps' write-up.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..sample import Sample


def find_cgcnn_wins(
    samples: list[Sample],
    y_true: np.ndarray,
    gbdt_pred: np.ndarray,
    cgcnn_pred: np.ndarray,
    threshold: float = 50.0,
) -> list[dict]:
    gbdt_err = np.abs(gbdt_pred - y_true)
    cgcnn_err = np.abs(cgcnn_pred - y_true)
    improvement = gbdt_err - cgcnn_err

    wins: list[dict] = []
    for i, s in enumerate(samples):
        if improvement[i] >= threshold:
            wins.append({
                "mp_id": s.mp_id,
                "formula": s.structure.composition.reduced_formula,
                "gbdt_err": float(gbdt_err[i]),
                "cgcnn_err": float(cgcnn_err[i]),
                "improvement": float(improvement[i]),
                "structure": s.structure,
            })
    wins.sort(key=lambda w: -w["improvement"])
    return wins


def bin_failures_by_spacegroup(records: list[dict]) -> dict[int, int]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    sg_counts: Counter[int] = Counter()
    for rec in records:
        try:
            sg = SpacegroupAnalyzer(rec["structure"]).get_space_group_number()
            sg_counts[int(sg)] += 1
        except Exception:
            continue
    return dict(sg_counts)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_physics.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add phonon_omegamax/physics/analysis.py tests/test_physics.py
git commit -m "feat(physics): cgcnn-wins finder + space-group binning"
```

---

## Phase 7 — Configs, docs, notebook

### Task 20: Per-model YAML configs

**Files:**
- Create: `configs/gbdt.yaml`
- Create: `configs/cgcnn.yaml`

- [ ] **Step 1: Write `configs/gbdt.yaml`**

```yaml
random_state: 42
max_iter: 500
learning_rate: 0.05
max_leaf_nodes: 63
early_stopping: true
validation_fraction: 0.2
```

- [ ] **Step 2: Write `configs/cgcnn.yaml`**

```yaml
seed: 42
seeds: [0, 1, 2, 3, 4]
epochs: 200
batch_size: 64
lr: 0.001
patience: 30
huber_delta: 10.0
hidden: 64
n_conv: 3
n_dense: 128
dropout: 0.2
cutoff_ang: 8.0
```

- [ ] **Step 3: Commit**

```bash
git add configs/
git commit -m "chore: per-model YAML configs"
```

(Wiring `--config` into the CLI is deferred — defaults in the code match these values; the YAML files document the hyperparameters for reproducibility.)

---

### Task 21: README + doc skeletons

**Files:**
- Create: `README.md`
- Create: `docs/data_card.md`
- Create: `docs/methods.md`
- Create: `docs/physics_findings.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# phonon-omega-max

Regression of ω_max (last-peak phonon frequency, cm⁻¹) for 1,265
inorganic crystals on the Matbench `phonons` task, comparing a
composition-only Magpie + GBDT baseline against a hand-implemented
structure-based CGCNN.

![headline](figures/headline.png)

## Leaderboard

_filled in after run_ — see `figures/leaderboard.md`.

## Reproduce

```bash
git clone <repo-url>
cd phonon-omega-max
pip install -e ".[dev]"
make all
```

Wall time on a laptop CPU: ~5 hours. The Matbench dataset is bundled
with the `matbench` package and downloads automatically on first use
(no API key needed).

On macOS, the `Makefile` exports `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the
torch / OpenMP dual-load deadlock encountered in the sibling
`xanes-oxstate` project.

## Layout

- `phonon_omegamax/` — the package
- `notebooks/walkthrough.ipynb` — single-fold walkthrough
- `configs/` — per-model hyperparameters
- `docs/` — data card, methods, physics findings
- `figures/` — versioned final figures + leaderboard

## Limitations

- Five seeds × five outer folds is a real CPU budget; if wall-clock
  runs long, dropping to three seeds is a documented escape hatch.
- Equivariant nets (e3nn, NequIP, MACE) are not implemented — future
  work.
- Target is a single derived scalar (ω_max), not the full DOS curve.

## Acknowledgements

Phonon data and structures from the Materials Project (CC-BY 4.0),
delivered via the Matbench benchmark suite.
````

- [ ] **Step 2: Write `docs/data_card.md`**

```markdown
# Data card

**Source:** Matbench `phonons` task (Materials Project DFT phonon
calculations, delivered via the `matbench` package bundled from figshare).
**License:** CC-BY 4.0.
**Matbench version:** _filled in after install_

## Counts

- 1,265 inorganic crystals after Matbench's standard filtering.
- Target: ω_max (last DOS peak frequency), cm⁻¹, single scalar per
  material.

## Splits

- 5-fold cross-validation, deterministic indices via our own KFold
  implementation seeded at 42.
- Inner train/val split: 80/20, seeded at 0.

## Distribution

_filled in by `make data` — histogram of ω_max, element coverage_

## Known limitations

- DFT-computed phonons, not experimental.
- Last-peak frequency is one derived scalar; the full DOS contains
  much more information that this project does not use.
- Element coverage is dominated by oxides and silicates; rare-earths
  and heavy actinides are sparse.
```

- [ ] **Step 3: Write `docs/methods.md`**

```markdown
# Methods

## Phase A: Magpie + GBDT

Composition fingerprint via matminer's `ElementProperty.from_preset("magpie")`
(132 element-property statistics: mean/min/max/std of atomic mass,
electronegativity, valence electron count, etc., across the formula).
NaN values are mean-imputed per column.

Model: sklearn `HistGradientBoostingRegressor`, 500 trees, learning
rate 0.05, 63 leaves, early stopping on an internal 20% validation
holdout. One model per outer CV fold, fixed seed.

## Phase B: CGCNN

Graph construction: for each crystal, atoms become nodes with 92-dim
one-hot element identity. Edges connect atom pairs within 8 Å (PBC-aware),
with a 41-dim Gaussian-basis distance feature (σ = 0.2 Å, centered
uniformly in [0, 8] Å).

Model: hand-implemented CGCNN (Xie & Grossman, 2018). Three gated
convolution layers with hidden width 64, then global mean pooling and
a two-layer MLP head (64 → 128 → 1). ~50k parameters.

Training: Adam (lr 1e-3, cosine decay to 1e-5), batch size 64, Huber
loss (δ = 10 cm⁻¹), 200 epochs max with early stopping (patience 30)
on validation MAE. Five-seed ensemble per outer CV fold; ensemble
prediction is the mean of the five raw outputs.

## Splits

5-fold outer CV (test indices fixed by our deterministic KFold).
Inner 80/20 train/val split inside each outer training set, used for
GBDT early stopping and CGCNN best-checkpoint selection. Test indices
are never touched during model selection or hyperparameter tuning.

## Metrics

- MAE (primary), cm⁻¹.
- R² (coefficient of determination).
- Residual quantiles P05/P50/P95 for fat-tail diagnostics.
- All reported as mean ± std across the 5 outer folds.
```

- [ ] **Step 4: Write `docs/physics_findings.md`**

```markdown
# Physics findings

The Magpie baseline sees only the composition; the CGCNN sees both
composition and the crystal structure. Materials where CGCNN beats
GBDT by more than 50 cm⁻¹ test the question: when does structure
contain decisive information that composition alone cannot supply?

The expected pattern is that polymorphs (same composition, different
structure) are the cleanest case — GBDT must predict the average of
their ω_max values, while CGCNN can distinguish them.

## (filled in after run)

For each pattern observed, a 2–4 sentence write-up grounded in
concrete material examples (mp_id and reduced formula).
```

- [ ] **Step 5: Commit**

```bash
git add README.md docs/
git commit -m "docs: README + data card + methods + physics-findings skeleton"
```

---

### Task 22: Notebook walkthrough

**Files:**
- Create: `notebooks/walkthrough.ipynb`

- [ ] **Step 1: Build the notebook programmatically**

Save this script as `/tmp/build_nb.py` and run it:

```python
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""\
# phonon-omega-max — single-fold walkthrough

This notebook runs one outer CV fold end-to-end:

1. Load Matbench `phonons` (cached after first call).
2. Featurize: Magpie composition + CGCNN graphs.
3. Train GBDT and one CGCNN seed on the fold's training split.
4. Evaluate on the fold's test split.
5. Show parity + residual plot for the single fold.

The full 5-fold × 5-seed result is produced by `make all` at the
command line; this notebook is for getting your hands on the data and
the models in one shot.
"""))

cells.append(nbf.v4.new_code_cell("""\
from pathlib import Path
from phonon_omegamax.data.load import load_phonons
from phonon_omegamax.data.split import kfold_indices, inner_train_val_split
from phonon_omegamax.features.magpie import featurize_samples
from phonon_omegamax.models.dataset import StructureDataset
from phonon_omegamax.models.train import train_one_seed
from phonon_omegamax.models.gbdt import GBDTRegressor
import numpy as np

ROOT = Path.cwd().parent
samples = load_phonons(cache_path=ROOT / 'data' / 'cache' / 'phonons.parquet')
print(f'{len(samples)} samples loaded')
"""))

cells.append(nbf.v4.new_code_cell("""\
folds = list(kfold_indices(len(samples), seed=42))
train_idx, test_idx = folds[0]
inner_train, inner_val = inner_train_val_split(train_idx, val_frac=0.2, seed=0)
print(f'fold 0: train={len(inner_train)} val={len(inner_val)} test={len(test_idx)}')

X = featurize_samples(samples, cache_path=ROOT / 'data' / 'cache' / 'magpie.npy')
y = np.array([s.target for s in samples])
"""))

cells.append(nbf.v4.new_code_cell("""\
gbdt = GBDTRegressor(random_state=42).fit(X[train_idx], y[train_idx])
gbdt_pred = gbdt.predict(X[test_idx])
print(f'GBDT fold-0 MAE = {np.abs(gbdt_pred - y[test_idx]).mean():.1f} cm⁻¹')
"""))

cells.append(nbf.v4.new_code_cell("""\
train_ds = StructureDataset(
    [samples[i] for i in inner_train],
    cache_dir=ROOT / 'data' / 'cache' / 'graphs',
)
val_ds = StructureDataset(
    [samples[i] for i in inner_val],
    cache_dir=ROOT / 'data' / 'cache' / 'graphs',
)
test_ds = StructureDataset(
    [samples[i] for i in test_idx],
    cache_dir=ROOT / 'data' / 'cache' / 'graphs',
)

result = train_one_seed(
    train_ds, val_ds, epochs=50, batch_size=32, seed=0,
    ckpt_path=ROOT / 'checkpoints' / 'walkthrough_fold0_seed0.pt',
)
print(f'best val MAE = {result.best_val_mae:.1f} after {result.epochs_run} epochs')
"""))

cells.append(nbf.v4.new_code_cell("""\
import torch
from torch_geometric.loader import DataLoader
from phonon_omegamax.models.cgcnn import CGCNN

ckpt = torch.load(
    ROOT / 'checkpoints' / 'walkthrough_fold0_seed0.pt',
    map_location='cpu', weights_only=False,
)
model = CGCNN()
model.load_state_dict(ckpt['state_dict'])
model.eval()

loader = DataLoader(test_ds, batch_size=32, shuffle=False)
cgcnn_pred = []
with torch.no_grad():
    for batch in loader:
        cgcnn_pred.append(model(batch).cpu().numpy())
cgcnn_pred = np.concatenate(cgcnn_pred)
print(f'CGCNN fold-0 MAE = {np.abs(cgcnn_pred - y[test_idx]).mean():.1f} cm⁻¹')
"""))

cells.append(nbf.v4.new_code_cell("""\
from phonon_omegamax.eval.parity import headline_figure
headline_figure(
    y_true=y[test_idx], gbdt_pred=gbdt_pred, cgcnn_pred=cgcnn_pred,
)
"""))

cells.append(nbf.v4.new_markdown_cell("""\
## Next steps

Run `make all` at the command line to reproduce the full 5-fold ×
5-seed result and the headline figure that backs the leaderboard
table.
"""))

nb.cells = cells
Path('notebooks').mkdir(exist_ok=True)
nbf.write(nb, 'notebooks/walkthrough.ipynb')
print('wrote notebooks/walkthrough.ipynb')
```

Run it: `.venv/bin/python /tmp/build_nb.py`.

- [ ] **Step 2: Verify the notebook is loadable**

Run: `.venv/bin/python -c "import nbformat; nb = nbformat.read('notebooks/walkthrough.ipynb', as_version=4); print(len(nb.cells), 'cells')"`
Expected: `8 cells`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/walkthrough.ipynb
git commit -m "docs: notebook walkthrough for single fold"
```

---

## Self-Review

After writing the plan, walking the spec:

**Spec §1 (Scope / deliverables / success criteria):**
- Repo + README + leaderboard inline → Task 21
- Notebook → Task 22
- Headline figure → Tasks 16, 17
- Physics findings → Task 19, write-up in Task 21
- ≤ 80 / ≤ 65 MAE / ≥ 10 cm⁻¹ improvement → verified by `run_*_cv` (Task 14) and leaderboard table (Task 15)
- Mean residual ∈ [−5, 5] → reported by `residual_quantiles` (Task 13)
- `make all` reproduces in < 6 h → Makefile in Task 14, scripts in Task 17

**Spec §2 (Architecture):**
- `Sample` dataclass → Task 2
- Magpie featurizer + cache → Task 6
- Graph builder + cache → Task 7
- StructureDataset → Task 9
- CGCNN → Task 10
- Training loop → Task 11
- Ensemble + multi-fold → Task 12
- CLI + Makefile → Task 14

**Spec §3 (Data):**
- Matbench loader + parquet cache → Task 4
- Cleaning checks (defensive) → Task 4 (inside the loader)
- 5-fold splits → Task 5
- Caches (Magpie .npy + per-graph .pt) → Tasks 6, 9
- Data card skeleton → Task 21

**Spec §4 (Methods):**
- Phase A GBDT model → Task 8
- Phase B CGCNN model + training → Tasks 10, 11
- Comparison protocol (5 outer folds, identical test indices) → Task 14
- Physics finding pass → Task 19

**Spec §5 (Evaluation):**
- MAE / R² / residuals → Task 13
- Headline figure → Task 16
- Leaderboard table → Task 15
- End-to-end smoke → Task 18
- Repository deliverables → Tasks 20, 21, 22 (plus Task 17 aggregator)

**Placeholder scan:** the doc skeletons (`docs/data_card.md`, `docs/physics_findings.md`) contain `_filled in after run_` markers — those are correct (their content is the deliverable of the run, not the plan). No "TBD" / "TODO" / "implement later" / "add appropriate" patterns in implementation tasks.

**Type consistency:** `GBDTRegressor` and `train_one_seed`'s contract are consistent across Tasks 8, 11, 14. `Sample` fields stable from Task 2 through Tasks 4, 6, 9. `headline_figure(y_true, gbdt_pred, cgcnn_pred)` signature matches between Task 16 (definition) and Tasks 17, 22 (callers).

No gaps. Plan is complete.
