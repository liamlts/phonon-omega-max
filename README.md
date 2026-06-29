# phonon-omega-max

[![CI](https://github.com/liamlts/phonon-omega-max/actions/workflows/ci.yml/badge.svg)](https://github.com/liamlts/phonon-omega-max/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Regression of ω_max (last-peak phonon frequency, cm⁻¹) for 1,265
inorganic crystals on the Matbench `phonons` task, comparing a
composition-only Magpie + GBDT baseline against a hand-implemented
structure-based CGCNN.

![headline](figures/headline.png)

## Headline result

A 30-line scikit-learn baseline (composition-only Magpie features and gradient
boosting) reaches MAE = 70.1 ± 10.3 cm⁻¹, R² = 0.91 on 5-fold CV, beating the
published Roost baseline (75.6 cm⁻¹) on the same Matbench `phonons` task.

| Model | MAE (cm⁻¹) | R² | n_folds |
|---|---|---|---|
| Magpie + GBDT (this repo) | 70.1 ± 10.3 | 0.910 | 5 |
| CGCNN (this repo, partial)    | 94.7 ± 12.4     | 0.799     | 3 |
| Roost (published)             | 75.6            | —         | 5 |
| CGCNN (published)             | 57.8            | —         | 5 |
| MEGNet (published)            | 49.9            | —         | 5 |
| ALIGNN (published)            | 29.5            | —         | 5 |

Full table in `figures/leaderboard.md`. Per-fold JSON in `metrics/`.

> The CGCNN result is partial (3 of 5 CV folds, 1–3 seeds per fold) and
> below the published config: 100 epochs with `patience=30` was the
> laptop-CPU budget, well below the ~500-epoch training the original
> CGCNN paper used. Treat the GBDT line, not the CGCNN line, as this
> repo's benchmark.

## Reproduce

```bash
git clone <repo-url>
cd phonon-omega-max
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install "numpy>=1.26,<2.0"   # torch 2.2.2 was built against NumPy 1.x
make all
```

Wall time on a laptop CPU: ~5 hours. The Matbench dataset is fetched
directly from Materials Project's ML data cache on first run (no API
key needed). All later runs reuse the parquet cache under
`data/cache/`.

On macOS, the Makefile exports `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `KMP_DUPLICATE_LIB_OK=TRUE` to keep torch and
the rest of the OpenMP-using stack from stepping on each other.

## Layout

- `phonon_omegamax/`: the package
- `notebooks/walkthrough.ipynb`: single-fold demo
- `configs/`: per-model hyperparameters (also documented in `docs/methods.md`)
- `docs/`: data card, methods, physics findings
- `figures/`: versioned headline figure + leaderboard table
- `metrics/`: per-fold JSON + summary JSON, written by `make all`

## Limitations

- Five seeds × five outer folds is a heavy CPU budget; if wall-clock
  runs long, dropping to three seeds is a documented escape hatch.
- Equivariant nets (e3nn, NequIP, MACE) are not implemented (future
  work).
- The target is a single derived scalar (ω_max), not the full DOS
  curve. The structure-vs-composition ablation is set up so that
  extending to DOS regression is a single-file change.

## Acknowledgements

Phonon data and structures from the Materials Project (CC-BY 4.0),
delivered via the same JSON the Matbench benchmark distributes.

## License

MIT. See [`LICENSE`](LICENSE). Materials Project data is CC-BY 4.0.
