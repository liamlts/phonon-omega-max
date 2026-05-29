# Data card

**Source:** Matbench `phonons` task (Materials Project DFT phonon
calculations). Fetched directly from
`https://ml.materialsproject.org/projects/matbench_phonons.json.gz`
on first run (no API key required); cached locally as a parquet.

**License:** CC-BY 4.0.

## Counts

- 1,265 inorganic crystals after Matbench's standard filtering.
- Target: ω_max (last DOS peak frequency), cm⁻¹, single scalar per
  material.

## Splits

- 5-fold cross-validation via the project's own deterministic KFold
  (`phonon_omegamax/data/split.py`), seeded at 42.
- Inner train/val split for early stopping: 80/20, seeded at 0.

## Distribution

_filled in after `make data` — histogram of ω_max, element coverage,
space-group breakdown._

## Known limitations

- DFT-computed phonons, not experimental.
- Last-peak frequency is one derived scalar; the full DOS contains
  much more information that this project does not use.
- The matbench Python package itself is not a dependency: it pins
  scipy==1.7.3 and other 2021-era versions with no Python 3.12 wheels.
  We fetch the same JSON file directly and parse via
  `monty.json.MontyDecoder`.
