"""Load the Matbench `phonons` dataset by direct figshare fetch.

The official matbench package is incompatible with modern Python (pins
scipy==1.7.3 etc., no Python 3.12 wheels). We fetch the same JSON file
directly and parse it with monty.json.MontyDecoder, which handles
pymatgen.Structure serialization round-tripping.

Cached as a parquet of (mp_id, structure_pickle, target) so subsequent
loads skip the figshare round-trip.
"""
from __future__ import annotations

import gzip
import json
import pickle
import urllib.request
from pathlib import Path

import pandas as pd

from ..sample import Sample


MATBENCH_PHONONS_URL = (
    "https://ml.materialsproject.org/projects/matbench_phonons.json.gz"
)


def _fetch_matbench_df() -> pd.DataFrame:
    """Download the matbench_phonons JSON and return a DataFrame
    with columns (mp_id, structure, target).

    Schema of the JSON (matbench convention):
        {
          "index": ["mb-phonons-0001", "mb-phonons-0002", ...],
          "columns": ["structure", "last phdos peak"],
          "data":    [[<Structure JSON>, <float>], ...]
        }
    """
    from monty.json import MontyDecoder

    req = urllib.request.Request(
        MATBENCH_PHONONS_URL,
        headers={"User-Agent": "phonon-omega-max/0.1 (research; +https://github.com/)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    payload = json.loads(gzip.decompress(raw).decode())
    decoder = MontyDecoder()

    mp_ids = list(payload["index"])
    cols = list(payload["columns"])
    data_rows = payload["data"]
    if "structure" not in cols or "last phdos peak" not in cols:
        raise RuntimeError(
            f"unexpected matbench schema; expected 'structure' and 'last phdos peak' "
            f"columns, got {cols}"
        )
    struct_col_idx = cols.index("structure")
    target_col_idx = cols.index("last phdos peak")

    structures = [decoder.process_decoded(row[struct_col_idx]) for row in data_rows]
    targets = [float(row[target_col_idx]) for row in data_rows]

    return pd.DataFrame({"mp_id": mp_ids, "structure": structures, "target": targets})


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

    df = _fetch_matbench_df()
    samples: list[Sample] = []
    for _, row in df.iterrows():
        try:
            samples.append(
                Sample(mp_id=str(row["mp_id"]), structure=row["structure"],
                       target=float(row["target"]))
            )
        except ValueError:
            continue  # skip non-positive / non-finite targets if any survive

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df = pd.DataFrame({
        "mp_id": [s.mp_id for s in samples],
        "structure_pickle": [pickle.dumps(s.structure) for s in samples],
        "target": [s.target for s in samples],
    })
    cache_df.to_parquet(cache_path)
    return samples
