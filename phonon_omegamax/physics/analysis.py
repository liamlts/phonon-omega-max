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
