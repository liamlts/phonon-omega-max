import pickle
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from phonon_omegamax.data.load import load_phonons


def _build_fake_df(samples):
    return pd.DataFrame(
        [{"mp_id": s.mp_id, "structure": s.structure, "target": s.target}
         for s in samples]
    )


def test_load_returns_list_of_samples(tmp_path, fake_samples):
    df = _build_fake_df(fake_samples)
    with patch("phonon_omegamax.data.load._fetch_matbench_df", return_value=df):
        out = load_phonons(cache_path=tmp_path / "phonons.parquet")
    assert len(out) == 12
    assert {s.mp_id for s in out} == {f"mp-fake-{i}" for i in range(12)}
    assert all(s.target > 0 for s in out)


def test_load_uses_cache_on_second_call(tmp_path, fake_samples):
    df = _build_fake_df(fake_samples)
    cache = tmp_path / "phonons.parquet"
    with patch("phonon_omegamax.data.load._fetch_matbench_df", return_value=df) as mk:
        first = load_phonons(cache_path=cache)
        second = load_phonons(cache_path=cache)
        # mk called exactly once (during first load).
        assert mk.call_count == 1
    assert [s.mp_id for s in first] == [s.mp_id for s in second]
