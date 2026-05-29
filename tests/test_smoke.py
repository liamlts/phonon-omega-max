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
