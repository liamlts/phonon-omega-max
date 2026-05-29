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
