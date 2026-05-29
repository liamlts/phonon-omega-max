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
