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
