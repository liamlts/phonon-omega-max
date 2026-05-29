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
