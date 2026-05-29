"""Convert pymatgen.Structure → PyG Data for CGCNN.

Atom features: 92-dim one-hot of element identity (atomic number 1–92).
Edges: undirected pairs within 8 Å cutoff.
Edge features: 41-dim Gaussian basis on bond length, σ = 0.2 Å.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from pymatgen.core import Structure

N_ELEMENTS = 92
N_EDGE_FEATS = 41
CUTOFF_ANG = 8.0
GAUSSIAN_SIGMA = 0.2
EDGE_CENTERS = np.linspace(0.0, CUTOFF_ANG, N_EDGE_FEATS)


def _one_hot_atomic_number(z: int) -> np.ndarray:
    v = np.zeros(N_ELEMENTS, dtype=np.float32)
    if 1 <= z <= N_ELEMENTS:
        v[z - 1] = 1.0
    return v


def _gaussian_basis(distances: np.ndarray) -> np.ndarray:
    # distances: [E]; output [E, N_EDGE_FEATS]
    d = distances[:, None]
    return np.exp(-((d - EDGE_CENTERS[None, :]) ** 2) / (2 * GAUSSIAN_SIGMA**2))


def structure_to_graph(struct: Structure) -> Data:
    z = np.array([site.specie.Z for site in struct], dtype=np.int64)
    x = np.stack([_one_hot_atomic_number(int(zi)) for zi in z])

    # PBC-aware neighbor list within CUTOFF_ANG.
    # get_neighbor_list returns (center_idx, neighbor_idx, image, distance).
    centers, neighbors, _, distances = struct.get_neighbor_list(r=CUTOFF_ANG)
    if len(centers) == 0:
        # Isolated structure (e.g. single atom): self-loop with zero edge feat.
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros(1, N_EDGE_FEATS, dtype=torch.float32)
    else:
        edge_index = torch.tensor(
            np.stack([centers, neighbors]), dtype=torch.long
        )
        edge_attr = torch.tensor(
            _gaussian_basis(distances.astype(np.float64)),
            dtype=torch.float32,
        )

    return Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
