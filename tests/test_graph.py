import numpy as np
import torch

from phonon_omegamax.features.graph import structure_to_graph


def test_graph_has_required_fields(nacl_structure):
    g = structure_to_graph(nacl_structure)
    # x: [N_atoms, 92], edge_index: [2, E], edge_attr: [E, 41]
    assert g.x.shape[0] == 2
    assert g.x.shape[1] == 92
    assert g.edge_index.shape[0] == 2
    assert g.edge_index.shape[1] >= 2
    assert g.edge_attr.shape[0] == g.edge_index.shape[1]
    assert g.edge_attr.shape[1] == 41


def test_graph_atom_features_are_one_hot(nacl_structure):
    g = structure_to_graph(nacl_structure)
    # Each row of x is a one-hot (sum to 1).
    assert torch.allclose(g.x.sum(dim=-1), torch.ones(2))


def test_graph_edge_attrs_in_gaussian_basis_range(nacl_structure):
    g = structure_to_graph(nacl_structure)
    # Gaussian basis values are non-negative and bounded by 1.
    assert (g.edge_attr >= 0).all()
    assert (g.edge_attr <= 1.0 + 1e-6).all()
