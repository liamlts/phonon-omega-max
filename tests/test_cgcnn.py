import torch
from torch_geometric.data import Batch

from phonon_omegamax.features.graph import structure_to_graph
from phonon_omegamax.models.cgcnn import CGCNN, CGCNNConv


def test_cgcnn_conv_preserves_node_count(nacl_structure):
    g = structure_to_graph(nacl_structure)
    h = torch.randn(g.x.shape[0], 64)
    conv = CGCNNConv(n_atom_feats=64, n_edge_feats=41)
    out = conv(h, g.edge_index, g.edge_attr)
    assert out.shape == h.shape


def test_cgcnn_forward_returns_scalar_per_graph(nacl_structure, mgo_structure):
    g1 = structure_to_graph(nacl_structure)
    g2 = structure_to_graph(mgo_structure)
    batch = Batch.from_data_list([g1, g2])
    model = CGCNN()
    out = model(batch)
    assert out.shape == (2,)


def test_cgcnn_param_count_under_200k():
    model = CGCNN()
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000, f"too many params: {n}"
