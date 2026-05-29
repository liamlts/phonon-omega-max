"""CGCNN (Xie & Grossman 2018) — hand-implemented for tunability.

Gated message passing: m_ij = sigmoid(W_f · z_ij) * softplus(W_s · z_ij),
where z_ij = concat([h_i, h_j, e_ij]).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


class CGCNNConv(MessagePassing):
    def __init__(self, n_atom_feats: int, n_edge_feats: int):
        super().__init__(aggr="add")
        in_dim = 2 * n_atom_feats + n_edge_feats
        self.lin_filter = nn.Linear(in_dim, n_atom_feats)
        self.lin_core = nn.Linear(in_dim, n_atom_feats)
        self.bn = nn.BatchNorm1d(n_atom_feats)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.softplus(self.bn(x + out))

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate = torch.sigmoid(self.lin_filter(z))
        core = F.softplus(self.lin_core(z))
        return gate * core


class CGCNN(nn.Module):
    def __init__(
        self,
        n_atom_feats: int = 92,
        n_edge_feats: int = 41,
        hidden: int = 64,
        n_conv: int = 3,
        n_dense: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.atom_embed = nn.Linear(n_atom_feats, hidden)
        self.convs = nn.ModuleList(
            [CGCNNConv(hidden, n_edge_feats) for _ in range(n_conv)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, n_dense),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_dense, 1),
        )

    def forward(self, data) -> torch.Tensor:
        h = self.atom_embed(data.x)
        for conv in self.convs:
            h = conv(h, data.edge_index, data.edge_attr)
        pooled = global_mean_pool(h, data.batch)
        return self.head(pooled).squeeze(-1)
