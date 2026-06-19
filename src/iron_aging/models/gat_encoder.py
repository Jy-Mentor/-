"""GATv2 编码器模块.

参考:
  - How Attentive are Graph Attention Networks? (Brody et al., ICLR 2022)
  - GitHub: https://github.com/tech-srl/how_attentive_are_gats
  - PyG 实现: torch_geometric.nn.GATv2Conv
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GATEncoder(nn.Module):
    """GATv2 编码器: 同类型节点内的动态局部注意力."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_dim, hidden_dim, heads=heads, dropout=dropout, share_weights=False
        )
        self.conv2 = GATv2Conv(
            hidden_dim * heads,
            out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
            share_weights=False,
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
