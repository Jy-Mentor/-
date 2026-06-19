"""HGT 编码器模块.

参考:
  - HGT: Heterogeneous Graph Transformer (Hu et al., WWW 2020)
  - GitHub: https://github.com/acbull/pyHGT
  - PyG 实现: torch_geometric.nn.HGTConv
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGTEncoder(nn.Module):
    """HGT 编码器: 跨类型语义注意力."""

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        metadata: tuple,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_layers: int = 2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
            )
        self.proj = Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_out = conv(x_dict, edge_index_dict)
            for k, v in x_out.items():
                if v is not None:
                    x_dict[k] = F.elu(v)
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }
        x_dict = {k: self.proj(v) for k, v in x_dict.items()}
        return x_dict
