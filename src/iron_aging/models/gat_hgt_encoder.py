"""GAT-HGT 融合编码器.

将异构图 Transformer (HGT) 与异构图注意力网络 (HeteroGAT) 并行编码,
通过可学习的门控机制按节点类型融合两种表示, 兼顾异构语义与同构结构信息.

参考:
- Hu et al. (2020) "Heterogeneous Graph Transformer", WWW.
- Velickovic et al. (2018) "Graph Attention Networks", ICLR.
- Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR.
- PyG HeteroConv: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.HeteroConv
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

from iron_aging.models.hgt_encoder import HGTEncoder


class GATHGTEncoder(nn.Module):
    """GAT-HGT 融合编码器.

    输入特征先由 HeteroLinkPredictionModel 投影到统一 hidden_dim,
    随后分别经过:
      - HGT 分支: 在异构边上学习类型感知表示.
      - HeteroGAT 分支: 按边类型分别学习局部注意力 (避免同质投影语义污染).
    最终按节点类型使用可学习门控融合两个分支.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        metadata: tuple,
        num_nodes_dict: dict[str, int],
        num_heads: int = 4,
        dropout: float = 0.2,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.num_nodes_dict = num_nodes_dict

        self.hgt_encoder = HGTEncoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            metadata=metadata,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
        )

        # HeteroGAT: 每种边类型拥有独立的 GATv2Conv 参数,
        # 避免将 gene/pathway/compound/disease/pocket 硬塞进同一注意力空间.
        self.gat_convs = nn.ModuleList()
        gat_hidden = hidden_dim // num_heads
        for _ in range(num_layers):
            conv_dict: dict[tuple[str, str, str], GATv2Conv] = {}
            for edge_type in self.edge_types:
                conv_dict[edge_type] = GATv2Conv(
                    hidden_dim,
                    gat_hidden,
                    heads=num_heads,
                    dropout=dropout,
                    add_self_loops=False,
                    share_weights=False,
                )
            self.gat_convs.append(HeteroConv(conv_dict, aggr="mean"))
        self.gat_proj = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, out_dim) for nt in self.node_types
        })

        self.fusion_gate = nn.ModuleDict({
            nt: nn.Linear(out_dim * 2, 1) for nt in self.node_types
        })
        self.dropout = dropout

    def _gat_forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """HeteroGAT 前向: 按边类型分别学习注意力."""
        gat_z = x_dict
        for conv in self.gat_convs:
            gat_z = conv(gat_z, edge_index_dict)
            gat_z = {k: F.elu(v) for k, v in gat_z.items()}
            gat_z = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in gat_z.items()
            }
        gat_z = {k: self.gat_proj[k](v) for k, v in gat_z.items()}
        return gat_z

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """前向传播, 返回融合后的节点嵌入字典."""
        hgt_z = self.hgt_encoder(x_dict, edge_index_dict)
        gat_z = self._gat_forward(x_dict, edge_index_dict)

        fused: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            h = hgt_z[nt]
            g = gat_z.get(nt, h)
            concat = torch.cat([h, g], dim=-1)
            gate = torch.sigmoid(self.fusion_gate[nt](concat))
            fused[nt] = gate * h + (1.0 - gate) * g
            fused[nt] = F.dropout(fused[nt], p=self.dropout, training=self.training)
        return fused

    @classmethod
    def from_hetero_data(
        cls,
        data: HeteroData,
        hidden_dim: int = 64,
        out_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> "GATHGTEncoder":
        """从 HeteroData 自动推断元数据与节点数."""
        metadata = data.metadata()
        num_nodes_dict = {nt: int(data[nt].num_nodes) for nt in data.node_types}
        return cls(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            metadata=metadata,
            num_nodes_dict=num_nodes_dict,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
        )
