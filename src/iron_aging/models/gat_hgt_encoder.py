"""GAT-HGT 融合编码器.

将异构图 Transformer (HGT) 与同构图注意力网络 (GATv2) 并行编码,
通过可学习的门控机制按节点类型融合两种表示, 兼顾异构语义与同构结构信息.

参考:
- Hu et al. (2020) "Heterogeneous Graph Transformer", WWW.
- Velickovic et al. (2018) "Graph Attention Networks", ICLR.
- Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from iron_aging.models.gat_encoder import GATEncoder
from iron_aging.models.hgt_encoder import HGTEncoder


class GATHGTEncoder(nn.Module):
    """GAT-HGT 融合编码器.

    输入特征先由 HeteroLinkPredictionModel 投影到统一 hidden_dim,
    随后分别经过:
      - HGT 分支: 在异构边上学习类型感知表示.
      - GAT 分支: 在全局同质投影上学习局部注意力表示.
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
        self.node_type_offset = self._compute_offsets(num_nodes_dict)

        self.hgt_encoder = HGTEncoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            metadata=metadata,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
        )
        # GAT 在统一 hidden_dim 特征上运行; GATEncoder 输出 out_dim
        self.gat_encoder = GATEncoder(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            heads=num_heads,
            dropout=dropout,
        )

        self.fusion_gate = nn.ModuleDict({
            nt: nn.Linear(out_dim * 2, 1) for nt in self.node_types
        })
        self.dropout = dropout

    @staticmethod
    def _compute_offsets(num_nodes_dict: dict[str, int]) -> dict[str, int]:
        """计算同质投影中的节点类型偏移."""
        offset = 0
        offsets: dict[str, int] = {}
        for nt, num in num_nodes_dict.items():
            offsets[nt] = offset
            offset += num
        return offsets

    def _to_homogeneous_edge_index(
        self, edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        """将异构边索引合并为同质边索引."""
        edge_list: list[torch.Tensor] = []
        for (src_type, _, dst_type), edge_index in edge_index_dict.items():
            shifted = torch.stack([
                edge_index[0] + self.node_type_offset[src_type],
                edge_index[1] + self.node_type_offset[dst_type],
            ])
            edge_list.append(shifted)
        return torch.cat(edge_list, dim=1)

    def _split_homogeneous_embeddings(
        self, z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """将同质节点嵌入按类型拆分回字典."""
        z_dict: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            offset = self.node_type_offset[nt]
            num = self.num_nodes_dict[nt]
            z_dict[nt] = z[offset : offset + num]
        return z_dict

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """前向传播, 返回融合后的节点嵌入字典."""
        hgt_z = self.hgt_encoder(x_dict, edge_index_dict)

        # GAT 同质投影: 按 node_type_offset 拼接所有节点
        x = torch.cat([x_dict[nt] for nt in self.node_types], dim=0)
        homogeneous_edge_index = self._to_homogeneous_edge_index(edge_index_dict)
        gat_z_hom = self.gat_encoder(x, homogeneous_edge_index)
        gat_z = self._split_homogeneous_embeddings(gat_z_hom)

        fused: dict[str, torch.Tensor] = {}
        for nt in self.node_types:
            h = hgt_z[nt]
            g = gat_z[nt]
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
