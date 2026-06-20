"""异质图链路预测模型.

将 HGT 编码器与链路预测头组合为端到端模型,
支持不同节点类型的输入特征投影.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from iron_aging.models.hgt_encoder import HGTEncoder
from iron_aging.models.link_predictor import LinkPredictor


class HeteroLinkPredictionModel(nn.Module):
    """异质图链路预测模型.

    1. 为每种节点类型投影输入特征到统一 hidden_dim.
    2. 使用 HGT 编码器学习节点嵌入.
    3. 使用 LinkPredictor 预测任意源-目标节点对间的边概率.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_dims: dict[str, int],
        hidden_dim: int,
        out_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]

        # 每种节点类型一个输入投影层
        self.input_proj = nn.ModuleDict(
            {
                nt: nn.Linear(in_dims.get(nt, hidden_dim), hidden_dim)
                for nt in self.node_types
            }
        )

        self.encoder = HGTEncoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            metadata=metadata,
            num_heads=num_heads,
            dropout=dropout,
            num_layers=num_layers,
        )
        self.predictor = LinkPredictor(in_dim=out_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """前向传播, 返回节点嵌入字典."""
        projected: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.input_proj:
                projected[nt] = self.input_proj[nt](x)
            else:
                projected[nt] = x
        return self.encoder(projected, edge_index_dict)

    @classmethod
    def from_hetero_data(
        cls,
        data: HeteroData,
        hidden_dim: int = 64,
        out_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> "HeteroLinkPredictionModel":
        """从 HeteroData 自动推断元数据与输入维度."""
        metadata = data.metadata()
        in_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
        return cls(
            metadata=metadata,
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
