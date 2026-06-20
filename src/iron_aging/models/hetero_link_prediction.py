"""异质图链路预测模型.

将 HGT/GAT 编码器与链路预测头组合为端到端模型,
支持不同节点类型的输入特征投影.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from iron_aging.models.gat_encoder import GATEncoder
from iron_aging.models.hgt_encoder import HGTEncoder
from iron_aging.models.link_predictor import LinkPredictor


class HeteroLinkPredictionModel(nn.Module):
    """异质图链路预测模型.

    1. 为每种节点类型投影输入特征到统一 hidden_dim.
    2. 使用 HGT 或 GAT 编码器学习节点嵌入.
    3. 使用 LinkPredictor 预测任意源-目标节点对间的边概率.
    """

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_dims: dict[str, int],
        hidden_dim: int,
        out_dim: int,
        encoder_type: str = "hgt",
        num_nodes_dict: dict[str, int] | None = None,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types = metadata[0]
        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "gat":
            if num_nodes_dict is None:
                msg = "GAT 编码器需要提供 num_nodes_dict"
                raise ValueError(msg)
            self.num_nodes_dict = num_nodes_dict
            self.node_type_offset = self._compute_offsets(num_nodes_dict)
            self.max_feat_dim = max(in_dims.values())
            proj_dim = self.max_feat_dim
        else:
            proj_dim = hidden_dim

        # 每种节点类型一个输入投影层
        self.input_proj = nn.ModuleDict(
            {
                nt: nn.Linear(in_dims.get(nt, proj_dim), proj_dim)
                for nt in self.node_types
            }
        )

        if self.encoder_type == "gat":
            self.encoder = GATEncoder(
                in_dim=self.max_feat_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                heads=num_heads,
                dropout=dropout,
            )
        else:
            self.encoder = HGTEncoder(
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                metadata=metadata,
                num_heads=num_heads,
                dropout=dropout,
                num_layers=num_layers,
            )
        self.predictor = LinkPredictor(in_dim=out_dim, hidden_dim=hidden_dim, dropout=dropout)

    @staticmethod
    def _compute_offsets(num_nodes_dict: dict[str, int]) -> dict[str, int]:
        """计算同质图中的节点类型偏移."""
        offset = 0
        offsets: dict[str, int] = {}
        for nt, num in num_nodes_dict.items():
            offsets[nt] = offset
            offset += num
        return offsets

    def _build_homogeneous_features(
        self, x_dict: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """将异构特征补零到统一维度后拼接为同质特征."""
        parts = []
        for nt in self.node_types:
            x = x_dict[nt]
            if x.shape[1] < self.max_feat_dim:
                pad = torch.zeros(
                    x.shape[0],
                    self.max_feat_dim - x.shape[1],
                    dtype=x.dtype,
                    device=x.device,
                )
                x = torch.cat([x, pad], dim=1)
            parts.append(x)
        return torch.cat(parts, dim=0)

    def _to_homogeneous_edge_index(
        self, edge_index_dict: dict[tuple[str, str, str], torch.Tensor]
    ) -> torch.Tensor:
        """将所有异构边合并为同质边索引."""
        edge_list: list[torch.Tensor] = []
        for (src_type, _, dst_type), edge_index in edge_index_dict.items():
            shifted = torch.stack(
                [
                    edge_index[0] + self.node_type_offset[src_type],
                    edge_index[1] + self.node_type_offset[dst_type],
                ]
            )
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
        """前向传播, 返回节点嵌入字典."""
        projected: dict[str, torch.Tensor] = {}
        for nt, x in x_dict.items():
            if nt in self.input_proj:
                projected[nt] = self.input_proj[nt](x)
            else:
                projected[nt] = x

        if self.encoder_type == "gat":
            x = self._build_homogeneous_features(projected)
            edge_index = self._to_homogeneous_edge_index(edge_index_dict)
            z = self.encoder(x, edge_index)
            return self._split_homogeneous_embeddings(z)

        return self.encoder(projected, edge_index_dict)

    @classmethod
    def from_hetero_data(
        cls,
        data: HeteroData,
        hidden_dim: int = 64,
        out_dim: int = 16,
        encoder_type: str = "hgt",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> "HeteroLinkPredictionModel":
        """从 HeteroData 自动推断元数据与输入维度."""
        metadata = data.metadata()
        in_dims = {nt: int(data[nt].x.shape[1]) for nt in data.node_types}
        num_nodes_dict = {nt: int(data[nt].num_nodes) for nt in data.node_types}
        return cls(
            metadata=metadata,
            in_dims=in_dims,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            encoder_type=encoder_type,
            num_nodes_dict=num_nodes_dict,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
