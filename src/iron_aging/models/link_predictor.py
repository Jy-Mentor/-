"""异质图链路预测头.

支持基因-通路归属预测与化合物-靶点结合预测, 采用 MLP 解码器.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinkPredictor(nn.Module):
    """链路预测头: 拼接两个节点嵌入, 输出结合/归属概率."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        """前向传播.

        Args:
            src_emb: 源节点嵌入 [N, in_dim].
            dst_emb: 目标节点嵌入 [N, in_dim].

        Returns:
            预测分数 [N, 1].
        """
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.mlp(combined)
