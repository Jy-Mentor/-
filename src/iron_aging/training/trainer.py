"""HGT-GAT 训练器模块.

提供可复用的训练循环, 支持链路预测任务与早停.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData

from iron_aging.training.losses import bce_with_logits_loss

logger = logging.getLogger(__name__)


class HGTTrainer:
    """HGT-GAT 链路预测训练器.

    负责模型训练、验证、早停与指标记录. 支持任意源-目标节点类型的边.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "cpu",
        early_stopping_patience: int = 20,
        edge_type: tuple[str, str, str] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.early_stopping_patience = early_stopping_patience
        self.edge_type = edge_type
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_ap": [],
        }
        self.best_val_metric = -np.inf
        self.patience_counter = 0

    def _infer_edge_type(
        self, edge_label_index: torch.Tensor | None = None
    ) -> tuple[str, str, str]:
        """推断边类型.

        优先级:
            1. 初始化时传入的 edge_type.
            2. 从 edge_label_index 的元数据推断 (PyG 边索引属性).
            3. 默认回退到 gene->pathway (兼容旧行为).
        """
        if self.edge_type is not None:
            return self.edge_type
        if (
            edge_label_index is not None
            and hasattr(edge_label_index, "edge_type")
            and edge_label_index.edge_type is not None
        ):
            return edge_label_index.edge_type
        return ("gene", "enriched_in", "pathway")

    def train_epoch(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> float:
        """执行一个训练 epoch.

        Args:
            data: PyG HeteroData.
            edge_label_index: 用于监督的边索引 [2, E].
            edge_labels: 边标签 [E].

        Returns:
            平均训练损失.
        """
        self.model.train()
        self.optimizer.zero_grad()

        data = data.to(self.device)
        edge_label_index = edge_label_index.to(self.device)
        edge_labels = edge_labels.to(self.device)

        src_type, _, dst_type = self._infer_edge_type(edge_label_index)

        # 模型应返回节点嵌入字典
        node_emb = self.model(data.x_dict, data.edge_index_dict)
        src_emb = node_emb[src_type][edge_label_index[0]]
        dst_emb = node_emb[dst_type][edge_label_index[1]]

        # 链路预测头假设为 self.model.predictor 或独立传入
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            raise ValueError("模型需包含 predictor 属性或传入独立链路预测头")

        logits = predictor(src_emb, dst_emb).squeeze(-1)
        loss = bce_with_logits_loss(logits, edge_labels)
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def evaluate(
        self,
        data: HeteroData,
        edge_label_index: torch.Tensor,
        edge_labels: torch.Tensor,
    ) -> dict[str, float]:
        """评估模型.

        Args:
            data: PyG HeteroData.
            edge_label_index: 边索引 [2, E].
            edge_labels: 边标签 [E].

        Returns:
            包含 loss/auc/ap 的字典.
        """
        self.model.eval()
        data = data.to(self.device)
        edge_label_index = edge_label_index.to(self.device)
        edge_labels = edge_labels.to(self.device)

        src_type, _, dst_type = self._infer_edge_type(edge_label_index)

        node_emb = self.model(data.x_dict, data.edge_index_dict)
        src_emb = node_emb[src_type][edge_label_index[0]]
        dst_emb = node_emb[dst_type][edge_label_index[1]]

        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            raise ValueError("模型需包含 predictor 属性")

        logits = predictor(src_emb, dst_emb).squeeze(-1)
        loss = float(bce_with_logits_loss(logits, edge_labels).item())
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = edge_labels.cpu().numpy()

        metrics = {"loss": loss}
        if len(np.unique(labels_np)) > 1:
            metrics["auc"] = float(roc_auc_score(labels_np, probs))
            metrics["ap"] = float(average_precision_score(labels_np, probs))
        else:
            metrics["auc"] = 0.0
            metrics["ap"] = 0.0

        return metrics

    def step(self, val_metric: float) -> bool:
        """早停步进.

        Args:
            val_metric: 验证指标 (越高越好).

        Returns:
            如果应继续训练返回 True, 触发早停返回 False.
        """
        self.history["val_auc"].append(val_metric)
        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        if self.patience_counter >= self.early_stopping_patience:
            logger.info("早停触发: %d 轮未提升", self.early_stopping_patience)
            return False
        return True
