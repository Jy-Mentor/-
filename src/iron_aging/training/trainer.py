"""HGT-GAT-RGCN 训练器模块.

提供可复用的训练循环, 支持链路预测任务与早停.
参考实践:
- SpotTarget (Zhu et al., WSDM 2024): 训练/测试时从消息传递图中排除目标边.
- PLNLP (Wang et al., 2021): pairwise ranking loss 更贴合 AUC 评估目标.
- PyG: structured negative sampling 生成更难负样本.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData

from iron_aging.training.losses import bce_with_logits_loss, margin_ranking_loss
from iron_aging.training.negative_sampling import structured_negative_sampling

logger = logging.getLogger(__name__)


class HGTTrainer:
    """异质图链路预测训练器.

    负责模型训练、验证、早停与指标记录. 支持任意源-目标节点类型的边.
    训练时动态采样负样本, 支持 BCE 或 pairwise ranking loss.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str = "cpu",
        early_stopping_patience: int = 20,
        edge_type: tuple[str, str, str] | None = None,
        loss_type: str = "bce",
        neg_sampling_ratio: float = 1.0,
        margin: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.early_stopping_patience = early_stopping_patience
        self.edge_type = edge_type
        self.loss_type = loss_type.lower()
        self.neg_sampling_ratio = neg_sampling_ratio
        self.margin = margin
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_ap": [],
        }
        self.best_val_metric = -np.inf
        self.patience_counter = 0
        self.best_state_dict: dict[str, Any] | None = None

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

    @staticmethod
    def _edges_to_tensor(
        edges: list[tuple[int, int]], device: torch.device
    ) -> torch.Tensor:
        """将边列表转为 [2, E] 张量."""
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=device)
        return torch.tensor(
            [[s for s, _ in edges], [d for _, d in edges]],
            dtype=torch.long,
            device=device,
        )

    def _sample_negatives(
        self,
        pos_edges: list[tuple[int, int]],
        num_src: int,
        num_dst: int,
        all_pos_edges: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        """为正样本动态采样负样本."""
        if self.neg_sampling_ratio <= 0 or not pos_edges:
            return []
        num_neg = max(1, int(len(pos_edges) * self.neg_sampling_ratio))
        return structured_negative_sampling(
            num_src=num_src,
            num_dst=num_dst,
            num_negatives=num_neg,
            positive_edges=all_pos_edges,
            rng=rng,
        )

    def _compute_scores(
        self,
        node_emb: dict[str, torch.Tensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """计算给定边的预测分数."""
        src_type, _, dst_type = self._infer_edge_type()
        src_emb = node_emb[src_type][edge_index[0]]
        dst_emb = node_emb[dst_type][edge_index[1]]
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            raise ValueError("模型需包含 predictor 属性或传入独立链路预测头")
        return predictor(src_emb, dst_emb).squeeze(-1)

    def train_epoch(
        self,
        data: HeteroData,
        pos_edges: list[tuple[int, int]],
        num_src: int,
        num_dst: int,
        all_pos_edges: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> float:
        """执行一个训练 epoch.

        Args:
            data: 训练图 (已移除目标监督边).
            pos_edges: 正样本边列表.
            num_src: 源节点数.
            num_dst: 目标节点数.
            all_pos_edges: 全部正样本边集合 (用于负采样过滤).
            rng: 随机数生成器.

        Returns:
            平均训练损失.
        """
        self.model.train()
        self.optimizer.zero_grad()

        data = data.to(self.device)

        neg_edges = self._sample_negatives(
            pos_edges, num_src, num_dst, all_pos_edges, rng
        )
        pos_edge_index = self._edges_to_tensor(pos_edges, self.device)
        neg_edge_index = self._edges_to_tensor(neg_edges, self.device)

        node_emb = self.model(data.x_dict, data.edge_index_dict)
        pos_score = self._compute_scores(node_emb, pos_edge_index)
        neg_score = self._compute_scores(node_emb, neg_edge_index)

        if self.loss_type == "ranking":
            loss = margin_ranking_loss(pos_score, neg_score, margin=self.margin)
        else:
            logits = torch.cat([pos_score, neg_score])
            labels = torch.cat(
                [
                    torch.ones_like(pos_score),
                    torch.zeros_like(neg_score),
                ]
            )
            loss = bce_with_logits_loss(logits, labels)

        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def evaluate(
        self,
        data: HeteroData,
        pos_edges: list[tuple[int, int]],
        neg_edges: list[tuple[int, int]],
    ) -> dict[str, float]:
        """评估模型.

        Args:
            data: 评估图 (已移除目标监督边).
            pos_edges: 正样本边列表.
            neg_edges: 负样本边列表.

        Returns:
            包含 loss/auc/ap 的字典.
        """
        self.model.eval()
        data = data.to(self.device)

        pos_edge_index = self._edges_to_tensor(pos_edges, self.device)
        neg_edge_index = self._edges_to_tensor(neg_edges, self.device)

        node_emb = self.model(data.x_dict, data.edge_index_dict)
        pos_score = self._compute_scores(node_emb, pos_edge_index)
        neg_score = self._compute_scores(node_emb, neg_edge_index)

        if self.loss_type == "ranking":
            loss = float(
                margin_ranking_loss(pos_score, neg_score, margin=self.margin).item()
            )
        else:
            logits = torch.cat([pos_score, neg_score])
            labels = torch.cat(
                [
                    torch.ones_like(pos_score),
                    torch.zeros_like(neg_score),
                ]
            )
            loss = float(bce_with_logits_loss(logits, labels).item())

        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        labels_np = torch.cat(
            [torch.ones_like(pos_score), torch.zeros_like(neg_score)]
        ).cpu().numpy()

        metrics: dict[str, float] = {"loss": loss}
        if len(np.unique(labels_np)) > 1:
            metrics["auc"] = float(roc_auc_score(labels_np, scores))
            metrics["ap"] = float(average_precision_score(labels_np, scores))
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
        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.patience_counter = 0
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
            return True
        self.patience_counter += 1
        if self.patience_counter >= self.early_stopping_patience:
            logger.info("早停触发: %d 轮未提升", self.early_stopping_patience)
            return False
        return True

    def fit(
        self,
        data: HeteroData,
        train_pos_edges: list[tuple[int, int]],
        val_pos_edges: list[tuple[int, int]],
        test_pos_edges: list[tuple[int, int]],
        train_neg_edges: list[tuple[int, int]],
        val_neg_edges: list[tuple[int, int]],
        test_neg_edges: list[tuple[int, int]],
        num_src: int,
        num_dst: int,
        epochs: int = 200,
        seed: int = 42,
    ) -> dict[str, list[float]]:
        """端到端单任务训练循环.

        Args:
            data: 训练图 (HeteroData, 已移除所有监督边).
            train_pos_edges: 训练正样本边.
            val_pos_edges: 验证正样本边.
            test_pos_edges: 测试正样本边.
            train_neg_edges: 训练负样本边 (ranking loss 时不使用, 动态采样替代).
            val_neg_edges: 验证负样本边.
            test_neg_edges: 测试负样本边.
            num_src: 源节点数量.
            num_dst: 目标节点数量.
            epochs: 最大训练轮数.
            seed: 随机种子.

        Returns:
            训练历史记录 (train_loss/val_loss/val_auc/val_ap).
        """
        all_pos_edges = set(train_pos_edges) | set(val_pos_edges) | set(test_pos_edges)
        rng = np.random.default_rng(seed)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(
                data, train_pos_edges, num_src, num_dst, all_pos_edges, rng
            )
            self.history["train_loss"].append(train_loss)

            val_metrics = self.evaluate(data, val_pos_edges, val_neg_edges)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auc"].append(val_metrics["auc"])
            self.history["val_ap"].append(val_metrics["ap"])

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "[epoch %d] train_loss=%.4f val_auc=%.4f val_ap=%.4f",
                    epoch,
                    train_loss,
                    val_metrics["auc"],
                    val_metrics["ap"],
                )

            if not self.step(val_metrics["auc"]):
                break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            logger.info("已加载最佳模型 (val_auc=%.4f)", self.best_val_metric)
        logger.info("训练结束, 最佳 val_auc=%.4f", self.best_val_metric)
        return self.history
