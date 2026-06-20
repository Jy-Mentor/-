"""HGT vs GAT 消融实验（改进版）.

参考：
- Hu et al. (2020) "Heterogeneous Graph Transformer", WWW.
- Gharizadeh et al. (2024) "HGTDR: Advancing drug repurposing with heterogeneous graph
  transformers", Bioinformatics.
- Velickovic et al. (2018) "Graph Attention Networks", ICLR.
- Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR.
- Li et al. (2023) "HeaRT: Heterogeneous Graph Level Link Prediction", NeurIPS.

本脚本在同样的训练/验证/测试边上对比 HGT（异质）与 GAT（同质投影）
在化合物-靶点（ct）和基因-基因（gp）边预测任务上的 AUC 与 AP.
相比初版改进：
1. 可靠负采样：负样本从真实节点空间中采样，并排除所有已知正样本（含验证/测试），
   避免数据泄漏。
2. MLP 链路预测头替代简单内积。
3. 早停与验证 AUC 选最优。
4. 同时输出 AUC 与 AP。
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HGTConv

# 抑制 OpenMP 多副本警告
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import module3_hgt  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN = 64
NUM_LAYERS = 2
HEADS = 4
DROPOUT = 0.3
EPOCHS = 200
LR = 0.001
WEIGHT_DECAY = 1e-5
PATIENCE = 20
NEGATIVE_RATIO = 1.0

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data() -> HeteroData:
    """使用 module3_hgt 构建带特征的 HeteroData."""
    logger.info("构建异质图...")
    graph_data = module3_hgt.build_heterogeneous_graph()
    pyg_data = module3_hgt.build_pyg_data(graph_data)
    logger.info("节点类型: %s", list(pyg_data.node_types))
    logger.info("边类型: %s", [str(et) for et in pyg_data.edge_types])
    return pyg_data


def get_target_edge_index(
    data: HeteroData, task: str
) -> tuple[tuple[str, str, str], int]:
    """返回目标任务对应的 edge_type 与边数."""
    if task == "ct":
        key = ("compound", "targets", "gene")
    elif task == "gp":
        key = ("gene", "coexp", "gene")
    else:
        raise ValueError(f"未知任务: {task}")
    if key not in data.edge_types:
        available = [str(et) for et in data.edge_types]
        raise KeyError(f"边类型 {key} 不存在，可用: {available}")
    return key, int(data[key].edge_index.shape[1])


def sample_reliable_negatives(
    num_src: int,
    num_dst: int,
    num_negatives: int,
    forbidden: set[tuple[int, int]],
    exclude_self_loops: bool = False,
    rng: np.random.Generator | None = None,
    max_trials: int = 1_000_000,
) -> list[tuple[int, int]]:
    """从节点空间中采样可靠负样本，排除所有已知正样本.

    Args:
        num_src: 源节点数量.
        num_dst: 目标节点数量.
        num_negatives: 需要采样的负样本数量.
        forbidden: 禁止采样的边集合（含训练/验证/测试所有已知正样本）.
        exclude_self_loops: 是否排除自环（gp 任务适用）.
        rng: 随机数生成器.
        max_trials: 最大尝试次数.

    Returns:
        负样本边列表.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    negatives: set[tuple[int, int]] = set()
    trials = 0
    while len(negatives) < num_negatives and trials < max_trials:
        src = int(rng.integers(0, num_src))
        dst = int(rng.integers(0, num_dst))
        if exclude_self_loops and src == dst:
            trials += 1
            continue
        edge = (src, dst)
        if edge not in forbidden:
            negatives.add(edge)
        trials += 1

    if len(negatives) < num_negatives:
        raise RuntimeError(
            f"无法采够 {num_negatives} 条负样本 (实际 {len(negatives)}); "
            "可能正样本过于稠密或搜索空间不足"
        )
    return list(negatives)


def prepare_link_prediction_data(
    data: HeteroData, task: str, rng: np.random.Generator | None = None
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, tuple[str, str, str], torch.Tensor,
]:
    """手动划分训练/验证/测试集，正负样本比例一致.

    Returns:
        train_idx, train_labels, val_idx, val_labels, test_idx, test_labels, key, train_pos_edges
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    key, _ = get_target_edge_index(data, task)
    src_type, _, dst_type = key
    num_src = data[src_type].num_nodes
    num_dst = data[dst_type].num_nodes

    # 读取所有正样本并去重（保留原始 edge_index 中的索引以便重建训练图）
    edge_index = data[key].edge_index.cpu().numpy()
    seen: set[tuple[int, int]] = set()
    unique_indices: list[int] = []
    unique_edges: list[tuple[int, int]] = []
    for i in range(edge_index.shape[1]):
        e = (int(edge_index[0, i]), int(edge_index[1, i]))
        if e not in seen:
            seen.add(e)
            unique_indices.append(i)
            unique_edges.append(e)
    pos_indices = np.array(unique_indices)
    positives = np.array(unique_edges)
    logger.info("任务 %s: %d 条去重正样本边", task.upper(), len(positives))

    # 生成可靠负样本：排除所有已知正样本
    forbidden = set(map(tuple, positives.tolist()))
    num_negatives = int(len(positives) * NEGATIVE_RATIO)
    negatives = sample_reliable_negatives(
        num_src=num_src,
        num_dst=num_dst,
        num_negatives=num_negatives,
        forbidden=forbidden,
        exclude_self_loops=(task == "gp"),
        rng=rng,
    )
    logger.info("任务 %s: 生成 %d 条可靠负样本", task.upper(), len(negatives))

    # 打乱并划分
    pos_order = np.arange(len(positives))
    neg_order = np.arange(len(negatives))
    rng.shuffle(pos_order)
    rng.shuffle(neg_order)

    n_pos = len(positives)
    n_neg = len(negatives)
    n_pos_train = int(n_pos * TRAIN_RATIO)
    n_pos_val = int(n_pos * VAL_RATIO)
    n_neg_train = int(n_neg * TRAIN_RATIO)
    n_neg_val = int(n_neg * VAL_RATIO)

    pos_train = positives[pos_order[:n_pos_train]]
    pos_val = positives[pos_order[n_pos_train : n_pos_train + n_pos_val]]
    pos_test = positives[pos_order[n_pos_train + n_pos_val :]]
    train_pos_indices = pos_indices[pos_order[:n_pos_train]]
    neg_array = np.array(negatives)
    neg_train = neg_array[neg_order[:n_neg_train]]
    neg_val = neg_array[neg_order[n_neg_train : n_neg_train + n_neg_val]]
    neg_test = neg_array[neg_order[n_neg_train + n_neg_val :]]

    def _to_tensor(edges_pos: np.ndarray, edges_neg: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.from_numpy(np.vstack([edges_pos, edges_neg]).T).long()
        labels = torch.cat([
            torch.ones(len(edges_pos)),
            torch.zeros(len(edges_neg)),
        ])
        return indices, labels

    train_idx, train_labels = _to_tensor(pos_train, neg_train)
    val_idx, val_labels = _to_tensor(pos_val, neg_val)
    test_idx, test_labels = _to_tensor(pos_test, neg_test)

    logger.info(
        "任务 %s 划分: train %d/%d, val %d/%d, test %d/%d",
        task.upper(),
        len(pos_train),
        len(neg_train),
        len(pos_val),
        len(neg_val),
        len(pos_test),
        len(neg_test),
    )
    train_pos_edges = torch.from_numpy(train_pos_indices).long()
    return train_idx, train_labels, val_idx, val_labels, test_idx, test_labels, key, train_pos_edges


def build_train_graph(
    data: HeteroData, key: tuple[str, str, str], train_pos_indices: torch.Tensor
) -> HeteroData:
    """构建训练阶段消息传递图：目标边类型仅保留训练正样本边.

    避免验证/测试边在训练时参与消息传递导致数据泄漏.
    """
    train_data = data.clone()
    src_type, rel_type, dst_type = key
    full_edge_index = data[key].edge_index
    train_data[src_type, rel_type, dst_type].edge_index = full_edge_index[:, train_pos_indices]
    # 同步更新反向边（若存在）
    rev_key = (dst_type, f"rev_{rel_type}", src_type)
    if rev_key in data.edge_types:
        rev_edge_index = data[rev_key].edge_index
        train_data[rev_key].edge_index = rev_edge_index[:, train_pos_indices]
    return train_data


class FeatureProjector(nn.Module):
    """将各类型节点特征投影到统一隐藏维度."""

    def __init__(self, data: HeteroData, hidden_dim: int) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict({
            nt: nn.Linear(data[nt].x.shape[1], hidden_dim)
            for nt in data.node_types
        })

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        return {nt: self.projectors[nt](data[nt].x) for nt in data.node_types}


class MLPLinkPredictor(nn.Module):
    """MLP 链路预测头."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([src_emb, dst_emb], dim=-1)
        return self.mlp(combined).squeeze(-1)


class HGTLP(nn.Module):
    """HGT 编码器 + MLP 预测头."""

    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = HIDDEN,
        num_layers: int = NUM_LAYERS,
        heads: int = HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.feature_projector = FeatureProjector(data, hidden_dim)
        self.convs = nn.ModuleList()
        metadata = data.metadata()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_dim, hidden_dim, metadata, heads=heads))
        self.predictor = MLPLinkPredictor(hidden_dim, hidden_dim, dropout)
        self.dropout = dropout

    def forward(
        self, data: HeteroData, edge_index_dict: dict | None = None
    ) -> dict[str, torch.Tensor]:
        x_dict = self.feature_projector(data)
        if edge_index_dict is None:
            edge_index_dict = data.edge_index_dict
        for conv in self.convs:
            x_out = conv(x_dict, edge_index_dict)
            for k, v in x_out.items():
                if v is not None:
                    x_dict[k] = F.elu(v)
            x_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training)
                for k, v in x_dict.items()
            }
        return x_dict


class GATLP(nn.Module):
    """同质 GATv2 编码器 + MLP 预测头."""

    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = HIDDEN,
        num_layers: int = NUM_LAYERS,
        heads: int = HEADS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        # 同质特征：每种节点类型特征补零到统一维度，再沿节点维度拼接
        self.max_feat_dim = max(data[nt].x.shape[1] for nt in data.node_types)
        in_channels = self.max_feat_dim
        # 同质图节点数
        self.num_nodes = sum(data[nt].num_nodes for nt in data.node_types)
        self.node_type_offset = self._compute_offsets(data)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim * heads
            self.convs.append(GATv2Conv(in_ch, hidden_dim, heads=heads, dropout=dropout))
        self.predictor = MLPLinkPredictor(hidden_dim * heads, hidden_dim, dropout)
        self.dropout = dropout

    @staticmethod
    def _compute_offsets(data: HeteroData) -> dict[str, int]:
        offset = 0
        offsets: dict[str, int] = {}
        for nt in data.node_types:
            offsets[nt] = offset
            offset += data[nt].num_nodes
        return offsets

    def _build_homogeneous_features(self, data: HeteroData) -> torch.Tensor:
        """将异构特征拼接为同质特征向量（缺失维度补零）."""
        max_dim = max(data[nt].x.shape[1] for nt in data.node_types)
        parts = []
        for nt in data.node_types:
            x = data[nt].x
            if x.shape[1] < max_dim:
                pad = torch.zeros(x.shape[0], max_dim - x.shape[1], dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=1)
            parts.append(x)
        return torch.cat(parts, dim=0)

    def forward(self, data: HeteroData, edge_index: torch.Tensor) -> torch.Tensor:
        x = self._build_homogeneous_features(data).to(next(self.parameters()).device)
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def to_homogeneous_edge_index(data: HeteroData, offsets: dict[str, int]) -> torch.Tensor:
    """将所有异构边合并为同质边索引."""
    edge_list: list[torch.Tensor] = []
    for (src_type, _, dst_type), edge_index in data.edge_index_dict.items():
        src_offset = offsets[src_type]
        dst_offset = offsets[dst_type]
        shifted = torch.stack([
            edge_index[0] + src_offset,
            edge_index[1] + dst_offset,
        ])
        edge_list.append(shifted)
    return torch.cat(edge_list, dim=1)


def evaluate(
    model: nn.Module,
    data: HeteroData,
    edge_label_index: torch.Tensor,
    edge_labels: torch.Tensor,
    src_type: str,
    dst_type: str,
    offsets: dict[str, int] | None = None,
    homogeneous_edge_index: torch.Tensor | None = None,
) -> dict[str, float]:
    """评估模型，返回 loss/auc/ap."""
    model.eval()
    with torch.no_grad():
        if isinstance(model, HGTLP):
            z_dict = model(data)
            src_emb = z_dict[src_type][edge_label_index[0]]
            dst_emb = z_dict[dst_type][edge_label_index[1]]
        else:
            z = model(data, homogeneous_edge_index)
            src_idx = edge_label_index[0] + offsets[src_type]
            dst_idx = edge_label_index[1] + offsets[dst_type]
            src_emb = z[src_idx]
            dst_emb = z[dst_idx]

        logits = model.predictor(src_emb, dst_emb)
        loss = F.binary_cross_entropy_with_logits(logits, edge_labels)
        probs = torch.sigmoid(logits).cpu().numpy()
        labels_np = edge_labels.cpu().numpy()

        metrics: dict[str, float] = {"loss": float(loss.item())}
        if len(np.unique(labels_np)) > 1:
            metrics["auc"] = float(roc_auc_score(labels_np, probs))
            metrics["ap"] = float(average_precision_score(labels_np, probs))
        else:
            metrics["auc"] = 0.0
            metrics["ap"] = 0.0
    return metrics


def train_model(
    model: nn.Module,
    data: HeteroData,
    train_data: HeteroData,
    train_idx: torch.Tensor,
    train_labels: torch.Tensor,
    val_idx: torch.Tensor,
    val_labels: torch.Tensor,
    src_type: str,
    dst_type: str,
    offsets: dict[str, int] | None = None,
    homogeneous_edge_index: torch.Tensor | None = None,
) -> dict[str, float]:
    """训练单个模型，返回验证集最优指标.

    训练阶段使用 train_data 进行消息传递（已剔除验证/测试边），
    评估阶段使用完整 data.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=PATIENCE // 2
    )

    data = data.to(DEVICE)
    train_data = train_data.to(DEVICE)
    train_idx = train_idx.to(DEVICE)
    train_labels = train_labels.to(DEVICE)
    val_idx = val_idx.to(DEVICE)
    val_labels = val_labels.to(DEVICE)

    # GAT 训练用同质边索引
    train_homogeneous_edge_index = None
    if not isinstance(model, HGTLP):
        train_homogeneous_edge_index = to_homogeneous_edge_index(train_data, offsets).to(DEVICE)

    best_val_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        if isinstance(model, HGTLP):
            z_dict = model(train_data)
            src_emb = z_dict[src_type][train_idx[0]]
            dst_emb = z_dict[dst_type][train_idx[1]]
        else:
            z = model(train_data, train_homogeneous_edge_index)
            src_idx = train_idx[0] + offsets[src_type]
            dst_idx = train_idx[1] + offsets[dst_type]
            src_emb = z[src_idx]
            dst_emb = z[dst_idx]

        logits = model.predictor(src_emb, dst_emb)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            val_metrics = evaluate(
                model, data, val_idx, val_labels, src_type, dst_type,
                offsets=offsets, homogeneous_edge_index=homogeneous_edge_index,
            )
            val_auc = val_metrics["auc"]
            scheduler.step(val_auc)
            logger.info(
                "[%s epoch %d] loss %.4f val_auc %.4f val_ap %.4f",
                model.__class__.__name__,
                epoch,
                loss.item(),
                val_auc,
                val_metrics["ap"],
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    logger.info("早停触发于 epoch %d", epoch)
                    break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    val_metrics = evaluate(
        model, data, val_idx, val_labels, src_type, dst_type,
        offsets=offsets, homogeneous_edge_index=homogeneous_edge_index,
    )
    return val_metrics


def run_ablation() -> dict[str, Any]:
    set_seed()
    data = build_data()
    results: dict[str, Any] = {"device": str(DEVICE), "seed": SEED, "tasks": {}}

    for task in ["gp", "ct"]:
        logger.info("=" * 60)
        logger.info("任务: %s", task.upper())
        key, num_edges = get_target_edge_index(data, task)
        src_type, _, dst_type = key
        logger.info("目标边 %s: %d 条", key, num_edges)

        (
            train_idx,
            train_labels,
            val_idx,
            val_labels,
            test_idx,
            test_labels,
            _,
            train_pos_indices,
        ) = prepare_link_prediction_data(data, task)

        # 构建训练消息传递图（剔除验证/测试目标边，避免数据泄漏）
        train_data = build_train_graph(data, key, train_pos_indices)

        # HGT 模型
        logger.info("训练 HGT...")
        hgt_model = HGTLP(data).to(DEVICE)
        hgt_val_metrics = train_model(
            hgt_model, data, train_data, train_idx, train_labels, val_idx, val_labels, src_type, dst_type
        )
        hgt_test_metrics = evaluate(
            hgt_model, data.to(DEVICE), test_idx.to(DEVICE), test_labels.to(DEVICE), src_type, dst_type
        )

        # GAT 模型
        logger.info("训练 GAT...")
        offsets = GATLP._compute_offsets(data)
        homogeneous_edge_index = to_homogeneous_edge_index(data, offsets).to(DEVICE)
        gat_model = GATLP(data).to(DEVICE)
        gat_val_metrics = train_model(
            gat_model,
            data,
            train_data,
            train_idx,
            train_labels,
            val_idx,
            val_labels,
            src_type,
            dst_type,
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
        )
        gat_test_metrics = evaluate(
            gat_model,
            data.to(DEVICE),
            test_idx.to(DEVICE),
            test_labels.to(DEVICE),
            src_type,
            dst_type,
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
        )

        results["tasks"][task] = {
            "edge_type": key,
            "num_pos_edges": num_edges,
            "train_samples": int(train_labels.shape[0]),
            "val_samples": int(val_labels.shape[0]),
            "test_samples": int(test_labels.shape[0]),
            "HGT": {
                "val_auc": hgt_val_metrics["auc"],
                "val_ap": hgt_val_metrics["ap"],
                "test_auc": hgt_test_metrics["auc"],
                "test_ap": hgt_test_metrics["ap"],
            },
            "GAT": {
                "val_auc": gat_val_metrics["auc"],
                "val_ap": gat_val_metrics["ap"],
                "test_auc": gat_test_metrics["auc"],
                "test_ap": gat_test_metrics["ap"],
            },
        }
        logger.info(
            "[%s] HGT val=%.4f test=%.4f (AP %.4f) | GAT val=%.4f test=%.4f (AP %.4f)",
            task.upper(),
            hgt_val_metrics["auc"],
            hgt_test_metrics["auc"],
            hgt_test_metrics["ap"],
            gat_val_metrics["auc"],
            gat_test_metrics["auc"],
            gat_test_metrics["ap"],
        )

    return results


def main() -> None:
    try:
        results = run_ablation()
        out_path = PROJECT_ROOT / "L3_results" / "ablation_hgt_vs_gat.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("结果已保存: %s", out_path)
        logger.info(json.dumps(results, indent=2, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
