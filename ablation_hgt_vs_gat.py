"""HGT vs GAT 消融实验（整改优化版）.

参考：
- Hu et al. (2020) "Heterogeneous Graph Transformer", WWW.
- Gharizadeh et al. (2024) "HGTDR: Advancing drug repurposing with heterogeneous graph
  transformers", Bioinformatics.
- Velickovic et al. (2018) "Graph Attention Networks", ICLR.
- Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR.
- Li et al. (2023) "HeaRT: Heterogeneous Graph Level Link Prediction", NeurIPS.

整改要点：
1. K-Fold 交叉验证：降低小样本随机划分带来的方差，尤其针对 ct 任务。
2. 分层抽样：按源节点（化合物/基因）分层划分，避免训练/验证/测试分布失衡。
3. 更严格训练图：可选移除与验证/测试节点相连的间接边，削弱信息泄漏。
4. 基线模型：增加 MLP-only（仅节点特征）与 Random 基线。
5. 生物学相关性指标：针对 BCP-ACSL4 等关键 pair 计算排名、Hits@K、MRR。
6. 可复现性：持久化划分索引、损失曲线、每折指标。
7. 图构建缓存：避免重复运行时的重复构建开销。
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
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, GCNConv, HGTConv

# 抑制 OpenMP 多副本警告
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import module3_hgt  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 同时将完整日志写入文件，便于后续审计与归档
_LOG_FILE = PROJECT_ROOT / "L3_results" / "ablation_run_v4.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(_file_handler)

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

# 交叉验证配置
N_FOLDS = 5
USE_CV = True  # 设为 False 可仅运行单次划分
STRICT_TRAIN_GRAPH = True  # 设为 True 可移除与 val/test 节点相连的间接边

# Node2Vec 配置
N2V_PRETRAIN_EPOCHS = 50
N2V_WALK_LENGTH = 10
N2V_CONTEXT_SIZE = 5
N2V_WALKS_PER_NODE = 10

# 关键生物学实体
TARGET_COMPOUND = "BCP"
TARGET_GENES = ["ACSL4", "GPX4", "PTGS2", "TFRC", "HMOX1", "TP53", "CDKN1A", "HIF1A", "NFE2L2"]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_data(cache_path: Path | None = None) -> tuple[HeteroData, dict[str, list[str]]]:
    """使用 module3_hgt 构建带特征的 HeteroData，支持缓存.

    Returns:
        pyg_data: PyG HeteroData 对象.
        node_names: 各节点类型的名称列表，用于后续生物学解释.
    """
    if cache_path is None:
        cache_path = PROJECT_ROOT / "L3_results" / "pyg_data_cache.pt"

    if cache_path.exists():
        logger.info("加载缓存的异质图: %s", cache_path)
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(cached, dict) and "data" in cached and "names" in cached:
                return cached["data"], cached["names"]
            if isinstance(cached, HeteroData):
                # 兼容旧缓存格式（无名称）
                return cached, {}
        except Exception:
            traceback.print_exc()
            logger.warning("缓存加载失败，重新构建图")

    logger.info("构建异质图...")
    graph_data = module3_hgt.build_heterogeneous_graph()
    pyg_data = module3_hgt.build_pyg_data(graph_data)
    node_names = {
        "gene": list(graph_data["gene"].get("names", [])),
        "compound": list(graph_data["compound"].get("names", [])),
        "pathway": list(graph_data["pathway"].get("names", [])),
        "disease": list(graph_data["disease"].get("names", [])),
        "pocket": list(graph_data["pocket"].get("names", [])),
    }
    logger.info("节点类型: %s", list(pyg_data.node_types))
    logger.info("边类型: %s", [str(et) for et in pyg_data.edge_types])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": pyg_data, "names": node_names}, cache_path)
    logger.info("图已缓存: %s", cache_path)
    return pyg_data, node_names


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


def _stratify_labels(pos_edges: np.ndarray, num_src: int, task: str) -> np.ndarray:
    """为分层抽样生成分层标签.

    ct 任务按源化合物（src）分层；gp 任务按源基因度分层桶.
    """
    if task == "ct":
        return pos_edges[:, 0]

    # gp 任务：按源节点度分 5 桶
    degrees = np.bincount(pos_edges[:, 0], minlength=num_src)
    src_degrees = degrees[pos_edges[:, 0]]
    bins = np.percentile(src_degrees, [20, 40, 60, 80])
    stratify = np.digitize(src_degrees, bins)
    return stratify


def prepare_link_prediction_data(
    data: HeteroData,
    task: str,
    rng: np.random.Generator | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[str, str, str],
    torch.Tensor,
    dict[str, Any],
]:
    """手动划分训练/验证/测试集，正负样本比例一致，按源节点分层.

    Returns:
        train_idx, train_labels, val_idx, val_labels, test_idx, test_labels, key,
        train_pos_edges, split_info
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

    # 按源节点分层抽样
    stratify_pos = _stratify_labels(positives, num_src, task)
    stratify_neg = _stratify_labels(np.array(negatives), num_src, task)

    n_pos = len(positives)
    n_neg = len(negatives)
    n_pos_train = int(n_pos * TRAIN_RATIO)
    n_pos_val = int(n_pos * VAL_RATIO)
    n_neg_train = int(n_neg * TRAIN_RATIO)
    n_neg_val = int(n_neg * VAL_RATIO)

    # 在分层约束下打乱：按标签组分别 shuffle
    pos_order = _stratified_shuffle(np.arange(n_pos), stratify_pos, rng)
    neg_order = _stratified_shuffle(np.arange(n_neg), stratify_neg, rng)

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

    split_info = {
        "pos_indices": pos_indices.tolist(),
        "positives": positives.tolist(),
        "negatives": negatives,
        "train_pos_idx_in_unique": pos_order[:n_pos_train].tolist(),
        "val_pos_idx_in_unique": pos_order[n_pos_train : n_pos_train + n_pos_val].tolist(),
        "test_pos_idx_in_unique": pos_order[n_pos_train + n_pos_val :].tolist(),
        "train_neg_idx_in_unique": neg_order[:n_neg_train].tolist(),
        "val_neg_idx_in_unique": neg_order[n_neg_train : n_neg_train + n_neg_val].tolist(),
        "test_neg_idx_in_unique": neg_order[n_neg_train + n_neg_val :].tolist(),
    }

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
    return train_idx, train_labels, val_idx, val_labels, test_idx, test_labels, key, train_pos_edges, split_info


def _stratified_shuffle(indices: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """在每组标签内部独立打乱，返回全局索引顺序."""
    order: list[int] = []
    for label in np.unique(labels):
        group = indices[labels == label].copy()
        rng.shuffle(group)
        order.extend(group.tolist())
    return np.array(order)


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


def build_strict_train_graph(
    data: HeteroData,
    key: tuple[str, str, str],
    train_pos_indices: torch.Tensor,
    val_idx: torch.Tensor,
    test_idx: torch.Tensor,
) -> HeteroData:
    """构建更严格的训练图：移除目标边，并移除与验证/测试节点相连的非目标间接边."""
    train_data = build_train_graph(data, key, train_pos_indices)
    src_type, rel_type, dst_type = key

    # 收集验证/测试中的节点
    val_src = set(val_idx[0].cpu().numpy().tolist())
    val_dst = set(val_idx[1].cpu().numpy().tolist())
    test_src = set(test_idx[0].cpu().numpy().tolist())
    test_dst = set(test_idx[1].cpu().numpy().tolist())
    protected_src = val_src | test_src
    protected_dst = val_dst | test_dst

    # 对非目标边类型，移除任何一端为保护节点的边
    for et in train_data.edge_types:
        if et == key or et == (dst_type, f"rev_{rel_type}", src_type):
            continue
        st, _, dt = et
        ei = train_data[et].edge_index.cpu().numpy()
        mask = np.ones(ei.shape[1], dtype=bool)
        if st == src_type:
            mask &= ~np.isin(ei[0], list(protected_src))
        if dt == dst_type:
            mask &= ~np.isin(ei[1], list(protected_dst))
        if st == dst_type:
            mask &= ~np.isin(ei[0], list(protected_dst))
        if dt == src_type:
            mask &= ~np.isin(ei[1], list(protected_src))
        train_data[et].edge_index = torch.from_numpy(ei[:, mask]).long().to(data[key].edge_index.device)

    logger.info("严格训练图已构建，移除了与 val/test 节点相连的间接边")
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


class GCNLP(nn.Module):
    """同质 GCN 编码器 + MLP 预测头."""

    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = HIDDEN,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.max_feat_dim = max(data[nt].x.shape[1] for nt in data.node_types)
        in_channels = self.max_feat_dim
        self.num_nodes = sum(data[nt].num_nodes for nt in data.node_types)
        self.node_type_offset = GATLP._compute_offsets(data)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_ch, hidden_dim))
        self.predictor = MLPLinkPredictor(hidden_dim, hidden_dim, dropout)
        self.dropout = dropout

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
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class Node2VecLP(nn.Module):
    """Node2Vec 结构嵌入 + MLP 预测头 (自实现, 不依赖 torch-cluster/pyg-lib)."""

    def __init__(
        self,
        data: HeteroData,
        edge_index: torch.Tensor,
        hidden_dim: int = HIDDEN,
        walk_length: int = N2V_WALK_LENGTH,
        context_size: int = N2V_CONTEXT_SIZE,
        walks_per_node: int = N2V_WALKS_PER_NODE,
        p: float = 1.0,
        q: float = 1.0,
        dropout: float = DROPOUT,
        train_node_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = sum(data[nt].num_nodes for nt in data.node_types)
        self.node_type_offset = GATLP._compute_offsets(data)
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.embedding = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.predictor = MLPLinkPredictor(hidden_dim, hidden_dim, dropout)
        self.dropout = dropout
        self._edge_index = edge_index.cpu()
        # 仅保留训练节点用于结构嵌入学习，避免验证/测试节点信息泄漏
        if train_node_mask is None:
            self.train_node_mask = torch.ones(self.num_nodes, dtype=torch.bool)
        else:
            self.train_node_mask = train_node_mask.cpu().bool()
        self.train_node_indices = self.train_node_mask.nonzero(as_tuple=False).flatten().to(DEVICE)
        self._adj = self._build_adj(edge_index)

    def _build_adj(self, edge_index: torch.Tensor) -> dict[int, np.ndarray]:
        """构建训练节点的邻居列表 (无向图，仅含训练节点之间的边)."""
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
        train_nodes = set(self.train_node_mask.nonzero(as_tuple=False).flatten().tolist())
        adj: dict[int, set[int]] = {i: set() for i in range(self.num_nodes)}
        for s, d in zip(src, dst):
            s_int, d_int = int(s), int(d)
            if s_int in train_nodes and d_int in train_nodes:
                adj[s_int].add(d_int)
                adj[d_int].add(s_int)
        return {k: np.array(list(v), dtype=np.int64) for k, v in adj.items()}

    def _random_walks(self, seed: int = SEED) -> np.ndarray:
        """生成 Node2Vec 随机游走序列 (仅在训练节点内游走，长度不足时以末尾节点填充)."""
        rng = np.random.default_rng(seed)
        walks: list[np.ndarray] = []
        nodes = self.train_node_mask.nonzero(as_tuple=False).flatten().cpu().numpy()
        if len(nodes) == 0:
            logger.warning("Node2Vec 训练节点掩码为空，无法生成随机游走")
            return np.empty((0, self.walk_length), dtype=np.int64)
        for _ in range(self.walks_per_node):
            rng.shuffle(nodes)
            for start in nodes:
                walk = [int(start)]
                if self.walk_length == 0:
                    walks.append(np.array(walk))
                    continue
                prev = -1
                cur = int(start)
                for _step in range(self.walk_length - 1):
                    neighbors = self._adj.get(cur, np.array([], dtype=np.int64))
                    if len(neighbors) == 0:
                        break
                    if prev < 0 or (self.p == 1.0 and self.q == 1.0):
                        nxt = int(rng.choice(neighbors))
                    else:
                        # biased second-order walk
                        weights = np.ones(len(neighbors), dtype=np.float64)
                        prev_neighbors = set(self._adj.get(prev, []))
                        for idx, nb in enumerate(neighbors):
                            if nb == prev:
                                weights[idx] = 1.0 / self.p
                            elif nb in prev_neighbors:
                                weights[idx] = 1.0
                            else:
                                weights[idx] = 1.0 / self.q
                        weights /= weights.sum()
                        nxt = int(rng.choice(neighbors, p=weights))
                    walk.append(nxt)
                    prev, cur = cur, nxt
                # 统一长度: 孤立/早期终止的游走后补最后一个节点
                while len(walk) < self.walk_length:
                    walk.append(walk[-1])
                walks.append(np.array(walk, dtype=np.int64))
        return np.stack(walks)

    def _pairs_from_walks(self, walks: np.ndarray) -> list[tuple[int, int]]:
        """从游走序列中提取 (center, context) 对."""
        pairs: list[tuple[int, int]] = []
        half = self.context_size // 2
        for walk in walks:
            for i, center in enumerate(walk):
                window = walk[max(0, i - half) : min(len(walk), i + half + 1)]
                for ctx in window:
                    if ctx != center:
                        pairs.append((int(center), int(ctx)))
        return pairs

    def forward(self, data: HeteroData, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        return self.embedding.weight

    def pretrain(
        self, epochs: int = N2V_PRETRAIN_EPOCHS, lr: float = 0.01, batch_size: int = 512
    ) -> list[float]:
        """使用 skip-gram 负采样预训练 Node2Vec 嵌入."""
        walks = self._random_walks()
        pairs = self._pairs_from_walks(walks)
        if not pairs:
            logger.warning("Node2Vec 未生成有效训练对, 跳过预训练")
            return []
        centers = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=DEVICE)
        contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=DEVICE)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: list[float] = []
        n_samples = len(pairs)
        for epoch in range(epochs):
            perm = torch.randperm(n_samples, device=DEVICE)
            total_loss = 0.0
            n_batches = 0
            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                c = centers[idx]
                pos = contexts[idx]
                # 负采样限制在训练节点内，避免更新验证/测试节点嵌入
                neg_idx = torch.randint(0, len(self.train_node_indices), (len(idx),), device=DEVICE)
                neg = self.train_node_indices[neg_idx]
                emb_c = self.embedding(c)
                emb_pos = self.embedding(pos)
                emb_neg = self.embedding(neg)
                pos_score = torch.sum(emb_c * emb_pos, dim=1)
                neg_score = torch.sum(emb_c * emb_neg, dim=1)
                loss = -(
                    F.logsigmoid(pos_score).mean() + F.logsigmoid(-neg_score).mean()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_batches += 1
            avg_loss = total_loss / max(n_batches, 1)
            history.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info("[Node2Vec pretrain epoch %d] loss %.4f", epoch + 1, avg_loss)
        return history


class MLPLP(nn.Module):
    """仅节点特征的 MLP 基线：无图消息传递."""

    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = HIDDEN,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.projector = FeatureProjector(data, hidden_dim)
        self.predictor = MLPLinkPredictor(hidden_dim, hidden_dim, dropout)

    def forward(self, data: HeteroData, _edge_index: Any | None = None) -> dict[str, torch.Tensor]:
        return self.projector(data)


class RandomLP(nn.Module):
    """随机基线：输出不可学习随机得分."""

    def __init__(self, data: HeteroData) -> None:
        super().__init__()
        self.data = data

    def forward(self, data: HeteroData, _edge_index: Any | None = None) -> dict[str, torch.Tensor]:
        return {nt: torch.zeros(data[nt].num_nodes, 1, device=DEVICE) for nt in data.node_types}

    def predict(self, src_emb: torch.Tensor, dst_emb: torch.Tensor) -> torch.Tensor:
        return torch.randn(src_emb.shape[0], device=src_emb.device)


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
        elif isinstance(model, (GATLP, GCNLP)):
            z = model(data, homogeneous_edge_index)
            src_idx = edge_label_index[0] + offsets[src_type]
            dst_idx = edge_label_index[1] + offsets[dst_type]
            src_emb = z[src_idx]
            dst_emb = z[dst_idx]
        elif isinstance(model, Node2VecLP):
            z = model(data)
            src_idx = edge_label_index[0] + offsets[src_type]
            dst_idx = edge_label_index[1] + offsets[dst_type]
            src_emb = z[src_idx]
            dst_emb = z[dst_idx]
        elif isinstance(model, MLPLP):
            z_dict = model(data)
            src_emb = z_dict[src_type][edge_label_index[0]]
            dst_emb = z_dict[dst_type][edge_label_index[1]]
        elif isinstance(model, RandomLP):
            src_emb = torch.zeros(edge_label_index.shape[1], 1, device=edge_label_index.device)
            dst_emb = torch.zeros(edge_label_index.shape[1], 1, device=edge_label_index.device)
        else:
            raise TypeError(f"未知模型类型: {type(model)}")

        if isinstance(model, RandomLP):
            logits = model.predict(src_emb, dst_emb)
        else:
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
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """训练单个模型，返回验证集最优指标及训练历史.

    训练阶段使用 train_data 进行消息传递（已剔除验证/测试边），
    评估阶段使用完整 data.
    """
    # 随机基线无需训练
    if isinstance(model, RandomLP):
        return {"auc": 0.5, "ap": 0.5, "loss": 0.0}, {"train_loss": [], "val_auc": [], "val_ap": []}

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

    # GAT/GCN 训练用同质边索引
    train_homogeneous_edge_index = None
    if isinstance(model, (GATLP, GCNLP, Node2VecLP)):
        train_homogeneous_edge_index = to_homogeneous_edge_index(train_data, offsets).to(DEVICE)

    # Node2Vec 先预训练结构嵌入，再冻结嵌入只训练预测头
    n2v_pretrain_history: list[float] = []
    if isinstance(model, Node2VecLP):
        logger.info("预训练 Node2Vec 嵌入...")
        model.to(DEVICE)
        n2v_pretrain_history = model.pretrain(epochs=N2V_PRETRAIN_EPOCHS, lr=0.01)
        for param in model.embedding.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=PATIENCE // 2
    )

    best_val_auc = -np.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_auc": [],
        "val_ap": [],
        "n2v_pretrain_loss": n2v_pretrain_history,
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        if isinstance(model, HGTLP):
            z_dict = model(train_data)
            src_emb = z_dict[src_type][train_idx[0]]
            dst_emb = z_dict[dst_type][train_idx[1]]
        elif isinstance(model, (GATLP, GCNLP)):
            z = model(train_data, train_homogeneous_edge_index)
            src_idx = train_idx[0] + offsets[src_type]
            dst_idx = train_idx[1] + offsets[dst_type]
            src_emb = z[src_idx]
            dst_emb = z[dst_idx]
        elif isinstance(model, Node2VecLP):
            z = model(train_data)
            src_idx = train_idx[0] + offsets[src_type]
            dst_idx = train_idx[1] + offsets[dst_type]
            src_emb = z[src_idx]
            dst_emb = z[dst_idx]
        elif isinstance(model, MLPLP):
            z_dict = model(train_data)
            src_emb = z_dict[src_type][train_idx[0]]
            dst_emb = z_dict[dst_type][train_idx[1]]
        else:
            raise TypeError(f"未知模型类型: {type(model)}")

        logits = model.predictor(src_emb, dst_emb)
        loss = F.binary_cross_entropy_with_logits(logits, train_labels)
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(
            model, data, val_idx, val_labels, src_type, dst_type,
            offsets=offsets, homogeneous_edge_index=homogeneous_edge_index,
        )
        val_auc = val_metrics["auc"]
        scheduler.step(val_auc)

        history["train_loss"].append(float(loss.item()))
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_metrics["ap"])

        if epoch % 10 == 0:
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
    return val_metrics, history


def _get_node_name_to_idx(node_names: dict[str, list[str]], node_type: str) -> dict[str, int]:
    """从节点名称列表中获取名称到索引的映射."""
    names = node_names.get(node_type, [])
    return {name: idx for idx, name in enumerate(names)}


def compute_bcp_metrics(
    model: nn.Module,
    data: HeteroData,
    task: str,
    node_names: dict[str, list[str]],
    offsets: dict[str, int] | None = None,
    homogeneous_edge_index: torch.Tensor | None = None,
) -> dict[str, Any]:
    """计算 BCP 对项目基因集的靶向相关性指标.

    仅对 ct（compound-targets-gene）任务有意义；gp 任务返回空指标。
    """
    src_type, _, dst_type = get_target_edge_index(data, task)[0]
    if src_type != "compound" or dst_type != "gene":
        return {"note": f"BCP metrics only defined for ct task, got {src_type}->{dst_type}"}

    compound2idx = _get_node_name_to_idx(node_names, src_type)
    gene2idx = _get_node_name_to_idx(node_names, dst_type)

    bcp_idx = compound2idx.get(TARGET_COMPOUND)
    if bcp_idx is None:
        logger.warning("未在图中找到化合物 %s", TARGET_COMPOUND)
        return {"error": f"compound {TARGET_COMPOUND} not found"}

    target_gene_indices = {g: gene2idx.get(g) for g in TARGET_GENES}
    target_gene_indices = {k: v for k, v in target_gene_indices.items() if v is not None}

    model.eval()
    with torch.no_grad():
        if isinstance(model, HGTLP):
            z_dict = model(data)
            src_emb = z_dict[src_type][bcp_idx].unsqueeze(0)
            dst_emb = z_dict[dst_type]
        elif isinstance(model, (GATLP, GCNLP)):
            z = model(data, homogeneous_edge_index)
            src_idx = bcp_idx + offsets[src_type]
            src_emb = z[src_idx].unsqueeze(0)
            dst_emb = z[offsets[dst_type] : offsets[dst_type] + data[dst_type].num_nodes]
        elif isinstance(model, Node2VecLP):
            z = model(data)
            src_idx = bcp_idx + offsets[src_type]
            src_emb = z[src_idx].unsqueeze(0)
            dst_emb = z[offsets[dst_type] : offsets[dst_type] + data[dst_type].num_nodes]
        elif isinstance(model, MLPLP):
            z_dict = model(data)
            src_emb = z_dict[src_type][bcp_idx].unsqueeze(0)
            dst_emb = z_dict[dst_type]
        elif isinstance(model, RandomLP):
            return {"note": "random baseline has no meaningful ranking"}
        else:
            raise TypeError(f"未知模型类型: {type(model)}")

        # 计算 BCP 对所有基因/化合物的得分
        src_emb = src_emb.expand(dst_emb.shape[0], -1)
        scores = torch.sigmoid(model.predictor(src_emb, dst_emb)).cpu().numpy()

    # 排名：按分数降序
    ranks = np.argsort(-scores)
    score_ranks = {gene: int(np.where(ranks == idx)[0][0]) + 1 for gene, idx in target_gene_indices.items()}

    metrics: dict[str, Any] = {
        "num_candidates": int(data[dst_type].num_nodes),
        "target_gene_ranks": score_ranks,
        "mean_reciprocal_rank": float(np.mean([1.0 / r for r in score_ranks.values()])) if score_ranks else 0.0,
    }
    for k in [1, 5, 10, 20, 50]:
        metrics[f"hits@{k}"] = int(sum(1 for r in score_ranks.values() if r <= k))

    return metrics


def _run_single_split(
    data: HeteroData,
    node_names: dict[str, list[str]],
    task: str,
    fold_idx: int | None = None,
) -> dict[str, Any]:
    """运行单次划分实验."""
    rng = np.random.default_rng(SEED if fold_idx is None else SEED + fold_idx)
    key, num_edges = get_target_edge_index(data, task)
    src_type, _, dst_type = key
    logger.info("=" * 60)
    logger.info("任务: %s (fold=%s)", task.upper(), fold_idx if fold_idx is not None else "single")
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
        split_info,
    ) = prepare_link_prediction_data(data, task, rng=rng)

    if STRICT_TRAIN_GRAPH:
        train_data = build_strict_train_graph(data, key, train_pos_indices, val_idx, test_idx)
    else:
        train_data = build_train_graph(data, key, train_pos_indices)

    offsets = GATLP._compute_offsets(data)
    homogeneous_edge_index = to_homogeneous_edge_index(data, offsets).to(DEVICE)
    train_homogeneous_edge_index = to_homogeneous_edge_index(train_data, offsets).to(DEVICE)
    data_dev = data.to(DEVICE)

    # 计算 Node2Vec 训练节点掩码，限制预训练仅在训练节点间游走
    num_nodes_total = sum(data[nt].num_nodes for nt in data.node_types)
    src_global = train_idx[0] + offsets[src_type]
    dst_global = train_idx[1] + offsets[dst_type]
    train_node_indices = torch.cat([src_global, dst_global]).unique()
    train_node_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    train_node_mask[train_node_indices] = True

    models: dict[str, nn.Module] = {
        "HGT": HGTLP(data).to(DEVICE),
        "GAT": GATLP(data).to(DEVICE),
        "GCN": GCNLP(data).to(DEVICE),
        "Node2Vec": Node2VecLP(data, train_homogeneous_edge_index, train_node_mask=train_node_mask).to(DEVICE),
        "MLP": MLPLP(data).to(DEVICE),
        "Random": RandomLP(data).to(DEVICE),
    }

    fold_result: dict[str, Any] = {
        "task": task,
        "fold": fold_idx,
        "num_pos_edges": num_edges,
        "train_samples": int(train_labels.shape[0]),
        "val_samples": int(val_labels.shape[0]),
        "test_samples": int(test_labels.shape[0]),
        "models": {},
    }

    for model_name, model in models.items():
        logger.info("训练 %s...", model_name)
        val_metrics, history = train_model(
            model,
            data_dev,
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
        test_metrics = evaluate(
            model,
            data_dev,
            test_idx.to(DEVICE),
            test_labels.to(DEVICE),
            src_type,
            dst_type,
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
        )
        bcp_metrics = compute_bcp_metrics(
            model, data_dev, task, node_names, offsets=offsets, homogeneous_edge_index=homogeneous_edge_index
        )

        fold_result["models"][model_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "history": history,
            "bcp_metrics": bcp_metrics,
        }
        logger.info(
            "[%s] val_auc=%.4f test_auc=%.4f test_ap=%.4f",
            model_name,
            val_metrics["auc"],
            test_metrics["auc"],
            test_metrics["ap"],
        )

    fold_result["split_info"] = split_info
    return fold_result


def _aggregate_cv_results(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 K-Fold 结果，计算均值与标准差."""
    aggregated: dict[str, Any] = {}
    task = fold_results[0]["task"]
    aggregated["task"] = task
    aggregated["num_folds"] = len(fold_results)

    model_names = list(fold_results[0]["models"].keys())
    for model_name in model_names:
        val_aucs = [f["models"][model_name]["val"]["auc"] for f in fold_results]
        test_aucs = [f["models"][model_name]["test"]["auc"] for f in fold_results]
        val_aps = [f["models"][model_name]["val"]["ap"] for f in fold_results]
        test_aps = [f["models"][model_name]["test"]["ap"] for f in fold_results]

        aggregated[model_name] = {
            "val_auc": {"mean": float(np.mean(val_aucs)), "std": float(np.std(val_aucs))},
            "test_auc": {"mean": float(np.mean(test_aucs)), "std": float(np.std(test_aucs))},
            "val_ap": {"mean": float(np.mean(val_aps)), "std": float(np.std(val_aps))},
            "test_ap": {"mean": float(np.mean(test_aps)), "std": float(np.std(test_aps))},
        }

        # BCP 指标聚合
        bcp_hits: dict[str, list[int]] = {}
        bcp_mrrs: list[float] = []
        for f in fold_results:
            bcp = f["models"][model_name].get("bcp_metrics", {})
            if "mean_reciprocal_rank" in bcp:
                bcp_mrrs.append(bcp["mean_reciprocal_rank"])
            for k in [1, 5, 10, 20, 50]:
                key = f"hits@{k}"
                bcp_hits.setdefault(key, []).append(bcp.get(key, 0))

        mrr_mean = float(np.mean(bcp_mrrs)) if bcp_mrrs else None
        mrr_std = float(np.std(bcp_mrrs)) if bcp_mrrs else None
        aggregated[model_name]["bcp"] = {
            "mean_reciprocal_rank": {"mean": mrr_mean, "std": mrr_std},
        }
        for key, vals in bcp_hits.items():
            aggregated[model_name]["bcp"][key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    return aggregated


def run_ablation() -> dict[str, Any]:
    set_seed()
    data, node_names = build_data()
    results: dict[str, Any] = {
        "device": str(DEVICE),
        "seed": SEED,
        "use_cv": USE_CV,
        "n_folds": N_FOLDS if USE_CV else 1,
        "strict_train_graph": STRICT_TRAIN_GRAPH,
        "tasks": {},
    }

    for task in ["gp", "ct"]:
        if USE_CV:
            # 使用基于源节点的分层 K-Fold
            key, num_edges = get_target_edge_index(data, task)
            src_type, _, dst_type = key
            num_src = data[src_type].num_nodes
            edge_index = data[key].edge_index.cpu().numpy()
            seen: set[tuple[int, int]] = set()
            positives: list[tuple[int, int]] = []
            for i in range(edge_index.shape[1]):
                e = (int(edge_index[0, i]), int(edge_index[1, i]))
                if e not in seen:
                    seen.add(e)
                    positives.append(e)
            positives = np.array(positives)
            stratify = _stratify_labels(positives, num_src, task)

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
            fold_results: list[dict[str, Any]] = []
            for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(positives, stratify)):
                # 进一步将 train_val 划分为 train/val（保持分层）
                train_val_pos = positives[train_val_idx]
                train_val_stratify = stratify[train_val_idx]
                n_train = int(len(train_val_pos) * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO))
                fold_rng = np.random.default_rng(SEED + fold_idx)
                train_order = _stratified_shuffle(np.arange(len(train_val_pos)), train_val_stratify, fold_rng)
                train_pos = train_val_pos[train_order[:n_train]]
                val_pos = train_val_pos[train_order[n_train:]]
                test_pos = positives[test_idx]

                fold_result = _run_fold_from_splits(data, node_names, task, fold_idx, train_pos, val_pos, test_pos)
                fold_results.append(fold_result)

            results["tasks"][task] = {
                "fold_results": fold_results,
                "aggregated": _aggregate_cv_results(fold_results),
            }
        else:
            single = _run_single_split(data, node_names, task)
            results["tasks"][task] = {
                "single": single,
            }

    return results


def _run_fold_from_splits(
    data: HeteroData,
    node_names: dict[str, list[str]],
    task: str,
    fold_idx: int,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    test_pos: np.ndarray,
) -> dict[str, Any]:
    """给定已划分好的正样本，生成负样本并运行一次 fold."""
    key, num_edges = get_target_edge_index(data, task)
    src_type, _, dst_type = key
    num_src = data[src_type].num_nodes
    num_dst = data[dst_type].num_nodes
    rng = np.random.default_rng(SEED + fold_idx)

    # 收集正样本
    full_positives = np.vstack([train_pos, val_pos, test_pos])
    forbidden = set(map(tuple, full_positives.tolist()))
    n_neg_total = int(len(full_positives) * NEGATIVE_RATIO)
    negatives = sample_reliable_negatives(
        num_src=num_src,
        num_dst=num_dst,
        num_negatives=n_neg_total,
        forbidden=forbidden,
        exclude_self_loops=(task == "gp"),
        rng=rng,
    )
    neg_array = np.array(negatives)

    # 按源节点分层为负样本分配 fold
    neg_stratify = _stratify_labels(neg_array, num_src, task)
    # 保持与正样本相同的 fold 边界比例
    n_neg = len(neg_array)
    n_neg_train = int(n_neg * len(train_pos) / len(full_positives))
    n_neg_val = int(n_neg * len(val_pos) / len(full_positives))
    neg_order = _stratified_shuffle(np.arange(n_neg), neg_stratify, rng)
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

    train_idx, train_labels = _to_tensor(train_pos, neg_train)
    val_idx, val_labels = _to_tensor(val_pos, neg_val)
    test_idx, test_labels = _to_tensor(test_pos, neg_test)

    # 恢复 train_pos_indices：找到 train_pos 在 data[key].edge_index 中的索引
    edge_index = data[key].edge_index.cpu().numpy()
    pos_to_idx: dict[tuple[int, int], int] = {}
    seen_local: set[tuple[int, int]] = set()
    for i in range(edge_index.shape[1]):
        e = (int(edge_index[0, i]), int(edge_index[1, i]))
        if e not in seen_local:
            seen_local.add(e)
            pos_to_idx[e] = i
    train_pos_indices_list = [pos_to_idx[tuple(e.tolist())] for e in train_pos]
    train_pos_indices = torch.tensor(train_pos_indices_list, dtype=torch.long)

    if STRICT_TRAIN_GRAPH:
        train_data = build_strict_train_graph(data, key, train_pos_indices, val_idx, test_idx)
    else:
        train_data = build_train_graph(data, key, train_pos_indices)

    offsets = GATLP._compute_offsets(data)
    homogeneous_edge_index = to_homogeneous_edge_index(data, offsets).to(DEVICE)
    train_homogeneous_edge_index = to_homogeneous_edge_index(train_data, offsets).to(DEVICE)
    data_dev = data.to(DEVICE)

    # 计算 Node2Vec 训练节点掩码，限制预训练仅在训练节点间游走
    num_nodes_total = sum(data[nt].num_nodes for nt in data.node_types)
    src_global = train_idx[0] + offsets[src_type]
    dst_global = train_idx[1] + offsets[dst_type]
    train_node_indices = torch.cat([src_global, dst_global]).unique()
    train_node_mask = torch.zeros(num_nodes_total, dtype=torch.bool)
    train_node_mask[train_node_indices] = True

    models: dict[str, nn.Module] = {
        "HGT": HGTLP(data).to(DEVICE),
        "GAT": GATLP(data).to(DEVICE),
        "GCN": GCNLP(data).to(DEVICE),
        "Node2Vec": Node2VecLP(data, train_homogeneous_edge_index, train_node_mask=train_node_mask).to(DEVICE),
        "MLP": MLPLP(data).to(DEVICE),
        "Random": RandomLP(data).to(DEVICE),
    }

    fold_result: dict[str, Any] = {
        "task": task,
        "fold": fold_idx,
        "num_pos_edges": num_edges,
        "train_samples": int(train_labels.shape[0]),
        "val_samples": int(val_labels.shape[0]),
        "test_samples": int(test_labels.shape[0]),
        "models": {},
    }

    for model_name, model in models.items():
        logger.info("训练 %s (fold %d)...", model_name, fold_idx)
        val_metrics, history = train_model(
            model,
            data_dev,
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
        test_metrics = evaluate(
            model,
            data_dev,
            test_idx.to(DEVICE),
            test_labels.to(DEVICE),
            src_type,
            dst_type,
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
        )
        bcp_metrics = compute_bcp_metrics(
            model, data_dev, task, node_names, offsets=offsets, homogeneous_edge_index=homogeneous_edge_index
        )

        fold_result["models"][model_name] = {
            "val": val_metrics,
            "test": test_metrics,
            "history": history,
            "bcp_metrics": bcp_metrics,
        }
        logger.info(
            "[%s fold %d] val_auc=%.4f test_auc=%.4f test_ap=%.4f",
            model_name,
            fold_idx,
            val_metrics["auc"],
            test_metrics["auc"],
            test_metrics["ap"],
        )

    return fold_result


def main() -> None:
    try:
        results = run_ablation()
        out_path = PROJECT_ROOT / "L3_results" / "ablation_hgt_vs_gat_v3.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("结果已保存: %s", out_path)
        logger.info(json.dumps(results["tasks"], indent=2, ensure_ascii=False))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
