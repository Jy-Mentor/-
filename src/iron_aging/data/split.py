"""可复现的链路预测数据拆分模块.

为异质图链路预测提供统一、可配置的训练/验证/测试拆分策略:
- random: 基于 sklearn.train_test_split 的随机划分.
- stratified: 按源节点 ID 分层随机划分, 保证每个源节点在训练/验证/测试中都有代表.

所有拆分函数均接受固定的 numpy Generator 或种子, 确保结果可复现.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _stratify_labels(pos_edges: np.ndarray, num_src: int, task: str) -> np.ndarray:
    """为分层抽样生成分层标签.

    ct 任务按源化合物(src)分层; gp 任务按源基因度分 5 桶.
    """
    if task == "ct":
        return pos_edges[:, 0]

    degrees = np.bincount(pos_edges[:, 0], minlength=num_src)
    src_degrees = degrees[pos_edges[:, 0]]
    bins = np.percentile(src_degrees, [20, 40, 60, 80])
    return np.digitize(src_degrees, bins)


def _stratified_shuffle(
    indices: np.ndarray, labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """在每组标签内部独立打乱, 返回全局索引顺序."""
    order: list[int] = []
    for label in np.unique(labels):
        group = indices[labels == label].copy()
        rng.shuffle(group)
        order.extend(group.tolist())
    return np.array(order)


def split_edges_random(
    edge_index: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """随机划分边索引为训练/验证/测试集."""
    num_edges = edge_index.shape[1]
    indices = np.arange(num_edges)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, random_state=seed
    )
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, random_state=seed
    )
    return edge_index[:, train_idx], edge_index[:, val_idx], edge_index[:, test_idx]


def split_edges_stratified(
    edge_index: torch.Tensor,
    num_src: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    task: str = "ct",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按源节点分层划分边索引.

    对 compound-target 等任务, 按源节点 ID 分层后再随机划分,
    避免某些源节点只出现在验证/测试集中导致分布偏移.
    """
    edges_np = edge_index.cpu().numpy().T
    stratify = _stratify_labels(edges_np, num_src, task)

    n_pos = len(edges_np)
    n_pos_train = int(n_pos * train_ratio)
    n_pos_val = int(n_pos * val_ratio)

    rng = np.random.default_rng(seed)
    order = _stratified_shuffle(np.arange(n_pos), stratify, rng)

    train_edges = edges_np[order[:n_pos_train]]
    val_edges = edges_np[order[n_pos_train : n_pos_train + n_pos_val]]
    test_edges = edges_np[order[n_pos_train + n_pos_val :]]

    return (
        torch.from_numpy(train_edges.T).long(),
        torch.from_numpy(val_edges.T).long(),
        torch.from_numpy(test_edges.T).long(),
    )


def split_edges(
    edge_index: torch.Tensor,
    num_src: int | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    mode: str = "random",
    task: str = "ct",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """统一的边拆分入口.

    Args:
        edge_index: 2 x E 的正样本边索引.
        num_src: 源节点数量, stratified 模式下必需.
        train_ratio: 训练集比例.
        val_ratio: 验证集比例.
        seed: 随机种子.
        mode: "random" 或 "stratified".
        task: 任务类型, 用于 stratified 模式的分层标签生成.

    Returns:
        train_ei, val_ei, test_ei: 三个 2 x E 的边索引张量.
    """
    if mode == "stratified":
        if num_src is None:
            raise ValueError("stratified 拆分需要提供 num_src")
        return split_edges_stratified(
            edge_index, num_src, train_ratio, val_ratio, seed, task
        )
    if mode != "random":
        logger.warning("未知拆分模式 %s, 回退到 random", mode)
    return split_edges_random(edge_index, train_ratio, val_ratio, seed)


def save_split_info(
    split_info: dict[str, Any], output_path: Any, encoding: str = "utf-8"
) -> None:
    """将拆分信息持久化到 JSON 文件, 便于审计与复现."""
    import json
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)
