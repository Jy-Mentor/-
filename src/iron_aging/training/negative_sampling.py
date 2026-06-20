"""负采样与边泄漏修复模块.

为链路预测任务生成负样本边, 并支持从训练图中移除验证/测试正样本边,
避免训练时的标签泄漏.
"""

from __future__ import annotations

import numpy as np
import torch


def negative_sample_edges(
    num_src: int,
    num_dst: int,
    num_negatives: int,
    positive_edges: set[tuple[int, int]],
    rng: np.random.Generator | int | None = None,
    max_trials: int = 100_000,
) -> list[tuple[int, int]]:
    """从源-目标节点空间中采样负样本边.

    Args:
        num_src: 源节点数量.
        num_dst: 目标节点数量.
        num_negatives: 需要采样的负样本数量.
        positive_edges: 已知正样本边集合, 避免重复采样.
        rng: 随机数生成器或种子.
        max_trials: 最大尝试次数, 防止无限循环.

    Returns:
        负样本边列表.
    """
    if rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, int):
        rng = np.random.default_rng(rng)

    negatives: set[tuple[int, int]] = set()
    trials = 0
    while len(negatives) < num_negatives and trials < max_trials:
        src = rng.integers(0, num_src)
        dst = rng.integers(0, num_dst)
        edge = (int(src), int(dst))
        if edge not in positive_edges:
            negatives.add(edge)
        trials += 1

    if len(negatives) < num_negatives:
        raise RuntimeError(
            f"无法采够 {num_negatives} 条负样本 (实际 {len(negatives)}), "
            "可能正样本过于稠密"
        )

    return list(negatives)


def build_link_prediction_labels(
    positive_edges: list[tuple[int, int]],
    negative_edges: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[float]]:
    """合并正负样本并生成标签.

    Args:
        positive_edges: 正样本边.
        negative_edges: 负样本边.

    Returns:
        (edges, labels), labels 中 1 表示正样本, 0 表示负样本.
    """
    edges = positive_edges + negative_edges
    labels = [1.0] * len(positive_edges) + [0.0] * len(negative_edges)
    return edges, labels


def remove_leaked_edges(
    edge_index: torch.Tensor | None,
    leak_set: set[tuple[int, int]],
    device: torch.device | None = None,
) -> torch.Tensor | None:
    """从边索引中移除属于验证/测试集的边, 防止训练时标签泄漏.

    参考: link prediction 标准做法 (Kipf & Welling, 2016).

    Args:
        edge_index: 2 x N 的边索引张量.
        leak_set: 需要移除的边集合, 元素为 (src, dst) 元组.
        device: 输出张量所在设备, 默认与输入相同.

    Returns:
        移除泄漏边后的边索引张量; 若输入为空或 leak_set 为空则原样返回.
    """
    if edge_index is None or len(leak_set) == 0:
        return edge_index

    if device is None:
        device = edge_index.device

    ei = edge_index.cpu().numpy()
    mask = np.ones(ei.shape[1], dtype=bool)
    for src, dst in leak_set:
        match = (ei[0] == src) & (ei[1] == dst)
        mask[match] = False
    return torch.from_numpy(ei[:, mask]).to(device)
