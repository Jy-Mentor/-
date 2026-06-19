"""负采样模块.

为链路预测任务生成负样本边, 确保不采样已知正样本.
"""

from __future__ import annotations

import numpy as np


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
