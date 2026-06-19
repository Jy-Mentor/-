"""可解释性分析模块.

提供基于梯度的边重要性估计, 用于解释模型预测.
"""

from __future__ import annotations

import torch


def compute_edge_attribution_gradient(
    model: torch.nn.Module,
    src_emb: torch.Tensor,
    dst_emb: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """计算边对链路预测输出的梯度归因.

    Args:
        model: 包含 predictor 的模型.
        src_emb: 源节点嵌入 [N, d].
        dst_emb: 目标节点嵌入 [N, d].
        edge_index: 边索引 [2, E], 对应需要解释的边.

    Returns:
        每条边的重要性分数 [E].
    """
    predictor = getattr(model, "predictor", None)
    if predictor is None:
        raise ValueError("模型需包含 predictor 属性")

    src_emb.requires_grad_(True)
    dst_emb.requires_grad_(True)

    logits = predictor(src_emb, dst_emb).squeeze(-1)
    # 对目标边索引对应的 logit 求和并反向传播
    selected = logits[edge_index[0]] if edge_index.size(1) == logits.size(0) else logits
    selected.sum().backward(retain_graph=True)

    if src_emb.grad is None or dst_emb.grad is None:
        raise RuntimeError("梯度未正确计算")

    importance = (src_emb.grad.abs().sum(dim=1) + dst_emb.grad.abs().sum(dim=1)) / 2.0
    return importance.detach()
