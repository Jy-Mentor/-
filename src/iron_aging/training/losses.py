"""训练损失函数模块."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_with_logits_loss(
    logits: torch.Tensor, labels: torch.Tensor, pos_weight: float | None = None
) -> torch.Tensor:
    """带可选正样本权重的 BCEWithLogitsLoss.

    Args:
        logits: 预测分数 [N, 1] 或 [N].
        labels: 真实标签, 与 logits 同形状.
        pos_weight: 正样本权重系数.

    Returns:
        标量损失.
    """
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if labels.dim() == 2 and labels.size(-1) == 1:
        labels = labels.squeeze(-1)

    weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device) if pos_weight is not None else None
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=weight)


def vib_kl_loss(mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1e-3) -> torch.Tensor:
    """变分信息瓶颈 KL 损失.

    Args:
        mu: 潜在变量均值.
        logvar: 潜在变量对数方差.
        beta: KL 项权重.

    Returns:
        标量 KL 损失.
    """
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * kl
