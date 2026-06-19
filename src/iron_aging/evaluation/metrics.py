"""评估指标模块.

提供 AUC、AP、分类指标等常用评估函数.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 ROC-AUC.

    Args:
        y_true: 真实标签.
        y_score: 预测概率.

    Returns:
        AUC 值; 若标签单一则返回 0.0.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 Average Precision.

    Args:
        y_true: 真实标签.
        y_score: 预测概率.

    Returns:
        AP 值; 若标签单一则返回 0.0.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def compute_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """计算二分类综合指标.

    Args:
        y_true: 真实标签.
        y_score: 预测概率.
        threshold: 二分类阈值.

    Returns:
        包含 auc/ap/accuracy/precision/recall 的字典.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "auc": compute_auc(y_true, y_score),
        "ap": compute_ap(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    return metrics
