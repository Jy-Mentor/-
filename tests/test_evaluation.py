"""测试评估层模块."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iron_aging.evaluation.explainability import compute_edge_attribution_gradient  # noqa: E402
from iron_aging.evaluation.metrics import (  # noqa: E402
    compute_ap,
    compute_auc,
    compute_classification_metrics,
)
from iron_aging.models.link_predictor import LinkPredictor  # noqa: E402


def test_compute_auc():
    """验证 AUC 计算."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    auc = compute_auc(y_true, y_score)
    assert 0.0 <= auc <= 1.0


def test_compute_ap():
    """验证 AP 计算."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    ap = compute_ap(y_true, y_score)
    assert 0.0 <= ap <= 1.0


def test_compute_classification_metrics():
    """验证分类综合指标."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    metrics = compute_classification_metrics(y_true, y_score)
    assert "auc" in metrics
    assert "ap" in metrics
    assert "accuracy" in metrics


def test_compute_edge_attribution_gradient():
    """验证梯度归因计算."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = LinkPredictor(in_dim=4, hidden_dim=8, dropout=0.0)

    model = DummyModel()
    src_emb = torch.randn(5, 4, requires_grad=True)
    dst_emb = torch.randn(5, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    importance = compute_edge_attribution_gradient(model, src_emb, dst_emb, edge_index)
    assert importance.shape == (5,)
