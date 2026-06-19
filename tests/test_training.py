"""测试训练层模块."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iron_aging.training.losses import bce_with_logits_loss, vib_kl_loss  # noqa: E402
from iron_aging.training.negative_sampling import (  # noqa: E402
    build_link_prediction_labels,
    negative_sample_edges,
)


def test_bce_with_logits_loss():
    """验证 BCE 损失计算."""
    logits = torch.tensor([0.0, 2.0, -1.0])
    labels = torch.tensor([0.0, 1.0, 0.0])
    loss = bce_with_logits_loss(logits, labels)
    assert loss.item() > 0


def test_vib_kl_loss():
    """验证 VIB KL 损失非负."""
    mu = torch.randn(10, 8)
    logvar = torch.randn(10, 8)
    loss = vib_kl_loss(mu, logvar, beta=1e-3)
    assert loss.item() >= 0


def test_negative_sample_edges():
    """验证负采样不命中正样本."""
    positive = {(0, 0), (1, 1), (2, 2)}
    negatives = negative_sample_edges(
        num_src=5, num_dst=5, num_negatives=10, positive_edges=positive, rng=42
    )
    assert len(negatives) == 10
    assert not (set(negatives) & positive)


def test_build_link_prediction_labels():
    """验证标签生成."""
    pos = [(0, 0), (1, 1)]
    neg = [(2, 3), (4, 5)]
    edges, labels = build_link_prediction_labels(pos, neg)
    assert len(edges) == 4
    assert labels == [1.0, 1.0, 0.0, 0.0]
