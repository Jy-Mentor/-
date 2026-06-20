"""测试训练层模块."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iron_aging.training.losses import bce_with_logits_loss, vib_kl_loss  # noqa: E402
from iron_aging.training.negative_sampling import (  # noqa: E402
    build_link_prediction_labels,
    negative_sample_edges,
    remove_leaked_edges,
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


def test_remove_leaked_edges():
    """验证移除泄漏边保留非泄漏边."""
    edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    leak_set = {(1, 1), (3, 3)}
    filtered = remove_leaked_edges(edge_index, leak_set)
    assert filtered is not None
    assert filtered.shape[1] == 2
    assert torch.equal(filtered, torch.tensor([[0, 2], [0, 2]]))


def test_remove_leaked_edges_empty_input():
    """验证空输入时 remove_leaked_edges 安全返回."""
    assert remove_leaked_edges(None, {(0, 0)}) is None
    edge_index = torch.tensor([[0, 1], [0, 1]])
    assert torch.equal(remove_leaked_edges(edge_index, set()), edge_index)


def test_hgt_trainer_with_custom_edge_type():
    """验证 HGTTrainer 支持自定义边类型."""
    from torch_geometric.data import HeteroData

    from iron_aging.models.link_predictor import LinkPredictor
    from iron_aging.training.trainer import HGTTrainer

    data = HeteroData()
    data["compound"].x = torch.randn(4, 8)
    data["gene"].x = torch.randn(5, 8)
    data["compound", "targets", "gene"].edge_index = torch.tensor([[0, 1], [0, 1]])

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = LinkPredictor(8, 16)

        def forward(self, x_dict, edge_index_dict):
            return x_dict

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = HGTTrainer(
        model, optimizer, edge_type=("compound", "targets", "gene")
    )

    edge_label_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    edge_labels = torch.tensor([1.0, 0.0, 1.0])

    loss = trainer.train_epoch(data, edge_label_index, edge_labels)
    assert loss > 0

    metrics = trainer.evaluate(data, edge_label_index, edge_labels)
    assert "auc" in metrics
    assert "ap" in metrics


def test_hgt_trainer_fit():
    """验证 HGTTrainer.fit 端到端训练循环."""
    from torch_geometric.data import HeteroData

    from iron_aging.models.link_predictor import LinkPredictor
    from iron_aging.training.trainer import HGTTrainer

    data = HeteroData()
    data["compound"].x = torch.randn(4, 8)
    data["gene"].x = torch.randn(5, 8)
    data["compound", "targets", "gene"].edge_index = torch.tensor([[0, 1], [0, 1]])

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.predictor = LinkPredictor(8, 16)

        def forward(self, x_dict, edge_index_dict):
            return x_dict

    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = HGTTrainer(
        model,
        optimizer,
        edge_type=("compound", "targets", "gene"),
        early_stopping_patience=5,
    )

    train_idx = torch.tensor([[0, 1], [0, 1]])
    train_labels = torch.tensor([1.0, 0.0])
    val_idx = torch.tensor([[0, 1], [1, 0]])
    val_labels = torch.tensor([1.0, 0.0])

    history = trainer.fit(data, train_idx, train_labels, val_idx, val_labels, epochs=10)

    assert 0 < len(history["train_loss"]) <= 10
    assert len(history["train_loss"]) == len(history["val_auc"])
    assert history["train_loss"][0] > history["train_loss"][-1]
