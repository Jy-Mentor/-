"""测试模型模块前向传播."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iron_aging.models.gat_encoder import GATEncoder  # noqa: E402
from iron_aging.models.hgt_encoder import HGTEncoder  # noqa: E402
from iron_aging.models.link_predictor import LinkPredictor  # noqa: E402


@pytest.fixture
def device():
    return torch.device("cpu")


def test_gat_encoder_forward(device):
    """验证 GAT 编码器前向传播."""
    encoder = GATEncoder(in_dim=8, hidden_dim=4, out_dim=2, heads=2, dropout=0.0)
    encoder = encoder.to(device)

    x = torch.randn(10, 8, device=device)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long, device=device)

    out = encoder(x, edge_index)
    assert out.shape == (10, 2)


def test_hgt_encoder_forward(device):
    """验证 HGT 编码器前向传播."""
    metadata = (["gene", "compound"], [("gene", "interacts", "gene"), ("compound", "targets", "gene")])
    encoder = HGTEncoder(hidden_dim=4, out_dim=2, metadata=metadata, num_heads=2, num_layers=1, dropout=0.0)
    encoder = encoder.to(device)

    x_dict = {
        "gene": torch.randn(5, 4, device=device),
        "compound": torch.randn(3, 4, device=device),
    }
    edge_index_dict = {
        ("gene", "interacts", "gene"): torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device),
        ("compound", "targets", "gene"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=device),
    }

    out = encoder(x_dict, edge_index_dict)
    assert out["gene"].shape == (5, 2)
    assert out["compound"].shape == (3, 2)


def test_link_predictor_forward(device):
    """验证链路预测头前向传播."""
    predictor = LinkPredictor(in_dim=4, hidden_dim=8, dropout=0.0)
    predictor = predictor.to(device)

    src_emb = torch.randn(10, 4, device=device)
    dst_emb = torch.randn(10, 4, device=device)

    out = predictor(src_emb, dst_emb)
    assert out.shape == (10, 1)
