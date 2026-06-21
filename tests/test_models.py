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


def test_rgcn_edge_type_mapping_stable(device):
    """回归测试: RGCN 关系 ID 固定由 metadata 顺序决定, 不受字典迭代顺序影响."""
    from torch_geometric.data import HeteroData

    from iron_aging.models.hetero_link_prediction import HeteroLinkPredictionModel

    metadata = (
        ["gene", "compound"],
        [
            ("gene", "interacts", "gene"),
            ("compound", "targets", "gene"),
        ],
    )
    model = HeteroLinkPredictionModel(
        metadata=metadata,
        in_dims={"gene": 4, "compound": 4},
        hidden_dim=8,
        out_dim=4,
        encoder_type="rgcn",
        num_nodes_dict={"gene": 3, "compound": 2},
        num_layers=1,
        dropout=0.0,
    ).to(device)

    edge_a = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device)
    edge_b = torch.tensor([[0], [0]], dtype=torch.long, device=device)

    # 顺序 1: A 在前, B 在后
    _, edge_type_1 = model._to_homogeneous_edge_index(
        {("gene", "interacts", "gene"): edge_a, ("compound", "targets", "gene"): edge_b}
    )
    # 顺序 2: B 在前, A 在后
    _, edge_type_2 = model._to_homogeneous_edge_index(
        {("compound", "targets", "gene"): edge_b, ("gene", "interacts", "gene"): edge_a}
    )

    # metadata 中 A 的 rel_id=0, B 的 rel_id=1
    assert edge_type_1[:2].tolist() == [0, 0]
    assert edge_type_1[2:].tolist() == [1]
    assert edge_type_2[:1].tolist() == [1]
    assert edge_type_2[1:].tolist() == [0, 0]

    # 完整前向传播不应报错, 且输出节点嵌入形状正确
    data = HeteroData()
    data["gene"].x = torch.randn(3, 4, device=device)
    data["compound"].x = torch.randn(2, 4, device=device)
    data["gene", "interacts", "gene"].edge_index = edge_a
    data["compound", "targets", "gene"].edge_index = edge_b

    out = model(data.x_dict, data.edge_index_dict)
    assert out["gene"].shape == (3, 4)
    assert out["compound"].shape == (2, 4)


def test_gat_hgt_encoder_forward(device):
    """验证 GAT-HGT 融合编码器前向传播与输出形状."""
    from torch_geometric.data import HeteroData

    from iron_aging.models.hetero_link_prediction import HeteroLinkPredictionModel

    metadata = (
        ["gene", "compound"],
        [
            ("gene", "interacts", "gene"),
            ("compound", "targets", "gene"),
        ],
    )
    model = HeteroLinkPredictionModel(
        metadata=metadata,
        in_dims={"gene": 4, "compound": 4},
        hidden_dim=8,
        out_dim=4,
        encoder_type="gat_hgt",
        num_nodes_dict={"gene": 3, "compound": 2},
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    ).to(device)

    data = HeteroData()
    data["gene"].x = torch.randn(3, 4, device=device)
    data["compound"].x = torch.randn(2, 4, device=device)
    data["gene", "interacts", "gene"].edge_index = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.long, device=device
    )
    data["compound", "targets", "gene"].edge_index = torch.tensor(
        [[0], [0]], dtype=torch.long, device=device
    )

    out = model(data.x_dict, data.edge_index_dict)
    assert out["gene"].shape == (3, 4)
    assert out["compound"].shape == (2, 4)
