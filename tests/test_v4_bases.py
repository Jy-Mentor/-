"""v4.0 分层架构基类与应用入口集成测试."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

# 确保项目根目录可被导入
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ruff: noqa: E402
import torch
from torch_geometric.data import HeteroData

from iron_aging.apps.hgt_pipeline import parse_args
from iron_aging.data.split import split_edges
from iron_aging.etl.base import DataSource, ETLResult
from iron_aging.models import HeteroLinkPredictionModel
from iron_aging.pipelines.base import Pipeline, PipelineConfig, PipelineResult
from iron_aging.pipelines.hgt_pipeline import _build_link_prediction_data


class MockDataSource(DataSource):
    """测试用数据源实现."""

    name = "mock"

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        n = params.get("n", 2)
        return [{"id": i, "value": i * 2} for i in range(n)]

    def normalize(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"gene": r["id"], "score": float(r["value"])} for r in raw]

    def validate(self, records: list[dict[str, Any]]) -> list[str]:
        errors = []
        for r in records:
            if r["score"] < 0:
                errors.append(f"负值: {r}")
        return errors


class FailingDataSource(DataSource):
    """模拟 fetch 失败的数据源."""

    name = "failing"

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        msg = params.get("error_message", "boom")
        raise RuntimeError(msg)

    def normalize(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return raw


class InvalidDataSource(DataSource):
    """模拟验证失败的数据源."""

    name = "invalid"

    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        return [{"id": 1, "value": -10}]

    def normalize(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"gene": r["id"], "score": float(r["value"])} for r in raw]

    def validate(self, records: list[dict[str, Any]]) -> list[str]:
        return ["negative score"] if any(r["score"] < 0 for r in records) else []


class MockPipeline(Pipeline):
    """测试用 Pipeline 实现."""

    name = "mock_pipeline"

    def run(self, config: PipelineConfig) -> PipelineResult:
        return PipelineResult(
            experiment_id=config.experiment_name,
            status="success",
            metrics={"seed": config.seed, "use_cache": config.use_cache},
        )


def test_etl_result_success_property() -> None:
    """ETLResult.success 应在无错误时返回 True."""
    result = ETLResult(source_name="test")
    assert result.success is True
    assert result.entity_counts == {}
    assert result.errors == []


def test_etl_result_failure_property() -> None:
    """ETLResult.success 应在有错误时返回 False."""
    result = ETLResult(source_name="test", errors=["missing column"])
    assert result.success is False


def test_datasource_run_success() -> None:
    """MockDataSource 应正确完成 fetch -> normalize -> validate."""
    source = MockDataSource()
    result = source.run(n=3)
    assert result.success is True
    assert result.entity_counts["normalized"] == 3
    assert len(result.errors) == 0


def test_datasource_run_validation_error() -> None:
    """验证失败应记录在 errors 中但不抛异常."""
    source = InvalidDataSource()
    result = source.run()
    assert result.success is False
    assert any("negative" in e for e in result.errors)


def test_datasource_run_fetch_failure() -> None:
    """fetch 异常应被捕获并记录."""
    source = FailingDataSource()
    result = source.run(error_message="network down")
    assert result.success is False
    assert "network down" in result.errors[0]


def test_pipeline_config_defaults() -> None:
    """PipelineConfig 应使用合理的默认值."""
    config = PipelineConfig(experiment_name="exp01")
    assert config.seed == 42
    assert config.use_cache is True
    assert config.model_config == {}
    assert config.training_config == {}


def test_pipeline_result_fields() -> None:
    """PipelineResult 应包含关键字段."""
    result = PipelineResult(experiment_id="exp01", status="success")
    assert result.metrics == {}
    assert result.model_path is None
    assert result.ranking_path is None


def test_mock_pipeline_run() -> None:
    """MockPipeline 应返回带正确 experiment_id 的结果."""
    config = PipelineConfig(experiment_name="test_pipeline", seed=123)
    pipeline = MockPipeline()
    result = pipeline.run(config)
    assert result.status == "success"
    assert result.experiment_id == "test_pipeline"
    assert result.metrics["seed"] == 123


def test_hgt_pipeline_argparse_defaults() -> None:
    """hgt_pipeline 默认参数解析应正确."""
    args = parse_args([])
    assert args.config == "config.yaml"
    assert args.device == "cuda"
    assert args.epochs is None
    assert args.clear_cache is False


def test_hgt_pipeline_argparse_custom() -> None:
    """hgt_pipeline 自定义参数解析应正确."""
    args = parse_args([
        "--config", "custom.yaml",
        "--device", "cpu",
        "--epochs", "10",
        "--clear-cache",
        "--encoder-type", "gat_hgt",
    ])
    assert args.config == "custom.yaml"
    assert args.device == "cpu"
    assert args.epochs == 10
    assert args.clear_cache is True
    assert args.encoder_type == "gat_hgt"


def test_hgt_pipeline_main_uses_v4_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """hgt_pipeline.main() 默认应调用 v4.0 HGTLinkPredictionPipeline."""
    fake_result = PipelineResult(
        experiment_id="fake",
        status="success",
        metrics={"best_val_auc": 0.8, "test": {"auc": 0.75}},
    )
    fake_pipeline_cls = mock.Mock()
    fake_pipeline_cls.return_value.run.return_value = fake_result
    monkeypatch.setattr(
        "iron_aging.pipelines.HGTLinkPredictionPipeline", fake_pipeline_cls
    )

    # 保留 fake module3_hgt, 避免 reload 时触发真实 module3_hgt
    fake_module = ModuleType("module3_hgt")
    fake_module.main = mock.Mock(return_value=0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "module3_hgt", fake_module)

    import importlib

    from iron_aging.apps import hgt_pipeline

    importlib.reload(hgt_pipeline)

    assert hgt_pipeline.main(["--epochs", "2"]) == 0
    fake_pipeline_cls.return_value.run.assert_called_once()


def test_hgt_pipeline_main_legacy_delegates_to_module3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hgt_pipeline.main() --legacy 应透传至 module3_hgt.main()."""
    fake_module = ModuleType("module3_hgt")
    fake_module.main = mock.Mock(return_value=0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "module3_hgt", fake_module)

    import importlib

    from iron_aging.apps import hgt_pipeline

    importlib.reload(hgt_pipeline)

    assert hgt_pipeline.main(["--legacy"]) == 0
    fake_module.main.assert_called_once()


def test_hetero_link_prediction_model_from_data() -> None:
    """HeteroLinkPredictionModel 应能从 HeteroData 自动推断维度并前向传播."""
    data = HeteroData()
    data["gene"].x = torch.randn(10, 8)
    data["compound"].x = torch.randn(5, 6)
    data["gene", "interacts", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])
    data["compound", "targets", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])

    model = HeteroLinkPredictionModel.from_hetero_data(data, hidden_dim=8, out_dim=4)
    out = model(data.x_dict, data.edge_index_dict)
    assert "gene" in out
    assert "compound" in out
    assert out["gene"].shape == (10, 4)


def test_gat_link_prediction_model_from_data() -> None:
    """HeteroLinkPredictionModel 使用 GAT 编码器时应正确前向传播."""
    data = HeteroData()
    data["gene"].x = torch.randn(10, 8)
    data["compound"].x = torch.randn(5, 6)
    data["gene", "interacts", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])
    data["compound", "targets", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])

    model = HeteroLinkPredictionModel.from_hetero_data(
        data, hidden_dim=8, out_dim=4, encoder_type="gat"
    )
    assert model.encoder_type == "gat"
    out = model(data.x_dict, data.edge_index_dict)
    assert "gene" in out
    assert "compound" in out
    assert out["gene"].shape == (10, 4)
    assert out["compound"].shape == (5, 4)


def test_rgcn_link_prediction_model_from_data() -> None:
    """HeteroLinkPredictionModel 使用 RGCN 编码器时应正确前向传播."""
    data = HeteroData()
    data["gene"].x = torch.randn(10, 8)
    data["compound"].x = torch.randn(5, 6)
    data["gene", "interacts", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])
    data["compound", "targets", "gene"].edge_index = torch.tensor([[0, 1], [1, 2]])

    model = HeteroLinkPredictionModel.from_hetero_data(
        data, hidden_dim=8, out_dim=4, encoder_type="rgcn"
    )
    assert model.encoder_type == "rgcn"
    out = model(data.x_dict, data.edge_index_dict)
    assert "gene" in out
    assert "compound" in out
    assert out["gene"].shape == (10, 4)
    assert out["compound"].shape == (5, 4)


def test_split_edges() -> None:
    """split_edges 应按比例划分边."""
    edge_index = torch.arange(100).unsqueeze(0).repeat(2, 1)
    train_ei, val_ei, test_ei = split_edges(
        edge_index, num_src=100, train_ratio=0.7, val_ratio=0.15, seed=42, mode="random"
    )
    assert train_ei.shape[1] == 70
    assert val_ei.shape[1] >= 10
    assert test_ei.shape[1] >= 10
    assert train_ei.shape[1] + val_ei.shape[1] + test_ei.shape[1] == 100


def test_split_edges_stratified() -> None:
    """split_edges stratified 模式应按源节点分层划分."""
    # 100 条边, src 为 0-99 各出现 1 次
    edge_index = torch.arange(100).unsqueeze(0).repeat(2, 1)
    train_ei, val_ei, test_ei = split_edges(
        edge_index, num_src=100, train_ratio=0.7, val_ratio=0.15, seed=42, mode="stratified"
    )
    assert train_ei.shape[1] + val_ei.shape[1] + test_ei.shape[1] == 100
    # 分层模式下每个集合都包含不同 src
    assert len(set(train_ei[0].tolist())) > 0
    assert len(set(val_ei[0].tolist())) > 0
    assert len(set(test_ei[0].tolist())) > 0


def test_build_link_prediction_data() -> None:
    """_build_link_prediction_data 应生成训练图并移除所有目标正样本边."""
    data = HeteroData()
    data["gene"].x = torch.randn(10, 4)
    data["pathway"].x = torch.randn(5, 4)
    data["gene"].num_nodes = 10
    data["pathway"].num_nodes = 5
    # 10 条唯一正样本边
    src = torch.arange(10)
    dst = torch.arange(10) % 5
    data["gene", "belongs_to", "pathway"].edge_index = torch.stack([src, dst])

    train_data, splits, num_src, num_dst = _build_link_prediction_data(
        data,
        ("gene", "belongs_to", "pathway"),
        train_ratio=0.5,
        val_ratio=0.25,
        seed=42,
    )
    train_ei = train_data["gene", "belongs_to", "pathway"].edge_index
    # SpotTarget: 训练图移除全部目标边 (train/val/test)
    assert train_ei.shape[1] == 0
    assert len(splits["train_pos_edges"]) == 5
    assert len(splits["val_pos_edges"]) == 2
    assert len(splits["test_pos_edges"]) == 3
    assert num_src == 10
    assert num_dst == 5
