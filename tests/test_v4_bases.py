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
from iron_aging.apps.hgt_pipeline import parse_args
from iron_aging.etl.base import DataSource, ETLResult
from iron_aging.pipelines.base import Pipeline, PipelineConfig, PipelineResult


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
    args = parse_args(["--config", "custom.yaml", "--device", "cpu", "--epochs", "10", "--clear-cache"])
    assert args.config == "custom.yaml"
    assert args.device == "cpu"
    assert args.epochs == 10
    assert args.clear_cache is True


def test_hgt_pipeline_main_delegates_to_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """hgt_pipeline.main() 第一阶段应透传至 module3_hgt.main()."""
    # 创建一个伪 module3_hgt 模块, 避免触发实际训练
    fake_module = ModuleType("module3_hgt")
    fake_module.main = mock.Mock(return_value=0)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "module3_hgt", fake_module)

    # reload hgt_pipeline 使其使用被 mock 的 module3_hgt
    import importlib

    from iron_aging.apps import hgt_pipeline

    importlib.reload(hgt_pipeline)

    assert hgt_pipeline.main(["--device", "cpu"]) == 0
    fake_module.main.assert_called_once()
