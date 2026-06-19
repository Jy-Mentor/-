"""Pipeline 抽象基类与通用数据结构."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Pipeline 配置."""

    experiment_name: str
    model_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    use_cache: bool = True
    seed: int = 42


@dataclass
class PipelineResult:
    """Pipeline 运行结果."""

    experiment_id: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    model_path: Path | None = None
    ranking_path: Path | None = None


class Pipeline(ABC):
    """Pipeline 抽象基类."""

    name: str = "abstract"

    @abstractmethod
    def run(self, config: PipelineConfig) -> PipelineResult:
        """执行 Pipeline 并返回结果."""
