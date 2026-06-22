"""ETL Source 抽象基类与结果结构.

所有外部数据源(ChEMBL, STRING, DisGeNET, OpenTargets, PubChem, GEO 等)
必须实现 DataSource 接口, 并通过 ETLRunner 统一调度.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ETLResult:
    """单次 ETL 运行结果."""

    source_name: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    cached: bool = False

    @property
    def success(self) -> bool:
        return not self.errors


class DataSource(ABC):
    """外部数据源抽象基类."""

    name: str = "abstract"

    @abstractmethod
    def fetch(self, **params: Any) -> list[dict[str, Any]]:
        """从外部源获取原始记录."""

    @abstractmethod
    def normalize(self, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将原始记录标准化为项目内部字段."""

    def validate(self, records: list[dict[str, Any]]) -> list[str]:
        """验证记录, 返回错误信息列表. 默认不报错."""
        return []

    def run(self, **params: Any) -> ETLResult:
        """执行 fetch -> normalize -> validate 并返回结果."""
        result = ETLResult(source_name=self.name)
        try:
            raw = self.fetch(**params)
            normalized = self.normalize(raw)
            errors = self.validate(normalized)
            if errors:
                result.errors.extend(errors)
            result.entity_counts["normalized"] = len(normalized)
        except Exception as e:  # noqa: BLE001 (ETL 编排层需聚合子源错误, 不中断整体流程)
            logger.exception("%s ETL 失败", self.name)
            result.errors.append(f"{self.name} ETL 失败: {e}")
        return result
