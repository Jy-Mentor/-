"""数据抽取层 (ETL) 抽象与具体实现."""

from iron_aging.etl.base import DataSource, ETLResult

__all__ = ["DataSource", "ETLResult"]
