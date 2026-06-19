"""项目配置加载模块.

所有超参数、路径统一从项目根目录的 config.yaml 加载,
支持通过环境变量覆盖关键路径.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from iron_aging import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """加载 YAML 配置文件.

    Args:
        path: 配置文件路径, 默认使用项目根目录 config.yaml.

    Returns:
        解析后的配置字典.
    """
    path = Path(path) if path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}
    return config


def get_path_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回路径配置, 支持环境变量覆盖."""
    config = config or load_config()
    paths = config.get("paths", {})

    # 关键路径允许通过环境变量覆盖
    env_overrides = {
        "geo_dir": os.environ.get("IRON_GEO_DIR"),
        "l1_results_dir": os.environ.get("IRON_L1_DIR"),
    }
    for key, value in env_overrides.items():
        if value:
            paths[key] = value
    return paths


def get_model_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回模型架构配置."""
    config = config or load_config()
    return config.get("model", {})


def get_training_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """返回训练配置."""
    config = config or load_config()
    return config.get("training", {})
