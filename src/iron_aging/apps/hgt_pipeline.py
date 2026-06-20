"""GAT-HGT 异构图训练与推理应用入口.

本模块为升级后的统一应用层入口, 第一阶段兼容调用 legacy module3_hgt.main(),
后续逐步替换为新模块化 pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保 src 与项目根目录可被导入
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (_SRC_DIR, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# ruff: noqa: E402
from iron_aging.utils.warnings import suppress_known_library_warnings

suppress_known_library_warnings()

import module3_hgt
from iron_aging.config import load_config
from iron_aging.data.graph_builder import clear_graph_cache

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description="运行 GAT-HGT 训练与评估流程")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="训练设备: cuda 或 cpu (默认: cuda)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数, 未指定则使用配置文件",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="清除图数据缓存后运行",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """主入口函数."""
    args = parse_args(argv)
    config = load_config(args.config)
    hidden_dim = config.get("model", {}).get("hidden_dim", 64)
    logger.info(f"配置加载完成, hidden_dim={hidden_dim}")

    if args.clear_cache:
        clear_graph_cache()
        logger.info("已清除图数据缓存")

    # 第一阶段: 透传至 legacy module3_hgt.main()
    # 后续将逐步替换为新模块化 pipeline
    return module3_hgt.main()


if __name__ == "__main__":
    raise SystemExit(main())
