"""GAT-HGT 异构图训练与推理应用入口.

本模块为 v4.0 统一应用层入口. 默认使用新模块化 pipeline,
保留 --legacy 标志兼容调用 legacy module3_hgt.main().
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
from iron_aging.pipelines import HGTLinkPredictionPipeline, PipelineConfig

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
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="使用 legacy module3_hgt.main() 而非 v4.0 pipeline",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="hgt_link_prediction",
        help="实验名称",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default=None,
        help="编码器类型: hgt | gat | rgcn | gat_hgt (默认使用配置文件)",
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

    if args.legacy:
        logger.info("使用 legacy module3_hgt.main()")
        return module3_hgt.main()

    # v4.0 pipeline
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    if args.encoder_type is not None:
        model_config["encoder_type"] = args.encoder_type

    pipeline_config = PipelineConfig(
        experiment_name=args.experiment_name,
        model_config=model_config,
        training_config=training_config,
        use_cache=True,
        seed=training_config.get("seed", 42),
    )
    pipeline = HGTLinkPredictionPipeline()
    result = pipeline.run(pipeline_config)
    if result.status != "success":
        logger.error("Pipeline 运行失败: %s", result.errors)
        return 1
    logger.info(
        "Pipeline 运行成功, 最佳 val_auc=%.4f, test_auc=%.4f",
        result.metrics.get("best_val_auc", 0.0),
        result.metrics.get("test", {}).get("auc", 0.0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
