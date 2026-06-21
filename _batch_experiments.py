"""批量运行链路预测实验：多 seed 稳定性 + 多编码器对比.

结果写入 L3_results/_batch_summary.json。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iron_aging.config import load_config
from iron_aging.pipelines import HGTLinkPredictionPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(name: str, encoder: str, seed: int, epochs: int = 50) -> dict:
    """运行单个实验并返回摘要."""
    config = load_config("config.yaml")
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    model_config["encoder_type"] = encoder
    training_config["epochs"] = epochs

    pipeline_config = PipelineConfig(
        experiment_name=name,
        model_config=model_config,
        training_config=training_config,
        use_cache=True,
        seed=seed,
    )

    pipeline = HGTLinkPredictionPipeline()
    result = pipeline.run(pipeline_config)

    return {
        "experiment": name,
        "encoder": encoder,
        "seed": seed,
        "epochs": epochs,
        "status": result.status,
        "best_val_auc": result.metrics.get("best_val_auc"),
        "test_auc": result.metrics.get("test", {}).get("auc"),
        "test_ap": result.metrics.get("test", {}).get("ap"),
        "model_path": str(result.model_path) if result.model_path else None,
    }


def main() -> None:
    summary: list[dict] = []

    # 1) 多 seed RGCN 稳定性
    seeds = [42, 2024, 12345]
    for seed in seeds:
        name = f"rgcn_seed{seed}"
        logger.info("=" * 60)
        logger.info("运行 %s", name)
        summary.append(run_experiment(name, "rgcn", seed))

    # 2) 同 seed 下 HGT/GAT/RGCN 对比
    encoders = ["hgt", "gat", "rgcn"]
    common_seed = 42
    for encoder in encoders:
        name = f"{encoder}_compare_seed{common_seed}"
        logger.info("=" * 60)
        logger.info("运行 %s", name)
        summary.append(run_experiment(name, encoder, common_seed))

    output_path = Path("L3_results") / "_batch_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("批量实验摘要已保存: %s", output_path)

    # 控制台打印表格
    print("\n" + "=" * 80)
    print(f"{'Experiment':<30} {'Encoder':<8} {'Seed':<8} {'Val AUC':<10} {'Test AUC':<10} {'Test AP':<10}")
    print("-" * 80)
    for r in summary:
        print(
            f"{r['experiment']:<30} {r['encoder']:<8} {r['seed']:<8} "
            f"{r['best_val_auc']:<10.4f} {r['test_auc']:<10.4f} {r['test_ap']:<10.4f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
