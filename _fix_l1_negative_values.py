"""修复 L1 表达数据中的负值.

表达量不应为负. 本脚本将 L1_genome_wide_de.csv 和 L1_gene_level_analysis.csv
中的 mean_case/mean_control 负值 clipped 到 0, 并记录警告.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent


def fix_csv(path: Path) -> None:
    if not path.exists():
        logger.warning(f"文件不存在: {path}")
        return
    df = pd.read_csv(path)
    if "mean_case" not in df.columns or "mean_control" not in df.columns:
        logger.warning(f"缺少 mean_case/mean_control 列: {path}")
        return

    n_neg_case = (df["mean_case"] < 0).sum()
    n_neg_control = (df["mean_control"] < 0).sum()
    if n_neg_case == 0 and n_neg_control == 0:
        logger.info(f"{path.name}: 无负值")
        return

    logger.warning(
        f"{path.name}: 发现负值, mean_case={n_neg_case}, mean_control={n_neg_control}, "
        f"将 clipped 到 0"
    )
    df["mean_case"] = df["mean_case"].clip(lower=0)
    df["mean_control"] = df["mean_control"].clip(lower=0)
    df.to_csv(path, index=False)
    logger.info(f"已修复并保存: {path}")


if __name__ == "__main__":
    fix_csv(ROOT / "L1" / "l1_results" / "L1_genome_wide_de.csv")
    fix_csv(ROOT / "L1" / "l1_results" / "L1_gene_level_analysis.csv")
    fix_csv(ROOT / "L3" / "L1_genome_wide_de.csv")
    fix_csv(ROOT / "L3" / "L1_gene_level_analysis.csv")
