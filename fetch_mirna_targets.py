"""从 miRTarBase 下载并构建 miRNA-靶基因边.

数据来源:
    miRTarBase 10.0 (curated experimentally validated microRNA-target interactions)
    https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/cache/download/10.0/hsa_MTI.csv

输出:
    network_files/mirna_target_edges.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iron_aging import NETWORK_DIR, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MIRTARBASE_URL = "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/cache/download/10.0/miRTarBase_SE_WR.csv"
DEFAULT_GENE_FILE = PROJECT_ROOT / "network_files" / "string_ppi_edges.csv"


def load_project_genes(path: Path) -> set[str]:
    """从 PPI 边文件加载项目基因符号集合."""
    df = pd.read_csv(path, dtype=str)
    genes: set[str] = set()
    for col in ("protein_A", "protein_B"):
        if col in df.columns:
            genes.update(df[col].dropna().str.strip().str.upper().unique())
    logger.info("从 %s 加载项目基因: %d 个", path, len(genes))
    return genes


def download_mirtarbase(output_dir: Path, force: bool = False) -> Path:
    """下载 miRTarBase 强证据子集 CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "miRTarBase_SE_WR.csv"
    if output_path.exists() and not force:
        logger.info("使用已存在文件: %s", output_path)
        return output_path

    logger.info("下载 miRTarBase: %s", MIRTARBASE_URL)
    try:
        with requests.get(MIRTARBASE_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info("下载完成: %s (%.2f MB)", output_path, output_path.stat().st_size / 1e6)
    except Exception:
        logger.error("下载 miRTarBase 失败")
        traceback.print_exc()
        raise
    return output_path


def assign_confidence(support_type: str | None) -> tuple[float, str]:
    """根据 miRTarBase support type 分配置信度."""
    if not support_type:
        return 0.5, "unknown"
    st = str(support_type).strip().lower()
    if "functional" in st:
        return 0.9, "high"
    if "non-functional" in st:
        return 0.4, "low"
    if "weak" in st:
        return 0.5, "medium"
    if "strong" in st:
        return 0.85, "high"
    return 0.6, "medium"


def process_mirtarbase(
    csv_path: Path,
    project_genes: set[str],
) -> pd.DataFrame:
    """读取 miRTarBase 强证据子集并过滤人类/项目基因."""
    logger.info("读取 miRTarBase: %s", csv_path)
    df = pd.read_csv(csv_path, dtype=str)
    logger.info("miRTarBase 原始记录数: %d", len(df))

    required = {"miRNA", "Target Gene", "Species (miRNA)"}
    missing = required - set(df.columns)
    if missing:
        msg = f"缺少必要列: {missing}, 实际列: {list(df.columns)}"
        raise ValueError(msg)

    # 仅保留人类 miRNA
    df = df[df["Species (miRNA)"].str.strip().str.lower() == "hsa"]
    logger.info("人类 miRNA 记录数: %d", len(df))

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        mirna = str(row.get("miRNA", "")).strip()
        gene = str(row.get("Target Gene", "")).strip().upper()
        if not mirna or not gene:
            continue
        if gene not in project_genes:
            continue
        key = (mirna, gene)
        if key in seen:
            continue
        seen.add(key)

        support_type = str(row.get("Support Type", "")).strip() or None
        confidence, level = assign_confidence(support_type)
        records.append({
            "mirna": mirna,
            "gene": gene,
            "support_type": support_type,
            "source": "miRTarBase_10.0_SE_WR",
            "score": confidence,
            "confidence": confidence,
            "confidence_level": level,
            "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        })

    result = pd.DataFrame(records)
    if result.empty:
        logger.warning("未找到项目基因相关的 miRNA 靶基因记录")
        result = pd.DataFrame(columns=[
            "mirna", "gene", "support_type", "source", "score",
            "confidence", "confidence_level", "download_date",
        ])
    else:
        result = result.sort_values(["mirna", "gene"]).reset_index(drop=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 miRTarBase 构建 miRNA-靶基因边")
    parser.add_argument(
        "--gene-file",
        type=str,
        default=str(DEFAULT_GENE_FILE),
        help="项目基因列表文件 (CSV, 包含 protein_A/protein_B 列)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(NETWORK_DIR),
        help="输出目录",
    )
    parser.add_argument(
        "--external-dir",
        type=str,
        default=str(PROJECT_ROOT / "external_data" / "mirnatarbase"),
        help="外部数据下载目录",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="强制重新下载",
    )
    args = parser.parse_args(argv)

    try:
        project_genes = load_project_genes(Path(args.gene_file))
        csv_path = download_mirtarbase(Path(args.external_dir), force=args.force_download)
        df = process_mirtarbase(csv_path, project_genes)
        output_path = Path(args.output_dir) / "mirna_target_edges.csv"
        df.to_csv(output_path, index=False)
        logger.info("输出 miRNA 靶基因边: %s, 共 %d 条", output_path, len(df))
        return 0
    except Exception:
        logger.error("构建 miRNA 靶基因边失败")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
