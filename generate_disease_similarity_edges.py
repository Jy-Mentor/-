"""基于疾病-基因关联的 Jaccard 相似性生成 disease-similar-disease 边.

输入:
    network_files/disease_gene_associations.csv

输出:
    network_files/disease_disease_similarity_edges.csv

规则:
1. 仅对疾病节点配置中存在的疾病生成边.
2. 使用两疾病共享基因数 / 并集基因数计算 Jaccard 相似性.
3. 阈值 >= 0.1 且至少共享 2 个基因.
4. 记录 source/confidence/download_date.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Set

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DISEASE_GENE_CSV = BASE_DIR / "network_files" / "disease_gene_associations.csv"
NODE_CONFIG = BASE_DIR / "network_files" / "graph_node_config.yaml"
OUTPUT_CSV = BASE_DIR / "network_files" / "disease_disease_similarity_edges.csv"


def load_disease_nodes(config_path: Path) -> set[str]:
    """从 graph_node_config.yaml 加载疾病节点列表."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    diseases = cfg.get("diseases", [])
    return {str(d).strip() for d in diseases if str(d).strip()}


def load_disease_genes(path: Path) -> dict[str, Set[str]]:
    """读取 disease -> set(gene) 映射."""
    df = pd.read_csv(path)
    mapping: dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        disease = str(row.get("disease", "")).strip()
        gene = str(row.get("gene", "")).strip().upper()
        if not disease or not gene:
            continue
        mapping.setdefault(disease, set()).add(gene)
    return mapping


def jaccard(a: Set[str], b: Set[str]) -> float:
    """计算 Jaccard 相似性."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def generate_edges(
    disease_genes: dict[str, Set[str]],
    disease_nodes: set[str],
    min_jaccard: float = 0.1,
    min_shared: int = 2,
) -> pd.DataFrame:
    """生成疾病-疾病相似性边."""
    records = []
    diseases = sorted(disease_nodes & set(disease_genes.keys()))
    for i, d1 in enumerate(diseases):
        for d2 in diseases[i + 1 :]:
            genes1 = disease_genes.get(d1, set())
            genes2 = disease_genes.get(d2, set())
            shared = genes1 & genes2
            if len(shared) < min_shared:
                continue
            score = jaccard(genes1, genes2)
            if score < min_jaccard:
                continue
            records.append(
                {
                    "disease_a": d1,
                    "disease_b": d2,
                    "score": round(score, 4),
                    "shared_genes": len(shared),
                    "source": "disease_gene_associations",
                    "confidence": round(score, 4),
                    "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                }
            )
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "disease_a",
                "disease_b",
                "score",
                "shared_genes",
                "source",
                "confidence",
                "download_date",
            ]
        )
    return df.sort_values(["disease_a", "disease_b"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 disease-disease 相似性边")
    parser.add_argument("--input", type=Path, default=DISEASE_GENE_CSV)
    parser.add_argument("--config", type=Path, default=NODE_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--min-jaccard", type=float, default=0.1)
    parser.add_argument("--min-shared", type=int, default=2)
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("输入文件不存在: %s", args.input)
        return 1
    if not args.config.exists():
        logger.error("节点配置文件不存在: %s", args.config)
        return 1

    disease_nodes = load_disease_nodes(args.config)
    disease_genes = load_disease_genes(args.input)
    logger.info("配置疾病节点: %d, 有基因关联的疾病: %d", len(disease_nodes), len(disease_genes))

    df = generate_edges(
        disease_genes,
        disease_nodes,
        min_jaccard=args.min_jaccard,
        min_shared=args.min_shared,
    )
    df.to_csv(args.output, index=False)
    logger.info("已写入 %d 条 disease-disease 相似性边: %s", len(df), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
