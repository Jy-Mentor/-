"""基于多源数据生成 disease-similar-disease 边.

输入:
    network_files/disease_gene_associations.csv   (项目 curated 疾病-基因)
    network_files/disgenet_disease_genes.csv      (DisGeNET curated 疾病-基因)
    network_files/ctd_compound_disease.csv        (CTD 化合物-疾病直接证据)

输出:
    network_files/disease_disease_similarity_edges.csv

规则:
1. 仅对疾病节点配置中存在的疾病生成边.
2. 多源相似性信号:
   - curated 疾病-基因 Jaccard
   - DisGeNET 疾病-基因 Jaccard (可选 min_disgenet_score 过滤)
   - CTD 化合物-疾病 Jaccard (共享治疗/标志物化合物)
3. 综合 score = 可用非零源的平均值; confidence = score.
4. 阈值 >= 0.1 且至少在一个来源中共享 >= 2 个基因/化合物.
5. 记录 sources/shared_genes/shared_compounds/confidence_level/download_date.
"""

from __future__ import annotations

import argparse
import logging
import traceback
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
NODE_CONFIG = BASE_DIR / "network_files" / "graph_node_config.yaml"
OUTPUT_CSV = BASE_DIR / "network_files" / "disease_disease_similarity_edges.csv"

CURATED_DISEASE_GENE_CSV = BASE_DIR / "network_files" / "disease_gene_associations.csv"
DISGENET_DISEASE_GENE_CSV = BASE_DIR / "network_files" / "disgenet_disease_genes.csv"
CTD_COMPOUND_DISEASE_CSV = BASE_DIR / "network_files" / "ctd_compound_disease.csv"


def load_disease_nodes(config_path: Path) -> set[str]:
    """从 graph_node_config.yaml 加载疾病节点列表."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    diseases = cfg.get("diseases", [])
    return {str(d).strip() for d in diseases if str(d).strip()}


def load_disease_genes(path: Path, min_score: float | None = None) -> dict[str, Set[str]]:
    """读取 disease -> set(gene) 映射, 可选按 score 过滤."""
    if not path.exists():
        logger.warning("疾病-基因文件不存在, 跳过: %s", path)
        return {}
    df = pd.read_csv(path, dtype=str)
    mapping: dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        disease = str(row.get("disease", "")).strip()
        gene = str(row.get("gene", "")).strip().upper()
        if not disease or not gene:
            continue
        if min_score is not None and "score" in df.columns:
            try:
                score = float(row.get("score", "0"))
            except (TypeError, ValueError):
                continue
            if score < min_score:
                continue
        mapping.setdefault(disease, set()).add(gene)
    return mapping


def load_disease_compounds(path: Path) -> dict[str, Set[str]]:
    """读取 disease -> set(compound) 映射."""
    if not path.exists():
        logger.warning("化合物-疾病文件不存在, 跳过: %s", path)
        return {}
    df = pd.read_csv(path, dtype=str)
    mapping: dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        disease = str(row.get("disease", "")).strip()
        compound = str(row.get("compound", "")).strip()
        if not disease or not compound:
            continue
        mapping.setdefault(disease, set()).add(compound)
    return mapping


def jaccard(a: Set[str], b: Set[str]) -> float:
    """计算 Jaccard 相似性."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def confidence_level(score: float) -> str:
    """根据综合得分分配置信度等级."""
    if score >= 0.50:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def generate_edges(
    curated_disease_genes: dict[str, Set[str]],
    disgenet_disease_genes: dict[str, Set[str]],
    disease_compounds: dict[str, Set[str]],
    disease_nodes: set[str],
    min_jaccard: float = 0.1,
    min_shared: int = 2,
) -> pd.DataFrame:
    """生成疾病-疾病相似性边, 综合 curated/DisGeNET 基因与 CTD 化合物信号."""
    records = []
    all_diseases = sorted(
        disease_nodes
        & (
            set(curated_disease_genes.keys())
            | set(disgenet_disease_genes.keys())
            | set(disease_compounds.keys())
        )
    )

    for i, d1 in enumerate(all_diseases):
        for d2 in all_diseases[i + 1 :]:
            sources: list[str] = []
            scores: list[float] = []

            curated_genes1 = curated_disease_genes.get(d1, set())
            curated_genes2 = curated_disease_genes.get(d2, set())
            shared_curated = curated_genes1 & curated_genes2
            if len(shared_curated) >= min_shared:
                score = jaccard(curated_genes1, curated_genes2)
                if score > 0:
                    sources.append("curated_disease_genes")
                    scores.append(score)

            disgenet_genes1 = disgenet_disease_genes.get(d1, set())
            disgenet_genes2 = disgenet_disease_genes.get(d2, set())
            shared_disgenet = disgenet_genes1 & disgenet_genes2
            if len(shared_disgenet) >= min_shared:
                score = jaccard(disgenet_genes1, disgenet_genes2)
                if score > 0:
                    sources.append("DisGeNET_disease_genes")
                    scores.append(score)

            compounds1 = disease_compounds.get(d1, set())
            compounds2 = disease_compounds.get(d2, set())
            shared_compounds = compounds1 & compounds2
            if len(shared_compounds) >= min_shared:
                score = jaccard(compounds1, compounds2)
                if score > 0:
                    sources.append("CTD_compound_disease")
                    scores.append(score)

            if not scores:
                continue

            combined_score = sum(scores) / len(scores)
            if combined_score < min_jaccard:
                continue

            records.append(
                {
                    "disease_a": d1,
                    "disease_b": d2,
                    "score": round(combined_score, 4),
                    "shared_genes": len(shared_curated | shared_disgenet),
                    "shared_compounds": len(shared_compounds),
                    "sources": ";".join(sources),
                    "confidence": round(combined_score, 4),
                    "confidence_level": confidence_level(combined_score),
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
                "shared_compounds",
                "sources",
                "confidence",
                "confidence_level",
                "download_date",
            ]
        )
    return df.sort_values(["disease_a", "disease_b"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 disease-disease 相似性边")
    parser.add_argument("--config", type=Path, default=NODE_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--curated-input", type=Path, default=CURATED_DISEASE_GENE_CSV)
    parser.add_argument("--disgenet-input", type=Path, default=DISGENET_DISEASE_GENE_CSV)
    parser.add_argument("--ctd-input", type=Path, default=CTD_COMPOUND_DISEASE_CSV)
    parser.add_argument("--min-jaccard", type=float, default=0.1)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--min-disgenet-score", type=float, default=0.21)
    args = parser.parse_args(argv)

    if not args.config.exists():
        logger.error("节点配置文件不存在: %s", args.config)
        return 1

    disease_nodes = load_disease_nodes(args.config)
    curated_disease_genes = load_disease_genes(args.curated_input)
    disgenet_disease_genes = load_disease_genes(args.disgenet_input, min_score=args.min_disgenet_score)
    disease_compounds = load_disease_compounds(args.ctd_input)

    logger.info(
        "配置疾病节点: %d, curated基因源: %d, DisGeNET基因源: %d, CTD化合物源: %d",
        len(disease_nodes),
        len(curated_disease_genes),
        len(disgenet_disease_genes),
        len(disease_compounds),
    )

    df = generate_edges(
        curated_disease_genes,
        disgenet_disease_genes,
        disease_compounds,
        disease_nodes,
        min_jaccard=args.min_jaccard,
        min_shared=args.min_shared,
    )
    df.to_csv(args.output, index=False)
    logger.info("已写入 %d 条 disease-disease 相似性边: %s", len(df), args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
