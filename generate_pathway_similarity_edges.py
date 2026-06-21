"""生成通路-通路相似性边 (Jaccard >= 0.2).

输入:
    network_files/kegg_pathway_genes.csv

输出:
    network_files/pathway_pathway_similarity_edges.csv
        pathway_A, pathway_B, jaccard, intersection_size, union_size,
        source, confidence, confidence_level, download_date

方法:
    - 按 pathway 聚合 gene_id 集合.
    - 两两计算 Jaccard 相似度 = |A ∩ B| / |A ∪ B|.
    - 保留 Jaccard >= 0.2 的无向边, 无自环.
    - confidence = jaccard.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "network_files" / "kegg_pathway_genes.csv"
OUTPUT_CSV = BASE_DIR / "network_files" / "pathway_pathway_similarity_edges.csv"
METADATA_JSON = BASE_DIR / "external_data" / "pathway_similarity_metadata.json"

JACCARD_THRESHOLD = 0.2


def load_pathway_genes(csv_path: Path) -> dict[str, set[str]]:
    """读取通路-基因映射, 返回 pathway -> gene_id 集合."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["pathway", "gene_id"])
    df["pathway"] = df["pathway"].astype(str).str.strip()
    df["gene_id"] = df["gene_id"].astype(str).str.strip()

    mapping: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        mapping.setdefault(row["pathway"], set()).add(row["gene_id"])
    return mapping


def confidence_level(score: float) -> str:
    if score >= 0.50:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def generate_edges(pathway_genes: dict[str, set[str]], threshold: float) -> pd.DataFrame:
    """生成 pathway-pathway Jaccard 相似性边."""
    pathways = sorted(pathway_genes.keys())
    records = []

    for i in range(len(pathways)):
        name_a = pathways[i]
        genes_a = pathway_genes[name_a]
        for j in range(i + 1, len(pathways)):
            name_b = pathways[j]
            genes_b = pathway_genes[name_b]
            intersection = len(genes_a & genes_b)
            union = len(genes_a | genes_b)
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard >= threshold:
                records.append(
                    {
                        "pathway_A": name_a,
                        "pathway_B": name_b,
                        "jaccard": round(jaccard, 4),
                        "intersection_size": intersection,
                        "union_size": union,
                        "source": "KEGG_pathway_Jaccard",
                        "confidence": round(jaccard, 4),
                        "confidence_level": confidence_level(jaccard),
                    }
                )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("jaccard", ascending=False).reset_index(drop=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate pathway-pathway similarity edges")
    parser.add_argument(
        "--threshold",
        type=float,
        default=JACCARD_THRESHOLD,
        help="Jaccard similarity threshold",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not INPUT_CSV.exists():
        logger.error("输入文件不存在: %s", INPUT_CSV)
        return 1

    pathway_genes = load_pathway_genes(INPUT_CSV)
    logger.info("读取通路: %d 个", len(pathway_genes))

    edges = generate_edges(pathway_genes, args.threshold)
    edges["download_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)

    edges.to_csv(OUTPUT_CSV, index=False)
    if edges.empty:
        logger.warning("未生成任何 pathway-pathway 相似性边 (阈值 %.2f)", args.threshold)
    else:
        logger.info("已写入 %s: %d 条边", OUTPUT_CSV, len(edges))

    metadata = {
        "source": "KEGG_pathway_Jaccard",
        "method": "Jaccard similarity between KEGG pathway gene sets",
        "input_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "threshold": args.threshold,
        "n_pathways": len(pathway_genes),
        "n_edges": len(edges),
        "download_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_distribution": edges["confidence_level"].value_counts().to_dict() if not edges.empty else {},
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已写入元数据: %s", METADATA_JSON)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
