"""生成通路-通路相似性边 (Jaccard >= threshold).

输入:
    network_files/kegg_pathway_genes.csv      (KEGG REST 通路-基因)
    network_files/gene_pathway_enrichment.csv (项目富集通路-基因, KEGG/Reactome/GO等)

输出:
    network_files/pathway_pathway_similarity_edges.csv
        pathway_A, pathway_B, jaccard, intersection_size, union_size,
        source, confidence, confidence_level, download_date

方法:
    - 合并 KEGG 通路基因与项目富集通路基因, 按 pathway 名称聚合 gene 集合.
    - 两两计算 Jaccard 相似度 = |A ∩ B| / |A ∪ B|.
    - 保留 Jaccard >= threshold 且 intersection_size >= min_intersection 的无向边, 无自环.
    - confidence = jaccard; confidence_level 按 0.5/0.3 分档.
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
KEGG_PATHWAY_GENES_CSV = BASE_DIR / "network_files" / "kegg_pathway_genes.csv"
ENRICHMENT_CSV = BASE_DIR / "network_files" / "gene_pathway_enrichment.csv"
OUTPUT_CSV = BASE_DIR / "network_files" / "pathway_pathway_similarity_edges.csv"
METADATA_JSON = BASE_DIR / "external_data" / "pathway_similarity_metadata.json"

JACCARD_THRESHOLD = 0.2
MIN_INTERSECTION = 2


def load_pathway_genes(csv_path: Path) -> dict[str, set[str]]:
    """读取通路-基因映射, 返回 pathway -> gene_id 集合."""
    if not csv_path.exists():
        logger.warning("通路-基因文件不存在, 跳过: %s", csv_path)
        return {}
    df = pd.read_csv(csv_path, dtype=str)
    df = df.dropna(subset=["pathway", "gene_id"])
    df["pathway"] = df["pathway"].astype(str).str.strip()
    df["gene_id"] = df["gene_id"].astype(str).str.strip().str.upper()

    mapping: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        mapping.setdefault(row["pathway"], set()).add(row["gene_id"])
    return mapping


def load_enrichment_pathway_genes(csv_path: Path) -> dict[str, set[str]]:
    """读取富集通路-基因映射, 返回 pathway -> gene symbol 集合."""
    if not csv_path.exists():
        logger.warning("富集通路文件不存在, 跳过: %s", csv_path)
        return {}
    df = pd.read_csv(csv_path, dtype=str)
    df = df.dropna(subset=["pathway", "gene"])
    df["pathway"] = df["pathway"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()

    mapping: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        mapping.setdefault(row["pathway"], set()).add(row["gene"])
    return mapping


def merge_pathway_sources(
    kegg_mapping: dict[str, set[str]],
    enrichment_mapping: dict[str, set[str]],
) -> dict[str, set[str]]:
    """合并两个来源的通路-基因映射, 同名通路取基因并集."""
    merged: dict[str, set[str]] = {k: set(v) for k, v in kegg_mapping.items()}
    for pathway, genes in enrichment_mapping.items():
        merged.setdefault(pathway, set()).update(genes)
    return merged


def confidence_level(score: float) -> str:
    if score >= 0.50:
        return "high"
    if score >= 0.30:
        return "medium"
    return "low"


def generate_edges(
    pathway_genes: dict[str, set[str]],
    threshold: float,
    min_intersection: int,
) -> pd.DataFrame:
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
            if intersection < min_intersection:
                continue
            jaccard = intersection / union
            if jaccard < threshold:
                continue
            records.append(
                {
                    "pathway_A": name_a,
                    "pathway_B": name_b,
                    "jaccard": round(jaccard, 4),
                    "intersection_size": intersection,
                    "union_size": union,
                    "source": "KEGG_enrichment_union_Jaccard",
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
    parser.add_argument(
        "--min-intersection",
        type=int,
        default=MIN_INTERSECTION,
        help="最小共享基因数",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    kegg_mapping = load_pathway_genes(KEGG_PATHWAY_GENES_CSV)
    enrichment_mapping = load_enrichment_pathway_genes(ENRICHMENT_CSV)
    pathway_genes = merge_pathway_sources(kegg_mapping, enrichment_mapping)
    logger.info(
        "读取通路: KEGG=%d, 富集=%d, 合并后=%d",
        len(kegg_mapping),
        len(enrichment_mapping),
        len(pathway_genes),
    )

    edges = generate_edges(pathway_genes, args.threshold, args.min_intersection)
    edges["download_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)

    edges.to_csv(OUTPUT_CSV, index=False)
    if edges.empty:
        logger.warning("未生成任何 pathway-pathway 相似性边 (阈值 %.2f)", args.threshold)
    else:
        logger.info("已写入 %s: %d 条边", OUTPUT_CSV, len(edges))

    metadata = {
        "source": "KEGG_enrichment_union_Jaccard",
        "method": "Jaccard similarity between unified KEGG + enrichment pathway gene sets",
        "input_files": [str(KEGG_PATHWAY_GENES_CSV), str(ENRICHMENT_CSV)],
        "output_file": str(OUTPUT_CSV),
        "threshold": args.threshold,
        "min_intersection": args.min_intersection,
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
