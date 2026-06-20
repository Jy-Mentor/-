"""清洗核心网络边文件.

对 _audit_network_edges.py 审计出的核心边文件进行标准化清洗:
- 基因名标准化(大写、去空格)
- 过滤到 98 核心铁衰老基因集
- 去重
- 去除非允许自环
- 补充 source/confidence/download_date 列

输出: 直接覆盖 network_files/ 下核心边文件, 原始文件备份到 .bak
"""

from __future__ import annotations

import logging
import shutil
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
GENE_LIST_PATH = PROJECT_ROOT / "铁衰老基因.txt"
DOWNLOAD_DATE = datetime.now().strftime("%Y-%m-%d")

# 允许自环的边类型(同源二聚体/同源互作)
SELF_LOOP_ALLOWED = {"gene_coexp_edges", "string_ppi_edges"}


def load_gene_set(path: Path) -> set[str]:
    """加载核心基因集."""
    genes: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            g = line.strip().upper()
            if g:
                genes.add(g)
    return genes


def normalize_gene(g) -> str | None:
    """标准化基因名."""
    if pd.isna(g):
        return None
    g = str(g).strip().upper()
    return g if g else None


def backup_file(path: Path) -> Path:
    """创建文件备份."""
    bak_path = path.with_suffix(path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy2(path, bak_path)
        logger.info("  已备份: %s", bak_path.name)
    return bak_path


def clean_celltype_markers(path: Path, core_genes: set[str]) -> pd.DataFrame | None:
    """清洗 celltype_marker_genes.csv."""
    df = pd.read_csv(path)
    if "celltype" not in df.columns or "gene" not in df.columns:
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["gene"] = df["gene"].apply(normalize_gene)
    df = df.dropna(subset=["celltype", "gene"])
    df["celltype"] = df["celltype"].astype(str).str.strip()
    df = df[df["gene"].isin(core_genes)]
    df = df.drop_duplicates(subset=["celltype", "gene"])

    df["source"] = "PanglaoDB_literature"
    df["confidence"] = 0.70
    df["download_date"] = DOWNLOAD_DATE
    return df[["celltype", "gene", "source", "confidence", "download_date"]].sort_values(
        ["celltype", "gene"]
    )


def clean_gene_coexp_edges(path: Path, core_genes: set[str]) -> pd.DataFrame | None:
    """清洗 gene_coexp_edges.csv."""
    df = pd.read_csv(path)
    if "gene_A" not in df.columns or "gene_B" not in df.columns:
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["gene_A"] = df["gene_A"].apply(normalize_gene)
    df["gene_B"] = df["gene_B"].apply(normalize_gene)
    df = df.dropna(subset=["gene_A", "gene_B"])
    df = df[df["gene_A"].isin(core_genes) & df["gene_B"].isin(core_genes)]

    # 去除自环(共表达允许自环,但此处 STRING score 自环无生物学意义)
    df = df[df["gene_A"] != df["gene_B"]]
    df = df.drop_duplicates(subset=["gene_A", "gene_B"])

    df["source"] = "STRING"
    df["confidence"] = (df["score"].clip(0, 1000) / 1000.0).round(4)
    df["download_date"] = DOWNLOAD_DATE
    return df[["gene_A", "gene_B", "score", "source", "confidence", "download_date"]].sort_values(
        ["gene_A", "gene_B"]
    )


def clean_gene_pathway_enrichment(path: Path, core_genes: set[str]) -> pd.DataFrame | None:
    """清洗 gene_pathway_enrichment.csv."""
    df = pd.read_csv(path)
    required = {"gene", "pathway", "source", "adj_p_value"}
    if not required.issubset(df.columns):
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["gene"] = df["gene"].apply(normalize_gene)
    df = df.dropna(subset=["gene", "pathway", "adj_p_value"])
    df = df[df["gene"].isin(core_genes)]
    df = df.drop_duplicates(subset=["gene", "pathway"])

    # confidence: adj_p_value 越小置信度越高, -log10(p) / 10 封顶 1.0, 保底 0.5
    neg_log_p = -np.log10(df["adj_p_value"].replace(0, 1e-300))
    df["confidence"] = (neg_log_p / 10.0).clip(0.5, 1.0).round(4)
    df["download_date"] = DOWNLOAD_DATE
    return df[["gene", "pathway", "source", "adj_p_value", "confidence", "download_date"]].sort_values(
        ["gene", "pathway"]
    )


def clean_ligand_receptor_pairs(path: Path, core_genes: set[str]) -> pd.DataFrame | None:
    """清洗 ligand_receptor_pairs.csv."""
    df = pd.read_csv(path)
    if "ligand" not in df.columns or "receptor" not in df.columns:
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["ligand"] = df["ligand"].apply(normalize_gene)
    df["receptor"] = df["receptor"].apply(normalize_gene)
    df = df.dropna(subset=["ligand", "receptor"])
    df = df[df["ligand"].isin(core_genes) & df["receptor"].isin(core_genes)]
    df = df[df["ligand"] != df["receptor"]]  # 去除自环
    df = df.drop_duplicates(subset=["ligand", "receptor"])

    df["source"] = "CellChatDB_CellPhoneDB_literature"
    df["confidence"] = 0.85
    df["download_date"] = DOWNLOAD_DATE
    return df[["ligand", "receptor", "source", "confidence", "download_date"]].sort_values(
        ["ligand", "receptor"]
    )


def clean_string_ppi_edges(path: Path, core_genes: set[str]) -> pd.DataFrame | None:
    """清洗 string_ppi_edges.csv."""
    df = pd.read_csv(path)
    if "protein_A" not in df.columns or "protein_B" not in df.columns:
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["protein_A"] = df["protein_A"].apply(normalize_gene)
    df["protein_B"] = df["protein_B"].apply(normalize_gene)
    df = df.dropna(subset=["protein_A", "protein_B"])
    df = df[df["protein_A"].isin(core_genes) & df["protein_B"].isin(core_genes)]
    # PPI 允许自环(同源二聚体)
    df = df.drop_duplicates(subset=["protein_A", "protein_B"])

    df["source"] = "STRING"
    df["confidence"] = (df["score"].clip(0, 1000) / 1000.0).round(4)
    df["download_date"] = DOWNLOAD_DATE
    return df[["protein_A", "protein_B", "score", "source", "confidence", "download_date"]].sort_values(
        ["protein_A", "protein_B"]
    )


def clean_trrust_tf_target(path: Path, core_genes: set[str]) -> pd.DataFrame | None:
    """清洗 trrust_tf_target.csv."""
    df = pd.read_csv(path)
    if "tf" not in df.columns or "target" not in df.columns:
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["tf"] = df["tf"].apply(normalize_gene)
    df["target"] = df["target"].apply(normalize_gene)
    df = df.dropna(subset=["tf", "target"])
    df = df[df["tf"].isin(core_genes) & df["target"].isin(core_genes)]
    df = df[df["tf"] != df["target"]]  # 去除自调控环
    df = df.drop_duplicates(subset=["tf", "target"])

    df["source"] = "TRRUST_literature"
    df["confidence"] = 0.80
    df["download_date"] = DOWNLOAD_DATE
    return df[["tf", "target", "source", "confidence", "download_date"]].sort_values(
        ["tf", "target"]
    )


def clean_disease_gene_associations(path: Path, core_genes: set[str]) -> pd.DataFrame | None:
    """清洗 disease_gene_associations.csv."""
    df = pd.read_csv(path)
    if "disease" not in df.columns or "gene" not in df.columns:
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["gene"] = df["gene"].apply(normalize_gene)
    df = df.dropna(subset=["disease", "gene"])
    df["disease"] = df["disease"].astype(str).str.strip()
    df = df[df["gene"].isin(core_genes)]
    df = df.drop_duplicates(subset=["disease", "gene"])

    df["source"] = "DisGeNET_GenAge_AlzGene_literature"
    df["confidence"] = 0.70
    df["download_date"] = DOWNLOAD_DATE
    return df[["disease", "gene", "source", "confidence", "download_date"]].sort_values(
        ["disease", "gene"]
    )


def clean_compound_target_edges(path: Path) -> pd.DataFrame | None:
    """补充 compound_target_edges.csv 的 download_date."""
    df = pd.read_csv(path)
    required = {"compound", "gene", "source", "confidence", "confidence_level"}
    if not required.issubset(df.columns):
        logger.error("  列名不符合预期: %s", list(df.columns))
        return None

    df["gene"] = df["gene"].apply(normalize_gene)
    df = df.dropna(subset=["compound", "gene"])
    df = df.drop_duplicates(subset=["compound", "gene"])
    df["download_date"] = DOWNLOAD_DATE
    return df[["compound", "gene", "source", "confidence", "confidence_level", "download_date"]].sort_values(
        ["compound", "gene"]
    )


CLEANERS = {
    "celltype_marker_genes.csv": clean_celltype_markers,
    "gene_coexp_edges.csv": clean_gene_coexp_edges,
    "gene_pathway_enrichment.csv": clean_gene_pathway_enrichment,
    "ligand_receptor_pairs.csv": clean_ligand_receptor_pairs,
    "string_ppi_edges.csv": clean_string_ppi_edges,
    "trrust_tf_target.csv": clean_trrust_tf_target,
    "disease_gene_associations.csv": clean_disease_gene_associations,
    "compound_target_edges.csv": clean_compound_target_edges,
}


def main() -> int:
    core_genes = load_gene_set(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    if not NETWORK_DIR.exists():
        logger.error("网络文件目录不存在: %s", NETWORK_DIR)
        return 1

    summary = []
    for filename, cleaner in CLEANERS.items():
        path = NETWORK_DIR / filename
        logger.info("=" * 60)
        logger.info("清洗 %s", filename)
        if not path.exists():
            logger.error("  文件不存在, 跳过")
            summary.append((filename, "missing", 0))
            continue

        try:
            backup_file(path)
            if filename == "compound_target_edges.csv":
                cleaned = cleaner(path)
            else:
                cleaned = cleaner(path, core_genes)

            if cleaned is None:
                summary.append((filename, "error", 0))
                continue

            cleaned.to_csv(path, index=False)
            logger.info("  → 清洗后保存 %d 条记录", len(cleaned))
            summary.append((filename, "ok", len(cleaned)))
        except Exception:
            logger.error("  清洗失败:\n%s", traceback.format_exc())
            summary.append((filename, "error", 0))

    logger.info("=" * 60)
    logger.info("核心边文件清洗汇总:")
    for filename, status, count in summary:
        logger.info("  %-40s %-10s %d", filename, status, count)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
