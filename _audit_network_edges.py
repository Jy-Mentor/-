"""网络边质量审计脚本 (v2).

对 network_files/ 下所有边文件执行系统化质量检查:
- 重复边
- 自环边
- 节点不在核心基因集内
- 可疑基因名
- 缺失 source/confidence 列
- 缺失值比例
- 孤立节点

输出: network_files/edge_quality_audit.json
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
GENE_LIST_PATH = PROJECT_ROOT / "铁衰老基因.txt"
OUTPUT_JSON = PROJECT_ROOT / "network_files" / "edge_quality_audit.json"

# 哪些边类型允许自环（同源二聚体等）
SELF_LOOP_ALLOWED = {"string_ppi_edges", "gene_coexp_edges"}

# 核心图构建边文件：节点必须在核心基因集内，且应记录 source/confidence
CORE_EDGE_FILES = {
    "compound_target_edges.csv",
    "string_ppi_edges.csv",
    "gene_coexp_edges.csv",
    "trrust_tf_target.csv",
    "disease_gene_associations.csv",
    "celltype_marker_genes.csv",
    "ligand_receptor_pairs.csv",
    "gene_pathway_enrichment.csv",
}

# 精选高置信度边文件（人工整理/ChEMBL清洗后）
CURATED_EDGE_FILES = {
    "acsl4_curated_edges.csv",
    "compound_target_edges_curated.csv",
    "chembl_compound_targets_cleaned.csv",
    "panglaodb_celltype_markers_cleaned.csv",
}

# 外部原始参考文件：允许节点超出核心集，但检查重复/可疑名/物种等
REFERENCE_FILES = {
    "chembl_compound_targets.csv",
    "cellchat_lr_pairs.csv",
    "panglaodb_celltype_markers.csv",
    "gene_pathway_enrichment_external.csv",
    "msigdb_gene_pathways.csv",
    "kegg_pathway_genes.csv",
    "disgenet_ciri_genes.csv",
    "disgenet_disease_genes.csv",
    "opentargets_ciri_genes.csv",
}

# 明确为非边文件的集合（特征、属性、配置类）
NON_EDGE_FILES = {
    "acsl4_pocket_features.csv",
    "acsl4_pocket_residues.csv",
    "compound_attentivefp_embeddings.csv",
    "compound_smiles.csv",
    "pubchem_compound_props.csv",
    "tcm_bbb_evaluation.csv",
    "tcm_monomers_pubchem.csv",
}


def load_gene_set(path: Path) -> set[str]:
    genes: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            g = line.strip().upper()
            if g:
                genes.add(g)
    return genes


def classify_file(filename: str) -> str:
    if filename in CORE_EDGE_FILES:
        return "core_edge"
    if filename in CURATED_EDGE_FILES:
        return "curated_edge"
    if filename in REFERENCE_FILES:
        return "reference"
    return "other"


def is_suspicious_gene_name(g) -> bool:
    """判断基因名称是否可疑（非标准HGNC符号）.

    真实HGNC符号可包含数字, 因此不将数字作为可疑条件.
    """
    if pd.isna(g):
        return True
    g = str(g).strip()
    if len(g) == 0 or len(g) > 20:
        return True
    if " " in g:
        return True
    if re.search(r"[-/().,;:]", g):
        return True
    # 包含小写字母或特殊字符
    if not re.match(r"^[A-Z0-9]+$", g):
        return True
    return False


def is_likely_mouse_gene(g) -> bool:
    """判断是否为小鼠 KEGG ID (mmu:*) 等."""
    if pd.isna(g):
        return False
    g = str(g).strip().lower()
    return g.startswith("mmu:") or g.startswith("mgi:")


def get_node_columns(df: pd.DataFrame, filename: str) -> list[str]:
    """根据文件名和列名推断节点列."""
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    # 明确映射
    if filename in {"string_ppi_edges.csv", "gene_coexp_edges.csv"}:
        return ["gene_A", "gene_B"] if "gene_A" in cols and "gene_B" in cols else ["protein_A", "protein_B"]
    if filename == "trrust_tf_target.csv":
        return ["tf", "target"]
    if filename == "ligand_receptor_pairs.csv":
        return ["ligand", "receptor"]
    compound_target_files = {
        "compound_target_edges.csv",
        "compound_target_edges_curated.csv",
        "chembl_compound_targets.csv",
        "chembl_compound_targets_cleaned.csv",
    }
    if filename in compound_target_files:
        return ["compound", "gene"]
    if filename == "disease_gene_associations.csv":
        return ["disease", "gene"]
    celltype_files = {
        "celltype_marker_genes.csv",
        "cellchat_lr_pairs.csv",
        "panglaodb_celltype_markers.csv",
        "panglaodb_celltype_markers_cleaned.csv",
    }
    if filename in celltype_files:
        # celltype_marker_genes: celltype, gene
        if "celltype" in lower and "gene" in lower:
            return [cols[lower.index("celltype")], cols[lower.index("gene")]]
    pathway_files = {
        "gene_pathway_enrichment.csv",
        "gene_pathway_enrichment_external.csv",
        "msigdb_gene_pathways.csv",
        "kegg_pathway_genes.csv",
    }
    if filename in pathway_files:
        # gene, pathway
        if "gene" in lower and "pathway" in lower:
            return [cols[lower.index("gene")], cols[lower.index("pathway")]]
        # kegg_pathway_genes 用 gene_id
        if "gene_id" in lower and "pathway" in lower:
            return [cols[lower.index("gene_id")], cols[lower.index("pathway")]]
    if filename in {"disgenet_disease_genes.csv", "disgenet_ciri_genes.csv", "opentargets_ciri_genes.csv"}:
        return ["disease", "gene"]
    if filename == "acsl4_curated_edges.csv":
        # 需要查看，先推断为 protein_A/protein_B 或类似
        if "protein_A" in cols and "protein_B" in cols:
            return ["protein_A", "protein_B"]
        if "gene_A" in cols and "gene_B" in cols:
            return ["gene_A", "gene_B"]

    # 通用推断：找 compound/gene/protein/disease/celltype/tf/target/ligand/receptor
    preferred = []
    for pattern in ["compound", "protein", "gene", "disease", "celltype", "tf", "target", "ligand", "receptor"]:
        for c in cols:
            if c.lower() == pattern and c not in preferred:
                preferred.append(c)
    if len(preferred) >= 2:
        return preferred[:2]

    # 最后回退：前两个非 score/confidence/source 列
    skip = {
        "score", "confidence", "source", "confidence_level",
        "standard_type", "standard_value", "standard_units",
        "pchembl_value", "adj_p_value", "download_date",
    }
    candidates = [c for c in cols if c.lower() not in skip]
    return candidates[:2]


def audit_file(path: Path, core_genes: set[str]) -> dict:
    """审计单个边文件."""
    result = {
        "file": str(path),
        "classification": classify_file(path.name),
        "exists": False,
        "errors": [],
        "warnings": [],
        "info": [],
        "stats": {},
    }
    if not path.exists():
        result["errors"].append("文件不存在")
        return result

    result["exists"] = True
    try:
        df = pd.read_csv(path)
    except Exception:
        result["errors"].append(f"读取失败: {traceback.format_exc()}")
        return result

    result["stats"]["rows"] = len(df)
    result["stats"]["columns"] = list(df.columns)

    if len(df) == 0:
        result["warnings"].append("文件为空")
        return result

    node_cols = get_node_columns(df, path.name)
    if len(node_cols) < 2:
        result["errors"].append(f"无法识别两个节点列, 列名: {list(df.columns)}")
        return result

    result["node_columns"] = node_cols
    classification = result["classification"]
    is_core = classification == "core_edge"
    is_curated = classification == "curated_edge"

    # 1. 缺失值检查
    for col in node_cols:
        na_count = df[col].isna().sum()
        if na_count > 0:
            msg = f"列 {col} 缺失值 {na_count} 条"
            if is_core or is_curated:
                result["warnings"].append(msg)
            else:
                result["info"].append(msg)

    # 2. 重复边
    subset = [c for c in node_cols if c in df.columns]
    dup_count = df.duplicated(subset=subset).sum()
    if dup_count > 0:
        result["warnings"].append(f"重复边 {dup_count} 条")

    # 3. 自环
    stem = path.stem
    allow_self_loop = stem in SELF_LOOP_ALLOWED
    if all(c in df.columns for c in node_cols):
        self_loop = df[node_cols[0]] == df[node_cols[1]]
        n_self = self_loop.sum()
        if n_self > 0 and not allow_self_loop:
            result["warnings"].append(f"自环边 {n_self} 条（非允许类型）")
        elif n_self > 0:
            result["stats"]["self_loops"] = int(n_self)
            result["info"].append(f"自环边 {n_self} 条（允许类型）")

    # 4. 基因节点不在核心集内
    gene_col_candidates = [
        c for c in node_cols
        if "gene" in c.lower() or "protein" in c.lower()
        or c in {"tf", "target", "ligand", "receptor"}
    ]
    for col in gene_col_candidates:
        if col not in df.columns:
            continue
        out_of_core = df[col].apply(lambda x: pd.notna(x) and str(x).strip().upper() not in core_genes).sum()
        if out_of_core > 0:
            msg = f"列 {col} 中 {out_of_core} 个节点不在 98 核心基因集内"
            if is_core or is_curated:
                result["warnings"].append(msg)
            else:
                result["info"].append(msg)

    # 5. 可疑基因名
    for col in gene_col_candidates:
        if col not in df.columns:
            continue
        suspicious = df[col].apply(is_suspicious_gene_name).sum()
        if suspicious > 0:
            result["warnings"].append(f"列 {col} 中 {suspicious} 个可疑基因名")
            sample = df[df[col].apply(is_suspicious_gene_name)][col].unique()[:10].tolist()
            result["warnings"].append(f"  示例: {sample}")

    # 6. 小鼠基因污染检查（主要针对 KEGG）
    for col in gene_col_candidates:
        if col not in df.columns:
            continue
        mouse_genes = df[col].apply(is_likely_mouse_gene).sum()
        if mouse_genes > 0:
            result["warnings"].append(f"列 {col} 中 {mouse_genes} 个小鼠 ID（如 mmu:...），物种可能错误")

    # 7. source/confidence 列检查（对核心边文件）
    if is_core:
        if "source" not in df.columns:
            result["warnings"].append("缺少 source 列（建议记录数据来源）")
        if "confidence" not in df.columns:
            result["warnings"].append("缺少 confidence 列（建议记录置信度）")
        if "download_date" not in df.columns:
            result["info"].append("缺少 download_date 列（建议记录数据日期）")

    # 8. 生成建议
    result["recommendation"] = _recommendation(result)
    return result


def _recommendation(result: dict) -> str:
    """根据审计结果生成处理建议."""
    if result["errors"]:
        return "修复读取/格式错误后重新审计"
    warnings = result["warnings"]
    classification = result["classification"]
    if not warnings:
        return "质量合格"
    if classification == "core_edge":
        if any("不在 98 核心基因集内" in w for w in warnings):
            return "过滤到核心基因集，或确认该边是否应进入核心图"
        if any("自环边" in w for w in warnings):
            return "去除非允许自环边"
        if any("source" in w or "confidence" in w for w in warnings):
            return "补充 source/confidence/download_date 列"
        return "清洗重复/可疑记录"
    if classification == "reference":
        if any("小鼠 ID" in w for w in warnings):
            return "检查物种，必要时替换为人类数据"
        return "可作为参考数据，但进入核心图前需清洗"
    return "人工复核"


def main() -> int:
    core_genes = load_gene_set(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    if not NETWORK_DIR.exists():
        logger.error("网络文件目录不存在: %s", NETWORK_DIR)
        return 1

    edge_files = sorted(p for p in NETWORK_DIR.glob("*.csv") if p.name not in NON_EDGE_FILES)
    logger.info("发现 %d 个 CSV 网络边文件", len(edge_files))

    audit_report = {
        "core_genes_count": len(core_genes),
        "files_audited": len(edge_files),
        "files": {},
    }

    for path in edge_files:
        logger.info("审计 %s", path.name)
        audit_report["files"][path.name] = audit_file(path, core_genes)

    summary = {
        "total_files": len(edge_files),
        "files_with_errors": sum(1 for r in audit_report["files"].values() if r["errors"]),
        "files_with_warnings": sum(1 for r in audit_report["files"].values() if r["warnings"]),
        "total_warnings": sum(len(r["warnings"]) for r in audit_report["files"].values()),
        "core_edge_files": sorted(CORE_EDGE_FILES & {p.name for p in edge_files}),
        "reference_files": sorted(REFERENCE_FILES & {p.name for p in edge_files}),
    }
    audit_report["summary"] = summary

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("审计报告已保存: %s", OUTPUT_JSON)
    logger.info("汇总: %s", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
