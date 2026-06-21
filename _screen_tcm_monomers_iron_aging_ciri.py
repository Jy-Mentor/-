"""基于真实网络数据筛选可能通过铁衰老路径治疗CIRI的中药单体.

输入数据:
  - network_files/compound_smiles.csv
  - network_files/compound_target_edges.csv
  - network_files/ctd_compound_disease.csv
  - network_files/ferrdb_compound_regulators.csv
  - network_files/disgenet_ciri_genes.csv
  - network_files/disease_gene_associations.csv
  - network_files/gene_pathway_enrichment.csv
  - 铁衰老基因.txt
  - L1/core_gene_set.csv

输出:
  - L3_results/tcm_monomer_screening/iron_aging_ciri_candidates.csv
  - L3_results/tcm_monomer_screening/candidate_summary.json

不生成任何模拟数据; 缺失文件会记录警告或异常.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
OUTPUT_DIR = PROJECT_ROOT / "L3_results" / "tcm_monomer_screening"

# 合成铁死亡工具化合物(非中药单体), 在候选筛选中排除
SYNTHETIC_FERROPTOSIS_TOOLS = {
    "Fer-1", "DFO", "Lip-1", "Erastin", "RSL3", "ML162",
    "Deferoxamine", "Ferrostatin-1", "Liproxstatin-1",
}

# 重点关注的CIRI相关通路(用于输出解释)
KEY_CIRI_PATHWAYS = {
    "Ferroptosis",
    "p53 signaling pathway",
    "MAPK signaling pathway",
    "HIF-1 signaling pathway",
    "TNF signaling pathway",
    "NF-kappa B signaling pathway",
    "NOD-like receptor signaling pathway",
    "Toll-like receptor signaling pathway",
    "PI3K-Akt signaling pathway",
    "Neurotrophin signaling pathway",
}


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """读取CSV; 缺失时抛出异常."""
    if not path.exists():
        raise FileNotFoundError(f"缺失必需文件: {path}")
    return pd.read_csv(path, **kwargs)


def load_compounds() -> pd.DataFrame:
    """加载化合物列表."""
    df = _read_csv(NETWORK_DIR / "compound_smiles.csv")
    required = {"compound", "cid", "CanonicalSMILES"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compound_smiles.csv 缺少列: {missing}")
    return df[["compound", "cid", "CanonicalSMILES"]].copy()


def load_iron_aging_genes() -> set[str]:
    """加载铁衰老基因列表."""
    gene_file = PROJECT_ROOT / "铁衰老基因.txt"
    if not gene_file.exists():
        logger.error("铁衰老基因文件不存在: %s", gene_file)
        return set()
    genes = {
        line.strip()
        for line in gene_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    logger.info("加载铁衰老基因 %d 个", len(genes))
    return genes


def load_ciri_genes() -> set[str]:
    """加载CIRI相关基因(来自DisGeNET和disease_gene_associations)."""
    genes: set[str] = set()

    disgenet_path = NETWORK_DIR / "disgenet_ciri_genes.csv"
    if disgenet_path.exists():
        df = _read_csv(disgenet_path)
        if "gene" in df.columns:
            genes.update(df["gene"].dropna().astype(str).unique())
        else:
            logger.warning("disgenet_ciri_genes.csv 缺少 gene 列")
    else:
        logger.warning("缺失文件: %s", disgenet_path)

    disease_assoc_path = NETWORK_DIR / "disease_gene_associations.csv"
    if disease_assoc_path.exists():
        df = _read_csv(disease_assoc_path)
        if "disease" in df.columns and "gene" in df.columns:
            ciri_assoc = df[df["disease"] == "CIRI"]
            genes.update(ciri_assoc["gene"].dropna().astype(str).unique())
        else:
            logger.warning("disease_gene_associations.csv 缺少 disease/gene 列")
    else:
        logger.warning("缺失文件: %s", disease_assoc_path)

    logger.info("加载 CIRI 相关基因 %d 个", len(genes))
    return genes


def load_compound_targets() -> pd.DataFrame:
    """加载化合物-靶点边."""
    df = _read_csv(NETWORK_DIR / "compound_target_edges.csv")
    required = {"compound", "gene", "source", "confidence", "confidence_level"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compound_target_edges.csv 缺少列: {missing}")
    return df[["compound", "gene", "source", "confidence", "confidence_level"]].copy()


def load_ctd_disease() -> pd.DataFrame:
    """加载CTD化合物-疾病关系."""
    path = NETWORK_DIR / "ctd_compound_disease.csv"
    if not path.exists():
        logger.warning("缺失文件: %s", path)
        return pd.DataFrame(columns=["compound", "disease", "direct_evidence", "confidence"])
    df = _read_csv(path)
    required = {"compound", "disease", "direct_evidence", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ctd_compound_disease.csv 缺少列: {missing}")
    return df[["compound", "disease", "direct_evidence", "confidence"]].copy()


def load_ferrdb_regulators() -> pd.DataFrame:
    """加载FerrDb化合物调控信息."""
    path = NETWORK_DIR / "ferrdb_compound_regulators.csv"
    if not path.exists():
        logger.warning("缺失文件: %s", path)
        return pd.DataFrame(columns=["compound", "role", "pmid", "source"])
    df = _read_csv(path)
    required = {"compound", "role", "pmid", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ferrdb_compound_regulators.csv 缺少列: {missing}")
    return df[["compound", "role", "pmid", "source"]].copy()


def load_gene_pathways() -> pd.DataFrame:
    """加载基因-通路富集注释."""
    path = NETWORK_DIR / "gene_pathway_enrichment.csv"
    if not path.exists():
        logger.warning("缺失文件: %s", path)
        return pd.DataFrame(columns=["gene", "pathway", "source", "adj_p_value", "confidence"])
    df = _read_csv(path)
    required = {"gene", "pathway", "source", "adj_p_value", "confidence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gene_pathway_enrichment.csv 缺少列: {missing}")
    return df[["gene", "pathway", "source", "adj_p_value", "confidence"]].copy()


def _map_confidence_level(level: str) -> float:
    """将confidence_level映射为数值权重."""
    mapping = {"high": 1.0, "medium": 0.7, "low": 0.4}
    return mapping.get(str(level).lower(), 0.5)


def screen_candidates(
    compounds_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    ctd_df: pd.DataFrame,
    ferrdb_df: pd.DataFrame,
    pathways_df: pd.DataFrame,
    iron_aging_genes: set[str],
    ciri_genes: set[str],
) -> pd.DataFrame:
    """筛选并评分候选中药单体."""
    # 排除合成铁死亡工具
    all_compounds = set(compounds_df["compound"].unique())
    tcm_compounds = sorted(all_compounds - SYNTHETIC_FERROPTOSIS_TOOLS)
    logger.info("化合物总数 %d, 排除合成工具后剩余 %d 个", len(all_compounds), len(tcm_compounds))

    # CTD中CIRI直接治疗证据
    ctd_ciri = set()
    if not ctd_df.empty and "disease" in ctd_df.columns:
        ctd_ciri = set(ctd_df[ctd_df["disease"] == "CIRI"]["compound"].unique())
    logger.info("CTD中直接标注 CIRI therapeutic 的化合物 %d 个", len(ctd_ciri))

    # FerrDb角色
    ferrdb_roles: dict[str, set[str]] = {}
    if not ferrdb_df.empty:
        for compound, sub in ferrdb_df.groupby("compound"):
            ferrdb_roles[compound] = set(sub["role"].dropna().unique())

    # 基因到通路映射
    gene_to_pathways = {}
    if not pathways_df.empty:
        gene_to_pathways = (
            pathways_df.groupby("gene")["pathway"].apply(lambda x: sorted(set(x))).to_dict()
        )

    records: list[dict] = []
    for compound in tcm_compounds:
        sub = targets_df[targets_df["compound"] == compound]

        known_targets = set(sub["gene"].dropna().unique()) if not sub.empty else set()
        iron_aging_targets = known_targets & iron_aging_genes
        ciri_targets = known_targets & ciri_genes
        bridge_targets = iron_aging_targets & ciri_targets

        # 置信度加权
        if not sub.empty:
            sub = sub.copy()
            sub["conf_weight"] = sub["confidence_level"].apply(_map_confidence_level)
            weighted_score = float((sub["confidence"] * sub["conf_weight"]).sum() / max(sub["conf_weight"].sum(), 1e-9))
            mean_confidence = float(sub["confidence"].mean())
            high_conf_pairs = int(((sub["confidence"] >= 0.8) | (sub["confidence_level"] == "high")).sum())
        else:
            weighted_score = 0.0
            mean_confidence = 0.0
            high_conf_pairs = 0

        # 通路覆盖
        covered_pathways: set[str] = set()
        for gene in known_targets:
            covered_pathways.update(gene_to_pathways.get(gene, []))
        key_ciri_pathways_hit = sorted(covered_pathways & KEY_CIRI_PATHWAYS)

        # FerrDb证据
        roles = ferrdb_roles.get(compound, set())
        ferrdb_role_str = ";".join(sorted(roles)) if roles else "NA"
        is_ferrdb_inducer = "inducer" in roles
        is_ferrdb_inhibitor = "inhibitor" in roles

        # 是否直接有CIRI治疗证据
        has_ctd_ciri = compound in ctd_ciri

        # 评分: 多证据综合
        # 基础分 = 铁衰老靶点数*2 + 桥接靶点数*3 + CIRI靶点数*1 + CTD证据*2
        # 加权: 平均置信度 * (1 + FerrDb证据)
        score = (
            len(iron_aging_targets) * 2.0
            + len(bridge_targets) * 3.0
            + len(ciri_targets) * 1.0
            + (2.0 if has_ctd_ciri else 0.0)
            + (1.0 if is_ferrdb_inducer else 0.0)
            + (0.5 if is_ferrdb_inhibitor else 0.0)
        )
        score *= mean_confidence if mean_confidence > 0 else 1.0

        records.append({
            "compound": compound,
            "cid": compounds_df.loc[compounds_df["compound"] == compound, "cid"].iloc[0]
            if compound in compounds_df["compound"].values else None,
            "known_target_count": len(known_targets),
            "iron_aging_target_count": len(iron_aging_targets),
            "ciri_target_count": len(ciri_targets),
            "bridge_target_count": len(bridge_targets),
            "iron_aging_targets": ";".join(sorted(iron_aging_targets)) if iron_aging_targets else "NA",
            "bridge_targets": ";".join(sorted(bridge_targets)) if bridge_targets else "NA",
            "ciri_targets": ";".join(sorted(ciri_targets)) if ciri_targets else "NA",
            "mean_confidence": round(mean_confidence, 4),
            "weighted_score": round(weighted_score, 4),
            "high_confidence_pairs": high_conf_pairs,
            "has_ctd_ciri_evidence": has_ctd_ciri,
            "ferrdb_role": ferrdb_role_str,
            "is_ferrdb_inducer": is_ferrdb_inducer,
            "key_ciri_pathways_hit": ";".join(key_ciri_pathways_hit) if key_ciri_pathways_hit else "NA",
            "key_pathway_count": len(key_ciri_pathways_hit),
            "candidate_score": round(score, 4),
        })

    df = pd.DataFrame(records)
    df = df.sort_values("candidate_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def main() -> int:
    try:
        compounds_df = load_compounds()
        iron_aging_genes = load_iron_aging_genes()
        ciri_genes = load_ciri_genes()
        targets_df = load_compound_targets()
        ctd_df = load_ctd_disease()
        ferrdb_df = load_ferrdb_regulators()
        pathways_df = load_gene_pathways()
    except Exception:
        logger.exception("加载输入数据失败")
        traceback.print_exc()
        return 1

    if compounds_df.empty:
        logger.error("化合物列表为空")
        return 1
    if targets_df.empty:
        logger.error("化合物-靶点边为空")
        return 1

    candidates_df = screen_candidates(
        compounds_df,
        targets_df,
        ctd_df,
        ferrdb_df,
        pathways_df,
        iron_aging_genes,
        ciri_genes,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "iron_aging_ciri_candidates.csv"
    candidates_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info("保存候选化合物表: %s (%d 行)", out_csv, len(candidates_df))

    # 汇总JSON
    summary = {
        "total_compounds": int(compounds_df["compound"].nunique()),
        "tcm_compounds screened": len(candidates_df),
        "iron_aging_genes_loaded": len(iron_aging_genes),
        "ciri_genes_loaded": len(ciri_genes),
        "known_compound_target_pairs": len(targets_df),
        "top_10_candidates": candidates_df.head(10).to_dict("records"),
        "note": (
            "Scores combine known iron-aging targets, CIRI targets, bridge targets, "
            "CTD evidence, and FerrDb roles; based solely on real network files."
        ),
    }
    out_json = OUTPUT_DIR / "candidate_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("保存汇总 JSON: %s", out_json)

    logger.info("\nTop 15 候选中药单体:\n%s", candidates_df.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
