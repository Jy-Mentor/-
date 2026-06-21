"""化合物-铁衰老靶点-CIRI 通路机制解释分析.

基于统一 GAT-HGT pipeline 的预测结果, 对 63 个单体化合物进行:
  1. 全基因排序与铁衰老基因排序
  2. 已知靶点 vs 新预测靶点对比
  3. 通路富集/参与度计算
  4. BBB 渗透性整合
  5. BCP/VC/Fer-1/DFO 多维度比较
  6. 结构化机制解释表输出

所有输入均来自真实文件; 缺失数据会记录警告, 不静默填充.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
L3_RESULTS = PROJECT_ROOT / "L3_results"
NETWORK_DIR = PROJECT_ROOT / "network_files"

# 关键化合物(含中药单体与合成铁死亡工具)
KEY_COMPOUNDS = ["BCP", "VC", "Fer-1", "DFO"]

# 核心铁死亡/铁衰老调控基因(用于机制解释重点展示)
CORE_REGULATORY_GENES = {
    "ACSL4", "GPX4", "LPCAT3", "FTH1", "FTL", "TFRC", "SLC7A11",
    "NFE2L2", "KEAP1", "HMOX1", "PTGS2", "CDKN1A", "TP53", "MTDH",
}

# 疾病->用于定义 CIRI/神经退行相关通路
CIRI_RELATED_DISEASES = {"CIRI", "AD", "Aging"}


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """读取 CSV; 缺失时抛出异常并打印 traceback."""
    if not path.exists():
        raise FileNotFoundError(f"缺失必需文件: {path}")
    return pd.read_csv(path, **kwargs)


def load_iron_aging_genes() -> set[str]:
    """加载铁衰老基因列表."""
    gene_file = PROJECT_ROOT / "铁衰老基因.txt"
    if not gene_file.exists():
        logger.error("铁衰老基因文件不存在: %s", gene_file)
        return set()
    genes = {line.strip() for line in gene_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    logger.info("加载铁衰老基因 %d 个", len(genes))
    return genes


def load_ciri_related_genes() -> set[str]:
    """加载 CIRI/AD/Aging 疾病相关基因."""
    disease_path = NETWORK_DIR / "disease_gene_associations.csv"
    if not disease_path.exists():
        logger.warning("疾病基因关联文件不存在: %s", disease_path)
        return set()
    df = _read_csv(disease_path)
    if "disease" not in df.columns or "gene" not in df.columns:
        logger.warning("disease_gene_associations.csv 缺少 disease/gene 列")
        return set()
    genes = set(df.loc[df["disease"].isin(CIRI_RELATED_DISEASES), "gene"].dropna().unique())
    logger.info("CIRI/AD/Aging 相关基因 %d 个", len(genes))
    return genes


def load_gene_pathways() -> pd.DataFrame:
    """加载基因-通路注释."""
    pathway_path = NETWORK_DIR / "gene_pathway_enrichment.csv"
    df = _read_csv(pathway_path)
    required = {"gene", "pathway"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"gene_pathway_enrichment.csv 缺少列: {missing}")
    return df[["gene", "pathway", "source", "adj_p_value", "confidence"]].copy()


def load_known_targets() -> pd.DataFrame:
    """加载已知化合物-靶点边."""
    ct_path = NETWORK_DIR / "compound_target_edges.csv"
    df = _read_csv(ct_path)
    required = {"compound", "gene"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compound_target_edges.csv 缺少列: {missing}")
    return df[["compound", "gene", "source", "confidence", "confidence_level"]].copy()


def load_bbb_data() -> pd.DataFrame:
    """加载 BBB 预测数据."""
    bbb_path = NETWORK_DIR / "tcm_bbb_evaluation.csv"
    df = _read_csv(bbb_path)
    required = {"compound", "BBB_score", "BBB_pass", "BBB_reasons"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tcm_bbb_evaluation.csv 缺少列: {missing}")
    return df[list(required)].copy()


def compute_per_compound_targets(
    scores_df: pd.DataFrame,
    iron_aging_genes: set[str],
    ciri_genes: set[str],
    known_targets_df: pd.DataFrame,
    top_k: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算每个化合物的 top 靶点及铁衰老相关靶点.

    Returns:
        per_compound_df: 每个化合物的 top 靶点(含已知/新预测标记、铁衰老标记).
        summary_df: 每个化合物的汇总统计.
    """
    known_pairs = set(
        zip(known_targets_df["compound"], known_targets_df["gene"])
    ) if not known_targets_df.empty else set()

    records: list[dict] = []
    summary_records: list[dict] = []

    for compound, sub in scores_df.groupby("compound_name"):
        sub = sub.sort_values("score", ascending=False).reset_index(drop=True)

        # 全基因 top-k
        top_k_df = sub.head(top_k).copy()
        top_k_df["rank"] = np.arange(1, len(top_k_df) + 1)
        top_k_df["is_iron_aging"] = top_k_df["gene_name"].isin(iron_aging_genes)
        top_k_df["is_ciri_related"] = top_k_df["gene_name"].isin(ciri_genes)
        top_k_df["is_known_target"] = top_k_df.apply(
            lambda r: (r["compound_name"], r["gene_name"]) in known_pairs, axis=1
        )

        # 铁衰老基因子集排序
        ia_sub = sub[sub["gene_name"].isin(iron_aging_genes)].copy()
        ia_sub["rank_in_iron_aging"] = np.arange(1, len(ia_sub) + 1)

        records.append(top_k_df)

        # 汇总
        n_ia_in_topk = int(top_k_df["is_iron_aging"].sum())
        n_known_in_topk = int(top_k_df["is_known_target"].sum())
        n_ciri_in_topk = int(top_k_df["is_ciri_related"].sum())

        summary_records.append({
            "compound": compound,
            "total_genes_scored": len(sub),
            "top_k": top_k,
            "top_k_iron_aging_count": n_ia_in_topk,
            "top_k_iron_aging_ratio": round(n_ia_in_topk / top_k, 4),
            "top_k_known_target_count": n_known_in_topk,
            "top_k_ciri_related_count": n_ciri_in_topk,
            "max_score": float(sub["score"].max()),
            "mean_score": float(sub["score"].mean()),
            "std_score": float(sub["score"].std()),
            "top_iron_aging_gene": ia_sub["gene_name"].iloc[0] if not ia_sub.empty else None,
            "top_iron_aging_score": float(ia_sub["score"].iloc[0]) if not ia_sub.empty else None,
        })

    per_compound_df = pd.concat(records, ignore_index=True)
    summary_df = pd.DataFrame(summary_records)
    return per_compound_df, summary_df


def compute_pathway_participation(
    per_compound_df: pd.DataFrame,
    gene_pathways_df: pd.DataFrame,
    ciri_genes: set[str],
    top_k: int = 50,
) -> pd.DataFrame:
    """计算每个化合物在 top-K 靶点所映射到的通路上的参与度.

    参与度 = 该通路中包含的 top-K 靶点数量 / top_K, 加权置信度取平均.
    """
    top_df = per_compound_df[per_compound_df["rank"] <= top_k].copy()
    merged = top_df.merge(
        gene_pathways_df,
        left_on="gene_name",
        right_on="gene",
        how="left",
        suffixes=("", "_pathway"),
    )

    # 标记 CIRI 相关通路: 包含至少一个 CIRI/AD/Aging 基因的通路
    ciri_pathways = set(
        gene_pathways_df[gene_pathways_df["gene"].isin(ciri_genes)]["pathway"].unique()
    )

    records: list[dict] = []
    for compound, sub in merged.groupby("compound_name"):
        # 只考虑有通路注释的基因
        annotated = sub.dropna(subset=["pathway"])
        if annotated.empty:
            logger.warning("化合物 %s 的 top-%d 靶点无通路注释", compound, top_k)
            continue

        pathway_stats = annotated.groupby("pathway").agg(
            target_count=("gene_name", "nunique"),
            avg_score=("score", "mean"),
            avg_confidence=("confidence", "mean"),
        ).reset_index()
        pathway_stats["compound"] = compound
        pathway_stats["participation_ratio"] = pathway_stats["target_count"] / top_k
        pathway_stats["is_ciri_related"] = pathway_stats["pathway"].isin(ciri_pathways)
        pathway_stats = pathway_stats.sort_values(
            ["is_ciri_related", "participation_ratio"], ascending=[False, False]
        )
        records.append(pathway_stats)

    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def build_mechanism_table(
    per_compound_df: pd.DataFrame,
    gene_pathways_df: pd.DataFrame,
    bbb_df: pd.DataFrame,
    key_compounds: list[str] | None = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """构建结构化的"化合物-铁衰老靶点-CIRI通路"解释表.

    每行代表一个关键化合物的核心靶点-通路对, 可直接支撑论文机制章节.
    """
    if key_compounds is None:
        key_compounds = KEY_COMPOUNDS

    # 通路映射字典
    gene_to_pathways = (
        gene_pathways_df.groupby("gene")["pathway"].apply(lambda x: sorted(set(x))).to_dict()
    )

    bbb_dict = (
        bbb_df.set_index("compound")[["BBB_score", "BBB_pass", "BBB_reasons"]].to_dict("index")
        if not bbb_df.empty else {}
    )

    records: list[dict] = []
    for compound in key_compounds:
        sub = per_compound_df[
            (per_compound_df["compound_name"] == compound)
            & (per_compound_df["is_iron_aging"])
        ].copy()
        sub = sub.sort_values("rank").head(top_k)

        if sub.empty:
            logger.warning("关键化合物 %s 在 top-%d 铁衰老靶点中无记录", compound, top_k)

        for _, row in sub.iterrows():
            gene = row["gene_name"]
            score = float(row["score"])
            rank = int(row["rank"])
            is_known = bool(row["is_known_target"])
            pathways = gene_to_pathways.get(gene, [])
            pathways_str = "; ".join(pathways[:5]) if pathways else "NA"

            records.append({
                "compound": compound,
                "compound_type": "TCM_monomer" if compound in ("BCP", "VC") else "synthetic_tool",
                "target_gene": gene,
                "predicted_score": round(score, 4),
                "overall_rank": rank,
                "is_known_target": is_known,
                "related_pathways": pathways_str,
                "BBB_score": bbb_dict.get(compound, {}).get("BBB_score"),
                "BBB_pass": bbb_dict.get(compound, {}).get("BBB_pass"),
            })

    return pd.DataFrame(records)


def compare_key_compounds(
    summary_df: pd.DataFrame,
    per_compound_df: pd.DataFrame,
    bbb_df: pd.DataFrame,
    gene_pathways_df: pd.DataFrame,
    top_k: int = 50,
) -> pd.DataFrame:
    """对 BCP/VC/Fer-1/DFO 进行多维度比较."""
    bbb_dict = (
        bbb_df.set_index("compound")[["BBB_score", "BBB_pass"]].to_dict("index")
        if not bbb_df.empty else {}
    )

    # 每个化合物在 top-k 中的铁衰老靶点列表
    top_ia_targets: dict[str, list[str]] = {}
    for compound in KEY_COMPOUNDS:
        sub = per_compound_df[
            (per_compound_df["compound_name"] == compound)
            & (per_compound_df["rank"] <= top_k)
            & (per_compound_df["is_iron_aging"])
        ]
        top_ia_targets[compound] = sub.sort_values("rank")["gene_name"].tolist()

    records: list[dict] = []
    for compound in KEY_COMPOUNDS:
        row = summary_df[summary_df["compound"] == compound]
        if row.empty:
            logger.warning("汇总表中缺少关键化合物 %s", compound)
            continue
        row = row.iloc[0]

        ia_targets = top_ia_targets.get(compound, [])
        # 计算前 5 个铁衰老靶点涉及的核心通路数
        core_pathways: set[str] = set()
        for gene in ia_targets[:5]:
            core_pathways.update(gene_pathways_df[gene_pathways_df["gene"] == gene]["pathway"].unique())

        records.append({
            "compound": compound,
            "compound_category": "TCM_monomer" if compound in ("BCP", "VC") else "synthetic_ferroptosis_tool",
            "top_k_iron_aging_count": int(row["top_k_iron_aging_count"]),
            "top_k_iron_aging_ratio": float(row["top_k_iron_aging_ratio"]),
            "top_k_known_target_count": int(row["top_k_known_target_count"]),
            "max_score": float(row["max_score"]),
            "mean_score": float(row["mean_score"]),
            "top_iron_aging_gene": row["top_iron_aging_gene"],
            "top_5_iron_aging_targets": ", ".join(ia_targets[:5]),
            "core_pathway_count_top5": len(core_pathways),
            "BBB_score": bbb_dict.get(compound, {}).get("BBB_score"),
            "BBB_pass": bbb_dict.get(compound, {}).get("BBB_pass"),
        })

    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description="化合物-铁衰老靶点-CIRI 通路机制解释分析")
    parser.add_argument(
        "--experiment",
        type=str,
        default="gat_hgt_iron_aging_seed42",
        help="实验目录名 (位于 L3_results/)",
    )
    parser.add_argument("--top-k", type=int, default=50, help="每个化合物取 top-K 靶点")
    parser.add_argument("--mechanism-top-k", type=int, default=20, help="机制解释表取 top-K 铁衰老靶点")
    args = parser.parse_args()

    exp_dir = L3_RESULTS / args.experiment
    scores_path = exp_dir / "all_compound_gene_scores.csv"
    if not scores_path.exists():
        logger.error("预测分数文件不存在: %s", scores_path)
        return 1

    scores_df = _read_csv(scores_path)
    required = {"compound_name", "gene_name", "score"}
    missing = required - set(scores_df.columns)
    if missing:
        logger.error("预测分数文件缺少列: %s", missing)
        return 1

    output_dir = exp_dir / "mechanism_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        iron_aging_genes = load_iron_aging_genes()
        ciri_genes = load_ciri_related_genes()
        known_targets_df = load_known_targets()
        gene_pathways_df = load_gene_pathways()
        bbb_df = load_bbb_data()
    except Exception:
        logger.exception("加载输入数据失败")
        traceback.print_exc()
        return 1

    # 1. 每个化合物 top 靶点与汇总
    per_compound_df, summary_df = compute_per_compound_targets(
        scores_df,
        iron_aging_genes,
        ciri_genes,
        known_targets_df,
        top_k=args.top_k,
    )

    per_compound_path = output_dir / f"per_compound_top{args.top_k}_targets.csv"
    per_compound_df.to_csv(per_compound_path, index=False, encoding="utf-8-sig")
    logger.info("保存 per-compound top targets: %s", per_compound_path)

    summary_path = output_dir / "compound_mechanism_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info("保存 compound mechanism summary: %s", summary_path)

    # 2. 通路参与度
    pathway_df = compute_pathway_participation(
        per_compound_df,
        gene_pathways_df,
        ciri_genes,
        top_k=args.top_k,
    )
    if not pathway_df.empty:
        pathway_path = output_dir / f"compound_pathway_participation_top{args.top_k}.csv"
        pathway_df.to_csv(pathway_path, index=False, encoding="utf-8-sig")
        logger.info("保存 pathway participation: %s", pathway_path)
    else:
        logger.warning("未生成通路参与度表")

    # 3. 关键化合物比较
    comparison_df = compare_key_compounds(
        summary_df,
        per_compound_df,
        bbb_df,
        gene_pathways_df,
        top_k=args.top_k,
    )
    comparison_path = output_dir / "key_compounds_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    logger.info("保存关键化合物比较: %s", comparison_path)

    # 4. 结构化机制解释表
    mechanism_df = build_mechanism_table(
        per_compound_df,
        gene_pathways_df,
        bbb_df,
        key_compounds=KEY_COMPOUNDS,
        top_k=args.mechanism_top_k,
    )
    mechanism_path = output_dir / f"mechanism_table_top{args.mechanism_top_k}.csv"
    mechanism_df.to_csv(mechanism_path, index=False, encoding="utf-8-sig")
    logger.info("保存机制解释表: %s", mechanism_path)

    # 5. 汇总 JSON
    summary = {
        "experiment": args.experiment,
        "compounds_scored": int(scores_df["compound_name"].nunique()),
        "genes_scored": int(scores_df["gene_name"].nunique()),
        "iron_aging_genes_in_graph": len(iron_aging_genes),
        "ciri_related_genes": len(ciri_genes),
        "known_compound_target_pairs": len(known_targets_df),
        "top_k": args.top_k,
        "key_compounds": KEY_COMPOUNDS,
        "per_compound_summary": summary_df.to_dict("records"),
        "key_compounds_comparison": comparison_df.to_dict("records"),
        "note": "Scores are model logits used for ranking, not calibrated probabilities.",
    }
    summary_path = output_dir / "mechanism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("保存机制汇总 JSON: %s", summary_path)

    # 打印关键化合物摘要
    logger.info("\n关键化合物比较摘要:\n%s", comparison_df.to_string(index=False))
    logger.info("\n机制解释表行数: %d", len(mechanism_df))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
