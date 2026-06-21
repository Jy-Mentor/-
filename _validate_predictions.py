"""对 top-k compound-target 预测进行外部数据库验证.

参考 DeepPurpose / DTINet 等 DTI 项目的做法：
- 不仅做精确名称匹配，还引入证据置信度加权
- 对化合物和基因名称做归一化（去空格、统一连接符、去除常见后缀）
- 输出 pair-level / compound-level / gene-level 三级证据统计
- 汇总各证据源、置信度分布及 top 有证据支持的预测

支持的证据源（按本地已有网络文件）:
- compound_target_edges.csv / compound_target_edges_curated.csv
- bindingdb_compound_targets.csv
- chembl_compound_targets_cleaned.csv
- ctd_compound_targets.csv
- drugbank_compound_targets.csv
- dgidb_compound_targets.csv
- drugtargetcommons_compound_targets.csv
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path

import pandas as pd

# 常见化合物别名，仅用于扩展本地数据库的精确匹配能力。
# 别名来源于网络文件中实际出现的记录及常见命名变体。
COMPOUND_ALIASES: dict[str, list[str]] = {
    "BCP": ["BCP", "CARYOPHYLLENE", "BETA-CARYOPHYLLENE", "Β-CARYOPHYLLENE"],
    "BETA-CARYOPHYLLENE": ["BCP", "CARYOPHYLLENE", "BETA-CARYOPHYLLENE", "Β-CARYOPHYLLENE"],
    "Β-CARYOPHYLLENE": ["BCP", "CARYOPHYLLENE", "BETA-CARYOPHYLLENE", "Β-CARYOPHYLLENE"],
    "VC": ["VC", "VITAMIN C", "ASCORBIC ACID", "ASCORBATE"],
    "VITAMIN C": ["VC", "VITAMIN C", "ASCORBIC ACID", "ASCORBATE"],
    "N-ACETYLCYSTEINE": ["N-ACETYLCYSTEINE", "NAC", "N-ACETYL CYSTEINE", "N-ACETYL-L-CYSTEINE"],
    "NAC": ["N-ACETYLCYSTEINE", "NAC", "N-ACETYL CYSTEINE", "N-ACETYL-L-CYSTEINE"],
    "EGCG": ["EGCG", "EPIGALLOCATECHIN GALLATE", "EPIGALLOCATECHIN-3-GALLATE"],
    "DANSOHENSU": ["DANSHENSU", "DANSOHENSU"],
    "SALIDROSIDE": ["SALIDROSIDE", "RHODIOLOSIDE"],
    "LUTEOLIN": ["LUTEOLIN", "2-(3,4-DIHYDROXYPHENYL)-5,7-DIHYDROXY-4H-1-BENZOPYRAN-4-ONE"],
    "FISETIN": ["FISETIN", "5-DESOXYQUERCETIN"],
    "CATECHIN": ["CATECHIN", "CATECHINIC ACID", "CATECHOL"],
    "QUERCETIN": ["QUERCETIN", "3,3',4',5,7-PENTAHYDROXYFLAVONE"],
    "GENISTEIN": ["GENISTEIN", "4',5,7-TRIHYDROXYISOFLAVONE"],
    "RESVERATROL": ["RESVERATROL", "TRANS-RESVERATROL"],
    "CURCUMIN": ["CURCUMIN", "DIFERULOYLMETHANE"],
}

# 证据源置信度映射（与网络文件中的 confidence_level 字段一致）
CONFIDENCE_WEIGHT: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.25,
}

DEFAULT_WEIGHT = 0.5  # 没有 confidence_level 时的默认权重


def normalize_name(s: str) -> str:
    """统一名称格式用于匹配：大写、去首尾空格、压缩连续空格."""
    return re.sub(r"\s+", " ", str(s).strip().upper())


def normalize_compound(name: str) -> str:
    """化合物名称归一化：统一连字符/下划线、去除常见盐形式后缀."""
    s = normalize_name(name)
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    # 去除常见盐/溶剂后缀（仅用于匹配，不改变原始输出）
    s = re.sub(r"\s+(HYDROCHLORIDE|HCL|SULFATE|SODIUM|CALCIUM|POTASSIUM|CITRATE|ACETATE)$", "", s)
    return s.strip()


def normalize_gene(name: str) -> str:
    """基因名称归一化：去除常见蛋白后缀."""
    s = normalize_name(name)
    s = re.sub(r"_HUMAN$", "", s)
    s = re.sub(r"\s+PROTEIN$", "", s)
    return s.strip()


def expand_compound_name(name: str) -> set[str]:
    """扩展化合物名称为可能的别名集合，用于交叉匹配."""
    normalized = normalize_compound(name)
    aliases = {normalized, normalize_name(name)}
    for key, alias_list in COMPOUND_ALIASES.items():
        if normalize_compound(key) == normalized or normalize_name(key) == normalize_name(name):
            aliases.update(normalize_compound(a) for a in alias_list)
            aliases.update(normalize_name(a) for a in alias_list)
    return aliases


def _confidence_to_weight(level: str | float | None) -> float:
    """将置信度字段转换为权重."""
    if level is None:
        return DEFAULT_WEIGHT
    if isinstance(level, (int, float)):
        return float(level)
    level_str = str(level).strip().lower()
    return CONFIDENCE_WEIGHT.get(level_str, DEFAULT_WEIGHT)


def load_external_pairs() -> tuple[pd.DataFrame, set[str], set[str]]:
    """加载所有本地化合物-靶点证据，并返回带权重的 pair 表、已知化合物集合、已知基因集合."""
    network_dir = Path("network_files")
    files = [
        ("compound_target_edges", network_dir / "compound_target_edges.csv"),
        ("compound_target_curated", network_dir / "compound_target_edges_curated.csv"),
        ("bindingdb", network_dir / "bindingdb_compound_targets.csv"),
        ("chembl", network_dir / "chembl_compound_targets_cleaned.csv"),
        ("ctd", network_dir / "ctd_compound_targets.csv"),
        ("drugbank", network_dir / "drugbank_compound_targets.csv"),
        ("dgidb", network_dir / "dgidb_compound_targets.csv"),
        ("drugtargetcommons", network_dir / "drugtargetcommons_compound_targets.csv"),
    ]

    rows: list[dict] = []
    known_compounds: set[str] = set()
    known_genes: set[str] = set()

    for source, path in files:
        if not path.exists():
            print(f"[跳过] 证据文件不存在: {path}")
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
            if "compound" not in df.columns or "gene" not in df.columns:
                print(f"[跳过] {path} 缺少 compound/gene 列")
                continue
            conf_col = "confidence_level" if "confidence_level" in df.columns else None
            cols = ["compound", "gene"] + ([conf_col] if conf_col else [])
            valid = df[cols].dropna()
            for idx, r in valid.iterrows():
                comp = normalize_name(r["compound"])
                comp_norm = normalize_compound(r["compound"])
                gene = normalize_gene(r["gene"])
                weight = _confidence_to_weight(r[conf_col]) if conf_col else DEFAULT_WEIGHT
                rows.append({
                    "compound": comp,
                    "compound_norm": comp_norm,
                    "gene": gene,
                    "source": source,
                    "weight": weight,
                })
                known_compounds.add(comp)
                known_compounds.add(comp_norm)
                known_genes.add(gene)
            print(f"[加载] {source}: {len(valid)} 条证据")
        except Exception:
            print(f"[警告] 加载 {path} 失败:")
            traceback.print_exc()

    if not rows:
        return (
            pd.DataFrame(columns=["compound", "compound_norm", "gene", "source", "weight"]),
            known_compounds,
            known_genes,
        )

    return pd.DataFrame(rows), known_compounds, known_genes


def _aggregate_pair_evidence(sub_df: pd.DataFrame) -> tuple[float, list[str], dict[str, float]]:
    """聚合同一 pair 的多源证据，返回加权分数、来源列表、来源权重字典."""
    if sub_df.empty:
        return 0.0, [], {}
    source_weights = (
        sub_df.groupby("source")["weight"]
        .max()
        .to_dict()
    )
    total_score = sum(source_weights.values())
    sources = sorted(source_weights.keys())
    return total_score, sources, source_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 top-k 预测的外部数据库支持")
    parser.add_argument("--experiment", type=str, default="hgt_compare_seed42")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="证据加权分阈值，>= 该值视为有支持")
    args = parser.parse_args()

    exp_dir = Path("L3_results") / args.experiment
    pred_path = exp_dir / "top100_novel_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")

    pred_df = pd.read_csv(pred_path)
    required_cols = {"compound_name", "gene_name", "score"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"预测文件缺少列: {missing}")

    pred_df["compound_key"] = pred_df["compound_name"].apply(normalize_name)
    pred_df["compound_norm"] = pred_df["compound_name"].apply(normalize_compound)
    pred_df["gene_key"] = pred_df["gene_name"].apply(normalize_gene)

    evidence_df, known_compounds, known_genes = load_external_pairs()

    # 为每个预测构建 pair 证据
    pair_scores: list[float] = []
    pair_sources: list[list[str]] = []
    pair_source_weights: list[dict[str, float]] = []
    for _, row in pred_df.iterrows():
        aliases = expand_compound_name(row["compound_name"])
        mask = (
            evidence_df["compound"].isin(aliases) | evidence_df["compound_norm"].isin(aliases)
        ) & (evidence_df["gene"] == row["gene_key"])
        score, sources, weights = _aggregate_pair_evidence(evidence_df.loc[mask])
        pair_scores.append(score)
        pair_sources.append(sources)
        pair_source_weights.append(weights)

    pred_df["evidence_score"] = pair_scores
    pred_df["pair_supported"] = pred_df["evidence_score"] >= args.score_threshold
    pred_df["external_sources"] = pair_sources
    pred_df["source_weights"] = pair_source_weights

    def compound_has_evidence(row: pd.Series) -> bool:
        aliases = expand_compound_name(row["compound_name"])
        return bool(aliases & known_compounds)

    def gene_has_evidence(row: pd.Series) -> bool:
        return row["gene_key"] in known_genes

    pred_df["compound_known"] = pred_df.apply(compound_has_evidence, axis=1)
    pred_df["gene_known"] = pred_df.apply(gene_has_evidence, axis=1)

    total = len(pred_df)
    pair_supported = int(pred_df["pair_supported"].sum())
    compound_known = int(pred_df["compound_known"].sum())
    gene_known = int(pred_df["gene_known"].sum())

    supported_df = pred_df[pred_df["pair_supported"]].copy()

    # 保存完整验证结果
    output_path = exp_dir / "top100_validation.csv"
    pred_df[[
        "compound_name", "gene_name", "score",
        "pair_supported", "evidence_score", "compound_known", "gene_known",
        "external_sources", "source_weights",
    ]].to_csv(output_path, index=False)
    print(f"Saved validation results: {output_path}")

    # 按证据源统计命中次数（按预测条数）
    source_counts: dict[str, int] = {}
    for sources in pred_df["external_sources"]:
        for src in sources:
            source_counts[src] = source_counts.get(src, 0) + 1

    # 按证据源统计加权分数
    source_weight_totals: dict[str, float] = {}
    for weights in pred_df["source_weights"]:
        for src, w in weights.items():
            source_weight_totals[src] = source_weight_totals.get(src, 0.0) + w

    summary = {
        "experiment": args.experiment,
        "score_threshold": args.score_threshold,
        "total_predictions": total,
        "pair_supported": pair_supported,
        "pair_support_rate": round(pair_supported / total, 4) if total else 0.0,
        "compound_known": compound_known,
        "compound_known_rate": round(compound_known / total, 4) if total else 0.0,
        "gene_known": gene_known,
        "gene_known_rate": round(gene_known / total, 4) if total else 0.0,
        "source_counts": source_counts,
        "source_weight_totals": {k: round(v, 3) for k, v in source_weight_totals.items()},
        "supported_predictions": supported_df.head(20)[[
            "compound_name", "gene_name", "score", "evidence_score", "external_sources"
        ]].to_dict("records") if not supported_df.empty else [],
    }
    summary_path = exp_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    print(
        f"\n验证结果（加权 pair 匹配，阈值={args.score_threshold}）: "
        f"{pair_supported}/{total} 条 top-100 新预测在本地外部数据库中找到证据 "
        f"(支持率 {summary['pair_support_rate']:.2%})"
    )
    print(
        f"化合物层面: {compound_known}/{total} "
        f"({summary['compound_known_rate']:.2%}) 的化合物在本地数据库中有任何靶点记录"
    )
    print(
        f"基因层面: {gene_known}/{total} ({summary['gene_known_rate']:.2%}) 的基因在本地数据库中被任何化合物靶向"
    )
    if source_counts:
        print("\n各证据源命中数:")
        for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"  - {src}: {cnt} 次")
    if supported_df.empty:
        print("\n没有 top-100 预测在本地外部数据库中获得 pair-level 支持。")
    else:
        print("\n有外部证据支持的 top 预测:")
        print(supported_df.head(20)[[
            "compound_name", "gene_name", "score", "evidence_score", "external_sources"
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
