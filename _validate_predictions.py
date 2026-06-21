"""对 top-k compound-target 预测进行外部数据库验证.

支持的证据源（按本地已有网络文件）:
- compound_target_edges.csv / compound_target_edges_curated.csv
- bindingdb_compound_targets.csv
- chembl_compound_targets_cleaned.csv
- ctd_compound_targets.csv
- drugbank_compound_targets.csv
- dgidb_compound_targets.csv
- drugtargetcommons_compound_targets.csv

验证维度:
1. pair-level: 化合物-基因对是否已在本地数据库中出现（精确匹配）
2. compound-level: 该化合物是否被本地数据库记录过任何靶点
3. gene-level: 该基因是否在本地数据库中被任何化合物靶向
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

# 常见化合物别名，仅用于扩展本地数据库的精确匹配能力。
# 别名来源于项目中实际出现的网络文件记录（如 CTD 使用 caryophyllene，
# 部分数据库使用 β-caryophyllene 等）。
COMPOUND_ALIASES: dict[str, list[str]] = {
    "BCP": ["BCP", "CARYOPHYLLENE", "BETA-CARYOPHYLLENE", "Β-CARYOPHYLLENE"],
    "BETA-CARYOPHYLLENE": ["BCP", "CARYOPHYLLENE", "BETA-CARYOPHYLLENE", "Β-CARYOPHYLLENE"],
    "Β-CARYOPHYLLENE": ["BCP", "CARYOPHYLLENE", "BETA-CARYOPHYLLENE", "Β-CARYOPHYLLENE"],
    "VC": ["VC", "VITAMIN C", "ASCORBIC ACID", "ASCORBATE"],
    "VITAMIN C": ["VC", "VITAMIN C", "ASCORBIC ACID", "ASCORBATE"],
    "N-ACETYLCYSTEINE": ["N-ACETYLCYSTEINE", "NAC", "N-ACETYL CYSTEINE"],
    "NAC": ["N-ACETYLCYSTEINE", "NAC", "N-ACETYL CYSTEINE"],
}


def normalize_name(s: str) -> str:
    """统一名称格式用于匹配."""
    return str(s).strip().upper()


def expand_compound_name(name: str) -> set[str]:
    """扩展化合物名称为可能的别名集合."""
    normalized = normalize_name(name)
    aliases = {normalized}
    for key, alias_list in COMPOUND_ALIASES.items():
        if normalize_name(key) == normalized:
            aliases.update(normalize_name(a) for a in alias_list)
    return aliases


def load_external_pairs() -> tuple[pd.DataFrame, set[str], set[str]]:
    """加载所有本地化合物-靶点证据，并返回 pair 表、已知化合物集合、已知基因集合."""
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
            valid = df[["compound", "gene"]].dropna()
            for _, r in valid.iterrows():
                comp = normalize_name(r["compound"])
                gene = normalize_name(r["gene"])
                rows.append({
                    "compound": comp,
                    "gene": gene,
                    "source": source,
                })
                known_compounds.add(comp)
                known_genes.add(gene)
            print(f"[加载] {source}: {len(valid)} 条证据")
        except Exception:
            print(f"[警告] 加载 {path} 失败:")
            traceback.print_exc()

    if not rows:
        return (
            pd.DataFrame(columns=["compound", "gene", "source"]),
            known_compounds,
            known_genes,
        )

    return pd.DataFrame(rows), known_compounds, known_genes


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 top-k 预测的外部数据库支持")
    parser.add_argument("--experiment", type=str, default="hgt_compare_seed42")
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
    pred_df["gene_key"] = pred_df["gene_name"].apply(normalize_name)

    evidence_df, known_compounds, known_genes = load_external_pairs()

    def find_pair_sources(row: pd.Series) -> list[str]:
        aliases = expand_compound_name(row["compound_name"])
        mask = evidence_df["compound"].isin(aliases) & (evidence_df["gene"] == row["gene_key"])
        return evidence_df.loc[mask, "source"].unique().tolist()

    def compound_has_evidence(row: pd.Series) -> bool:
        aliases = expand_compound_name(row["compound_name"])
        return bool(aliases & known_compounds)

    def gene_has_evidence(row: pd.Series) -> bool:
        return row["gene_key"] in known_genes

    pred_df["external_sources"] = pred_df.apply(find_pair_sources, axis=1)
    pred_df["pair_supported"] = pred_df["external_sources"].apply(lambda x: len(x) > 0)
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
        "pair_supported", "compound_known", "gene_known", "external_sources"
    ]].to_csv(output_path, index=False)
    print(f"Saved validation results: {output_path}")

    # 按证据源统计
    source_counts: dict[str, int] = {}
    for sources in pred_df["external_sources"]:
        for src in sources:
            source_counts[src] = source_counts.get(src, 0) + 1

    summary = {
        "experiment": args.experiment,
        "total_predictions": total,
        "pair_supported": pair_supported,
        "pair_support_rate": round(pair_supported / total, 4) if total else 0.0,
        "compound_known": compound_known,
        "compound_known_rate": round(compound_known / total, 4) if total else 0.0,
        "gene_known": gene_known,
        "gene_known_rate": round(gene_known / total, 4) if total else 0.0,
        "source_counts": source_counts,
        "supported_predictions": supported_df.head(20)[[
            "compound_name", "gene_name", "score", "external_sources"
        ]].to_dict("records") if not supported_df.empty else [],
    }
    summary_path = exp_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    print(
        f"\n验证结果（精确 pair 匹配）: {pair_supported}/{total} 条 top-100 新预测在本地外部数据库中找到证据 "
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
            print(f"  - {src}: {cnt}")
    if not supported_df.empty:
        print("\n有外部证据支持的 top 预测:")
        print(supported_df.head(20)[["compound_name", "gene_name", "score", "external_sources"]].to_string(index=False))


if __name__ == "__main__":
    main()
