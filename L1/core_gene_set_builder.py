#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心基因集整理脚本
=====================================================================
整合来源：
  1. IDSP 基因集 (idsp_gene_sets.py)
     - PURE_FERROPTOSIS (72)  — FerrDb V2 铁死亡核心
     - PURE_SENESCENCE  (140) — CellAge + SenMayo 衰老核心
     - SHARED_GENES     (12)  — 铁死亡 ∩ 衰老 桥接基因
  2. 铁衰老基因.txt (96)      — 铁衰老候选基因列表
  3. L1 跨数据集分析
     - L1_gene_level_analysis.csv   — 5个数据集差异表达
     - L1_rra_gene_consistency.csv  — RRA跨数据集一致性
     - L1_dual_comparison_summary.csv — 双评分统计

输出：
  - L1/core_gene_set.csv          — 完整核心基因集（含分类与证据层级）
=====================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "l1_results"

# ================================================================
# 1. 加载 IDSP 基因集
# ================================================================
from idsp_gene_sets import PURE_FERROPTOSIS, PURE_SENESCENCE, SHARED_GENES

# ================================================================
# 2. 加载铁衰老基因.txt
# ================================================================
iron_aging_path = BASE_DIR.parent / "铁衰老基因.txt"
with open(iron_aging_path, "r", encoding="utf-8") as f:
    IRON_AGING_GENES = {line.strip() for line in f if line.strip()}
print(f"铁衰老基因.txt: {len(IRON_AGING_GENES)} genes")

# ================================================================
# 3. 加载 L1 分析结果
# ================================================================
df_gene = pd.read_csv(RESULTS_DIR / "L1_gene_level_analysis.csv")
df_rra = pd.read_csv(RESULTS_DIR / "L1_rra_gene_consistency.csv")

# 计算每个基因在多少个数据集中显著 (padj < 0.05)
gene_sig = df_gene[df_gene["padj"] < 0.05].groupby("gene")["dataset"].nunique()
gene_sig.name = "n_sig_datasets"

# 计算每个基因在所有数据集中的平均 |log2FC|
gene_avg_fc = df_gene.groupby("gene")["log2FC"].apply(lambda x: np.mean(np.abs(x)))
gene_avg_fc.name = "mean_abs_log2FC"

# RRA consistency
rra_dict = df_rra.set_index("gene")[["n_datasets", "mean_normalized_rank", "rra_padj"]].to_dict("index")

# ================================================================
# 4. 构建核心基因集
# ================================================================

def classify_gene(gene):
    """分类基因：纯铁死亡 / 纯衰老 / 共享 / 铁衰老候选"""
    categories = []

    in_ferr = gene in PURE_FERROPTOSIS
    in_sene = gene in PURE_SENESCENCE
    in_shared = gene in SHARED_GENES
    in_iron_aging = gene in IRON_AGING_GENES

    if in_shared:
        categories.append("桥接基因")
    if in_ferr:
        categories.append("纯铁死亡")
    if in_sene:
        categories.append("纯衰老")
    if in_iron_aging:
        categories.append("铁衰老候选")
    # 不在任何IDSP中的铁衰老候选
    if in_iron_aging and not in_ferr and not in_sene and not in_shared:
        categories.append("铁衰老独有")

    return " | ".join(categories) if categories else "其他"


def calc_evidence_level(gene):
    """
    证据层级：
      Tier1: IDSP + 铁衰老 + L1多数据集显著
      Tier2: IDSP + 铁衰老（双证据）
      Tier3: IDSP 或 铁衰老 任一 + L1显著
      Tier4: 仅IDSP或仅铁衰老
    """
    in_idsp = gene in PURE_FERROPTOSIS or gene in PURE_SENESCENCE or gene in SHARED_GENES
    in_iron = gene in IRON_AGING_GENES
    sig_count = gene_sig.get(gene, 0)

    if in_idsp and in_iron and sig_count >= 3:
        return "Tier1_核心"
    elif in_idsp and in_iron:
        return "Tier2_双验证"
    elif (in_idsp or in_iron) and sig_count >= 2:
        return "Tier3_L1支持"
    elif in_idsp or in_iron:
        return "Tier4_单证据"
    else:
        return "Tier5_其他"


# 收集所有基因
all_genes = set()
all_genes.update(PURE_FERROPTOSIS)
all_genes.update(PURE_SENESCENCE)
all_genes.update(SHARED_GENES)
all_genes.update(IRON_AGING_GENES)

records = []
for gene in sorted(all_genes):
    records.append({
        "gene": gene,
        "category": classify_gene(gene),
        "in_ferroptosis": gene in PURE_FERROPTOSIS,
        "in_senescence": gene in PURE_SENESCENCE,
        "in_shared": gene in SHARED_GENES,
        "in_iron_aging_list": gene in IRON_AGING_GENES,
        "n_sig_datasets_L1": gene_sig.get(gene, 0),
        "mean_abs_log2FC": round(gene_avg_fc.get(gene, np.nan), 3) if gene in gene_avg_fc.index else np.nan,
        "rra_padj": round(rra_dict[gene]["rra_padj"], 4) if gene in rra_dict else np.nan,
        "evidence_tier": calc_evidence_level(gene),
    })

df_core = pd.DataFrame(records)

# 按证据层级排序
tier_order = {"Tier1_核心": 0, "Tier2_双验证": 1, "Tier3_L1支持": 2, "Tier4_单证据": 3, "Tier5_其他": 4}
df_core["_tier_sort"] = df_core["evidence_tier"].map(tier_order)
df_core = df_core.sort_values(["_tier_sort", "n_sig_datasets_L1", "gene"],
                                ascending=[True, False, True]).drop(columns=["_tier_sort"])

# ================================================================
# 5. 保存
# ================================================================
output_path = BASE_DIR / "core_gene_set.csv"
df_core.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n核心基因集已保存: {output_path}")
print(f"总计: {len(df_core)} 个基因")

# ================================================================
# 6. 汇总统计
# ================================================================
print("\n" + "=" * 60)
print("  核心基因集汇总统计")
print("=" * 60)

for tier in ["Tier1_核心", "Tier2_双验证", "Tier3_L1支持", "Tier4_单证据", "Tier5_其他"]:
    subset = df_core[df_core["evidence_tier"] == tier]
    if len(subset) > 0:
        print(f"\n[{tier}] ({len(subset)} genes):")
        genes_str = ", ".join(subset["gene"].tolist())
        if len(genes_str) > 120:
            print(f"  {genes_str[:120]}...")
        else:
            print(f"  {genes_str}")

print("\n" + "=" * 60)
print("  分类统计")
print("=" * 60)
cat_counts = df_core["category"].value_counts()
for cat, cnt in cat_counts.items():
    print(f"  {cat}: {cnt}")

print(f"\nIDSP 基因集: 铁死亡={len(PURE_FERROPTOSIS)}, 衰老={len(PURE_SENESCENCE)}, 共享={len(SHARED_GENES)}")
print(f"铁衰老基因.txt: {len(IRON_AGING_GENES)}")
print(f"整合后核心基因集: {len(df_core)}")
