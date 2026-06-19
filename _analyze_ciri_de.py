"""分析 L3/L1_genome_wide_de.csv，提取 CIRI 差异表达基因并检查与铁衰老基因集的overlap。

数据来源:
  - L3/L1_genome_wide_de.csv: L1 模块对 5 个脑缺血/卒中数据集 (GSE16561, GSE37587,
    GSE61616, GSE97537, GSE104036) 的差分表达分析结果。
  - 铁衰老基因.txt: 项目铁死亡/衰老核心基因集。

筛选标准:
  - padj < 0.05
  - |log2FC| > 0.5
  - 在至少 1 个数据集中显著 (可调整为 ≥2 个以提高稳健性)
"""
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent

de = pd.read_csv(BASE_DIR / "L3" / "L1_genome_wide_de.csv")
print("Datasets:", de["dataset"].unique())
print("Total DE rows:", len(de))
print("Unique genes:", de["gene"].nunique())

# 显著性过滤
sig = de[(de["padj"] < 0.05) & (de["log2FC"].abs() > 0.5)].copy()
print("Significant rows:", len(sig))
print("Significant unique genes:", sig["gene"].nunique())

# 每个基因在多少个数据集中显著
gene_counts = sig.groupby("gene")["dataset"].nunique().reset_index()
gene_counts.columns = ["gene", "n_datasets"]
gene_counts = gene_counts.sort_values(["n_datasets", "gene"], ascending=[False, True])
print("\nTop recurring DE genes:")
print(gene_counts.head(30).to_string(index=False))

# 铁衰老基因集
ferroaging_file = BASE_DIR / "铁衰老基因.txt"
ferro = set()
with open(ferroaging_file, "r", encoding="utf-8") as f:
    for line in f:
        g = line.strip()
        if g:
            ferro.add(g.upper())
print(f"\nFerroaging genes: {len(ferro)}")

# 交集
de_genes = set(gene_counts["gene"].tolist())
overlap = de_genes & ferro
print(f"DE genes overlapping with ferroaging: {len(overlap)}")
print("Overlapping genes:", sorted(overlap))

# 铁衰老基因中的 DE 汇总
ferro_de = sig[sig["gene"].isin(ferro)].copy()
ferro_de["abs_l2fc"] = ferro_de["log2FC"].abs()
agg = (
    ferro_de.groupby("gene")
    .agg(
        n_datasets=("dataset", "nunique"),
        mean_log2FC=("log2FC", "mean"),
        max_abs_log2FC=("abs_l2fc", "max"),
        min_padj=("padj", "min"),
    )
    .sort_values(["n_datasets", "min_padj"], ascending=[False, True])
)
print("\nFerroaging DE summary:")
print(agg.to_string())

# 不同重现性阈值下的交集数量
print("\nReproducibility thresholds:")
for min_ds in [1, 2, 3, 4, 5]:
    ds_genes = set(gene_counts[gene_counts["n_datasets"] >= min_ds]["gene"])
    ds_overlap = ds_genes & ferro
    print(f"  >= {min_ds} datasets: {len(ds_genes)} DE genes, {len(ds_overlap)} in ferroaging")
