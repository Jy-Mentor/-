"""将 L1 DE 提取的 CIRI 相关基因追加到 disease_gene_associations.csv。

本脚本仅用于数据持久化; module3_hgt.py 运行时会自动从 L1_genome_wide_de.csv
重新提取 CIRI-gene 关联。
"""
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
CSV_FILE = BASE_DIR / "network_files" / "disease_gene_associations.csv"
DE_FILE = BASE_DIR / "L3" / "L1_genome_wide_de.csv"

# 加载现有疾病-基因关联
if CSV_FILE.exists():
    df_existing = pd.read_csv(CSV_FILE)
else:
    df_existing = pd.DataFrame(columns=["disease", "gene"])

# 加载并过滤 DE 数据
de_df = pd.read_csv(DE_FILE)
sig = de_df[(de_df["padj"] < 0.05) & (de_df["log2FC"].abs() > 0.5)].copy()
gene_ds_counts = sig.groupby("gene")["dataset"].nunique()
ciri_genes = gene_ds_counts[gene_ds_counts >= 2].index.tolist()

# 追加 CIRI 行 (去重)
new_rows = [{"disease": "CIRI", "gene": g.upper()} for g in ciri_genes]
df_new = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True)
df_new["gene"] = df_new["gene"].str.upper()
df_new = df_new.drop_duplicates()

df_new.to_csv(CSV_FILE, index=False)
print(f"已更新 {CSV_FILE.name}: {len(df_new)} 行")
print(f"  - CIRI 基因数: {len(ciri_genes)}")
print(f"  - 疾病分布:\n{df_new['disease'].value_counts().to_string()}")
