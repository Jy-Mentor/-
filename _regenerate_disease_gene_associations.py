"""重新生成 disease_gene_associations.csv.

来源:
- AD/Aging: DisGeNET curated 数据 (github mirror)
- CIRI: L3/L1_genome_wide_de.csv 元分析差异基因
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# 1. 读取 DisGeNET 真实数据
disgenet = pd.read_csv(ROOT / 'network_files' / 'disgenet_disease_genes.csv')
# 筛选 AD 和 Aging
ad_aging = disgenet[disgenet['disease'].isin(['AD', 'Aging'])].copy()
ad_aging['source'] = 'DisGeNET_curated_github_mirror'
ad_aging['confidence'] = ad_aging['score']
ad_aging['download_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

# 2. 从 L1 DE 提取 CIRI 基因
de = pd.read_csv(ROOT / 'L3' / 'L1_genome_wide_de.csv')
sig = de[(de['padj'] < 0.05) & (de['log2FC'].abs() > 0.5)].copy()
gene_ds_counts = sig.groupby('gene')['dataset'].nunique()
ciri_genes = gene_ds_counts[gene_ds_counts >= 2].index.tolist()

ciri_rows = []
for gene in ciri_genes:
    ciri_rows.append({
        'disease': 'CIRI',
        'gene': gene.upper(),
        'disease_name': 'cerebral ischemia-reperfusion injury (GEO DE meta)',
        'disease_id': 'NA',
        'score': 0.0,
        'source': 'GEO_DE_meta_analysis',
        'confidence': 0.7,
        'download_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    })
ciri_df = pd.DataFrame(ciri_rows)

# 3. 合并并选择所需列
combined = pd.concat([
    ad_aging[['disease', 'gene', 'disease_name', 'disease_id', 'score', 'source', 'confidence', 'download_date']],
    ciri_df
], ignore_index=True)

# 只保留核心基因集中的基因
core_genes = set(pd.read_csv(ROOT / '铁衰老基因.txt', header=None)[0].str.strip().str.upper())
combined = combined[combined['gene'].isin(core_genes)].copy()

# 去重
combined = combined.drop_duplicates(subset=['disease', 'gene']).sort_values(['disease', 'gene'])

# 4. 保存
out_path = ROOT / 'network_files' / 'disease_gene_associations.csv'
combined.to_csv(out_path, index=False)
print(f"已生成 {out_path}: {len(combined)} 条关联")
print(combined['disease'].value_counts())
