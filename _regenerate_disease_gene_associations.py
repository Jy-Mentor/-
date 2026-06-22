"""重新生成 disease_gene_associations.csv.

来源:
- AD/Aging: DisGeNET curated 数据 (github mirror)
- CIRI-DisGeNET: DisGeNET 中 stroke / brain ischemia / cerebral infarction 条目,
  作为 CIRI 的疾病-基因关联
- CIRI-GEO: L3/L1_genome_wide_de.csv 元分析差异基因,
  保持严格阈值 (padj<0.05, |log2FC|>0.5, >=2 数据集)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent

# 1. 读取 DisGeNET 真实数据
disgenet = pd.read_csv(ROOT / 'network_files' / 'disgenet_disease_genes.csv')

# 1a. AD/Aging (原有)
ad_aging = disgenet[disgenet['disease'].isin(['AD', 'Aging'])].copy()
ad_aging['source'] = 'DisGeNET_curated_github_mirror'
ad_aging['confidence'] = ad_aging['score']
ad_aging['download_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

# 1b. CIRI from DisGeNET: stroke / brain ischemia / cerebral infarction 视为 CIRI 相关
#     不降低 DisGeNET curated 阈值, 仅排除缺失 score 的记录
ciri_disgenet = disgenet[disgenet['disease'] == 'CIRI'].copy()
ciri_disgenet = ciri_disgenet[ciri_disgenet['score'].notna() & (ciri_disgenet['score'] > 0)].copy()
ciri_disgenet['source'] = 'DisGeNET_curated_github_mirror_CIRI'
ciri_disgenet['confidence'] = ciri_disgenet['score']
ciri_disgenet['download_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')

# 2. 从 L1 DE 提取 CIRI 基因 (严格阈值不变)
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
    ciri_disgenet[['disease', 'gene', 'disease_name', 'disease_id', 'score', 'source', 'confidence', 'download_date']],
    ciri_df
], ignore_index=True)

# 只保留核心基因集中的基因
core_genes = set(pd.read_csv(ROOT / '铁衰老基因.txt', header=None)[0].str.strip().str.upper())
combined = combined[combined['gene'].isin(core_genes)].copy()

# 去重: 同一疾病-基因保留置信度最高的一条
combined = combined.sort_values('confidence', ascending=False)
combined = combined.drop_duplicates(subset=['disease', 'gene']).sort_values(['disease', 'gene'])

def main() -> int:
    # 4. 保存
    out_path = ROOT / 'network_files' / 'disease_gene_associations.csv'
    combined.to_csv(out_path, index=False)
    print(f"已生成 {out_path}: {len(combined)} 条关联")
    print(combined['disease'].value_counts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
