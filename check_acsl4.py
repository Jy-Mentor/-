import pandas as pd
from pathlib import Path

# 检查ACSL4的连接情况
hub_file = Path('L3_results/L3_hub_gene_ranking.csv')
df = pd.read_csv(hub_file)

print('=== ACSL4 详细分析 ===')
acsl4 = df[df['gene'] == 'ACSL4'].iloc[0]
print(f"排名: {acsl4['rank']}/285")
print(f"Hub Score: {acsl4['hub_score']:.4f}")
print(f"Embedding Norm: {acsl4['embedding_norm']:.4f}")
print(f"Degree: {acsl4['degree']}")
print(f"Bio Prior: {acsl4['bio_prior']}")

print()
print('=== Top 5 基因对比 ===')
for _, row in df.head(5).iterrows():
    print(f"{row['gene']}: hub={row['hub_score']:.3f}, emb_norm={row['embedding_norm']:.3f}, deg={row['degree']}, bio={row['bio_prior']}")

print()
print('=== 铁死亡核心基因排名 ===')
ferro_genes = ['ACSL4', 'GPX4', 'TFRC', 'SLC7A11', 'NFE2L2', 'KEAP1', 'HMOX1', 'PTGS2']
for g in ferro_genes:
    row = df[df['gene'] == g]
    if len(row) > 0:
        r = row.iloc[0]
        print(f"{g}: rank={r['rank']}, hub={r['hub_score']:.3f}, deg={r['degree']}")
