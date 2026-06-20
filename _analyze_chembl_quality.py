"""分析 ChEMBL 化合物-靶点数据质量."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CSV_PATH = Path("network_files/chembl_compound_targets.csv")
GENE_LIST_PATH = Path("铁衰老基因.txt")


def load_gene_set(path: Path) -> set[str]:
    genes: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            g = line.strip().upper()
            if g:
                genes.add(g)
    return genes


def is_suspicious_gene_name(g) -> bool:
    """判断基因名称是否可疑（非标准HGNC符号）."""
    if pd.isna(g):
        return True
    g = str(g).strip()
    if len(g) > 15:
        return True
    if " " in g:
        return True
    if any(c.isdigit() for c in g):
        return True
    if not g.isupper():
        return True
    if re.search(r"[-/().,;:]", g):
        return True
    return False


def main() -> int:
    df = pd.read_csv(CSV_PATH)
    core_genes = load_gene_set(GENE_LIST_PATH)

    print(f"总行数: {len(df)}")
    print(f"化合物数: {df['compound'].nunique()}")
    print(f"唯一 gene 数: {df['gene'].nunique()}")
    print(f"在 98 铁衰老基因集中的比例: {df['gene'].isin(core_genes).mean():.2%}")

    suspicious = df["gene"].apply(is_suspicious_gene_name)
    print(f"\n可疑基因名比例: {suspicious.mean():.2%} ({suspicious.sum()}/{len(df)})")

    print("\n前50个唯一 gene:")
    unique_genes = [g for g in df["gene"].unique() if pd.notna(g)]
    for i, g in enumerate(sorted(unique_genes)[:50], 1):
        marker = "*" if is_suspicious_gene_name(g) else " "
        print(f"  {marker}{i:2d}. {g}")

    print("\n示例可疑条目:")
    sample = df[suspicious][["compound", "gene", "target_chembl_id", "standard_type", "pchembl_value"]].head(30)
    print(sample.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
