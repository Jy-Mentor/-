"""清洗 PanglaoDB 细胞类型标记基因文件.

输入: network_files/panglaodb_celltype_markers.csv
输出: network_files/panglaodb_celltype_markers_cleaned.csv

清洗规则:
1. 去除 gene 为占位符或非基因符号的行(如 PROTEIN-CODING GENE)。
2. 去除空值、重复行。
3. 优先保留人类(Hs/Homo sapiens)数据。
4. 标准化基因符号为大写。
5. 为每行记录 source= PanglaoDB 与清洗日期。
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "network_files" / "panglaodb_celltype_markers.csv"
OUTPUT_CSV = PROJECT_ROOT / "network_files" / "panglaodb_celltype_markers_cleaned.csv"
METADATA_JSON = PROJECT_ROOT / "external_data" / "panglaodb_cleaned_metadata.json"

# 已知占位符/非基因符号
PLACEHOLDERS = {
    "PROTEIN-CODING GENE", "PROTEIN CODING GENE", "ANTISENSE LNCRNA GENE",
    "NON-CODING RNA", "LINCRNA GENE", "PREDICTED GENE", "PSEUDOGENE",
    "MIRNA GENE", "SNORNA GENE", "TRNA GENE", "RRNA GENE",
}


def is_valid_gene_symbol(g: str) -> bool:
    """判断是否为合理基因符号."""
    if pd.isna(g):
        return False
    g = str(g).strip().upper()
    if len(g) == 0 or len(g) > 20:
        return False
    if " " in g:
        return False
    if re.search(r"[-/().,;:]", g):
        return False
    if g in PLACEHOLDERS:
        return False
    # 允许大写字母+数字
    if not re.match(r"^[A-Z0-9]+$", g):
        return False
    return True


def is_human(species: str) -> bool:
    s = str(species).strip().lower()
    return "hs" in s or "homo" in s or "human" in s


def main() -> int:
    if not INPUT_CSV.exists():
        logger.error("输入文件不存在: %s", INPUT_CSV)
        return 1

    df = pd.read_csv(INPUT_CSV)
    logger.info("原始 PanglaoDB 记录: %d", len(df))

    # 标准化列名
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"celltype", "gene"}
    if not required.issubset(set(df.columns)):
        logger.error("缺少必要列: %s", required - set(df.columns))
        return 1

    # 清洗 gene
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    df = df[df["gene"].apply(is_valid_gene_symbol)]
    logger.info("去除占位符后: %d", len(df))

    # 优先人类
    has_species = "species" in df.columns
    if has_species:
        df["is_human"] = df["species"].apply(is_human)
        # 先按人类排序, 再去重保留人类优先
        df = df.sort_values("is_human", ascending=False)

    # 去重
    subset = ["celltype", "gene"]
    df = df.drop_duplicates(subset=subset, keep="first")
    logger.info("去重后: %d", len(df))

    # 选择输出列
    out_cols = ["celltype", "gene"]
    if has_species:
        out_cols.append("species")
    out_df = df[out_cols].copy()
    out_df["source"] = "PanglaoDB"
    out_df = out_df.sort_values(["celltype", "gene"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("已写入清洗后文件: %s (%d 条)", OUTPUT_CSV, len(out_df))

    metadata = {
        "source": "PanglaoDB",
        "input_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "clean_date": pd.Timestamp.now().isoformat(),
        "stats": {
            "raw_rows": len(pd.read_csv(INPUT_CSV)),
            "cleaned_rows": len(out_df),
            "unique_celltypes": int(out_df["celltype"].nunique()),
            "unique_genes": int(out_df["gene"].nunique()),
        },
    }
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已写入元数据: %s", METADATA_JSON)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
