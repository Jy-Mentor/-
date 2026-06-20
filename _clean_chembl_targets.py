"""清洗 ChEMBL 化合物-靶点数据.

问题: fetch_chembl_compound_targets.py 在无法解析 target component 的 GENE_SYMBOL 时,
      回退到 target_pref_name(蛋白描述/别名/细胞系), 导致 79.6% 的 gene 字段不是标准基因符号.

策略:
1. 用 mygene.info 将别名/描述映射到 HGNC 官方符号.
2. 仅保留成功映射到官方符号且落在 98 铁衰老核心基因集内的边.
3. 同一 compound-gene 保留 pchembl 最高记录; 无 pchembl 时按 standard_value 排序.
4. 输出带 source/confidence/download_date 的清洗后文件.

输入: network_files/chembl_compound_targets.csv
输出: network_files/chembl_compound_targets_cleaned.csv
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import mygene
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "network_files" / "chembl_compound_targets.csv"
OUTPUT_CSV = PROJECT_ROOT / "network_files" / "chembl_compound_targets_cleaned.csv"
METADATA_JSON = PROJECT_ROOT / "external_data" / "chembl_cleaned_metadata.json"
GENE_LIST_PATH = PROJECT_ROOT / "铁衰老基因.txt"

# 置信度阈值
PChEMBL_HIGH = 7.0   # <100 nM
PChEMBL_MED = 6.0    # <1 uM
PChEMBL_LOW = 5.0    # <10 uM


def load_gene_set(path: Path) -> set[str]:
    genes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


def normalize_gene_name(g: str) -> str:
    """标准化基因名输入, 去掉常见噪声."""
    if pd.isna(g):
        return ""
    g = str(g).strip().upper()
    # 如果整串包含空格且很长, 多半是描述, 仍然交给 mygene 尝试
    return g


def query_official_symbols(gene_names: set[str]) -> dict[str, str | None]:
    """用 mygene.info 批量查询基因名 -> 官方 HGNC 符号."""
    mg = mygene.MyGeneInfo()
    unique_names = sorted(g for g in gene_names if g)
    logger.info("向 mygene.info 查询 %d 个唯一基因名...", len(unique_names))

    mapping: dict[str, str | None] = {}
    batch_size = 1000
    for i in range(0, len(unique_names), batch_size):
        batch = unique_names[i : i + batch_size]
        try:
            res = mg.querymany(
                batch,
                scopes="symbol,alias,name,ensembl.protein,ensembl.gene,uniprot",
                fields="symbol,HGNC",
                species="human",
                verbose=False,
                as_dataframe=False,
            )
            for item in res:
                q = str(item.get("query", "")).strip().upper()
                sym = item.get("symbol")
                if sym:
                    sym = str(sym).strip().upper()
                # 优先保存首次命中的官方符号
                if q and q not in mapping:
                    mapping[q] = sym
        except Exception:
            logger.warning("mygene.info 查询批次失败 (idx=%d)", i)
            traceback.print_exc()

    n_mapped = sum(1 for v in mapping.values() if v)
    logger.info("mygene.info 映射完成: %d/%d 成功", n_mapped, len(unique_names))
    return mapping


def assign_confidence(pchembl: float | None, standard_value: float | None) -> tuple[float, str]:
    """根据活性数据计算置信度分数和等级."""
    if pd.notna(pchembl):
        p = float(pchembl)
        if p >= PChEMBL_HIGH:
            return 0.95, "high"
        if p >= PChEMBL_MED:
            return 0.80, "medium"
        if p >= PChEMBL_LOW:
            return 0.60, "low"
        return 0.30, "very_low"

    if pd.notna(standard_value):
        sv = float(standard_value)
        if sv <= 100:
            return 0.90, "high"
        if sv <= 1000:
            return 0.75, "medium"
        if sv <= 10000:
            return 0.55, "low"
        return 0.25, "very_low"

    return 0.10, "unknown"


def main() -> int:
    core_genes = load_gene_set(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    df = pd.read_csv(INPUT_CSV)
    logger.info("读取 ChEMBL 原始记录: %d 条", len(df))

    # 1. 标准化 gene 字段
    df["gene_raw"] = df["gene"].astype(str).str.strip().str.upper()

    # 2. 用 mygene.info 解析官方符号
    unique_raw_genes = set(df["gene_raw"].unique())
    raw_to_official = query_official_symbols(unique_raw_genes)
    df["gene_official"] = df["gene_raw"].map(raw_to_official)

    n_mapped_total = df["gene_official"].notna().sum()
    logger.info("成功映射到官方符号: %d/%d 条", n_mapped_total, len(df))

    # 3. 仅保留官方符号且在核心基因集内的记录
    df_clean = df[df["gene_official"].isin(core_genes)].copy()
    logger.info("落在核心基因集内的记录: %d 条", len(df_clean))

    if df_clean.empty:
        logger.warning("清洗后无有效记录, 请检查 mygene 映射或核心基因集")
        OUTPUT_CSV.write_text("compound,gene,target_chembl_id,molecule_chembl_id,standard_type,standard_value,standard_units,pchembl_value,source,confidence,confidence_level\n")
        return 0

    # 4. 同一 compound-gene 保留最佳活性记录
    df_clean["pchembl_value"] = pd.to_numeric(df_clean["pchembl_value"], errors="coerce")
    df_clean["standard_value"] = pd.to_numeric(df_clean["standard_value"], errors="coerce")
    # 排序: pchembl 降序, 其次 standard_value 升序
    df_clean = df_clean.sort_values(
        by=["compound", "gene_official", "pchembl_value", "standard_value"],
        ascending=[True, True, False, True],
    )
    df_clean = df_clean.drop_duplicates(subset=["compound", "gene_official"], keep="first")

    # 5. 计算置信度
    confidences = []
    levels = []
    for _, row in df_clean.iterrows():
        c, lv = assign_confidence(row.get("pchembl_value"), row.get("standard_value"))
        confidences.append(c)
        levels.append(lv)
    df_clean["confidence"] = confidences
    df_clean["confidence_level"] = levels

    # 6. 选择输出列
    out_df = df_clean[
        [
            "compound",
            "gene_official",
            "target_chembl_id",
            "molecule_chembl_id",
            "standard_type",
            "standard_value",
            "standard_units",
            "pchembl_value",
            "source",
            "confidence",
            "confidence_level",
        ]
    ].copy()
    out_df = out_df.rename(columns={"gene_official": "gene"})
    out_df = out_df.sort_values(["compound", "gene", "confidence"], ascending=[True, True, False])

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("已写入清洗后 ChEMBL 边: %s (%d 条)", OUTPUT_CSV, len(out_df))

    # 7. 元数据
    metadata = {
        "source": "ChEMBL",
        "input_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "clean_date": pd.Timestamp.now().isoformat(),
        "mapping_tool": "mygene.info",
        "stats": {
            "raw_records": len(df),
            "raw_unique_genes": df["gene_raw"].nunique(),
            "mapped_to_official": int(n_mapped_total),
            "in_core_gene_set": len(df_clean),
            "unique_compounds": int(out_df["compound"].nunique()),
            "unique_genes": int(out_df["gene"].nunique()),
        },
        "confidence_distribution": out_df["confidence_level"].value_counts().to_dict(),
    }
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已写入元数据: %s", METADATA_JSON)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
