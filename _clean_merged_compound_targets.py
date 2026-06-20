"""清洗合并后的 compound_target_edges.csv.

输入:
    network_files/compound_target_edges_curated.csv   (高置信度人工整理)
    network_files/chembl_compound_targets_cleaned.csv (ChEMBL 清洗后)
    external_data/stitch/stitch_compound_targets.csv  (STITCH, 可选)

输出:
    network_files/compound_target_edges.csv           (清洗后)
    external_data/compound_target_merged_metadata.json

清洗规则:
1. 仅保留 gene 在 98 铁衰老核心基因集内。
2. 去除明显非标准基因符号的描述性文本(空格、过长、含标点)。
3. 同一 compound-gene 对去重, source 合并。
4. 为每条边分配置信度:
   - curated: 0.95
   - ChEMBL: 使用清洗后的 confidence
   - STITCH: stitch_score/1000, 截断到 [0.40, 0.90]
5. 输出带 source/confidence/confidence_level 的边。
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
NETWORK_DIR = PROJECT_ROOT / "network_files"
EXTERNAL_DIR = PROJECT_ROOT / "external_data"
GENE_LIST_PATH = PROJECT_ROOT / "铁衰老基因.txt"

CURATED_CSV = NETWORK_DIR / "compound_target_edges_curated.csv"
CHEMBL_CLEANED_CSV = NETWORK_DIR / "chembl_compound_targets_cleaned.csv"
STITCH_CSV = EXTERNAL_DIR / "stitch" / "stitch_compound_targets.csv"
OUTPUT_CSV = NETWORK_DIR / "compound_target_edges.csv"
METADATA_JSON = EXTERNAL_DIR / "compound_target_merged_metadata.json"


def load_gene_set(path: Path) -> set[str]:
    genes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


def is_valid_gene_symbol(g: str) -> bool:
    """判断是否为标准基因符号(排除描述性文本)."""
    if pd.isna(g):
        return False
    g = str(g).strip()
    if len(g) == 0 or len(g) > 20:
        return False
    if " " in g:
        return False
    if re.search(r"[-/().,;:]", g):
        return False
    # 允许大写字母+数字
    if not re.match(r"^[A-Z0-9]+$", g):
        return False
    return True


def load_curated(core_genes: set[str]) -> pd.DataFrame:
    df = pd.read_csv(CURATED_CSV)
    df["compound"] = df["compound"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    df = df[df["gene"].isin(core_genes) & df["gene"].apply(is_valid_gene_symbol)]
    df["source"] = "curated"
    df["confidence"] = 0.95
    df["confidence_level"] = "high"
    logger.info("curated 有效边: %d", len(df))
    return df


def load_chembl_cleaned(core_genes: set[str]) -> pd.DataFrame:
    df = pd.read_csv(CHEMBL_CLEANED_CSV)
    df["compound"] = df["compound"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    df = df[df["gene"].isin(core_genes) & df["gene"].apply(is_valid_gene_symbol)]
    df["source"] = "ChEMBL"
    if "confidence" not in df.columns:
        df["confidence"] = 0.60
    if "confidence_level" not in df.columns:
        df["confidence_level"] = "low"
    logger.info("ChEMBL 清洗后有效边: %d", len(df))
    return df


def load_stitch(core_genes: set[str]) -> pd.DataFrame:
    if not STITCH_CSV.exists():
        logger.warning("STITCH 文件不存在, 跳过: %s", STITCH_CSV)
        return pd.DataFrame(columns=["compound", "gene", "source", "confidence", "confidence_level", "stitch_score"])

    df = pd.read_csv(STITCH_CSV)
    df["compound"] = df["compound"].astype(str).str.strip()
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    df = df[df["gene"].isin(core_genes) & df["gene"].apply(is_valid_gene_symbol)]

    # 按 stitch_score 计算置信度
    if "stitch_score" in df.columns:
        df["confidence"] = df["stitch_score"].astype(float) / 1000.0
        df["confidence"] = df["confidence"].clip(0.40, 0.90)
    else:
        df["confidence"] = 0.50

    df["source"] = "STITCH"
    df["confidence_level"] = df["confidence"].apply(
        lambda x: "high" if x >= 0.80 else ("medium" if x >= 0.60 else "low")
    )
    logger.info("STITCH 有效边: %d", len(df))
    return df


def deduplicate_keep_highest_confidence(df: pd.DataFrame) -> pd.DataFrame:
    """同一 compound-gene 保留置信度最高记录; 如并列, source 合并."""
    if df.empty:
        return df

    # 先按置信度排序
    df = df.sort_values("confidence", ascending=False)
    # 取每个 compound-gene 的最高置信度行
    best = df.drop_duplicates(subset=["compound", "gene"], keep="first").copy()

    # 合并 source: 收集所有 source
    source_map = {}
    for _, row in df.iterrows():
        key = (row["compound"], row["gene"])
        source_map.setdefault(key, set()).add(str(row["source"]))

    best["source"] = best.apply(
        lambda row: "|".join(sorted(source_map.get((row["compound"], row["gene"]), {row["source"]}))),
        axis=1,
    )
    return best


def main() -> int:
    core_genes = load_gene_set(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    curated = load_curated(core_genes)
    chembl = load_chembl_cleaned(core_genes)
    stitch = load_stitch(core_genes)

    # 统一列
    common_cols = ["compound", "gene", "source", "confidence", "confidence_level"]
    curated = curated[[c for c in common_cols if c in curated.columns]]
    chembl = chembl[[c for c in common_cols if c in chembl.columns]]
    stitch = stitch[[c for c in common_cols if c in stitch.columns]]

    merged = pd.concat([curated, chembl, stitch], ignore_index=True)
    logger.info("合并去重前: %d 条", len(merged))

    cleaned = deduplicate_keep_highest_confidence(merged)
    cleaned = cleaned.sort_values(["compound", "gene"]).reset_index(drop=True)
    logger.info("合并去重后: %d 条", len(cleaned))

    # 备份原文件
    if OUTPUT_CSV.exists():
        backup_path = OUTPUT_CSV.with_suffix(".csv.bak")
        backup_path.write_text(OUTPUT_CSV.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("已备份原文件: %s", backup_path)

    cleaned.to_csv(OUTPUT_CSV, index=False)
    logger.info("已写入清洗后 compound-target 边: %s", OUTPUT_CSV)

    metadata = {
        "source": "curated + ChEMBL(cleaned) + STITCH",
        "clean_date": pd.Timestamp.now().isoformat(),
        "stats": {
            "curated": len(curated),
            "chembl_cleaned": len(chembl),
            "stitch": len(stitch),
            "merged_unique": len(cleaned),
            "unique_compounds": int(cleaned["compound"].nunique()),
            "unique_genes": int(cleaned["gene"].nunique()),
        },
        "confidence_distribution": cleaned["confidence_level"].value_counts().to_dict(),
        "source_distribution": {k: int(v) for k, v in cleaned["source"].value_counts().items()},
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
