"""Integrate FerrDb V3 (early preview) ferroptosis regulators into the project gene set.

Data source: http://www.zhounan.org/ferrdb/current/extdownload/ferroptosis_early_preview_upto20231231.zip
Downloaded from the FerrDb official download page; preview covers curation up to 2023/12/31.
"""
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
EXTERNAL_DIR = PROJECT_ROOT / "external_data"
FERRDB_DIR = EXTERNAL_DIR / "ferrdb"
FERRDB_BASE = FERRDB_DIR / "ferroptosis_preview_extracted" / "ferroptosis_early_preview_upto20231231"
CORE_GENE_PATH = PROJECT_ROOT / "铁衰老基因.txt"

GENE_OUTPUT = NETWORK_DIR / "ferrdb_v3_gene_regulators.csv"
COMPOUND_OUTPUT = NETWORK_DIR / "ferrdb_compound_regulators.csv"
METADATA_OUTPUT = FERRDB_DIR / "ferrdb_download_metadata.json"


def normalize_confidence(value: Any) -> str:
    """统一 FerrDb confidence 字符串."""
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().lower()
    mapping = {
        "validated": "Validated",
        "validate": "Validated",
        "screened": "Screened",
        "predicted": "Predicted",
        "deduced": "Deduced",
        "deduecd": "Deduced",
    }
    return mapping.get(v, str(value).strip())


def confidence_score(conf: str) -> float:
    """将 confidence 映射为数值分数."""
    return {
        "Validated": 1.0,
        "Screened": 0.7,
        "Predicted": 0.5,
        "Deduced": 0.4,
        "Unknown": 0.0,
    }.get(conf, 0.0)


def load_ferrdb_genes() -> pd.DataFrame:
    """读取 FerrDb 基因调控因子文件."""
    files = {
        "driver": ("driver.csv", "Symbol_or_reported_abbr"),
        "suppressor": ("suppressor.csv", "Symbol"),
        "marker": ("marker.csv", "Symbol"),
        "unclassified": ("unclassified.reg.csv", "Symbol"),
    }
    records = []
    for category, (fname, sym_col) in files.items():
        path = FERRDB_BASE / fname
        df = pd.read_csv(path)
        df["category"] = category
        df["gene"] = df[sym_col].astype(str).str.strip().str.upper()
        df["Confidence_norm"] = df["Confidence"].apply(normalize_confidence)
        df["Exp_organism_norm"] = df["Exp_organism"].astype(str).str.strip()
        records.append(
            df[
                [
                    "gene",
                    "category",
                    "Confidence_norm",
                    "Experiments",
                    "Exp_organism_norm",
                    "PMID",
                    "HGNC_ID",
                    "ENSG_stable",
                    "UniProtAC",
                    "Pathway",
                ]
            ].copy()
        )
    return pd.concat(records, ignore_index=True)


def _join_unique(values, skip: set[str] | None = None) -> str:
    """将 Series 中的非空、非占位值去重后用 '|' 连接."""
    skip = skip or set()
    items = sorted({str(v).strip() for v in values if pd.notna(v) and str(v).strip() and str(v).strip() not in skip})
    return "|".join(items)


def build_regulator_table(df: pd.DataFrame) -> pd.DataFrame:
    """按基因聚合,生成 regulator 汇总表."""
    df = df.copy()
    df["conf_score"] = df["Confidence_norm"].apply(confidence_score)

    agg = df.groupby("gene").agg(
        categories=("category", lambda x: "|".join(sorted(set(x)))),
        max_confidence=("conf_score", "max"),
        confidence_levels=("Confidence_norm", lambda x: "|".join(sorted(set(x)))),
        experiment_count=("Experiments", "sum"),
        organisms=("Exp_organism_norm", lambda x: _join_unique(x, skip={"nan"})),
        pmids=("PMID", lambda x: _join_unique(x, skip={"_NA_", ""})),
        hgnc_ids=("HGNC_ID", lambda x: _join_unique(x, skip={"_NA_", ""})),
        ensg_ids=("ENSG_stable", lambda x: _join_unique(x, skip={"_NA_", ""})),
        uniprot_ids=("UniProtAC", lambda x: _join_unique(x, skip={"_NA_", ""})),
        pathway_notes=("Pathway", lambda x: _join_unique(x, skip={"_NA_", ""})),
        row_count=("gene", "count"),
    )
    agg = agg.reset_index()
    agg["max_confidence"] = agg["max_confidence"].round(3)
    return agg


def load_existing_compounds() -> set[str]:
    """读取项目当前化合物列表."""
    path = NETWORK_DIR / "compound_smiles.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(df.iloc[:, 0].astype(str).str.strip().str.lower())


def build_compound_regulators(existing_compounds: set[str]) -> pd.DataFrame:
    """从 FerrDb inducer/inhibitor 中提取与项目化合物匹配的条目."""
    records = []
    for category, fname in [("inducer", "inducer.csv"), ("inhibitor", "inhibitor.csv")]:
        df = pd.read_csv(FERRDB_BASE / fname)
        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            molecule = str(row.get("Molecule", "")).strip()
            cid = str(row.get("PubChem_CID", "")).strip()
            matched_name = name if name.lower() in existing_compounds else None
            matched_molecule = molecule if molecule.lower() in existing_compounds else None
            if matched_name or matched_molecule:
                records.append(
                    {
                        "compound": matched_name or matched_molecule,
                        "ferrdb_name": name,
                        "ferrdb_molecule": molecule,
                        "pubchem_cid": cid,
                        "role": category,
                        "pmid": row.get("PMID"),
                        "source": "FerrDb",
                        "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    if not FERRDB_BASE.exists():
        raise FileNotFoundError(f"FerrDb 预览数据未解压: {FERRDB_BASE}")

    # 1. 处理基因调控因子
    logger.info("读取 FerrDb 基因调控因子...")
    df_genes = load_ferrdb_genes()
    logger.info("原始记录: %d 条, 唯一基因: %d", len(df_genes), df_genes["gene"].nunique())

    regulator_table = build_regulator_table(df_genes)
    regulator_table = regulator_table.sort_values(["max_confidence", "experiment_count"], ascending=False)
    regulator_table.to_csv(GENE_OUTPUT, index=False)
    logger.info("已保存 regulator 表: %s (%d genes)", GENE_OUTPUT, len(regulator_table))

    # 2. 生成扩展基因集: 保留原 98 个核心基因 + FerrDb 人类 Validated/Screened driver/suppressor/marker
    with open(CORE_GENE_PATH, encoding="utf-8") as f:
        core_genes = {line.strip().upper() for line in f if line.strip()}
    logger.info("现有核心基因: %d", len(core_genes))

    human_high_conf = df_genes[
        df_genes["Exp_organism_norm"].str.contains("Human", case=False, na=False)
        & df_genes["category"].isin({"driver", "suppressor", "marker"})
        & df_genes["Confidence_norm"].isin({"Validated", "Screened"})
    ]
    expanded_genes = sorted(core_genes | set(human_high_conf["gene"].unique()))
    logger.info("扩展后基因集: %d (新增 %d)", len(expanded_genes), len(expanded_genes) - len(core_genes))

    # 3. 化合物调控因子匹配
    existing_compounds = load_existing_compounds()
    compound_regs = build_compound_regulators(existing_compounds)
    if not compound_regs.empty:
        compound_regs.to_csv(COMPOUND_OUTPUT, index=False)
        logger.info("已保存化合物调控因子匹配表: %s (%d rows)", COMPOUND_OUTPUT, len(compound_regs))
    else:
        logger.warning("未找到与项目化合物匹配的 FerrDb inducer/inhibitor")

    # 4. 元数据
    metadata = {
        "source": "FerrDb",
        "version": "V3 early preview (up to 2023-12-31)",
        "url": "http://www.zhounan.org/ferrdb/current/extdownload/ferroptosis_early_preview_upto20231231.zip",
        "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "total_regulator_records": int(len(df_genes)),
        "unique_regulators": int(df_genes["gene"].nunique()),
        "human_validated_screened_driver_suppressor_marker": int(len(human_high_conf["gene"].unique())),
        "expanded_gene_set_size": len(expanded_genes),
        "core_gene_overlap": len(core_genes & set(df_genes["gene"].unique())),
        "citation": (
            "Zhou N, Peng L, Luo Q, et al. FerrDb V3: expanding the manually curated resource "
            "for regulators and disease associations from ferroptosis to regulated cell death. "
            "Nucleic Acids Res. 2026;54(D1):D572-D582."
        ),
        "pmid": "41171133",
    }
    with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("已保存元数据: %s", METADATA_OUTPUT)

    # 5. 输出扩展基因集提示 (不直接覆盖原文件, 留待人工确认后再切换)
    expanded_path = PROJECT_ROOT / "铁衰老基因_ferrdb_v3_expanded.txt"
    with open(expanded_path, "w", encoding="utf-8") as f:
        for g in expanded_genes:
            f.write(f"{g}\n")
    logger.info("扩展基因集已写入 (供复核): %s", expanded_path)
    logger.info(
        "如需采用扩展基因集, 请将 %s 替换为 %s 并重新运行 generate_all_network_files.py",
        CORE_GENE_PATH.name,
        expanded_path.name,
    )


if __name__ == "__main__":
    main()
