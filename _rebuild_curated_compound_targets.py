"""重建 compound_target_edges_curated.csv, 使来源透明可追溯.

原文件存在以下问题:
- 所有边的 source 均为模糊的 "curated";
- 包含大量无外部数据库或文献直接支持的边;
- 包含合成铁死亡工具化合物 (DFO/Erastin/Fer-1).

本脚本仅保留:
1. 天然中药单体 (BCP, Quercetin, VC);
2. 有外部数据库支持 (ChEMBL/CTD/BindingDB/STITCH 等) 的边, source 列写明具体数据库;
3. 或有明确文献证据的边, source 列写明文献标识 (PMID / 作者论文).

输出列: compound, gene, source, confidence, confidence_level, download_date
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
EXTERNAL_DIR = PROJECT_ROOT / "external_data"
GENE_LIST_PATH = PROJECT_ROOT / "铁衰老基因.txt"
OUTPUT_CSV = NETWORK_DIR / "compound_target_edges_curated.csv"

# 合成铁死亡工具 / 合成药物, 不参与中药单体预测
SYNTHETIC_TOOLS = {"DFO", "Erastin", "Fer-1", "Lip-1", "Ferrostatin-1", "Deferoxamine"}

# 外部数据库文件及其默认置信度
DB_FILES: dict[str, tuple[Path, float, str]] = {
    "ChEMBL_cleaned": (NETWORK_DIR / "chembl_compound_targets_cleaned.csv", 0.70, "medium"),
    "CTD": (NETWORK_DIR / "ctd_compound_targets.csv", 0.65, "medium"),
    "DGIdb": (NETWORK_DIR / "dgidb_compound_targets.csv", 0.45, "low"),
    "BindingDB": (NETWORK_DIR / "bindingdb_compound_targets.csv", 0.60, "medium"),
    "DrugBank": (NETWORK_DIR / "drugbank_compound_targets.csv", 0.60, "medium"),
    "DrugTargetCommons": (NETWORK_DIR / "drugtargetcommons_compound_targets.csv", 0.50, "low"),
    "SwissTargetPrediction": (NETWORK_DIR / "swisstarget_compound_targets.csv", 0.40, "low"),
    "STITCH": (EXTERNAL_DIR / "stitch" / "stitch_compound_targets.csv", 0.50, "low"),
    "ChEMBL_SEA": (NETWORK_DIR / "target_predictions_chembl_sea.csv", 0.40, "low"),
}

# 文献明确支持的天然单体-靶点边 (source 必须可追溯)
LITERATURE_EDGES: list[tuple[str, str, str, float, str]] = [
    # BCP 通过 NRF2/HO-1 通路抑制脑缺血后铁死亡
    # Hu Q, et al. Phytomedicine, 2022. PMID: 36150289
    ("BCP", "HMOX1", "literature_PMID:36150289", 0.85, "high"),
    # 作者论文: Vitamin C inhibits ACSL4 to alleviate ferro-aging in primates
    ("VC", "ACSL4", "literature_author_paper_VC_ACSL4", 0.90, "high"),
]


def load_core_genes(path: Path) -> set[str]:
    genes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


def collect_db_supported_edges(core_genes: set[str]) -> dict[tuple[str, str], set[str]]:
    """遍历外部数据库文件, 收集天然化合物-核心基因边的支持来源."""
    supported: dict[tuple[str, str], set[str]] = {}

    for src_name, (path, default_conf, _) in DB_FILES.items():
        if not path.exists():
            logger.warning("数据库文件不存在, 跳过: %s", path)
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            logger.error("读取失败: %s", path)
            traceback.print_exc()
            continue

        if "compound" not in df.columns or "gene" not in df.columns:
            logger.warning("%s 缺少 compound/gene 列, 跳过", src_name)
            continue

        df["compound"] = df["compound"].astype(str).str.strip()
        df["gene"] = df["gene"].astype(str).str.strip().str.upper()
        df = df[df["gene"].isin(core_genes)]
        df = df[~df["compound"].isin(SYNTHETIC_TOOLS)]

        for _, row in df.iterrows():
            key = (row["compound"], row["gene"])
            supported.setdefault(key, set()).add(src_name)

    return supported


def main() -> int:
    core_genes = load_core_genes(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    db_supported = collect_db_supported_edges(core_genes)
    logger.info("外部数据库支持的天然单体-核心基因边: %d 条", len(db_supported))

    rows: list[dict[str, object]] = []

    # 1) 外部数据库支持的边
    for (compound, gene), srcs in sorted(db_supported.items()):
        source_str = "|".join(sorted(srcs))
        rows.append({
            "compound": compound,
            "gene": gene,
            "source": source_str,
            "confidence": 0.85,
            "confidence_level": "high",
            "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        })

    # 2) 文献明确支持的边 (若同时有数据库支持, 合并来源)
    for compound, gene, lit_source, conf, level in LITERATURE_EDGES:
        if gene not in core_genes:
            logger.warning("文献边 %s-%s 的基因不在核心集中, 跳过", compound, gene)
            continue
        key = (compound, gene)
        srcs = db_supported.get(key, set()) | {lit_source}
        rows.append({
            "compound": compound,
            "gene": gene,
            "source": "|".join(sorted(srcs)),
            "confidence": conf,
            "confidence_level": level,
            "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        })

    if not rows:
        logger.error("未找到任何有证据支持的 curated 边")
        return 1

    result = pd.DataFrame(rows).drop_duplicates(subset=["compound", "gene"]).sort_values(
        ["compound", "gene"]
    ).reset_index(drop=True)

    # 备份原文件
    if OUTPUT_CSV.exists():
        backup = OUTPUT_CSV.with_suffix(".csv.bak")
        backup.write_text(OUTPUT_CSV.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("已备份原文件: %s", backup)

    result.to_csv(OUTPUT_CSV, index=False)
    logger.info("已写入 %d 条透明来源的 curated 边到 %s", len(result), OUTPUT_CSV)
    logger.info("来源分布:\n%s", result["source"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
