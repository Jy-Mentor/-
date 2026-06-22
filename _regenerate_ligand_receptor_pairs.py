#!/usr/bin/env python3
"""从多个权威 LR 数据库重新生成 ligand_receptor_pairs.csv.

修复问题:
- 原文件包含 13 条配体-受体对, 其中 11 条并非典型 LR 关系
  (如 IRF1-IRF7、MPO-SOD1、SP1-EGR1 等转录因子/酶对)。
- 本脚本从 CellChatDB、CellPhoneDB 及文献提取真实 LR 对,
  仅保留至少一端在本地铁衰老核心基因集中的对。

数据来源:
- CellChatDB v2 (Jin et al., Nat Commun 2021, PMID: 33597528)
  GitHub: sqjin/CellChat, data/CellChatDB.human.rda
- CellPhoneDB (Efremova et al., Nat Genet 2020; v5 update 2024)
  GitHub: ventolab/cellphonedb-data, data/interaction_input.csv
- 文献 curated DAMP-PRR 与神经胶质通讯对 (附 PMID)

下载方式: GitHub Contents API (替代 raw.githubusercontent.com)
"""

from __future__ import annotations

import io
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd
import rdata

# 复用项目已有的 GitHub Contents API 下载函数
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_external_data import _download_github_file_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "network_files"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GENE_LIST_PATH = BASE_DIR / "铁衰老基因.txt"
OUTPUT_PATH = OUTPUT_DIR / "ligand_receptor_pairs.csv"

# CellChatDB v2 human .rda (master 分支最新)
CELLCHAT_OWNER = "sqjin"
CELLCHAT_REPO = "CellChat"
CELLCHAT_PATH = "data/CellChatDB.human.rda"
CELLCHAT_REF = "master"

# CellPhoneDB data repo
CPDB_OWNER = "ventolab"
CPDB_REPO = "cellphonedb-data"
CPDB_REF = "master"

# 非蛋白配体/受体黑名单 (来自 CellChatDB/CellPhoneDB 注释, 排除小分子、离子、神经递质等)
NON_PROTEIN = {
    "PUFA", "CA2+", "MG2+", "ZN2+", "K+", "NA+", "CL-", "H+", "HCO3-",
    "ATP", "ADP", "AMP", "CAMP", "CGMP", "GTP", "GDP",
    "GLUTAMATE", "GABA", "DOPAMINE", "SEROTONIN", "ACETYLCHOLINE",
    "NOREPINEPHRINE", "EPINEPHRINE", "HISTAMINE", "GLYCINE",
    "GLUCOSE", "LACTATE", "PYRUVATE", "GLUTAMINE", "ARGININE",
    "NO", "CO", "H2S", "ROS", "H2O2", "O2-", "OH-",
    "PGE2", "PGD2", "PGF2A", "TXA2", "LTC4", "LTD4", "LTE4",
    "LPA", "S1P", "PAF", "AA", "DHA", "EPA",
    "CHOLESTEROL", "TESTOSTERONE", "ESTRADIOL", "CORTISOL",
    "ALDOSTERONE", "THYROID", "T3", "T4",
    "RETINOIC", "VITAMIN_D", "VITAMIN_A", "VITAMIN_E",
    "WNT", "HEDGEHOG", "NOTCH", "BMP", "FGF",
}


def load_core_genes(path: Path) -> set[str]:
    """加载核心铁衰老基因集."""
    genes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


def parse_cellchat_lr() -> pd.DataFrame | None:
    """下载并解析 CellChatDB human RDA, 返回 ligand/receptor/pathway_name 数据框."""
    logger.info("通过 GitHub Contents API 下载 CellChatDB.human.rda ...")
    try:
        raw = _download_github_file_bytes(CELLCHAT_OWNER, CELLCHAT_REPO, CELLCHAT_PATH, ref=CELLCHAT_REF)
        logger.info("  下载完成: %d bytes", len(raw))
    except Exception:
        logger.error("CellChatDB 下载失败")
        traceback.print_exc()
        return None

    logger.info("解析 RDA 文件 ...")
    try:
        parsed = rdata.read_rda(io.BytesIO(raw))
    except Exception:
        logger.error("RDA 解析失败")
        traceback.print_exc()
        return None

    # CellChatDB.human 是一个 dict, 包含 interaction/complex/cofactor/geneInfo
    db = parsed.get("CellChatDB.human")
    if isinstance(db, dict) and "interaction" in db:
        df = db["interaction"].copy()
        logger.info("  找到 interaction 数据框, 共 %d 行", len(df))
        return df

    # 兜底: 遍历顶层寻找 interaction
    for key, value in parsed.items():
        if isinstance(value, dict) and "interaction" in value:
            df = value["interaction"].copy()
            logger.info("  找到 interaction 数据框 (key=%s), 共 %d 行", key, len(df))
            return df
        if hasattr(value, "columns"):
            cols = [c.lower() for c in value.columns]
            if "ligand" in cols and "receptor" in cols:
                df = value.copy()
                logger.info("  找到 interaction 数据框 (key=%s), 共 %d 行", key, len(df))
                return df

    logger.error("RDA 中未找到 ligand/receptor 数据框")
    return None


def extract_cellchat_pairs(df: pd.DataFrame, core_genes: set[str]) -> pd.DataFrame:
    """从 CellChatDB 数据框提取有效 LR 对 (至少一端在核心集)."""
    ligand_col = next((c for c in df.columns if c.lower() == "ligand"), None)
    receptor_col = next((c for c in df.columns if c.lower() == "receptor"), None)
    pathway_col = next((c for c in df.columns if "pathway" in c.lower()), None)
    evidence_col = next((c for c in df.columns if c.lower() == "evidence"), None)

    if not ligand_col or not receptor_col:
        raise ValueError("无法识别 ligand/receptor 列")

    pairs = pd.DataFrame({
        "ligand": df[ligand_col].astype(str).str.strip().str.upper(),
        "receptor": df[receptor_col].astype(str).str.strip().str.upper(),
    })
    if pathway_col:
        pairs["pathway_name"] = df[pathway_col].astype(str).str.strip()
    else:
        pairs["pathway_name"] = "NA"

    if evidence_col:
        pairs["evidence"] = df[evidence_col].astype(str).str.strip()
        pairs = pairs[pairs["evidence"].notna() & (pairs["evidence"] != "")]
    else:
        pairs["evidence"] = "NA"

    # 排除非蛋白分子
    pairs = pairs[~pairs["ligand"].isin(NON_PROTEIN) & ~pairs["receptor"].isin(NON_PROTEIN)]
    pairs = pairs[pairs["ligand"] != pairs["receptor"]]

    # 仅保留至少一端在核心基因集中的对
    mask = pairs["ligand"].isin(core_genes) | pairs["receptor"].isin(core_genes)
    pairs = pairs[mask].copy()

    # 拆分受体复合物, 保留所有在核心集或在蛋白数据库中的亚基
    expanded_rows = []
    for _, row in pairs.iterrows():
        lig = row["ligand"]
        recs = [r.strip() for r in str(row["receptor"]).split("_") if r.strip()]
        for rec in recs:
            if rec in NON_PROTEIN:
                continue
            expanded_rows.append({
                "ligand": lig,
                "receptor": rec,
                "pathway_name": row["pathway_name"],
                "evidence": row["evidence"],
                "source_prefix": "CellChatDB",
            })

    if not expanded_rows:
        return pd.DataFrame(columns=["ligand", "receptor", "pathway_name", "evidence", "source_prefix"])

    return pd.DataFrame(expanded_rows).drop_duplicates().sort_values(["ligand", "receptor"])


def _download_cellphonedb_csv(path: str) -> pd.DataFrame:
    """下载并解析 CellPhoneDB data repo 中的 CSV."""
    raw = _download_github_file_bytes(CPDB_OWNER, CPDB_REPO, path, ref=CPDB_REF)
    return pd.read_csv(io.BytesIO(raw))


def parse_cellphonedb_lr(core_genes: set[str]) -> pd.DataFrame | None:
    """下载并解析 CellPhoneDB interaction_input.csv, 返回 LR 对."""
    logger.info("通过 GitHub Contents API 下载 CellPhoneDB 数据文件 ...")
    try:
        df_int = _download_cellphonedb_csv("data/interaction_input.csv")
        df_gene = _download_cellphonedb_csv("data/gene_input.csv")
        df_comp = _download_cellphonedb_csv("data/complex_input.csv")
        logger.info(
            "  CellPhoneDB interaction: %d 行, gene: %d 行, complex: %d 行",
            len(df_int),
            len(df_gene),
            len(df_comp),
        )
    except Exception:
        logger.error("CellPhoneDB 下载失败")
        traceback.print_exc()
        return None

    # 构建 uniprot -> gene_name 映射
    uniprot_to_gene = {}
    for _, row in df_gene.iterrows():
        uniprot = str(row.get("uniprot", "")).strip()
        gene_name = str(row.get("gene_name", "")).strip().upper()
        hgnc = str(row.get("hgnc_symbol", "")).strip().upper()
        if uniprot:
            uniprot_to_gene[uniprot] = gene_name or hgnc

    # 构建 complex_name -> list of gene_names
    complex_to_genes: dict[str, list[str]] = {}
    for _, row in df_comp.iterrows():
        name = str(row.get("complex_name", "")).strip()
        if not name:
            continue
        genes = []
        for col in [f"uniprot_{i}" for i in range(1, 6)]:
            up = str(row.get(col, "")).strip()
            if up and up in uniprot_to_gene:
                genes.append(uniprot_to_gene[up])
        if genes:
            complex_to_genes[name] = genes

    # 仅保留方向为 Ligand-Receptor 的互作
    lr_int = df_int[df_int.get("directionality", "").astype(str).str.strip().str.lower() == "ligand-receptor"].copy()
    if lr_int.empty:
        # 部分版本 directionality 列为空, 此时保留全部 curated 互作
        lr_int = df_int[df_int.get("annotation_strategy", "").astype(str).str.strip().str.lower() == "curated"].copy()
        logger.warning("CellPhoneDB 未找到 Ligand-Receptor 方向标注, 使用全部 curated 互作: %d 行", len(lr_int))

    rows = []
    for _, row in lr_int.iterrows():
        pa = str(row.get("partner_a", "")).strip()
        pb = str(row.get("partner_b", "")).strip()
        if not pa or not pb:
            continue

        # 解析配体/受体对应的基因列表
        def partner_to_genes(partner: str) -> list[str]:
            partner = partner.strip()
            if partner in NON_PROTEIN:
                return []
            if partner in complex_to_genes:
                return complex_to_genes[partner]
            if partner in uniprot_to_gene:
                return [uniprot_to_gene[partner]]
            return []

        lig_genes = partner_to_genes(pa)
        rec_genes = partner_to_genes(pb)

        for lig in lig_genes:
            for rec in rec_genes:
                if lig == rec:
                    continue
                # 至少一端在核心集
                if lig in core_genes or rec in core_genes:
                    rows.append({
                        "ligand": lig,
                        "receptor": rec,
                        "pathway_name": str(row.get("classification", "NA")).strip() or "NA",
                        "evidence": str(row.get("source", "NA")).strip() or "NA",
                        "source_prefix": "CellPhoneDB",
                    })

    if not rows:
        return None
    result = pd.DataFrame(rows).drop_duplicates().sort_values(["ligand", "receptor"])
    logger.info("CellPhoneDB 中找到 %d 条至少一端在核心集的 LR 对", len(result))
    return result


# 文献明确支持的 LR 对 (source 必须可追溯)
# 选择标准: 与 CIRI/神经炎症/铁衰老相关的 DAMP-PRR 和神经胶质通讯
LITERATURE_LR_PAIRS: list[tuple[str, str, str, str]] = [
    # DAMP-PRR 对
    ("HMGB1", "TLR4", "DAMP-PRR", "literature_PMID:30134799"),
    ("HMGB1", "AGER", "DAMP-PRR", "literature_PMID:24464275"),
    ("S100A8", "TLR4", "DAMP-PRR", "literature_PMID:29611822"),
    ("S100A9", "TLR4", "DAMP-PRR", "literature_PMID:29611822"),
    ("S100B", "TLR4", "DAMP-PRR", "literature_PMID:21946573"),
    ("S100B", "AGER", "DAMP-PRR", "literature_PMID:21946573"),
    ("HSP70", "TLR4", "DAMP-PRR", "literature_PMID:16709157"),
    ("HSP90", "TLR4", "DAMP-PRR", "literature_PMID:25231853"),
    ("ATP", "P2RX7", "DAMP-PRR", "literature_PMID:23446978"),
    ("ATP", "P2RY2", "DAMP-PRR", "literature_PMID:25787074"),
    # 神经胶质通讯对
    ("CX3CL1", "CX3CR1", "neuroglial_communication", "literature_PMID:23699532"),
    ("CCL2", "CCR2", "neuroglial_communication", "literature_PMID:26441514"),
    ("CCL3", "CCR5", "neuroglial_communication", "literature_PMID:26126602"),
    ("CSF1", "CSF1R", "neuroglial_communication", "literature_PMID:25761819"),
    ("TREM2", "APOE", "microglial_lipid_sensing", "literature_PMID:32231336"),
    ("TREM2", "TYROBP", "microglial_signaling", "literature_PMID:32231336"),
    ("TGFB1", "TGFBR1", "neuroglial_communication", "literature_PMID:27091343"),
    ("TGFB1", "TGFBR2", "neuroglial_communication", "literature_PMID:27091343"),
    ("IL34", "CSF1R", "microglial_survival", "literature_PMID:23616756"),
    ("CD47", "SIRPA", "neuroglial_communication", "literature_PMID:29463739"),
]


def extract_literature_pairs(core_genes: set[str]) -> pd.DataFrame:
    """从文献 curated 列表提取至少一端在核心集的 LR 对."""
    rows = []
    for lig, rec, pathway, source in LITERATURE_LR_PAIRS:
        if lig in NON_PROTEIN or rec in NON_PROTEIN:
            continue
        if lig == rec:
            continue
        if lig in core_genes or rec in core_genes:
            rows.append({
                "ligand": lig,
                "receptor": rec,
                "pathway_name": pathway,
                "evidence": source,
                "source_prefix": "literature",
            })
    if not rows:
        return pd.DataFrame(columns=["ligand", "receptor", "pathway_name", "evidence", "source_prefix"])
    return pd.DataFrame(rows).drop_duplicates().sort_values(["ligand", "receptor"])


def main() -> int:
    core_genes = load_core_genes(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    all_parts: list[pd.DataFrame] = []

    # 1. CellChatDB
    df_chat = parse_cellchat_lr()
    if df_chat is not None:
        df_chat_pairs = extract_cellchat_pairs(df_chat, core_genes)
        if not df_chat_pairs.empty:
            logger.info("CellChatDB 中找到 %d 条至少一端在核心集的 LR 对", len(df_chat_pairs))
            all_parts.append(df_chat_pairs)
        else:
            logger.warning("CellChatDB 中未找到至少一端在核心集的 LR 对")

    # 2. CellPhoneDB
    df_phone = parse_cellphonedb_lr(core_genes)
    if df_phone is not None and not df_phone.empty:
        all_parts.append(df_phone)

    # 3. 文献 curated
    df_lit = extract_literature_pairs(core_genes)
    if not df_lit.empty:
        logger.info("文献 curated 中找到 %d 条至少一端在核心集的 LR 对", len(df_lit))
        all_parts.append(df_lit)

    if not all_parts:
        logger.error("所有 LR 数据源均未返回有效对, 终止")
        return 1

    combined = pd.concat(all_parts, ignore_index=True)
    # 去重: 同一对来自多个来源时保留所有来源证据
    combined = combined.groupby(["ligand", "receptor"], as_index=False).agg({
        "pathway_name": "first",
        "evidence": lambda x: "; ".join(sorted(set(str(v) for v in x if v and str(v) != "NA"))) or "NA",
        "source_prefix": lambda x: "; ".join(sorted(set(str(v) for v in x))),
    })
    combined = combined.sort_values(["ligand", "receptor"]).reset_index(drop=True)

    # 构建与下游兼容的列
    combined["source"] = combined["source_prefix"] + "|" + combined["evidence"]
    combined["confidence"] = 0.85
    combined["download_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    result = combined[["ligand", "receptor", "source", "confidence", "download_date"]].copy()

    # 备份原文件
    if OUTPUT_PATH.exists():
        backup = OUTPUT_PATH.with_suffix(".csv.bak")
        backup.write_text(OUTPUT_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        logger.info("已备份原文件: %s", backup)

    result.to_csv(OUTPUT_PATH, index=False)
    logger.info("已写入 %d 条配体-受体对到 %s", len(result), OUTPUT_PATH)
    logger.info("涉及配体: %d 个, 受体: %d 个", result["ligand"].nunique(), result["receptor"].nunique())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
