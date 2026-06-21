#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
L2 机制层: ACSL4是ISP的核心执行者
=============================================================================
架构:
  1. Python 加载 GEO 表达矩阵 → 保存 CSV 供 R WGCNA 使用
  2. 调用 R WGCNA 脚本 (真 WGCNA + modulePreservation)
  3. Python 读取 R 结果 → ACSL4 表达分析 + GPX4 趋势分析
  4. 输出数据供 R 绘图

方法升级:
  - 旧版: Python 假 WGCNA (correlation + hclust)
  - 新版: R WGCNA 包 (pickSoftThreshold + TOM + blockwiseModules)
  - 新增: modulePreservation (Zsummary) 跨物种保守性统计

依赖: numpy, pandas, scipy, matplotlib
=============================================================================
"""

import gzip
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# 加载统一项目配置 (避免硬编码绝对路径)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _l2_config import get_l2_config

_L2_CFG = get_l2_config(Path(__file__).resolve().parent)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================
def _open_text_or_gzip(path: str):
    """根据扩展名智能打开普通文本或 gzip 压缩文件."""
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="latin-1")
    return open(path, "r", encoding="utf-8", errors="replace")


def _parse_gpl6883_annotation(annot_path: str) -> Dict[str, str]:
    """解析 GPL6883 平台注释文件, 返回 probe_id -> gene_symbol 映射.

    同时兼容 GEO 下载的 gzip 压缩格式 (含 !platform_table_begin)
    与用户提供的纯文本格式 (含 # 注释头与制表符表头).
    """
    probe_map: Dict[str, str] = {}
    if not annot_path or not os.path.exists(annot_path):
        logger.warning(f"GPL6883 注释文件不存在: {annot_path}")
        return probe_map

    with _open_text_or_gzip(annot_path) as f:
        in_table = False
        header_parsed = False
        gs_idx: Optional[int] = None
        for line in f:
            current_line = line.strip()
            if not current_line:
                continue

            # GEO gzip 格式入口
            if current_line == "!platform_table_begin":
                in_table = True
                header_line = f.readline()
                if not header_line:
                    break
                header = header_line.strip().split("\t")
                gs_idx = next(
                    (
                        i
                        for i, h in enumerate(header)
                        if "gene symbol" in h.lower() or "symbol" == h.lower().strip()
                    ),
                    None,
                )
                header_parsed = True
                continue

            # 纯文本格式: 跳过 # 注释, 首条非注释行为表头
            if current_line.startswith("#"):
                continue
            if not header_parsed:
                header = current_line.split("\t")
                gs_idx = next(
                    (
                        i
                        for i, h in enumerate(header)
                        if "gene symbol" in h.lower() or "symbol" == h.lower().strip()
                    ),
                    None,
                )
                header_parsed = True
                continue

            if in_table and current_line == "!platform_table_end":
                break

            fields = current_line.split("\t")
            if gs_idx is None or len(fields) <= gs_idx:
                continue
            probe = fields[0].strip('"').strip()
            gene = fields[gs_idx].strip('"').strip()
            if gene:
                probe_map[probe] = gene

    logger.info(f"GPL6883 注释解析完成: {len(probe_map)} 条 probe -> gene 映射")
    return probe_map


# ============================================================
# 路径配置 (从 config.yaml 加载, 支持环境变量覆盖)
# ============================================================
IRON_AGING_DIR = _L2_CFG["project_root"]
L1_OUTPUT_DIR = _L2_CFG["l1_results_dir"]
L2_OUTPUT_DIR = _L2_CFG["output_dir"]
L2_WGCNA_INPUT = _L2_CFG["wgcna_input_dir"]
L2_WGCNA_OUTPUT = _L2_CFG["wgcna_output_dir"]

DATA_DIRS = {
    k: str(v) for k, v in _L2_CFG["datasets"].items() if k not in ("GSE174574",)
}

GPL6883_ANNOT = str(_L2_CFG["annotations"].get("GPL6883", ""))
GPL1355_FILE = str(_L2_CFG["annotations"].get("GPL1355", ""))
RSCRIPT = str(_L2_CFG["rscript"])

# 基因集定义 (从 config.yaml 加载, 保留默认值以兼容旧逻辑)
_gene_sets = _L2_CFG.get("gene_sets", {})
ISP_CORE_GENES = _gene_sets.get(
    "isp_core",
    [
        "ACSL4",
        "PTGS2",
        "HMOX1",
        "TFRC",
        "ALOX15",
        "HMGB1",
        "TLR4",
        "NLRP3",
        "HIF1A",
        "KEAP1",
        "SOD1",
        "MPO",
        "CDO1",
        "DPP4",
        "NOX4",
    ],
)

PURE_FERROPTOSIS = [
    "GPX4",
    "SLC7A11",
    "SLC3A2",
    "FSP1",
    "CHAC1",
    "GCLC",
    "GCLM",
    "HSPB1",
    "SLC1A4",
    "SLC1A5",
    "SLC39A14",
    "SLC40A1",
    "STEAP3",
    "CISD1",
    "CISD2",
    "NCOA4",
    "HMGCR",
    "LPCAT3",
    "ACSL4",
    "ALOX5",
]

PURE_SENESCENCE = [
    "CDKN1A",
    "CDKN2A",
    "CDKN2B",
    "LMNB1",
    "IL6",
    "IL1B",
    "IL8",
    "CXCL8",
    "CCL2",
    "CCL8",
    "MMP3",
    "MMP9",
    "MMP12",
    "TIMP1",
    "SERPINE1",
    "IGFBP3",
    "IGFBP7",
    "GADD45A",
    "GADD45B",
    "GDF15",
]

DISCRIMINATOR_GENES = _gene_sets.get(
    "discriminator", ["GPX4", "ACSL4", "PTGS2", "HMOX1", "TFRC", "CDKN1A", "IL6"]
)

# ============================================================
# GEO 数据加载 (复用 L1 逻辑)
# ============================================================


def find_file(dir_path: str, keywords: List[str]) -> Optional[str]:
    if not os.path.isdir(dir_path):
        return None
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if all(k.lower() in f.lower() for k in keywords):
                return os.path.join(root, f)
    return None


def parse_series_matrix(filepath: str) -> pd.DataFrame:
    open_func = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"
    with open_func(filepath, mode, encoding="latin-1") as f:
        content = f.read()
    lines = content.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("!series_matrix_table_begin"):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            header_idx = j
            break
    if header_idx is None:
        raise ValueError(f"无法找到series_matrix_table_begin: {filepath}")
    header = lines[header_idx].strip().split("\t")
    header = [h.strip('"').strip() for h in header]
    data_lines = []
    for i in range(header_idx + 1, len(lines)):
        if lines[i].startswith("!series_matrix_table_end"):
            break
        if lines[i].strip():
            data_lines.append(lines[i])
    data, index = [], []
    for line in data_lines:
        fields = line.strip().split("\t")
        if len(fields) < 2:
            continue
        probe_id = fields[0].strip('"').strip()
        index.append(probe_id)
        values = [float(v) if v not in ("null", "") else np.nan for v in fields[1:]]
        if len(values) < len(header) - 1:
            values.extend([np.nan] * (len(header) - 1 - len(values)))
        data.append(values[: len(header) - 1])
    df = pd.DataFrame(data, index=index, columns=header[1:])
    return df


def parse_gpl1355_annotation(filepath: str) -> Dict[str, str]:
    probe_map = {}
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        in_table = False
        gene_col = 5
        for line in f:
            current_line = line.strip()
            if current_line.startswith("ID"):
                in_table = True
                header = current_line.split("\t")
                try:
                    gene_col = next(
                        i
                        for i, h in enumerate(header)
                        if "gene symbol" in h.lower() or "symbol" in h.lower()
                    )
                except StopIteration:
                    pass
                continue
            if not in_table or not current_line:
                continue
            fields = current_line.split("\t")
            if len(fields) <= max(gene_col, 0):
                continue
            probe = fields[0]
            gene = fields[gene_col].strip('"').strip()
            if gene:
                probe_map[probe] = gene.split("///")[0].strip()
    return probe_map


def collapse_probes(expr_df: pd.DataFrame, probe_map: Dict[str, str]) -> pd.DataFrame:
    mapped = expr_df[expr_df.index.isin(probe_map.keys())].copy()
    if mapped.empty:
        return expr_df
    gene_series = pd.Series(mapped.index.map(probe_map), index=mapped.index)
    gene_series = gene_series.dropna()
    mapped = mapped.loc[gene_series.index]
    mapped.index = gene_series.str.upper()
    mapped = mapped.groupby(mapped.index).max()
    return mapped


def load_expression_matrix(ds_name: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """加载表达矩阵并返回 (全基因表达矩阵, case_cols, control_cols)"""
    if ds_name == "GSE16561":
        sm_file = find_file(DATA_DIRS["GSE16561"], ["series_matrix"])
        expr_df = parse_series_matrix(sm_file)
        with gzip.open(sm_file, "rt", encoding="latin-1") as f:
            lines = f.readlines()
        desc_line, sample_line = None, None
        for current_line in lines:
            if current_line.startswith("!Sample_description"):
                desc_line = current_line.strip().split("\t")
            if current_line.startswith("!Sample_geo_accession"):
                sample_line = current_line.strip().split("\t")
        case_cols, control_cols = [], []
        for i, gsm in enumerate(sample_line[1:], 1):
            gsm = gsm.strip('"').strip()
            desc = desc_line[i].strip('"').strip() if i < len(desc_line) else ""
            if "Stroke" in desc or "stroke" in desc:
                case_cols.append(gsm)
            else:
                control_cols.append(gsm)
        avail = set(expr_df.columns)
        case_cols = [c for c in case_cols if c in avail]
        control_cols = [c for c in control_cols if c in avail]
        probe_map = _parse_gpl6883_annotation(GPL6883_ANNOT)
        expr_gene = collapse_probes(expr_df, probe_map)

    elif ds_name == "GSE37587":
        sm_file = find_file(DATA_DIRS["GSE37587"], ["series_matrix"])
        expr_df = parse_series_matrix(sm_file)
        with gzip.open(sm_file, "rt", encoding="latin-1") as f:
            lines = f.readlines()
        sample_line, desc_line = None, None
        for current_line in lines:
            if current_line.startswith("!Sample_geo_accession"):
                sample_line = [x.strip('"').strip() for x in current_line.strip().split("\t")]
            if current_line.startswith("!Sample_description"):
                desc_line = [x.strip('"').strip() for x in current_line.strip().split("\t")]
        case_cols, control_cols = [], []
        for i, gsm in enumerate(sample_line[1:], 1):
            desc = desc_line[i] if i < len(desc_line) else ""
            desc_lower = desc.lower()
            is_baseline = any(
                kw in desc_lower for kw in ["baseline", "hour 0", "0 hour"]
            )
            is_followup = any(
                kw in desc_lower for kw in ["follow-up", "follow up", "followup"]
            )
            if is_followup:
                case_cols.append(gsm)
            elif is_baseline:
                control_cols.append(gsm)
        case_cols = [c for c in case_cols if c in expr_df.columns]
        control_cols = [c for c in control_cols if c in expr_df.columns]
        probe_map = _parse_gpl6883_annotation(GPL6883_ANNOT)
        expr_gene = collapse_probes(expr_df, probe_map)

    elif ds_name == "GSE61616":
        sm_file = find_file(DATA_DIRS["GSE61616"], ["series_matrix"])
        expr_df = parse_series_matrix(sm_file)
        with gzip.open(sm_file, "rt", encoding="latin-1") as f:
            lines = f.readlines()
        sample_acc, sample_title = None, None
        for current_line in lines:
            if current_line.startswith("!Sample_geo_accession"):
                sample_acc = [x.strip('"').strip() for x in current_line.strip().split("\t")]
            if current_line.startswith("!Sample_title"):
                sample_title = [x.strip('"').strip() for x in current_line.strip().split("\t")]
        case_cols, control_cols = [], []
        for i, gsm in enumerate(sample_acc[1:], 1):
            title = sample_title[i].lower() if i < len(sample_title) else ""
            if "sham" in title:
                control_cols.append(gsm)
            elif any(kw in title for kw in ["mcao", "model", "stroke"]):
                case_cols.append(gsm)
        case_cols = [c for c in case_cols if c in expr_df.columns]
        control_cols = [c for c in control_cols if c in expr_df.columns]
        probe_map = {}
        if os.path.exists(GPL1355_FILE):
            probe_map = parse_gpl1355_annotation(GPL1355_FILE)
        expr_gene = collapse_probes(expr_df, probe_map)

    elif ds_name == "GSE97537":
        sm_file = find_file(DATA_DIRS["GSE97537"], ["series_matrix"])
        expr_df = parse_series_matrix(sm_file)
        with gzip.open(sm_file, "rt", encoding="latin-1") as f:
            lines = f.readlines()
        sample_acc, sample_title = None, None
        for current_line in lines:
            if current_line.startswith("!Sample_geo_accession"):
                sample_acc = [x.strip('"').strip() for x in current_line.strip().split("\t")]
            if current_line.startswith("!Sample_title"):
                sample_title = [x.strip('"').strip() for x in current_line.strip().split("\t")]
        case_cols, control_cols = [], []
        for i, gsm in enumerate(sample_acc[1:], 1):
            title = sample_title[i].lower() if i < len(sample_title) else ""
            if "sham" in title:
                control_cols.append(gsm)
            elif any(kw in title for kw in ["mcao", "model", "stroke"]):
                case_cols.append(gsm)
        case_cols = [c for c in case_cols if c in expr_df.columns]
        control_cols = [c for c in control_cols if c in expr_df.columns]
        probe_map = {}
        if os.path.exists(GPL1355_FILE):
            probe_map = parse_gpl1355_annotation(GPL1355_FILE)
        expr_gene = collapse_probes(expr_df, probe_map)

    elif ds_name == "GSE104036":
        counts_file = Path(DATA_DIRS["GSE104036"]) / "GSE104036_TC-RNAseq_counts.txt.gz"
        if not counts_file.exists():
            counts_file = find_file(DATA_DIRS["GSE104036"], ["counts", "txt"])
            if counts_file:
                counts_file = Path(counts_file)
        if counts_file and counts_file.exists():
            expr_gene = pd.read_csv(
                str(counts_file), sep="\t", index_col=0, compression="gzip"
            )
        else:
            sm_file = find_file(DATA_DIRS["GSE104036"], ["series_matrix"])
            expr_gene = parse_series_matrix(sm_file)
        expr_gene.columns = [c.strip('"').strip() for c in expr_gene.columns]
        expr_gene.index = [str(idx).strip('"').strip() for idx in expr_gene.index]
        expr_gene.index = expr_gene.index.str.upper()
        flat = expr_gene.values.flatten()
        flat = flat[~np.isnan(flat)]
        if (
            np.max(flat) > 50
            and np.median(flat) > 5
            and np.mean(flat == np.floor(flat)) > 0.5
        ):
            col_sums = expr_gene.sum()
            cpm = expr_gene.div(col_sums, axis=1) * 1e6
            expr_gene = np.log2(cpm + 1)
        all_cols = expr_gene.columns.tolist()
        control_cols = sorted(
            [
                c
                for c in all_cols
                if re.match(r"^S\d+", str(c)) or "sham" in str(c).lower()
            ]
        )
        ipsi_candidates = [
            c
            for c in all_cols
            if "sham" not in str(c).lower() and not re.match(r"^C\d+", str(c))
        ]
        case_cols = ipsi_candidates
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    logger.info(
        f"  [{ds_name}] 加载完成: {expr_gene.shape}, case={len(case_cols)}, ctrl={len(control_cols)}"
    )
    return expr_gene, case_cols, control_cols


# ============================================================
# Step 1: 导出表达矩阵 CSV 供 R WGCNA 使用
# ============================================================


def export_expression_for_wgcna():
    """导出各数据集全基因表达矩阵"""
    logger.info("===== Step 1: 导出表达矩阵 CSV =====")
    for ds_name in DATA_DIRS:
        try:
            expr_gene, case_cols, control_cols = load_expression_matrix(ds_name)
            # 使用所有样本 (case + control)
            all_cols = [c for c in case_cols + control_cols if c in expr_gene.columns]
            if len(all_cols) < 5:
                logger.warning(f"  [{ds_name}] 样本数不足: {len(all_cols)}, 跳过")
                continue
            expr_sub = expr_gene[all_cols].copy()
            # 去除全 NaN 行
            expr_sub = expr_sub.dropna(how="all")
            # 去除重复行名 (gene symbols)
            expr_sub = expr_sub[~expr_sub.index.duplicated(keep="first")]
            # 确保 index 是字符串
            expr_sub.index = expr_sub.index.astype(str)
            out_file = L2_WGCNA_INPUT / f"{ds_name}_expr.csv"
            expr_sub.to_csv(out_file)
            logger.info(f"  [{ds_name}] 已导出: {expr_sub.shape}, {out_file}")
        except Exception as e:
            logger.error(f"  [{ds_name}] 导出失败: {e}")


# ============================================================
# Step 2: 调用 R WGCNA 脚本
# ============================================================


def run_r_wgcna():
    """调用 R 脚本执行 WGCNA + modulePreservation"""
    logger.info("\n===== Step 2: 执行 R WGCNA + modulePreservation =====")
    r_script = IRON_AGING_DIR / "L2_WGCNA_modulePreservation.R"
    if not r_script.exists():
        logger.error(f"R 脚本不存在: {r_script}")
        return False

    cmd = [RSCRIPT, str(r_script)]
    logger.info(f"  执行: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(IRON_AGING_DIR),
        timeout=3600,
        encoding="utf-8",
        errors="replace",
    )
    logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if result.returncode != 0:
        logger.error(f"R 脚本返回错误码: {result.returncode}")
        return False
    logger.info("  R WGCNA 完成")
    return True


# ============================================================
# Step 3: 读取 R 结果 + ACSL4 表达 + GPX4 趋势
# ============================================================


def read_wgcna_results():
    """读取 R WGCNA 输出"""
    results = {}
    for ds_name in DATA_DIRS:
        assign_file = L2_WGCNA_OUTPUT / f"{ds_name}_module_assignment.csv"
        if assign_file.exists():
            results[ds_name] = pd.read_csv(assign_file)
            logger.info(f"  [{ds_name}] 模块分配: {len(results[ds_name])} 基因")

    # 读取汇总
    summary_file = L2_WGCNA_OUTPUT / "L2_WGCNA_summary.csv"
    if summary_file.exists():
        summary = pd.read_csv(summary_file)
        logger.info(f"\n  WGCNA 汇总:\n{summary.to_string()}")
    return results


def acsl4_expression_analysis(expr_gene, ds_name, case_cols, control_cols, dual_scores):
    """ACSL4 在 ISP 高/低组中的表达差异"""
    if "ACSL4" not in expr_gene.index:
        return None

    ds_scores = dual_scores[dual_scores["dataset"] == ds_name].copy()
    if ds_scores.empty:
        return None

    valid_idsp = ds_scores["idsp_index"].dropna()
    if len(valid_idsp) < 4:
        return None

    q75, q25 = valid_idsp.quantile(0.75), valid_idsp.quantile(0.25)
    available_cols = set(expr_gene.columns)

    # 获取高 ISP 和低 ISP 的样本名 (从 sample 列)
    high_idx = valid_idsp[valid_idsp >= q75].index
    low_idx = valid_idsp[valid_idsp <= q25].index
    high_samples = ds_scores.loc[high_idx, "sample"].tolist()
    low_samples = ds_scores.loc[low_idx, "sample"].tolist()

    high_cols = [c for c in high_samples if c in available_cols]
    low_cols = [c for c in low_samples if c in available_cols]

    if len(high_cols) < 2 or len(low_cols) < 2:
        return None

    acsl4_high = expr_gene.loc["ACSL4", high_cols].values.astype(float)
    acsl4_low = expr_gene.loc["ACSL4", low_cols].values.astype(float)
    acsl4_high = acsl4_high[~np.isnan(acsl4_high)]
    acsl4_low = acsl4_low[~np.isnan(acsl4_low)]

    _, pval = stats.ttest_ind(acsl4_high, acsl4_low, equal_var=False)
    _, pval_mw = stats.mannwhitneyu(acsl4_high, acsl4_low, alternative="two-sided")

    species = (
        "Human"
        if ds_name in ["GSE16561", "GSE37587"]
        else "Rat"
        if ds_name in ["GSE61616", "GSE97537"]
        else "Mouse"
    )

    return {
        "dataset": ds_name,
        "species": species,
        "acsl4_high_isp_mean": float(np.mean(acsl4_high)),
        "acsl4_low_isp_mean": float(np.mean(acsl4_low)),
        "acsl4_fold_change": float(np.mean(acsl4_high) - np.mean(acsl4_low)),
        "acsl4_isp_pvalue": pval,
        "acsl4_isp_mw_pvalue": pval_mw,
        "n_high_isp": len(acsl4_high),
        "n_low_isp": len(acsl4_low),
        "acsl4_values": list(acsl4_high) + list(acsl4_low),
        "groups": ["High_ISP"] * len(acsl4_high) + ["Low_ISP"] * len(acsl4_low),
    }


def gpx4_trend_analysis(expr_gene, ds_name, dual_scores):
    """GPX4 趋势: ISP中不降 vs 铁死亡中下降"""
    if "GPX4" not in expr_gene.index:
        return None

    ds_scores = dual_scores[dual_scores["dataset"] == ds_name].copy()
    if ds_scores.empty:
        return None

    valid_idsp = ds_scores["idsp_index"].dropna()
    if len(valid_idsp) < 4:
        return None

    q75, _q50, q25 = (
        valid_idsp.quantile(0.75),
        valid_idsp.quantile(0.50),
        valid_idsp.quantile(0.25),
    )
    available_cols = set(expr_gene.columns)

    def get_gpx4(samples):
        cols = [c for c in samples if c in available_cols]
        if len(cols) < 2:
            return np.nan, np.nan, np.nan
        vals = expr_gene.loc["GPX4", cols].values.astype(float)
        vals = vals[~np.isnan(vals)]
        return np.mean(vals), np.std(vals), len(vals)

    high_isp = valid_idsp[valid_idsp >= q75].index
    mid_isp = valid_idsp[(valid_idsp > q25) & (valid_idsp < q75)].index
    low_isp = valid_idsp[valid_idsp <= q25].index

    gpx4_high_mean, gpx4_high_std, n_high = get_gpx4(high_isp)
    gpx4_mid_mean, gpx4_mid_std, n_mid = get_gpx4(mid_isp)
    gpx4_low_mean, gpx4_low_std, n_low = get_gpx4(low_isp)

    if n_high >= 2 and n_low >= 2:
        gpx4_high_vals = expr_gene.loc[
            "GPX4", [c for c in high_isp if c in available_cols]
        ].values.astype(float)
        gpx4_low_vals = expr_gene.loc[
            "GPX4", [c for c in low_isp if c in available_cols]
        ].values.astype(float)
        gpx4_high_vals = gpx4_high_vals[~np.isnan(gpx4_high_vals)]
        gpx4_low_vals = gpx4_low_vals[~np.isnan(gpx4_low_vals)]
        _, gpx4_pval = stats.ttest_ind(gpx4_high_vals, gpx4_low_vals, equal_var=False)
    else:
        gpx4_pval = np.nan

    trend = "UNCHANGED"
    if not np.isnan(gpx4_high_mean) and not np.isnan(gpx4_low_mean):
        diff = gpx4_high_mean - gpx4_low_mean
        if diff > 0.1:
            trend = "UP"
        elif diff < -0.1:
            trend = "DOWN"

    supports_isp = trend != "DOWN" or (not np.isnan(gpx4_pval) and gpx4_pval > 0.05)
    species = (
        "Human"
        if ds_name in ["GSE16561", "GSE37587"]
        else "Rat"
        if ds_name in ["GSE61616", "GSE97537"]
        else "Mouse"
    )

    return {
        "dataset": ds_name,
        "species": species,
        "gpx4_high_isp": float(gpx4_high_mean)
        if not np.isnan(gpx4_high_mean)
        else np.nan,
        "gpx4_mid_isp": float(gpx4_mid_mean) if not np.isnan(gpx4_mid_mean) else np.nan,
        "gpx4_low_isp": float(gpx4_low_mean) if not np.isnan(gpx4_low_mean) else np.nan,
        "gpx4_high_std": float(gpx4_high_std)
        if not np.isnan(gpx4_high_std)
        else np.nan,
        "gpx4_mid_std": float(gpx4_mid_std) if not np.isnan(gpx4_mid_std) else np.nan,
        "gpx4_low_std": float(gpx4_low_std) if not np.isnan(gpx4_low_std) else np.nan,
        "gpx4_trend": trend,
        "gpx4_pvalue": gpx4_pval,
        "supports_isp": supports_isp,
        "n_high": int(n_high) if not np.isnan(n_high) else 0,
        "n_mid": int(n_mid) if not np.isnan(n_mid) else 0,
        "n_low": int(n_low) if not np.isnan(n_low) else 0,
    }


def analyze_all_datasets():
    """对每个数据集执行 ACSL4 + GPX4 分析"""
    logger.info("\n===== Step 3: ACSL4 表达 + GPX4 趋势分析 =====")

    # 加载 L1 双评分
    dual_scores_file = L1_OUTPUT_DIR / "L1_dual_scores_all_datasets.csv"
    if not dual_scores_file.exists():
        logger.error(f"L1 双评分文件不存在: {dual_scores_file}")
        return None, None
    dual_scores = pd.read_csv(dual_scores_file)

    all_acsl4 = []
    all_gpx4 = []
    all_acsl4_data = []  # 供 R 绘图

    for ds_name in DATA_DIRS:
        try:
            expr_gene, case_cols, control_cols = load_expression_matrix(ds_name)

            # ACSL4 表达分析
            acsl4_res = acsl4_expression_analysis(
                expr_gene, ds_name, case_cols, control_cols, dual_scores
            )
            if acsl4_res:
                all_acsl4.append(
                    {
                        k: v
                        for k, v in acsl4_res.items()
                        if k not in ("acsl4_values", "groups")
                    }
                )
                # 详细数据供 R 绘图
                for val, grp in zip(acsl4_res["acsl4_values"], acsl4_res["groups"]):
                    all_acsl4_data.append(
                        {
                            "dataset": ds_name,
                            "species": acsl4_res["species"],
                            "group": grp,
                            "ACSL4_expression": val,
                        }
                    )

            # GPX4 趋势
            gpx4_res = gpx4_trend_analysis(expr_gene, ds_name, dual_scores)
            if gpx4_res:
                all_gpx4.append(gpx4_res)

            logger.info(
                f"  [{ds_name}] ACSL4={'OK' if acsl4_res else 'FAIL'}, GPX4={'OK' if gpx4_res else 'FAIL'}"
            )

        except Exception as e:
            logger.error(f"  [{ds_name}] 分析失败: {e}")

    # 保存 ACSL4 结果
    if all_acsl4:
        acsl4_df = pd.DataFrame(all_acsl4)
        acsl4_df.to_csv(L2_OUTPUT_DIR / "L2_ACSL4_expression_summary.csv", index=False)
        logger.info(f"\n  ACSL4 汇总: {len(all_acsl4)} 数据集")

    if all_acsl4_data:
        acsl4_data_df = pd.DataFrame(all_acsl4_data)
        acsl4_data_df.to_csv(
            L2_OUTPUT_DIR / "L2_ACSL4_expression_data.csv", index=False
        )
        logger.info(f"  ACSL4 详细数据: {len(all_acsl4_data)} 行")

    # 保存 GPX4 结果
    if all_gpx4:
        gpx4_df = pd.DataFrame(all_gpx4)
        gpx4_df.to_csv(L2_OUTPUT_DIR / "L2_GPX4_trend_data.csv", index=False)
        logger.info(f"  GPX4 汇总: {len(all_gpx4)} 数据集")

        # 判定
        n_isp = sum(1 for g in all_gpx4 if g["supports_isp"])
        logger.info(f"\n  GPX4 支持 ISP 假说: {n_isp}/{len(all_gpx4)} 数据集")

    return all_acsl4, all_gpx4


# ============================================================
# Step 4: 保存跨物种热图数据
# ============================================================


def prepare_cross_species_heatmap():
    """准备跨物种基因重叠热图数据"""
    logger.info("\n===== Step 4: 准备跨物种热图数据 =====")

    # 读取 WGCNA 汇总获取 ACSL4 模块信息
    overlap_file = L2_WGCNA_OUTPUT / "L2_cross_species_overlap.csv"
    if overlap_file.exists():
        overlap_df = pd.read_csv(overlap_file)
        overlap_df.to_csv(L2_OUTPUT_DIR / "L2_cross_species_overlap.csv", index=False)
        logger.info(f"  跨物种重叠: {len(overlap_df)} 对比较")
        # 生成 Jaccard 矩阵供热图
        jaccard_pivot = overlap_df.pivot_table(
            index="dataset_a", columns="dataset_b", values="jaccard", aggfunc="first"
        )
        # 填充对称矩阵
        for ds_a in jaccard_pivot.index:
            for ds_b in jaccard_pivot.columns:
                if (
                    pd.isna(jaccard_pivot.loc[ds_a, ds_b])
                    and ds_b in jaccard_pivot.index
                    and ds_a in jaccard_pivot.columns
                ):
                    jaccard_pivot.loc[ds_a, ds_b] = jaccard_pivot.loc[ds_b, ds_a]
        jaccard_pivot.to_csv(L2_OUTPUT_DIR / "L2_jaccard_heatmap_data.csv")
        logger.info(f"  Jaccard 矩阵: {jaccard_pivot.shape}")
    else:
        logger.warning("  无跨物种重叠文件")

    # 也复制 Zsummary 文件（如果有）
    preservation_files = list(L2_WGCNA_OUTPUT.glob("modulePreservation_*.csv"))
    if preservation_files:
        all_pres = []
        for f in preservation_files:
            df = pd.read_csv(f)
            all_pres.append(df)
        combined = pd.concat(all_pres, ignore_index=True)
        combined.to_csv(L2_OUTPUT_DIR / "L2_modulePreservation_all.csv", index=False)
        logger.info(f"  模块保留合并: {len(combined)} 行")


# ============================================================
# 主流程
# ============================================================


def main():
    logger.info("=" * 60)
    logger.info("L2: ACSL4是ISP的核心执行者")
    logger.info("=" * 60)

    # Step 1: 导出表达矩阵
    export_expression_for_wgcna()

    # Step 2: 运行 R WGCNA
    if not run_r_wgcna():
        logger.error("R WGCNA 失败，终止")
        return

    # Step 3: 读取 WGCNA 结果
    read_wgcna_results()

    # Step 4: ACSL4 表达 + GPX4 趋势
    all_acsl4, all_gpx4 = analyze_all_datasets()

    # Step 5: 跨物种热图数据
    prepare_cross_species_heatmap()

    # Step 6: 调用 R 绘图脚本生成图2A-D
    r_plot_script = IRON_AGING_DIR / "L2_plot_figures.R"
    if r_plot_script.exists():
        logger.info("\n===== Step 6: R 绘图 =====")
        result = subprocess.run(
            [RSCRIPT, str(r_plot_script)],
            capture_output=True,
            text=True,
            cwd=str(IRON_AGING_DIR),
            timeout=600,
            encoding="utf-8",
            errors="replace",
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        if result.returncode == 0:
            logger.info("  R 绘图完成")
        else:
            logger.error(f"  R 绘图失败: {result.returncode}")

    logger.info("\n===== L2 Python 完成 =====")
    logger.info("输出文件:")
    for f in sorted(L2_OUTPUT_DIR.glob("*")):
        logger.info(f"  {f.name}")


if __name__ == "__main__":
    main()
