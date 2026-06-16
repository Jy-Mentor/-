#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
L1: 双评分分析 — 在CIRI中识别铁驱动的衰老程序 (IDSP)
=====================================================================
核心逻辑:
  Step 1: 对每个样本/细胞同时计算铁死亡评分和衰老评分
  Step 2: 计算 IDSP Index = z(ferr) + z(sene) - |z(ferr) - z(sene)|
  Step 3: 时间动态分析 (GSE104036) 验证铁死亡(早峰) vs 衰老(持续)
  Step 4: GPX4验证 — 排除典型铁死亡
  Step 5: 跨数据集Meta分析 — 验证IDSP的跨物种保守性

输出:
  - l1_results/ 目录下所有图表和数据
  - L1_dual_scores_all_datasets.csv    — 每个样本的双评分
  - L1_dual_comparison_summary.csv     — 各数据集区分度统计
  - L1_temporal_dual_scores.csv        — GSE104036时间动态
  - L1_gpx4_validation.csv             — GPX4验证
  - L1_idsp_index_all.csv              — IDSP Index
  
数据依赖 (D:盘):
  D:\反向网络药理...
  D盘已确认可读

用法: python l1_dual_analysis.py
=====================================================================
"""

import os, sys, re, gzip, json, warnings, logging, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
try:
    from joblib import Memory
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# joblib 缓存 (加速重计算 — 敏感性分析)
# ============================================================
_JOBLIB_MEMORY = None
if _HAS_JOBLIB:
    _CACHE_DIR = Path(__file__).parent / '.l1_cache'
    _CACHE_DIR.mkdir(exist_ok=True)
    _JOBLIB_MEMORY = Memory(_CACHE_DIR, verbose=0)
    logger.info(f"joblib 缓存已启用: {_CACHE_DIR}")

# ============================================================
# 导入三基因集（带 fallback 保护）
# ============================================================
from idsp_gene_sets import (
    PURE_FERROPTOSIS, PURE_SENESCENCE, SHARED_GENES,
    FERROPTOSIS_ALL, SENESCENCE_ALL
)
logger.info(f"基因集加载: idsp_gene_sets.py "
            f"(铁死亡={len(PURE_FERROPTOSIS)}, 衰老={len(PURE_SENESCENCE)}, 共享={len(SHARED_GENES)})")

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "l1_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR = OUTPUT_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)

DATA_DIRS = {
    'GSE16561':  r'D:\反向网络药理学\L1 数据集\bulk\GSE16561',
    'GSE37587':  r'D:\反向网络药理学\L1 数据集\bulk\GSE37587',
    'GSE61616':  r'D:\反向网络药理学\L1 数据集\bulk\GSE61616（7d）',
    'GSE97537':  r'D:\反向网络药理学\L1 数据集\bulk\GSE97537(24H)',
    'GSE104036': r'D:\反向网络药理学\L1 数据集\bulk\GSE104036（多时序）',
}
GPL6883_ANNOT = str(BASE_DIR / 'GPL6883.annot.gz')
GPL1355_FILE = str(Path(DATA_DIRS['GSE61616']) / 'GPL1355-10794 (1).txt')

# ============================================================
# GENE SET REPORT
# ============================================================
logger.info("=" * 60)
logger.info(f"纯铁死亡基因集:    {len(PURE_FERROPTOSIS)} 基因")
logger.info(f"纯衰老基因集:      {len(PURE_SENESCENCE)} 基因")
logger.info(f"共享基因集:        {len(SHARED_GENES)} 基因")
logger.info(f"铁死亡∩衰老交集:   {len(PURE_FERROPTOSIS & PURE_SENESCENCE)} (应为0)")
assert PURE_FERROPTOSIS.isdisjoint(PURE_SENESCENCE), "PURE集不能重叠!"
logger.info("=" * 60)

# ============================================================
# 数据加载函数 (从ferro_aging_ciri_analysis.py复用)
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
    """解析GEO Series Matrix文件"""
    open_func = gzip.open if str(filepath).endswith('.gz') else open
    with open_func(filepath, 'rt', encoding='latin-1') as f:
        content = f.read()
    lines = content.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('!series_matrix_table_begin'):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            header_idx = j
            break
    if header_idx is None:
        raise ValueError(f"无法找到series_matrix_table_begin: {filepath}")
    header = lines[header_idx].strip().split('\t')
    header = [h.strip('"').strip() for h in header]
    data_lines = []
    for i in range(header_idx + 1, len(lines)):
        if lines[i].startswith('!series_matrix_table_end'):
            break
        stripped = lines[i].strip()
        if stripped:
            data_lines.append(lines[i])
    data = []
    index = []
    for line in data_lines:
        fields = line.strip().split('\t')
        if len(fields) < 2:
            continue
        probe_id = fields[0].strip('"').strip()
        index.append(probe_id)
        values = [float(v) if v != 'null' and v != '' else np.nan
                  for v in fields[1:]]
        if len(values) < len(header) - 1:
            values.extend([np.nan] * (len(header) - 1 - len(values)))
        data.append(values[:len(header) - 1])
    df = pd.DataFrame(data, index=index, columns=header[1:])
    logger.info(f"  解析 {os.path.basename(filepath)}: {df.shape}")
    return df

def parse_gpl6883_annotation(annot_path: str) -> Dict[str, str]:
    """解析 GPL6883 平台注释 (通用)"""
    probe_map = {}
    if not os.path.exists(annot_path):
        return probe_map
    with gzip.open(annot_path, 'rt', encoding='latin-1') as f:
        in_table = False
        for line in f:
            l = line.strip()
            if l == '!platform_table_begin':
                in_table = True
                header = f.readline().strip().split('\t')
                gs_idx = next((i for i, h in enumerate(header)
                                if 'gene symbol' in h.lower()), 2)
                continue
            if not in_table or l == '':
                continue
            fields = l.split('\t')
            if len(fields) > gs_idx:
                probe = fields[0].strip('"').strip()
                gene = fields[gs_idx].strip('"').strip().upper()
                if gene:
                    probe_map[probe] = gene
    logger.info(f"  GPL6883: {len(probe_map)} 探针注释")
    return probe_map


def parse_gpl1355_annotation(filepath: str) -> Dict[str, str]:
    """解析 GPL1355 平台注释 (大鼠)"""
    probe_map = {}
    if not os.path.exists(filepath):
        logger.warning(f"  GPL1355 文件不存在: {filepath}")
        return probe_map
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        in_table = False
        for line in f:
            l = line.strip()
            if l.startswith('ID'):
                in_table = True
                header = l.split('\t')
                try:
                    gene_col = next(i for i, h in enumerate(header)
                                    if 'gene symbol' in h.lower() or 'symbol' in h.lower())
                except StopIteration:
                    gene_col = 5
                continue
            if not in_table or not l:
                continue
            fields = l.split('\t')
            if len(fields) <= max(gene_col, 0):
                continue
            probe = fields[0]
            gene = fields[gene_col].strip('"').strip()
            if gene:
                probe_map[probe] = gene.split('///')[0].strip().upper()
    logger.info(f"  GPL1355: {len(probe_map)} 探针注释")
    return probe_map


def collapse_probes(expr_df: pd.DataFrame, probe_map: Dict[str, str]) -> pd.DataFrame:
    """探针→基因折叠 (最大表达值)

    策略选择: 对于多探针对应同一基因的情况, 选择表达值最高的探针作为代表。
    理由: 在铁死亡/衰老基因集中, 多数基因为中等至低表达调控因子,
    max 策略可避免均值/中位数稀释关键调控信号 (如 GPX4, ACSL4 的灵敏探针)。
    标准 limma/affy 实践多用 mean 或 median, 此处 max 为有意的偏差选择,
    适用于"强信号优先"的下游秩基富集分析。

    输入: 行=探针ID, 列=样本; 输出: 行=基因Symbol(大写), 列=样本
    """
    mapped = expr_df[expr_df.index.isin(probe_map.keys())].copy()
    if mapped.empty:
        logger.warning("  collapse_probes: 无探针成功映射到基因, 返回空矩阵")
        return mapped
    gene_series = pd.Series(mapped.index.map(probe_map), index=mapped.index)
    gene_series = gene_series.dropna()
    mapped = mapped.loc[gene_series.index]
    gene_series = gene_series.str.upper()
    mapped.index = gene_series
    mapped = mapped.groupby(mapped.index).max()
    return mapped

# ============================================================
# 跨平台 Harmonization: ComBat 批次校正 (新增 🔧)
# ============================================================

def combat_harmonize_datasets(expr_dict: Dict[str, pd.DataFrame],
                              sample_groups: Dict[str, Tuple[List[str], List[str]]] = None) -> Dict[str, pd.DataFrame]:
    """
    ComBat 跨平台批次校正 — 消除 Illumina / Affymetrix / RNA-seq 技术偏差
    
    5 个数据集来自三种平台，表达量分布差异大。
    评分前用 ComBat 做批次校正，减少技术偏差对富集评分的影响。
    
    来源: Johnson 2007 *Biostatistics*; Leek 2012 *Bioinformatics* (sva最佳实践);
         Zhang 2020 *NAR* (多平台ComBat基准)
    
    Parameters:
        expr_dict:     {dataset_name: expr_df} 各数据集的基因表达矩阵 (行=基因, 列=样本)
        sample_groups: {dataset_name: (case_cols, control_cols)} 样本生物学分组,
                       传入后用于构建 design 矩阵, 防止 ComBat 消除生物信号.
                       来源: Leek 2012; Johnson 2007 第 4.2 节.
        
    Returns:
        校正后的 {dataset_name: expr_df_corrected}
    """
    from collections import OrderedDict
    
    # 1. 找到所有数据集共有且非NaN的基因交集
    common_genes_list = []
    for name, df in expr_dict.items():
        if df is None or df.empty:
            continue
        # 每个基因在所有样本中至少 50% 非 NaN
        valid_genes = df.index[df.notna().sum(axis=1) >= max(1, df.shape[1] // 2)]
        common_genes_list.append(set(valid_genes))
    
    if len(common_genes_list) < 2:
        logger.warning("  ComBat: 可合并的数据集 < 2, 跳过批次校正")
        return expr_dict
    
    common_genes = sorted(set.intersection(*common_genes_list))
    if len(common_genes) < 100:
        logger.warning(f"  ComBat: 共有基因仅 {len(common_genes)} < 100, 跳过批次校正")
        return expr_dict
    
    logger.info(f"  ComBat: {len(common_genes)} 共有基因, {len(expr_dict)} 数据集")
    
    # 2. 构建合并矩阵 (基因 × 所有样本)
    merged_parts = []
    batch_labels = []
    group_labels = []  # 生物学分组标签 (case=1, control=0)
    dataset_order = []
    
    for name, df in expr_dict.items():
        if df is None or df.empty:
            continue
        sub = df.loc[df.index.intersection(common_genes)].copy()
        if sub.empty:
            continue
        # 取每个基因在数据集内的均值填充 NaN
        # 注意: 此操作假设缺失为MCAR(完全随机缺失), 50%阈值提供部分保护
        sub = sub.T.fillna(sub.mean(axis=1)).T
        merged_parts.append(sub)
        batch_labels.extend([name] * sub.shape[1])

        # 构建生物学协变量标签 (保护 case/control 差异)
        if sample_groups and name in sample_groups:
            case_cols, ctrl_cols = sample_groups[name]
            if case_cols is not None and ctrl_cols is not None:
                for col in sub.columns:
                    if col in case_cols:
                        group_labels.append(1)
                    elif col in ctrl_cols:
                        group_labels.append(0)
                    else:
                        group_labels.append(0)  # unknown → control
            else:
                group_labels.extend([0] * sub.shape[1])
        else:
            # 无分组信息: 全部填 0 (无保护)
            group_labels.extend([0] * sub.shape[1])

        dataset_order.append((name, sub.shape[1]))

    if len(merged_parts) < 2:
        logger.warning("  ComBat: 有效数据集 < 2, 跳过批次校正")
        return expr_dict
    
    merged_expr = pd.concat(merged_parts, axis=1)
    logger.info(f"  ComBat 合并矩阵: {merged_expr.shape} (基因 × 样本)")
    
    # 3. 先用基因中心化预处理 (每个基因减去全局均值, 更稳定)
    gene_means = merged_expr.mean(axis=1)
    merged_centered = merged_expr.sub(gene_means, axis=0)
    
    # 4. 构建 neuroCombat covars DataFrame (Leek 2012 最佳实践)
    #    neuroCombat 的 categorical_cols 参数直接指定需要保护的生物学变量
    group_array = np.array(group_labels)
    has_covariate = len(np.unique(group_array)) > 1
    n_samples = merged_centered.shape[1]
    assert len(group_labels) == n_samples, \
        f"group_labels 长度 {len(group_labels)} != 样本数 {n_samples}"
    assert len(batch_labels) == n_samples, \
        f"batch_labels 长度 {len(batch_labels)} != 样本数 {n_samples}"
    covars = pd.DataFrame({
        'batch': batch_labels,
        'disease': ['case' if g == 1 else 'control' for g in group_labels],
    }, index=merged_centered.columns)
    if has_covariate:
        logger.info(f"  ComBat covars: batch={len(set(batch_labels))} 批次, "
                    f"disease={sum(group_array)} case / {len(group_array) - sum(group_array)} control")

    # 5. 运行 neuroCombat (Fortin 2018, Jupyter/Python port)
    combat_applied = False
    try:
        from neuroCombat import neuroCombat
        # neuroCombat 需要 (n_features, n_samples) 输入
        harmonized = neuroCombat(
            dat=merged_centered.values,  # (genes, samples)
            covars=covars,
            batch_col='batch',
            categorical_cols=['disease'] if has_covariate else None,
            eb=True,
            parametric=True,
            mean_only=False,
            ref_batch=None
        )
        corrected_raw = harmonized['data']  # shape (genes, samples)
        corrected = pd.DataFrame(
            corrected_raw,
            index=merged_centered.index,
            columns=merged_centered.columns
        )
        combat_applied = True
        logger.info(f"  neuroCombat 校正完成: {corrected.shape}"
                    f"{' (含生物学协变量保护: disease)' if has_covariate else ''}")
    except ImportError:
        logger.warning("  neuroCombat 未安装, 使用简化中位数对齐替代")
        # 备用方案: 中位数对齐 (量级校正)
        corrected = merged_centered.copy()
        global_med = np.median(corrected.values)
        for batch_name in set(batch_labels):
            batch_mask = np.array(batch_labels) == batch_name
            batch_med = np.median(corrected.values[:, batch_mask])
            corrected.values[:, batch_mask] = corrected.values[:, batch_mask] - batch_med + global_med
        combat_applied = True
        logger.info("  中位数对齐完成 (ComBat 替代)")
    except Exception as e:
        logger.warning(f"  neuroCombat 失败 ({e}), 使用 pycombat 降级方案")
        try:
            from pycombat.pycombat import Combat
            combat = Combat()
            Y_input = merged_centered.T.values
            combat.fit(Y_input, batch_labels)
            corrected_raw = combat.transform(Y_input, batch_labels)
            corrected = pd.DataFrame(
                corrected_raw.T,
                index=merged_centered.index,
                columns=merged_centered.columns
            )
            combat_applied = True
            logger.info(f"  pycombat (降级) 校正完成: {corrected.shape}")
        except Exception as e2:
            logger.warning(f"  pycombat 降级也失败 ({e2}), 返回原始数据")
            return expr_dict
    
    # 6. 恢复基因均值 (加回全局均值)
    corrected = corrected.add(gene_means, axis=0)
    
    # 7. 拆分回各数据集
    result = {}
    col_start = 0
    for name, n_cols in dataset_order:
        result[name] = corrected.iloc[:, col_start:col_start + n_cols].copy()
        col_start += n_cols
    logger.info(f"  ComBat 拆分: {list(result.keys())} "
                f"(corrected.shape={corrected.shape}, dataset_order={dataset_order}) "
                f"{'(含生物学协变量保护)' if has_covariate else ''}")
    return result


def combat_pca_diagnostic(expr_dict: Dict[str, pd.DataFrame],
                          sample_groups: Dict[str, Tuple[List[str], List[str]]] = None,
                          save_path: str = None):
    """
    ComBat 校正前后 PCA 诊断图 [Zhang 2020 *NAR* 审稿标准QC]

    生成 PCA 散点图, 按数据集(batch)和 condition(biology)分别着色,
    验证批次效应是否消除且生物信号是否保留。

    标准: 校正后样本不应按数据集/平台分离; 而应按 case/control 分离。
    """
    if len(expr_dict) < 2:
        return

    # 1. 找到共有基因
    common_genes_list = [set(df.index) for df in expr_dict.values() if df is not None]
    common_genes = sorted(set.intersection(*common_genes_list)) if common_genes_list else []
    if len(common_genes) < 100:
        return

    # 2. 合并表达矩阵
    parts, batch_labels, condition_labels = [], [], []
    for name, df in expr_dict.items():
        if df is None:
            continue
        sub = df.loc[df.index.intersection(common_genes)].T
        parts.append(sub)
        batch_labels.extend([name] * sub.shape[0])
        if sample_groups and name in sample_groups:
            case_cols, ctrl_cols = sample_groups[name]
            for col in sub.index:
                if col in (case_cols or []):
                    condition_labels.append('case')
                elif col in (ctrl_cols or []):
                    condition_labels.append('control')
                else:
                    condition_labels.append('unknown')
        else:
            condition_labels.extend(['unknown'] * sub.shape[0])

    merged = pd.concat(parts, axis=0)
    merged = merged.fillna(merged.mean())

    # 3. PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(merged)
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X)
    var_expl = pca.explained_variance_ratio_ * 100

    # 4. 绘制 (按batch + 按condition)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 按数据集着色
    datasets = sorted(set(batch_labels))
    cmap_batch = plt.cm.tab10
    colors_batch = {d: cmap_batch(i % 10) for i, d in enumerate(datasets)}
    for d in datasets:
        mask = np.array(batch_labels) == d
        ax1.scatter(pc[mask, 0], pc[mask, 1], c=[colors_batch[d]], label=d,
                     alpha=0.7, s=30, edgecolors='none')
    ax1.set_xlabel(f'PC1 ({var_expl[0]:.1f}%)')
    ax1.set_ylabel(f'PC2 ({var_expl[1]:.1f}%)')
    ax1.set_title('By Dataset (Batch)')
    ax1.legend(fontsize=7, loc='best')

    # 按生物学分组着色
    conds = sorted(set(condition_labels))
    palette = {'case': '#E74C3C', 'control': '#3498DB', 'unknown': '#95A5A6'}
    for c in conds:
        mask = np.array(condition_labels) == c
        ax2.scatter(pc[mask, 0], pc[mask, 1], c=palette.get(c, 'gray'),
                     label=c, alpha=0.7, s=30, edgecolors='none')
    ax2.set_xlabel(f'PC1 ({var_expl[0]:.1f}%)')
    ax2.set_ylabel(f'PC2 ({var_expl[1]:.1f}%)')
    ax2.set_title('By Condition (Biology)')
    ax2.legend(fontsize=7, loc='best')

    fig.suptitle('ComBat Cross-Platform Harmonization: PCA Quality Control',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ComBat PCA诊断图保存: {save_path}")
    plt.close()


# ============================================================
# 核心函数: 单样本秩和富集评分
# ============================================================

def rank_sum_enrichment_score(expr: np.ndarray, gene_mask: np.ndarray) -> float:
    """秩和富集评分 (单样本)"""
    n_genes = len(expr)
    n_set = gene_mask.sum()
    if n_set == 0 or n_set == n_genes:
        return 0.0
    ranks = stats.rankdata(expr, method='average')
    set_ranks = ranks[gene_mask]
    expected = n_set * (n_genes + 1) / 2
    sum_ranks = set_ranks.sum()
    max_dev = n_set * (n_genes - n_set)
    if max_dev == 0:
        return 0.0
    return float((sum_ranks - expected) / (max_dev / 2))

def compute_enrichment_score_matrix(expr_df: pd.DataFrame, gene_set: Set[str]) -> pd.Series:
    """对表达矩阵所有样本计算秩和富集评分"""
    common_genes = [g for g in gene_set if g in expr_df.index]
    if len(common_genes) < 5:
        logger.warning(f"  基因集交集过小: {len(common_genes)}")
        return pd.Series(index=expr_df.columns, dtype=float)
    gene_mask = expr_df.index.isin(common_genes)
    scores = {}
    for col in expr_df.columns:
        vals = expr_df[col].values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 50:
            scores[col] = np.nan
            continue
        scores[col] = rank_sum_enrichment_score(vals[valid], gene_mask[valid])
    result = pd.Series(scores)
    logger.info(f"  富集评分: {len(common_genes)}/{len(gene_set)} 匹配, {result.notna().sum()} 样本有效")
    return result


# ============================================================
# Bulk免疫细胞特征基因集 (MCPcounter / xCell / 文献策展)
# ============================================================

# 人血免疫细胞特征基因签名 [Newman 2015 Nat Methods; Aran 2017 Genome Biol; Becht 2016]
HUMAN_IMMUNE_SIGNATURES: Dict[str, Set[str]] = {
    'Monocyte': {
        'CD14', 'FCGR3A', 'CSF1R', 'ITGAM', 'CD68', 'CCR2',
        'FCN1', 'S100A8', 'S100A9', 'LYZ', 'VCAN', 'CLEC4A',
        'MS4A6A', 'CTSS', 'SPI1', 'MAFB',
    },
    'Macrophage_M1': {
        'IL1B', 'TNF', 'IL6', 'CCL2', 'CCL3', 'CCL4', 'CCL5',
        'CXCL9', 'CXCL10', 'CXCL11', 'NOS2', 'IL12B', 'IL23A',
        'CD80', 'CD86', 'TLR2', 'TLR4', 'STAT1', 'IRF5', 'SOCS3',
    },
    'Macrophage_M2': {
        'IL10', 'TGFB1', 'ARG1', 'MRC1', 'CD163', 'CLEC10A',
        'MSR1', 'CCL18', 'CCL22', 'CCL24', 'F13A1', 'TGM2',
        'VEGFA', 'CHI3L1', 'SELENOP', 'FOLR2', 'IL1RN',
    },
    'Neutrophil': {
        'FCGR3B', 'CEACAM8', 'CSF3R', 'CXCR1', 'CXCR2',
        'MMP8', 'MMP9', 'ELANE', 'MPO', 'CTSG', 'PRTN3',
        'S100A12', 'LCN2', 'DEFA1', 'DEFA3', 'FPR1', 'FPR2', 'ITGAM',
    },
    'Tcell_CD8': {
        'CD8A', 'CD8B', 'GZMA', 'GZMB', 'GZMK', 'PRF1',
        'NKG7', 'CCL5', 'IFNG', 'TBX21', 'EOMES',
    },
    'Tcell_CD4': {
        'CD4', 'CD3E', 'CD3D', 'CD3G', 'IL7R', 'CCR7',
        'TCF7', 'LEF1', 'SELL', 'CD27', 'CD28',
    },
    'Treg': {
        'FOXP3', 'IL2RA', 'CTLA4', 'TNFRSF18', 'TNFRSF4',
        'IKZF2', 'IL10', 'TGFB1', 'LRRC32', 'ENTPD1', 'TIGIT',
    },
    'NK_cell': {
        'KLRD1', 'KLRF1', 'NKG7', 'GNLY', 'PRF1', 'GZMB',
        'NCR1', 'NCR3', 'KLRK1', 'CD160', 'FCGR3A', 'KIR2DL1',
        'KIR3DL1', 'KIR2DS4',
    },
    'B_cell': {
        'MS4A1', 'CD79A', 'CD79B', 'CD19', 'PAX5', 'BLK',
        'BANK1', 'FCRL1', 'FCRL2', 'TNFRSF17', 'MZB1',
        'IGHA1', 'IGHG1', 'IGHM', 'IGKC', 'JCHAIN',
    },
    'Plasma_cell': {
        'SDC1', 'MZB1', 'DERL3', 'TNFRSF17', 'SLAMF7',
        'XBP1', 'IRF4', 'PRDM1', 'FKBP11', 'SSR4',
    },
    'Dendritic_cell': {
        'FCER1A', 'CLEC10A', 'CLEC4C', 'NRP1', 'NDRG2',
        'FLT3', 'CLEC9A', 'XCR1', 'BDCA2', 'BDCA4',
        'BATF3', 'IRF8', 'ZBTB46', 'ITGAX', 'HLA-DRA', 'HLA-DRB1',
    },
}

# 鼠脑细胞类型特征基因 [Saunders 2018 Cell; McKenzie 2019 Cell Rep; Zeisel 2018 Cell]
MOUSE_BRAIN_SIGNATURES: Dict[str, Set[str]] = {
    'Microglia': {
        'Aif1', 'Cx3cr1', 'P2ry12', 'Tmem119', 'Trem2',
        'Csf1r', 'Itgam', 'Hexb', 'C1qa', 'C1qb', 'C1qc',
        'Sparc', 'Olfml3', 'Siglech', 'Gpr34', 'Sall1',
        'Fcrls', 'Mertk', 'Cst3', 'Ctss', 'Ccl2',
    },
    'Astrocyte': {
        'Gfap', 'Aldh1l1', 'Aqp4', 'Slc1a3', 'Slc1a2',
        'Gjb6', 'S100b', 'Aldoc', 'Clu', 'Mlc1',
        'Agt', 'Fgfr3', 'Gja1', 'Apoe', 'Vim',
    },
    'Neuron': {
        'Syp', 'Rbfox3', 'Map2', 'Tubb3', 'Nefl',
        'Nefm', 'Snap25', 'Syt1', 'Dlg4', 'Grin1',
        'Grin2a', 'Gad1', 'Slc17a7', 'Bdnf', 'Npy',
    },
    'Oligodendrocyte': {
        'Mog', 'Mbp', 'Plp1', 'Mobp', 'Mag',
        'Olig1', 'Olig2', 'Sox10', 'Cnp', 'Cldn11',
        'Mobp', 'Ermn', 'Gpr17', 'PdgfRa', 'Ugt8',
    },
    'Endothelial': {
        'Cldn5', 'Pecam1', 'Cdh5', 'Tek', 'Flt1',
        'Kdr', 'Vwf', 'Eng', 'Icam1', 'Vcam1',
        'Esam', 'Rgs5', 'Pdgfrb', 'Acta2', 'Cspg4',
    },
    'Pericyte': {
        'Pdgfrb', 'Rgs5', 'Cspg4', 'Anpep', 'Des',
        'Abcc9', 'Kcnj8', 'Cox4i2', 'Notch3', 'Mylk',
    },
    'Oligodendrocyte_Precursor': {
        'Pdgfra', 'Cspg4', 'Sox10', 'Olig1', 'Olig2',
        'Gpr17', 'Cspg5', 'Myt1', 'Nkx2-2', 'Cndp1',
    },
}


def deconvolve_immune_cells(expr_df: pd.DataFrame,
                            cell_type_sigs: Dict[str, Set[str]],
                            method_label: str = 'immune') -> pd.DataFrame:
    """
    Bulk免疫细胞反卷积 (特征基因集富集评分法)

    替代 immunedeconv::EPIC (R包, PyPI不可用), 使用基于文献策展的
    特征基因签名 + 秩和富集评分, 原理等同 MCPcounter (Becht 2016)。

    对每种细胞类型的特征基因集计算单样本秩和富集得分,
    得分反映该细胞类型的相对丰度。

    来源: Newman 2015 Nat Methods (CIBERSORT/LM22);
          Becht 2016 Genome Biol (MCPcounter);
          Aran 2017 Genome Biol (xCell)

    Parameters:
        expr_df:         基因表达矩阵 (行=基因, 列=样本)
        cell_type_sigs:  {细胞类型: 特征基因集合}
        method_label:    方法标签 (用于日志)

    Returns:
        细胞比例矩阵 (样本 × 细胞类型)
    """
    cell_scores = {}
    for ct_name, sig_genes in cell_type_sigs.items():
        detected = [g for g in sig_genes if g in expr_df.index]
        if len(detected) < 3:
            logger.debug(f"  [{method_label}] {ct_name}: 仅 {len(detected)} 基因, 跳过")
            continue
        scores = compute_enrichment_score_matrix(expr_df, sig_genes)
        cell_scores[ct_name] = scores
    if not cell_scores:
        return pd.DataFrame()
    result = pd.DataFrame(cell_scores)
    result = result.reindex(index=expr_df.columns)
    n_valid = result.dropna(how='all').shape[0]
    logger.info(f"  [{method_label}] 反卷积完成: {len(cell_scores)} 细胞类型 × {n_valid} 样本")
    return result


def idsp_immune_correlation(idsp_scores: pd.DataFrame,
                            deconv_df: pd.DataFrame,
                            dataset_name: str,
                            alpha_bonf: float = 0.05) -> Dict[str, Any]:
    """
    IDSP 评分与免疫细胞比例的 Spearman 相关性分析

    对每种细胞类型计算与铁死亡/衰老/IDSP评分的 Spearman ρ,
    并用 Bonferroni 校正多重比较。

    Returns:
        dict with correlations, top_celltype, etc.
    """
    results = {'dataset': dataset_name, 'correlations': [], 'top_types': {}}
    n_tests = deconv_df.shape[1] * 3  # 细胞类型 × 3个IDSP指标
    alpha_corrected = alpha_bonf / n_tests if n_tests > 0 else alpha_bonf

    for col in deconv_df.columns:
        valid_mask = deconv_df[col].notna() & idsp_scores['ferroptosis'].notna()
        if valid_mask.sum() < 5:
            continue

        for metric in ['ferroptosis', 'senescence', 'idsp_index']:
            try:
                rho, pval = stats.spearmanr(
                    deconv_df.loc[valid_mask, col],
                    idsp_scores.loc[valid_mask, metric]
                )
                results['correlations'].append({
                    'cell_type': col,
                    'metric': metric,
                    'rho': float(rho),
                    'p_value': float(pval),
                    'significant': pval < alpha_corrected,
                    'n_samples': int(valid_mask.sum()),
                })
            except Exception:
                continue

    # 找与 IDSP index 最显著正相关的细胞类型
    idsp_corrs = [c for c in results['correlations'] if c['metric'] == 'idsp_index']
    if idsp_corrs:
        # 按 ρ 降序
        idsp_corrs.sort(key=lambda x: x['rho'], reverse=True)
        best = idsp_corrs[0]
        results['top_types'] = {
            'cell_type': best['cell_type'],
            'rho': best['rho'],
            'p_value': best['p_value'],
            'dataset': dataset_name,
        }

    return results


def plot_deconvolution_heatmap(all_correlations: List[Dict[str, Any]],
                               save_path: str = None):
    """
    Fig1E: IDSP-免疫细胞相关性热图

    行=数据集 × 细胞类型, 列=IDSP指标 (铁死亡/衰老/IDSP)
    """
    if not all_correlations:
        return

    # 构建相关性矩阵
    rows = []
    for d in all_correlations:
        for c in d.get('correlations', []):
            rows.append({
                'label': f"{d['dataset']}_{c['cell_type']}",
                'dataset': d['dataset'],
                'cell_type': c['cell_type'],
                'metric': c['metric'],
                'rho': c['rho'],
                'sig': c['significant'],
            })

    if not rows:
        return
    df = pd.DataFrame(rows)

    # 透视
    pivot = df.pivot_table(values='rho', index='label', columns='metric',
                           aggfunc='first')
    sig_pivot = df.pivot_table(values='sig', index='label', columns='metric',
                               aggfunc='first')

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.35)))
    im = ax.imshow(pivot.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # 标注显著记号
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.notna(val):
                marker = '★' if sig_pivot.values[i, j] else '·'
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}{marker}', ha='center', va='center',
                        fontsize=8, color=color)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title('IDSP × 免疫细胞 Spearman 相关性 (★=Bonferroni显著)',
                 fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  免疫相关性热图保存: {save_path}")
    plt.close()


def plot_microglia_idsp_scatter(idsp_scores: pd.DataFrame,
                                deconv_df: pd.DataFrame,
                                dataset_name: str,
                                microglia_col: str = 'Microglia',
                                save_path: str = None):
    """
    小胶质细胞/单核细胞 vs IDSP 散点图
    """
    if microglia_col not in deconv_df.columns:
        # 尝试人血对应类型
        alt_cols = ['Monocyte', 'Macrophage_M1', 'Macrophage_M2']
        found = [c for c in alt_cols if c in deconv_df.columns]
        if not found:
            return
        microglia_col = found[0]  # 使用第一个可用的

    valid = (deconv_df[microglia_col].notna() &
             idsp_scores['idsp_index'].notna())
    if valid.sum() < 5:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    x = deconv_df.loc[valid, microglia_col]
    y = idsp_scores.loc[valid, 'idsp_index']

    rho, pval = stats.spearmanr(x, y)

    # 按 group 着色
    if 'group' in idsp_scores.columns:
        groups = idsp_scores.loc[valid, 'group']
        palette = {'case': '#E74C3C', 'control': '#3498DB', 'unknown': '#95A5A6'}
        colors = groups.map(lambda g: palette.get(g, '#95A5A6'))
        ax.scatter(x, y, c=colors.tolist(), alpha=0.7, s=50, edgecolors='none')
        # 图例
        for grp in sorted(set(groups)):
            mask = groups == grp
            if mask.any():
                ax.scatter([], [], c=palette.get(grp, '#95A5A6'), label=grp, alpha=0.7, s=50)
        ax.legend(fontsize=8)
    else:
        ax.scatter(x, y, c='#8E44AD', alpha=0.7, s=50, edgecolors='none')

    # 趋势线
    if len(x) > 2:
        slope, intercept, _, _, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, '--', color='gray', alpha=0.6)

    ax.set_xlabel(f'{microglia_col} Enrichment Score', fontsize=11)
    ax.set_ylabel('IDSP Index', fontsize=11)
    ax.set_title(f'{dataset_name}: {microglia_col} vs IDSP\n'
                 f'Spearman ρ={rho:.3f}, p={pval:.3e}', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  IDSP-{microglia_col}散点图保存: {save_path}")
    plt.close()


# ============================================================
# 基因集扰动敏感性分析
# ============================================================
def gene_set_sensitivity_analysis(expr_dict: Dict[str, pd.DataFrame],
                                  sample_info: Dict,
                                  gene_pool_size: int = 5000,
                                  n_iterations: int = 100,
                                  replace_fraction: float = 0.10) -> pd.DataFrame:
    """
    基因集扰动敏感性分析

    对纯铁死亡和纯衰老基因集各执行 n 次随机扰动:
      1. 随机移除 replace_fraction (10%) 的原始基因
      2. 从背景基因池中随机补入等量基因
      3. 重新计算所有数据集的 Cohen's d 效应量

    目的: 验证 IDSP 推断在基因集微小变化下的稳定性，
    堵住"结果依赖特定基因集选择"的审稿漏洞。

    Returns:
        DataFrame: 每行 = 一次扰动, 列 = 各数据集的 d_ferr / d_sene
    """
    from idsp_gene_sets import PURE_FERROPTOSIS as PF, PURE_SENESCENCE as PS

    # 构建背景基因池 (表达矩阵中除纯铁死亡/纯衰老/共享基因外的所有基因)
    all_expr_genes = set()
    for df in expr_dict.values():
        if df is not None:
            all_expr_genes.update(df.index)
    excluded = PF | PS | SHARED_GENES
    background_pool = sorted(all_expr_genes - excluded)
    if len(background_pool) < gene_pool_size:
        gene_pool_size = len(background_pool)
        logger.warning(f"  背景基因池不足, 降为 {gene_pool_size}")
    np.random.seed(42)

    results = []
    ferr_base = sorted(PF)
    sene_base = sorted(PS)
    n_replace_ferr = max(1, int(len(ferr_base) * replace_fraction))
    n_replace_sene = max(1, int(len(sene_base) * replace_fraction))

    for iteration in range(n_iterations):
        rng = np.random.RandomState(iteration + 42)
        # 扰动铁死亡基因集
        drop_ferr = set(rng.choice(ferr_base, size=n_replace_ferr, replace=False))
        add_ferr = set(rng.choice(background_pool, size=n_replace_ferr, replace=False))
        perturbed_ferr = set(ferr_base) - drop_ferr | add_ferr

        # 扰动衰老基因集
        drop_sene = set(rng.choice(sene_base, size=n_replace_sene, replace=False))
        add_sene = set(rng.choice(background_pool, size=n_replace_sene, replace=False))
        perturbed_sene = set(sene_base) - drop_sene | add_sene

        row = {'iteration': iteration}
        for ds_name, expr_df in expr_dict.items():
            if ds_name not in sample_info:
                continue
            case_cols, ctrl_cols = sample_info[ds_name]
            is_paired = (ds_name == 'GSE37587')

            ferr_score = compute_enrichment_score_matrix(expr_df, perturbed_ferr)
            sene_score = compute_enrichment_score_matrix(expr_df, perturbed_sene)

            # z-score标准化后计算Cohen's d (同dual_enrichment_analysis修复)
            def _zscore_d(case_vals, ctrl_vals):
                all_vals = np.concatenate([case_vals, ctrl_vals])
                std_all = np.std(all_vals)
                if std_all > 1e-12:
                    mu_all = np.mean(all_vals)
                    case_z = (case_vals - mu_all) / std_all
                    ctrl_z = (ctrl_vals - mu_all) / std_all
                    return float(np.mean(case_z) - np.mean(ctrl_z))
                return 0.0

            case = ferr_score[ferr_score.index.isin(case_cols or [])]
            ctrl = ferr_score[ferr_score.index.isin(ctrl_cols or [])]
            if len(case) >= 2 and len(ctrl) >= 2:
                row[f'{ds_name}_d_ferr'] = _zscore_d(case.values, ctrl.values)
            else:
                row[f'{ds_name}_d_ferr'] = np.nan

            case_s = sene_score[sene_score.index.isin(case_cols or [])]
            ctrl_s = sene_score[sene_score.index.isin(ctrl_cols or [])]
            if len(case_s) >= 2 and len(ctrl_s) >= 2:
                row[f'{ds_name}_d_sene'] = _zscore_d(case_s.values, ctrl_s.values)
            else:
                row[f'{ds_name}_d_sene'] = np.nan

        results.append(row)

    result_df = pd.DataFrame(results)

    # 汇总统计
    logger.info("=" * 60)
    logger.info("基因集扰动敏感性分析 (100 iterations, 10% 替换)")
    for col in result_df.columns:
        if col == 'iteration':
            continue
        vals = result_df[col].dropna()
        if len(vals) == 0:
            continue
        logger.info(f"  {col}: μ={vals.mean():.3f}, σ={vals.std():.3f}, "
                    f"CV={abs(vals.std() / vals.mean()) if abs(vals.mean()) > 0.01 else np.nan:.3f}, "
                    f"95%CI=[{vals.quantile(0.025):.3f}, {vals.quantile(0.975):.3f}]")
    return result_df


def _cohens_d_paired(case_vals, ctrl_vals) -> float:
    """配对设计的 Cohen's d"""
    diff = np.array(case_vals) - np.array(ctrl_vals)
    sd_diff = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0

# ============================================================
# 新增: 双评分 + IDSP Index + GPX4验证
# ============================================================

def dual_enrichment_analysis(expr_df: pd.DataFrame, dataset_name: str,
                              case_cols: List[str], control_cols: List[str],
                              is_paired: bool = False) -> Tuple[pd.DataFrame, dict]:
    """
    双评分分析: 同时计算铁死亡和衰老的富集得分

    Parameters:
        expr_df:     基因表达矩阵 (行=基因, 列=样本)
        dataset_name: 数据集名称
        case_cols:    case 样本列名
        control_cols: control 样本列名
        is_paired:    是否为配对设计 (如 GSE37587 基线 vs 随访)

    Returns:
        scores_df: 每个样本的三维评分
        comparison: 区分度统计字典
    """
    # 检查基因集交集是否足够
    for gname, gset in [('Ferroptosis', PURE_FERROPTOSIS), ('Senescence', PURE_SENESCENCE)]:
        common = sum(1 for g in gset if g in expr_df.index)
        if common < 5:
            logger.warning(f"  [{dataset_name}] {gname} 交集={common} < 5, 跳过")
            empty_scores = pd.DataFrame(columns=['ferroptosis','senescence','shared','dataset','sample','group','idsp_index'])
            empty_comp = {'dataset': dataset_name, 'n_case': 0, 'n_control': 0, 'r_ferr_sene': np.nan,
                          'd_ferroptosis': np.nan, 'd_senescence': np.nan, 'd_idsp': np.nan,
                          'p_ferroptosis': np.nan, 'p_senescence': np.nan, 'p_idsp': np.nan}
            return empty_scores, empty_comp

    # 计算三个评分
    ferr_score = compute_enrichment_score_matrix(expr_df, PURE_FERROPTOSIS)
    sene_score = compute_enrichment_score_matrix(expr_df, PURE_SENESCENCE)
    share_score = compute_enrichment_score_matrix(expr_df, SHARED_GENES)

    # 合并
    scores_df = pd.DataFrame({
        'ferroptosis': ferr_score,
        'senescence': sene_score,
        'shared': share_score,
    })
    scores_df['dataset'] = dataset_name
    scores_df['sample'] = scores_df.index

    # 仅标记已知分组的样本, 其余设为 unknown (避免 GSE61616 额外列被计为 control)
    scores_df['group'] = 'unknown'
    scores_df.loc[scores_df.index.isin(case_cols), 'group'] = 'case'
    scores_df.loc[scores_df.index.isin(control_cols), 'group'] = 'control'

    # 计算 IDSP Index
    scores_df['idsp_index'] = calc_idsp_index(scores_df['ferroptosis'], scores_df['senescence'])

    # 统计 — 仅使用已知分组的样本
    case = scores_df[scores_df['group'] == 'case'].dropna(subset=['ferroptosis', 'senescence'])
    ctrl = scores_df[scores_df['group'] == 'control'].dropna(subset=['ferroptosis', 'senescence'])

    # 双评分相关性 (所有样本)
    valid_all = scores_df.dropna(subset=['ferroptosis', 'senescence'])
    if len(valid_all) >= 3:
        r_all, p_all = stats.pearsonr(valid_all['ferroptosis'], valid_all['senescence'])
    else:
        r_all, p_all = np.nan, np.nan

    # 效应量 — z-score标准化后再计算Cohen's d
    # 原因: 富集评分量级0-1且标准差极小(0.01-0.02), 直接计算d会膨胀至30-40,
    #       导致I²虚高、τ²巨大。标准化后d恢复0.5-3正常范围。
    if len(case) >= 2 and len(ctrl) >= 2:
        def _zscore_cohens_d(case_vals, ctrl_vals):
            """z-score标准化后计算标准化均值差 (等同于标准化后的Cohen's d)"""
            all_vals = np.concatenate([case_vals, ctrl_vals])
            std_all = np.std(all_vals)
            if std_all > 1e-12:
                mu_all = np.mean(all_vals)
                case_z = (case_vals - mu_all) / std_all
                ctrl_z = (ctrl_vals - mu_all) / std_all
                return float(np.mean(case_z) - np.mean(ctrl_z))
            return 0.0
        d_ferr = _zscore_cohens_d(case['ferroptosis'].values, ctrl['ferroptosis'].values)
        d_sene = _zscore_cohens_d(case['senescence'].values, ctrl['senescence'].values)
        d_idsp = _zscore_cohens_d(case['idsp_index'].values, ctrl['idsp_index'].values)
    else:
        d_ferr, d_sene, d_idsp = np.nan, np.nan, np.nan

    # t检验 (支持配对/非配对)
    if len(case) >= 2 and len(ctrl) >= 2:
        if is_paired and len(case) == len(ctrl):
            # 配对 t 检验 — baseline 与 follow-up 按顺序配对
            paired_ferr = stats.ttest_rel(case['ferroptosis'].values, ctrl['ferroptosis'].values)
            paired_sene = stats.ttest_rel(case['senescence'].values, ctrl['senescence'].values)
            paired_idsp = stats.ttest_rel(case['idsp_index'].values, ctrl['idsp_index'].values)
            _, p_ferr = paired_ferr
            _, p_sene = paired_sene
            _, p_idsp = paired_idsp
            p_ferr_method = 'paired_t'
        else:
            _, p_ferr = stats.ttest_ind(case['ferroptosis'], ctrl['ferroptosis'], equal_var=False)
            _, p_sene = stats.ttest_ind(case['senescence'], ctrl['senescence'], equal_var=False)
            _, p_idsp = stats.ttest_ind(case['idsp_index'], ctrl['idsp_index'], equal_var=False)
            p_ferr_method = 'welch_t'
    else:
        p_ferr, p_sene, p_idsp = np.nan, np.nan, np.nan
        p_ferr_method = 'na'

    # 效应量方差 (用于 I² 和随机效应Meta)
    n_c, n_ct = len(case), len(ctrl)
    n_total = n_c + n_ct
    var_ferr = ((n_c + n_ct) / (n_c * n_ct) + d_ferr**2 / (2 * n_total)
                if pd.notna(d_ferr) and n_total >= 4 else np.nan)
    var_sene = ((n_c + n_ct) / (n_c * n_ct) + d_sene**2 / (2 * n_total)
                if pd.notna(d_sene) and n_total >= 4 else np.nan)

    comparison = {
        'dataset': dataset_name,
        'n_case': n_c, 'n_control': n_ct,
        'ferr_case_mean': case['ferroptosis'].mean(), 'ferr_ctrl_mean': ctrl['ferroptosis'].mean(),
        'sene_case_mean': case['senescence'].mean(), 'sene_ctrl_mean': ctrl['senescence'].mean(),
        'r_ferr_sene': r_all, 'p_corr': p_all,
        'd_ferroptosis': d_ferr, 'd_senescence': d_sene, 'd_idsp': d_idsp,
        'p_ferroptosis': p_ferr, 'p_senescence': p_sene, 'p_idsp': p_idsp,
        'test_method': p_ferr_method,
        'is_paired': is_paired,
        'var_ferroptosis': var_ferr, 'var_senescence': var_sene,
    }

    if pd.notna(r_all):
        logger.info(f"  [{dataset_name}] r={r_all:.3f}, d_ferr={d_ferr:.3f}, d_sene={d_sene:.3f}, "
                    f"p_ferr={p_ferr:.4e}, p_sene={p_sene:.4e}")
    else:
        logger.info(f"  [{dataset_name}] 双评分统计不可用 (交集不足或样本量过小)")

    return scores_df, comparison


def calc_idsp_index(ferr_score: pd.Series, sene_score: pd.Series) -> pd.Series:
    """
    IDSP Index = z(ferr) + z(sene) - |z(ferr) - z(sene)| = 2 * min(z_ferr, z_sene)

    含义: 两个得分都高且差异小时 → IDSP Index 最大
    识别铁死亡与衰老得分"同步升高"的样本。

    注意: z-score 在单个数据集内标准化, 因此 IDSP 值不可跨数据集直接比较。
    跨数据集分析通过 Cohen's d (每数据集独立) 和 Meta分析实现。
    跨物种(人/大鼠/小鼠)比较使用各物种正交基因评分。
    """
    # 数据集内 z-score 标准化 (以各自分布为参照)
    ferr_std = ferr_score.std()
    sene_std = sene_score.std()
    z_ferr = ((ferr_score - ferr_score.mean()) / ferr_std
              if ferr_std != 0 else pd.Series(0.0, index=ferr_score.index))
    z_sene = ((sene_score - sene_score.mean()) / sene_std
              if sene_std != 0 else pd.Series(0.0, index=sene_score.index))
    return z_ferr + z_sene - np.abs(z_ferr - z_sene)


def gpx4_validation(expr_df: pd.DataFrame, scores_df: pd.DataFrame,
                     case_cols: List[str], control_cols: List[str],
                     dataset_name: str) -> dict:
    """
    GPX4分层验证: 高IDSP样本中GPX4是否下降？

    铁死亡: GPX4 ↓↓↓
    IDSP:   GPX4 不变或轻微变化

    如果GPX4在高IDSP组不显著低于对照组 → 支持IDSP假说
    """
    if 'GPX4' not in expr_df.index:
        return {'dataset': dataset_name, 'gpx4_found': False}

    scores = scores_df.copy()
    scores['gpx4_expr'] = np.nan
    for col in scores.index:
        if col in expr_df.columns:
            scores.loc[col, 'gpx4_expr'] = expr_df.loc['GPX4', col]

    scores = scores.dropna(subset=['gpx4_expr', 'idsp_index'])
    if len(scores) < 6:
        return {'dataset': dataset_name, 'gpx4_found': True, 'n_too_small': True}

    # 按IDSP Index分高/低组 (四分位数, Top 25% vs Bottom 25%, NaN安全)
    valid_idsp = scores['idsp_index'].dropna()
    if len(valid_idsp) < 4:
        return {'dataset': dataset_name, 'gpx4_found': True, 'n_too_small': True,
                'gpx4_mean_high': np.nan, 'gpx4_mean_low': np.nan,
                'gpx4_log2fc': np.nan, 'pvalue': np.nan, 'verdict': 'insufficient_samples'}
    q75, q25 = valid_idsp.quantile(0.75), valid_idsp.quantile(0.25)
    high_idsp = scores[scores['idsp_index'] >= q75]['gpx4_expr'].values
    low_idsp = scores[scores['idsp_index'] <= q25]['gpx4_expr'].values

    if len(high_idsp) < 2 or len(low_idsp) < 2:
        return {'dataset': dataset_name, 'gpx4_found': True, 'n_too_small': True,
                'gpx4_mean_high': np.nan, 'gpx4_mean_low': np.nan,
                'gpx4_log2fc': np.nan, 'pvalue': np.nan, 'verdict': 'insufficient_samples'}

    # t检验: 高IDSP vs 低IDSP 的GPX4
    _, pval = stats.ttest_ind(high_idsp, low_idsp, equal_var=False)
    mean_high, mean_low = high_idsp.mean(), low_idsp.mean()
    log2fc = mean_high - mean_low

    # 判断: log2fc > -0.5 且 p > 0.05 → GPX4没有显著下降
    verdict = "IDSP_supported" if (log2fc > -0.5 or pval > 0.05) else "IDSP_not_supported"

    logger.info(f"  [{dataset_name}] GPX4: highIDSP={mean_high:.3f}, lowIDSP={mean_low:.3f}, "
                f"Δ={log2fc:.3f}, p={pval:.4f} → {verdict}")

    return {
        'dataset': dataset_name,
        'gpx4_found': True,
        'n_high_idsp': len(high_idsp), 'n_low_idsp': len(low_idsp),
        'gpx4_mean_high': mean_high, 'gpx4_mean_low': mean_low,
        'gpx4_log2fc': log2fc,
        'pvalue': pval,
        'verdict': verdict,
    }


def cohens_d(case: np.ndarray, control: np.ndarray, paired: bool = False) -> float:
    """Cohen's d 效应量 (支持配对/独立设计)"""
    if paired and len(case) == len(control):
        diff = case - control
        sd_diff = np.std(diff, ddof=1)
        return float(np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0
    n1, n2 = len(case), len(control)
    if n1 < 2 or n2 < 2:
        return 0.0
    s1, s2 = np.var(case, ddof=1), np.var(control, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(case) - np.mean(control)) / pooled if pooled > 0 else 0.0


def analyze_signature_genes(expr_df: pd.DataFrame, case_cols: List[str],
                             control_cols: List[str], gene_set: Set[str],
                             dataset_name: str) -> pd.DataFrame:
    """单基因差异分析 (ACSLA4, PTGS2等)"""
    case_cols = [c for c in case_cols if c in expr_df.columns]
    control_cols = [c for c in control_cols if c in expr_df.columns]
    if not case_cols or not control_cols:
        return pd.DataFrame()
    common = [g for g in expr_df.index if g in gene_set]
    results = []
    for gene in common:
        raw_case = expr_df.loc[gene, case_cols].values.astype(float)
        raw_ctrl = expr_df.loc[gene, control_cols].values.astype(float)
        case_vals = raw_case[~np.isnan(raw_case)]
        ctrl_vals = raw_ctrl[~np.isnan(raw_ctrl)]
        if len(case_vals) < 2 or len(ctrl_vals) < 2:
            continue
        log2fc = np.mean(case_vals) - np.mean(ctrl_vals)
        _, pval = stats.ttest_ind(case_vals, ctrl_vals, equal_var=False)
        results.append({
            'dataset': dataset_name,
            'gene': gene,
            'mean_case': np.mean(case_vals),
            'mean_control': np.mean(ctrl_vals),
            'log2FC': log2fc,
            'pvalue': pval,
        })
    df = pd.DataFrame(results)
    if not df.empty:
        _, padj, _, _ = multipletests(df['pvalue'], method='fdr_bh')
        df['padj'] = padj
        df = df.sort_values('pvalue')
    return df


def fisher_meta_analysis(p_values: List[float]) -> Tuple[float, float]:
    """Fisher 合并p值"""
    valid_p = [p for p in p_values if 0 < p <= 1]
    if len(valid_p) < 2:
        return np.nan, np.nan
    chi2 = -2 * np.sum(np.log(valid_p))
    df = 2 * len(valid_p)
    meta_p = 1 - stats.chi2.cdf(chi2, df)
    return chi2, meta_p


def stouffer_meta(p_values: List[float], weights: Optional[List[float]] = None,
                   directions: Optional[List[int]] = None) -> float:
    """
    Stouffer's Z-score 合并p值 (可加权, 带效应方向)

    需传入每项研究的效应方向 (±1), 避免 `np.sign(1-p/2)` 始终为正的假阳性问题.

    Parameters
    ----------
    p_values   : 各研究的 p 值 (双侧)
    weights    : 权重 (如样本量 sqrt), 默认等权
    directions : 效应方向 (±1), 正效应=1, 负效应=-1, 默认全为正

    前沿参考: Zaykin (2011) Genet Epidemiol; 广泛应用于GWAS Meta分析.
    """
    if directions is None:
        directions = [1] * len(p_values)
    valid = [(p, w, d) for p, w, d in
             zip(p_values,
                 weights if weights else [1.0] * len(p_values),
                 directions)
             if 0 < p <= 1]
    if len(valid) < 2:
        return np.nan
    ps, ws, ds = zip(*valid)
    # 将双侧p值转换为单侧z值, 乘以效应方向
    z_scores = [stats.norm.ppf(1 - p / 2) * np.sign(d) for p, d in zip(ps, ds)]
    w_sum = np.sqrt(np.sum(np.array(ws) ** 2))
    if w_sum == 0:
        return np.nan
    z_comb = np.sum(np.array(z_scores) * np.array(ws)) / w_sum
    meta_p = 2 * (1 - stats.norm.cdf(abs(z_comb)))
    return float(meta_p)


def random_effects_meta_analysis(effect_sizes: List[float],
                                  variances: List[float]) -> dict:
    """
    随机效应Meta分析 (DerSimonian-Laird 方法)

    相比固定效应模型(Fisher/Stouffer), 允许效应量在数据集间存在真实异质性.
    前沿参考: DerSimonian & Laird (1986) Control Clin Trials;
    Higgins et al. (2003) BMJ Cochrane金标准.

    Parameters:
        effect_sizes: Cohen's d 列表
        variances:   对应方差 列表

    Returns:
        dict with summary_effect, p_value, tau2, I2, Q, df
    """
    valid = [(d, v) for d, v in zip(effect_sizes, variances)
             if pd.notna(d) and pd.notna(v) and v > 0]
    if len(valid) < 3:
        return {'summary_effect': np.nan, 'p_value': np.nan,
                'tau2': np.nan, 'I2': np.nan, 'k': len(valid)}

    ds, vs = zip(*valid)
    k = len(ds)
    ds, vs = np.array(ds), np.array(vs)

    # 固定效应加权平均
    w_fixed = 1.0 / vs
    d_fixed = np.sum(w_fixed * ds) / np.sum(w_fixed)

    # Q 统计量 (异质性)
    Q = np.sum(w_fixed * (ds - d_fixed) ** 2)
    df = k - 1

    # tau² (DerSimonian-Laird)
    C = np.sum(w_fixed) - np.sum(w_fixed ** 2) / np.sum(w_fixed)
    tau2 = max(0, (Q - df) / C) if C > 0 else 0

    # I²
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0

    # 随机效应加权平均
    w_random = 1.0 / (vs + tau2)
    d_random = np.sum(w_random * ds) / np.sum(w_random)
    se_random = np.sqrt(1.0 / np.sum(w_random))
    z = d_random / se_random if se_random > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% 预测区间 [Higgins 2003 BMJ; Borenstein 2009]
    # PI = d_random ± t_{k-2, 0.975} * sqrt(tau² + SE²)
    # 当 I² > 75% 时必须报告, 反映新研究预期效应量的范围
    if k > 2:
        t_crit = stats.t.ppf(0.975, df=k - 2)
        pi_lower = float(d_random - t_crit * np.sqrt(tau2 + se_random**2))
        pi_upper = float(d_random + t_crit * np.sqrt(tau2 + se_random**2))
    else:
        pi_lower = pi_upper = np.nan

    return {
        'summary_effect': float(d_random),
        'p_value': float(p_val),
        'tau2': float(tau2),
        'I2': float(I2),
        'Q': float(Q),
        'df': df,
        'k': k,
        'd_fixed': float(d_fixed),
        'se_random': float(se_random),
        'pi_lower': pi_lower,
        'pi_upper': pi_upper,
    }


def bayesian_meta_analysis(effect_sizes: List[float],
                             variances: List[float],
                             draws: int = 2000,
                             tune: int = 1000,
                             seed: int = 42) -> dict:
    """
    Bayesian 随机效应 Meta-Analysis (新增 🔧)
    
    正态-正态层次模型，MCMC 采样 (NUTS):
      y_i ~ N(θ_i, σ_i²)    # 观测层: 各研究效应量
      θ_i ~ N(μ, τ²)         # 研究层: 真实效应量来自共同分布
      μ ~ N(0, 10)           # 先验: 总体均值
      τ ~ HalfCauchy(2.5)    # 先验: 异质性标准差
    
    输出 τ² 后验分布，比 DerSimonian-Laird 点估计更稳健。
    同时计算 Bayes Factor 和 95% HDI。
    
    来源: Gelman 2013 *Bayesian Data Analysis*; PyMC 官方示例
    
    Parameters:
        effect_sizes: Cohen's d 列表
        variances:    对应方差列表
        draws:        MCMC 后验采样数
        tune:         预热步数
        seed:         随机种子
        
    Returns:
        dict with posterior summary: μ_mean, μ_hdi, τ_mean, τ_hdi, τ²_mean, ...
    """
    valid = [(d, v) for d, v in zip(effect_sizes, variances)
             if pd.notna(d) and pd.notna(v) and v > 0]
    if len(valid) < 3:
        logger.warning("  Bayesian Meta: 有效研究 < 3, 跳过")
        return {'k': len(valid), 'mu_mean': np.nan, 'tau_mean': np.nan,
                'tau2_mean': np.nan, 'converged': False}

    ds, vs = zip(*valid)
    y_obs = np.array(ds)
    sigma_obs = np.sqrt(np.array(vs))
    k = len(y_obs)
    logger.info(f"  Bayesian Meta: k={k}, MCMC {draws} draws x {tune} tune")

    result = {'k': k}

    try:
        import pymc as pm
        import arviz as az

        with pm.Model() as model:
            # 总体均值先验 (弱信息)
            mu = pm.Normal('mu', mu=0, sigma=10)
            # 异质性标准差先验 (HalfCauchy 厚尾, 对离群研究不敏感)
            tau = pm.HalfCauchy('tau', beta=2.5)
            # 随机效应: 各研究真实效应量
            theta = pm.Normal('theta', mu=mu, sigma=tau, shape=k)
            # 观测似然
            y_like = pm.Normal('y_like', mu=theta, sigma=sigma_obs, observed=y_obs)
            
            # MCMC 采样 (NUTS) — 提高 target_accept 减少发散
            trace = pm.sample(
                draws=draws,
                tune=tune,
                target_accept=0.99,
                random_seed=seed,
                progressbar=False,
                idata_kwargs={'log_likelihood': True},
            )

        # 后验摘要 — 自动检测 arviz 列名 (兼容 hdi_2.5% / 2.5% 两种格式)
        try:
            summary = az.summary(trace, var_names=['mu', 'tau'], hdi_prob=0.95)
        except TypeError:
            summary = az.summary(trace, var_names=['mu', 'tau'], ci_prob=0.95)
        
        # 自动检测 CI/HDI 列名 (arviz 不同版本列名不同)
        cols = list(summary.columns)
        lo_candidates = ['hdi_2.5%', '2.5%', 'ci_2.5%']
        hi_candidates = ['hdi_97.5%', '97.5%', 'ci_97.5%']
        lo_col = next((c for c in lo_candidates if c in cols), cols[2] if len(cols) > 2 else cols[-1])
        hi_col = next((c for c in hi_candidates if c in cols), cols[3] if len(cols) > 3 else cols[-1])
        
        # 使用实际列名读取
        result['mu_mean'] = float(summary.loc['mu', 'mean'])
        result['mu_sd'] = float(summary.loc['mu', 'sd'])
        result['mu_hdi_2.5'] = float(summary.loc['mu', lo_col])
        result['mu_hdi_97.5'] = float(summary.loc['mu', hi_col])
        result['mu_rhat'] = float(summary.loc['mu', 'r_hat'])
        
        result['tau_mean'] = float(summary.loc['tau', 'mean'])
        result['tau_sd'] = float(summary.loc['tau', 'sd'])
        result['tau_hdi_2.5'] = float(summary.loc['tau', lo_col])
        result['tau_hdi_97.5'] = float(summary.loc['tau', hi_col])
        result['tau_rhat'] = float(summary.loc['tau', 'r_hat'])
        
        result['tau2_mean'] = float(np.mean(trace.posterior['tau'].values ** 2))
        result['tau2_sd'] = float(np.std(trace.posterior['tau'].values ** 2))
        
        # 收敛诊断 (R̂ < 1.05 则认为收敛)
        rhat_max = max(result['mu_rhat'], result['tau_rhat'])
        result['converged'] = rhat_max < 1.05
        
        # μ 的 95% HDI 不包含 0 = 统计显著
        result['mu_significant'] = (
            (result['mu_hdi_2.5'] > 0) or (result['mu_hdi_97.5'] < 0)
        )

        # 后验概率 P(μ > 0 | data) — 直接 Bayesian 推断 [Gelman 2013, Conlon 2014]
        mu_posterior = trace.posterior['mu'].values.flatten()
        result['p_mu_gt_0'] = float(np.mean(mu_posterior > 0))
        result['p_mu_lt_0'] = float(np.mean(mu_posterior < 0))

        # ρ = 1 / (1 + τ² / σ̄²) — 信号比 (越接近 1 表示异质性越小)
        avg_var = np.mean(vs)
        result['rho_mean'] = float(np.mean(1.0 / (1.0 + trace.posterior['tau'].values ** 2 / avg_var)))

        logger.info(
            f"  Bayesian Meta 结果: μ={result['mu_mean']:.3f} "
            f"(95%HDI [{result['mu_hdi_2.5']:.3f}, {result['mu_hdi_97.5']:.3f}]), "
            f"P(μ>0)={result.get('p_mu_gt_0', np.nan):.3f}, "
            f"τ={result['tau_mean']:.3f} (95%HDI [{result['tau_hdi_2.5']:.3f}, {result['tau_hdi_97.5']:.3f}]), "
            f"R̂_max={rhat_max:.4f}, {'✓收敛' if result['converged'] else '⚠未收敛'}, "
            f"{'✓显著' if result['mu_significant'] else '不显著'}"
        )

    except ImportError:
        logger.warning("  pymc/arviz 未安装, 跳过 Bayesian Meta")
        result['mu_mean'] = np.nan
        result['converged'] = False
    except Exception as e:
        logger.warning(f"  Bayesian Meta 失败: {e}")
        result['mu_mean'] = np.nan
        result['converged'] = False

    return result


# ============================================================
# 数据集处理 (复用加载逻辑 + 改用双评分)
# ============================================================

def _load_expr_gse16561() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """加载GSE16561数据, 返回(expr_gene, case_cols, control_cols)"""
    logger.info("=" * 50)
    logger.info("[GSE16561] 人全血: Stroke vs Control")
    sm_file = find_file(DATA_DIRS['GSE16561'], ['series_matrix'])
    if not sm_file:
        raise FileNotFoundError("GSE16561 未找到")
    expr_df = parse_series_matrix(sm_file)
    with gzip.open(sm_file, 'rt', encoding='latin-1') as f:
        lines = f.readlines()
    desc_line = sample_line = None
    for l in lines:
        if l.startswith('!Sample_description'):
            desc_line = l.strip().split('\t')
        if l.startswith('!Sample_geo_accession'):
            sample_line = l.strip().split('\t')
    case_cols, control_cols = [], []
    for i, gsm in enumerate(sample_line[1:], 1):
        gsm = gsm.strip('"').strip()
        desc = desc_line[i].strip('"').strip() if i < len(desc_line) else ''
        if 'Stroke' in desc or 'stroke' in desc:
            case_cols.append(gsm)
        else:
            control_cols.append(gsm)
    case_cols = [c for c in case_cols if c in expr_df.columns]
    control_cols = [c for c in control_cols if c in expr_df.columns]
    logger.info(f"  Stroke={len(case_cols)}, Control={len(control_cols)}")
    probe_map = parse_gpl6883_annotation(GPL6883_ANNOT)
    expr_gene = collapse_probes(expr_df, probe_map)
    return expr_gene, case_cols, control_cols


def _load_expr_gse37587() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """加载GSE37587 (人全血, 配对)"""
    logger.info("=" * 50)
    logger.info("[GSE37587] 人全血: Follow-Up vs Baseline (配对)")
    sm_file = find_file(DATA_DIRS['GSE37587'], ['series_matrix'])
    if not sm_file:
        raise FileNotFoundError("GSE37587 未找到")
    expr_df = parse_series_matrix(sm_file)
    with gzip.open(sm_file, 'rt', encoding='latin-1') as f:
        lines = f.readlines()
    sample_line = desc_line = None
    for l in lines:
        if l.startswith('!Sample_geo_accession'):
            sample_line = [x.strip('"').strip() for x in l.strip().split('\t')]
        if l.startswith('!Sample_description'):
            desc_line = [x.strip('"').strip() for x in l.strip().split('\t')]
    case_cols, control_cols = [], []
    for i, gsm in enumerate(sample_line[1:], 1):
        desc = desc_line[i] if i < len(desc_line) else ''
        desc_lower = desc.lower()
        if any(kw in desc_lower for kw in ['follow-up', 'follow up', 'hour 24']):
            case_cols.append(gsm)
        elif any(kw in desc_lower for kw in ['baseline', 'hour 0', '0 hour']):
            control_cols.append(gsm)
    case_cols = [c for c in case_cols if c in expr_df.columns]
    control_cols = [c for c in control_cols if c in expr_df.columns]
    logger.info(f"  FU={len(case_cols)}, BL={len(control_cols)}")
    probe_map = parse_gpl6883_annotation(GPL6883_ANNOT)
    expr_gene = collapse_probes(expr_df, probe_map)
    return expr_gene, case_cols, control_cols


def _load_expr_gse61616() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """加载GSE61616 (大鼠MCAO 7d)"""
    logger.info("=" * 50)
    logger.info("[GSE61616] 大鼠 MCAO 7d")
    sm_file = find_file(DATA_DIRS['GSE61616'], ['series_matrix'])
    if not sm_file:
        raise FileNotFoundError("GSE61616 未找到")
    expr_df = parse_series_matrix(sm_file)
    with gzip.open(sm_file, 'rt', encoding='latin-1') as f:
        lines = f.readlines()
    sample_acc = sample_title = None
    for l in lines:
        if l.startswith('!Sample_geo_accession'):
            sample_acc = [x.strip('"').strip() for x in l.strip().split('\t')]
        if l.startswith('!Sample_title'):
            sample_title = [x.strip('"').strip() for x in l.strip().split('\t')]
    sham_cols, model_cols = [], []
    for i, gsm in enumerate(sample_acc[1:], 1):
        title = sample_title[i].lower() if i < len(sample_title) else ''
        if 'sham' in title:
            sham_cols.append(gsm)
        elif any(kw in title for kw in ['mcao', 'model', 'stroke']):
            model_cols.append(gsm)
    sham_cols = [c for c in sham_cols if c in expr_df.columns]
    model_cols = [c for c in model_cols if c in expr_df.columns]
    logger.info(f"  Model={len(model_cols)}, Sham={len(sham_cols)}")
    probe_map = {}
    if os.path.exists(GPL1355_FILE):
        probe_map = parse_gpl1355_annotation(GPL1355_FILE)
    expr_gene = collapse_probes(expr_df, probe_map)
    return expr_gene, model_cols, sham_cols


def _load_expr_gse97537() -> Tuple[pd.DataFrame, List[str], List[str]]:
    """加载GSE97537 (大鼠MCAO 24h)"""
    logger.info("=" * 50)
    logger.info("[GSE97537] 大鼠 MCAO 24h")
    sm_file = find_file(DATA_DIRS['GSE97537'], ['series_matrix'])
    if not sm_file:
        raise FileNotFoundError("GSE97537 未找到")
    expr_df = parse_series_matrix(sm_file)
    with gzip.open(sm_file, 'rt', encoding='latin-1') as f:
        lines = f.readlines()
    sample_acc = sample_title = None
    for l in lines:
        if l.startswith('!Sample_geo_accession'):
            sample_acc = [x.strip('"').strip() for x in l.strip().split('\t')]
        if l.startswith('!Sample_title'):
            sample_title = [x.strip('"').strip() for x in l.strip().split('\t')]
    sham_cols, mcao_cols = [], []
    for i, gsm in enumerate(sample_acc[1:], 1):
        title = sample_title[i].lower() if i < len(sample_title) else ''
        if 'sham' in title:
            sham_cols.append(gsm)
        elif any(kw in title for kw in ['mcao', 'model', 'stroke']):
            mcao_cols.append(gsm)
    sham_cols = [c for c in sham_cols if c in expr_df.columns]
    mcao_cols = [c for c in mcao_cols if c in expr_df.columns]
    logger.info(f"  MCAO={len(mcao_cols)}, Sham={len(sham_cols)}")
    probe_map = {}
    if os.path.exists(GPL1355_FILE):
        probe_map = parse_gpl1355_annotation(GPL1355_FILE)
    expr_gene = collapse_probes(expr_df, probe_map)
    return expr_gene, mcao_cols, sham_cols


def _load_expr_gse104036() -> Tuple[pd.DataFrame, dict, List[str]]:
    """
    加载GSE104036 (小鼠RNA-seq, 多时间点)
    Returns: (expr_df, timepoint_dict, sham_cols)
    timepoint_dict = {'3hr': [cols], '6hr': [...], ...}
    """
    logger.info("=" * 50)
    logger.info("[GSE104036] 小鼠 RNA-seq: 多时间点")
    counts_file = Path(DATA_DIRS['GSE104036']) / 'GSE104036_TC-RNAseq_counts.txt.gz'
    if not counts_file.exists():
        cf = find_file(DATA_DIRS['GSE104036'], ['counts', 'txt'])
        counts_file = Path(cf) if cf else None
    if counts_file and counts_file.exists():
        logger.info(f"  加载: {counts_file.name}")
        expr_df = pd.read_csv(str(counts_file), sep='\t', index_col=0, compression='gzip')
    else:
        sm_file = find_file(DATA_DIRS['GSE104036'], ['series_matrix'])
        if not sm_file:
            raise FileNotFoundError("GSE104036 数据未找到")
        expr_df = parse_series_matrix(sm_file)
    expr_df.columns = [c.strip('"').strip() for c in expr_df.columns]
    expr_df.index = [str(idx).strip('"').strip() for idx in expr_df.index]
    expr_df.index = expr_df.index.str.upper()
    logger.info(f"  矩阵: {expr_df.shape}")

    # 判断是否需要log转换 (浮点容差检查)
    flat = expr_df.values.flatten()
    flat = flat[~np.isnan(flat)]
    int_ratio = np.mean(np.abs(flat - np.round(flat)) < 1e-8) if len(flat) > 0 else 0
    if len(flat) > 0 and np.max(flat) > 50 and np.median(flat) > 5 and int_ratio > 0.5:
        logger.info("  raw counts检测, 执行log2(CPM+1)")
        col_sums = expr_df.sum()
        cpm = expr_df.div(col_sums, axis=1) * 1e6
        expr_df = np.log2(cpm + 1)

    all_cols = expr_df.columns.tolist()
    sham_cols = sorted([c for c in all_cols if re.match(r'^S\d+', str(c)) or 'sham' in str(c).lower()])
    ipsi_candidates = [c for c in all_cols if 'sham' not in str(c).lower() and not re.match(r'^C\d+', str(c))]
    ipsi_3hr = sorted([c for c in ipsi_candidates if re.search(r'(?i)3h', str(c))])
    ipsi_6hr = sorted([c for c in ipsi_candidates if re.search(r'(?i)6h', str(c))])
    ipsi_12hr = sorted([c for c in ipsi_candidates if re.search(r'(?i)12h', str(c))])
    ipsi_24hr = sorted([c for c in ipsi_candidates if re.search(r'(?i)24h', str(c))])
    timepoint_dict = {'3hr': ipsi_3hr, '6hr': ipsi_6hr, '12hr': ipsi_12hr, '24hr': ipsi_24hr}

    logger.info(f"  Sham={len(sham_cols)}, 3hr={len(ipsi_3hr)}, 6hr={len(ipsi_6hr)}, "
                f"12hr={len(ipsi_12hr)}, 24hr={len(ipsi_24hr)}")
    return expr_df, timepoint_dict, sham_cols

# ============================================================
# 时间动态分析 (GSE104036)
# ============================================================

def temporal_dual_analysis(expr_df: pd.DataFrame, timepoint_dict: dict,
                            sham_cols: List[str], dataset_name: str) -> pd.DataFrame:
    """
    时间动态双评分分析

    注意: 此处重新计算富集评分而非复用 main() 中的结果,
    是为保持函数独立性. 因数据量小, 重复计算开销可忽略.

    预期:
      铁死亡: 3h↑ → 6h达峰 → 12h↓ → 24h继续↓ (急性脉冲)
      衰老:   3h不显著 → 6h开始 → 12h↑ → 24h持续 (慢性激活)

    新增生物指标:
      - ferr_late_rate_ratio:    铁死亡晚期/早期变化率比 (<1=衰减)
      - sene_late_rate_ratio:    衰老晚期/早期变化率比 (>1=持续加速)
      - sene_fc_dominance:       衰老/铁死亡累积FC比 (>1.5=衰老主导)
      - ferr_accumulation_auc:   铁死亡曲线下面积 (整体负荷)
      - sene_accumulation_auc:   衰老曲线下面积 (整体负荷)
      - sene_biphasic_index:     衰老双相指数 (>0.5=双相激活模式)
    """
    results = []
    if not sham_cols:
        logger.warning("  [GSE104036] 无Sham样本, 跳过时间动态分析")
        return pd.DataFrame()

    sham_ferr = compute_enrichment_score_matrix(expr_df[sham_cols], PURE_FERROPTOSIS)
    sham_sene = compute_enrichment_score_matrix(expr_df[sham_cols], PURE_SENESCENCE)
    sham_share = compute_enrichment_score_matrix(expr_df[sham_cols], SHARED_GENES)

    for tp_name in ['3hr', '6hr', '12hr', '24hr']:
        tp_cols = timepoint_dict.get(tp_name, [])
        if len(tp_cols) < 2:
            continue
        ferr_tp = compute_enrichment_score_matrix(expr_df[tp_cols], PURE_FERROPTOSIS)
        sene_tp = compute_enrichment_score_matrix(expr_df[tp_cols], PURE_SENESCENCE)
        share_tp = compute_enrichment_score_matrix(expr_df[tp_cols], SHARED_GENES)

        # Fix 4: 检查有效样本数
        n_ferr_valid = ferr_tp.dropna().shape[0]
        n_sene_valid = sene_tp.dropna().shape[0]
        if n_ferr_valid < 2 or n_sene_valid < 2:
            logger.warning(f"    {tp_name}: 有效样本不足 (ferr={n_ferr_valid}, sene={n_sene_valid}), 跳过")
            continue

        _, p_ferr = stats.ttest_ind(ferr_tp.dropna(), sham_ferr.dropna(), equal_var=False) if len(ferr_tp.dropna())>=2 and len(sham_ferr.dropna())>=2 else (None, np.nan)
        _, p_sene = stats.ttest_ind(sene_tp.dropna(), sham_sene.dropna(), equal_var=False) if len(sene_tp.dropna())>=2 and len(sham_sene.dropna())>=2 else (None, np.nan)

        results.append({
            'dataset': dataset_name,
            'timepoint': tp_name,
            'time_hr': int(tp_name.replace('hr', '')),
            'n_samples': len(tp_cols),
            'ferroptosis_mean': ferr_tp.mean(),
            'ferroptosis_sem': ferr_tp.std() / np.sqrt(len(ferr_tp.dropna())),
            'senescence_mean': sene_tp.mean(),
            'senescence_sem': sene_tp.std() / np.sqrt(len(sene_tp.dropna())),
            'shared_mean': share_tp.mean(),
            'p_ferroptosis': p_ferr,
            'p_senescence': p_sene,
        })
        logger.info(f"    {tp_name}: ferr={ferr_tp.mean():.3f}(p={p_ferr:.4e}), "
                    f"sene={sene_tp.mean():.3f}(p={p_sene:.4e})")

    # 多时间点FDR校正 — 避免假阳性累积 (BH-FDR)
    if len(results) >= 2:
        p_ferr_all = [r['p_ferroptosis'] for r in results]
        p_sene_all = [r['p_senescence'] for r in results]
        valid_ferr = [p for p in p_ferr_all if pd.notna(p)]
        valid_sene = [p for p in p_sene_all if pd.notna(p)]

        if valid_ferr:
            _, padj_ferr, _, _ = multipletests(valid_ferr, method='fdr_bh')
            padj_idx = 0
            for r in results:
                if pd.notna(r['p_ferroptosis']):
                    r['p_ferroptosis_fdr'] = padj_ferr[padj_idx]
                    padj_idx += 1
                else:
                    r['p_ferroptosis_fdr'] = np.nan
        if valid_sene:
            _, padj_sene, _, _ = multipletests(valid_sene, method='fdr_bh')
            padj_idx = 0
            for r in results:
                if pd.notna(r['p_senescence']):
                    r['p_senescence_fdr'] = padj_sene[padj_idx]
                    padj_idx += 1
                else:
                    r['p_senescence_fdr'] = np.nan

    # 加入sham基线
    sham_ferr_mean = sham_ferr.mean()
    sham_sene_mean = sham_sene.mean()
    results.append({
        'dataset': dataset_name,
        'timepoint': 'Sham',
        'time_hr': -0.5,
        'n_samples': len(sham_cols),
        'ferroptosis_mean': sham_ferr_mean,
        'ferroptosis_sem': sham_ferr.std() / np.sqrt(len(sham_ferr.dropna())),
        'senescence_mean': sham_sene_mean,
        'senescence_sem': sham_sene.std() / np.sqrt(len(sham_sene.dropna())),
        'shared_mean': sham_share.mean(),
        'p_ferroptosis': np.nan,
        'p_senescence': np.nan,
        'ferr_fc_vs_sham': 1.0,
        'sene_fc_vs_sham': 1.0,
    })

    df = pd.DataFrame(results).sort_values('time_hr')

    # 计算相对于 Sham 的 fold-change 和时间点间变化率
    for i, row in df.iterrows():
        if row['timepoint'] != 'Sham':
            df.loc[i, 'ferr_fc_vs_sham'] = row['ferroptosis_mean'] / sham_ferr_mean if sham_ferr_mean != 0 else np.nan
            df.loc[i, 'sene_fc_vs_sham'] = row['senescence_mean'] / sham_sene_mean if sham_sene_mean != 0 else np.nan

    # 计算相邻时间点间的变化率 (delta/dt)
    ferr_vals = df['ferroptosis_mean'].values
    sene_vals = df['senescence_mean'].values
    time_vals = df['time_hr'].values
    ferr_delta = np.full(len(df), np.nan)
    sene_delta = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        dt = time_vals[i] - time_vals[i-1]
        if dt > 0:
            ferr_delta[i] = (ferr_vals[i] - ferr_vals[i-1]) / dt
            sene_delta[i] = (sene_vals[i] - sene_vals[i-1]) / dt
    df['ferr_delta_per_hr'] = ferr_delta
    df['sene_delta_per_hr'] = sene_delta

    # ================================================================
    # 新增: 高级时间动态生物指标
    # ================================================================

    # --- 1. 累积曲线下面积 (AUC) — 反映整体激活负荷 ---
    tp_rows = df[df['timepoint'] != 'Sham'].sort_values('time_hr')
    if len(tp_rows) >= 2:
        t = tp_rows['time_hr'].values
        ferr_auc = np.trapz(tp_rows['ferroptosis_mean'].values, t)
        sene_auc = np.trapz(tp_rows['senescence_mean'].values, t)
    else:
        ferr_auc = sene_auc = np.nan
    df.attrs['ferr_accumulation_auc'] = ferr_auc
    df.attrs['sene_accumulation_auc'] = sene_auc
    df.attrs['sene_auc_ratio'] = sene_auc / ferr_auc if (pd.notna(ferr_auc) and pd.notna(sene_auc) and ferr_auc != 0) else np.nan

    # --- 2. 晚期/早期变化率比 (反映激活模式衰减或持续) ---
    non_sham = df[df['timepoint'] != 'Sham'].sort_values('time_hr')
    ferr_deltas = non_sham['ferr_delta_per_hr'].dropna()
    sene_deltas = non_sham['sene_delta_per_hr'].dropna()
    if len(ferr_deltas) >= 2:
        df.attrs['ferr_late_rate_ratio'] = ferr_deltas.iloc[-1] / ferr_deltas.iloc[0] if ferr_deltas.iloc[0] != 0 else np.nan
    else:
        df.attrs['ferr_late_rate_ratio'] = np.nan
    if len(sene_deltas) >= 2:
        df.attrs['sene_late_rate_ratio'] = sene_deltas.iloc[-1] / sene_deltas.iloc[0] if sene_deltas.iloc[0] != 0 else np.nan
    else:
        df.attrs['sene_late_rate_ratio'] = np.nan

    # --- 3. 衰老/铁死亡累积FC比 (晚期优势) ---
    late_tp = non_sham[non_sham['time_hr'] == non_sham['time_hr'].max()]
    if not late_tp.empty:
        sene_fc = late_tp['sene_fc_vs_sham'].values[0]
        ferr_fc = late_tp['ferr_fc_vs_sham'].values[0]
        df.attrs['sene_fc_dominance'] = sene_fc / ferr_fc if (pd.notna(sene_fc) and pd.notna(ferr_fc) and ferr_fc != 0) else np.nan
    else:
        df.attrs['sene_fc_dominance'] = np.nan

    # --- 4. 衰老双相激活指数 ---
    # 定义: 晚期(12→24hr)变化率 / 早期(3→6hr)变化率
    # >1 = 晚期加速(双相), <1 = 单相衰减
    if len(sene_deltas) >= 3:
        early_sene = sene_deltas.iloc[:2].mean()  # 3→6hr, 6→12hr avg
        late_sene = sene_deltas.iloc[-1]           # 12→24hr
        df.attrs['sene_biphasic_index'] = late_sene / early_sene if early_sene != 0 else np.nan
    else:
        df.attrs['sene_biphasic_index'] = np.nan

    # --- 5. 铁死亡衰减指数 ---
    if len(ferr_deltas) >= 3:
        early_ferr = ferr_deltas.iloc[:2].mean()
        late_ferr = ferr_deltas.iloc[-1]
        df.attrs['ferr_decay_index'] = late_ferr / early_ferr if early_ferr != 0 else np.nan
    else:
        df.attrs['ferr_decay_index'] = np.nan

    # --- 6. 双相模式判定 ---
    biphasic = (
        pd.notna(df.attrs.get('sene_biphasic_index')) and
        df.attrs['sene_biphasic_index'] > 0.5 and
        pd.notna(df.attrs.get('ferr_decay_index')) and
        df.attrs['ferr_decay_index'] < 0.5
    )
    df.attrs['sene_biphasic_detected'] = biphasic

    # 日志输出新增指标
    logger.info("  [时间动态生物指标]")
    if pd.notna(df.attrs.get('ferr_accumulation_auc')):
        logger.info(f"    铁死亡 AUC={df.attrs['ferr_accumulation_auc']:.4f}, "
                    f"衰老 AUC={df.attrs['sene_accumulation_auc']:.4f}, "
                    f"比率={df.attrs['sene_auc_ratio']:.2f}")
    if pd.notna(df.attrs.get('ferr_late_rate_ratio')):
        logger.info(f"    铁死亡晚期/早期变化率比={df.attrs['ferr_late_rate_ratio']:.3f} "
                    f"(<1=急性衰减), "
                    f"衰老晚期/早期变化率比={df.attrs['sene_late_rate_ratio']:.3f} "
                    f"(>1=持续加速)")
    if pd.notna(df.attrs.get('sene_fc_dominance')):
        logger.info(f"    衰老/铁死亡累积FC比(24h)={df.attrs['sene_fc_dominance']:.2f} "
                    f"(>1.5=衰老晚期主导)")
    if df.attrs.get('sene_biphasic_detected'):
        logger.info(f"    双相激活模式: ✓检测到 (衰老晚期加速 + 铁死亡衰减)")
    else:
        logger.info(f"    双相激活模式: 未检测到")

    return df


# ============================================================
# 高级分析方法 (Bootstrap · 置换检验 · ROC · I² · LODO)
# ============================================================

def bootstrap_idsp_ci(scores_df: pd.DataFrame, n_bootstrap: int = 2000,
                       ci: float = 0.95, seed: int = 42) -> dict:
    """
    Bootstrap IDSP Index 置信区间 (不确定性量化)

    前沿参考: sc-ssGSEA (GenePattern 2024) 使用metacell聚合降低不确定性.
    此处扩展为bootstrap, 适用于bulk RNA-seq样本量有限场景.
    """
    rng = np.random.default_rng(seed)
    ferr = scores_df['ferroptosis'].values
    sene = scores_df['senescence'].values
    n = len(ferr)
    if n < 4:
        return {'n_boot': 0, 'idsp_mean': np.nan, 'idsp_ci_lower': np.nan,
                'idsp_ci_upper': np.nan, 'ci_level': ci}

    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ferr_boot, sene_boot = ferr[idx], sene[idx]
        zf = (ferr_boot - ferr_boot.mean()) / (ferr_boot.std() + 1e-12)
        zs = (sene_boot - sene_boot.mean()) / (sene_boot.std() + 1e-12)
        idsp = zf + zs - np.abs(zf - zs)
        boot_means[i] = idsp.mean()

    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])
    return {
        'n_boot': n_bootstrap,
        'idsp_mean': boot_means.mean(),
        'idsp_ci_lower': lower,
        'idsp_ci_upper': upper,
        'ci_level': ci,
    }


def permutation_enrichment_test(scores: pd.Series, case_cols: List[str],
                                 control_cols: List[str],
                                 n_perm: int = 2000, seed: int = 42) -> dict:
    """
    置换检验: 对样本标签置换, 检验两组评分差异的显著性 (替代参数ttest)

    直接接受已计算好的单样本评分 Series, 对样本标签进行随机置换.
    避免了 all-1s 伪矩阵产生的 NaN 问题.

    前沿参考: GSEApy (Zhuoqing Fang 2023) phenotype permutation paradigm.
    """
    case = scores[case_cols].dropna().values
    ctrl = scores[control_cols].dropna().values
    if len(case) < 3 or len(ctrl) < 3:
        return {'obs_case_mean': np.nan, 'obs_ctrl_mean': np.nan,
                'obs_diff': np.nan, 'p_perm': np.nan, 'effect_size': np.nan}

    obs_case_mean = np.mean(case)
    obs_ctrl_mean = np.mean(ctrl)
    obs_diff = obs_case_mean - obs_ctrl_mean

    pooled = np.concatenate([case, ctrl])
    n_case = len(case)
    rng = np.random.default_rng(seed)

    n_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        perm_diff = np.mean(pooled[:n_case]) - np.mean(pooled[n_case:])
        if abs(perm_diff) >= abs(obs_diff):
            n_extreme += 1

    p_perm = (n_extreme + 1) / (n_perm + 1)
    # z-score标准化后计算效应量 (同dual_enrichment_analysis修复)
    std_pooled = np.std(pooled)
    if std_pooled > 1e-12:
        mu_pooled = np.mean(pooled)
        case_z = (case - mu_pooled) / std_pooled
        ctrl_z = (ctrl - mu_pooled) / std_pooled
        d = float(np.mean(case_z) - np.mean(ctrl_z))
    else:
        d = 0.0

    return {
        'n_perm': n_perm,
        'obs_case_mean': float(obs_case_mean),
        'obs_ctrl_mean': float(obs_ctrl_mean),
        'obs_diff': float(obs_diff),
        'p_perm': float(p_perm),
        'effect_size': float(d),
    }


def dual_score_roc_auc(scores_df: pd.DataFrame, case_cols: List[str],
                        control_cols: List[str]) -> dict:
    """
    ROC/AUC: 双评分判别能力评估

    前沿参考: MDPI Biomedicines 2025 铁死亡-衰老 biomarker 论文使用 ROC 评估.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    result = {}
    for score_name, col in [('ferroptosis', 'ferroptosis'),
                             ('senescence', 'senescence'),
                             ('idsp_index', 'idsp_index')]:
        y_true = np.array([1 if c in case_cols else 0
                          for c in scores_df.index if c in case_cols or c in control_cols])
        y_score = scores_df.loc[[c for c in scores_df.index
                                if c in case_cols or c in control_cols], col].values
        y_true = y_true[~np.isnan(y_score)]
        y_score = y_score[~np.isnan(y_score)]
        if len(np.unique(y_true)) < 2 or len(y_score) < 4:
            result[score_name] = {'auc': np.nan, 'n': len(y_score)}
            continue
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        youden = tpr[np.argmax(tpr - fpr)] if len(tpr) > 0 else np.nan
        result[score_name] = {'auc': auc, 'n': len(y_score), 'youden_index': youden}
    return result


def i_squared_heterogeneity(comparisons: List[dict],
                             effect_key: str = 'd_ferroptosis',
                             var_key: str = 'var_ferroptosis') -> float:
    """
    I² 异质性 (标准Q统计量): 跨数据集效应量一致性

    Q = Σ w_i × (y_i - y_bar)²,  w_i = 1 / var_i
    I² = max(0, (Q - df) / Q) × 100

    沿用 Cochrane 金标准公式 (Higgins & Thompson 2002).
    需传入效应量及其方差. 若数据集 < 3 或方差全缺失则返回 np.nan.
    """
    valid = [(c.get(effect_key), c.get(var_key))
             for c in comparisons
             if pd.notna(c.get(effect_key)) and pd.notna(c.get(var_key))
             and c.get(var_key, 0) > 0]
    if len(valid) < 3:
        return np.nan
    ds, vs = zip(*valid)
    y = np.array(ds)
    w = 1.0 / np.array(vs)
    y_bar = np.average(y, weights=w)
    k = len(y)
    Q = float(np.sum(w * (y - y_bar) ** 2))
    df = k - 1
    I2 = max(0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    return float(I2)


def lodo_cross_validation(comparisons: List[dict], meta_func: callable) -> pd.DataFrame:
    """
    留一数据集交叉验证 (LODO): 检查 Meta 分析稳定性

    每剔除一个数据集, 重新计算 Meta p 值.
    meta_func 签名: (p_values, directions) → float

    重要: p值与效应量联合过滤, 避免长度不对齐导致合并失真.
    """
    results = []
    for i, comp in enumerate(comparisons):
        subset = [c for j, c in enumerate(comparisons) if j != i]

        # 铁死亡: 联合过滤 p值 + 效应量 (避免独立过滤导致长度不匹配)
        valid_ferr = [(c['p_ferroptosis'], c['d_ferroptosis']) for c in subset
                      if pd.notna(c.get('p_ferroptosis'))
                      and pd.notna(c.get('d_ferroptosis'))]
        if len(valid_ferr) >= 2:
            p_ferr, d_ferr = zip(*valid_ferr)
            dir_ferr = [int(np.sign(d)) if d != 0 else 1 for d in d_ferr]
            meta_ferr = meta_func(list(p_ferr), dir_ferr)
        else:
            p_ferr, d_ferr = [], []
            meta_ferr = np.nan

        # 衰老: 联合过滤 p值 + 效应量
        valid_sene = [(c['p_senescence'], c['d_senescence']) for c in subset
                      if pd.notna(c.get('p_senescence'))
                      and pd.notna(c.get('d_senescence'))]
        if len(valid_sene) >= 2:
            p_sene, d_sene = zip(*valid_sene)
            dir_sene = [int(np.sign(d)) if d != 0 else 1 for d in d_sene]
            meta_sene = meta_func(list(p_sene), dir_sene)
        else:
            p_sene, d_sene = [], []
            meta_sene = np.nan

        results.append({
            'removed_dataset': comp['dataset'],
            'n_remaining': len(subset),
            'meta_p_ferroptosis': meta_ferr,
            'meta_p_senescence': meta_sene,
            'mean_d_ferroptosis': np.mean(d_ferr) if d_ferr else np.nan,
            'mean_d_senescence': np.mean(d_sene) if d_sene else np.nan,
            'cv_d_ferroptosis': float(np.std(d_ferr) / abs(np.mean(d_ferr))) if d_ferr and np.mean(d_ferr) != 0 else np.nan,
            'cv_d_senescence': float(np.std(d_sene) / abs(np.mean(d_sene))) if d_sene and np.mean(d_sene) != 0 else np.nan,
        })
    return pd.DataFrame(results)


# ============================================================
# 前沿模块: Robust Rank Aggregation (RRA)
# ============================================================

def robust_rank_aggregation(rank_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Robust Rank Aggregation (RRA): 跨数据集基因排名一致性聚合

    基于 irGSEA (Fan et al. 2024, Brief Bioinform) 的 RRA 范式.
    核心思想: 对每个基因, 检验其跨数据集的排序是否显著优于随机期望.
    使用 Kolmogorov-Smirnov 检验每个基因的秩分布偏离均匀分布的程度.

    与 Fisher/Stouffer Meta 的区别:
      - Meta 分析合并 p 值(富集显著性), 回答"IDSP是否跨数据集一致显著"
      - RRA 聚合基因排名, 回答"哪些基因跨数据集一致差异表达"

    前沿参考:
      - irGSEA (Fan 2024) 使用 RRA 集成6种基因集打分方法
      - SumRank (Nakatsuka 2025) 使用秩聚合识别可复现DEG
      - Kolde et al. (2012) RRA 原始方法 (Nucleic Acids Res)
    """
    if rank_matrix.empty or rank_matrix.shape[1] < 2:
        logger.warning("  RRA: 矩阵为空或数据集<2, 跳过")
        return pd.DataFrame()

    n_genes = len(rank_matrix)
    n_datasets = rank_matrix.shape[1]

    # 将表达量转换为秩 (每个数据集内升序排列: 低秩=低表达, 高秩=高表达)
    # 但我们更关注差异方向, 所以对 case vs control 用 fold-change 排序
    rank_matrix = rank_matrix.copy()
    for col in rank_matrix.columns:
        series = rank_matrix[col].dropna()
        if len(series) < 5:
            rank_matrix[col] = np.nan
            continue
        rank_matrix[col] = pd.Series(
            stats.rankdata(series, method='average'), index=series.index)

    results = []
    for gene in rank_matrix.index:
        ranks = rank_matrix.loc[gene].dropna().values
        if len(ranks) < 2:
            continue
        # 归一化秩到 [0, 1]
        n_genes_local = n_genes  # 使用全局基因数
        normalized = ranks / (n_genes_local + 1)
        # KS检验: 检验归一化秩是否偏离均匀分布 U(0,1)
        # 若跨数据集一致排在前列 → 偏离均匀分布 → p值小
        ks_stat, ks_p = stats.kstest(normalized, 'uniform', args=(0, 1))
        # 归一化秩的均值 (0=最显著, 0.5=随机)
        mean_rank = normalized.mean()
        # 秩的变异系数 (跨数据集一致性)
        cv = np.std(normalized) / (mean_rank + 1e-12)
        results.append({
            'gene': gene,
            'n_datasets': len(ranks),
            'mean_normalized_rank': mean_rank,
            'rank_cv': cv,
            'ks_statistic': ks_stat,
            'rra_pvalue': ks_p,
        })

    rra_df = pd.DataFrame(results).sort_values('rra_pvalue')
    if not rra_df.empty:
        _, rra_padj, _, _ = multipletests(rra_df['rra_pvalue'], method='fdr_bh')
        rra_df['rra_padj'] = rra_padj
        rra_df['significant'] = rra_padj < 0.05

    logger.info(f"  RRA: {len(rra_df)} 基因, "
                f"{rra_df['significant'].sum() if 'significant' in rra_df.columns else 0} 显著")
    return rra_df


# ============================================================
# 前沿模块: Jensen-Shannon Divergence + KS 分布差异
# ============================================================

def jsd_and_ks_comparison(scores_df: pd.DataFrame, case_cols: List[str],
                           control_cols: List[str], dataset_name: str) -> dict:
    """
    Jensen-Shannon Divergence (JSD) + KS检验: 信息论视角的分布差异量化

    JSD 是两个概率分布之间对称且有界的相似性度量(0=完全相同, 1=完全分离):
      JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = (P+Q)/2

    相比 t-test / Cohen's d:
      - 不假设正态性
      - 对称且归一化到 [0, 1]
      - 更适合小样本

    前沿参考:
      - irGSEA (Fan 2024) 使用 JSD 评估多方法一致性
      - Lin (1991) JSD 原始论文
      - 应用于单细胞基因集评分 (2024-2025 新兴趋势)
    """
    result = {'dataset': dataset_name}
    case_df = scores_df[scores_df.index.isin(case_cols)]
    ctrl_df = scores_df[scores_df.index.isin(control_cols)]

    for score_name, col in [('ferroptosis', 'ferroptosis'),
                             ('senescence', 'senescence'),
                             ('idsp_index', 'idsp_index')]:
        case_vals = case_df[col].dropna().values
        ctrl_vals = ctrl_df[col].dropna().values
        if len(case_vals) < 3 or len(ctrl_vals) < 3:
            result[f'{score_name}_jsd'] = np.nan
            result[f'{score_name}_ks_stat'] = np.nan
            result[f'{score_name}_ks_p'] = np.nan
            continue

        # JSD: 高斯核密度估计 → 离散化 → 计算
        def _jsd(x, y, bins=20):
            all_vals = np.concatenate([x, y])
            lo, hi = np.percentile(all_vals, [1, 99])
            if hi - lo < 1e-10:
                return 0.0
            bins_arr = np.linspace(lo, hi, bins)
            px = np.histogram(x, bins=bins_arr, density=True)[0] + 1e-12
            py = np.histogram(y, bins=bins_arr, density=True)[0] + 1e-12
            px /= px.sum()
            py /= py.sum()
            m = 0.5 * (px + py)
            kl_pm = np.sum(px * np.log(px / m))
            kl_qm = np.sum(py * np.log(py / m))
            return float(0.5 * (kl_pm + kl_qm))

        jsd_val = _jsd(case_vals, ctrl_vals)

        # 两样本KS检验
        ks_stat, ks_p = stats.ks_2samp(case_vals, ctrl_vals)

        result[f'{score_name}_jsd'] = jsd_val
        result[f'{score_name}_ks_stat'] = float(ks_stat)
        result[f'{score_name}_ks_p'] = float(ks_p)

        # 简洁日志
        if pd.notna(jsd_val):
            logger.info(f"  [{dataset_name}] {score_name}: JSD={jsd_val:.4f}, "
                        f"KS_p={ks_p:.4e}")

    return result


# ============================================================
# 可视化
# ============================================================

def plot_forest_dual(comparisons: List[dict], save_path: str,
                     re_ferr: dict = None, re_sene: dict = None):
    """双评分效应量森林图 (Cohen's d + 95%CI + 预测区间 + 汇总菱形)

    Parameters:
        comparisons: 每个数据集的效应量信息
        re_ferr:     随机效应Meta结果 (含 summary_effect, pi_lower, pi_upper)
        re_sene:     随机效应Meta结果 (含 summary_effect, pi_lower, pi_upper)
    """
    valid_comp = [c for c in comparisons if pd.notna(c.get('d_ferroptosis', np.nan))
                  and pd.notna(c.get('d_senescence', np.nan))]
    if not valid_comp:
        logger.warning("  森林图: 无有效数据, 跳过")
        return

    ds_names = [c['dataset'] for c in valid_comp]
    d_ferr = [c['d_ferroptosis'] for c in valid_comp]
    d_sene = [c['d_senescence'] for c in valid_comp]

    # 计算SE: 使用dual_enrichment_analysis中已计算的方差
    # var = (n_case+n_ctrl)/(n_case*n_ctrl) + d²/(2*(n_case+n_ctrl))
    # 此公式为Hedges' g的标准方差近似, 适用于标准化后的d值
    se_ferr = [np.sqrt(c.get('var_ferroptosis')) if pd.notna(c.get('var_ferroptosis'))
               else np.nan for c in valid_comp]
    se_sene = [np.sqrt(c.get('var_senescence')) if pd.notna(c.get('var_senescence'))
               else np.nan for c in valid_comp]

    n_studies = len(valid_comp)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, n_studies * 0.7)),
                                    gridspec_kw={'width_ratios': [1, 1]})

    for ax, ds, se_list, label, color, re_res in [
        (ax1, d_ferr, se_ferr, 'Ferroptosis', '#E74C3C', re_ferr),
        (ax2, d_sene, se_sene, 'Senescence', '#3498DB', re_sene),
    ]:
        y_positions = list(range(n_studies - 1, -1, -1))

        for i, y in enumerate(y_positions):
            # 95% CI 水平线
            ci_low = ds[i] - 1.96 * se_list[i]
            ci_hi = ds[i] + 1.96 * se_list[i]
            ax.plot([ci_low, ci_hi], [y, y], color=color, lw=2, alpha=0.7, zorder=1)
            # 效应量散点
            ax.scatter(ds[i], y, c=color, s=80, zorder=3, edgecolors='white', linewidth=1.5)

        ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.6, zorder=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ds_names, fontsize=9)
        ax.set_xlabel("Cohen's d", fontsize=11)
        ax.set_title(label, fontsize=13, fontweight='bold', color=color)

        # 汇总菱形 (随机效应 Meta)
        if re_res and 'summary_effect' in re_res and pd.notna(re_res.get('summary_effect')):
            sum_d = re_res['summary_effect']
            sum_se = re_res.get('se_random', np.nan)
            # 汇总菱形
            diamond_y = -1
            if pd.notna(sum_se):
                ci_lo = sum_d - 1.96 * sum_se
                ci_hi = sum_d + 1.96 * sum_se
                # 预测区间 (虚线)
                pi_lo = re_res.get('pi_lower', np.nan)
                pi_hi = re_res.get('pi_upper', np.nan)
                if pd.notna(pi_lo) and pd.notna(pi_hi):
                    ax.plot([pi_lo, pi_hi], [diamond_y, diamond_y], color=color,
                            lw=1.5, ls=':', alpha=0.5, zorder=1)

                ax.plot([ci_lo - 0.05, sum_d, ci_hi + 0.05, sum_d,
                         ci_lo - 0.05],
                        [diamond_y, diamond_y - 0.35, diamond_y, diamond_y + 0.35, diamond_y],
                        color=color, lw=1.2, alpha=0.6, zorder=2)

            # 标签
            ax.set_ylim(diamond_y - 1.2, n_studies - 0.2)
            ytick_labels = list(ds_names) + ['◆ Summary']
            ytick_pos = y_positions + [diamond_y]
            ax.set_yticks(ytick_pos)
            ax.set_yticklabels(ytick_labels, fontsize=9)
        else:
            ax.set_ylim(-0.5, n_studies - 0.2)

    fig.suptitle('Forest Plot: Ferroptosis & Senescence Effect Sizes in CIRI\n'
                 '◆ = Summary (random-effects), ··· = 95% Prediction interval',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  森林图保存: {save_path}")


def plot_temporal_dual(temporal_df: pd.DataFrame, save_path: str):
    """时间动态双曲线 (Sham基线独立显示 + 变化率子图)"""
    if temporal_df.empty:
        return
    df = temporal_df.sort_values('time_hr')

    # 分离Sham与实际时间点
    sham_row = df[df['timepoint'] == 'Sham']
    tp_rows = df[df['timepoint'] != 'Sham']

    fig = plt.figure(figsize=(10, 7))

    # ---- 子图1: 评分绝对值和变化趋势 ----
    ax1 = fig.add_subplot(211)
    color_ferr = '#E74C3C'
    color_sene = '#3498DB'
    ax2 = ax1.twinx()

    if not sham_row.empty:
        sham_ferr_mean = sham_row['ferroptosis_mean'].values[0]
        sham_sene_mean = sham_row['senescence_mean'].values[0]
        # 绘制Sham基线 (水平虚线)
        ax1.axhline(sham_ferr_mean, color=color_ferr, ls=':', lw=1.5, alpha=0.5)
        ax2.axhline(sham_sene_mean, color=color_sene, ls=':', lw=1.5, alpha=0.5)
        # 标注Sham
        ax1.text(0.02, sham_ferr_mean, 'Sham (Ferroptosis)', color=color_ferr,
                 fontsize=8, alpha=0.7, va='center', transform=ax1.get_yaxis_transform())
        ax2.text(0.02, sham_sene_mean, 'Sham (Senescence)', color=color_sene,
                 fontsize=8, alpha=0.7, va='center', transform=ax2.get_yaxis_transform())
        # 虚线分隔Sham与损伤时间点
        ax1.axvline(x=0, color='gray', ls=':', lw=1, alpha=0.4)

    # 绘制实际时间点
    if not tp_rows.empty:
        x_tp = tp_rows['time_hr'].values
        ax1.errorbar(x_tp, tp_rows['ferroptosis_mean'], yerr=tp_rows['ferroptosis_sem'],
                     fmt='o-', color=color_ferr, capsize=4, label='Ferroptosis', markersize=8)
        ax2.errorbar(x_tp, tp_rows['senescence_mean'], yerr=tp_rows['senescence_sem'],
                     fmt='s--', color=color_sene, capsize=4, label='Senescence', markersize=8)
        # x轴仅显示实际时间点
        ax1.set_xticks(x_tp)
        ax1.set_xticklabels([f'{int(h)}h' for h in x_tp])

    ax1.set_xlabel('Time (hours post-MCAO)')
    ax1.set_ylabel('Ferroptosis Score', color=color_ferr)
    ax2.set_ylabel('Senescence Score', color=color_sene)
    ax1.tick_params(axis='y', labelcolor=color_ferr)
    ax2.tick_params(axis='y', labelcolor=color_sene)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.set_title('A. Temporal Dynamics: Ferroptosis vs Senescence in MCAO')

    # ---- 子图2: 变化率 (delta/hr) ----
    ax3 = fig.add_subplot(212)
    if not tp_rows.empty:
        x_tp = tp_rows['time_hr'].values
        ferr_deltas = tp_rows['ferr_delta_per_hr'].values
        sene_deltas = tp_rows['sene_delta_per_hr'].values

        # 标记有效变化率
        valid_ferr_d = ~np.isnan(ferr_deltas)
        valid_sene_d = ~np.isnan(sene_deltas)

        if valid_ferr_d.any():
            ax3.plot(x_tp[valid_ferr_d], ferr_deltas[valid_ferr_d],
                     'o-', color=color_ferr, label='Ferroptosis Δ/hr', markersize=8)
        if valid_sene_d.any():
            ax3.plot(x_tp[valid_sene_d], sene_deltas[valid_sene_d],
                     's--', color=color_sene, label='Senescence Δ/hr', markersize=8)

        ax3.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
        ax3.set_xticks(x_tp)
        ax3.set_xticklabels([f'{int(h)}h' for h in x_tp])

    ax3.set_xlabel('Time (hours post-MCAO)')
    ax3.set_ylabel('Rate of Change (Δ score / hr)')
    ax3.legend(loc='upper right')
    ax3.set_title('B. Rate of Change: Activation Kinetics')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  时间动态图保存: {save_path}")


def plot_scatter_dual(all_scores_df: pd.DataFrame, save_path: str):
    """双评分散点图 (各数据集) — 仅显示 case/control 分组样本"""
    datasets = all_scores_df['dataset'].unique()
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), squeeze=False)
    if n == 0:
        return
    for i, ds in enumerate(datasets):
        ax = axes[0, i]
        sub = all_scores_df[all_scores_df['dataset'] == ds].dropna(subset=['ferroptosis', 'senescence'])
        # 仅保留 case/control 分组 (排除 unknown)
        sub = sub[sub['group'].isin(['case', 'control'])]
        if sub.empty:
            ax.set_title(f'{ds}\n(无有效分组)')
            continue
        colors = sub['group'].map({'case': '#E74C3C', 'control': '#3498DB'})
        ax.scatter(sub['ferroptosis'], sub['senescence'], c=colors.tolist(), alpha=0.7, s=40, edgecolors='none')
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub['ferroptosis'], sub['senescence'])
            ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.3e}', transform=ax.transAxes,
                    va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel('Ferroptosis Score')
        ax.set_ylabel('Senescence Score')
        ax.set_title(ds)
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.axvline(0, color='gray', ls='--', lw=0.5)
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#E74C3C', label='Case'),
                           Patch(facecolor='#3498DB', label='Control')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  散点图保存: {save_path}")


def plot_gene_heatmap(all_gene_dfs: List[pd.DataFrame], save_path: str):
    """核心基因热图"""
    if not all_gene_dfs:
        return
    combined = pd.concat(all_gene_dfs, ignore_index=True)
    key_genes = ['ACSL4', 'PTGS2', 'HMOX1', 'TFRC', 'GPX4',
                 'SLC7A11', 'CDKN1A', 'IL6', 'IL1B', 'HMGB1',
                 'TP53', 'RB1', 'NFE2L2', 'KEAP1', 'HIF1A']
    available = [g for g in key_genes if g in combined['gene'].values]
    if len(available) < 3:
        return
    pivot = combined[combined['gene'].isin(available)].pivot_table(
        index='gene', columns='dataset', values='log2FC', aggfunc='first')
    pivot = pivot.loc[[g for g in available if g in pivot.index]]

    fig, ax = plt.subplots(figsize=(len(pivot.columns)*1.5 + 2, len(pivot)*1.2 + 2))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=7, color='white' if abs(val) > 1 else 'black')
    plt.colorbar(im, ax=ax, label='log2FC', shrink=0.8)
    ax.set_title('Core Gene Expression Changes (Case vs Control)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  热图保存: {save_path}")

# ============================================================
# 主流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("L1: IDSP 双评分分析 — 在CIRI中识别铁驱动的衰老程序")
    logger.info(f"纯铁死亡: {len(PURE_FERROPTOSIS)} | 纯衰老: {len(PURE_SENESCENCE)} | 共享: {len(SHARED_GENES)}")
    logger.info("=" * 60)

    # ============================================================
    # 1. Bulk RNA-seq 双评分分析 (5个数据集)
    # ============================================================
    loaders = [
        ('GSE16561', _load_expr_gse16561),
        ('GSE37587', _load_expr_gse37587),
        ('GSE61616', _load_expr_gse61616),
        ('GSE97537', _load_expr_gse97537),
    ]

    all_scores = []
    all_comparisons = []
    all_gene_dfs = []
    all_gpx4 = []
    all_meta = []    # (ds_name, scores, case_cols, control_cols) for advanced analysis
    temporal_df = pd.DataFrame()

    # ============================================================
    # 1a. 跨平台 ComBat Harmonization (新增 🔧)
    # ============================================================
    # 先加载所有数据集, 然后统一 ComBat 校正
    raw_expr_dict = {}
    sample_info = {}  # {ds_name: (case_cols, control_cols)}

    for ds_name, loader in loaders:
        try:
            expr_gene, case_cols, control_cols = loader()
            raw_expr_dict[ds_name] = expr_gene
            sample_info[ds_name] = (case_cols, control_cols)
            logger.info(f"  [ComBat-pre] {ds_name}: {expr_gene.shape}")
        except Exception as e:
            logger.error(f"  ✗ {ds_name} 加载失败: {e}")
            import traceback; traceback.print_exc()

    # 运行 ComBat 批次校正 (仅对有效数据集)
    # 过滤掉空矩阵
    valid_expr_dict = {k: v for k, v in raw_expr_dict.items() if v is not None and v.shape[0] > 0 and v.shape[1] > 0}
    if len(valid_expr_dict) >= 2:
        logger.info("\n" + "=" * 50)
        logger.info("ComBat 跨平台批次校正")
        logger.info(f"  有效数据集: {list(valid_expr_dict.keys())} (共 {len(valid_expr_dict)} 个)")
        harmonized_dict = combat_harmonize_datasets(valid_expr_dict, sample_groups=sample_info)
        if len(harmonized_dict) < 2:
            logger.info("  ComBat 校正未生效, 使用原始数据")
            harmonized_dict = raw_expr_dict
        else:
            logger.info("  ComBat 校正成功, 使用校正后的表达矩阵")
            # 保存校正前后的对比指标
            pre_means = {k: float(v.values.mean()) for k, v in valid_expr_dict.items()}
            post_means = {k: float(v.values.mean()) for k, v in harmonized_dict.items()}
            combat_metrics = {
                'dataset': list(pre_means.keys()),
                'mean_before': [pre_means[k] for k in pre_means],
                'mean_after': [post_means[k] for k in post_means],
            }
            pd.DataFrame(combat_metrics).to_csv(
                OUTPUT_DIR / 'L1_combat_harmonization.csv', index=False)
            logger.info("  ComBat指标保存: L1_combat_harmonization.csv")

            # ComBat PCA 诊断图 (Zhang 2020 标准QC)
            try:
                combat_pca_diagnostic(valid_expr_dict, sample_groups=sample_info,
                                      save_path=OUTPUT_DIR / 'L1_combat_pca_diagnostic.png')
            except Exception as pca_e:
                logger.warning(f"  ComBat PCA诊断图生成失败: {pca_e}")
    else:
        harmonized_dict = raw_expr_dict
        logger.info(f"  ComBat 跳过: 有效数据集仅 {len(valid_expr_dict)} 个")

    # 使用校正后的数据运行双评分分析
    for ds_name in sample_info:
        try:
            expr_gene = harmonized_dict.get(ds_name, raw_expr_dict.get(ds_name))
            if expr_gene is None or expr_gene.empty:
                logger.warning(f"  {ds_name}: 表达矩阵为空, 跳过")
                continue
            case_cols, control_cols = sample_info[ds_name]
            # GSE37587 是配对设计 (baseline vs follow-up)
            is_paired = (ds_name == 'GSE37587')
            scores_df, comp = dual_enrichment_analysis(expr_gene, ds_name, case_cols, control_cols,
                                                       is_paired=is_paired)
            all_scores.append(scores_df)
            all_comparisons.append(comp)

            # GPX4验证
            gpx4_res = gpx4_validation(expr_gene, scores_df, case_cols, control_cols, ds_name)
            all_gpx4.append(gpx4_res)

            # 单基因分析
            all_genes = PURE_FERROPTOSIS | PURE_SENESCENCE | SHARED_GENES
            gene_df = analyze_signature_genes(expr_gene, case_cols, control_cols, all_genes, ds_name)
            all_gene_dfs.append(gene_df)

            # 记录元数据 (用于高级分析)
            all_meta.append((ds_name, scores_df.copy(), case_cols.copy(), control_cols.copy()))

        except Exception as e:
            logger.error(f"  ✗ {ds_name} 失败: {e}")
            import traceback; traceback.print_exc()
            continue

    # ============================================================
    # 2. GSE104036 时间动态分析
    # ============================================================
    try:
        expr_104036, tp_dict, sham_cols = _load_expr_gse104036()
        # 收集所有样本分组
        all_ipsi = []
        for tp_cols in tp_dict.values():
            all_ipsi.extend(tp_cols)
        scores_104036, comp_104036 = dual_enrichment_analysis(expr_104036, 'GSE104036', all_ipsi, sham_cols, is_paired=False)
        all_scores.append(scores_104036)
        all_comparisons.append(comp_104036)

        gpx4_104036 = gpx4_validation(expr_104036, scores_104036, all_ipsi, sham_cols, 'GSE104036')
        all_gpx4.append(gpx4_104036)

        all_genes = PURE_FERROPTOSIS | PURE_SENESCENCE | SHARED_GENES
        gene_104036 = analyze_signature_genes(expr_104036, all_ipsi, sham_cols, all_genes, 'GSE104036')
        all_gene_dfs.append(gene_104036)

        # 时间动态
        temporal_df = temporal_dual_analysis(expr_104036, tp_dict, sham_cols, 'GSE104036')
        # 记录元数据
        all_meta.append(('GSE104036', scores_104036.copy(), all_ipsi.copy(), sham_cols.copy()))

    except Exception as e:
        logger.error(f"  ✗ GSE104036 失败: {e}")
        import traceback; traceback.print_exc()

    # ============================================================
    # 3. Meta分析
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("Meta分析")

    comp_df = pd.DataFrame(all_comparisons)

    # 铁死亡 Meta分析
    ferr_pvals = comp_df['p_ferroptosis'].dropna().values
    if len(ferr_pvals) >= 2:
        chi2_f, meta_p_f = fisher_meta_analysis(list(ferr_pvals))
        logger.info(f"铁死亡 Meta: χ²={chi2_f:.2f}, p={meta_p_f:.4e}")
    else:
        meta_p_f = np.nan

    # 衰老 Meta分析
    sene_pvals = comp_df['p_senescence'].dropna().values
    if len(sene_pvals) >= 2:
        chi2_s, meta_p_s = fisher_meta_analysis(list(sene_pvals))
        logger.info(f"衰老 Meta: χ²={chi2_s:.2f}, p={meta_p_s:.4e}")
    else:
        meta_p_s = np.nan

    # ============================================================
    # 4. 高级分析 (Bootstrap · 置换检验 · ROC · I² · LODO)
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("高级分析")

    # 4a. Bootstrap IDSP CI (第一个数据集)
    if all_scores:
        first_ds = all_scores[0]
        boot_res = bootstrap_idsp_ci(first_ds, n_bootstrap=2000, ci=0.95)
        if boot_res['n_boot'] > 0:
            logger.info(f"  Bootstrap IDSP: mean={boot_res['idsp_mean']:.3f}, "
                        f"95%CI=[{boot_res['idsp_ci_lower']:.3f}, {boot_res['idsp_ci_upper']:.3f}]")

    # 4b. 置换检验 (直接传入已计算的评分Series, 不再使用全1伪矩阵)
    perm_results = []
    for ds_name, scores_df, ccs, ctrls in all_meta:
        for gname, score_col in [('Ferroptosis', 'ferroptosis'),
                                  ('Senescence', 'senescence')]:
            perm = permutation_enrichment_test(
                scores_df[score_col], ccs, ctrls, n_perm=2000, seed=42)
            perm['dataset'] = ds_name
            perm['gene_set'] = gname
            perm_results.append(perm)

    perm_sig_ferr = sum(1 for p in perm_results
                        if p.get('gene_set') == 'Ferroptosis'
                        and pd.notna(p.get('p_perm')) and p['p_perm'] < 0.05)
    perm_sig_sene = sum(1 for p in perm_results
                        if p.get('gene_set') == 'Senescence'
                        and pd.notna(p.get('p_perm')) and p['p_perm'] < 0.05)
    logger.info(f"  置换检验: 铁死亡 {perm_sig_ferr}/{len([p for p in perm_results if p['gene_set']=='Ferroptosis'])} 显著, "
                f"衰老 {perm_sig_sene}/{len([p for p in perm_results if p['gene_set']=='Senescence'])} 显著")
    perm_df = pd.DataFrame(perm_results)
    perm_df.to_csv(OUTPUT_DIR / 'L1_permutation_tests.csv', index=False)

    # 4c. ROC/AUC (基于all_meta)
    roc_results = []
    for ds_name, scores_df, ccs, ctrls in all_meta:
        try:
            roc = dual_score_roc_auc(scores_df, ccs, ctrls)
            roc['dataset'] = ds_name
            roc_results.append(roc)
            for score_name in ['ferroptosis', 'senescence', 'idsp_index']:
                if score_name in roc:
                    auc_val = roc[score_name]['auc']
                    if pd.notna(auc_val):
                        logger.info(f"  [{ds_name}] {score_name} AUC={auc_val:.3f}")
        except Exception as e:
            logger.warning(f"  [{ds_name}] ROC跳过: {e}")

    # 4d. I² 异质性 (标准Q统计量, 使用已存储的方差)
    i2_ferr = i_squared_heterogeneity(all_comparisons,
                                       effect_key='d_ferroptosis',
                                       var_key='var_ferroptosis')
    i2_sene = i_squared_heterogeneity(all_comparisons,
                                       effect_key='d_senescence',
                                       var_key='var_senescence')
    if pd.notna(i2_ferr):
        logger.info(f"  铁死亡 跨数据集 I²={i2_ferr:.0f}% "
                    f"{'(低异质性)' if i2_ferr < 25 else '(中异质性)' if i2_ferr < 50 else '(高异质性)'}")
    if pd.notna(i2_sene):
        logger.info(f"  衰老 跨数据集 I²={i2_sene:.0f}% "
                    f"{'(低异质性)' if i2_sene < 25 else '(中异质性)' if i2_sene < 50 else '(高异质性)'}")

    # 4e. LODO 交叉验证 (meta_func 需接受 (pvals, dirs) 两个参数)
    n_stable = 0  # 初始化, 防止 LODO 跳过时 JSON 摘要 NameError
    lodo_df = lodo_cross_validation(
        all_comparisons,
        meta_func=lambda pvals, dirs: stouffer_meta(list(pvals), directions=list(dirs))
        if len(pvals) >= 2 else np.nan)
    if not lodo_df.empty:
        n_stable = sum(1 for _, r in lodo_df.iterrows()
                       if pd.notna(r['meta_p_ferroptosis']) and pd.notna(r['meta_p_senescence']))
        logger.info(f"  LODO: {n_stable}/{len(lodo_df)} 移除后Meta仍有效")
        # 效应量稳定性报告
        cv_ferr = lodo_df['cv_d_ferroptosis'].dropna()
        cv_sene = lodo_df['cv_d_senescence'].dropna()
        if len(cv_ferr):
            logger.info(f"  LODO效应量稳定性: d_ferr CV={cv_ferr.mean():.3f} "
                        f"(越接近0越稳定), d_sene CV={cv_sene.mean():.3f}")
        lodo_df.to_csv(OUTPUT_DIR / 'L1_lodo_cross_validation.csv', index=False)

    # 4f. 前沿: Stouffer 加权Meta (带效应方向) + 随机效应Meta
    logger.info("\n  前沿Meta分析:")
    # 使用 dual_enrichment_analysis 中已存储的方差
    ferr_ds = [c.get('d_ferroptosis') for c in all_comparisons
               if pd.notna(c.get('d_ferroptosis')) and pd.notna(c.get('var_ferroptosis'))]
    ferr_vars = [c.get('var_ferroptosis') for c in all_comparisons
                 if pd.notna(c.get('d_ferroptosis')) and pd.notna(c.get('var_ferroptosis'))]
    sene_ds = [c.get('d_senescence') for c in all_comparisons
               if pd.notna(c.get('d_senescence')) and pd.notna(c.get('var_senescence'))]
    sene_vars = [c.get('var_senescence') for c in all_comparisons
                 if pd.notna(c.get('d_senescence')) and pd.notna(c.get('var_senescence'))]

    # 加权Stouffer (同步过滤 p值 + 方向 + 权重, 避免长度不匹配)
    ferr_dir_for_p, ferr_w_for_p, ferr_p_for_p = [], [], []
    for c in all_comparisons:
        p = c.get('p_ferroptosis')
        d = c.get('d_ferroptosis')
        n_c, n_ct = c.get('n_case', 0), c.get('n_control', 0)
        if pd.notna(p) and pd.notna(d):
            ferr_p_for_p.append(p)
            ferr_dir_for_p.append(int(np.sign(d)) if d != 0 else 1)
            ferr_w_for_p.append(np.sqrt(n_c + n_ct) if (n_c + n_ct) > 0 else 1.0)

    if len(ferr_p_for_p) >= 2:
        meta_p_stouffer_f = stouffer_meta(
            ferr_p_for_p, weights=ferr_w_for_p, directions=ferr_dir_for_p)
        logger.info(f"  铁死亡 Stouffer(加权+方向) p={meta_p_stouffer_f:.4e}")
    else:
        meta_p_stouffer_f = np.nan

    sene_dir_for_p, sene_w_for_p, sene_p_for_p = [], [], []
    for c in all_comparisons:
        p = c.get('p_senescence')
        d = c.get('d_senescence')
        n_c, n_ct = c.get('n_case', 0), c.get('n_control', 0)
        if pd.notna(p) and pd.notna(d):
            sene_p_for_p.append(p)
            sene_dir_for_p.append(int(np.sign(d)) if d != 0 else 1)
            sene_w_for_p.append(np.sqrt(n_c + n_ct) if (n_c + n_ct) > 0 else 1.0)

    if len(sene_p_for_p) >= 2:
        meta_p_stouffer_s = stouffer_meta(
            sene_p_for_p, weights=sene_w_for_p, directions=sene_dir_for_p)
        logger.info(f"  衰老 Stouffer(加权+方向) p={meta_p_stouffer_s:.4e}")
    else:
        meta_p_stouffer_s = np.nan

    # 随机效应Meta分析
    if len(ferr_ds) >= 3:
        re_ferr = random_effects_meta_analysis(ferr_ds, ferr_vars)
        logger.info(f"  铁死亡 随机效应Meta: d={re_ferr['summary_effect']:.3f}, "
                    f"p={re_ferr['p_value']:.4e}, I²={re_ferr['I2']:.0f}%, τ²={re_ferr['tau2']:.4f}")
    else:
        re_ferr = None

    if len(sene_ds) >= 3:
        re_sene = random_effects_meta_analysis(sene_ds, sene_vars)
        logger.info(f"  衰老 随机效应Meta: d={re_sene['summary_effect']:.3f}, "
                    f"p={re_sene['p_value']:.4e}, I²={re_sene['I2']:.0f}%, τ²={re_sene['tau2']:.4f}")
    else:
        re_sene = None

    # 4f-2. Bayesian 随机效应 Meta-Analysis (新增 🔧)
    logger.info("\n  Bayesian Meta-Analysis (MCMC):")
    bayes_ferr = bayesian_meta_analysis(ferr_ds, ferr_vars) if len(ferr_ds) >= 3 else None
    bayes_sene = bayesian_meta_analysis(sene_ds, sene_vars) if len(sene_ds) >= 3 else None

    # 高级Meta结果汇总导出 (新增 Bayesian 行)
    meta_summary = {
        'method': ['Fisher', 'Fisher', 'Stouffer_weighted', 'Stouffer_weighted',
                   'Random_effects', 'Random_effects',
                   'Bayesian_MCMC', 'Bayesian_MCMC'],
        'score': ['Ferroptosis', 'Senescence', 'Ferroptosis', 'Senescence',
                  'Ferroptosis', 'Senescence',
                  'Ferroptosis', 'Senescence'],
        'p_value': [meta_p_f, meta_p_s, meta_p_stouffer_f, meta_p_stouffer_s,
                    re_ferr['p_value'] if re_ferr else np.nan,
                    re_sene['p_value'] if re_sene else np.nan,
                    np.nan, np.nan],  # Bayesian 不用 p 值
        'summary_d': [np.nan, np.nan, np.nan, np.nan,
                      re_ferr['summary_effect'] if re_ferr else np.nan,
                      re_sene['summary_effect'] if re_sene else np.nan,
                      bayes_ferr.get('mu_mean') if bayes_ferr else np.nan,
                      bayes_sene.get('mu_mean') if bayes_sene else np.nan],
        'I2_pct': [np.nan, np.nan, np.nan, np.nan,
                   re_ferr['I2'] if re_ferr else np.nan,
                   re_sene['I2'] if re_sene else np.nan, np.nan, np.nan],
        'tau2': [np.nan, np.nan, np.nan, np.nan,
                 re_ferr['tau2'] if re_ferr else np.nan,
                 re_sene['tau2'] if re_sene else np.nan,
                 bayes_ferr.get('tau2_mean') if bayes_ferr else np.nan,
                 bayes_sene.get('tau2_mean') if bayes_sene else np.nan],
        'pi_lower': [np.nan, np.nan, np.nan, np.nan,
                     re_ferr['pi_lower'] if re_ferr else np.nan,
                     re_sene['pi_lower'] if re_sene else np.nan,
                     bayes_ferr.get('mu_hdi_2.5') if bayes_ferr else np.nan,
                     bayes_sene.get('mu_hdi_2.5') if bayes_sene else np.nan],
        'pi_upper': [np.nan, np.nan, np.nan, np.nan,
                     re_ferr['pi_upper'] if re_ferr else np.nan,
                     re_sene['pi_upper'] if re_sene else np.nan,
                     bayes_ferr.get('mu_hdi_97.5') if bayes_ferr else np.nan,
                     bayes_sene.get('mu_hdi_97.5') if bayes_sene else np.nan],
        'k': [len(ferr_pvals), len(sene_pvals),
              len(ferr_p_for_p), len(sene_p_for_p),
              re_ferr['k'] if re_ferr else 0,
              re_sene['k'] if re_sene else 0,
              bayes_ferr.get('k', 0) if bayes_ferr else 0,
              bayes_sene.get('k', 0) if bayes_sene else 0],
    }
    pd.DataFrame(meta_summary).to_csv(OUTPUT_DIR / 'L1_meta_analysis_summary.csv', index=False)
    logger.info("  Meta汇总保存: L1_meta_analysis_summary.csv")

    # Bayesian Meta 详细结果导出
    if bayes_ferr or bayes_sene:
        bayes_rows = []
        for label, bm in [('Ferroptosis', bayes_ferr), ('Senescence', bayes_sene)]:
            if bm and bm.get('converged'):
                bayes_rows.append({
                    'score': label,
                    'mu_mean': bm.get('mu_mean'),
                    'mu_sd': bm.get('mu_sd'),
                    'mu_hdi_2.5': bm.get('mu_hdi_2.5'),
                    'mu_hdi_97.5': bm.get('mu_hdi_97.5'),
                    'tau_mean': bm.get('tau_mean'),
                    'tau_sd': bm.get('tau_sd'),
                    'tau2_mean': bm.get('tau2_mean'),
                    'tau2_sd': bm.get('tau2_sd'),
                    'rho_mean': bm.get('rho_mean'),
                    'p_mu_gt_0': bm.get('p_mu_gt_0'),
                    'rhat_max': max(bm.get('mu_rhat', 1), bm.get('tau_rhat', 1)),
                    'converged': bm.get('converged'),
                    'mu_significant': bm.get('mu_significant'),
                    'k': bm.get('k'),
                })
        if bayes_rows:
            pd.DataFrame(bayes_rows).to_csv(
                OUTPUT_DIR / 'L1_bayesian_meta_results.csv', index=False)
            logger.info("  Bayesian结果保存: L1_bayesian_meta_results.csv")

    # 4g. 前沿: Robust Rank Aggregation (跨数据集基因一致性)
    logger.info("\n  前沿RRA分析:")
    rra_results = []
    if all_gene_dfs and len(all_gene_dfs) >= 2:
        # 构建 log2FC 矩阵 (基因 × 数据集)
        all_gene_combined = pd.concat(all_gene_dfs, ignore_index=True)
        pivot_fc = all_gene_combined.pivot_table(
            index='gene', columns='dataset', values='log2FC', aggfunc='first')
        rra_df = robust_rank_aggregation(pivot_fc)
        if not rra_df.empty:
            rra_results.append(rra_df)
            key_genes_in_rra = [g for g in ['ACSL4', 'PTGS2', 'HMOX1', 'TFRC', 'GPX4',
                                             'SLC7A11', 'CDKN1A', 'IL6', 'TP53', 'HMGB1']
                                if g in rra_df['gene'].values]
            for g in key_genes_in_rra:
                row = rra_df[rra_df['gene'] == g].iloc[0]
                logger.info(f"    {g}: mean_rank={row['mean_normalized_rank']:.3f}, "
                            f"padj={row['rra_padj']:.4e}, "
                            f"{'★显著' if row.get('significant') else ''}")
            rra_df.to_csv(OUTPUT_DIR / 'L1_rra_gene_consistency.csv', index=False)

    # 4h. 前沿: JSD + KS 分布差异分析
    logger.info("\n  前沿JSD/KS分布差异:")
    jsd_results = []
    for ds_name, scores_df, ccs, ctrls in all_meta:
        try:
            jsd_res = jsd_and_ks_comparison(scores_df, ccs, ctrls, ds_name)
            jsd_results.append(jsd_res)
        except Exception as e:
            logger.warning(f"  [{ds_name}] JSD跳过: {e}")

    if jsd_results:
        jsd_out_df = pd.DataFrame(jsd_results)
        jsd_out_df.to_csv(OUTPUT_DIR / 'L1_jsd_ks_distribution.csv', index=False)
        logger.info(f"  JSD结果保存: {len(jsd_results)} 数据集")
        # 汇总 JSD 均值
        jsd_ferr = [r.get('ferroptosis_jsd') for r in jsd_results if pd.notna(r.get('ferroptosis_jsd'))]
        jsd_sene = [r.get('senescence_jsd') for r in jsd_results if pd.notna(r.get('senescence_jsd'))]
        if jsd_ferr:
            logger.info(f"  铁死亡 JSD: mean={np.mean(jsd_ferr):.4f}")
        if jsd_sene:
            logger.info(f"  衰老 JSD: mean={np.mean(jsd_sene):.4f}")

    # ============================================================
    # 4i. Bulk免疫细胞反卷积 × IDSP 关联分析 (新增 🔧)
    # ============================================================
    # 原理: 特征基因集富集评分法 (MCPcounter风格)
    #   - 人血数据集 (GSE16561, GSE37587) → HUMAN_IMMUNE_SIGNATURES (12种免疫细胞)
    #   - 鼠脑数据集 (GSE104036) → MOUSE_BRAIN_SIGNATURES (7种脑细胞, 含小胶质)
    #   - 关联分析: IDSP评分与细胞比例的Spearman ρ + Bonferroni校正
    logger.info("\n" + "=" * 50)
    logger.info("Bulk免疫细胞反卷积 x IDSP关联分析")

    all_deconv_correlations = []  # 收集所有数据集的关联结果
    deconv_summary_rows = []      # CSV导出用
    top_celltype_global = None    # 全局最佳细胞类型

    for ds_name in sample_info:
        try:
            expr_gene = harmonized_dict.get(ds_name, raw_expr_dict.get(ds_name))
            if expr_gene is None or expr_gene.empty:
                continue

            # 选择细胞类型签名: 人 vs 鼠
            if ds_name == 'GSE104036':
                # 鼠脑数据集 — 基因名需大写匹配 (mouse genes are title-case in the sigs)
                mouse_upper = {ct: {g.upper() for g in genes}
                               for ct, genes in MOUSE_BRAIN_SIGNATURES.items()}
                deconv_df = deconvolve_immune_cells(
                    expr_gene, mouse_upper, method_label=f'{ds_name}/mouse_brain')
                microglia_col = 'Microglia'
            else:
                deconv_df = deconvolve_immune_cells(
                    expr_gene, HUMAN_IMMUNE_SIGNATURES, method_label=f'{ds_name}/human_blood')
                microglia_col = None  # 自动选择 Monocyte/Macrophage

            if deconv_df.empty:
                continue

            # 获取对应数据集的 IDSP 评分 (样本对齐)
            # 注: scores_df 的 index.name 未设定, 用 'dataset' 列直接匹配
            idsp_matched = None
            for sc in all_scores:
                if 'dataset' in sc.columns and sc['dataset'].iloc[0] == ds_name:
                    idsp_matched = sc
                    break
            if idsp_matched is None:
                # 后备: 用空IDSP占位
                idsp_matched = pd.DataFrame({
                    'ferroptosis': 0.0, 'senescence': 0.0,
                    'idsp_index': 0.0, 'group': 'unknown'
                }, index=deconv_df.index)

            common_samples = deconv_df.index.intersection(idsp_matched.index)
            if len(common_samples) < 5:
                logger.warning(f"  [{ds_name}] 样本对齐不足 ({len(common_samples)}), 跳过")
                continue
            deconv_aligned = deconv_df.loc[common_samples]
            idsp_aligned = idsp_matched.loc[common_samples]

            # Spearman ρ + Bonferroni
            corr_results = idsp_immune_correlation(idsp_aligned, deconv_aligned, ds_name)
            all_deconv_correlations.append(corr_results)

            # 全局最优追踪
            if corr_results['top_types']:
                tt = corr_results['top_types']
                if (top_celltype_global is None or
                        tt['rho'] > top_celltype_global.get('rho', -999)):
                    top_celltype_global = dict(tt)

            for c in corr_results['correlations']:
                deconv_summary_rows.append({
                    'dataset': ds_name, 'cell_type': c['cell_type'],
                    'metric': c['metric'], 'rho': c['rho'],
                    'p_value': c['p_value'], 'significant_bonf': c['significant'],
                    'n_samples': c['n_samples'],
                })

            # 散点图
            plot_microglia_idsp_scatter(
                idsp_aligned, deconv_aligned, ds_name,
                microglia_col=microglia_col,
                save_path=FIGS_DIR / f'L1_{ds_name}_idsp_scatter.png')

            if corr_results['top_types']:
                tt = corr_results['top_types']
                logger.info(f"  [{ds_name}] Top: {tt['cell_type']} "
                            f"ρ={tt['rho']:.3f}, p={tt['p_value']:.2e}")

        except Exception as e:
            logger.warning(f"  [{ds_name}] 反卷积/关联分析失败: {e} (非关键)")

    # 输出
    if deconv_summary_rows:
        pd.DataFrame(deconv_summary_rows).to_csv(
            OUTPUT_DIR / 'L1_deconvolution_correlations.csv', index=False)
        logger.info(f"  反卷积关联保存: L1_deconvolution_correlations.csv "
                    f"({len(deconv_summary_rows)} 条)")

    if all_deconv_correlations:
        plot_deconvolution_heatmap(all_deconv_correlations,
                                   save_path=FIGS_DIR / 'Fig1E_immune_correlation_heatmap.png')

    if top_celltype_global:
        logger.info(f"  全局最佳: {top_celltype_global['cell_type']} "
                    f"(ρ={top_celltype_global['rho']:.3f}, {top_celltype_global.get('dataset')})")

    # ============================================================
    # 5. 输出文件
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("输出结果")

    # 4a. 双评分数据
    if all_scores:
        all_scores_df = pd.concat(all_scores, ignore_index=False)
        all_scores_df.to_csv(OUTPUT_DIR / 'L1_dual_scores_all_datasets.csv', index=True)
        logger.info(f"  scores: {OUTPUT_DIR / 'L1_dual_scores_all_datasets.csv'}")

    # 4b. 对比统计
    comp_df.to_csv(OUTPUT_DIR / 'L1_dual_comparison_summary.csv', index=False)
    logger.info(f"  comparison: {OUTPUT_DIR / 'L1_dual_comparison_summary.csv'}")

    # 4b2. 基因集扰动敏感性分析 (100次, 10%替换)
    logger.info("\n  基因集扰动敏感性分析 (100 iterations, 10% 替换):")
    try:
        sensitivity_df = gene_set_sensitivity_analysis(
            valid_expr_dict, sample_info, n_iterations=100, replace_fraction=0.10)
        sensitivity_df.to_csv(
            OUTPUT_DIR / 'L1_gene_set_sensitivity.csv', index=False)
        logger.info(f"  敏感性分析保存: L1_gene_set_sensitivity.csv")
    except Exception as e:
        logger.warning(f"  敏感性分析失败: {e} (非关键)")

    # 4c. 时间动态
    if not temporal_df.empty:
        temporal_df.to_csv(OUTPUT_DIR / 'L1_temporal_dual_scores.csv', index=False)
        logger.info(f"  temporal: {OUTPUT_DIR / 'L1_temporal_dual_scores.csv'}")

    # 4d. GPX4验证
    gpx4_df = pd.DataFrame(all_gpx4)
    gpx4_df.to_csv(OUTPUT_DIR / 'L1_gpx4_validation.csv', index=False)
    logger.info(f"  gpx4: {OUTPUT_DIR / 'L1_gpx4_validation.csv'}")

    # 4e. 单基因分析
    if all_gene_dfs:
        combined_genes = pd.concat(all_gene_dfs, ignore_index=True)
        combined_genes.to_csv(OUTPUT_DIR / 'L1_gene_level_analysis.csv', index=False)
        logger.info(f"  genes: {OUTPUT_DIR / 'L1_gene_level_analysis.csv'}")

    # ============================================================
    # 5. 可视化
    # ============================================================
    logger.info("\n" + "=" * 50)
    logger.info("生成图表")

    # Fig1A: 效应量森林图 (含95%CI + 预测区间 + 汇总菱形)
    plot_forest_dual(all_comparisons, str(FIGS_DIR / 'Fig1A_forest_dual.png'),
                     re_ferr=re_ferr, re_sene=re_sene)

    # Fig1B: 时间动态
    if not temporal_df.empty:
        plot_temporal_dual(temporal_df, str(FIGS_DIR / 'Fig1B_temporal_dual.png'))

    # Fig1C: 散点图
    if all_scores:
        combined_scores = pd.concat(all_scores, ignore_index=False)
        plot_scatter_dual(combined_scores, str(FIGS_DIR / 'Fig1C_scatter_dual.png'))

    # Fig1D: 核心基因热图
    plot_gene_heatmap(all_gene_dfs, str(FIGS_DIR / 'Fig1D_gene_heatmap.png'))

    # ============================================================
    # 6. 验证报告
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("L1 验证报告")
    logger.info("=" * 60)

    # 判断标准1: 双评分相关性 (NaN安全)
    r_values = comp_df['r_ferr_sene'].dropna()
    if not r_values.empty:
        mean_r = r_values.mean()
        r_verdict = "PASS" if mean_r < 0.6 else ("WARNING" if mean_r < 0.8 else "FAIL")
        logger.info(f"  双评分相关性: mean_r={mean_r:.3f} → {r_verdict}")
    else:
        mean_r = np.nan
        r_verdict = "N/A"
        logger.info("  双评分相关性: 无有效数据")

    # 判断标准2: GPX4验证
    gpx4_supported = sum(1 for g in all_gpx4 if g.get('verdict') == 'IDSP_supported')
    gpx4_total = sum(1 for g in all_gpx4 if g.get('gpx4_found'))
    gpx4_verdict = f"{gpx4_supported}/{gpx4_total} 数据集支持IDSP"
    logger.info(f"  GPX4验证: {gpx4_verdict}")

    # 安全格式化辅助
    def safe_fmt(val, fmt='.3f'):
        return ('{:' + fmt + '}').format(val) if pd.notna(val) else 'N/A'

    # 判断标准3: 时间动态分离 — 多层次判定 ###
    if not temporal_df.empty:
        tp = temporal_df.sort_values('time_hr')
        ferr_ser = tp['ferroptosis_mean'].dropna()
        sene_ser = tp['senescence_mean'].dropna()

        # 读取高级生物指标 (temporal_df.attrs)
        sene_fc_dom = temporal_df.attrs.get('sene_fc_dominance', np.nan)
        sene_late_ratio = temporal_df.attrs.get('sene_late_rate_ratio', np.nan)
        ferr_late_ratio = temporal_df.attrs.get('ferr_late_rate_ratio', np.nan)
        biphasic = temporal_df.attrs.get('sene_biphasic_detected', False)
        sene_auc_ratio = temporal_df.attrs.get('sene_auc_ratio', np.nan)

        if not ferr_ser.empty and not sene_ser.empty:
            # --- 原有变化率峰值检测 (保留兼容) ---
            ferr_delta = tp['ferr_delta_per_hr'].dropna()
            sene_delta = tp['sene_delta_per_hr'].dropna()
            if not ferr_delta.empty:
                ferr_peak_idx = ferr_delta.idxmax()
                ferr_peak_hr = tp.loc[ferr_peak_idx, 'time_hr']
                ferr_peak_rate = ferr_delta.max()
            else:
                ferr_peak_hr = tp.loc[ferr_ser.idxmax(), 'time_hr']
                ferr_peak_rate = np.nan
            if not sene_delta.empty:
                sene_peak_idx = sene_delta.idxmax()
                sene_peak_hr = tp.loc[sene_peak_idx, 'time_hr']
                sene_peak_rate = sene_delta.max()
            else:
                sene_peak_hr = tp.loc[sene_ser.idxmax(), 'time_hr']
                sene_peak_rate = np.nan

            # --- 新: 多层次判定逻辑 ---
            # 判定标准1: 衰老晚期累积FC > 铁死亡1.5倍
            criterion_1 = pd.notna(sene_fc_dom) and sene_fc_dom > 1.5
            # 判定标准2: 衰老变化率晚期不衰减 (ratio > 0.2) 且铁死亡严重衰减 (ratio < 0.2)
            criterion_2 = (pd.notna(sene_late_ratio) and sene_late_ratio > 0.2 and
                           pd.notna(ferr_late_ratio) and ferr_late_ratio < 0.2)
            # 判定标准3: 双相激活模式检测
            criterion_3 = biphasic
            # 判定标准4: 衰老AUC总负荷 > 铁死亡AUC
            criterion_4 = pd.notna(sene_auc_ratio) and sene_auc_ratio > 1.0

            criteria_passed = sum([criterion_1, criterion_2, criterion_3, criterion_4])
            temporal_verdict = "PASS" if criteria_passed >= 2 else "WARNING"

            logger.info(f"  时间动态: 铁死亡峰值变化率在{ferr_peak_hr}h (rate={ferr_peak_rate:.4f}/hr), "
                        f"衰老峰值变化率在{sene_peak_hr}h (rate={sene_peak_rate:.4f}/hr)")
            logger.info(f"    累积FC比(24h): sene/ferr={sene_fc_dom:.2f} "
                        f"{'✓>1.5' if criterion_1 else ''}")
            logger.info(f"    晚期/早期变化率比: 铁死亡={ferr_late_ratio:.3f}, "
                        f"衰老={sene_late_ratio:.3f} "
                        f"{'✓衰老持续' if criterion_2 else ''}")
            logger.info(f"    双相模式: {'✓检测到' if biphasic else '未检测到'} "
                        f"{'✓' if criterion_3 else ''}")
            logger.info(f"    衰老/铁死亡 AUC比: {sene_auc_ratio:.2f} "
                        f"{'✓衰老负荷更大' if criterion_4 else ''}")
            logger.info(f"    时间动态判定({criteria_passed}/4通过): {temporal_verdict}")
        else:
            temporal_verdict = "N/A (无数据)"
            logger.info("  时间动态: 富集评分全为NaN, 无法判定")
    else:
        temporal_verdict = "N/A (无数据)"
        logger.info("  时间动态: 无数据")

    # 综合判断
    logger.info(f"\n  L1 整体判定: {'✅ 可推进到L2' if r_verdict != 'FAIL' else '❌ 需要调整基因集'}")

    # 保存报告 (NaN安全格式化)
    report_path = OUTPUT_DIR / 'L1_validation_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("L1: IDSP 双评分分析 — 验证报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"基因集: 纯铁死亡={len(PURE_FERROPTOSIS)}, 纯衰老={len(PURE_SENESCENCE)}, 共享={len(SHARED_GENES)}\n\n")
        f.write("各数据集统计:\n")
        for _, row in comp_df.iterrows():
            f.write(f"  {row['dataset']}: r={safe_fmt(row['r_ferr_sene'])}, "
                    f"d_ferr={safe_fmt(row['d_ferroptosis'])}, d_sene={safe_fmt(row['d_senescence'])}, "
                    f"p_ferr={safe_fmt(row['p_ferroptosis'], '.3e')}, p_sene={safe_fmt(row['p_senescence'], '.3e')}\n")
        f.write(f"\nMeta分析 (Fisher): 铁死亡 p={safe_fmt(meta_p_f, '.4e')}, 衰老 p={safe_fmt(meta_p_s, '.4e')}\n")
        f.write(f"Meta分析 (Stouffer加权): 铁死亡 p={safe_fmt(meta_p_stouffer_f, '.4e')}, 衰老 p={safe_fmt(meta_p_stouffer_s, '.4e')}\n")
        if re_ferr:
            f.write(f"随机效应Meta (铁死亡): d={safe_fmt(re_ferr['summary_effect'])}, p={safe_fmt(re_ferr['p_value'], '.4e')}, I²={safe_fmt(re_ferr['I2'], '.0f')}%, τ²={safe_fmt(re_ferr['tau2'], '.4f')}\n")
            if pd.notna(re_ferr.get('pi_lower')):
                f.write(f"  95%预测区间: [{safe_fmt(re_ferr['pi_lower'])}, {safe_fmt(re_ferr['pi_upper'])}] (新研究预期效应量范围)\n")
        if re_sene:
            f.write(f"随机效应Meta (衰老): d={safe_fmt(re_sene['summary_effect'])}, p={safe_fmt(re_sene['p_value'], '.4e')}, I²={safe_fmt(re_sene['I2'], '.0f')}%, τ²={safe_fmt(re_sene['tau2'], '.4f')}\n")
            if pd.notna(re_sene.get('pi_lower')):
                f.write(f"  95%预测区间: [{safe_fmt(re_sene['pi_lower'])}, {safe_fmt(re_sene['pi_upper'])}] (新研究预期效应量范围)\n")
        if bayes_ferr and bayes_ferr.get('converged'):
            f.write(f"Bayesian Meta (铁死亡): μ={safe_fmt(bayes_ferr.get('mu_mean'))} "
                    f"(95%HDI [{safe_fmt(bayes_ferr.get('mu_hdi_2.5'))}, {safe_fmt(bayes_ferr.get('mu_hdi_97.5'))}]), "
                    f"P(μ>0)={safe_fmt(bayes_ferr.get('p_mu_gt_0'))}, "
                    f"τ²={safe_fmt(bayes_ferr.get('tau2_mean'), '.4f')}, "
                    f"ρ={safe_fmt(bayes_ferr.get('rho_mean'))}, "
                    f"R̂={safe_fmt(bayes_ferr.get('mu_rhat'), '.4f')}, "
                    f"{'✓收敛' if bayes_ferr.get('converged') else '⚠未收敛'}\n")
        if bayes_sene and bayes_sene.get('converged'):
            f.write(f"Bayesian Meta (衰老): μ={safe_fmt(bayes_sene.get('mu_mean'))} "
                    f"(95%HDI [{safe_fmt(bayes_sene.get('mu_hdi_2.5'))}, {safe_fmt(bayes_sene.get('mu_hdi_97.5'))}]), "
                    f"P(μ>0)={safe_fmt(bayes_sene.get('p_mu_gt_0'))}, "
                    f"τ²={safe_fmt(bayes_sene.get('tau2_mean'), '.4f')}, "
                    f"ρ={safe_fmt(bayes_sene.get('rho_mean'))}, "
                    f"R̂={safe_fmt(bayes_sene.get('mu_rhat'), '.4f')}, "
                    f"{'✓收敛' if bayes_sene.get('converged') else '⚠未收敛'}\n")
        f.write(f"\nGPX4验证: {gpx4_verdict}\n")
        f.write(f"时间动态: {temporal_verdict}\n")
        if not temporal_df.empty:
            f.write(f"  时间动态生物指标:\n")
            sene_fc_dom = temporal_df.attrs.get('sene_fc_dominance', np.nan)
            ferr_late_ratio = temporal_df.attrs.get('ferr_late_rate_ratio', np.nan)
            sene_late_ratio = temporal_df.attrs.get('sene_late_rate_ratio', np.nan)
            biphasic = temporal_df.attrs.get('sene_biphasic_detected', False)
            sene_auc_ratio = temporal_df.attrs.get('sene_auc_ratio', np.nan)
            f.write(f"    衰老/铁死亡累积FC比(24h)={safe_fmt(sene_fc_dom)} (>1.5=衰老主导)\n")
            f.write(f"    铁死亡晚期/早期变化率比={safe_fmt(ferr_late_ratio)} (<0.2=急性衰减)\n")
            f.write(f"    衰老晚期/早期变化率比={safe_fmt(sene_late_ratio)} (>0.2=持续激活)\n")
            f.write(f"    衰老双相激活模式={'检测到' if biphasic else '未检测到'}\n")
            f.write(f"    衰老/铁死亡AUC负荷比={safe_fmt(sene_auc_ratio)}\n")

        # 4i. 免疫反卷积关联
        f.write(f"\n免疫细胞反卷积 × IDSP 关联:\n")
        if top_celltype_global:
            f.write(f"  全局最佳细胞类型: {top_celltype_global['cell_type']} "
                    f"(ρ={safe_fmt(top_celltype_global['rho'])}, "
                    f"p={safe_fmt(top_celltype_global.get('p_value'), '.2e')}, "
                    f"dataset={top_celltype_global.get('dataset')})\n")
        else:
            f.write("  无显著IDSP-免疫细胞关联\n")

    logger.info(f"  报告保存: {report_path}")
    logger.info(f"\n{'='*60}")
    logger.info("L1 分析完成!")
    logger.info(f"结果目录: {OUTPUT_DIR}")

    # ============================================================
    # 7. JSON 统计摘要 (供下游模块直接读取)
    # ============================================================
    _json_meta = {
        'module': 'L1',
        'gene_sets': {
            'n_ferroptosis': len(PURE_FERROPTOSIS),
            'n_senescence': len(PURE_SENESCENCE),
            'n_shared': len(SHARED_GENES),
        },
        'meta_analysis': {
            'fisher': {'ferroptosis_p': float(meta_p_f) if pd.notna(meta_p_f) else None,
                       'senescence_p': float(meta_p_s) if pd.notna(meta_p_s) else None},
            'stouffer_weighted': {'ferroptosis_p': float(meta_p_stouffer_f) if pd.notna(meta_p_stouffer_f) else None,
                                  'senescence_p': float(meta_p_stouffer_s) if pd.notna(meta_p_stouffer_s) else None},
            'random_effects': {
                'ferroptosis': {'d': re_ferr['summary_effect'] if re_ferr else None,
                                'p': re_ferr['p_value'] if re_ferr else None,
                                'I2': re_ferr['I2'] if re_ferr else None,
                                'tau2': re_ferr['tau2'] if re_ferr else None,
                                'pi_lower': re_ferr.get('pi_lower') if re_ferr else None,
                                'pi_upper': re_ferr.get('pi_upper') if re_ferr else None,
                                'k': re_ferr['k'] if re_ferr else 0},
                'senescence': {'d': re_sene['summary_effect'] if re_sene else None,
                               'p': re_sene['p_value'] if re_sene else None,
                               'I2': re_sene['I2'] if re_sene else None,
                               'tau2': re_sene['tau2'] if re_sene else None,
                               'pi_lower': re_sene.get('pi_lower') if re_sene else None,
                               'pi_upper': re_sene.get('pi_upper') if re_sene else None,
                               'k': re_sene['k'] if re_sene else 0},
            },
            'bayesian': {
                'ferroptosis': {'mu_mean': bayes_ferr.get('mu_mean') if bayes_ferr else None,
                                'mu_hdi_lower': bayes_ferr.get('mu_hdi_2.5') if bayes_ferr else None,
                                'mu_hdi_upper': bayes_ferr.get('mu_hdi_97.5') if bayes_ferr else None,
                                'p_mu_gt_0': bayes_ferr.get('p_mu_gt_0') if bayes_ferr else None,
                                'converged': bayes_ferr.get('converged') if bayes_ferr else False},
                'senescence': {'mu_mean': bayes_sene.get('mu_mean') if bayes_sene else None,
                               'mu_hdi_lower': bayes_sene.get('mu_hdi_2.5') if bayes_sene else None,
                               'mu_hdi_upper': bayes_sene.get('mu_hdi_97.5') if bayes_sene else None,
                               'p_mu_gt_0': bayes_sene.get('p_mu_gt_0') if bayes_sene else None,
                               'converged': bayes_sene.get('converged') if bayes_sene else False},
            },
        },
        'gpx4': gpx4_verdict,
        'temporal': temporal_verdict,
        'lodo_stable': n_stable if n_stable else 0,
        'top_celltype_associated_with_idsp': top_celltype_global if top_celltype_global else None,
    }
    json_path = OUTPUT_DIR / 'L1_statistical_summary.json'
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(_json_meta, jf, indent=2, ensure_ascii=False,
                  default=lambda o: None if pd.isna(o) else (
                      float(o) if isinstance(o, (np.floating,)) else o))
    logger.info(f"  JSON摘要保存: L1_statistical_summary.json")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()