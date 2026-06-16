#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
模块二: 单细胞分辨率的脑I/R铁衰老细胞图谱
=====================================================================
数据集: GSE174574 (小鼠 MCAO vs Sham, 10X scRNA-seq, 皮层+海马)
样本: 3 MCAO + 3 Sham

分析流程:
  Step A: scRNA-seq 标准处理 (QC → 归一化 → 聚类 → 标注)
  Step B: 每个细胞计算铁衰老双评分
  Step B2: 共识衰老评分 (SenMayo + CellAge + FRIDMAN 取中位数)
  Step C: 差异分析 + 通路富集
  Step D: 拟时序分析

输出:
  - module2_results/ 目录下所有图表和数据
  - Fig2A: UMAP 细胞类型标注 + 铁衰老评分覆盖
  - Fig2B: 小提琴图 各细胞类型铁衰老评分 (Sham vs MCAO)
  - Fig2C: 拟时序图 铁衰老动态轨迹
  - Fig2D: 气泡图 核心基因在各细胞类型表达
=====================================================================
"""

import os, sys, warnings, logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 基因集定义 (内联, 复用L1)
# ============================================================
PURE_FERROPTOSIS = {
    'ACSL4', 'PTGS2', 'HMOX1', 'TFRC', 'SLC7A11', 'CHAC1', 'SLC3A2',
    'FTH1', 'FTL', 'NFE2L2', 'GPX4', 'DPP4', 'ALOX5', 'ALOX12',
    'SAT1', 'SLC40A1', 'STEAP3', 'CBS', 'CSE1L', 'HSPB1',
    'VDAC2', 'VDAC3', 'GOT1', 'GCLC', 'GCLM', 'ABCC1', 'ABCC2',
    'ATF3', 'ATF4', 'DDIT3', 'SESN2', 'FANCD2', 'CDO1',
    'ZEB1', 'SNAI1', 'MMP9', 'RGS4', 'SQSTM1', 'NCOA4', 'BECN1',
    'PRNP', 'ADIPOQ', 'PLIN2', 'LPIN1', 'LPIN2', 'PNPLA2',
    'MAP1LC3A', 'MAP1LC3B', 'GABARAP', 'GABARAPL1',
    'ATG3', 'ATG5', 'ATG7', 'BAP1', 'TRIB3', 'KEAP1',
    'TFAM', 'PPARGC1A', 'SIRT1', 'FOXO1', 'FOXO3',
    'PRKAA1', 'PRKAA2', 'NFKB1', 'RELA', 'BNIP3', 'BNIP3L',
    'HSP90AA1', 'HSPA5', 'HSPD1', 'EIF2AK3', 'EIF2A',
}

PURE_SENESCENCE = {
    'CDKN2A', 'CDKN2B', 'CDKN1A', 'CDKN1B', 'RB1', 'E2F1', 'E2F2',
    'E2F3', 'CCND1', 'CCNE1', 'CCNA2', 'CCNB1', 'CDK4', 'CDK6',
    'CDK2', 'TP53', 'MDM2', 'MDM4', 'CHEK1', 'CHEK2',
    'ATM', 'ATR', 'H2AX', 'GADD45A', 'GADD45B', 'SERPINE1',
    'IGFBP3', 'IGFBP5', 'IGFBP7', 'IL6', 'IL1A', 'IL1B',
    'CCL2', 'CCL3', 'CCL4', 'CXCL1', 'CXCL2', 'CXCL10',
    'MMP1', 'MMP2', 'MMP3', 'MMP10', 'MMP12', 'MMP13',
    'TIMP1', 'TIMP2', 'FN1', 'COL1A1', 'COL1A2', 'COL3A1',
    'LMNB1', 'HMGB1', 'HMGA1', 'HMGA2', 'SIRT6',
    'FOXO4', 'STAT3', 'JAK2', 'MAPK1', 'MAPK3', 'MAPK8',
    'MAPK14', 'AKT1', 'MTOR', 'RPS6KB1', 'PTEN', 'TSC1', 'TSC2',
    'CREB1', 'ATF2', 'JUN', 'FOS', 'MYC', 'MAX', 'MNT',
    'HDAC1', 'HDAC2', 'HDAC3', 'EP300', 'CREBBP', 'BRD4',
    'PARP1', 'BUB1B', 'BUB1', 'BUB3', 'CDC20', 'MAD2L1',
    'PLK1', 'AURKA', 'AURKB', 'TOP2A', 'MKI67', 'PCNA',
    'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7',
    'RFC1', 'RFC2', 'RFC3', 'RFC4', 'RFC5',
    'RPA1', 'RPA2', 'RPA3', 'LIG1', 'LIG3', 'LIG4',
    'XRCC1', 'XRCC6', 'XRCC5', 'PRKDC', 'NBN', 'MRE11',
    'RAD50', 'RAD51', 'BRCA1', 'BRCA2', 'BLM', 'WRN',
    'TERF1', 'TERF2', 'TERT', 'CD38', 'CD4', 'CD8A',
    'CSF2', 'CSF3', 'IFNG', 'TNF', 'TGFB1', 'VEGFA',
    'ICAM1', 'VCAM1', 'SELE', 'IL18', 'IL10', 'TNFRSF1A',
}

SHARED_GENES = {
    'TP53', 'CDKN1A', 'RB1', 'CD74', 'S100A8', 'IFNG',
    'IRF1', 'TLR4', 'NLRP3', 'HIF1A', 'KEAP1', 'SOD1',
}

# 共识衰老基因集 (SenMayo + CellAge + FRIDMAN - 取中位数)
# SenMayo: Saul 2022 Nat Commun 核心衰老基因
SENMAYO_GENES = {
    'ACVR1B', 'ANG', 'ANGPTL4', 'AREG', 'AXL', 'BEX3', 'BMP2', 'BMP6',
    'BRIX1', 'BTG2', 'C3', 'CAPN2', 'CCL1', 'CCL13', 'CCL16', 'CCL2',
    'CCL20', 'CCL24', 'CCL26', 'CCL3', 'CCL4', 'CCL7', 'CCL8', 'CCR1',
    'CD55', 'CDKN1A', 'CDKN2B', 'CEBPB', 'CETN3', 'CLU', 'COL1A1',
    'COL3A1', 'CSF1', 'CSF2', 'CSF2RB', 'CTSH', 'CXCL1', 'CXCL10',
    'CXCL11', 'CXCL12', 'CXCL13', 'CXCL14', 'CXCL16', 'CXCL2', 'CXCL3',
    'CXCL5', 'CXCL8', 'CXCL9', 'CXCR3', 'DDIT4', 'DKK1', 'EDN1',
    'EGF', 'EGFR', 'EREG', 'ESM1', 'ETS2', 'FAS', 'FGF2', 'FGF7',
    'GADD45A', 'GADD45B', 'GDF15', 'GEM', 'GPC4', 'HGF', 'HMGB1',
    'ICAM1', 'ICAM3', 'IGF1', 'IGF2', 'IGFBP1', 'IGFBP2', 'IGFBP3',
    'IGFBP4', 'IGFBP5', 'IGFBP6', 'IGFBP7', 'IL10', 'IL13', 'IL15',
    'IL18', 'IL1A', 'IL1B', 'IL6', 'IL7', 'IL8', 'INHBA', 'IRS1',
    'ITGA2', 'KITLG', 'LCP1', 'MIF', 'MMP1', 'MMP10', 'MMP12', 'MMP13',
    'MMP14', 'MMP2', 'MMP3', 'MMP7', 'MMP9', 'NAP1L4', 'NRG1', 'PAPPA',
    'PECAM1', 'PLAU', 'PLAUR', 'PTGES', 'S100A8', 'SERPINB4', 'SERPINE1',
    'SERPINE2', 'SPP1', 'SPX', 'TIMP1', 'TIMP2', 'TNF', 'TNFRSF10C',
    'TNFRSF11B', 'TNFRSF1A', 'TNFSF11', 'VEGFA', 'VEGFC', 'VGF', 'WNT16',
    'WNT2', 'IGFBP2', 'CXCL8', 'MDK', 'PTX3', 'TGFB1', 'TGFB2',
}

# CellAge: Avelar 2020 Genome Biol 衰老基因
CELLAGE_GENES = {
    'AKT1', 'AR', 'ATM', 'BRAF', 'BRCA1', 'CAV1', 'CCNA2', 'CCNB1',
    'CCND1', 'CCNE1', 'CCNE2', 'CDC25A', 'CDC25C', 'CDC6', 'CDK1',
    'CDK2', 'CDK4', 'CDK6', 'CDKN1A', 'CDKN1B', 'CDKN2A', 'CDKN2B',
    'CDKN2C', 'CDKN2D', 'CHEK1', 'CHEK2', 'CITED2', 'CREBBP', 'E2F1',
    'E2F2', 'E2F3', 'E2F4', 'E2F5', 'EGFR', 'EP300', 'ERBB2', 'ETS1',
    'ETS2', 'EZH2', 'FOS', 'FOXM1', 'FOXO1', 'FOXO3', 'FOXO4', 'GATA4',
    'HDAC1', 'HDAC2', 'HIF1A', 'HMGA1', 'HMGA2', 'HRAS', 'IGFBP3',
    'IGFBP5', 'IGFBP7', 'IL6', 'JUN', 'JUNB', 'JUND', 'KLF4', 'KRAS',
    'LMNB1', 'MAP2K1', 'MAP2K2', 'MAPK1', 'MAPK14', 'MAPK3', 'MAPK8',
    'MCM2', 'MCM3', 'MCM4', 'MCM5', 'MCM6', 'MCM7', 'MDM2', 'MKI67',
    'MMP1', 'MMP2', 'MMP3', 'MMP9', 'MYC', 'MYCN', 'NFKB1', 'NFKB2',
    'NOTCH1', 'NOTCH2', 'NOTCH3', 'NRAS', 'PCNA', 'PIK3CA', 'PLK1',
    'PTEN', 'RB1', 'RELA', 'SIRT1', 'SIRT6', 'SMAD2', 'SMAD3', 'SMAD4',
    'SP1', 'STAT3', 'TBX2', 'TBX3', 'TERT', 'TGFB1', 'TGFB2', 'TGFB3',
    'TGFBR1', 'TGFBR2', 'TNF', 'TP53', 'TP63', 'TP73', 'TWIST1',
    'VEGFA', 'WRN', 'YBX1', 'ZEB1', 'ZEB2',
}

# FRIDMAN: 衰老核心基因集 (Fridman & Tainsky 2008)
FRIDMAN_GENES = {
    'CDKN1A', 'CDKN2A', 'CDKN2B', 'TP53', 'RB1', 'CCND1', 'CCNE1',
    'CDK4', 'CDK6', 'CDK2', 'E2F1', 'E2F2', 'E2F3', 'MYC', 'FOS',
    'JUN', 'STAT3', 'NFKB1', 'RELA', 'SIRT1', 'LMNB1', 'HMGB1',
    'HMGA1', 'HMGA2', 'IL6', 'IL1B', 'TNF', 'TGFB1', 'CXCL8',
    'CCL2', 'MMP1', 'MMP2', 'MMP3', 'MMP9', 'SERPINE1', 'TIMP1',
    'TIMP2', 'IGFBP3', 'IGFBP5', 'IGFBP7', 'VEGFA', 'FGF2', 'EGF',
    'HGF', 'IGF1', 'PDGFA', 'PDGFB', 'AREG', 'EREG', 'HBEGF',
    'CDKN1B', 'CDKN2C', 'CDKN2D', 'CCNA2', 'CCNB1', 'CCNB2',
    'CDC2', 'CDC20', 'BUB1', 'BUB1B', 'PLK1', 'AURKA', 'AURKB',
    'MKI67', 'PCNA', 'TOP2A', 'BRCA1', 'BRCA2', 'ATM', 'ATR',
    'CHEK1', 'CHEK2', 'GADD45A', 'GADD45B', 'MDM2', 'PTEN',
    'AKT1', 'MTOR', 'RPS6KB1', 'MAPK1', 'MAPK3', 'MAPK14',
}

# 脑细胞类型 marker 基因 (小鼠)
BRAIN_CELL_MARKERS = {
    'Neuron':       ['Rbfox3', 'Syt1', 'Snap25', 'Syp', 'Tubb3', 'Map2', 'Nefh', 'Nefl'],
    'Microglia':    ['Cx3cr1', 'Aif1', 'Tmem119', 'P2ry12', 'Csf1r', 'Itgam', 'Hexb', 'C1qa'],
    'Astrocyte':    ['Gfap', 'Aqp4', 'S100b', 'Aldh1l1', 'Slc1a3', 'Glast', 'Ntsr2', 'Cx43'],
    'Oligodendrocyte': ['Mbp', 'Mog', 'Plp1', 'Cnp', 'Mag', 'Olig1', 'Olig2', 'Sox10'],
    'Endothelial':  ['Cldn5', 'Pecam1', 'Flt1', 'Cdh5', 'Tek', 'Esam', 'Vwf', 'Eng'],
    'Pericyte':     ['Pdgfrb', 'Cspg4', 'Anpep', 'Rgs5', 'Acta2', 'Des', 'Notch3', 'Kcnj8'],
}

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "module2_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR = OUTPUT_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# 10X 数据路径
DATA_10X = Path(r'D:\反向网络药理学\L1 数据集\RNA-seq\GSE174574_10X_organized')
SERIES_MATRIX = Path(r'D:\反向网络药理学\L1 数据集\RNA-seq\GSE174574_series_matrix.txt\GSE174574_series_matrix.txt')

# 样本列表
SAMPLES = {
    'MCAO_1': 'MCAO',
    'MCAO_2': 'MCAO',
    'MCAO_3': 'MCAO',
    'Sham_1': 'Sham',
    'Sham_2': 'Sham',
    'Sham_3': 'Sham',
}

# ============================================================
# 辅助函数
# ============================================================

def rank_sum_enrichment_score(expr: np.ndarray, gene_mask: np.ndarray) -> float:
    """单样本秩和富集评分"""
    n_genes = len(expr)
    n_set = gene_mask.sum()
    if n_set == 0 or n_set == n_genes:
        return 0.0
    valid = ~np.isnan(expr)
    if valid.sum() < 10:
        return 0.0
    ranks = stats.rankdata(expr[valid], method='average')
    mask_valid = gene_mask[valid]
    set_ranks = ranks[mask_valid]
    n_valid = len(ranks)
    n_set_valid = mask_valid.sum()
    if n_set_valid == 0:
        return 0.0
    expected = n_set_valid * (n_valid + 1) / 2.0
    sum_ranks = set_ranks.sum()
    max_dev = n_set_valid * (n_valid - n_set_valid)
    if max_dev == 0:
        return 0.0
    return float((sum_ranks - expected) / (max_dev / 2.0))


def calc_idsp_index(ferr_score: np.ndarray, sene_score: np.ndarray) -> np.ndarray:
    """IDSP Index = z(ferr) + z(sene) - |z(ferr) - z(sene)|"""
    ferr_std = np.nanstd(ferr_score)
    sene_std = np.nanstd(sene_score)
    z_ferr = ((ferr_score - np.nanmean(ferr_score)) / ferr_std) if ferr_std > 0 else np.zeros_like(ferr_score)
    z_sene = ((sene_score - np.nanmean(sene_score)) / sene_std) if sene_std > 0 else np.zeros_like(sene_score)
    return z_ferr + z_sene - np.abs(z_ferr - z_sene)


def cohens_d(case: np.ndarray, control: np.ndarray) -> float:
    """Cohen's d 效应量"""
    n1, n2 = len(case), len(control)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = np.var(case, ddof=1)
    s2 = np.var(control, ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(case) - np.mean(control)) / pooled if pooled > 0 else 0.0


# ============================================================
# Step A: 加载 + QC + 标准处理
# ============================================================

def load_and_preprocess(data_dir: Path, sample_dict: Dict[str, str]) -> 'sc.AnnData':
    """加载10X数据, 合并, QC, 归一化, 聚类"""
    import scanpy as sc

    logger.info("=" * 60)
    logger.info("Step A: 加载10X数据 + 标准处理")

    adatas = []
    for sample_name, condition in sample_dict.items():
        sample_path = data_dir / sample_name
        if not sample_path.exists():
            logger.warning(f"  样本目录不存在: {sample_path}")
            continue
        logger.info(f"  加载 {sample_name} ({condition})...")
        adata = sc.read_10x_mtx(sample_path, var_names='gene_symbols', cache=False)
        adata.var_names_make_unique()
        adata.obs['sample'] = sample_name
        adata.obs['condition'] = condition
        adatas.append(adata)

    adata = adatas[0].concatenate(adatas[1:], batch_key='sample', index_unique=None)
    logger.info(f"  合并后: {adata.shape[0]} 细胞, {adata.shape[1]} 基因")

    # 基因名大写统一
    adata.var_names = [g.upper() for g in adata.var_names]
    # Ensure unique var_names after concatenation
    adata.var_names_make_unique()

    # QC 指标
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS', 'RPL'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt', 'ribo'], percent_top=None, log1p=False, inplace=True)

    # QC 过滤
    logger.info(f"  QC前: {adata.n_obs} 细胞")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    logger.info(f"  min_genes>=200: {adata.n_obs} 细胞")

    # 线粒体过滤 (脑组织允许稍高)
    adata = adata[adata.obs['pct_counts_mt'] < 20, :].copy()
    logger.info(f"  pct_mt<20: {adata.n_obs} 细胞")

    # 基因数过滤
    adata = adata[adata.obs['n_genes_by_counts'] < 6000, :].copy()
    logger.info(f"  n_genes<6000: {adata.n_obs} 细胞")

    # 归一化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 高变基因 (use seurat flavor, simpler)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat')
    adata.raw = adata

    # PCA
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata, min_dist=0.3, spread=1.0)

    # Leiden 聚类
    sc.tl.leiden(adata, resolution=0.8, key_added='leiden')

    return adata


# ============================================================
# Step A2: 细胞类型标注
# ============================================================

def annotate_cell_types(adata, markers: Dict[str, List[str]]) -> 'sc.AnnData':
    """基于 marker 基因评分标注细胞类型"""
    import scanpy as sc

    logger.info("=" * 60)
    logger.info("Step A2: 细胞类型标注")

    # 为每个细胞类型计算 marker 基因平均表达
    score_dict = {}
    for ct_name, ct_genes in markers.items():
        ct_genes_upper = [g.upper() for g in ct_genes]
        available = [g for g in ct_genes_upper if g in adata.var_names]
        if len(available) < 2:
            logger.warning(f"  {ct_name}: 仅 {len(available)} 个 marker 可用, 跳过")
            continue
        # 计算每个细胞的平均 marker 表达
        expr = adata[:, available].X
        if issparse(expr):
            expr = expr.toarray()
        score_dict[ct_name] = np.mean(expr, axis=1)
        logger.info(f"  {ct_name}: {len(available)}/{len(ct_genes)} marker 可用")

    if not score_dict:
        logger.error("  无有效细胞类型标注!")
        return adata

    scores_df = pd.DataFrame(score_dict, index=adata.obs_names)
    adata.obs['cell_type'] = scores_df.idxmax(axis=1)

    # 置信度过滤: 最高分需 > 第2高分的 1.2 倍
    sorted_scores = np.sort(scores_df.values, axis=1)
    confidence = sorted_scores[:, -1] / (sorted_scores[:, -2] + 1e-12)
    low_conf_mask = confidence < 1.2
    if low_conf_mask.sum() > 0:
        adata.obs.loc[low_conf_mask, 'cell_type'] = 'Unassigned'
        logger.info(f"  Unassigned (低置信度): {low_conf_mask.sum()} 细胞")

    # 统计
    for ct in adata.obs['cell_type'].value_counts().index:
        n = adata.obs['cell_type'].value_counts()[ct]
        logger.info(f"    {ct}: {n} 细胞")

    return adata


# ============================================================
# Step B: 铁衰老双评分
# ============================================================

def compute_ferroptosis_aging_scores(adata) -> 'sc.AnnData':
    """每个细胞计算铁死亡 + 衰老 + 共识衰老评分"""
    logger.info("=" * 60)
    logger.info("Step B: 铁衰老双评分")

    # 获取表达矩阵
    X = adata.X
    if issparse(X):
        X = X.toarray()

    # 铁死亡评分
    ferro_genes = [g for g in PURE_FERROPTOSIS if g in adata.var_names]
    if len(ferro_genes) < 5:
        logger.error(f"  铁死亡基因仅 {len(ferro_genes)} 个可用!")
        return adata
    ferro_mask = np.array([g in ferro_genes for g in adata.var_names])
    logger.info(f"  铁死亡基因: {len(ferro_genes)}/{len(PURE_FERROPTOSIS)} 可用")

    ferro_scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        ferro_scores[i] = rank_sum_enrichment_score(X[i], ferro_mask)
    adata.obs['ferroptosis_score'] = ferro_scores

    # 衰老评分
    sene_genes = [g for g in PURE_SENESCENCE if g in adata.var_names]
    sene_mask = np.array([g in sene_genes for g in adata.var_names])
    logger.info(f"  衰老基因: {len(sene_genes)}/{len(PURE_SENESCENCE)} 可用")

    sene_scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        sene_scores[i] = rank_sum_enrichment_score(X[i], sene_mask)
    adata.obs['senescence_score'] = sene_scores

    # 共识衰老评分 (SenMayo + CellAge + FRIDMAN 取中位数)
    consensus_scores = np.zeros((X.shape[0], 3))
    for idx, (gset_name, gset) in enumerate([
        ('SenMayo', SENMAYO_GENES),
        ('CellAge', CELLAGE_GENES),
        ('FRIDMAN', FRIDMAN_GENES),
    ]):
        available = [g for g in gset if g in adata.var_names]
        gs_mask = np.array([g in available for g in adata.var_names])
        for i in range(X.shape[0]):
            consensus_scores[i, idx] = rank_sum_enrichment_score(X[i], gs_mask)
        logger.info(f"  {gset_name}: {len(available)}/{len(gset)} 可用")

    adata.obs['consensus_senescence'] = np.median(consensus_scores, axis=1)

    # IDSP Index
    adata.obs['idsp_index'] = calc_idsp_index(
        adata.obs['ferroptosis_score'].values,
        adata.obs['senescence_score'].values
    )

    logger.info(f"  铁死亡评分: mean={adata.obs['ferroptosis_score'].mean():.3f}")
    logger.info(f"  衰老评分: mean={adata.obs['senescence_score'].mean():.3f}")
    logger.info(f"  共识衰老评分: mean={adata.obs['consensus_senescence'].mean():.3f}")
    logger.info(f"  IDSP Index: mean={adata.obs['idsp_index'].mean():.3f}")

    return adata


# ============================================================
# Step C: 差异分析
# ============================================================

def differential_analysis(adata) -> pd.DataFrame:
    """每细胞类型 MCAO vs Sham 差异分析"""
    import scanpy as sc

    logger.info("=" * 60)
    logger.info("Step C: 差异分析 (MCAO vs Sham per cell type)")

    results = []

    for ct in adata.obs['cell_type'].unique():
        sub = adata[adata.obs['cell_type'] == ct].copy()
        mcao = sub[sub.obs['condition'] == 'MCAO']
        sham = sub[sub.obs['condition'] == 'Sham']

        if mcao.n_obs < 10 or sham.n_obs < 10:
            continue

        for score_name in ['ferroptosis_score', 'senescence_score',
                           'consensus_senescence', 'idsp_index']:
            mcao_vals = mcao.obs[score_name].dropna().values
            sham_vals = sham.obs[score_name].dropna().values

            if len(mcao_vals) < 3 or len(sham_vals) < 3:
                continue

            _, pval = stats.ttest_ind(mcao_vals, sham_vals, equal_var=False)
            d = cohens_d(mcao_vals, sham_vals)

            results.append({
                'cell_type': ct,
                'score': score_name,
                'mcao_mean': np.mean(mcao_vals),
                'sham_mean': np.mean(sham_vals),
                'log2FC': np.mean(mcao_vals) - np.mean(sham_vals),
                'cohens_d': d,
                'pvalue': pval,
                'n_mcao': len(mcao_vals),
                'n_sham': len(sham_vals),
            })

    df = pd.DataFrame(results)
    if not df.empty:
        _, padj, _, _ = multipletests(df['pvalue'], method='fdr_bh')
        df['padj'] = padj
        df['significant'] = df['padj'] < 0.05
        df = df.sort_values('pvalue')

    logger.info(f"  差异分析结果: {len(df)} 项, {df['significant'].sum() if not df.empty else 0} 显著")

    return df


# ============================================================
# 可视化
# ============================================================

def plot_umap_cell_types(adata, save_path: str):
    """Fig2A: UMAP 细胞类型"""
    import scanpy as sc
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Panel 1: 细胞类型
    sc.pl.umap(adata, color='cell_type', ax=axes[0], show=False,
               title='Cell Types', legend_loc='right margin',
               palette='tab10', frameon=False)
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # Panel 2: 铁死亡评分
    sc.pl.umap(adata, color='ferroptosis_score', ax=axes[1], show=False,
               title='Ferroptosis Score', cmap='Reds', vmin=-1, vmax=1,
               frameon=False)
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    # Panel 3: IDSP Index
    sc.pl.umap(adata, color='idsp_index', ax=axes[2], show=False,
               title='IDSP Index', cmap='RdYlBu_r', vmin=-2, vmax=2,
               frameon=False)
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')

    plt.suptitle('Fig2A: UMAP — Cell Types & Ferroptosis-Aging Scores', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig2A 保存: {save_path}")


def plot_violin_scores(adata, save_path: str):
    """Fig2B: 小提琴图 各细胞类型铁衰老评分 (Sham vs MCAO)"""
    ct_order = sorted(adata.obs['cell_type'].unique())
    n_ct = len(ct_order)

    fig, axes = plt.subplots(2, 2, figsize=(max(12, n_ct * 2.5), 12))
    scores = ['ferroptosis_score', 'senescence_score', 'consensus_senescence', 'idsp_index']
    titles = ['Ferroptosis Score', 'Senescence Score', 'Consensus Senescence', 'IDSP Index']
    cmaps = ['Reds', 'Blues', 'Purples', 'RdYlBu_r']

    for idx, (score, title, cmap) in enumerate(zip(scores, titles, cmaps)):
        ax = axes[idx // 2, idx % 2]
        positions = []
        labels = []
        colors = []

        for i, ct in enumerate(ct_order):
            sub = adata[adata.obs['cell_type'] == ct]
            mcao = sub[sub.obs['condition'] == 'MCAO'].obs[score].dropna().values
            sham = sub[sub.obs['condition'] == 'Sham'].obs[score].dropna().values

            if len(mcao) < 3 or len(sham) < 3:
                continue

            pos_mcao = i * 2.5 - 0.5
            pos_sham = i * 2.5 + 0.5

            vp1 = ax.violinplot(mcao, positions=[pos_mcao], showmeans=True,
                                showmedians=True, widths=0.8)
            vp2 = ax.violinplot(sham, positions=[pos_sham], showmeans=True,
                                showmedians=True, widths=0.8)

            for body in vp1['bodies']:
                body.set_facecolor('#E74C3C')
                body.set_alpha(0.7)
            for body in vp2['bodies']:
                body.set_facecolor('#3498DB')
                body.set_alpha(0.7)

            # 显著性标注
            _, pval = stats.ttest_ind(mcao, sham, equal_var=False)
            if pval < 0.05:
                y_max = max(np.max(mcao), np.max(sham))
                ax.text(i * 2.5, y_max + 0.1, '*' if pval > 0.01 else '**' if pval > 0.001 else '***',
                        ha='center', fontsize=10, fontweight='bold')

            positions.extend([pos_mcao, pos_sham])
            labels.extend([f'{ct[:8]}\nMCAO', f'{ct[:8]}\nSham'])

        ax.set_xticks([i * 2.5 for i in range(len(ct_order))])
        ax.set_xticklabels([ct[:10] for ct in ct_order], rotation=45, ha='right', fontsize=8)
        ax.set_title(title)
        ax.axhline(y=0, color='gray', ls='--', lw=0.5)

        # 图例
        legend_elements = [
            mpatches.Patch(facecolor='#E74C3C', alpha=0.7, label='MCAO'),
            mpatches.Patch(facecolor='#3498DB', alpha=0.7, label='Sham'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.suptitle('Fig2B: Ferroptosis-Aging Scores per Cell Type (MCAO vs Sham)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig2B 保存: {save_path}")


def plot_bubble_genes(adata, save_path: str):
    """Fig2D: 气泡图 核心基因在各细胞类型表达"""
    core_genes = ['ACSL4', 'PTGS2', 'HMOX1', 'TFRC', 'GPX4', 'SLC7A11',
                  'CDKN1A', 'IL6', 'IL1B', 'HMGB1', 'TP53', 'NFE2L2',
                  'HIF1A', 'KEAP1', 'SOD1', 'CXCL10', 'IGFBP7', 'MMP9']
    available_genes = [g for g in core_genes if g in adata.var_names]
    if len(available_genes) < 5:
        logger.warning("  核心基因可用不足5个, 跳过热图")
        return

    ct_order = sorted([ct for ct in adata.obs['cell_type'].unique() if ct != 'Unassigned'])

    # 计算每个细胞类型 × 基因的平均表达和表达比例
    mean_expr = np.zeros((len(ct_order), len(available_genes)))
    pct_expr = np.zeros((len(ct_order), len(available_genes)))

    X = adata.X
    if issparse(X):
        X = X.toarray()

    for i, ct in enumerate(ct_order):
        ct_mask = adata.obs['cell_type'] == ct
        ct_X = X[ct_mask]
        for j, gene in enumerate(available_genes):
            gene_idx = list(adata.var_names).index(gene)
            gene_expr = ct_X[:, gene_idx]
            mean_expr[i, j] = np.mean(gene_expr)
            pct_expr[i, j] = np.mean(gene_expr > 0) * 100

    # 气泡图
    fig, ax = plt.subplots(figsize=(max(10, len(available_genes) * 0.7),
                                    max(5, len(ct_order) * 0.6)))

    # 归一化均值用于颜色
    mean_norm = (mean_expr - mean_expr.min()) / (mean_expr.max() - mean_expr.min() + 1e-12)

    for i in range(len(ct_order)):
        for j in range(len(available_genes)):
            size = pct_expr[i, j] / 100 * 15
            if size > 0.5:
                ax.scatter(j, i, s=size * 20, c=[plt.cm.Reds(mean_norm[i, j])],
                           edgecolors='gray', linewidth=0.5, alpha=0.8)

    ax.set_xticks(range(len(available_genes)))
    ax.set_xticklabels(available_genes, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(ct_order)))
    ax.set_yticklabels(ct_order, fontsize=9)
    ax.set_title('Fig2D: Core Ferroptosis-Aging Gene Expression\n(Dot size = % expressing, Color = mean expression)')
    ax.set_xlim(-0.5, len(available_genes) - 0.5)
    ax.set_ylim(-0.5, len(ct_order) - 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig2D 保存: {save_path}")


def plot_pseudotime(adata, save_path: str):
    """Fig2C: 拟时序分析 (扩散伪时间)"""
    import scanpy as sc

    logger.info("  计算扩散伪时间...")
    # 使用 diffusion pseudotime
    try:
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata, n_branchings=0, n_dcs=10)
    except Exception as e:
        logger.warning(f"  拟时序计算失败: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # DPT 伪时间
    sc.pl.umap(adata, color='dpt_pseudotime', ax=axes[0], show=False,
               title='Diffusion Pseudotime', cmap='viridis', frameon=False)
    axes[0].set_xlabel('UMAP1')
    axes[0].set_ylabel('UMAP2')

    # 铁死亡评分 vs 伪时间
    sc.pl.umap(adata, color='ferroptosis_score', ax=axes[1], show=False,
               title='Ferroptosis Score', cmap='Reds', frameon=False)
    axes[1].set_xlabel('UMAP1')
    axes[1].set_ylabel('UMAP2')

    # IDSP Index vs 伪时间
    sc.pl.umap(adata, color='idsp_index', ax=axes[2], show=False,
               title='IDSP Index', cmap='RdYlBu_r', frameon=False)
    axes[2].set_xlabel('UMAP1')
    axes[2].set_ylabel('UMAP2')

    plt.suptitle('Fig2C: Pseudotime & Ferroptosis-Aging Trajectory', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig2C 保存: {save_path}")


def plot_condition_umap(adata, save_path: str):
    """补充: UMAP 按 condition 分面"""
    import scanpy as sc
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, cond in enumerate(['Sham', 'MCAO']):
        sub = adata[adata.obs['condition'] == cond]
        sc.pl.umap(sub, color='cell_type', ax=axes[idx], show=False,
                   title=f'{cond}', palette='tab10', frameon=False)
        axes[idx].set_xlabel('UMAP1')
        axes[idx].set_ylabel('UMAP2')

    plt.suptitle('UMAP by Condition', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Condition UMAP 保存: {save_path}")


def plot_condition_violin(adata, save_path: str):
    """补充: 各细胞类型按条件的小提琴图 (单面板)"""
    ct_order = sorted([ct for ct in adata.obs['cell_type'].unique() if ct != 'Unassigned'])
    n_ct = len(ct_order)

    fig, axes = plt.subplots(2, 2, figsize=(max(14, n_ct * 2), 10))
    scores = ['ferroptosis_score', 'senescence_score', 'consensus_senescence', 'idsp_index']
    titles = ['Ferroptosis Score', 'Senescence Score', 'Consensus Senescence (SenMayo+CellAge+FRIDMAN)', 'IDSP Index']

    for idx, (score, title) in enumerate(zip(scores, titles)):
        ax = axes[idx // 2, idx % 2]
        data_mcao = []
        data_sham = []
        labels = []

        for ct in ct_order:
            sub = adata[adata.obs['cell_type'] == ct]
            mcao = sub[sub.obs['condition'] == 'MCAO'].obs[score].dropna().values
            sham = sub[sub.obs['condition'] == 'Sham'].obs[score].dropna().values

            if len(mcao) >= 3 and len(sham) >= 3:
                data_mcao.append(mcao)
                data_sham.append(sham)
                labels.append(ct[:12])

        positions_mcao = [i * 3 - 0.5 for i in range(len(labels))]
        positions_sham = [i * 3 + 0.5 for i in range(len(labels))]

        vp1 = ax.violinplot(data_mcao, positions=positions_mcao, showmeans=True, showmedians=True, widths=0.8)
        vp2 = ax.violinplot(data_sham, positions=positions_sham, showmeans=True, showmedians=True, widths=0.8)

        for body in vp1['bodies']:
            body.set_facecolor('#E74C3C')
            body.set_alpha(0.7)
        for body in vp2['bodies']:
            body.set_facecolor('#3498DB')
            body.set_alpha(0.7)

        # 显著性标注
        for i, (m, s) in enumerate(zip(data_mcao, data_sham)):
            _, pval = stats.ttest_ind(m, s, equal_var=False)
            if pval < 0.05:
                y_max = max(np.max(m), np.max(s))
                ax.text(i * 3, y_max + 0.1, '*' if pval > 0.01 else '**' if pval > 0.001 else '***',
                        ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks([i * 3 for i in range(len(labels))])
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.axhline(y=0, color='gray', ls='--', lw=0.5)

        legend_elements = [
            mpatches.Patch(facecolor='#E74C3C', alpha=0.7, label='MCAO'),
            mpatches.Patch(facecolor='#3498DB', alpha=0.7, label='Sham'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

    plt.suptitle('Fig2B: Ferroptosis-Aging Scores per Cell Type (MCAO vs Sham)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig2B (violin) 保存: {save_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    import scanpy as sc
    sc.settings.figdir = str(FIGS_DIR)
    sc.settings.verbosity = 1

    logger.info("=" * 60)
    logger.info("模块二: 单细胞分辨率的脑I/R铁衰老细胞图谱")
    logger.info(f"数据: GSE174574 (小鼠 MCAO vs Sham)")
    logger.info(f"输出: {OUTPUT_DIR}")
    logger.info("=" * 60)

    # ============================================================
    # Step A: 加载 + QC + 预处理
    # ============================================================
    if not DATA_10X.exists():
        logger.error(f"数据目录不存在: {DATA_10X}")
        sys.exit(1)

    adata = load_and_preprocess(DATA_10X, SAMPLES)

    # 保存原始数据
    adata.write(OUTPUT_DIR / 'adata_preprocessed.h5ad', compression='gzip')
    logger.info(f"  预处理数据保存: adata_preprocessed.h5ad")

    # ============================================================
    # Step A2: 细胞类型标注
    # ============================================================
    adata = annotate_cell_types(adata, BRAIN_CELL_MARKERS)

    # ============================================================
    # Step B: 铁衰老双评分
    # ============================================================
    adata = compute_ferroptosis_aging_scores(adata)

    # ============================================================
    # Step C: 差异分析
    # ============================================================
    diff_df = differential_analysis(adata)

    # ============================================================
    # Step D: 拟时序分析
    # ============================================================
    logger.info("=" * 60)
    logger.info("Step D: 拟时序分析")

    try:
        sc.tl.diffmap(adata)
        sc.tl.dpt(adata, n_branchings=0, n_dcs=10)
        logger.info("  扩散伪时间计算完成")
    except Exception as e:
        logger.warning(f"  拟时序失败: {e}")

    # ============================================================
    # 保存完整数据
    # ============================================================
    adata.write(OUTPUT_DIR / 'adata_final.h5ad', compression='gzip')
    logger.info(f"  完整数据保存: adata_final.h5ad")

    # 导出评分数据
    scores_df = adata.obs[['sample', 'condition', 'cell_type',
                           'ferroptosis_score', 'senescence_score',
                           'consensus_senescence', 'idsp_index']].copy()
    scores_df.to_csv(OUTPUT_DIR / 'L2_cell_scores.csv')
    logger.info(f"  细胞评分保存: L2_cell_scores.csv")

    # 导出差异分析
    if not diff_df.empty:
        diff_df.to_csv(OUTPUT_DIR / 'L2_differential_analysis.csv', index=False)
        logger.info(f"  差异分析保存: L2_differential_analysis.csv")

    # 导出细胞类型统计
    ct_stats = adata.obs.groupby(['cell_type', 'condition']).agg(
        n_cells=('ferroptosis_score', 'count'),
        ferroptosis_mean=('ferroptosis_score', 'mean'),
        ferroptosis_std=('ferroptosis_score', 'std'),
        senescence_mean=('senescence_score', 'mean'),
        senescence_std=('senescence_score', 'std'),
        consensus_mean=('consensus_senescence', 'mean'),
        idsp_mean=('idsp_index', 'mean'),
        idsp_std=('idsp_index', 'std'),
    ).reset_index()
    ct_stats.to_csv(OUTPUT_DIR / 'L2_cell_type_summary.csv', index=False)
    logger.info(f"  细胞类型统计保存: L2_cell_type_summary.csv")

    # ============================================================
    # 可视化
    # ============================================================
    logger.info("=" * 60)
    logger.info("生成图表")

    # Fig2A: UMAP
    plot_umap_cell_types(adata, str(FIGS_DIR / 'Fig2A_umap_scores.png'))

    # Fig2B: 小提琴图
    plot_condition_violin(adata, str(FIGS_DIR / 'Fig2B_violin_scores.png'))

    # Fig2C: 拟时序
    plot_pseudotime(adata, str(FIGS_DIR / 'Fig2C_pseudotime.png'))

    # Fig2D: 气泡图
    plot_bubble_genes(adata, str(FIGS_DIR / 'Fig2D_bubble_genes.png'))

    # 补充: Condition UMAP
    plot_condition_umap(adata, str(FIGS_DIR / 'Fig2_condition_umap.png'))

    # ============================================================
    # 总结报告
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("模块二 分析完成!")
    logger.info(f"  细胞总数: {adata.n_obs}")
    logger.info(f"  基因总数: {adata.n_vars}")
    logger.info(f"  细胞类型: {list(adata.obs['cell_type'].unique())}")

    # 铁衰老评分统计
    for ct in sorted(adata.obs['cell_type'].unique()):
        sub = adata[adata.obs['cell_type'] == ct]
        mcao = sub[sub.obs['condition'] == 'MCAO']
        sham = sub[sub.obs['condition'] == 'Sham']
        if mcao.n_obs >= 3 and sham.n_obs >= 3:
            d_idsp = cohens_d(mcao.obs['idsp_index'].values, sham.obs['idsp_index'].values)
            logger.info(f"  {ct:20s}: IDSP d={d_idsp:+.3f}, "
                        f"MCAO={mcao.obs['idsp_index'].mean():+.3f}, "
                        f"Sham={sham.obs['idsp_index'].mean():+.3f}")

    logger.info(f"\n结果目录: {OUTPUT_DIR}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()