#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
模块二: 单细胞分辨率脑I/R铁衰老细胞图谱 v2.0
=============================================================================
数据: GSE174574 (3 Sham + 3 MCAO, 24h tMCAO, 小鼠皮层, 10X scRNA-seq)
流程:
  1. 数据加载 + QC + 归一化 (修复解压健壮性/ComBat)
  2. 铁衰老评分 (全基因集·含随机背景显著性)
  3. 聚类 + 细胞类型标注 (HVG)
  4. scTour 拟定序 + DPT 备用
  5. LIANA 跨细胞通讯预验证
  6. 差异表达 + 伪bulk + LR接口 (模块三)
  7. CytoTRACE导出 + R配套脚本
  8. 输出全部图表数据
=============================================================================
"""
import os, sys, gc, tarfile, shutil, warnings, logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats, sparse
import scanpy as sc
import anndata

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 路径配置
# ============================================================
IRON_DIR = Path(r"c:\Users\Jy-Mentor-7\Desktop\铁衰老")
DATA_DIR = Path(r"D:\反向网络药理学\L1 数据集\RNA-seq")
RAW_TAR = DATA_DIR / "GSE174574_RAW.tar"
EXTRACT_DIR = IRON_DIR / "GSE174574_extracted"
MODULE2_OUT = IRON_DIR / "module2_results"
R_PLOT_DIR = IRON_DIR / "module2_plot_data"
MODULE3_IFACE = IRON_DIR / "module3_interface"

for d in [EXTRACT_DIR, MODULE2_OUT, R_PLOT_DIR, MODULE3_IFACE]:
    d.mkdir(parents=True, exist_ok=True)

sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=100, facecolor='white')

# ============================================================
# 基因集
# ============================================================
FERROPTOSIS_GENES = [
    "ACSL4","GPX4","SLC7A11","TFRC","HMOX1","PTGS2","FTH1","FTL",
    "SLC3A2","FSP1","CHAC1","GCLC","GCLM","HSPB1","SAT1","ALOX15",
    "LPCAT3","NCOA4","STEAP3","SLC40A1","SLC39A14","CISD1","CISD2",
    "NOX4","DPP4","CDO1","MAP1LC3A","ATG5","ATG7","SQSTM1"
]

SENESCENCE_GENES = [
    "CDKN1A","CDKN2A","CDKN2B","LMNB1","IL6","IL1B","CXCL8",
    "CCL2","CCL8","MMP3","MMP9","MMP12","TIMP1","SERPINE1",
    "IGFBP3","IGFBP7","GADD45A","GDF15","TNF","TGFB1",
    "VEGFA","CXCL1","CXCL2","CXCL10","ICAM1","VCAM1",
    "CSF1","CSF2","MIF","HMGB1","B2M","H2AFX"
]

FRIDMAN_GENES = [
    "CDKN1A","CDKN2A","GLB1","LMNB1","TP53","RB1","HMGA2",
    "TERF2","TERT","SIRT1","FOXO3","NFKB1","RELA",
    "MAPK14","MAPK8","AKT1","MTOR","ULK1","ATG5","BECN1"
]

CELLAGE_GENES = [
    "CDKN1A","CDKN2A","LMNB1","GLB1","IL6","IL1B","TNF",
    "MMP3","SERPINE1","IGFBP3","GDF15","VEGFA","CXCL8"
]

CONSENSUS_SENESCENCE = list(set(SENESCENCE_GENES + FRIDMAN_GENES))

CELL_MARKERS = {
    'Neuron':        ['RBFOX3','MAP2','SYP','TUBB3','SNAP25','NEUN'],
    'Microglia':     ['CX3CR1','AIF1','TMEM119','CSF1R','P2RY12'],
    'Astrocyte':     ['GFAP','AQP4','ALDH1L1','SLC1A3','S100B'],
    'Oligodendrocyte': ['MBP','PLP1','MOG','OLIG2','CNP'],
    'Endothelial':   ['CLDN5','PECAM1','CDH5','FLT1','TEK'],
    'OPC':           ['PDGFRA','CSPG4','OLIG1','SOX10'],
    'Pericyte':      ['PDGFRB','RGS5','VTN','ANPEP'],
}

MICROGLIA_ACTIVATION = [
    'AIF1','CX3CR1','TREM2','CD68','ITGAM','FCGR3',
    'TLR2','TLR4','P2RY12','SPP1','APOE','CSF1R'
]

# ============================================================
# 工具函数: 基因名匹配
# ============================================================
def match_genes(gene_list: List[str], var_names: pd.Index) -> List[str]:
    """大小写不敏感的基因名匹配，返回存在于var_names中的基因名list"""
    vn_upper = {v.upper(): v for v in var_names}
    matched = []
    for g in gene_list:
        if g.upper() in vn_upper:
            matched.append(vn_upper[g.upper()])
    return matched

def match_genes_loose(gene_list: List[str], var_names: pd.Index) -> List[str]:
    """先精确大写匹配，失败则尝试to_lower匹配"""
    result = match_genes(gene_list, var_names)
    if len(result) == 0:
        vn_lower = {v.lower(): v for v in var_names}
        for g in gene_list:
            if g.lower() in vn_lower:
                result.append(vn_lower[g.lower()])
    return result

# ============================================================
# Step 1: 解压并加载 10X 数据 (健壮性升级)
# ============================================================
def extract_and_load() -> anndata.AnnData:
    logger.info("===== Step 1: 加载 10X scRNA-seq =====")
    
    # 解压 (仅当目录不存在或为空)
    need_extract = (
        not EXTRACT_DIR.exists() or 
        not list(EXTRACT_DIR.glob("*.gz"))
    )
    if need_extract:
        if EXTRACT_DIR.exists():
            shutil.rmtree(EXTRACT_DIR)
        EXTRACT_DIR.mkdir(parents=True)
        logger.info("  解压 RAW tar...")
        with tarfile.open(str(RAW_TAR), 'r') as tar:
            tar.extractall(path=str(EXTRACT_DIR))
        logger.info("  解压完成")

    # 识别GSM样本
    files = sorted(EXTRACT_DIR.glob("*.gz"))
    gsm_names = set()
    for f in files:
        parts = f.name.split('_')
        for i, p in enumerate(parts):
            if p.startswith('GSM') and len(p) >= 9:
                gsm_name = '_'.join(parts[:i+2]) if i+1 < len(parts) else p
                gsm_names.add(gsm_name)
                break

    if not gsm_names:
        # 回退: 检查子目录
        for sub in EXTRACT_DIR.iterdir():
            if sub.is_dir() and list(sub.glob("*.gz")):
                gsm_names.add(sub.name)

    logger.info(f"  检测到 {len(gsm_names)} 个 GSM 样本")

    adatas = []
    for gsm_name in sorted(gsm_names):
        gsm_dir = EXTRACT_DIR / gsm_name
        gsm_dir.mkdir(exist_ok=True)

        # 文件映射: 用 copy2 而非 rename 保留原文件
        for suffix in ['matrix.mtx.gz','barcodes.tsv.gz','genes.tsv.gz','features.tsv.gz']:
            src = EXTRACT_DIR / f"{gsm_name}_{suffix}"
            dst = gsm_dir / ('features.tsv.gz' if suffix == 'genes.tsv.gz' else suffix)
            if src.exists() and not dst.exists():
                try:
                    shutil.copy2(str(src), str(dst))
                except Exception:
                    # 硬链接/重命名回退
                    try:
                        src.rename(dst)
                    except Exception as e:
                        logger.warning(f"  文件复制失败 {suffix}: {e}")

        # 验证必要文件
        mtx_file = gsm_dir / "matrix.mtx.gz"
        bc_file = gsm_dir / "barcodes.tsv.gz"
        ft_file = gsm_dir / "features.tsv.gz"

        missing = []
        for fn in [mtx_file, bc_file, ft_file]:
            if not fn.exists():
                missing.append(fn.name)
        if missing:
            logger.warning(f"  [{gsm_name}] 缺少文件: {missing}, 跳过")
            continue

        try:
            from scipy.io import mmread
            mtx = mmread(str(mtx_file))
            barcodes = pd.read_csv(str(bc_file), header=None, sep='\t')[0].tolist()
            features = pd.read_csv(str(ft_file), header=None, sep='\t')
            gene_symbols = features.iloc[:, 1].values
            gene_ids = features.iloc[:, 0].values

            adata = anndata.AnnData(
                mtx.tocsr().T,
                obs=pd.DataFrame(index=barcodes),
                var=pd.DataFrame(index=gene_ids)
            )
            adata.var['gene_symbols'] = gene_symbols
            adata.var_names = gene_symbols
            adata.var_names_make_unique()

            new_names = [f"{gsm_name}_{n.split('-')[-1]}" for n in adata.obs_names]
            adata.obs_names = new_names
            adata.obs_names_make_unique()
            adata.obs['sample'] = gsm_name
            condition = 'MCAO' if 'MCAO' in gsm_name.upper() else 'Sham'
            adata.obs['condition'] = condition
            adatas.append(adata)
            logger.info(f"  [{gsm_name}] {adata.n_obs} cells x {adata.n_vars} genes, {condition}")
        except Exception as e:
            logger.error(f"  [{gsm_name}] 加载失败: {e}")

    if not adatas:
        raise RuntimeError("无有效样本")

    adata = adatas[0].concatenate(adatas[1:], batch_key='batch', join='inner')
    logger.info(f"  合并: {adata.n_obs} cells x {adata.n_vars} genes")
    return adata

# ============================================================
# Step 2: QC + 归一化 (ComBat 健壮)
# ============================================================
def qc_and_normalize(adata: anndata.AnnData) -> anndata.AnnData:
    logger.info("\n===== Step 2: QC + 归一化 =====")
    adata.var_names = adata.var_names.astype(str).str.upper()
    adata = adata[:, ~adata.var_names.duplicated()].copy()

    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.var['ribo'] = adata.var_names.str.startswith(('RPS','RPL'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo'], inplace=True)

    logger.info(f"  过滤前: {adata.n_obs} cells")
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata = adata[adata.obs.pct_counts_mt < 20, :].copy()
    adata = adata[adata.obs.n_genes_by_counts < 5000, :].copy()
    logger.info(f"  过滤后: {adata.n_obs} cells, {adata.n_vars} genes")

    # 保存原始计数 (CytoTRACE用)
    adata.raw = adata.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor='seurat_v3',
                                batch_key='batch' if 'batch' in adata.obs.columns else None)
    logger.info(f"  高变基因: {adata.var.highly_variable.sum()}")
    return adata

def run_combat_safe(adata_hvg: anndata.AnnData) -> None:
    """安全的 ComBat 批次校正"""
    try:
        # 确保 batch 列存在且为字符串
        if 'batch' not in adata_hvg.obs.columns:
            logger.warning("  batch 列缺失, 跳过 ComBat")
            return
        adata_hvg.obs['batch'] = adata_hvg.obs['batch'].astype(str)
        sc.pp.combat(adata_hvg, key='batch')
        logger.info("  ComBat 完成")
    except Exception as e:
        logger.warning(f"  ComBat 失败 ({e}), 跳过")

# ============================================================
# Step 3: 铁衰老评分 (全基因集 + 随机背景显著性)
# ============================================================
def compute_iron_aging_scores(adata: anndata.AnnData) -> anndata.AnnData:
    logger.info("\n===== Step 3: 铁衰老评分 + 代谢 =====")

    def score_genes_safe(adata_obj, gene_list, score_name):
        """安全评分: 匹配存在的基因, 返回score向量"""
        valid = match_genes_loose(gene_list, adata_obj.var_names)
        if len(valid) >= 3:
            sc.tl.score_genes(adata_obj, gene_list=valid, score_name=score_name)
            return len(valid), adata_obj.obs[score_name].values.copy()
        return 0, None

    # --- 铁死亡评分 ---
    n_ferr, scores_ferr = score_genes_safe(adata, FERROPTOSIS_GENES, 'ferroptosis_score')
    if n_ferr:
        logger.info(f"  铁死亡评分: {n_ferr}/{len(FERROPTOSIS_GENES)} 基因")

    # --- 衰老评分 ---
    n_sene, scores_sene = score_genes_safe(adata, SENESCENCE_GENES, 'senescence_score')
    if n_sene:
        logger.info(f"  衰老评分: {n_sene}/{len(SENESCENCE_GENES)} 基因")

    # --- 共识衰老: 分别打分, 保留各组列 ---
    consensus_cfg = {
        'sene_senmayo':  SENESCENCE_GENES,
        'sene_cellage':  CELLAGE_GENES,
        'sene_fridman':  FRIDMAN_GENES,
    }
    sub_scores = {}
    for key, gene_list in consensus_cfg.items():
        n_valid, vec = score_genes_safe(adata, gene_list, key)
        if vec is not None:
            sub_scores[key] = vec
            adata.obs[key] = vec  # 保留各组评分列
            logger.info(f"  {key}: {n_valid} 基因")

    if len(sub_scores) >= 2:
        score_matrix = np.column_stack(list(sub_scores.values()))
        adata.obs['consensus_senescence'] = np.median(score_matrix, axis=1)
        logger.info(f"  共识衰老评分: {len(sub_scores)} 基因集中位数")

    # --- ISP Index ---
    if 'ferroptosis_score' in adata.obs and 'senescence_score' in adata.obs:
        zf = stats.zscore(adata.obs['ferroptosis_score'].values, nan_policy='omit')
        zs = stats.zscore(adata.obs['senescence_score'].values, nan_policy='omit')
        adata.obs['isp_index'] = 2 * np.minimum(zf, zs)

    # ============ 铁死亡代谢通路评分 ============
    logger.info("  代谢通路评分...")
    METABOLISM_SETS = {
        'glutathione_metabolism': ['GCLC', 'GCLM', 'GSS', 'GSR', 'GPX4', 'SLC7A11'],
        'fatty_acid_elongation':  ['ELOVL6', 'ELOVL5', 'FADS1', 'FADS2', 'SCD1', 'ACSL4'],
        'iron_metabolism':        ['TFRC', 'FTH1', 'FTL', 'SLC40A1', 'SLC39A14', 'HAMP'],
    }
    for mname, mgenes in METABOLISM_SETS.items():
        valid_m = match_genes_loose(mgenes, adata.var_names)
        if len(valid_m) >= 3:
            sc.tl.score_genes(adata, gene_list=valid_m, score_name=mname)
            logger.info(f"    {mname}: {len(valid_m)}/{len(mgenes)} 基因")

    # 代谢-铁衰老相关性
    score_names = ['ferroptosis_score','senescence_score','isp_index']
    met_names = list(METABOLISM_SETS.keys())
    corr_rows = []
    for sn in score_names:
        if sn not in adata.obs:
            continue
        for mn in met_names:
            if mn not in adata.obs:
                continue
            r, p = stats.pearsonr(
                adata.obs[sn].fillna(0).values,
                adata.obs[mn].fillna(0).values
            )
            corr_rows.append({'score': sn, 'metabolism': mn, 'pearson_r': round(r, 4), 'p_value': round(p, 6)})
    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(MODULE2_OUT / "metabolism_correlations.csv", index=False)
        logger.info(f"    代谢相关性: {MODULE2_OUT / 'metabolism_correlations.csv'}")

    return adata

# ============================================================
# Step 4: 聚类 + 细胞标注
# ============================================================
def cluster_and_annotate(adata: anndata.AnnData) -> anndata.AnnData:
    logger.info("\n===== Step 4: 聚类 + 细胞标注 =====")
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=30)
    run_combat_safe(adata_hvg)
    sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata_hvg)
    sc.tl.leiden(adata_hvg, resolution=0.6)

    # 标记物打分
    score_names = []
    for ct, markers in CELL_MARKERS.items():
        valid = match_genes(markers, adata_hvg.var_names)
        if len(valid) >= 2:
            sc.tl.score_genes(adata_hvg, gene_list=valid, score_name=f'score_{ct}')
            score_names.append(f'score_{ct}')

    if score_names:
        cluster_scores = adata_hvg.obs.groupby('leiden')[score_names].mean()
        cluster_to_ct = cluster_scores.idxmax(axis=1).str.replace('score_', '')
        adata_hvg.obs['cell_type'] = adata_hvg.obs['leiden'].map(cluster_to_ct)
    else:
        adata_hvg.obs['cell_type'] = adata_hvg.obs['leiden'].apply(lambda x: f'Cluster_{x}')

    logger.info(f"  细胞类型:\n{adata_hvg.obs['cell_type'].value_counts().to_string()}")
    return adata_hvg

# ============================================================
# Step 5: scTour / DPT 拟时序
# ============================================================
def run_sctour(adata: anndata.AnnData) -> anndata.AnnData:
    logger.info("\n===== Step 5: 拟时序 =====")
    try:
        import sctour as sct

        if adata.raw is not None:
            adata_tour = adata.raw.to_adata().copy()
        else:
            adata_tour = adata.copy()
            adata_tour.X = np.expm1(adata.X.toarray() if sparse.issparse(adata.X) else adata.X)

        adata_tour.obs = adata.obs.copy()
        sc.pp.calculate_qc_metrics(adata_tour, percent_top=None, log1p=False, inplace=True)
        sc.pp.filter_genes(adata_tour, min_cells=20)
        sc.pp.highly_variable_genes(adata_tour, flavor='seurat_v3', n_top_genes=2000, subset=True)

        if sparse.issparse(adata_tour.X):
            adata_tour.X = np.log1p(adata_tour.X.toarray())
        else:
            adata_tour.X = np.log1p(adata_tour.X)

        n_cells = adata_tour.n_obs
        nepoch = 100 if n_cells > 30000 else 200
        percent = 0.1 if n_cells > 30000 else 0.2

        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
        logger.info(f"    GPU: {use_gpu}")

        tnode = sct.train.Trainer(
            adata_tour, loss_mode='mse',
            percent=percent, nepoch=nepoch,
            use_gpu=use_gpu, random_state=42,
            batch_size=2048 if n_cells > 30000 else 1024
        )
        tnode.train()
        adata.obs['sctour_ptime'] = tnode.get_time()
        logger.info("  scTour 完成")
        return adata

    except Exception as e:
        logger.warning(f"  scTour 失败: {e}, 使用 DPT")
        gc.collect()
        return dpt_fallback(adata)

def dpt_fallback(adata: anndata.AnnData) -> anndata.AnnData:
    logger.info("  使用 DPT")
    if 'X_umap' not in adata.obsm:
        sc.tl.umap(adata)
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)
    return adata

# ============================================================
# Step 6.5: 随机背景显著性 (后置于聚类)
# ============================================================
def background_significance(adata: anndata.AnnData, n_permutations: int = 500):
    """对铁死亡/衰老/共识评分做随机基因集置换检验"""
    logger.info("\n===== Step 6.5: 随机背景显著性 =====")
    if 'cell_type' not in adata.obs:
        logger.warning("  无 cell_type, 跳过 background_significance")
        return
    assert 'cell_type' in adata.obs.columns, \
        "background_significance requires 'cell_type' in adata.obs"

    all_genes = list(adata.var_names)
    cell_types = sorted(adata.obs['cell_type'].unique())
    score_sets = {
        'ferroptosis':  match_genes_loose(FERROPTOSIS_GENES, adata.var_names),
        'senescence':   match_genes_loose(SENESCENCE_GENES, adata.var_names),
        'consensus':    match_genes_loose(CONSENSUS_SENESCENCE, adata.var_names),
    }
    score_columns = {
        'ferroptosis':  'ferroptosis_score',
        'senescence':   'senescence_score',
        'consensus':    'consensus_senescence',
    }

    perm_results = []
    bar_data = []
    rng = np.random.RandomState(42)

    for set_name, gene_set in score_sets.items():
        if len(gene_set) < 5 or score_columns[set_name] not in adata.obs:
            continue

        real_scores = adata.obs.groupby('cell_type')[score_columns[set_name]].mean()

        null_dist = {ct: [] for ct in cell_types}
        for i in range(n_permutations):
            perm_genes = rng.choice(all_genes, size=len(gene_set), replace=False)
            sc.tl.score_genes(adata, gene_list=list(perm_genes), score_name='_perm_tmp')
            for ct in cell_types:
                null_dist[ct].append(
                    float(adata.obs.loc[adata.obs['cell_type'] == ct, '_perm_tmp'].mean())
                )
            del adata.obs['_perm_tmp']
            if (i + 1) % 50 == 0:
                logger.info(f"    {set_name}: {i+1}/{n_permutations} done")

        for ct in cell_types:
            observed = float(real_scores[ct])
            null_vec = np.array(null_dist[ct])
            p_val = (np.sum(null_vec >= observed) + 1) / (n_permutations + 1)
            null_mean = float(np.mean(null_vec))
            null_std = float(np.std(null_vec))
            perm_results.append({
                'gene_set': set_name,
                'cell_type': ct,
                'observed_mean': round(observed, 4),
                'null_mean': round(null_mean, 4),
                'null_std': round(null_std, 4),
                'empirical_p': round(p_val, 6),
                'significant': bool(p_val < 0.05),
            })
            bar_data.append({
                'cell_type': ct,
                'score_type': set_name,
                'observed': round(observed, 4),
                'expected': round(null_mean, 4),
                'p_value': round(p_val, 6),
                'significant': '+' if p_val < 0.05 else '',
            })
        n_sig = sum(1 for r in perm_results if r['gene_set'] == set_name and r['significant'])
        logger.info(f"    {set_name}: {n_sig}/{len(cell_types)} CT 显著")

    if perm_results:
        pd.DataFrame(perm_results).to_csv(MODULE2_OUT / "iron_aging_significance.csv", index=False)
        pd.DataFrame(bar_data).to_csv(R_PLOT_DIR / "significance_bar.csv", index=False)
        logger.info(f"    背景显著性: {MODULE2_OUT / 'iron_aging_significance.csv'}")
        logger.info(f"    柱状图数据: {R_PLOT_DIR / 'significance_bar.csv'}")
# ============================================================
def run_liana_lr_analysis(adata_full: anndata.AnnData, adata_hvg: anndata.AnnData):
    """LIANA 配体-受体分析, 筛选 MCAO vs Sham 差异通讯对"""
    logger.info("\n===== Step 6: LIANA 跨细胞通讯 =====")
    try:
        import liana as li
    except ImportError:
        logger.warning("  LIANA 未安装, 跳过 (pip install liana)")
        return

    try:
        # 确保 cell_type 列可用
        if 'cell_type' not in adata_hvg.obs:
            logger.warning("  无 cell_type 列, 跳过 LIANA")
            return
        adata_hvg.obs['cell_type'] = adata_hvg.obs['cell_type'].astype(str)

        # --- 从 raw 层提取原始计数 (LIANA 依赖原始计数估计配-受体表达) ---
        if adata_full.raw is None:
            logger.warning("  adata_full 无 raw 层, 跳过 LIANA")
            return

        hvgs = list(adata_hvg.var_names)
        raw_var_names = list(adata_full.raw.var_names)
        hvgs_in_raw = [g for g in hvgs if g in raw_var_names]
        if len(hvgs_in_raw) < 50:
            logger.warning("  raw 层 HVG 基因太少, 跳过 LIANA")
            return

        raw_gene_idx = [raw_var_names.index(g) for g in hvgs_in_raw]
        raw_counts_hvg = adata_full.raw.X[:, raw_gene_idx].tocsr()

        # 构���带原始计数的 AnnData
        adata_raw = anndata.AnnData(
            X=raw_counts_hvg,
            obs=adata_full.obs.copy(),
            var=pd.DataFrame(index=hvgs_in_raw)
        )
        adata_raw.obs['cell_type'] = adata_hvg.obs['cell_type']

        # MCAO / Sham 子集 (原始计数)
        adata_mcao = adata_raw[adata_raw.obs['condition'] == 'MCAO', :].copy()
        adata_sham = adata_raw[adata_raw.obs['condition'] == 'Sham', :].copy()

        if adata_mcao.n_obs < 10 or adata_sham.n_obs < 10:
            logger.warning("  MCAO/Sham 细胞数不足")
            return

        # LIANA 共识方法 (仅一次 inplace=False, 不修改输入)
        lr_mcao = li.mt.rank_aggregate(
            adata_mcao, groupby='cell_type',
            resource_name='consensus',
            expr_prop=0.1, verbose=False,
            use_raw=False, return_all_lrs=True, inplace=False
        )

        lr_sham = li.mt.rank_aggregate(
            adata_sham, groupby='cell_type',
            resource_name='consensus',
            expr_prop=0.1, verbose=False,
            use_raw=False, return_all_lrs=True, inplace=False
        )

        # 差异通讯: MCAO vs Sham
        merged = lr_mcao.merge(
            lr_sham,
            on=['source','target','ligand_complex','receptor_complex'],
            suffixes=('_mcao','_sham')
        )
        merged['magnitude_rank_diff'] = merged['magnitude_rank_mcao'] - merged['magnitude_rank_sham']
        merged = merged.sort_values('magnitude_rank_diff')
        logger.info(f"  LR 差异对: {len(merged)}")

        # 保存完整结果
        merged.to_csv(MODULE2_OUT / "LIANA_all_LR_pairs.csv", index=False)

        # 提取铁衰老相关 LR (全大写集合, 兼容 LIANA 命名)
        iron_aging_upper = set(
            g.upper() for g in match_genes_loose(
                FERROPTOSIS_GENES + SENESCENCE_GENES, adata_full.var_names
            )
        )
        def is_iron_aging(row):
            ligands = str(row['ligand_complex']).upper()
            receptors = str(row['receptor_complex']).upper()
            # 拆分复合体名 (如 "HMOX1_TFRC") 后逐个判断
            lr_all = ligands.replace('_', ' ') + ' ' + receptors.replace('_', ' ')
            for token in lr_all.split():
                if token.strip() in iron_aging_upper:
                    return True
            return False

        merged['iron_aging_related'] = merged.apply(is_iron_aging, axis=1)
        ia_lr = merged[merged['iron_aging_related']].copy()

        if len(ia_lr) > 0:
            ia_lr.to_csv(MODULE3_IFACE / "iron_aging_LR_pairs.csv", index=False)
            logger.info(f"  铁衰老相关 LR: {len(ia_lr)} 对")

            lr_list = ia_lr[['source','target','ligand_complex','receptor_complex',
                             'magnitude_rank_mcao','magnitude_rank_sham','magnitude_rank_diff']].copy()
            lr_list.to_csv(MODULE3_IFACE / "lr_edges.csv", index=False)
            logger.info(f"  模块三 LR 候选边: {MODULE3_IFACE / 'lr_edges.csv'}")
        else:
            top50 = merged.head(50)[['source','target','ligand_complex','receptor_complex',
                                     'magnitude_rank_mcao','magnitude_rank_sham','magnitude_rank_diff']].copy()
            top50.to_csv(MODULE3_IFACE / "lr_edges.csv", index=False)
            logger.info(f"  无铁衰老 LR, 输出 top50")

        # LIANA 气泡图 & 网络数据
        lr_bubble = merged[['source','target','ligand_complex','receptor_complex',
                            'magnitude_rank_mcao','magnitude_rank_sham','magnitude_rank_diff']].head(200).copy()
        lr_bubble.to_csv(R_PLOT_DIR / "liana_bubble.csv", index=False)
        lr_net = merged[['source','target','ligand_complex','receptor_complex','magnitude_rank_diff']].copy()
        lr_net.columns = ['source','target','ligand','receptor','weight']
        lr_net['weight'] = -lr_net['weight']  # 负值越大 = MCAO 通讯更强
        lr_net.to_csv(R_PLOT_DIR / "liana_network.csv", index=False)
        logger.info(f"  LIANA 气泡/网络数据已保存")

    except Exception as e:
        logger.warning(f"  LIANA 失败: {e}")

# ============================================================
# Step 7: 差异表达 + 伪bulk + 接口数据
# ============================================================
def differential_and_interfaces(adata_full: anndata.AnnData, adata: anndata.AnnData):
    logger.info("\n===== Step 7: DEG + 伪bulk + 接口 =====")

    # --- ISP 高 vs 低 DEG ---
    isp = adata.obs.get('isp_index', pd.Series(dtype=float)).dropna()
    if len(isp) > 0:
        q75, q25 = isp.quantile(0.75), isp.quantile(0.25)
        adata.obs['isp_group'] = 'Mid'
        adata.obs.loc[isp[isp >= q75].index, 'isp_group'] = 'High'
        adata.obs.loc[isp[isp <= q25].index, 'isp_group'] = 'Low'

    adata_hvg = adata[:, adata.var.highly_variable].copy() if 'highly_variable' in adata.var else adata.copy()

    # MCAO vs Sham DEG (per cell type)
    deg_ct_list = []
    for ct in adata.obs['cell_type'].unique():
        ct_adata = adata_hvg[adata_hvg.obs['cell_type'] == ct, :].copy()
        if ct_adata.n_obs < 30:
            continue
        if ct_adata.obs['condition'].nunique() < 2:
            continue
        try:
            sc.tl.rank_genes_groups(ct_adata, 'condition', groups=['MCAO'],
                                    reference='Sham', method='wilcoxon')
            deg = sc.get.rank_genes_groups_df(ct_adata, group='MCAO')
            deg['cell_type'] = ct
            deg_ct_list.append(deg)
        except Exception as e:
            logger.warning(f"  [{ct}] DEG 失败: {e}")

    if deg_ct_list:
        deg_all = pd.concat(deg_ct_list, ignore_index=True)
        deg_all.to_csv(MODULE2_OUT / "cell_type_DEG_MCAO_vs_Sham.csv", index=False)
        # 筛选显著
        deg_sig = deg_all[(deg_all['logfoldchanges'].abs() > 0.25) & (deg_all['pvals_adj'] < 0.05)]
        deg_sig.to_csv(MODULE3_IFACE / "cell_type_DEG_significant.csv", index=False)
        logger.info(f"  MCAO vs Sham DEG: {len(deg_all)} 基因, 显著 {len(deg_sig)}")
    else:
        logger.warning("  DEG 结果为空")

    # --- 伪bulk表达矩阵 (per cell_type × condition) ---
    pseudobulk_matrices = {}
    for ct in adata_full.obs['cell_type'].unique():
        for cond in ['Sham', 'MCAO']:
            mask = (adata_full.obs['cell_type'] == ct) & (adata_full.obs['condition'] == cond)
            n_cells = mask.sum()
            if n_cells < 5:
                continue
            if sparse.issparse(adata_full[mask].X):
                pb = adata_full[mask].X.mean(axis=0).A1
            else:
                pb = adata_full[mask].X.mean(axis=0)
            key = f"{ct}_{cond}"
            pseudobulk_matrices[key] = pb

    if pseudobulk_matrices:
        pb_df = pd.DataFrame(pseudobulk_matrices, index=adata_full.var_names)
        pb_df.to_csv(MODULE3_IFACE / "pseudobulk_celltype_condition.csv")
        logger.info(f"  伪bulk: {pb_df.shape[1]} 组 x {pb_df.shape[0]} 基因")
    else:
        logger.warning("  伪bulk 数据不足")

    # --- Raw counts 伪bulk (模块三备用) ---
    if adata_full.raw is not None:
        raw_pb_matrices = {}
        for ct in adata_full.obs['cell_type'].unique():
            for cond in ['Sham', 'MCAO']:
                mask = (adata_full.obs['cell_type'] == ct) & (adata_full.obs['condition'] == cond)
                if mask.sum() < 5:
                    continue
                raw_expr = adata_full.raw[mask].X
                if sparse.issparse(raw_expr):
                    raw_pb = raw_expr.mean(axis=0).A1
                else:
                    raw_pb = raw_expr.mean(axis=0)
                raw_pb_matrices[f"{ct}_{cond}"] = raw_pb
        if raw_pb_matrices:
            raw_pb_df = pd.DataFrame(raw_pb_matrices, index=adata_full.raw.var_names)
            raw_pb_df.to_csv(MODULE3_IFACE / "pseudobulk_raw_counts.csv")
            logger.info(f"  Raw伪bulk: {raw_pb_df.shape[1]} 组 x {raw_pb_df.shape[0]} 基因")

    # --- 小胶质激活 × 铁死亡特征 ---
    if 'Microglia' in adata.obs['cell_type'].values:
        mg = adata_hvg[adata_hvg.obs['cell_type'] == 'Microglia', :].copy()
        act_genes = match_genes(MICROGLIA_ACTIVATION, mg.var_names)
        if act_genes:
            sc.tl.score_genes(mg, gene_list=act_genes, score_name='mg_activation')
            mg_data = mg.obs[['ferroptosis_score','senescence_score','mg_activation',
                               'consensus_senescence','isp_index','condition']].copy()
            mg_data.to_csv(R_PLOT_DIR / "microglia_activation.csv")
            logger.info(f"  小胶质激活数据: {len(mg_data)} 细胞")

    # --- 代谢-铁衰老联合散点图数据 ---
    met_names = ['glutathione_metabolism', 'fatty_acid_elongation', 'iron_metabolism']
    if all(c in adata_full.obs for c in ['isp_index'] + met_names):
        scat_cols = ['isp_index', 'cell_type', 'condition'] + [c for c in met_names if c in adata_full.obs]
        scat_data = adata_full.obs[scat_cols].copy()
        scat_data.to_csv(R_PLOT_DIR / "metabolism_scatter.csv", index=False)
        logger.info(f"  代谢散点数据: {R_PLOT_DIR / 'metabolism_scatter.csv'}")

# ============================================================
# Step 8: CytoTRACE 数据导出 + R 配套
# ============================================================
def export_cytotrace_data(adata_full: anndata.AnnData):
    logger.info("\n===== Step 8: CytoTRACE 导出 =====")
    if adata_full.raw is None:
        logger.warning("  无 raw 层, 跳过")
        return

    raw = adata_full.raw.X
    if not sparse.issparse(raw):
        raw = sparse.csr_matrix(raw)

    # 保存为 .mtx (稀疏, ~1GB而非9GB dense; mmwrite 需相对路径)
    from scipy.io import mmwrite
    import os as _os
    _cwd = _os.getcwd()
    try:
        _os.chdir(str(MODULE3_IFACE))
        mmwrite("raw_counts_for_cytotrace.mtx", raw)
    finally:
        _os.chdir(_cwd)
    logger.info(f"  Raw counts MTX: {MODULE3_IFACE / 'raw_counts_for_cytotrace.mtx'}")

    # 保存基因名和细胞名
    pd.Series(adata_full.raw.var_names).to_csv(
        MODULE3_IFACE / "raw_genes.csv", index=False, header=False
    )
    pd.Series(adata_full.raw.obs_names).to_csv(
        MODULE3_IFACE / "raw_cells.csv", index=False, header=False
    )
    logger.info(f"  基因/细胞名: {MODULE3_IFACE}")

    # R 配套脚本
    r_script = MODULE3_IFACE / "cytotrace_analysis.R"
    with open(r_script, 'w') as f:
        f.write("""suppressPackageStartupMessages({
  library(CytoTRACE)
  library(Matrix)
  library(readr)
})
cat("Loading sparse matrix...\\n")
counts <- readMM("raw_counts_for_cytotrace.mtx")
genes <- read_csv("raw_genes.csv", col_names = FALSE)[[1]]
cells <- read_csv("raw_cells.csv", col_names = FALSE)[[1]]
rownames(counts) <- genes
colnames(counts) <- cells
cat(sprintf("Matrix: %d genes x %d cells\\n", nrow(counts), ncol(counts)))
results <- CytoTRACE(counts, ncores = 4, subsamplesize = 1000)
write.csv(results$CytoTRACE, "cytotrace_scores.csv")
cat("CytoTRACE done.\\n")
""")
    logger.info(f"  R 脚本: {r_script}")

# ============================================================
# Step 9: 保存结果
# ============================================================
def save_results(adata: anndata.AnnData, adata_full: anndata.AnnData = None):
    logger.info("\n===== Step 9: 保存结果 =====")
    if adata_full is None:
        adata_full = adata

    # 主 AnnData (HVG)
    adata.write(MODULE2_OUT / "GSE174574_processed.h5ad", compression='gzip')

    # 全基因 AnnData (保留 raw + scores)
    adata_full.write(MODULE2_OUT / "GSE174574_full.h5ad", compression='gzip')

    # --- 细胞元数据 ---
    obs_cols = ['sample','condition','leiden','cell_type']
    score_cols = ['ferroptosis_score','senescence_score','consensus_senescence','isp_index',
                  'sene_senmayo','sene_cellage','sene_fridman',
                  'glutathione_metabolism','fatty_acid_elongation','iron_metabolism']
    time_cols = ['sctour_ptime'] if 'sctour_ptime' in adata.obs else \
                ['dpt_pseudotime'] if 'dpt_pseudotime' in adata.obs else []
    isp_cols = ['isp_group'] if 'isp_group' in adata.obs else []

    cols_to_save = obs_cols + [c for c in score_cols + time_cols + isp_cols if c in adata.obs]
    cell_meta = adata.obs[cols_to_save].copy()

    if 'X_umap' in adata.obsm:
        cell_meta['UMAP1'] = adata.obsm['X_umap'][:, 0]
        cell_meta['UMAP2'] = adata.obsm['X_umap'][:, 1]
    cell_meta.to_csv(R_PLOT_DIR / "cell_metadata.csv")
    logger.info(f"  细胞元数据: {R_PLOT_DIR / 'cell_metadata.csv'}")

    # --- 细胞类型评分汇总 ---
    aggs = {c: ['mean','std'] for c in score_cols if c in adata.obs}
    if aggs:
        summary_df = adata.obs.groupby('cell_type').agg(aggs).round(4)
        summary_df.to_csv(R_PLOT_DIR / "cell_type_score_summary.csv")
        logger.info(f"  评分汇总: {R_PLOT_DIR / 'cell_type_score_summary.csv'}")

    # --- 气泡图数据 (全基因集) ---
    core_genes = ["ACSL4","GPX4","PTGS2","HMOX1","TFRC","CDKN1A","IL6","CDKN2A"]
    available = match_genes_loose(core_genes, adata_full.var_names)
    if available:
        bubble_data = []
        for ct in adata_full.obs['cell_type'].unique():
            ct_mask = adata_full.obs['cell_type'] == ct
            for g in available:
                expr = adata_full[ct_mask, g].X
                if sparse.issparse(expr):
                    expr = expr.toarray().ravel()
                bubble_data.append({
                    'cell_type': ct,
                    'gene': g,
                    'mean_expr': float(np.mean(expr)),
                    'pct_expr': float(np.sum(expr > 0) / len(expr) * 100)
                })
        pd.DataFrame(bubble_data).to_csv(R_PLOT_DIR / "bubble_data.csv", index=False)
        logger.info(f"  气泡图: {len(available)} 基因")

    # --- 细胞类型 × 条件 ---
    ct_cond = pd.crosstab(adata.obs['cell_type'], adata.obs['condition'])
    ct_cond.to_csv(R_PLOT_DIR / "cell_type_condition_counts.csv")

    # --- 共识评分每亚组 ---
    consensus_detail_cols = ['sene_senmayo','sene_cellage','sene_fridman']
    if all(c in adata.obs for c in consensus_detail_cols):
        consensus_detail = adata.obs.groupby('cell_type')[consensus_detail_cols].mean().round(4)
        consensus_detail.to_csv(R_PLOT_DIR / "consensus_senescence_detail.csv")
        logger.info("  共识评分详情已保存")

    logger.info("  保存完成")

# ============================================================
# 主流程
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("模块二 v2.0: 单细胞铁衰老细胞图谱")
    logger.info("=" * 60)

    # Step 1-2
    adata = extract_and_load()
    adata = qc_and_normalize(adata)

    # Step 3: Scoring FIRST (全基因集, 含代谢评分)
    adata_full = compute_iron_aging_scores(adata)

    # Step 4: Cluster on HVGs + transfer cell_type back
    adata = cluster_and_annotate(adata_full)
    
    # Transfer scores & cell_type between both
    for col in ['ferroptosis_score','senescence_score','consensus_senescence',
                'isp_index','sene_senmayo','sene_cellage','sene_fridman',
                'cell_type','glutathione_metabolism','fatty_acid_elongation','iron_metabolism']:
        if col in adata.obs and col not in adata_full.obs:
            adata_full.obs[col] = adata.obs[col]
        elif col in adata_full.obs and col not in adata.obs:
            adata.obs[col] = adata_full.obs[col]

    # Step 4.5: 随机背景显著性 (需要 cell_type)
    background_significance(adata_full, n_permutations=500)

    # Step 5: Pseudotime
    adata = run_sctour(adata)

    # Step 6: LIANA
    run_liana_lr_analysis(adata_full, adata)

    # Step 7: DEG + pseudo-bulk + interfaces
    differential_and_interfaces(adata_full, adata)

    # Step 8: CytoTRACE export
    export_cytotrace_data(adata_full)

    # Step 9: Save everything
    save_results(adata, adata_full)

    logger.info("\n===== 模块二 v2.0 完成 =====")
    logger.info(f"输出: {MODULE2_OUT}")
    logger.info(f"R绘图: {R_PLOT_DIR}")
    logger.info(f"模块三: {MODULE3_IFACE}")

if __name__ == '__main__':
    main()
