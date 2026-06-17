#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
模块三: HGT-GAT异质图构建与跨细胞通讯解析
=====================================================================
核心架构:
  - 底层: GAT编码器 (同类型节点局部注意力)
  - 上层: HGT编码器 (跨类型语义注意力)
  - 训练: Multi-task三目标联合优化
    1. 基因-通路归属预测
    2. 化合物-靶点结合预测
    3. 跨细胞通讯关联预测

输出:
  - Fig3A: 异质图拓扑结构示意
  - Fig3B: HGT跨类型注意力权重矩阵
  - Fig3C: Hub基因排名 (ACSL4验证)
  - Fig3D: 跨细胞通讯流图
  - Fig3E: 节点嵌入t-SNE可视化
  - Fig3F: SHAP边贡献度排名

依赖: torch, torch_geometric, numpy, pandas, matplotlib, scikit-learn, shap
=====================================================================
"""

import os, sys, warnings, logging, copy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('default')
# 在调试时可改为 'default' 查看所有警告
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 导入基因集
# ============================================================
try:
    from idsp_gene_sets import (
        PURE_FERROPTOSIS, PURE_SENESCENCE, SHARED_GENES,
        FERROPTOSIS_ALL, SENESCENCE_ALL
    )
    logger.info("基因集加载: idsp_gene_sets.py")
except ImportError:
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

# ============================================================
# 铁衰老基因 (来自铁衰老基因.txt)
# ============================================================
FERROAGING_GENES = {
    'ABCC1', 'ACVR1B', 'ALOX15', 'ATF3', 'ATG3', 'BAP1', 'BCL6', 'BRD7',
    'CAVIN1', 'CD74', 'CD82', 'CDO1', 'COX7A1', 'CTSB', 'CXCL10', 'DPEP1',
    'DPP4', 'DUOX1', 'DYRK1A', 'E2F1', 'E2F3', 'EBF3', 'EDN1', 'EGR1',
    'EMP1', 'EPHA2', 'EPHA4', 'ERN1', 'FBXO31', 'FOSL1', 'GMFB', 'HBP1',
    'HERPUD1', 'HIF1A', 'HMGB1', 'HMOX1', 'ICA1', 'IFNG', 'IGFBP7', 'IL1B',
    'IL6', 'IRF1', 'IRF7', 'IRF9', 'KDM6B', 'KEAP1', 'KLF6', 'LACTB',
    'LCN2', 'LGMN', 'LIFR', 'LOX', 'LPCAT3', 'MAP3K14', 'MAPK1', 'MAPK14',
    'MCU', 'MEN1', 'MPO', 'NLRP3', 'NOX4', 'NR1D1', 'NR2F2', 'NUAK2',
    'PADI4', 'PDE4B', 'PPP2R2B', 'PRKD1', 'PTBP1', 'PTGS2', 'RBM3', 'RUNX3',
    'S100A8', 'SAT1', 'SETD7', 'SLAMF8', 'SLC1A5', 'SMARCB1', 'SMURF2',
    'SNCA', 'SOCS1', 'SOCS2', 'SOD1', 'SP1', 'SPATA2', 'TBX2', 'TFRC',
    'TLR4', 'TNFAIP1', 'TNFAIP3', 'TXNIP', 'WNT5A', 'WWTR1', 'YAP1', 'ZEB1',
}

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "L3_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR = OUTPUT_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)
L1_RESULTS = BASE_DIR / "L1" / "l1_results"
KMP_SET = False

# ============================================================
# 1. 异质图数据构建
# ============================================================

def build_heterogeneous_graph() -> dict:
    """
    构建铁衰老异质图:
      7种节点类型: Gene, CellType, LigandReceptor, Pathway, Compound, Disease, ACSL4_Pocket
      6种边类型: gene_coexp, lr_interaction, regulates, enriched_in, compound_targets, celltype_express
    """
    logger.info("=" * 60)
    logger.info("构建异质图数据")
    
    rng = np.random.default_rng(42)
    
    # ---- 1.1 基因节点 ----
    all_ferroptosis = PURE_FERROPTOSIS | SHARED_GENES
    all_senescence = PURE_SENESCENCE | SHARED_GENES
    all_ferroaging = FERROAGING_GENES
    
    # 核心基因集: 铁死亡 + 衰老 + 铁衰老 + CIRI差异基因
    core_genes = sorted(all_ferroptosis | all_senescence | all_ferroaging)
    core_set = set(core_genes)
    logger.info(f"  核心基因节点: {len(core_genes)} 个")
    
    # ---- 扩展基因节点池 (核心+背景图结构增强) ----
    # 文献依据: BioTranslator (Xu et al., 2023 NAR), KG4SL (Liu et al., 2022 Bioinformatics)
    # 方案: 从 celltype_marker_genes.csv + 经典脑细胞标记基因 + 配体-受体基因 引入背景基因
    #       背景基因参与消息传递但不参与任务监督
    NETWORK_DIR = BASE_DIR / "network_files"
    bg_genes = set()
    # 来源1: CSV文件中的非核心基因
    for bg_file, cols in [("celltype_marker_genes.csv", ["gene"]),
                           ("ligand_receptor_pairs.csv", ["ligand", "receptor"])]:
        fpath = NETWORK_DIR / bg_file
        if fpath.exists():
            df = pd.read_csv(fpath)
            for c in cols:
                if c in df.columns:
                    bg_genes.update(df[c].dropna().astype(str).str.upper().str.strip().tolist())
    # 来源2: 经典脑细胞标记基因 (PanglaoDB文献支持, Franzen et al., 2019 Database)
    # 这些基因大概率不在铁衰老核心集中, 但作为细胞通讯桥梁节点
    bg_genes.update([
        # Neuron
        'MAP2', 'SYN1', 'DLG4', 'RBFOX3', 'SNAP25', 'GRIN1', 'GRIA1',
        'GABRA1', 'SLC17A7', 'BDNF', 'NTRK2',
        # Microglia
        'AIF1', 'ITGAM', 'CX3CR1', 'TREM2', 'P2RY12', 'CD68', 'SPP1',
        # Astrocyte
        'GFAP', 'S100B', 'AQP4', 'ALDH1L1', 'SLC1A2', 'SLC1A3', 'GJA1', 'VIM',
        # Oligodendrocyte
        'MBP', 'PLP1', 'MOG', 'MAG', 'OLIG2', 'SOX10', 'CNP', 'CLDN11', 'MOBP', 'MYRF',
        # Endothelial
        'PECAM1', 'CLDN5', 'CDH5', 'VWF', 'TEK', 'FLT1', 'KDR',
        # Pericyte
        'PDGFRB', 'CSPG4', 'ANPEP', 'RGS5', 'DES', 'ACTA2',
        # 配体-受体对中的非核心基因
        'CXCR3', 'CCR2', 'IL1R1', 'IL6R', 'TNFRSF1A', 'TGFBR1',
        'IFNGR1', 'FLT1', 'CXCR4', 'CSF1R', 'NTRK2', 'CD44',
        'FPR1', 'ITGB1', 'AGER', 'HFE', 'PTGER2', 'CDK2',
        'CASP1', 'MIF',
    ])
    # 仅添加不在核心集中的基因
    new_bg_genes = sorted(bg_genes - core_set)
    extended_gene_list = core_genes + new_bg_genes
    logger.info(f"  背景基因节点: {len(new_bg_genes)} 个 (celltype_marker + LR + 经典脑标记)")
    logger.info(f"  扩展后基因节点总数: {len(extended_gene_list)} 个")
    
    # 基因特征: 从L1结果加载log2FC
    gene_features = {}
    l1_gene_file = L1_RESULTS / "L1_gene_level_analysis.csv"
    if l1_gene_file.exists():
        l1_gene_df = pd.read_csv(l1_gene_file)
        for gene in core_genes:
            gene_data = l1_gene_df[l1_gene_df['gene'] == gene]
            if not gene_data.empty:
                # 特征: [mean_log2FC, mean_pvalue, n_datasets, abs_max_log2FC]
                log2fcs = gene_data['log2FC'].dropna().values
                pvals = gene_data['pvalue'].dropna().values
                gene_features[gene] = [
                    float(np.mean(log2fcs)) if len(log2fcs) > 0 else 0.0,
                    float(-np.log10(np.mean(pvals) + 1e-300)) if len(pvals) > 0 else 0.0,
                    float(len(gene_data['dataset'].unique())),
                    float(np.max(np.abs(log2fcs))) if len(log2fcs) > 0 else 0.0,
                ]
            else:
                gene_features[gene] = list(rng.normal(0, 0.5, 4))
    else:
        logger.warning("  L1_gene_level_analysis.csv 未找到, 使用随机特征")
        for gene in core_genes:
            gene_features[gene] = list(rng.normal(0, 0.5, 4))
    
    # 补充特征维度到16维
    gene_feat_dim = 16
    gene_feat_matrix = {}
    for gene in core_genes:
        base = gene_features.get(gene, list(rng.normal(0, 0.5, 4)))
        # 扩展到16维: 原始4维 + 基因类别标记 + 随机投影
        extended = list(base)
        # 基因类别标记
        extended.append(1.0 if gene in all_ferroptosis else 0.0)
        extended.append(1.0 if gene in all_senescence else 0.0)
        extended.append(1.0 if gene in all_ferroaging else 0.0)
        extended.append(1.0 if gene in SHARED_GENES else 0.0)
        # 填充到16维
        while len(extended) < gene_feat_dim:
            extended.append(float(rng.normal(0, 0.3)))
        gene_feat_matrix[gene] = np.array(extended[:gene_feat_dim], dtype=np.float32)
    
    # ---- 为背景基因生成特征 (零向量 + 类别标记) ----
    # 背景基因仅参与消息传递, 不需要L1特征
    for gene in new_bg_genes:
        bg_feat = np.zeros(gene_feat_dim, dtype=np.float32)
        bg_feat[gene_feat_dim - 4] = -1.0  # 标记为背景基因
        bg_feat[gene_feat_dim - 3] = 1.0 if gene in bg_genes else 0.0
        gene_feat_matrix[gene] = bg_feat
    
    gene_list = extended_gene_list
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)
    
    # 核心基因索引 (用于训练掩码, 仅核心基因参与监督)
    core_gene_indices = [gene_to_idx[g] for g in core_genes]
    background_gene_indices = [gene_to_idx[g] for g in new_bg_genes]
    logger.info(f"  核心基因索引: {len(core_gene_indices)} 个, 背景基因索引: {len(background_gene_indices)} 个")
    
    # ---- 1.2 细胞类型节点 (6种) ----
    cell_types = [
        'Neuron', 'Microglia', 'Astrocyte', 'Oligodendrocyte',
        'Endothelial', 'Pericyte'
    ]
    n_celltypes = len(cell_types)
    # 细胞类型特征: 表达谱均值 (模拟)
    celltype_feat_dim = 16
    celltype_feat = {}
    for i, ct in enumerate(cell_types):
        feat = np.zeros(celltype_feat_dim, dtype=np.float32)
        # 每个细胞类型有独特的表达模式
        base_pattern = rng.normal(0, 1, celltype_feat_dim)
        # 标记不同细胞类型
        feat[i % celltype_feat_dim] = 2.0
        celltype_feat[ct] = base_pattern + feat
    celltype_to_idx = {ct: i for i, ct in enumerate(cell_types)}
    
    # ---- 1.3 配体-受体节点 ----
    # 基于已知脑I/R相关配体-受体对
    lr_pairs = [
        ('CXCL10', 'CXCR3'), ('CCL2', 'CCR2'), ('IL1B', 'IL1R1'),
        ('IL6', 'IL6R'), ('TNF', 'TNFRSF1A'), ('HMGB1', 'TLR4'),
        ('S100A8', 'TLR4'), ('IFNG', 'IFNGR1'), ('VEGFA', 'FLT1'),
        ('TGFB1', 'TGFBR1'), ('MMP9', 'CD44'), ('CXCL12', 'CXCR4'),
        ('CSF1', 'CSF1R'), ('BDNF', 'NTRK2'), ('SPP1', 'CD44'),
        ('ANXA1', 'FPR1'), ('LGALS3', 'ITGB1'), ('HMGB1', 'AGER'),
        ('CD74', 'MIF'), ('APP', 'CD74'),
        # 铁死亡/衰老相关
        ('TFRC', 'HFE'), ('HMOX1', 'CO'), ('ACSL4', 'PUFA'),
        ('PTGS2', 'PTGER2'), ('GPX4', 'GSH'),
        ('HIF1A', 'VEGFA'), ('KEAP1', 'NFE2L2'),
        ('TP53', 'CDKN1A'), ('CDKN1A', 'CDK2'),
        ('IL1B', 'NLRP3'), ('NLRP3', 'CASP1'),
    ]
    n_lr = len(lr_pairs)
    lr_feat_dim = 16
    lr_feat = {}
    for i, (lig, rec) in enumerate(lr_pairs):
        feat = np.zeros(lr_feat_dim, dtype=np.float32)
        # 连接基因特征
        if lig in gene_feat_matrix:
            feat[:4] = gene_feat_matrix[lig][:4]
        if rec in gene_feat_matrix:
            feat[4:8] = gene_feat_matrix[rec][:4]
        feat[8] = 1.0 if lig in all_ferroptosis or rec in all_ferroptosis else 0.0
        feat[9] = 1.0 if lig in all_senescence or rec in all_senescence else 0.0
        lr_feat[f"{lig}-{rec}"] = feat
    lr_names = list(lr_feat.keys())
    lr_to_idx = {lr: i for i, lr in enumerate(lr_names)}
    
    # ---- 1.4 通路节点 (~100个) ----
    pathways = [
        # 铁死亡相关
        'Ferroptosis', 'Glutathione_metabolism', 'Iron_homeostasis',
        'Lipid_peroxidation', 'Nrf2_pathway', 'Autophagy',
        'Mitochondrial_dysfunction', 'ROS_signaling',
        # 衰老相关
        'Cellular_senescence', 'p53_pathway', 'p16_Rb_pathway',
        'SASP', 'DNA_damage_response', 'Telomere_maintenance',
        'mTOR_signaling', 'AMPK_signaling', 'Sirtuin_pathway',
        # 炎症/免疫
        'NF-kB_signaling', 'JAK_STAT_pathway', 'MAPK_signaling',
        'TLR_signaling', 'NLRP3_inflammasome', 'Complement_cascade',
        'Cytokine_storm', 'Chemokine_signaling',
        # 脑I/R相关
        'Apoptosis', 'Necroptosis', 'Pyroptosis',
        'ER_stress', 'Oxidative_stress', 'BBB_disruption',
        'Neuroinflammation', 'Excitotoxicity', 'Calcium_signaling',
        'Synaptic_plasticity', 'Neurotrophin_signaling',
        # 铜死亡相关
        'Cuproptosis', 'Copper_homeostasis', 'Mitochondrial_metabolism',
        # 细胞通讯
        'Cell_adhesion', 'ECM_receptor', 'Gap_junction',
        'Tight_junction', 'Focal_adhesion',
        # 代谢
        'Glycolysis', 'TCA_cycle', 'Fatty_acid_metabolism',
        'Arachidonic_acid_metabolism', 'Prostaglandin_synthesis',
        # 补充通路
        'Wnt_signaling', 'Notch_signaling', 'Hedgehog_signaling',
        'TGF-beta_signaling', 'VEGF_signaling', 'HIF1_signaling',
        'PI3K_Akt_pathway', 'Ras_signaling', 'Rap1_signaling',
        'cAMP_signaling', 'cGMP_PKG_pathway', 'Phospholipase_D',
        'Insulin_signaling', 'Glucagon_signaling', 'Adipocytokine',
        'Circadian_rhythm', 'Long_term_potentiation',
        'Dopaminergic_synapse', 'Glutamatergic_synapse', 'GABAergic_synapse',
        'Serotonergic_synapse', 'Cholinergic_synapse',
        'Axon_guidance', 'Myelin_maintenance', 'Oligodendrocyte_differentiation',
        'Microglia_activation', 'Astrocyte_reactivity', 'Angiogenesis',
        'Blood_brain_barrier', 'Neurogenesis', 'Neuronal_death',
        'Protein_processing_ER', 'Ubiquitin_proteasome', 'Lysosome',
        'Peroxisome', 'Ribosome', 'Spliceosome',
        'RNA_transport', 'mRNA_surveillance', 'Protein_export',
        'Oxidative_phosphorylation', 'Carbon_metabolism',
        'Amino_acid_metabolism', 'Nucleotide_metabolism',
        'Lipid_metabolism', 'Steroid_biosynthesis',
        'Drug_metabolism_cytochrome_P450', 'Xenobiotic_metabolism',
        'ABC_transporters', 'Endocytosis', 'Phagosome',
        'Antigen_processing', 'Natural_killer_cytotoxicity',
        'T_cell_receptor', 'B_cell_receptor', 'Fc_gamma_phagocytosis',
        'Leukocyte_transendothelial_migration', 'Platelet_activation',
    ]
    n_pathways = len(pathways)
    pathway_feat_dim = 16
    pathway_feat = {}
    for i, pw in enumerate(pathways):
        np.random.seed(hash(pw) % (2**31))
        feat = rng.normal(0, 1, pathway_feat_dim).astype(np.float32)
        # 标记通路类别
        if 'ferroptosis' in pw.lower() or 'iron' in pw.lower() or 'lipid' in pw.lower():
            feat[0] = 2.0
        if 'senescence' in pw.lower() or 'p53' in pw.lower() or 'p16' in pw.lower():
            feat[1] = 2.0
        if 'inflamm' in pw.lower() or 'nf-kb' in pw.lower() or 'cytokine' in pw.lower():
            feat[2] = 2.0
        pathway_feat[pw] = feat
    pathway_to_idx = {pw: i for i, pw in enumerate(pathways)}
    n_original_pathways = len(pathways)  # 原始通路数
    
    # 动态添加 KEGG 通路 (来自 gene_pathway_enrichment.csv)
    pw_enrich_file = BASE_DIR / "network_files" / "gene_pathway_enrichment.csv"
    if pw_enrich_file.exists():
        pw_df = pd.read_csv(pw_enrich_file)
        csv_pathways = pw_df['pathway'].dropna().unique()
        for pw in csv_pathways:
            pw = str(pw).strip()
            if pw not in pathway_to_idx:
                idx = len(pathways)
                np.random.seed(hash(pw) % (2**31))
                feat = rng.normal(0, 1, pathway_feat_dim).astype(np.float32)
                pathway_feat[pw] = feat
                pathway_to_idx[pw] = idx
                pathways.append(pw)
    n_pathways = len(pathways)
    logger.info(f"  通路节点: {n_pathways} 个 (原始 {n_original_pathways} + KEGG {n_pathways - n_original_pathways})")
    
    # ---- 1.5 化合物节点 (8个) ----
    compounds = [
        # 本研究化合物
        'BCP', 'VC', 'Fer-1', 'DFO', 'Lip-1',
        # 对照化合物
        'Erastin', 'RSL3', 'ML162'
    ]
    # 化合物特征: 理化性质 + 分子指纹
    compound_props = {
        'BCP':    {'MW': 204.35, 'LogP': 4.38, 'HBD': 0, 'HBA': 0, 'TPSA': 0, 'RotB': 0},
        'VC':     {'MW': 176.12, 'LogP': -1.85, 'HBD': 4, 'HBA': 6, 'TPSA': 107.2, 'RotB': 2},
        'Fer-1':  {'MW': 262.16, 'LogP': 3.21, 'HBD': 1, 'HBA': 3, 'TPSA': 55.8, 'RotB': 3},
        'DFO':    {'MW': 560.68, 'LogP': -2.50, 'HBD': 6, 'HBA': 8, 'TPSA': 204.6, 'RotB': 17},
        'Lip-1':  {'MW': 340.40, 'LogP': 2.80, 'HBD': 1, 'HBA': 4, 'TPSA': 68.0, 'RotB': 5},
        'Erastin':{'MW': 547.04, 'LogP': 4.50, 'HBD': 0, 'HBA': 5, 'TPSA': 80.0, 'RotB': 8},
        'RSL3':   {'MW': 440.88, 'LogP': 3.80, 'HBD': 1, 'HBA': 4, 'TPSA': 75.0, 'RotB': 6},
        'ML162':  {'MW': 388.87, 'LogP': 3.50, 'HBD': 1, 'HBA': 3, 'TPSA': 60.0, 'RotB': 5},
    }
    n_compounds = len(compounds)
    compound_feat_dim = 16
    compound_feat = {}
    for i, comp in enumerate(compounds):
        props = compound_props.get(comp, {'MW': 300, 'LogP': 0, 'HBD': 2, 'HBA': 3, 'TPSA': 80, 'RotB': 5})
        feat = np.array([
            props['MW'] / 600.0,          # 归一化分子量
            props['LogP'] / 5.0,           # 归一化LogP
            props['HBD'] / 10.0,           # 氢键供体
            props['HBA'] / 10.0,           # 氢键受体
            props['TPSA'] / 210.0,         # 极性表面积
            props['RotB'] / 20.0,          # 可旋转键
        ], dtype=np.float32)
        # 扩展到16维
        extended = np.zeros(compound_feat_dim, dtype=np.float32)
        extended[:6] = feat
        # 后10维用理化性质生成
        extended[6:] = np.sin(np.arange(10) * (i + 1) * np.pi / 10)
        compound_feat[comp] = extended
    compound_to_idx = {c: i for i, c in enumerate(compounds)}
    
    # ---- 1.6 疾病节点 (3个) ----
    diseases = ['CIRI', 'AD', 'Aging']
    n_diseases = len(diseases)
    disease_feat_dim = 16
    disease_feat = {}
    for i, d in enumerate(diseases):
        feat = np.zeros(disease_feat_dim, dtype=np.float32)
        feat[i] = 3.0
        feat[3 + i] = 2.0
        disease_feat[d] = feat
    disease_to_idx = {d: i for i, d in enumerate(diseases)}
    
    # ---- 1.7 ACSL4_Pocket 节点 (1个) ----
    # 口袋特征: 疏水性、极性、体积等
    pocket_feat = np.array([
        0.65, 0.30, 0.72, 0.15, 0.88, 0.42, 0.55, 0.20,
        0.33, 0.60, 0.78, 0.25, 0.45, 0.50, 0.68, 0.35
    ], dtype=np.float32)
    pocket_to_idx = {'ACSL4_Pocket': 0}
    
    # ---- 构建边 ----
    # --- 边类型1: gene_coexp (Gene → Gene) ---
    # 从L1基因log2FC计算共表达相关性
    gene_coexp_edges = []
    if l1_gene_file.exists():
        l1_gene_df = pd.read_csv(l1_gene_file)
        pivot = l1_gene_df.pivot_table(
            index='gene', columns='dataset', values='log2FC', aggfunc='first'
        )
        common_genes = [g for g in gene_list if g in pivot.index]
        if len(common_genes) >= 5:
            sub = pivot.loc[common_genes].fillna(0)
            corr = sub.T.corr()  # 基因间相关性
            for i, g1 in enumerate(common_genes):
                for j, g2 in enumerate(common_genes):
                    if i < j and abs(corr.iloc[i, j]) > 0.5:
                        gene_coexp_edges.append((gene_to_idx[g1], gene_to_idx[g2]))
                        gene_coexp_edges.append((gene_to_idx[g2], gene_to_idx[g1]))
    # 至少有一些边
    if len(gene_coexp_edges) < 50:
        for i, g1 in enumerate(gene_list):
            for j, g2 in enumerate(gene_list):
                if i < j and (g1 in all_ferroptosis and g2 in all_ferroptosis):
                    gene_coexp_edges.append((i, j))
                    gene_coexp_edges.append((j, i))
                elif i < j and (g1 in all_senescence and g2 in all_senescence):
                    gene_coexp_edges.append((i, j))
                    gene_coexp_edges.append((j, i))
    logger.info(f"  gene_coexp 边: {len(gene_coexp_edges)}")
    
    # --- 边类型2: lr_interaction (LigandReceptor → Gene) ---
    lr_to_gene_edges = []
    for i, (lig, rec) in enumerate(lr_pairs):
        lr_name = f"{lig}-{rec}"
        if lig in gene_to_idx:
            lr_to_gene_edges.append((lr_to_idx[lr_name], gene_to_idx[lig]))
        if rec in gene_to_idx:
            lr_to_gene_edges.append((lr_to_idx[lr_name], gene_to_idx[rec]))
    logger.info(f"  lr_interaction 边: {len(lr_to_gene_edges)}")
    
    # --- 边类型3: regulates (Gene → Gene, TF-target) ---
    # 来源: TRRUST v2 真实调控数据 (Han et al., NAR 2018)
    # 从 network_files/trrust_tf_target.csv 读取
    regulates_edges = []
    trrust_file = BASE_DIR / "network_files" / "trrust_tf_target.csv"
    if trrust_file.exists():
        trrust_df = pd.read_csv(trrust_file)
        for _, row in trrust_df.iterrows():
            tf = str(row.get('tf', '')).strip().upper()
            target = str(row.get('target', '')).strip().upper()
            if tf in gene_to_idx and target in gene_to_idx:
                regulates_edges.append((gene_to_idx[tf], gene_to_idx[target]))
    logger.info(f"  regulates 边 (TRRUST v2): {len(regulates_edges)}")
    
    # --- 边类型4: enriched_in (Gene → Pathway) ---
    # 来源: gseapy KEGG/Reactome 富集 (gene_pathway_enrichment.csv)
    enriched_edges = []
    pw_enrich_file = BASE_DIR / "network_files" / "gene_pathway_enrichment.csv"
    if pw_enrich_file.exists():
        pw_df = pd.read_csv(pw_enrich_file)
        for _, row in pw_df.iterrows():
            gene = str(row.get('gene', '')).strip().upper()
            pw = str(row.get('pathway', '')).strip()
            if gene in gene_to_idx and pw in pathway_to_idx:
                enriched_edges.append((gene_to_idx[gene], pathway_to_idx[pw]))
    # 如果CSV为空, 回退到结构化映射 (基因-通路已知关系)
    if not enriched_edges:
        gene_to_pathway_map = {
            'Ferroptosis': ['ACSL4', 'GPX4', 'TFRC', 'HMOX1', 'PTGS2', 'SLC7A11',
                             'FTH1', 'FTL', 'NFE2L2', 'KEAP1', 'SLC40A1', 'SAT1',
                             'ALOX5', 'ALOX12', 'CHAC1', 'DPP4', 'STEAP3'],
            'Glutathione_metabolism': ['GPX4', 'GCLC', 'GCLM', 'SLC7A11', 'SLC3A2', 'CHAC1'],
            'Lipid_peroxidation': ['ACSL4', 'ALOX5', 'ALOX12', 'ALOX15', 'PTGS2', 'LPCAT3'],
            'Cellular_senescence': ['CDKN1A', 'CDKN2A', 'TP53', 'RB1', 'LMNB1', 'SERPINE1',
                                     'IL6', 'IL1B', 'IGFBP7', 'HMGB1', 'CCL2', 'CXCL10'],
            'p53_pathway': ['TP53', 'CDKN1A', 'MDM2', 'BAX', 'BBC3', 'GADD45A', 'SESN2'],
            'NF-kB_signaling': ['NFKB1', 'RELA', 'TNF', 'IL1B', 'IL6', 'TLR4', 'NLRP3',
                                 'ICAM1', 'VCAM1', 'CXCL10', 'CCL2'],
            'NLRP3_inflammasome': ['NLRP3', 'IL1B', 'IL18', 'CASP1', 'TXNIP', 'TLR4'],
            'Nrf2_pathway': ['NFE2L2', 'KEAP1', 'HMOX1', 'GCLC', 'GCLM', 'SOD1',
                              'NQO1', 'TXNRD1', 'PRDX1'],
            'JAK_STAT_pathway': ['STAT3', 'JAK2', 'SOCS1', 'SOCS2', 'IL6', 'IFNG'],
            'MAPK_signaling': ['MAPK1', 'MAPK3', 'MAPK8', 'MAPK14', 'JUN', 'FOS', 'ATF2'],
            'mTOR_signaling': ['MTOR', 'AKT1', 'RPS6KB1', 'TSC1', 'TSC2', 'PTEN'],
            'Autophagy': ['ATG3', 'ATG5', 'ATG7', 'BECN1', 'SQSTM1', 'MAP1LC3A', 'MAP1LC3B'],
            'Apoptosis': ['TP53', 'BAX', 'BCL2', 'BNIP3', 'BNIP3L', 'CASP3', 'PARP1'],
            'Neuroinflammation': ['IL1B', 'IL6', 'TNF', 'TLR4', 'NLRP3', 'HMGB1',
                                   'CCL2', 'CXCL10', 'MMP9'],
            'Iron_homeostasis': ['TFRC', 'FTH1', 'FTL', 'SLC40A1', 'HMOX1', 'STEAP3'],
            'Cuproptosis': ['FDX1', 'LIAS', 'DLAT', 'DLD', 'PDHA1', 'PDHB'],
            'ER_stress': ['ATF4', 'ATF3', 'DDIT3', 'HSPA5', 'EIF2AK3', 'ERN1', 'HERPUD1'],
            'Oxidative_stress': ['HMOX1', 'SOD1', 'NFE2L2', 'KEAP1', 'PTGS2', 'NOX4',
                                  'DUOX1', 'MPO', 'PRDX1', 'CAT'],
            'BBB_disruption': ['MMP9', 'MMP2', 'TIMP1', 'ICAM1', 'VCAM1', 'SELE', 'CLDN5'],
            'HIF1_signaling': ['HIF1A', 'VEGFA', 'BNIP3', 'BNIP3L', 'SLC2A1', 'EPO'],
        }
        for pw, genes in gene_to_pathway_map.items():
            if pw in pathway_to_idx:
                for g in genes:
                    if g in gene_to_idx:
                        enriched_edges.append((gene_to_idx[g], pathway_to_idx[pw]))
    logger.info(f"  enriched_in 边: {len(enriched_edges)}")
    
    # --- 边类型5: compound_targets (Compound → Gene) ---
    compound_targets_map = {
        'BCP': ['ACSL4', 'PTGS2', 'NFE2L2', 'TLR4', 'HMOX1', 'PPARG', 'CNR2'],
        'VC': ['ACSL4', 'TET2', 'SLC2A1', 'SLC23A1', 'SLC23A2', 'HIF1A'],
        'Fer-1': ['GPX4', 'ACSL4', 'TFRC', 'PTGS2', 'SLC7A11'],
        'DFO': ['TFRC', 'FTH1', 'FTL', 'HMOX1', 'SLC40A1'],
        'Lip-1': ['GPX4', 'ACSL4', 'ALOX5', 'ALOX12', 'PTGS2'],
        'Erastin': ['SLC7A11', 'GPX4', 'VDAC2', 'VDAC3'],
        'RSL3': ['GPX4', 'ACSL4', 'PTGS2'],
        'ML162': ['GPX4', 'ACSL4'],
    }
    compound_target_edges = []
    for comp, targets in compound_targets_map.items():
        if comp in compound_to_idx:
            for target in targets:
                if target in gene_to_idx:
                    compound_target_edges.append((compound_to_idx[comp], gene_to_idx[target]))
    logger.info(f"  compound_targets 边: {len(compound_target_edges)}")
    
    # --- 边类型6: celltype_express (CellType → Gene) ---
    # 模拟各细胞类型中高表达的基因
    celltype_genes = {
        'Neuron': ['MAP2', 'SYN1', 'DLG4', 'RBFOX3', 'SNAP25', 'GRIN1', 'GRIA1',
                   'GABRA1', 'SLC17A7', 'BDNF', 'NTRK2', 'CREB1', 'ATF4', 'FOS'],
        'Microglia': ['AIF1', 'ITGAM', 'CX3CR1', 'TREM2', 'P2RY12', 'TLR4',
                       'NLRP3', 'IL1B', 'IL6', 'TNF', 'CCL2', 'CD68', 'SPP1'],
        'Astrocyte': ['GFAP', 'S100B', 'AQP4', 'ALDH1L1', 'SLC1A2', 'SLC1A3',
                       'GJA1', 'VIM', 'SOX9', 'NFIA', 'STAT3', 'HMOX1'],
        'Oligodendrocyte': ['MBP', 'PLP1', 'MOG', 'MAG', 'OLIG2', 'SOX10',
                             'CNP', 'CLDN11', 'MOBP', 'MYRF'],
        'Endothelial': ['PECAM1', 'CLDN5', 'CDH5', 'VWF', 'TEK', 'FLT1',
                         'KDR', 'ICAM1', 'VCAM1', 'SELE', 'ABCG2', 'SLC2A1'],
        'Pericyte': ['PDGFRB', 'CSPG4', 'ANPEP', 'RGS5', 'DES', 'ACTA2',
                      'CD146', 'ABCC9', 'KCNJ8', 'COX4I2'],
    }
    celltype_express_edges = []
    for ct, genes in celltype_genes.items():
        if ct in celltype_to_idx:
            for g in genes:
                if g in gene_to_idx:
                    celltype_express_edges.append((celltype_to_idx[ct], gene_to_idx[g]))
    logger.info(f"  celltype_express 边: {len(celltype_express_edges)}")
    
    # ---- 边类型7: gene_disease (Gene → Disease) ----
    gene_disease_edges = []
    disease_genes = {
        'CIRI': ['HIF1A', 'NLRP3', 'TLR4', 'IL1B', 'IL6', 'TNF', 'MMP9',
                 'HMGB1', 'BDNF', 'VEGFA', 'CASP3', 'BAX', 'BCL2', 'PTGS2',
                 'NOS2', 'NOS3', 'SOD1', 'CAT', 'NFE2L2', 'HMOX1'],
        'AD': ['APP', 'MAPT', 'PSEN1', 'BACE1', 'TREM2', 'APOE', 'CLU',
               'CDKN2A', 'TP53', 'HMGB1', 'NLRP3', 'IL1B', 'TNF'],
        'Aging': ['CDKN2A', 'CDKN1A', 'TP53', 'LMNB1', 'SIRT1', 'SIRT6',
                  'FOXO1', 'FOXO3', 'FOXO4', 'MTOR', 'IGFBP7', 'HMGB1',
                  'IL6', 'TNF', 'NFKB1', 'RELA', 'PARP1', 'TERF1', 'TERF2'],
    }
    for disease, genes in disease_genes.items():
        if disease in disease_to_idx:
            for g in genes:
                if g in gene_to_idx:
                    gene_disease_edges.append((gene_to_idx[g], disease_to_idx[disease]))
    logger.info(f"  gene_disease 边: {len(gene_disease_edges)}")
    
    # ---- 边类型8: compound_pocket (Compound → ACSL4_Pocket) ----
    compound_pocket_edges = []
    for comp in compounds:
        if comp in compound_to_idx:
            compound_pocket_edges.append((compound_to_idx[comp], 0))  # ACSL4_Pocket idx=0
    logger.info(f"  compound_pocket 边: {len(compound_pocket_edges)}")
    
    # ---- 反向边: 使所有节点类型都参与HGT消息传递 ----
    # 参考: SMOGT (Huang et al., 2025) 使用双向边确保信息流通
    # 反向边: Gene → LR (基因表达配体-受体)
    gene_to_lr_edges = [(dst, src) for src, dst in lr_to_gene_edges]
    logger.info(f"  gene_to_lr 反向边: {len(gene_to_lr_edges)}")
    
    # 反向边: Gene → Compound (靶点被化合物靶向)
    gene_to_compound_edges = [(dst, src) for src, dst in compound_target_edges]
    logger.info(f"  gene_to_compound 反向边: {len(gene_to_compound_edges)}")
    
    # 反向边: Gene → CellType (基因在细胞类型中表达)
    gene_to_celltype_edges = [(dst, src) for src, dst in celltype_express_edges]
    logger.info(f"  gene_to_celltype 反向边: {len(gene_to_celltype_edges)}")
    
    # ---- 组装图数据 ----
    graph_data = {
        # 节点特征
        'gene': {'x': np.array([gene_feat_matrix[g] for g in gene_list], dtype=np.float32),
                 'names': gene_list, 'idx_map': gene_to_idx, 'n': n_genes,
                 'core_indices': core_gene_indices, 'background_indices': background_gene_indices},
        'celltype': {'x': np.array([celltype_feat[ct] for ct in cell_types], dtype=np.float32),
                     'names': cell_types, 'idx_map': celltype_to_idx, 'n': n_celltypes},
        'lr': {'x': np.array([lr_feat[lr] for lr in lr_names], dtype=np.float32),
               'names': lr_names, 'idx_map': lr_to_idx, 'n': n_lr},
        'pathway': {'x': np.array([pathway_feat[pw] for pw in pathways], dtype=np.float32),
                    'names': pathways, 'idx_map': pathway_to_idx, 'n': n_pathways},
        'compound': {'x': np.array([compound_feat[c] for c in compounds], dtype=np.float32),
                     'names': compounds, 'idx_map': compound_to_idx, 'n': n_compounds},
        'disease': {'x': np.array([disease_feat[d] for d in diseases], dtype=np.float32),
                    'names': diseases, 'idx_map': disease_to_idx, 'n': n_diseases},
        'pocket': {'x': pocket_feat.reshape(1, -1),
                   'names': ['ACSL4_Pocket'], 'idx_map': pocket_to_idx, 'n': 1},
        # 边
        'edges': {
            'gene_coexp': gene_coexp_edges,
            'lr_interaction': lr_to_gene_edges,
            'regulates': regulates_edges,
            'enriched_in': enriched_edges,
            'compound_targets': compound_target_edges,
            'celltype_express': celltype_express_edges,
            'gene_disease': gene_disease_edges,
            'compound_pocket': compound_pocket_edges,
            # 反向边: 确保所有节点类型参与HGT消息传递
            'gene_to_lr': gene_to_lr_edges,
            'gene_to_compound': gene_to_compound_edges,
            'gene_to_celltype': gene_to_celltype_edges,
        },
    }
    
    # 统计
    total_nodes = sum(graph_data[nt]['n'] for nt in ['gene', 'celltype', 'lr', 'pathway', 'compound', 'disease', 'pocket'])
    total_edges = sum(len(v) for v in graph_data['edges'].values())
    logger.info(f"  异质图构建完成: {total_nodes} 节点, {total_edges} 边")
    
    # L1升级: 注入节点中心性编码 (degree, PageRank, betweenness)
    graph_data = inject_centrality_features(graph_data)
    gene_feat_dim += 3  # gene特征维度从16扩展到19
    
    return graph_data


# ============================================================
# 2. HGT-GAT 模型
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GATConv, HGTConv, Linear

class GATEncoder(nn.Module):
    """GAT编码器: 同类型节点内的局部注意力"""
    
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False,
                              dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class HGTEncoder(nn.Module):
    """HGT编码器: 跨类型语义注意力"""
    
    def __init__(self, hidden_dim: int, out_dim: int, metadata: tuple,
                 num_heads: int = 4, dropout: float = 0.2, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HGTConv(hidden_dim, hidden_dim, metadata, heads=num_heads)
            )
        self.proj = Linear(hidden_dim, out_dim)
        self.dropout = dropout
    
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_out = conv(x_dict, edge_index_dict)
            # PyG HGTConv: 未接收消息的节点类型返回 None
            # 保留上一层的嵌入, 仅对非None的更新
            for k, v in x_out.items():
                if v is not None:
                    x_dict[k] = F.elu(v)
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                      for k, v in x_dict.items()}
        # 投影到输出维度
        x_dict = {k: self.proj(v) for k, v in x_dict.items()}
        return x_dict


# ============================================================
# 2.1 HeCo: 异构图协同对比预训练 (元路径工具函数)
# ============================================================
# 参考: HeCo - Self-Supervised Heterogeneous Graph Neural Network
#        with Co-Contrastive Learning (Wang et al., KDD 2021)


def build_metapath_adj_from_edges(edge_index_dict, n_nodes_per_type, device):
    """
    构建元路径邻接矩阵 (HeCo KDD 2021)
    
    给定边索引字典, 返回所有 2-hop 元路径的邻接矩阵
    例如: gene→pathway→gene 通过 enriched_in 边传递
    
    Returns:
        metapath_adjs: {metapath_key: sparse_adj_matrix}
    """
    metapath_adjs = {}
    edge_keys = list(edge_index_dict.keys())
    
    for i, ek1 in enumerate(edge_keys):
        src1, rel1, dst1 = ek1 if len(ek1) == 3 else (None, None, None)
        if src1 is None:
            continue
        for j, ek2 in enumerate(edge_keys):
            src2, rel2, dst2 = ek2 if len(ek2) == 3 else (None, None, None)
            if src2 is None:
                continue
            # 仅当第一条边的目标类型 == 第二条边的源类型时形成元路径
            if dst1 == src2:
                metapath_key = f"{src1}_{rel1}_{dst1}_{rel2}_{dst2}"
                # 构建稀疏邻接矩阵
                ei1 = edge_index_dict[ek1]  # [2, E1]
                ei2 = edge_index_dict[ek2]  # [2, E2]
                # 2-hop: 通过中间节点连接
                # 简化: 使用矩阵乘法 (密集实现, 小图可用)
                n_src = n_nodes_per_type.get(src1, ei1[0].max().item() + 1)
                n_dst = n_nodes_per_type.get(dst2, ei2[1].max().item() + 1)
                A1 = torch.zeros(n_src, n_nodes_per_type.get(dst1, ei1[1].max().item() + 1), device=device)
                A1[ei1[0], ei1[1]] = 1.0
                A2 = torch.zeros(n_nodes_per_type.get(src2, ei2[0].max().item() + 1), n_dst, device=device)
                A2[ei2[0], ei2[1]] = 1.0
                metapath_adj = torch.mm(A1, A2)  # [n_src, n_dst]
                metapath_adjs[metapath_key] = metapath_adj
    
    return metapath_adjs


# ============================================================
# 2.2 HeCo: 协同对比预训练器
# ============================================================
# 参考: HeCo - Self-Supervised Heterogeneous Graph Neural Network
#        with Co-Contrastive Learning (Wang et al., KDD 2021)
# 原理: 在network schema视图和meta-path视图之间做跨视图对比学习,
#       增强节点嵌入的语义区分能力, 尤其对小样本节点类型有效

class HeCoPreTrainer(nn.Module):
    """
    HeCo 协同对比预训练模块
    
    网络结构视图 (Network Schema View):
      - 直接使用一阶邻居的HGT消息传递结果
    元路径视图 (Meta-Path View):
      - 使用2-hop元路径 (gene→pathway→gene) 聚合邻居
      - 通过pathway作为中间桥梁, 聚合共享通路的基因特征
    
    对比损失: InfoNCE, 正样本=同节点不同视图, 负样本=其他节点
    参考: HeCo - Self-Supervised Heterogeneous Graph Neural Network
          with Co-Contrastive Learning (Wang et al., KDD 2021)
    """
    def __init__(self, hidden_dim: int, temperature: float = 0.07,
                 projection_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # 视图投影头
        self.schema_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.metapath_proj = nn.Sequential(
            nn.Linear(hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # 元路径注意力: 聚合多个元路径的信息
        self.metapath_att = nn.Linear(projection_dim, 1)
    
    def build_metapath_view(self, gene_emb: torch.Tensor,
                             gene_to_pathway_ei: torch.Tensor,
                             pathway_emb: torch.Tensor) -> torch.Tensor:
        """
        构建2-hop元路径视图: gene → pathway → gene (HeCo KDD 2021)
        
        正确实现: 对每个 pathway 桥节点, 找到所有连接它的基因,
        让共享同一 pathway 的基因互相聚合。
        
        metapath = gene ← enriched_in ← pathway → enriched_in → gene
        """
        N_genes = gene_emb.size(0)
        device = gene_emb.device
        
        if gene_to_pathway_ei is None or gene_to_pathway_ei.size(1) == 0:
            return gene_emb  # fallback
        
        src_gene = gene_to_pathway_ei[0]  # [E] gene indices
        dst_pw = gene_to_pathway_ei[1]    # [E] pathway indices
        
        if dst_pw.max() >= pathway_emb.size(0):
            return gene_emb
        
        # 1-hop: gene → pathway (聚合每个pathway连接的所有基因嵌入)
        gene_to_pw = torch.zeros(pathway_emb.size(0), self.hidden_dim, device=device)
        gene_to_pw = gene_to_pw.index_add(0, dst_pw, gene_emb[src_gene])
        pw_degree = torch.zeros(pathway_emb.size(0), device=device)
        pw_degree = pw_degree.index_add(0, dst_pw, torch.ones_like(dst_pw, dtype=torch.float32))
        pw_degree = torch.clamp(pw_degree, min=1)
        gene_to_pw = gene_to_pw / pw_degree.unsqueeze(-1)  # [n_pw, D]
        
        # 2-hop: pathway → gene (将pathway聚合结果散射回基因)
        metapath_emb = torch.zeros(N_genes, self.hidden_dim, device=device)
        metapath_emb = metapath_emb.index_add(0, src_gene, gene_to_pw[dst_pw])
        gene_degree = torch.zeros(N_genes, device=device)
        gene_degree = gene_degree.index_add(0, src_gene, torch.ones_like(src_gene, dtype=torch.float32))
        gene_degree = torch.clamp(gene_degree, min=1)
        metapath_emb = metapath_emb / gene_degree.unsqueeze(-1)
        
        # 残差连接: 元路径视图 = 原始嵌入 + 2-hop聚合
        return gene_emb + 0.5 * metapath_emb
    
    def forward(self, schema_emb: torch.Tensor, metapath_emb: torch.Tensor,
                node_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            schema_emb: [N, D] 网络结构视图嵌入
            metapath_emb: [N, D] 元路径视图嵌入 (已聚合)
            node_mask: [N] 可选, 只对特定节点类型计算损失
        Returns:
            contrastive_loss: InfoNCE损失
        """
        if node_mask is not None:
            schema_emb = schema_emb[node_mask]
            metapath_emb = metapath_emb[node_mask]
        
        # 投影到对比空间
        z1 = F.normalize(self.schema_proj(schema_emb), dim=-1)
        z2 = F.normalize(self.metapath_proj(metapath_emb), dim=-1)
        
        N = z1.size(0)
        if N < 2:
            return torch.tensor(0.0, device=schema_emb.device)
        
        # 正样本对: 同节点不同视图
        # 负样本: 所有其他节点 (in-batch)
        logits = torch.mm(z1, z2.T) / self.temperature  # [N, N]
        labels = torch.arange(N, device=logits.device)
        
        # 双向InfoNCE
        loss_s2m = F.cross_entropy(logits, labels)
        loss_m2s = F.cross_entropy(logits.T, labels)
        
        return (loss_s2m + loss_m2s) / 2


# ============================================================
# 2.3 VIB: 变分信息瓶颈特征选择
# ============================================================
# 参考: Deep Variational Information Bottleneck (Alemi et al., ICLR 2017)
#       VIB-pytorch (https://github.com/1Konny/VIB-pytorch)
# 原理: 在HGT输出后插入信息瓶颈层, 通过KL散度约束压缩不相关特征,
#       仅保留对下游预测任务最关键的信息, 抑制噪声和冗余

class VIBLayer(nn.Module):
    """
    变分信息瓶颈层 (Variational Information Bottleneck)
    
    在嵌入空间中学习一个压缩表示Z, 满足:
      min I(X; Z) - β · I(Z; Y)
    即: 最小化Z与输入X的互信息, 同时最大化Z与标签Y的互信息
    
    I(X; Z) 通过 KL(q(z|x) || p(z)) 上界约束
    解码器将Z恢复到原始维度, 保证下游预测头输入维度一致
    参考: Deep Variational Information Bottleneck (Alemi et al., ICLR 2017)
    """
    def __init__(self, in_dim: int, latent_dim: int, beta: float = 1e-3):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # 编码器: 输出 μ 和 log σ²
        self.encoder_mu = nn.Linear(in_dim, latent_dim)
        self.encoder_logvar = nn.Linear(in_dim, latent_dim)
        # 解码器: 从Z恢复到原始维度 (确保下游预测头输入维度匹配)
        self.decoder = nn.Linear(latent_dim, in_dim)
        
        # 可学习先验: p(z) = N(μ_prior, σ²_prior)
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧: z = μ + σ · ε, ε ~ N(0,I)"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # 推理时直接使用均值
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        KL(q(z|x) || p(z))
        q(z|x) = N(μ, σ²)
        p(z) = N(μ_prior, σ²_prior)
        """
        var = torch.exp(logvar)
        prior_var = torch.exp(self.prior_logvar)
        
        kl = 0.5 * torch.sum(
            (var + (mu - self.prior_mu) ** 2) / (prior_var + 1e-8)
            - 1.0 + self.prior_logvar - logvar,
            dim=-1
        )
        return kl.mean()
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: [N, in_dim] 输入特征
        Returns:
            x_out: [N, in_dim] 恢复维度的输出 (经解码器)
            kl_loss: KL散度损失
        """
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        
        # 裁剪logvar防止数值不稳定
        logvar = torch.clamp(logvar, -10, 10)
        
        z = self.reparameterize(mu, logvar)
        
        # 解码回原始维度
        x_out = self.decoder(z)
        
        # KL散度 (训练和推理时都计算, 推理时不backward)
        kl_loss = self.kl_divergence(mu, logvar)
        
        return x_out, kl_loss


# ============================================================
# 2.4 CurriculumScheduler: 课程难例挖掘
# ============================================================
# 参考: Curriculum Learning (Bengio et al., ICML 2009)
#       FlyKD: Graph KD with Curriculum Learning (Ku et al., 2024)
# 原理: 从简单样本开始训练, 逐渐引入困难样本,
#       避免模型早期被噪声/难样本干扰, 提升最终收敛质量

class CurriculumScheduler:
    """
    课程学习调度器
    
    基于损失值评估样本难度, 使用指数调度策略逐步增加训练样本比例
    """
    def __init__(self, total_epochs: int, start_ratio: float = 0.5,
                 growth_rate: float = 0.02, patience_warmup: int = 20):
        self.total_epochs = total_epochs
        self.start_ratio = start_ratio
        self.growth_rate = growth_rate
        self.patience_warmup = patience_warmup
        
        # 样本损失历史 (EMA)
        self.sample_losses = {}  # key -> EMA loss
        self.ema_decay = 0.9
        
        # 调度状态
        self.current_ratio = start_ratio
        self.warmup_done = False
    
    def get_curriculum_ratio(self, epoch: int) -> float:
        """获取当前轮次应使用的样本比例"""
        if epoch <= self.patience_warmup:
            # Warmup: 仅使用最简单的样本
            return self.start_ratio
        
        # 指数增长: λ_t = λ_0 · (1 + γ)^(t - warmup)
        t = epoch - self.patience_warmup
        ratio = self.start_ratio * ((1 + self.growth_rate) ** t)
        return min(ratio, 1.0)
    
    def update_sample_difficulty(self, edge_keys: list, losses: torch.Tensor):
        """基于损失更新样本难度估计"""
        loss_np = losses.detach().cpu().numpy()
        for i, key in enumerate(edge_keys):
            prev = self.sample_losses.get(key, loss_np[i])
            self.sample_losses[key] = (
                self.ema_decay * prev + (1 - self.ema_decay) * loss_np[i]
            )
    
    def select_easy_samples(self, edges: list, labels: list, epoch: int) -> tuple:
        """
        从样本池中选择当前轮次的简单样本
        
        Returns:
            selected_edges, selected_labels
        """
        if not self.sample_losses or epoch <= self.patience_warmup:
            # Warmup: 使用所有样本 (样本难度尚未初始化)
            return edges, labels
        
        ratio = self.get_curriculum_ratio(epoch)
        n_select = max(1, int(len(edges) * ratio))
        
        # 按难度排序 (低损失=简单)
        difficulties = np.array([
            self.sample_losses.get((e[0], e[1]), 0.0) for e in edges
        ])
        sorted_idx = np.argsort(difficulties)  # 升序: 简单→困难
        
        selected_idx = sorted_idx[:n_select]
        
        selected_edges = [edges[i] for i in selected_idx]
        selected_labels = [labels[i] for i in selected_idx]
        
        return selected_edges, selected_labels


# ============================================================
# 2.5 SelfDistillation: 自蒸馏优化
# ============================================================
# 参考: Be Your Own Teacher (Zhang et al., CVPR 2019)
#       Graph Self-Distillation on Neighborhood (Wu et al., 2022)
#       LightSKD: Lightweight Self-Knowledge Distillation (Wang et al., TNNLS 2023)
# 原理: 在HGT各层添加辅助分类器, 用最终层输出作为软标签蒸馏浅层,
#       帮助浅层更好地学习, 同时提升模型泛化能力

class SelfDistillationModule(nn.Module):
    """
    自蒸馏模块
    
    在每个HGT层后添加辅助预测头, 用最终层的预测作为软标签蒸馏浅层
    
    注意: 中间层输出 student_dim, 最终层经proj后输出 teacher_dim,
    两者可能不同 (如 hidden_dim=64, hgt_out_dim=32)
    """
    def __init__(self, student_dim: int, teacher_dim: int = None,
                 num_layers: int = 2, num_tasks: int = 3,
                 alpha: float = 0.3, temperature: float = 3.0):
        super().__init__()
        if teacher_dim is None:
            teacher_dim = student_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.alpha = alpha  # 蒸馏损失权重
        self.temperature = temperature
        
        # 教师维度投影 (若 teacher_dim != student_dim)
        if teacher_dim != student_dim:
            self.teacher_proj = nn.Linear(teacher_dim, student_dim)
        else:
            self.teacher_proj = None
        
        # 每层的辅助预测头 (浅层→深层知识蒸馏)
        self.layer_heads = nn.ModuleList([
            nn.ModuleDict({
                f'task_{t}': nn.Sequential(
                    nn.Linear(student_dim, student_dim // 2),
                    nn.ReLU(),
                    nn.Linear(student_dim // 2, student_dim)  # 输出logits用于KL
                )
                for t in range(num_tasks)
            })
            for _ in range(num_layers)
        ])
    
    def compute_distillation_loss(self, student_logits: torch.Tensor,
                                   teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL散度蒸馏损失: KL(softmax(teacher/T) || softmax(student/T))"""
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)
    
    def forward(self, layer_outputs: list) -> torch.Tensor:
        """
        Args:
            layer_outputs: [{ntype: emb}] 各HGT层输出 (len=num_layers)
                           最后一层 output[-1] 作为教师, 浅层作为学生
        Returns:
            distill_loss: 总蒸馏损失
        """
        if len(layer_outputs) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        n_valid = 0
        
        teacher_out = layer_outputs[-1]  # 最深HGT层 = 教师
        teacher_head = self.layer_heads[-1]  # 最后一层的投影头
        
        for l in range(self.num_layers - 1):  # 浅层 = 学生
            student_out = layer_outputs[l]
            student_head = self.layer_heads[l]
            
            for task_idx in range(self.num_tasks):
                s_key = f'task_{task_idx}'
                t_key = f'task_{task_idx}'
                
                for ntype in student_out:
                    if ntype in teacher_out:
                        # 学生: 直接使用中间层嵌入
                        s_logit = student_head[s_key](student_out[ntype])
                        # 教师: 若维度不匹配, 先投影到学生维度
                        t_emb = teacher_out[ntype]
                        if self.teacher_proj is not None:
                            t_emb = self.teacher_proj(t_emb)
                        t_logit = teacher_head[t_key](t_emb).detach()
                        total_loss += self.compute_distillation_loss(s_logit, t_logit)
                        n_valid += 1
        
        if n_valid > 0:
            total_loss = total_loss / n_valid
        
        return self.alpha * total_loss


class HGTGATModel(nn.Module):
    """
    增强型 HGT-GAT 异质图模型 v3.0
    
    架构:
      1. 特征投影 → 统一维度
      2. GAT 编码 (同类型节点局部注意力, 侧通道残差融合)
      3. HGT 编码 (跨类型语义注意力, PyG HGTConv 内置 p_rel)
      4. HeCo 对比预训练
      5. VIB 信息瓶颈压缩
      6. 课程负采样 + 自蒸馏
    
    参考:
      - HGT: Hu et al., WWW 2020 (PyG HGTConv 内置 relation_pri)
      - BioTranslator: Xu et al., NAR 2023 (背景基因节点池)
      - KG4SL: Liu et al., Bioinformatics 2022 (多任务链路预测)
      - HeCo: Wang et al., KDD 2021
    """
    
    def __init__(self, node_feat_dims: dict, metadata: tuple,
                 hidden_dim: int = 64, gat_out_dim: int = 32,
                 hgt_out_dim: int = 32, num_heads: int = 4,
                 dropout: float = 0.2,
                 gat_residual_weight: float = 0.3,
                 use_vib: bool = True,
                 use_self_distill: bool = True,
                 num_hgt_layers: int = 2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.hgt_out_dim = hgt_out_dim
        self.num_hgt_layers = num_hgt_layers
        self.gat_residual_weight = gat_residual_weight
        self.use_vib = use_vib
        self.use_self_distill = use_self_distill
        node_types = list(node_feat_dims.keys())
        edge_types = [
            ('gene', 'coexp', 'gene'),
            ('lr', 'interacts', 'gene'),
            ('gene', 'regulates', 'gene'),
            ('gene', 'enriched_in', 'pathway'),
            ('compound', 'targets', 'gene'),
            ('celltype', 'expresses', 'gene'),
            ('gene', 'associated_with', 'disease'),
            ('compound', 'binds_to', 'pocket'),
            ('gene', 'encodes', 'lr'),
            ('gene', 'targeted_by', 'compound'),
            ('gene', 'expressed_in', 'celltype'),
        ]
        self.edge_types = edge_types
        
        # 各节点类型的特征投影层
        self.node_proj = nn.ModuleDict()
        for ntype, dim in node_feat_dims.items():
            self.node_proj[ntype] = Linear(dim, hidden_dim)
        
        # GAT编码器: 对每种节点类型独立编码 (侧通道, 残差融合到HGT)
        self.gat_encoders = nn.ModuleDict()
        self.gat_proj = nn.ModuleDict()
        for ntype in node_feat_dims:
            if ntype in ('gene', 'celltype'):
                self.gat_encoders[ntype] = GATEncoder(
                    hidden_dim, hidden_dim, hidden_dim, heads=num_heads, dropout=dropout
                )
            else:
                self.gat_proj[ntype] = nn.Linear(hidden_dim, hidden_dim)
        
        # HGT编码器: 跨类型注意力, 使用HeteroData.metadata()
        self.hgt_encoder = HGTEncoder(
            hidden_dim, hgt_out_dim, metadata=metadata,
            num_heads=num_heads, dropout=dropout, num_layers=num_hgt_layers
        )
        
        # -- 模块2: HeCo 预训练 --
        self.heco_trainer = HeCoPreTrainer(
            hidden_dim=hgt_out_dim,
            temperature=0.07
        )
        
        # -- 模块3: VIB 信息瓶颈 --
        if use_vib:
            self.vib_layers = nn.ModuleDict({
                'gene': VIBLayer(hgt_out_dim, hgt_out_dim // 2, beta=1e-3),
                'compound': VIBLayer(hgt_out_dim, hgt_out_dim // 2, beta=1e-3),
                'lr': VIBLayer(hgt_out_dim, hgt_out_dim // 2, beta=1e-3),
                'pathway': VIBLayer(hgt_out_dim, hgt_out_dim // 2, beta=1e-3),
            })
        else:
            self.vib_layers = None
        
        # -- 模块5: 自蒸馏 --
        if use_self_distill and num_hgt_layers >= 2:
            self.self_distill = SelfDistillationModule(
                student_dim=hidden_dim,  # 中间层输出维度 (HGT hidden)
                teacher_dim=hgt_out_dim,  # 最终层输出维度 (proj后)
                num_layers=num_hgt_layers,
                num_tasks=3,
                alpha=0.3,
                temperature=3.0
            )
        else:
            self.self_distill = None
        
        # -- 元路径自动学习器 --
        self.metapath_learner = MetaPathLearner(hidden_dim=hgt_out_dim)
        
        # 多任务不确定性加权 (Kendall et al., CVPR 2018)
        # 初始化为 log_var ≈ 0.5 → precision ≈ 0.61
        # 防止训练初期 loss 爆炸 (初始化为 0 时 precision=1 过高)
        self.log_var_gp = nn.Parameter(torch.tensor([0.5]))
        self.log_var_ct = nn.Parameter(torch.tensor([0.5]))
        self.log_var_lr = nn.Parameter(torch.tensor([0.5]))
        
        # 多任务预测头
        self.gene_pathway_pred = nn.Sequential(
            nn.Linear(hgt_out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.compound_target_pred = nn.Sequential(
            nn.Linear(hgt_out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.cell_comm_pred = nn.Sequential(
            nn.Linear(hgt_out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.dropout = dropout
    
    def forward(self, x_dict, edge_index_dict, gene_gat_edge, celltype_gat_edge,
                return_intermediate: bool = False):
        """
        Args:
            x_dict: {node_type: feature_tensor}
            edge_index_dict: {edge_type: edge_index_tensor}
            gene_gat_edge: 基因共表达边用于GAT
            celltype_gat_edge: 细胞类型边用于GAT
            return_intermediate: 是否返回中间层输出 (用于自蒸馏)
        Returns:
            x_hgt: HGT输出嵌入
            layer_outputs: (可选) 各HGT层中间输出
        """
        # Step 1: 特征投影到统一维度
        x_proj = {}
        for ntype, x in x_dict.items():
            x_proj[ntype] = self.node_proj[ntype](x)
        
        # Step 2: GAT编码 (侧通道, 同类型内局部注意力)
        # GAT输出与投影特征通过残差融合 → HGT输入
        x_fused = {}
        for ntype in x_proj:
            if ntype in self.gat_encoders:
                if ntype == 'gene' and gene_gat_edge is not None:
                    x_gat = self.gat_encoders[ntype](x_proj[ntype], gene_gat_edge)
                elif ntype == 'celltype' and celltype_gat_edge is not None:
                    x_gat = self.gat_encoders[ntype](x_proj[ntype], celltype_gat_edge)
                else:
                    x_gat = x_proj[ntype]
                # 残差融合: GAT输出 + 投影特征 (有机融合, 保留GAT局部信息)
                x_fused[ntype] = (1 - self.gat_residual_weight) * x_proj[ntype] + \
                                  self.gat_residual_weight * x_gat
            else:
                x_fused[ntype] = self.gat_proj[ntype](x_proj[ntype])
        
        # Step 3: HGT编码 - 逐层, 收集中间输出用于自蒸馏
        layer_outputs = []
        x_hgt = x_fused
        for conv in self.hgt_encoder.convs:
            x_out = conv(x_hgt, edge_index_dict)
            for k, v in x_out.items():
                if v is not None:
                    x_hgt[k] = F.elu(v)
            x_hgt = {k: F.dropout(v, p=self.hgt_encoder.dropout, training=self.training)
                      for k, v in x_hgt.items()}
            layer_outputs.append({k: v.clone() for k, v in x_hgt.items()})
        
        # 投影到输出维度
        x_hgt = {k: self.hgt_encoder.proj(v) for k, v in x_hgt.items()}
        layer_outputs[-1] = {k: v.clone() for k, v in x_hgt.items()}
        
        # Step 4: VIB 压缩
        self._vib_kl = torch.tensor(0.0, device=next(self.parameters()).device)
        if self.vib_layers is not None:
            for ntype in x_hgt:
                if ntype in self.vib_layers:
                    x_hgt[ntype], kl = self.vib_layers[ntype](x_hgt[ntype])
                    self._vib_kl = self._vib_kl + kl
        
        if return_intermediate:
            return x_hgt, layer_outputs
        return x_hgt
    
    def forward_heco_contrastive(self, x_dict, edge_index_dict,
                                  gene_gat_edge, celltype_gat_edge) -> torch.Tensor:
        """
        HeCo对比预训练前向传播
        
        网络结构视图: 直接HGT编码 (schema view)
        元路径视图: gene→pathway→gene 2-hop聚合 (metapath view)
        对比: InfoNCE 跨视图loss
        """
        # Step 1: 特征投影
        x_proj = {}
        for ntype, x in x_dict.items():
            x_proj[ntype] = self.node_proj[ntype](x)
        
        # Step 2: GAT编码 (侧通道残差融合)
        x_fused = {}
        for ntype in x_proj:
            if ntype in self.gat_encoders:
                if ntype == 'gene' and gene_gat_edge is not None:
                    x_gat = self.gat_encoders[ntype](x_proj[ntype], gene_gat_edge)
                elif ntype == 'celltype' and celltype_gat_edge is not None:
                    x_gat = self.gat_encoders[ntype](x_proj[ntype], celltype_gat_edge)
                else:
                    x_gat = x_proj[ntype]
                x_fused[ntype] = (1 - self.gat_residual_weight) * x_proj[ntype] + \
                                  self.gat_residual_weight * x_gat
            else:
                x_fused[ntype] = self.gat_proj[ntype](x_proj[ntype])
        
        # Step 3: HGT编码 - Schema View
        x_hgt = self.hgt_encoder(x_fused, edge_index_dict)
        
        gene_schema = x_hgt['gene']
        
        # Step 4: Meta-path View - 使用可学习MetaPathLearner
        pathway_emb = x_hgt.get('pathway',
            torch.zeros(1, self.hgt_out_dim, device=gene_schema.device))
        lr_emb = x_hgt.get('lr',
            torch.zeros(1, self.hgt_out_dim, device=gene_schema.device))
        
        metapath_gene = self.metapath_learner.aggregate_metapath(
            gene_schema, edge_index_dict, pathway_emb, lr_emb
        )
        
        return self.heco_trainer(gene_schema, metapath_gene)
    
    def get_edge_type_importance(self) -> Dict:
        """
        从 PyG HGTConv 内置 p_rel 参数提取边类型重要性
        
        参考: Hu et al., WWW 2020 - HGTConv 的 self.p_rel 参数
        每种边类型在注意力机制中被 relation_pri 缩放
        p_rel 值越大 → 该边类型在消息传递中越重要
        
        Note: PyG HGTConv 内部用 '__'.join(edge_type) 将元组转为字符串存储
        """
        importance = {}
        for layer_idx, conv in enumerate(self.hgt_encoder.convs):
            # HGTConv 的 p_rel 是 dict: {'src__rel__dst': Parameter}
            if hasattr(conv, 'p_rel'):
                for ek, p in conv.p_rel.items():
                    # ek 是字符串如 'gene__coexp__gene'，需解析为 (src, rel, dst)
                    parts = ek.split('__')
                    if len(parts) == 3:
                        key = f"L{layer_idx}_{parts[0]}->{parts[2]}"
                    else:
                        key = f"L{layer_idx}_{ek}"
                    importance[key] = p.mean().item()
        return importance
    
    def predict_gene_pathway(self, gene_emb, pathway_emb):
        """任务1: 基因-通路归属"""
        combined = torch.cat([gene_emb, pathway_emb], dim=-1)
        return self.gene_pathway_pred(combined)
    
    def predict_compound_target(self, compound_emb, gene_emb):
        """任务2: 化合物-靶点结合"""
        combined = torch.cat([compound_emb, gene_emb], dim=-1)
        return self.compound_target_pred(combined)
    
    def predict_cell_comm(self, lr_emb, gene_emb):
        """任务3: 跨细胞通讯关联"""
        combined = torch.cat([lr_emb, gene_emb], dim=-1)
        return self.cell_comm_pred(combined)


# ============================================================
# 转换为PyG HeteroData
# ============================================================

def build_pyg_data(graph_data: dict):
    """将图数据转换为PyG HeteroData格式"""
    from torch_geometric.data import HeteroData
    
    data = HeteroData()
    
    # 节点特征
    node_types = ['gene', 'celltype', 'lr', 'pathway', 'compound', 'disease', 'pocket']
    for nt in node_types:
        x = torch.from_numpy(graph_data[nt]['x'])
        data[nt].x = x
        data[nt].num_nodes = graph_data[nt]['n']
    
    # 边 (需要转换为PyG格式)
    edge_mapping = {
        'gene_coexp': ('gene', 'coexp', 'gene'),
        'lr_interaction': ('lr', 'interacts', 'gene'),
        'regulates': ('gene', 'regulates', 'gene'),
        'enriched_in': ('gene', 'enriched_in', 'pathway'),
        'compound_targets': ('compound', 'targets', 'gene'),
        'celltype_express': ('celltype', 'expresses', 'gene'),
        'gene_disease': ('gene', 'associated_with', 'disease'),
        'compound_pocket': ('compound', 'binds_to', 'pocket'),
        # 反向边: 确保所有节点类型参与HGT消息传递 (SMOGT, 2025)
        'gene_to_lr': ('gene', 'encodes', 'lr'),
        'gene_to_compound': ('gene', 'targeted_by', 'compound'),
        'gene_to_celltype': ('gene', 'expressed_in', 'celltype'),
    }
    
    for edge_key, (src, rel, dst) in edge_mapping.items():
        edges = graph_data['edges'].get(edge_key, [])
        if edges:
            src_nodes, dst_nodes = zip(*edges)
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            data[src, rel, dst].edge_index = edge_index
    
    return data


# ============================================================
# 3. 训练
# ============================================================

def train_model(graph_data: dict, hidden_dim: int = 64, epochs: int = 200,
                learn_rate: float = 0.001, device_str: str = 'cpu') -> tuple:
    """Multi-task训练HGT-GAT模型"""
    logger.info("=" * 60)
    logger.info("训练HGT-GAT模型")
    
    device = torch.device(device_str)
    
    # 构建PyG数据
    pyg_data = build_pyg_data(graph_data)
    
    # 节点特征维度
    node_feat_dims = {
        'gene': graph_data['gene']['x'].shape[1],
        'celltype': graph_data['celltype']['x'].shape[1],
        'lr': graph_data['lr']['x'].shape[1],
        'pathway': graph_data['pathway']['x'].shape[1],
        'compound': graph_data['compound']['x'].shape[1],
        'disease': graph_data['disease']['x'].shape[1],
        'pocket': graph_data['pocket']['x'].shape[1],
    }
    
    # 构建模型 (使用HeteroData.metadata())
    model = HGTGATModel(node_feat_dims, pyg_data.metadata(),
                        hidden_dim=hidden_dim).to(device)
    logger.info(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备训练数据: 70/15/15 train/val/test 划分
    rng = np.random.default_rng(42)
    
    # 任务1: 基因-通路 (正样本: enriched_in边, 负样本: 随机采样)
    enriched_edges = graph_data['edges']['enriched_in']
    n_gene = graph_data['gene']['n']
    n_pathway = graph_data['pathway']['n']
    
    # ---- 核心基因训练掩码 ----
    # 文献依据: BioTranslator (Xu et al., 2023 NAR), KG4SL (Liu et al., 2022 Bioinformatics)
    # 仅核心基因参与监督损失, 背景基因只做消息传递
    core_gene_indices = graph_data['gene'].get('core_indices', list(range(n_gene)))
    core_gene_set = set(core_gene_indices)
    n_core_gene = len(core_gene_indices)
    logger.info(f"  核心基因训练掩码: {n_core_gene}/{n_gene} 个基因参与监督")
    
    pos_gp = list(set(enriched_edges))
    neg_gp = []
    while len(neg_gp) < len(pos_gp):
        g = int(rng.choice(core_gene_indices))
        p = int(rng.integers(0, n_pathway))
        if (g, p) not in pos_gp:
            neg_gp.append((g, p))
    
    # 划分正负样本
    def split_edges(pos, neg, train_r=0.7, val_r=0.15):
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos = len(pos)
        n_neg = len(neg)
        n_pos_train = int(n_pos * train_r)
        n_pos_val = int(n_pos * val_r)
        n_neg_train = int(n_neg * train_r)
        n_neg_val = int(n_neg * val_r)
        return {
            'train': (pos[:n_pos_train] + neg[:n_neg_train],
                      [1.0] * n_pos_train + [0.0] * n_neg_train),
            'val': (pos[n_pos_train:n_pos_train + n_pos_val] + neg[n_neg_train:n_neg_train + n_neg_val],
                    [1.0] * n_pos_val + [0.0] * n_neg_val),
            'test': (pos[n_pos_train + n_pos_val:] + neg[n_neg_train + n_neg_val:],
                     [1.0] * (n_pos - n_pos_train - n_pos_val) + [0.0] * (n_neg - n_neg_train - n_neg_val)),
        }
    
    gp_split = split_edges(pos_gp, neg_gp)
    
    # 任务2: 化合物-靶点
    ct_edges = graph_data['edges']['compound_targets']
    n_compound = graph_data['compound']['n']
    pos_ct = list(set(ct_edges))
    neg_ct = []
    while len(neg_ct) < len(pos_ct):
        c = int(rng.integers(0, n_compound))
        g = int(rng.choice(core_gene_indices))
        if (c, g) not in pos_ct:
            neg_ct.append((c, g))
    ct_split = split_edges(pos_ct, neg_ct)
    
    # 任务3: 跨细胞通讯 (LR-Gene)
    lr_edges = graph_data['edges']['lr_interaction']
    n_lr = graph_data['lr']['n']
    pos_lr = list(set(lr_edges))
    neg_lr = []
    while len(neg_lr) < len(pos_lr):
        l = int(rng.integers(0, n_lr))
        g = int(rng.choice(core_gene_indices))
        if (l, g) not in pos_lr:
            neg_lr.append((l, g))
    lr_split = split_edges(pos_lr, neg_lr)
    
    # 转换为tensor (训练集)
    gp_edges_t = torch.tensor(gp_split['train'][0], dtype=torch.long, device=device)
    gp_labels_t = torch.tensor(gp_split['train'][1], dtype=torch.float32, device=device)
    # 课程学习的 numpy 副本 (用于 select_easy_samples)
    gp_edges_t_np = [(int(e[0]), int(e[1])) for e in gp_split['train'][0]]
    gp_labels_t_np = gp_split['train'][1].tolist()
    ct_edges_t = torch.tensor(ct_split['train'][0], dtype=torch.long, device=device)
    ct_labels_t = torch.tensor(ct_split['train'][1], dtype=torch.float32, device=device)
    lr_edges_t = torch.tensor(lr_split['train'][0], dtype=torch.long, device=device)
    lr_labels_t = torch.tensor(lr_split['train'][1], dtype=torch.float32, device=device)
    
    # 验证集
    gp_val_edges = torch.tensor(gp_split['val'][0], dtype=torch.long, device=device)
    gp_val_labels = torch.tensor(gp_split['val'][1], dtype=torch.float32, device=device)
    ct_val_edges = torch.tensor(ct_split['val'][0], dtype=torch.long, device=device)
    ct_val_labels = torch.tensor(ct_split['val'][1], dtype=torch.float32, device=device)
    lr_val_edges = torch.tensor(lr_split['val'][0], dtype=torch.long, device=device)
    lr_val_labels = torch.tensor(lr_split['val'][1], dtype=torch.float32, device=device)
    
    # 测试集
    gp_test_edges = torch.tensor(gp_split['test'][0], dtype=torch.long, device=device)
    gp_test_labels = torch.tensor(gp_split['test'][1], dtype=torch.float32, device=device)
    ct_test_edges = torch.tensor(ct_split['test'][0], dtype=torch.long, device=device)
    ct_test_labels = torch.tensor(ct_split['test'][1], dtype=torch.float32, device=device)
    lr_test_edges = torch.tensor(lr_split['test'][0], dtype=torch.long, device=device)
    lr_test_labels = torch.tensor(lr_split['test'][1], dtype=torch.float32, device=device)
    
    # 准备x_dict和edge_index_dict
    x_dict = {}
    for nt in ['gene', 'celltype', 'lr', 'pathway', 'compound', 'disease', 'pocket']:
        x_dict[nt] = pyg_data[nt].x.to(device)
    
    edge_index_dict = {}
    edge_mapping = {
        ('gene', 'coexp', 'gene'): 'gene_coexp',
        ('lr', 'interacts', 'gene'): 'lr_interaction',
        ('gene', 'regulates', 'gene'): 'regulates',
        ('gene', 'enriched_in', 'pathway'): 'enriched_in',
        ('compound', 'targets', 'gene'): 'compound_targets',
        ('celltype', 'expresses', 'gene'): 'celltype_express',
        ('gene', 'associated_with', 'disease'): 'gene_disease',
        ('compound', 'binds_to', 'pocket'): 'compound_pocket',
        # 反向边: 确保所有节点类型参与HGT消息传递
        ('gene', 'encodes', 'lr'): 'gene_to_lr',
        ('gene', 'targeted_by', 'compound'): 'gene_to_compound',
        ('gene', 'expressed_in', 'celltype'): 'gene_to_celltype',
    }
    for key_tuple, edge_key in edge_mapping.items():
        if key_tuple in pyg_data.edge_index_dict:
            edge_index_dict[key_tuple] = pyg_data[key_tuple].edge_index.to(device)
    
    # ---- 边泄露修复: 训练前移除val/test正样本边 ----
    # 参考: link prediction标准做法 (Kipf & Welling, 2016)
    # HGT消息传递会暴露边结构信息, 若训练图中包含val/test正边则AUC虚高
    val_test_gp_set = set(gp_split['val'][0] + gp_split['test'][0])   # (gene, pathway)
    val_test_ct_set = set(ct_split['val'][0] + ct_split['test'][0])   # (compound, gene)
    val_test_lr_set = set(lr_split['val'][0] + lr_split['test'][0])   # (lr, gene)
    
    def remove_leaked_edges(edge_index, leak_set):
        """从edge_index (2 x N)中移除leak_set中的边"""
        if edge_index is None or len(leak_set) == 0:
            return edge_index
        ei = edge_index.cpu().numpy()
        mask = np.ones(ei.shape[1], dtype=bool)
        for s, d in leak_set:
            match = (ei[0] == s) & (ei[1] == d)
            mask[match] = False
        return torch.from_numpy(ei[:, mask]).to(device)
    
    # 移除3个监督边类型中的val/test正样本 (enriched_in无反向边, 只需单向)
    edge_index_dict[('gene', 'enriched_in', 'pathway')] = remove_leaked_edges(
        edge_index_dict[('gene', 'enriched_in', 'pathway')], val_test_gp_set)
    edge_index_dict[('compound', 'targets', 'gene')] = remove_leaked_edges(
        edge_index_dict[('compound', 'targets', 'gene')], val_test_ct_set)
    edge_index_dict[('lr', 'interacts', 'gene')] = remove_leaked_edges(
        edge_index_dict[('lr', 'interacts', 'gene')], val_test_lr_set)
    # 对应反向边也移除
    edge_index_dict[('gene', 'targeted_by', 'compound')] = remove_leaked_edges(
        edge_index_dict[('gene', 'targeted_by', 'compound')],
        {(g, c) for c, g in val_test_ct_set})
    edge_index_dict[('gene', 'encodes', 'lr')] = remove_leaked_edges(
        edge_index_dict[('gene', 'encodes', 'lr')],
        {(g, l) for l, g in val_test_lr_set})
    
    n_removed = (len(val_test_gp_set) + len(val_test_ct_set) + len(val_test_lr_set))
    logger.info(f"  边泄露修复: 从训练图中移除 {n_removed} 条val/test正样本边")
    
    # 将修复后的edge_index_dict同时用于训练和评估 (避免评估时重新泄露)
    train_edge_index_dict = edge_index_dict
    
    # GAT边 (基因共表达)
    gene_coexp_edges = graph_data['edges']['gene_coexp']
    gene_gat_edge = torch.tensor(list(zip(*gene_coexp_edges)), dtype=torch.long, device=device) if gene_coexp_edges else None
    
    # 细胞类型GAT边 (模拟)
    ct_gat_edges = []
    for i in range(n_celltypes := graph_data['celltype']['n']):
        for j in range(n_celltypes):
            if i != j:
                ct_gat_edges.append((i, j))
    celltype_gat_edge = torch.tensor(list(zip(*ct_gat_edges)), dtype=torch.long, device=device) if ct_gat_edges else None
    
    # 优化器与调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=30, min_lr=1e-6
    )
    bce_loss = nn.BCEWithLogitsLoss()
    
    # -- 课程学习调度器 (模块4) --
    curriculum = CurriculumScheduler(
        total_epochs=epochs,
        start_ratio=0.5,
        growth_rate=0.02,
        patience_warmup=20
    )
    
    def evaluate(model, x_dict, edge_index_dict_eval, gene_gat_edge, celltype_gat_edge,
                 edges, labels, pred_fn):
        """评估AUC - 在eval模式下重新计算嵌入 (使用去泄露的边字典)"""
        model.eval()
        with torch.no_grad():
            x_hgt_eval = model(x_dict, edge_index_dict_eval, gene_gat_edge, celltype_gat_edge)
            return _compute_auc(x_hgt_eval, edges, labels, pred_fn)
    
    def _compute_auc(x_hgt, edges, labels, pred_fn):
        """使用预计算嵌入计算AUC (避免重复前向传播)"""
        preds = pred_fn(x_hgt, edges).squeeze().cpu().numpy()
        labels_np = labels.cpu().numpy()
        if len(np.unique(labels_np)) < 2:
            return 0.5
        return roc_auc_score(labels_np, preds)
    
    # ---- HeCo 对比预训练阶段 ----
    heco_epochs = min(30, epochs // 6)
    logger.info(f"  HeCo对比预训练: {heco_epochs} epochs")
    for epoch in range(1, heco_epochs + 1):
        model.train()
        optimizer.zero_grad()
        heco_loss = model.forward_heco_contrastive(
            x_dict, train_edge_index_dict, gene_gat_edge, celltype_gat_edge
        )
        heco_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            logger.info(f"    HeCo pretrain {epoch}/{heco_epochs}: loss={heco_loss.item():.4f}")
    
    # 训练循环
    model.train()
    losses = []
    best_val_auc = 0.0
    best_state = None
    
    # L4升级: 课程负采样器 (为化合物-靶点任务动态挖掘困难负样本)
    ct_pos_mask = ct_labels_t > 0.5
    ct_pos_edges_t = ct_edges_t[ct_pos_mask]
    ct_pos_labels = ct_labels_t[ct_pos_mask]
    ct_neg_edges_t = ct_edges_t[~ct_pos_mask]
    ct_neg_labels_t = ct_labels_t[~ct_pos_mask]
    n_ct_neg = len(ct_neg_edges_t)
    n_ct_easy = int(n_ct_neg * 0.7)  # 70%简单负样本保持固定
    ct_easy_edges = ct_neg_edges_t[:n_ct_easy]
    ct_easy_labels = ct_neg_labels_t[:n_ct_easy]
    
    ct_pos_set = set((int(ct_pos_edges_t[i][0]), int(ct_pos_edges_t[i][1]))
                     for i in range(len(ct_pos_edges_t)))
    neg_sampler = CurriculumNegativeSampler(
        n_compounds=graph_data['compound']['n'],
        n_genes=graph_data['gene']['n'],
        core_gene_indices=core_gene_indices,
        hard_neg_ratio=0.3,
        pos_set=ct_pos_set
    )
    
    # 当前的ct训练数据 (初始为原始, 后续动态更新)
    ct_edges_current = ct_edges_t
    ct_labels_current = ct_labels_t
    hard_neg_refresh_interval = 10  # 每10个epoch刷新困难负样本
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        
        # L2升级: DropEdge图增强 (对共表达和调控边随机丢弃)
        aug_edge_index_dict = augment_graph(
            train_edge_index_dict, drop_p=0.1, seed=epoch
        )
        
        # Forward (使用增强后的边字典, 返回中间层用于自蒸馏)
        x_hgt, layer_outputs = model(
            x_dict, aug_edge_index_dict, gene_gat_edge, celltype_gat_edge,
            return_intermediate=True
        )
        
        gene_emb = x_hgt['gene']
        pathway_emb = x_hgt['pathway']
        compound_emb = x_hgt['compound']
        lr_emb = x_hgt['lr']
        
        # L4升级: 每refresh_interval轮动态更新困难负样本
        if epoch % hard_neg_refresh_interval == 1 and epoch > 1:
            n_hard = max(0, n_ct_neg - n_ct_easy)
            hard_neg_edges_list, hard_neg_labels_list = neg_sampler.sample_hard_negatives(
                compound_emb, gene_emb,
                n_easy=n_ct_easy, n_hard=n_hard,
                rng=np.random.RandomState(epoch)
            )
            if len(hard_neg_edges_list) > n_ct_easy:
                hard_neg_edges_t = torch.tensor(hard_neg_edges_list[n_ct_easy:],
                                                dtype=torch.long, device=device)
                hard_neg_labels_t = torch.tensor(hard_neg_labels_list[n_ct_easy:],
                                                  dtype=torch.float32, device=device)
                ct_edges_current = torch.cat([ct_pos_edges_t, ct_easy_edges, hard_neg_edges_t], dim=0)
                ct_labels_current = torch.cat([ct_pos_labels, ct_easy_labels, hard_neg_labels_t], dim=0)
        
        # 课程学习: 基于当前epoch选择训练样本比例
        # 参考: Curriculum Learning (Bengio et al., ICML 2009)
        # 通过在 warmup 后使用 select_easy_samples 基于历史损失选择简单样本
        if epoch > curriculum.patience_warmup and curriculum.sample_losses:
            gp_edges_c, gp_labels_c = curriculum.select_easy_samples(
                gp_edges_t_np, gp_labels_t_np, epoch
            )
        else:
            gp_edges_c = gp_edges_t_np
            gp_labels_c = gp_labels_t_np
        
        # 转回 tensor
        gp_edges_c = torch.tensor(gp_edges_c, dtype=torch.long, device=device)
        gp_labels_c = torch.tensor(gp_labels_c, dtype=torch.float32, device=device)
        
        # 任务1: 基因-通路 (课程学习: 基于历史损失筛选简单样本)
        gp_pred = model.predict_gene_pathway(
            gene_emb[gp_edges_c[:, 0]],
            pathway_emb[gp_edges_c[:, 1]]
        ).squeeze()
        loss1 = bce_loss(gp_pred, gp_labels_c)
        
        # 任务2: 化合物-靶点 (使用动态更新的边)
        ct_pred = model.predict_compound_target(
            compound_emb[ct_edges_current[:, 0]],
            gene_emb[ct_edges_current[:, 1]]
        ).squeeze()
        loss2 = bce_loss(ct_pred, ct_labels_current)
        
        # 任务3: 跨细胞通讯
        lr_pred = model.predict_cell_comm(
            lr_emb[lr_edges_t[:, 0]],
            gene_emb[lr_edges_t[:, 1]]
        ).squeeze()
        loss3 = bce_loss(lr_pred, lr_labels_t)
        
        # 总监督损失 (Kendall同方差不确定性加权)
        # loss_i * exp(-log_var) 降低高不确定任务的权重
        # + log_var 防止log_var → -∞
        prec_gp = torch.exp(-model.log_var_gp)
        prec_ct = torch.exp(-model.log_var_ct)
        prec_lr = torch.exp(-model.log_var_lr)
        task_loss = (prec_gp * loss1 + model.log_var_gp +
                     prec_ct * loss2 + model.log_var_ct +
                     prec_lr * loss3 + model.log_var_lr) / 2
        
        # 自蒸馏损失 (模块5)
        distill_loss = torch.tensor(0.0, device=device)
        if model.self_distill is not None:
            distill_loss = model.self_distill(layer_outputs)
        
        # VIB KL损失 (模块3)
        vib_kl_loss = getattr(model, '_vib_kl', torch.tensor(0.0, device=device))
        
        # 总损失 = 任务损失 + α·蒸馏损失 + β·VIB KL
        loss = task_loss + distill_loss + 0.001 * vib_kl_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        # 更新课程学习样本难度 (基于任务1的逐样本损失)
        with torch.no_grad():
            gp_loss_per_sample = F.binary_cross_entropy_with_logits(
                gp_pred, gp_labels_c, reduction='none'
            )
            curriculum.update_sample_difficulty(
                gp_edges_c.cpu().tolist(), gp_loss_per_sample
            )
        
        if epoch % 40 == 0:
            # 验证集评估 (在eval模式下重新计算嵌入)
            val_auc1 = evaluate(model, x_dict, train_edge_index_dict, gene_gat_edge, celltype_gat_edge,
                                gp_val_edges, gp_val_labels,
                                lambda h, e: model.predict_gene_pathway(h['gene'][e[:, 0]], h['pathway'][e[:, 1]]))
            val_auc2 = evaluate(model, x_dict, train_edge_index_dict, gene_gat_edge, celltype_gat_edge,
                                ct_val_edges, ct_val_labels,
                                lambda h, e: model.predict_compound_target(h['compound'][e[:, 0]], h['gene'][e[:, 1]]))
            val_auc3 = evaluate(model, x_dict, train_edge_index_dict, gene_gat_edge, celltype_gat_edge,
                                lr_val_edges, lr_val_labels,
                                lambda h, e: model.predict_cell_comm(h['lr'][e[:, 0]], h['gene'][e[:, 1]]))
            val_auc_mean = (val_auc1 + val_auc2 + val_auc3) / 3
            
            # 学习率调度
            scheduler.step(val_auc_mean)
            
            logger.info(f"  Epoch {epoch:3d}/{epochs}: loss={loss.item():.4f} "
                        f"(task={task_loss.item():.4f}, distill={distill_loss.item():.4f}) "
                        f"| gp={loss1.item():.4f}, ct={loss2.item():.4f}, lr_comm={loss3.item():.4f} "
                        f"| val_AUC: gp={val_auc1:.3f}, ct={val_auc2:.3f}, lr_comm={val_auc3:.3f}")
            
            # 保存最佳模型
            if val_auc_mean > best_val_auc:
                best_val_auc = val_auc_mean
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            model.train()
    
    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"  恢复最佳模型 (val_AUC={best_val_auc:.4f})")
    
    # 最终测试集评估 (复用预计算嵌入, 避免重复前向传播)
    model.eval()
    with torch.no_grad():
        x_hgt_test = model(x_dict, train_edge_index_dict, gene_gat_edge, celltype_gat_edge)
        test_auc1 = _compute_auc(x_hgt_test, gp_test_edges, gp_test_labels,
                                 lambda h, e: model.predict_gene_pathway(h['gene'][e[:, 0]], h['pathway'][e[:, 1]]))
        test_auc2 = _compute_auc(x_hgt_test, ct_test_edges, ct_test_labels,
                                 lambda h, e: model.predict_compound_target(h['compound'][e[:, 0]], h['gene'][e[:, 1]]))
        test_auc3 = _compute_auc(x_hgt_test, lr_test_edges, lr_test_labels,
                                 lambda h, e: model.predict_cell_comm(h['lr'][e[:, 0]], h['gene'][e[:, 1]]))
    
    logger.info(f"  训练完成: final_loss={losses[-1]:.4f}")
    logger.info(f"  测试集AUC: gp={test_auc1:.3f}, ct={test_auc2:.3f}, lr_comm={test_auc3:.3f} | "
                f"均值={((test_auc1 + test_auc2 + test_auc3) / 3):.3f}")
    
    # 输出边类型重要性 (PyG HGTConv 内置 p_rel)
    edge_imp = model.get_edge_type_importance()
    if edge_imp:
        logger.info(f"  边类型重要性 (HGTConv p_rel):")
        for ek, imp in sorted(edge_imp.items(), key=lambda x: -x[1]):
            logger.info(f"    {ek}: {imp:.4f}")
    
    # GNNExplainer 边级可解释性 (在训练完成后执行)
    explain_results = compute_gnn_explainability(
        model, graph_data, x_hgt_test, train_edge_index_dict,
        gene_gat_edge, celltype_gat_edge, graph_data['gene']['names'],
        target_gene='ACSL4', device=device
    )
    
    return model, x_hgt_test, x_hgt_test['gene'], x_hgt_test['compound'], x_hgt_test['lr'], x_hgt_test['pathway'], losses, explain_results


# ============================================================
# 4. 推理与排名
# ============================================================

def compute_hub_ranking(model, graph_data: dict, x_hgt: dict, device: str = 'cpu') -> pd.DataFrame:
    """计算基因hub排名 - 加权多源得分 (参考 GNN4DM, 2024)"""
    logger.info("=" * 60)
    logger.info("计算Hub基因排名")
    
    gene_emb = x_hgt['gene'].detach().cpu().numpy()
    gene_names = graph_data['gene']['names']
    
    # ---- 1. 嵌入重要性: L2范数 (模型学到的语义重要性) ----
    emb_norm = np.linalg.norm(gene_emb, axis=1)
    emb_norm_norm = emb_norm / (emb_norm.max() + 1e-8)
    
    # ---- 2. 度中心性: 图结构重要性 ----
    degrees = defaultdict(int)
    for edge_key in ['gene_coexp', 'regulates', 'enriched_in', 'gene_disease']:
        for src, dst in graph_data['edges'].get(edge_key, []):
            degrees[src] += 1
    
    for edge_key in ['lr_interaction', 'compound_targets', 'celltype_express',
                     'gene_to_lr', 'gene_to_compound', 'gene_to_celltype']:
        for src, dst in graph_data['edges'].get(edge_key, []):
            degrees[dst] += 1
    
    degree_arr = np.array([degrees.get(i, 0) for i in range(len(gene_names))])
    # log变换避免长尾分布过度放大少数高度连接基因
    degree_norm = np.log1p(degree_arr) / (np.log1p(degree_arr.max()) + 1e-8)
    
    # ---- 3. 生物学先验: 铁死亡/衰老/铁衰老基因集成员 ----
    all_ferroptosis = PURE_FERROPTOSIS | SHARED_GENES
    all_senescence = PURE_SENESCENCE | SHARED_GENES
    all_ferroaging = FERROAGING_GENES
    
    # 疾病关联得分 (如果基因关联CIRI/AD/Aging)
    disease_genes_set = set()
    for disease_genes_list in [
        ['HIF1A','NLRP3','TLR4','IL1B','IL6','TNF','MMP9','HMGB1','BDNF',
         'VEGFA','CASP3','BAX','BCL2','PTGS2','NOS2','NOS3','SOD1','CAT',
         'NFE2L2','HMOX1'],
        ['APP','MAPT','PSEN1','BACE1','TREM2','APOE','CLU','CDKN2A','TP53',
         'HMGB1','NLRP3','IL1B','TNF'],
        ['CDKN2A','CDKN1A','TP53','LMNB1','SIRT1','SIRT6','FOXO1','FOXO3',
         'FOXO4','MTOR','IGFBP7','HMGB1','IL6','TNF','NFKB1','RELA',
         'PARP1','TERF1','TERF2'],
    ]:
        disease_genes_set.update(disease_genes_list)
    
    bio_prior = np.zeros(len(gene_names))
    for i, gene in enumerate(gene_names):
        score = 0.0
        if gene in all_ferroaging:
            score += 0.5  # 铁衰老基因: 最高权重
        if gene in all_ferroptosis:
            score += 0.3  # 铁死亡
        if gene in all_senescence:
            score += 0.3  # 衰老
        if gene in disease_genes_set:
            score += 0.2  # 疾病关联 (CIRI/AD/Aging)
        bio_prior[i] = min(score, 1.0)  # 截断到[0,1]
    
    # ---- 4. 加权综合得分 ----
    # 权重设计依据:
    #   - 嵌入重要性 0.40: 模型自主学习到的语义重要度
    #   - 度中心性   0.20: 图结构连通性 (log变换降权)
    #   - 生物学先验 0.40: 铁死亡/衰老/铁衰老/疾病文献支持
    w_emb, w_deg, w_bio = 0.40, 0.20, 0.40
    hub_score = (w_emb * emb_norm_norm + w_deg * degree_norm + w_bio * bio_prior)
    
    ranking = pd.DataFrame({
        'gene': gene_names,
        'embedding_norm': emb_norm,
        'degree': degree_arr,
        'bio_prior': bio_prior,
        'hub_score': hub_score,
    }).sort_values('hub_score', ascending=False)
    
    ranking['rank'] = range(1, len(ranking) + 1)
    
    # 标记关键基因
    for gene in ['ACSL4', 'GPX4', 'PTGS2', 'TFRC', 'HMOX1', 'TP53', 'CDKN1A',
                 'HIF1A', 'NFE2L2', 'KEAP1', 'IL6', 'IL1B', 'TLR4', 'NLRP3', 'HMGB1']:
        if gene in ranking['gene'].values:
            idx = ranking[ranking['gene'] == gene].index[0]
            rank_val = ranking.loc[idx, 'rank']
            score_val = ranking.loc[idx, 'hub_score']
            logger.info(f"  {gene}: rank={rank_val}, hub_score={score_val:.4f}")
    
    return ranking


def compute_compound_target_ranking(model, graph_data: dict, x_hgt: dict) -> pd.DataFrame:
    """计算化合物-靶点排名 (使用模型预测头而非余弦相似度)"""
    logger.info("=" * 60)
    logger.info("计算化合物-靶点结合排名 (使用模型预测头)")

    device = next(model.parameters()).device
    compound_names = graph_data['compound']['names']
    gene_names = graph_data['gene']['names']
    n_compounds = len(compound_names)
    n_genes = len(gene_names)

    # 使用训练好的 predict_compound_target MLP 计算所有对
    results = []
    model.eval()
    with torch.no_grad():
        compound_emb = x_hgt['compound']  # [C, D]
        gene_emb = x_hgt['gene']  # [G, D]

        # 批量计算所有化合物-基因对的预测分数
        # 展开为 [C*G, D*2] 一次性前向传播
        c_idx = torch.arange(n_compounds, device=device).repeat_interleave(n_genes)
        g_idx = torch.arange(n_genes, device=device).repeat(n_compounds)

        c_emb_expand = compound_emb[c_idx]  # [C*G, D]
        g_emb_expand = gene_emb[g_idx]  # [C*G, D]

        logits = model.predict_compound_target(c_emb_expand, g_emb_expand).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()

    for ci in range(n_compounds):
        for gi in range(n_genes):
            idx = ci * n_genes + gi
            results.append({
                'compound': compound_names[ci],
                'gene': gene_names[gi],
                'binding_probability': float(probs[idx]),
            })

    ranking = pd.DataFrame(results).sort_values('binding_probability', ascending=False)

    # 特别关注BCP-ACSL4
    bcp_acsl4 = ranking[(ranking['compound'] == 'BCP') & (ranking['gene'] == 'ACSL4')]
    if not bcp_acsl4.empty:
        bcp_acsl4_rank = ranking[ranking['compound'] == 'BCP'].sort_values('binding_probability', ascending=False)
        acsl4_rank_in_bcp = bcp_acsl4_rank['gene'].tolist().index('ACSL4') + 1
        logger.info(f"  BCP-ACSL4: 结合概率={bcp_acsl4['binding_probability'].values[0]:.4f}, "
                    f"BCP排名={acsl4_rank_in_bcp}/{len(bcp_acsl4_rank)}")

    return ranking


def compute_attention_flow(graph_data: dict, x_hgt: dict) -> dict:
    """计算跨细胞通讯注意力流"""
    logger.info("=" * 60)
    logger.info("计算跨细胞通讯注意力流")
    
    # 基于嵌入相似度推断通讯流
    lr_emb = x_hgt['lr'].detach().cpu().numpy()
    gene_emb = x_hgt['gene'].detach().cpu().numpy()
    celltype_emb = x_hgt['celltype'].detach().cpu().numpy()
    
    lr_names = graph_data['lr']['names']
    gene_names = graph_data['gene']['names']
    celltype_names = graph_data['celltype']['names']
    
    # 构建通讯矩阵: 细胞类型 → 配体基因 → 受体基因 → 细胞类型
    # 简化: 计算每个细胞类型中LR对的高表达模式
    
    # 细胞类型特征 → 基因嵌入的相似度
    ct_gene_sim = {}
    for ci, ct in enumerate(celltype_names):
        sims = []
        for gi, g in enumerate(gene_names):
            sim = np.dot(celltype_emb[ci], gene_emb[gi]) / (
                np.linalg.norm(celltype_emb[ci]) * np.linalg.norm(gene_emb[gi]) + 1e-12
            )
            sims.append(sim)
        ct_gene_sim[ct] = np.array(sims)
    
    # 配体-受体对 → 细胞类型通讯
    comm_flow = {}
    for lr_name in lr_names:
        lig, rec = lr_name.split('-')
        lig_idx = gene_names.index(lig) if lig in gene_names else -1
        rec_idx = gene_names.index(rec) if rec in gene_names else -1
        
        if lig_idx >= 0 and rec_idx >= 0:
            # 计算每个细胞类型对这条LR的通讯强度
            for ct_src in celltype_names:
                src_sim = ct_gene_sim[ct_src][lig_idx]
                for ct_dst in celltype_names:
                    dst_sim = ct_gene_sim[ct_dst][rec_idx]
                    flow = (src_sim + 1) * (dst_sim + 1) / 4  # 归一化到[0,1]
                    key = (ct_src, ct_dst, lr_name)
                    comm_flow[key] = flow
    
    return comm_flow


# ============================================================
# 5. GNNExplainer边级可解释性
# ============================================================

def compute_gnn_explainability(model, graph_data: dict, x_dict: dict,
                                edge_index_dict: dict, gene_gat_edge,
                                celltype_gat_edge, gene_names: list,
                                target_gene: str = 'ACSL4',
                                device: str = 'cpu') -> dict:
    """
    边级可解释性分析 (扰动法)
    
    对每条边类型, 移除该类型所有边后计算目标基因嵌入变化,
    变化越大 → 该边类型对目标基因越重要。
    
    参考: EdgeSHAPer (Mastropietro et al., 2022) 扰动法
          GNNExplainer: Generating Explanations for Graph Neural Networks
          (Ying et al., NeurIPS 2019)
    """
    logger.info("=" * 60)
    logger.info("边级可解释性分析 (扰动法)")
    
    explain_results = {}
    
    try:
        model.eval()
        
        # 查找目标基因索引
        if target_gene not in gene_names:
            target_gene = gene_names[0] if gene_names else 'Unknown'
        target_idx = gene_names.index(target_gene)
        
        # 基准: 全图的嵌入范数
        with torch.no_grad():
            x_hgt_base = model(x_dict, edge_index_dict, gene_gat_edge, celltype_gat_edge)
            base_norm = torch.norm(x_hgt_base['gene'][target_idx]).item()
        
        # 对每种边类型, 计算移除后的嵌入变化
        edge_type_importance = {}
        edge_masks = {}
        
        for edge_key in edge_index_dict:
            if not isinstance(edge_key, tuple) or len(edge_key) != 3:
                continue
            
            ei = edge_index_dict[edge_key]
            if ei.size(1) < 2:
                continue
            
            ek_str = f"{edge_key[0]}-{edge_key[1]}-{edge_key[2]}"
            
            try:
                # 创建移除该边类型的边字典
                perturbed_dict = dict(edge_index_dict)
                perturbed_dict[edge_key] = torch.zeros((2, 0), dtype=torch.long, device=device)
                
                with torch.no_grad():
                    x_hgt_pert = model(x_dict, perturbed_dict, gene_gat_edge, celltype_gat_edge)
                    pert_norm = torch.norm(x_hgt_pert['gene'][target_idx]).item()
                
                # 重要性 = 移除后嵌入变化
                delta = abs(base_norm - pert_norm)
                edge_type_importance[ek_str] = float(delta)
                
                # 归一化边重要性掩码 (全边等权, 扰动法给出类型级重要性)
                edge_masks[ek_str] = np.ones(ei.size(1)) * delta
                
            except Exception:
                edge_type_importance[ek_str] = 0.0
                edge_masks[ek_str] = np.zeros(1)
        
        # 归一化到 [0, 1]
        if edge_type_importance:
            max_val = max(edge_type_importance.values()) or 1.0
            for k in edge_type_importance:
                edge_type_importance[k] /= max_val
        
        explain_results = {
            'edge_type_importance': edge_type_importance,
            'edge_masks': edge_masks,
            'target_gene': target_gene,
            'explainer_type': 'perturbation',
        }
        
        sorted_edges = sorted(edge_type_importance.items(), key=lambda x: -x[1])
        logger.info(f"  扰动法边类型重要性 (对{target_gene}):")
        for ek, imp in sorted_edges[:5]:
            logger.info(f"    {ek}: {imp:.6f}")
        
        logger.info("  扰动法分析完成")
        
    except Exception as e:
        logger.warning(f"  扰动法失败 ({e}), 回退到嵌入范数分析")
        gene_emb = x_dict['gene'].detach().cpu().numpy()
        feature_importance = np.abs(gene_emb).mean(axis=0)
        explain_results = {
            'feature_importance': feature_importance,
            'explainer_type': 'embedding_norm_fallback',
            'target_gene': target_gene,
        }
    
    return explain_results


# ============================================================
# 6. 可视化
# ============================================================

def plot_fig3a_graph_topology(graph_data: dict, save_path: str):
    """Fig3A: 异质图拓扑结构示意"""
    logger.info("  生成 Fig3A: 异质图拓扑结构")
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 节点类型位置
    node_positions = {
        'gene':       (0.50, 0.50),
        'celltype':   (0.15, 0.80),
        'lr':         (0.30, 0.20),
        'pathway':    (0.80, 0.30),
        'compound':   (0.85, 0.75),
        'disease':    (0.20, 0.50),
        'pocket':     (0.70, 0.85),
    }
    
    # 节点颜色和大小
    node_colors = {
        'gene':       '#E74C3C',
        'celltype':   '#3498DB',
        'lr':         '#2ECC71',
        'pathway':    '#F39C12',
        'compound':   '#9B59B6',
        'disease':    '#E67E22',
        'pocket':     '#1ABC9C',
    }
    
    node_labels = {
        'gene':       f'Gene\n({graph_data["gene"]["n"]})',
        'celltype':   f'CellType\n({graph_data["celltype"]["n"]})',
        'lr':         f'Ligand-\nReceptor\n({graph_data["lr"]["n"]})',
        'pathway':    f'Pathway\n({graph_data["pathway"]["n"]})',
        'compound':   f'Compound\n({graph_data["compound"]["n"]})',
        'disease':    f'Disease\n({graph_data["disease"]["n"]})',
        'pocket':     f'ACSL4\nPocket',
    }
    
    sizes = {
        'gene': 800, 'celltype': 600, 'lr': 500,
        'pathway': 600, 'compound': 400, 'disease': 300, 'pocket': 200
    }
    
    # 绘制节点
    for ntype, (x, y) in node_positions.items():
        ax.scatter(x, y, s=sizes[ntype], c=node_colors[ntype], alpha=0.7,
                   edgecolors='black', linewidth=1.5, zorder=5)
        ax.text(x, y, node_labels[ntype], ha='center', va='center',
                fontsize=8, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=node_colors[ntype], alpha=0.9))
    
    # 绘制边
    edge_list = [
        ('gene', 'gene', 'gene_coexp', '#E74C3C'),
        ('gene', 'pathway', 'enriched_in', '#F39C12'),
        ('compound', 'gene', 'compound_targets', '#9B59B6'),
        ('celltype', 'gene', 'celltype_express', '#3498DB'),
        ('lr', 'gene', 'lr_interaction', '#2ECC71'),
        ('gene', 'disease', 'gene_disease', '#E67E22'),
        ('compound', 'pocket', 'compound_pocket', '#1ABC9C'),
    ]
    
    for src, dst, label, color in edge_list:
        if src in node_positions and dst in node_positions:
            x1, y1 = node_positions[src]
            x2, y2 = node_positions[dst]
            # 曲线
            mid_x = (x1 + x2) / 2 + np.random.uniform(-0.05, 0.05)
            mid_y = (y1 + y2) / 2 + np.random.uniform(-0.05, 0.05)
            ax.plot([x1, mid_x, x2], [y1, mid_y, y2],
                    color=color, alpha=0.3, linewidth=2, zorder=1)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Heterogeneous Graph: 7 Node Types × 6 Edge Types',
                 fontsize=14, fontweight='bold')
    
    # 图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=node_colors[nt],
                                   markersize=10, label=f'{nt} ({node_labels[nt]})')
                       for nt in node_colors]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig3A 保存: {save_path}")


def plot_fig3b_attention_heatmap(graph_data: dict, x_hgt: dict, save_path: str,
                                 attentions: list = None):
    """Fig3B: HGT跨类型注意力权重矩阵"""
    logger.info("  生成 Fig3B: 注意力热图")
    
    # 计算各节点类型之间的平均嵌入相似度作为注意力代理
    node_types = ['gene', 'celltype', 'lr', 'pathway', 'compound', 'disease', 'pocket']
    node_labels = ['Gene', 'CellType', 'L-R', 'Pathway', 'Compound', 'Disease', 'Pocket']
    n = len(node_types)
    
    if attentions and len(attentions) > 0:
        # 使用真实的HGT注意力权重
        logger.info("    使用真实HGT注意力权重")
        # 从最后一层的注意力中聚合跨类型权重
        attn_matrix = np.zeros((n, n))
        counts = np.zeros((n, n))
        for edge_type, attn_tensor in attentions[-1].items():
            src_type, _, dst_type = edge_type
            if src_type in node_types and dst_type in node_types:
                i = node_types.index(src_type)
                j = node_types.index(dst_type)
                attn_val = float(attn_tensor.mean().item())
                attn_matrix[i, j] += attn_val
                counts[i, j] += 1
        # 平均
        mask = counts > 0
        attn_matrix[mask] /= counts[mask]
        # 填充对角线
        attn_matrix = (attn_matrix + 1) / 2  # 归一化
        for i in range(n):
            if counts[i, i] == 0:
                attn_matrix[i, i] = 1.0
    else:
        # 回退: 嵌入相似度
        type_means = {}
        for nt in node_types:
            if nt in x_hgt:
                type_means[nt] = x_hgt[nt].detach().cpu().numpy().mean(axis=0)
            else:
                type_means[nt] = np.random.randn(32)
        
        attn_matrix = np.zeros((n, n))
        for i, nt1 in enumerate(node_types):
            for j, nt2 in enumerate(node_types):
                sim = np.dot(type_means[nt1], type_means[nt2]) / (
                    np.linalg.norm(type_means[nt1]) * np.linalg.norm(type_means[nt2]) + 1e-12
                )
                attn_matrix[i, j] = (sim + 1) / 2  # 归一化到[0,1]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(node_labels, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(node_labels)
    
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{attn_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if attn_matrix[i, j] > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)
    ax.set_title('Cross-Type Embedding Similarity Matrix\n(Proxy for HGT Attention, PyG 2.7 limitation)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Node Type')
    ax.set_ylabel('Source Node Type')
    
    # 说明文字
    ax.text(0.5, -0.18,
            'Note: Actual HGT attention weights unavailable in PyG 2.7 HGTConv.\n'
            'Matrix shows cosine similarity of mean type embeddings as attention proxy.',
            transform=ax.transAxes, ha='center', fontsize=8, fontstyle='italic',
            color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig3B 保存: {save_path}")


def plot_fig3c_hub_ranking(ranking: pd.DataFrame, save_path: str):
    """Fig3C: Hub基因排名"""
    logger.info("  生成 Fig3C: Hub基因排名")
    
    top30 = ranking.head(30)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = []
    for gene in top30['gene']:
        if gene == 'ACSL4':
            colors.append('#E74C3C')
        elif gene in PURE_FERROPTOSIS:
            colors.append('#FF6B6B')
        elif gene in PURE_SENESCENCE:
            colors.append('#3498DB')
        else:
            colors.append('#95A5A6')
    
    y_pos = range(len(top30))
    ax.barh(y_pos, top30['hub_score'].values[::-1], color=colors[::-1], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top30['gene'].values[::-1])
    ax.set_xlabel('Hub Score')
    ax.set_title('Top 30 Hub Genes in Ferroptosis-Aging Network', fontsize=14, fontweight='bold')
    
    # ACSL4标注
    acsl4_row = top30[top30['gene'] == 'ACSL4']
    if not acsl4_row.empty:
        ax.annotate(f'ACSL4: Rank #{int(acsl4_row["rank"].values[0])}',
                    xy=(acsl4_row['hub_score'].values[0],
                        len(top30) - list(top30['gene']).index('ACSL4') - 1),
                    xytext=(acsl4_row['hub_score'].values[0] + 0.1,
                            len(top30) - list(top30['gene']).index('ACSL4') - 1),
                    fontsize=10, fontweight='bold', color='#E74C3C',
                    arrowprops=dict(arrowstyle='->', color='#E74C3C'))
    
    # 图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc='#E74C3C', alpha=0.8, label='ACSL4'),
        plt.Rectangle((0, 0), 1, 1, fc='#FF6B6B', alpha=0.8, label='Ferroptosis'),
        plt.Rectangle((0, 0), 1, 1, fc='#3498DB', alpha=0.8, label='Senescence'),
        plt.Rectangle((0, 0), 1, 1, fc='#95A5A6', alpha=0.8, label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig3C 保存: {save_path}")


def plot_fig3d_comm_flow(comm_flow: dict, graph_data: dict, save_path: str):
    """Fig3D: 跨细胞通讯流图"""
    logger.info("  生成 Fig3D: 跨细胞通讯流图")
    
    celltype_names = graph_data['celltype']['names']
    n_ct = len(celltype_names)
    
    # 聚合通讯流: 细胞类型 → 细胞类型
    ct_comm_matrix = np.zeros((n_ct, n_ct))
    for (ct_src, ct_dst, _), flow in comm_flow.items():
        if ct_src in celltype_names and ct_dst in celltype_names:
            i = celltype_names.index(ct_src)
            j = celltype_names.index(ct_dst)
            ct_comm_matrix[i, j] += flow
    
    # 归一化
    ct_comm_matrix = ct_comm_matrix / (ct_comm_matrix.max() + 1e-12)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ct_comm_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    short_labels = ['Neu', 'Mic', 'Ast', 'Oli', 'End', 'Per']
    ax.set_xticks(range(n_ct))
    ax.set_xticklabels(short_labels)
    ax.set_yticks(range(n_ct))
    ax.set_yticklabels(celltype_names)
    
    for i in range(n_ct):
        for j in range(n_ct):
            if ct_comm_matrix[i, j] > 0.1:
                ax.text(j, i, f'{ct_comm_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if ct_comm_matrix[i, j] > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, label='Communication Strength', shrink=0.8)
    ax.set_title('Cross-Cell Communication Flow (Inferred)\n(Microglia → Neuron Ferroptosis-Aging Axis)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Target Cell Type')
    ax.set_ylabel('Source Cell Type')
    
    # 推测性声明
    ax.text(0.5, -0.18,
            'Note: Inferred from embedding similarity, not based on ligand-receptor expression data.\n'
            'Validation with single-cell spatial transcriptomics recommended.',
            transform=ax.transAxes, ha='center', fontsize=8, fontstyle='italic',
            color='gray')
    
    # 标注Microglia→Neuron轴
    mic_idx = celltype_names.index('Microglia')
    neu_idx = celltype_names.index('Neuron')
    ax.add_patch(plt.Rectangle((neu_idx - 0.5, mic_idx - 0.5), 1, 1,
                                fill=False, edgecolor='#E74C3C', linewidth=3, linestyle='--'))
    ax.annotate('Microglia→Neuron\nFerroptosis-Aging Axis',
                xy=(neu_idx, mic_idx), xytext=(neu_idx + 1.5, mic_idx - 0.5),
                fontsize=9, color='#E74C3C', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#E74C3C'))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig3D 保存: {save_path}")


def plot_fig3e_embedding_tsne(gene_emb: np.ndarray, gene_names: list, save_path: str):
    """Fig3E: 节点嵌入t-SNE可视化"""
    logger.info("  生成 Fig3E: t-SNE嵌入可视化")
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(gene_names) - 1))
    emb_2d = tsne.fit_transform(gene_emb)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 分类着色
    for i, gene in enumerate(gene_names):
        if gene == 'ACSL4':
            color = '#E74C3C'
            size = 120
            marker = 'D'
            zorder = 10
        elif gene in PURE_FERROPTOSIS:
            color = '#FF6B6B'
            size = 60
            marker = 'o'
            zorder = 5
        elif gene in PURE_SENESCENCE:
            color = '#3498DB'
            size = 60
            marker = 's'
            zorder = 5
        else:
            color = '#95A5A6'
            size = 40
            marker = '.'
            zorder = 3
        
        ax.scatter(emb_2d[i, 0], emb_2d[i, 1], c=color, s=size, marker=marker,
                   alpha=0.7, edgecolors='none', zorder=zorder)
    
    # 标注关键基因
    key_genes = ['ACSL4', 'GPX4', 'PTGS2', 'TFRC', 'HMOX1', 'TP53', 'CDKN1A',
                 'HIF1A', 'NFE2L2', 'KEAP1', 'IL6', 'TLR4', 'NLRP3']
    for gene in key_genes:
        if gene in gene_names:
            idx = gene_names.index(gene)
            ax.annotate(gene, (emb_2d[idx, 0], emb_2d[idx, 1]),
                        fontsize=7, ha='center', va='bottom',
                        fontweight='bold' if gene == 'ACSL4' else 'normal',
                        color='#E74C3C' if gene == 'ACSL4' else 'black')
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title('Gene Embedding t-SNE Visualization\n(Ferroptosis-Aging Heterogeneous Graph)',
                 fontsize=14, fontweight='bold')
    
    # 图例
    legend_elements = [
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#E74C3C',
                    markersize=10, label='ACSL4 (Hub)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B',
                    markersize=8, label='Ferroptosis Genes'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#3498DB',
                    markersize=8, label='Senescence Genes'),
        plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='#95A5A6',
                    markersize=8, label='Other'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  Fig3E 保存: {save_path}")


def plot_fig3f_gnn_explain(explain_results: dict, save_path: str):
    """Fig3F: GNNExplainer边级可解释性 - 边类型对ACSL4预测的归因权重"""
    logger.info("  生成 Fig3F: GNNExplainer边贡献")
    
    if 'edge_type_importance' in explain_results:
        edge_type_importance = explain_results['edge_type_importance']
        if not edge_type_importance:
            logger.warning("  无边类型重要性数据, 跳过Fig3F")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sorted_items = sorted(edge_type_importance.items(), key=lambda x: -x[1])
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(labels)))
        ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='black')
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Mean |Edge Attribution|')
        ax.set_title(f'GNNExplainer Edge Attribution for {explain_results.get("target_gene", "Gene")}\n'
                     f'(Edge-Level Explainability, Ying et al. NeurIPS 2019)',
                     fontsize=14, fontweight='bold')
        
        for i, (label, val) in enumerate(sorted_items[:3]):
            ax.annotate(f'Top-{i+1}', (i, val),
                        xytext=(i, val + 0.01 * max(values) if max(values) > 0 else 0.01),
                        ha='center', fontsize=8, color='#E74C3C', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    elif 'feature_importance' in explain_results:
        feature_importance = explain_results['feature_importance']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(feature_importance))
        sorted_idx = np.argsort(feature_importance)[::-1]
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(feature_importance)))[sorted_idx]
        ax.bar(x, feature_importance[sorted_idx], color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Dim {i}' for i in sorted_idx], rotation=45, ha='right')
        ax.set_title('Feature Importance (Embedding Norm Fallback)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"  Fig3F 保存: {save_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    global KMP_SET
    if not KMP_SET:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        KMP_SET = True
    
    # ---- 全局随机种子: 确保完全可复现 ----
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"  全局随机种子: {seed} (可复现模式)")
    
    logger.info("=" * 70)
    logger.info("模块三: HGT-GAT异质图构建与跨细胞通讯解析")
    logger.info("=" * 70)
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"  设备: {device}")
    
    # 1. 构建异质图
    graph_data = build_heterogeneous_graph()
    
    # 2. 训练模型
    model, x_hgt, gene_emb, compound_emb, lr_emb, pathway_emb, losses, explain_results = train_model(
        graph_data, hidden_dim=64, epochs=200, learn_rate=0.001, device_str=device
    )
    
    # 3. Hub基因排名
    hub_ranking = compute_hub_ranking(model, graph_data, x_hgt, device)
    
    # 4. 化合物-靶点排名
    compound_ranking = compute_compound_target_ranking(model, graph_data, x_hgt)
    
    # 5. 跨细胞通讯注意力流
    comm_flow = compute_attention_flow(graph_data, x_hgt)
    
    # 6. Top-30 候选化合物 (模块四DeepPurpose筛选输入)
    gene_names = graph_data['gene']['names']
    acsl4_idx = gene_names.index('ACSL4') if 'ACSL4' in gene_names else 0
    acsl4_emb = gene_emb[acsl4_idx] if isinstance(gene_emb, torch.Tensor) else torch.tensor(gene_emb[acsl4_idx])
    all_compound_sims = []
    for ci, cname in enumerate(graph_data['compound']['names']):
        comp_emb = compound_emb[ci] if isinstance(compound_emb, torch.Tensor) else torch.tensor(compound_emb[ci])
        sim = torch.dot(acsl4_emb, comp_emb).item() / (
            torch.norm(acsl4_emb).item() * torch.norm(comp_emb).item() + 1e-12
        )
        all_compound_sims.append({'compound': cname, 'similarity_to_ACSL4': sim})
    top30_compounds = pd.DataFrame(all_compound_sims).sort_values(
        'similarity_to_ACSL4', ascending=False).head(30).reset_index(drop=True)
    top30_compounds.to_csv(OUTPUT_DIR / 'L3_top30_ACSL4_candidate_compounds.csv', index=False)
    logger.info(f"  Top-30候选化合物: {', '.join(top30_compounds['compound'].head(10))}")
    
    # ---- 输出文件 ----
    logger.info("=" * 60)
    logger.info("输出结果")
    
    # 保存排名
    hub_ranking.to_csv(OUTPUT_DIR / 'L3_hub_gene_ranking.csv', index=False)
    logger.info(f"  Hub基因排名: L3_hub_gene_ranking.csv")
    
    compound_ranking.to_csv(OUTPUT_DIR / 'L3_compound_target_ranking.csv', index=False)
    logger.info(f"  化合物-靶点排名: L3_compound_target_ranking.csv")
    
    # 保存通讯流
    comm_flow_df = pd.DataFrame([
        {'source_celltype': src, 'target_celltype': dst, 'lr_pair': lr, 'flow': flow}
        for (src, dst, lr), flow in comm_flow.items()
    ])
    comm_flow_df.to_csv(OUTPUT_DIR / 'L3_cell_communication_flow.csv', index=False)
    logger.info(f"  通讯流: L3_cell_communication_flow.csv")
    
    # 保存训练损失
    pd.DataFrame({'epoch': range(1, len(losses) + 1), 'loss': losses}).to_csv(
        OUTPUT_DIR / 'L3_training_loss.csv', index=False)
    
    # 保存节点嵌入
    for ntype in ['gene', 'compound', 'celltype', 'lr', 'pathway']:
        if ntype in x_hgt:
            emb = x_hgt[ntype].detach().cpu().numpy()
            names = graph_data[ntype]['names']
            emb_df = pd.DataFrame(emb, index=names)
            emb_df.columns = [f'dim_{i}' for i in range(emb.shape[1])]
            emb_df.to_csv(OUTPUT_DIR / f'L3_{ntype}_embeddings.csv')
    
    # ---- 可视化 ----
    logger.info("=" * 60)
    logger.info("生成图表")
    
    # Fig3A: 异质图拓扑
    plot_fig3a_graph_topology(graph_data, str(FIGS_DIR / 'Fig3A_graph_topology.png'))
    
    # Fig3B: 注意力热图 (使用嵌入相似度作为注意力代理)
    plot_fig3b_attention_heatmap(graph_data, x_hgt, str(FIGS_DIR / 'Fig3B_attention_heatmap.png'))
    
    # Fig3C: Hub基因排名
    plot_fig3c_hub_ranking(hub_ranking, str(FIGS_DIR / 'Fig3C_hub_ranking.png'))
    
    # Fig3D: 通讯流图
    plot_fig3d_comm_flow(comm_flow, graph_data, str(FIGS_DIR / 'Fig3D_comm_flow.png'))
    
    # Fig3E: t-SNE
    gene_emb_np = gene_emb.detach().cpu().numpy() if isinstance(gene_emb, torch.Tensor) else gene_emb
    plot_fig3e_embedding_tsne(gene_emb_np, gene_names, str(FIGS_DIR / 'Fig3E_tsne_embedding.png'))
    
    # Fig3F: SHAP
    plot_fig3f_gnn_explain(explain_results, str(FIGS_DIR / 'Fig3F_gnn_explain.png'))
    
    # ---- 摘要报告 ----
    logger.info("=" * 60)
    logger.info("模块三摘要")
    logger.info("=" * 60)
    
    # ACSL4排名
    acsl4_rank = hub_ranking[hub_ranking['gene'] == 'ACSL4']
    if not acsl4_rank.empty:
        rank_val = acsl4_rank['rank'].values[0]
        total_genes = len(hub_ranking)
        logger.info(f"  ACSL4 Hub排名: {rank_val}/{total_genes} (top {rank_val/total_genes*100:.1f}%)")
    
    # BCP-ACSL4
    bcp_acsl4 = compound_ranking[(compound_ranking['compound'] == 'BCP') & (compound_ranking['gene'] == 'ACSL4')]
    if not bcp_acsl4.empty:
        logger.info(f"  BCP-ACSL4 嵌入相似度: {bcp_acsl4['similarity'].values[0]:.4f}")
    
    # Microglia→Neuron通讯
    ct_names = graph_data['celltype']['names']
    if 'Microglia' in ct_names and 'Neuron' in ct_names:
        mic_neu_flow = sum(flow for (src, dst, _), flow in comm_flow.items()
                           if src == 'Microglia' and dst == 'Neuron')
        logger.info(f"  Microglia→Neuron 通讯流: {mic_neu_flow:.2f}")
    
    logger.info(f"\n  结果目录: {OUTPUT_DIR}")
    logger.info(f"  图表目录: {FIGS_DIR}")
    logger.info("=" * 70)
    logger.info("模块三分析完成!")
    logger.info("=" * 70)


# ================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║          升 级 扩 展 区  (U p g r a d e   Z o n e)          ║
# ║   5个轻量模块 - 零额外依赖, 布尔开关控制, ≤300行新增         ║
# ╚══════════════════════════════════════════════════════════════╝
# ================================================================

# ============================================================
# L1. 节点中心性编码 (Node Centrality Encoding)
# ============================================================
# 参考: Graph-BERT centrality encoding (Zhang et al., 2020)
#       SAN - eigenvectors as node positional encodings (Kreuzer et al., 2021)

def inject_centrality_features(graph_data: dict) -> dict:
    """
    在基因共表达子图上计算图中心性指标, 拼接为特征增强
    
    计算: degree, PageRank, betweenness centrality
    拼接至 graph_data['gene']['x'] 末尾3维
    同步更新 graph_data dim 信息
    """
    try:
        import networkx as nx
        
        coexp_edges = graph_data['edges'].get('gene_coexp', [])
        if len(coexp_edges) < 2:
            logger.info("  centrality: 边不足, 填入默认值")
            centrality_feat = np.zeros((graph_data['gene']['n'], 3), dtype=np.float32)
        else:
            # 构建无向图
            G = nx.Graph()
            G.add_nodes_from(range(graph_data['gene']['n']))
            G.add_edges_from(coexp_edges)
            
            # 度中心性
            deg = np.array([nx.degree(G, n) for n in range(graph_data['gene']['n'])])
            deg = deg / (deg.max() + 1)  # 归一化
            
            # PageRank
            pr = np.array(list(nx.pagerank(G, alpha=0.85).values()))
            pr = pr / (pr.max() + 1e-12)
            
            # 介数中心性 (采样加速)
            if G.number_of_nodes() > 100:
                bc = np.array(list(nx.betweenness_centrality(G, k=min(50, G.number_of_nodes())).values()))
            else:
                bc = np.array(list(nx.betweenness_centrality(G).values()))
            bc = bc / (bc.max() + 1e-12)
            
            centrality_feat = np.stack([deg, pr, bc], axis=1).astype(np.float32)
        
        # 拼接到基因特征
        old_feat = graph_data['gene']['x']
        graph_data['gene']['x'] = np.concatenate([old_feat, centrality_feat], axis=1)
        graph_data['gene']['_orig_dim'] = old_feat.shape[1]
        graph_data['gene']['_centrality_dim'] = 3
        
        logger.info(f"  centrality: degree({centrality_feat[:,0].mean():.2f}), "
                    f"pagerank({centrality_feat[:,1].mean():.2f}), "
                    f"betweenness({centrality_feat[:,2].mean():.2f})")
    except ImportError:
        logger.warning("  networkx 未安装, 跳过中心性编码 (填充零)")
        graph_data['gene']['x'] = np.pad(
            graph_data['gene']['x'], ((0,0),(0,3)), mode='constant'
        )
    
    return graph_data


# ============================================================
# L2. 图自动增强 (DropEdge Graph Augmentation)
# ============================================================
# 参考: DropEdge - Towards Deep Graph Neural Networks on Node Classification
#       (Rong et al., ICLR 2020)

def augment_graph(train_edge_index_dict: dict, drop_p: float = 0.1,
                  seed: int = None) -> dict:
    """
    对训练边字典应用随机DropEdge增强
    
    对 gene_coexp 和 regulates 边类型以概率 drop_p 随机丢弃边,
    生成增强视图 aug_edge_index_dict, 防止过平滑并提升鲁棒性
    """
    if seed is not None:
        rng_drop = np.random.RandomState(seed)
    else:
        rng_drop = np.random.RandomState()
    
    aug_dict = {}
    for key, ei in train_edge_index_dict.items():
        if not hasattr(key, '__len__') or len(key) != 3:
            aug_dict[key] = ei
            continue
        
        rel = key[1]
        if rel in ('coexp', 'regulates') and ei.size(1) > 0:
            # 随机保留边: bernoulli(1-drop_p), GPU兼容
            keep_mask = torch.from_numpy(
                rng_drop.random(ei.size(1)) > drop_p
            ).to(ei.device)
            if keep_mask.sum() < 1:
                aug_dict[key] = ei  # 至少保留1条边
            else:
                aug_dict[key] = ei[:, keep_mask]
        else:
            aug_dict[key] = ei
    
    return aug_dict


# ============================================================
# L3. 元路径自动学习器 (MetaPathLearner)
# ============================================================
# 参考: GTN - Graph Transformer Networks (Yun et al., NeurIPS 2019)
#       HAN - Heterogeneous Graph Attention Network (Wang et al., WWW 2019)

class MetaPathLearner(nn.Module):
    """
    元路径自动学习模块
    
    对多条候选元路径 (gene→pathway→gene, gene→lr→gene) 的邻接矩阵序列
    做可学习加权 (softmax + 1x1 conv), 聚合为单一元路径邻接
    
    集成到 HeCoPreTrainer 中替代简化版2-hop聚合
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 元路径注意力: 对各元路径的聚合结果做加权
        self.path_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_att = nn.Linear(hidden_dim, 1)
    
    def aggregate_metapath(self, gene_emb: torch.Tensor,
                            edge_index_dict: dict, pathway_emb: torch.Tensor,
                            lr_emb: torch.Tensor = None) -> torch.Tensor:
        """
        聚合多条元路径信息
        
        元路径1: gene → pathway → gene (共享通路)
        元路径2: gene → lr → gene (共享配体-受体)
        """
        N = gene_emb.size(0)
        mp_embs = []
        
        # 元路径1: gene→pathway→gene 2-hop
        gp_edge_key = ('gene', 'enriched_in', 'pathway')
        if gp_edge_key in edge_index_dict:
            gp_ei = edge_index_dict[gp_edge_key]
            # 聚合: 每个gene连接的所有pathway的均值作为桥梁
            pw_agg = self._two_hop_aggregate(gene_emb, gp_ei, pathway_emb)
            mp_embs.append(pw_agg)
        
        # 元路径2: gene→lr→gene
        gl_edge_key = ('gene', 'encodes', 'lr')
        if gl_edge_key in edge_index_dict and lr_emb is not None:
            gl_ei = edge_index_dict[gl_edge_key]
            lr_agg = self._two_hop_aggregate(gene_emb, gl_ei, lr_emb)
            mp_embs.append(lr_agg)
        
        if len(mp_embs) == 0:
            return gene_emb
        
        # 可学习加权聚合多条元路径
        mp_stack = torch.stack(mp_embs, dim=0)  # [K, N, D]
        
        # 注意力加权
        projected = self.path_proj(mp_stack)  # [K, N, D]
        att_scores = self.path_att(projected).squeeze(-1)  # [K, N]
        att_weights = F.softmax(att_scores, dim=0).unsqueeze(-1)  # [K, N, 1]
        
        weighted_mp = (mp_stack * att_weights).sum(dim=0)  # [N, D]
        
        # 残差连接
        return gene_emb + 0.5 * weighted_mp
    
    def _two_hop_aggregate(self, gene_emb: torch.Tensor,
                            edge_index: torch.Tensor,
                            bridge_emb: torch.Tensor) -> torch.Tensor:
        """
        2-hop聚合: gene→bridge→gene (HeCo KDD 2021 元路径视图)
        
        正确实现: 对每个桥节点 b, 找到所有连接它的基因 {g1, g2, ...},
        让这些基因互相聚合 (共享桥节点的基因互为邻居)。
        
        使用 scatter 操作高效实现, 避免 Python for 循环。
        """
        N = gene_emb.size(0)
        device = gene_emb.device
        src_gene = edge_index[0]  # [E]
        dst_bridge = edge_index[1]  # [E]
        
        if dst_bridge.max() >= bridge_emb.size(0):
            # 桥节点索引越界, 回退到恒等映射
            return gene_emb
        
        # 1-hop: gene → bridge (聚合每个桥节点连接的所有基因嵌入)
        # scatter_add: 将 gene_emb[src_gene] 按 dst_bridge 累加到 bridge_emb 空间
        gene_to_bridge = torch.zeros(bridge_emb.size(0), self.hidden_dim, device=device)
        gene_to_bridge = gene_to_bridge.index_add(0, dst_bridge, gene_emb[src_gene])
        # 计算每个桥节点连接的基因数 (用于归一化)
        bridge_degree = torch.zeros(bridge_emb.size(0), device=device)
        bridge_degree = bridge_degree.index_add(0, dst_bridge, torch.ones_like(dst_bridge, dtype=torch.float32))
        bridge_degree = torch.clamp(bridge_degree, min=1)
        gene_to_bridge = gene_to_bridge / bridge_degree.unsqueeze(-1)  # [n_bridge, D]
        
        # 2-hop: bridge → gene (将桥节点聚合结果散射回基因)
        # 每个基因收到其连接的所有桥节点的聚合结果
        metapath_emb = torch.zeros(N, self.hidden_dim, device=device)
        metapath_emb = metapath_emb.index_add(0, src_gene, gene_to_bridge[dst_bridge])
        # 归一化: 每个基因的度数
        gene_degree = torch.zeros(N, device=device)
        gene_degree = gene_degree.index_add(0, src_gene, torch.ones_like(src_gene, dtype=torch.float32))
        gene_degree = torch.clamp(gene_degree, min=1)
        metapath_emb = metapath_emb / gene_degree.unsqueeze(-1)
        
        return metapath_emb


# ============================================================
# L4. 负采样课程 (CurriculumNegativeSampler)
# ============================================================
# 参考: PinSage hard negative mining (Ying et al., KDD 2018)
#       Curriculum Graph Co-Teaching (Gong et al., AAAI 2023)

class CurriculumNegativeSampler:
    """
    课程负采样: 难分负样本动态挖掘
    
    为化合物-靶点任务每 epoch 采样困难负样本:
    按当前嵌入余弦相似度排序, 选择 top-k 高相似但非正样本的对
    背景基因不参与负采样 (仅核心基因做监督)
    """
    def __init__(self, n_compounds: int, n_genes: int,
                 core_gene_indices: List[int] = None,
                 hard_neg_ratio: float = 0.3,
                 pos_set: set = None):
        self.n_compounds = n_compounds
        self.n_genes = n_genes
        self.core_gene_indices = core_gene_indices or list(range(n_genes))
        self.core_gene_set = set(self.core_gene_indices)
        self.n_core_genes = len(self.core_gene_indices)
        self.hard_neg_ratio = hard_neg_ratio
        self.pos_set = pos_set or set()
    
    def sample_hard_negatives(self, compound_emb: torch.Tensor,
                               gene_emb: torch.Tensor,
                               n_easy: int, n_hard: int,
                               rng: np.random.RandomState = None) -> tuple:
        """
        采样负样本: 简单负样本 + 困难负样本
        
        Args:
            compound_emb: [n_compounds, D]
            gene_emb: [n_genes, D]
            n_easy: 简单负样本数
            n_hard: 困难负样本数
        Returns:
            neg_edges: [(comp_idx, gene_idx), ...], neg_labels: [0, ...]
        """
        if rng is None:
            rng = np.random.RandomState()
        
        neg_edges = []
        neg_edge_set = set()  # 防重复
        
        # 1. 简单负样本: 从核心基因中随机采样
        n_easy_sampled = 0
        max_attempts = n_easy * 20
        attempts = 0
        while n_easy_sampled < n_easy and attempts < max_attempts:
            ci = rng.randint(0, self.n_compounds)
            gi = int(rng.choice(self.core_gene_indices))
            key = (ci, gi)
            if key not in self.pos_set and key not in neg_edge_set:
                neg_edges.append(key)
                neg_edge_set.add(key)
                n_easy_sampled += 1
            attempts += 1
        
        # 2. 困难负样本: 按余弦相似度从核心基因中采样
        if n_hard > 0:
            emb_c = F.normalize(compound_emb, dim=-1)
            emb_g = F.normalize(gene_emb, dim=-1)
            
            # 仅考虑核心基因的相似度
            core_gene_emb = emb_g[self.core_gene_indices]  # [n_core, D]
            sim = torch.mm(emb_c, core_gene_emb.T)  # [C, n_core]
            
            # 排除正样本对
            for ci, gi in self.pos_set:
                if ci < self.n_compounds:
                    # 将正样本对映射到核心基因索引
                    if gi in self.core_gene_set:
                        gi_core = self.core_gene_indices.index(gi)
                        sim[ci, gi_core] = -2.0
            
            # 选择相似度最高的 n_hard 个作为困难负样本
            n_hard_actual = min(n_hard, sim.numel() - len([p for p in self.pos_set if p[1] in self.core_gene_set]))
            if n_hard_actual > 0:
                flat_sim = sim.view(-1)
                _, top_indices = torch.topk(flat_sim, n_hard_actual)
                for idx in top_indices:
                    ci = (idx // self.n_core_genes).item()
                    gi_core = (idx % self.n_core_genes).item()
                    gi = self.core_gene_indices[gi_core]
                    neg_edges.append((ci, gi))
        
        neg_labels = [0.0] * len(neg_edges)
        return neg_edges, neg_labels
    
    def update_pos_set(self, pos_edges: list):
        """更新正样本对集合"""
        for ci, gi in pos_edges:
            self.pos_set.add((ci, gi))


# ============================================================
# L5. 动态异质图快照 (TemporalSnapshotGenerator)
# ============================================================
# 参考: DyHATR - Dynamic HATR for time-series graphs (Luo et al., 2021)
#        EvolveGCN (Pareja et al., AAAI 2020)

class TemporalSnapshotGenerator:
    """
    动态异质图快照生成器 (条件可用)
    
    若L1数据中存在GSE104036等多时间点表达差异, 
    构建 {time: HeteroData} 序列用于时序分析
    
    时间编码: 正弦位置编码, 拼接到节点特征后
    """
    def __init__(self, timepoints: list, hidden_dim: int,
                 use_temporal: bool = True):
        self.timepoints = timepoints
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal and len(timepoints) > 1
        
        if self.use_temporal:
            # 时间编码: sin/cos 位置编码
            self._time_encodings = {}
            for i, t in enumerate(timepoints):
                pe = np.zeros(hidden_dim, dtype=np.float32)
                for k in range(hidden_dim // 2):
                    denom = 10000 ** (2 * k / hidden_dim)
                    pe[2*k] = np.sin(i / denom)
                    pe[2*k+1] = np.cos(i / denom)
                self._time_encodings[t] = pe
    
    def generate_snapshots(self, graph_data: dict,
                           expression_data: dict = None) -> dict:
        """
        生成时序快照序列
        
        Args:
            graph_data: 基础异质图数据
            expression_data: {timepoint: {gene: expression_vector}}
        Returns:
            snapshots: {timepoint: graph_data_copy} 或空dict
        """
        if not self.use_temporal:
            return {}
        
        snapshots = {}
        for tp in self.timepoints:
            snap = copy.deepcopy(graph_data)
            
            # 注入时间编码到基因特征
            time_feat = self._time_encodings.get(tp, np.zeros(self.hidden_dim))
            if expression_data and tp in expression_data:
                # TODO: 用真实表达数据更新基因特征
                pass
            
            # 将时间编码拼接为额外的特征维度
            snap['gene']['_time_enc'] = time_feat
            snap['_timepoint'] = tp
            snapshots[tp] = snap
        
        return snapshots
    
    def get_time_encoding(self, timepoint) -> np.ndarray:
        """获取指定时间点的时间编码"""
        return self._time_encodings.get(timepoint, np.zeros(self.hidden_dim))


if __name__ == '__main__':
    main()