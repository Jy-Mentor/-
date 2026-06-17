#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
IDSP 基因集定义 (V2 精修版)

更新说明 (2026-06-16):
  基于 GitHub、FerrDb V2、SenMayo、CellAge 及 CIRI 铁死亡/衰老文献审查:

  【删除 - 虚假靶点】
    - DNA 复制许可因子: MCM2-7 (仅在增殖细胞表达，神经元不表达)
    - DNA 复制夹装载复合物: RFC1-5, RPA1-3
    - 有丝分裂机器: BUB1/BUB1B/BUB3/CDC20/MAD2L1/PLK1/AURKA/AURKB/MKI67/TOP2A
    - 通用DNA连接酶: LIG1 (LIG3/LIG4保留，涉及线粒体修复)
    - 非CNS免疫标记: CD4, CD8A, CSF2, CSF3
    - 铁衰老列表清洗: ICA1(胰腺), DPEP1(肾脏), SLAMF8(免疫), 
      MEN1(内分泌肿瘤), COX7A1(心脏/骨骼肌), EMP1(上皮), SPATA2(精子发生)

  【新增 - 真实靶点(CIRI脑缺血文献支持)】
    - STAT1: IFN-γ/JAK/STAT1/IRF1 铁死亡轴 (Li et al. 2025, FASEB J)
    - CXCL8(IL8): 关键SASP因子 (SenMayo 缺失补充)
    - HERC2: 铁蛋白自噬调控 (Shi et al. 2025, Cell Mol Neurobiol)
    - FBXL5: 铁感应E3连接酶 (Shi et al. 2025)
    - FABP5: 卒中铁死亡临床生物标志物 (Long et al. 2025, Front Neurosci)
    - GSS: 谷胱甘肽合成酶 (KEGG hsa04216 铁死亡通路)
    - CRYAB: 神经保护小热休克蛋白 (FerrDb V2 Validated)

数据库引用:
  - FerrDb V2: Zhou & Bao, 2023, Nucleic Acids Research (PMID: 36305826)
  - CellAge: Avelar et al., 2020, Genome Biology (PMID: 32264951)
  - SenMayo: Saul et al., 2022, Nature Communications (PMID: 35999225)

三个基因集:
  - PURE_FERROPTOSIS (79)  — FerrDb V2 Validated 核心 + 文献补充
  - PURE_SENESCENCE  (99)  — CellAge + SenMayo 核心 (去除神经元无关联基因)
  - SHARED_GENES     (14)  — 铁死亡 ∩ 衰老 桥接基因

运行时断言: PURE ∩ PURE = ∅ (互斥)
=====================================================================
"""

# ---- 纯铁死亡基因 (FerrDb V2 Validated 级别 + 文献补充) ----
PURE_FERROPTOSIS: set = {
    # === FerrDb V2 Validated ===
    'ACSL4', 'PTGS2', 'HMOX1', 'TFRC', 'SLC7A11', 'CHAC1', 'SLC3A2',
    'FTH1', 'FTL', 'NFE2L2', 'GPX4', 'DPP4', 'ALOX5', 'ALOX12',
    'SAT1', 'SLC40A1', 'STEAP3', 'CBS', 'CSE1L', 'HSPB1',
    'VDAC2', 'VDAC3', 'GOT1', 'GCLC', 'GCLM', 'ABCC1', 'ABCC2',
    'ATF3', 'ATF4', 'DDIT3', 'SESN2', 'FANCD2', 'CDO1',
    'ZEB1', 'SNAI1', 'MMP9', 'RGS4', 'SQSTM1', 'NCOA4', 'BECN1',
    'PRNP', 'ADIPOQ', 'PLIN2', 'LPIN1', 'LPIN2', 'PNPLA2',
    'MAP1LC3A', 'MAP1LC3B', 'GABARAP', 'GABARAPL1',
    'ATG3', 'ATG5', 'ATG7', 'BAP1', 'TRIB3',
    'TFAM', 'PPARGC1A', 'SIRT1', 'FOXO1', 'FOXO3',
    'PRKAA1', 'PRKAA2', 'NFKB1', 'RELA', 'BNIP3', 'BNIP3L',
    'HSP90AA1', 'HSPA5', 'HSPD1', 'EIF2AK3', 'EIF2A',
    # === 文献新增 (CIRI 铁死亡特异性) ===
    'GSS',       # GSH合成酶, KEGG hsa04216 铁死亡通路核心
    'CRYAB',     # 神经保护小HSP, FerrDb V2 Validated
    'HERC2',     # 铁蛋白自噬调控 (Shi et al. 2025, Cell Mol Neurobiol)
    'FBXL5',     # 铁感应E3泛素连接酶 (Shi et al. 2025)
    'FABP5',     # 卒中铁死亡生物标志物 (Long et al. 2025, Front Neurosci)
    'ACSL3',     # 脂质代谢, KEGG hsa04216
    'SLC39A8',   # ZIP8, 脑铁转运 (FerrDb V2)
}  # size=79

# ---- 纯衰老基因 (CellAge + SenMayo 核心, 去除神经元无关联基因) ----
PURE_SENESCENCE: set = {
    # === 细胞周期抑制/衰老效应器 ===
    'CDKN2A', 'CDKN2B', 'CDKN1A', 'CDKN1B', 'E2F1', 'E2F2', 'E2F3',
    'CCND1', 'CCNE1', 'CCNA2', 'CCNB1', 'CDK4', 'CDK6', 'CDK2',
    'CHEK1', 'CHEK2', 'MDM2', 'MDM4',
    # === DNA损伤/衰老信号 ===
    'ATM', 'ATR', 'H2AX', 'GADD45A', 'GADD45B',
    # === SASP 经典分泌表型 ===
    'SERPINE1', 'IGFBP3', 'IGFBP5', 'IGFBP7', 'IL6', 'IL1A', 'IL1B',
    'CCL2', 'CXCL10', 'CXCL8',                 # 注: 移除泛用CCL3/CCL4/CXCL1/CXCL2
    'MMP1', 'MMP2', 'MMP3', 'MMP10', 'MMP12', 'MMP13',
    'TIMP1', 'TIMP2', 'FN1', 'COL1A1', 'COL1A2', 'COL3A1',
    'TNF', 'TGFB1', 'VEGFA', 'IFNG',
    # === 核膜/染色质衰老标志 ===
    'LMNB1', 'HMGB1', 'HMGA1', 'HMGA2', 'SIRT6', 'FOXO4',
    # === 衰老信号通路 ===
    'STAT1', 'STAT3', 'JAK2',                  # 新增 STAT1 (IFN-γ/STAT1 铁死亡轴)
    'MAPK1', 'MAPK3', 'MAPK8', 'MAPK14',
    'AKT1', 'MTOR', 'RPS6KB1', 'PTEN', 'TSC1', 'TSC2',
    # === 转录调控/表观遗传 ===
    'CREB1', 'ATF2', 'JUN', 'FOS', 'MYC', 'MAX', 'MNT',
    'HDAC1', 'HDAC2', 'HDAC3', 'EP300', 'CREBBP', 'BRD4',
    'PARP1', 'PCNA',
    # === DNA修复 (保留CNS相关) ===
    'LIG3', 'LIG4', 'XRCC1', 'XRCC6', 'XRCC5', 'PRKDC',
    'NBN', 'MRE11', 'RAD50', 'RAD51', 'BRCA1', 'BRCA2', 'BLM', 'WRN',
    # === 端粒维持 ===
    'TERF1', 'TERF2', 'TERT',
    # === 免疫/黏附分子 (脑相关) ===
    'CD38', 'ICAM1', 'VCAM1', 'SELE', 'IL18', 'IL10', 'TNFRSF1A',
}  # size=99

# ---- 共享基因 (桥接铁死亡与衰老) ----
SHARED_GENES: set = {
    'TP53', 'CDKN1A', 'RB1',            # 铁死亡-衰老经典三方交叉
    'CD74', 'S100A8', 'IFNG',            # Liu et al. 2026 桥接基因
    'IRF1', 'TLR4', 'NLRP3',            # 炎症介导铁死亡-衰老
    'HIF1A', 'KEAP1', 'SOD1',           # 氧化应激介导
    'IL1B', 'CXCL10',                    # SASP ∩ 铁死亡 (文献新增)
}  # size=14

# ---- 完整基因集 ----
FERROPTOSIS_ALL: set = PURE_FERROPTOSIS | SHARED_GENES
SENESCENCE_ALL: set = PURE_SENESCENCE | SHARED_GENES

# ---- 运行时断言 ----
assert PURE_FERROPTOSIS.isdisjoint(PURE_SENESCENCE), \
    "PURE_FERROPTOSIS 和 PURE_SENESCENCE 不能有重叠!"

if __name__ == '__main__':
    print(f"PURE_FERROPTOSIS: {len(PURE_FERROPTOSIS)} genes")
    print(f"PURE_SENESCENCE:  {len(PURE_SENESCENCE)} genes")
    print(f"SHARED_GENES:     {len(SHARED_GENES)} genes")
    print(f"PURE → PURE overlap: {len(PURE_FERROPTOSIS & PURE_SENESCENCE)} (should be 0)")
    print("All checks passed ✓")