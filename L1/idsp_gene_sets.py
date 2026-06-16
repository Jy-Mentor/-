#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
IDSP 基因集定义

三个基因集:
  - PURE_FERROPTOSIS (73)  — FerrDb V2 Validated 核心
  - PURE_SENESCENCE  (140) — CellAge + SenMayo 核心
  - SHARED_GENES     (12)  — 铁死亡 ∩ 衰老 共享

运行时断言: PURE ∩ PURE = ∅ (互斥)
=====================================================================
"""

# ---- 纯铁死亡基因 (FerrDb V2 Validated 级别) ----
PURE_FERROPTOSIS: set = {
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
}  # size=72 (注意: 原ATM已移至PURE_SENESCENCE)

# ---- 纯衰老基因 (CellAge + SenMayo 核心) ----
PURE_SENESCENCE: set = {
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
}  # size=140

# ---- 共享基因 (桥接铁死亡与衰老) ----
SHARED_GENES: set = {
    'TP53', 'CDKN1A', 'RB1', 'CD74', 'S100A8', 'IFNG',
    'IRF1', 'TLR4', 'NLRP3', 'HIF1A', 'KEAP1', 'SOD1',
}  # size=12

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