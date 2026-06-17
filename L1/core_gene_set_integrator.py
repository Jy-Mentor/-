#!/usr/bin/env python3
"""
核心基因集整合脚本
=========================================================
整合以下来源的基因信息，输出分级的核心基因集：

数据来源：
  1. IDSP基因集（idsp_gene_sets.py）：铁死亡/衰老/共享
  2. 铁衰老基因.txt：人工筛选的铁衰老相关基因
  3. L1跨数据集差异表达分析（L1_gene_level_analysis.csv）
  4. L3 GNN Hub基因排名（L3_hub_gene_ranking.csv）

输出：
  - core_gene_set_tiered.csv    — 分级核心基因集
  - core_gene_set_summary.xlsx  — Excel汇总（多sheet）

用法：
  python core_gene_set_integrator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 路径配置
#