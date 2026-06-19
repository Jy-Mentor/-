"""铁衰老 × CIRI × GAT-HGT 系统升级包.

本包为现有脚本式项目的模块化重构结果, 目标是通过 GAT-HGT 异构图网络
发现能够靶向铁衰老通路、缓解脑缺血再灌注损伤(CIRI)的中药单体.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NETWORK_DIR = PROJECT_ROOT / "network_files"
L3_DIR = PROJECT_ROOT / "L3"
L3_RESULTS_DIR = PROJECT_ROOT / "L3_results"

__version__ = "4.0.0-dev"
