#!/usr/bin/env python3
"""GAT-HGT 异构图训练流程入口.

本脚本为项目根目录的快捷入口, 实际逻辑位于 src/iron_aging/apps/hgt_pipeline.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保 src 在路径中
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from iron_aging.apps.hgt_pipeline import main  # noqa: E402 (先插入 sys.path, 再导入项目模块)

if __name__ == "__main__":
    raise SystemExit(main())
