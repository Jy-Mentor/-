#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2 模块统一配置加载器
========================
从 config.yaml 读取 L2 相关路径与参数, 支持环境变量覆盖,
避免在 L2 脚本中硬编码绝对路径.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_l2_config(project_root: Path = None) -> Dict[str, Any]:
    """加载 L2 配置, 并解析为绝对路径.

    参数:
      project_root: 项目根目录. 默认为当前文件所在目录.

    返回:
      包含绝对路径与参数的字典.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent

    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml 不存在: {config_path}")

    cfg = _load_yaml(config_path)
    l2_cfg = cfg.get("l2", {})

    # 允许环境变量覆盖
    geo_dir = Path(
        os.environ.get(
            "IRON_GEO_DIR", l2_cfg.get("geo_dir", "D:/反向网络药理学/L1 数据集")
        )
    )
    if not geo_dir.is_absolute():
        geo_dir = project_root / geo_dir

    rscript = Path(
        os.environ.get("RSCRIPT", l2_cfg.get("rscript", "C:/R/R-4.5.2/bin/Rscript.exe"))
    )

    def _resolve(p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute():
            return pp
        # 先尝试相对于 geo_dir, 再相对于项目根目录
        candidate = geo_dir / pp
        if (
            candidate.exists()
            or str(pp).startswith("bulk/")
            or str(pp).startswith("RNA-seq/")
        ):
            return candidate
        return project_root / pp

    datasets = {k: _resolve(v) for k, v in l2_cfg.get("datasets", {}).items()}
    annotations = {k: _resolve(v) for k, v in l2_cfg.get("annotations", {}).items()}

    output_dir = project_root / l2_cfg.get("output_dir", "L2_results")
    wgcna_input_dir = project_root / l2_cfg.get("wgcna_input_dir", "L2_WGCNA_input")
    wgcna_output_dir = project_root / l2_cfg.get("wgcna_output_dir", "L2_WGCNA_output")
    module2_output_dir = project_root / l2_cfg.get(
        "module2_output_dir", "module2_results"
    )
    module2_plot_dir = project_root / l2_cfg.get(
        "module2_plot_dir", "module2_plot_data"
    )
    module3_interface_dir = project_root / l2_cfg.get(
        "module3_interface_dir", "module3_interface"
    )
    l1_results_dir = project_root / l2_cfg.get("l1_results_dir", "L1/l1_results")

    for d in [
        output_dir,
        wgcna_input_dir,
        wgcna_output_dir,
        module2_output_dir,
        module2_plot_dir,
        module3_interface_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "geo_dir": geo_dir,
        "rscript": rscript,
        "output_dir": output_dir,
        "wgcna_input_dir": wgcna_input_dir,
        "wgcna_output_dir": wgcna_output_dir,
        "module2_output_dir": module2_output_dir,
        "module2_plot_dir": module2_plot_dir,
        "module3_interface_dir": module3_interface_dir,
        "l1_results_dir": l1_results_dir,
        "datasets": datasets,
        "annotations": annotations,
        "gene_sets": l2_cfg.get("gene_sets", {}),
        "raw_config": l2_cfg,
    }
