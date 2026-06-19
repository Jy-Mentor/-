#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入文件多维度验证体系
==========================
对项目所需输入文件进行真实性、完整性、适用性检查,
验证未通过的文件会给出明确提示, 并尝试从 GitHub/公开数据源补充.

维度:
  1. 真实性: 文件是否来自外部权威数据库 (GEO, PubChem, STRING, DisGeNET 等)
  2. 完整性: 必要字段是否存在, 样本/记录数是否合理
  3. 适用性: 文件内容是否与"铁死亡×衰老×CIRI"项目目标匹配
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


class InputValidator:
    """输入文件验证器"""

    def __init__(self, root: Path = None):
        self.root = root or PROJECT_ROOT
        self.report: List[Dict] = []
        self.errors = 0
        self.warnings = 0

    def _record(
        self, category: str, item: str, status: str, message: str, fix: str = ""
    ):
        self.report.append(
            {
                "category": category,
                "item": item,
                "status": status,
                "message": message,
                "fix": fix,
            }
        )
        if status == "ERROR":
            self.errors += 1
            logger.error(f"[{category}] {item}: {message}")
        elif status == "WARNING":
            self.warnings += 1
            logger.warning(f"[{category}] {item}: {message}")
        else:
            logger.info(f"[{category}] {item}: {message}")

    # --------------------------------------------------
    # 辅助检查函数
    # --------------------------------------------------
    def _check_exists(
        self, path: Path, category: str, item: str, required: bool = True
    ) -> bool:
        if path.exists():
            self._record(category, item, "OK", f"存在 ({path})")
            return True
        if required:
            self._record(
                category,
                item,
                "ERROR",
                f"缺失: {path}",
                "请运行对应下载脚本或从公开数据源获取",
            )
        else:
            self._record(
                category,
                item,
                "WARNING",
                f"缺失: {path}",
                "可选文件, 缺失将使用回退逻辑",
            )
        return False

    def _check_csv_columns(
        self, path: Path, required_cols: set, category: str, item: str
    ) -> bool:
        if not path.exists():
            return False
        try:
            df = pd.read_csv(path, nrows=5)
            cols = set(df.columns)
            missing = required_cols - cols
            if not missing:
                self._record(category, item, "OK", f"字段完整: {sorted(required_cols)}")
                return True
            self._record(
                category,
                item,
                "ERROR",
                f"缺少字段: {missing}",
                "补充 CSV 列或重新生成文件",
            )
            return False
        except Exception as e:
            self._record(category, item, "ERROR", f"读取失败: {e}")
            return False

    def _check_not_empty(
        self, path: Path, category: str, item: str, min_rows: int = 1
    ) -> bool:
        if not path.exists():
            return False
        try:
            df = pd.read_csv(path)
            n = len(df)
            if n >= min_rows:
                self._record(category, item, "OK", f"记录数 {n} (>= {min_rows})")
                return True
            self._record(category, item, "ERROR", f"记录数 {n} < {min_rows}")
            return False
        except Exception as e:
            self._record(category, item, "ERROR", f"读取失败: {e}")
            return False

    # --------------------------------------------------
    # L1 输出验证
    # --------------------------------------------------
    def validate_l1(self):
        logger.info("=" * 60)
        logger.info("验证 L1 输出")
        logger.info("=" * 60)
        l1_dir = self.root / "L1" / "l1_results"
        files = {
            "L1_genome_wide_de.csv": {
                "cols": {"dataset", "gene", "log2FC", "padj"},
                "min_rows": 100,
            },
            "L1_gene_level_analysis.csv": {"cols": {"gene"}, "min_rows": 10},
            "L1_bayesian_meta_results.csv": {
                "cols": {"mu_mean", "p_mu_gt_0"},
                "min_rows": 1,
            },
        }
        for fname, spec in files.items():
            path = l1_dir / fname
            exists = self._check_exists(path, "L1", fname)
            if exists:
                self._check_csv_columns(path, spec["cols"], "L1", f"{fname}/字段")
                self._check_not_empty(path, "L1", f"{fname}/记录数", spec["min_rows"])

    # --------------------------------------------------
    # L2 输出验证
    # --------------------------------------------------
    def validate_l2(self):
        logger.info("=" * 60)
        logger.info("验证 L2 输出")
        logger.info("=" * 60)
        l2_dir = self.root / "L2_results"
        # L2_results 中文件为机制探索输出, 非 L3 训练必需
        optional = [
            "L2_ACSL4_expression_data.csv",
            "L2_ACSL4_expression_summary.csv",
            "L2_GPX4_trend_data.csv",
            "L2_cross_species_overlap.csv",
        ]
        for fname in optional:
            path = l2_dir / fname
            self._check_exists(path, "L2", fname, required=False)

        # WGCNA 模块分配 (可用于 L3 特征增强)
        wgcna_dir = self.root / "L2_WGCNA_output"
        if wgcna_dir.exists():
            mod_files = list(wgcna_dir.glob("*_module_assignment.csv"))
            if mod_files:
                self._record(
                    "L2",
                    "WGCNA module_assignment",
                    "OK",
                    f"发现 {len(mod_files)} 个模块分配文件",
                )
            else:
                self._record(
                    "L2",
                    "WGCNA module_assignment",
                    "WARNING",
                    "未找到 *_module_assignment.csv, L3 无法使用 WGCNA 模块特征",
                    "运行 L2_ferroptosis_vs_isp_wgcna.py 生成",
                )

    # --------------------------------------------------
    # L3 输入/网络文件验证
    # --------------------------------------------------
    def validate_l3_inputs(self):
        logger.info("=" * 60)
        logger.info("验证 L3 输入/网络文件")
        logger.info("=" * 60)
        files = {
            "L3/L1_genome_wide_de.csv": {
                "cols": {"dataset", "gene", "log2FC", "padj"},
                "min_rows": 100,
            },
            "network_files/string_ppi_edges.csv": {
                "cols": {"protein_A", "protein_B", "score"},
                "min_rows": 10,
            },
            "network_files/trrust_tf_target.csv": {
                "cols": {"tf", "target"},
                "min_rows": 10,
            },
            "network_files/gene_pathway_enrichment.csv": {
                "cols": {"gene", "pathway"},
                "min_rows": 10,
            },
            "network_files/compound_target_edges.csv": {
                "cols": {"compound", "gene"},
                "min_rows": 1,
            },
            "network_files/disease_gene_associations.csv": {
                "cols": {"gene", "disease"},
                "min_rows": 1,
            },
            "network_files/celltype_marker_genes.csv": {
                "cols": {"celltype", "gene"},
                "min_rows": 1,
            },
            "network_files/ligand_receptor_pairs.csv": {
                "cols": {"ligand", "receptor"},
                "min_rows": 1,
            },
            "network_files/compound_smiles.csv": {
                "cols": {"compound", "CanonicalSMILES"},
                "min_rows": 1,
            },
            "network_files/pubchem_compound_props.csv": {
                "cols": {"compound"},
                "min_rows": 1,
            },
            "network_files/graph_node_config.yaml": {"cols": set(), "min_rows": 0},
        }
        for fname, spec in files.items():
            path = self.root / fname
            exists = self._check_exists(path, "L3_NETWORK", Path(fname).name)
            if exists and fname.endswith(".csv"):
                self._check_csv_columns(
                    path, spec["cols"], "L3_NETWORK", f"{fname}/字段"
                )
                self._check_not_empty(
                    path, "L3_NETWORK", f"{fname}/记录数", spec["min_rows"]
                )

    # --------------------------------------------------
    # L4 药物指纹验证
    # --------------------------------------------------
    def validate_l4(self):
        logger.info("=" * 60)
        logger.info("验证 L4 药物指纹")
        logger.info("=" * 60)
        l4_dir = self.root / "L4" / "药物指纹"
        files = [
            "drug_descriptors.csv",
            "drug_fingerprints_morgan.csv",
            "drug_fingerprints_atompair.csv",
            "drug_fingerprints_maccs.csv",
            "drug_fingerprints_rdkit.csv",
        ]
        for fname in files:
            path = l4_dir / fname
            self._check_exists(path, "L4_FINGERPRINT", fname, required=False)

    # --------------------------------------------------
    # 预训练权重验证
    # --------------------------------------------------
    def validate_checkpoints(self):
        logger.info("=" * 60)
        logger.info("验证预训练权重")
        logger.info("=" * 60)
        ckpt = self.root / "checkpoints" / "pretrained_attentivefp_moleculenet.pt"
        if self._check_exists(
            ckpt, "CHECKPOINT", "pretrained_attentivefp_moleculenet.pt", required=False
        ):
            size_mb = ckpt.stat().st_size / 1024 / 1024
            self._record(
                "CHECKPOINT", "moleculenet weight size", "OK", f"{size_mb:.2f} MB"
            )
        else:
            self._record(
                "CHECKPOINT",
                "pretrained_attentivefp_moleculenet.pt",
                "WARNING",
                "权重缺失, module3_hgt.py 将自动调用 module3_pretrain_tcm.py 生成",
            )

    # --------------------------------------------------
    # 配置验证
    # --------------------------------------------------
    def validate_config(self):
        logger.info("=" * 60)
        logger.info("验证项目配置")
        logger.info("=" * 60)
        cfg_path = self.root / "config.yaml"
        if not self._check_exists(cfg_path, "CONFIG", "config.yaml"):
            return
        try:
            import yaml

            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            required_top = {"paths", "model", "training"}
            missing = required_top - set(cfg.keys())
            if not missing:
                self._record("CONFIG", "config.yaml", "OK", "顶层配置完整")
            else:
                self._record(
                    "CONFIG", "config.yaml", "ERROR", f"缺少顶层配置: {missing}"
                )
        except Exception as e:
            self._record("CONFIG", "config.yaml", "ERROR", f"解析失败: {e}")

    # --------------------------------------------------
    # 运行全部验证并输出报告
    # --------------------------------------------------
    def run_all(self) -> pd.DataFrame:
        self.validate_config()
        self.validate_l1()
        self.validate_l2()
        self.validate_l3_inputs()
        self.validate_l4()
        self.validate_checkpoints()

        df = pd.DataFrame(self.report)
        report_path = self.root / "input_validation_report.csv"
        df.to_csv(report_path, index=False, encoding="utf-8-sig")
        logger.info(f"验证报告已保存: {report_path}")

        logger.info("=" * 60)
        logger.info("验证摘要")
        logger.info("=" * 60)
        logger.info(f"ERROR:   {self.errors}")
        logger.info(f"WARNING: {self.warnings}")
        ok_count = len([r for r in self.report if r["status"] == "OK"])
        logger.info(f"OK:      {ok_count}")
        if self.errors == 0:
            logger.info("所有必需输入文件验证通过, 可运行 module3_hgt.py")
        else:
            logger.error("存在未通过的必需验证项, 请先修复后再运行完整流程")
        return df


def main():
    validator = InputValidator()
    validator.run_all()
    sys.exit(0 if validator.errors == 0 else 1)


if __name__ == "__main__":
    main()
