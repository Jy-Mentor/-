#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
L2 双模块联合分析与质量检查
=============================================================================
任务目标:
    同时调用并协同使用两个 L2 模块:
      - L2_ferroptosis_vs_isp_wgcna.py : 大样本转录组 WGCNA + ACSL4/GPX4 趋势
      - module2_sc.py                  : 单细胞分辨率脑 I/R 铁衰老细胞图谱

执行内容:
    1. 输入数据多维度验证 (存在性、格式、完整性、准确性)
    2. 依次/并行运行两个 L2 模块
    3. 输出结果质量检查 (数据结构、字段完整性、记录数、质量标准)
    4. 联合分析: 整合 WGCNA ACSL4 模块基因与单细胞差异基因/铁衰老评分
    5. 生成标准化 JSON/CSV 质量报告

输出:
    - L2_integrated_report.json   : 联合分析综合报告
    - L2_integrated_summary.csv   : 两模块输出汇总
    - L2_joint_genes.csv          : WGCNA-单细胞联合基因集
=============================================================================
"""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _l2_config import get_l2_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
_CFG = get_l2_config(Path(__file__).resolve().parent)
PROJECT_ROOT = _CFG["project_root"]
L1_RESULTS_DIR = _CFG["l1_results_dir"]
L2_OUTPUT_DIR = _CFG["output_dir"]
WGCNA_INPUT_DIR = _CFG["wgcna_input_dir"]
WGCNA_OUTPUT_DIR = _CFG["wgcna_output_dir"]
MODULE2_OUT_DIR = _CFG["module2_output_dir"]
MODULE2_PLOT_DIR = _CFG["module2_plot_dir"]
MODULE3_IFACE_DIR = _CFG["module3_interface_dir"]
DATASETS = _CFG["datasets"]
RSCRIPT = _CFG["rscript"]

REPORT_DIR = PROJECT_ROOT / "L2_integrated_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. 输入数据验证
# ---------------------------------------------------------------------------
REQUIRED_L1_FILES = {
    "L1_dual_scores_all_datasets.csv": [
        "dataset", "sample", "group", "ferroptosis", "senescence", "shared", "idsp_index"
    ],
    "L1_genome_wide_de.csv": ["dataset", "gene", "log2FC", "padj"],
}

REQUIRED_BULK_DATASETS = ["GSE16561", "GSE37587", "GSE61616", "GSE97537", "GSE104036"]

REQUIRED_ANNOTATIONS = {
    "GPL6883": ["Gene ID", "Gene symbol"],
    "GPL1355": ["ID", "Gene Symbol"],
}

REQUIRED_R_SCRIPT = "L2_WGCNA_modulePreservation.R"


def _check_csv_columns(path: Path, expected: List[str]) -> Tuple[bool, str]:
    """验证 CSV 文件存在且包含预期列."""
    if not path.exists():
        return False, f"文件不存在: {path}"
    try:
        df = pd.read_csv(path, nrows=2)
    except Exception as e:
        return False, f"读取失败 {path}: {e}"
    missing = [c for c in expected if c not in df.columns]
    if missing:
        return False, f"{path} 缺失列: {missing}"
    return True, f"OK: {path}"


def _validate_l1_outputs(errors: List[str]) -> Dict[str, str]:
    """验证 L1 输出文件."""
    status = {}
    for fname, cols in REQUIRED_L1_FILES.items():
        ok, msg = _check_csv_columns(L1_RESULTS_DIR / fname, cols)
        status[fname] = msg
        if not ok:
            errors.append(msg)
    return status


def _validate_bulk_datasets(errors: List[str]) -> Dict[str, Dict[str, Any]]:
    """验证 bulk GEO 数据集."""
    status = {}
    for ds in REQUIRED_BULK_DATASETS:
        path = DATASETS.get(ds)
        exists = path is not None and path.exists()
        status[ds] = {"path": str(path) if path else None, "exists": exists}
        if not exists:
            errors.append(f"缺失 bulk 数据集 {ds}: {path}")
    return status


def _validate_single_cell(errors: List[str]) -> Dict[str, Any]:
    """验证单细胞数据集."""
    sc_path = DATASETS.get("GSE174574")
    sc_exists = sc_path is not None and sc_path.exists()
    if not sc_exists:
        errors.append(f"缺失单细胞数据集 GSE174574: {sc_path}")
    return {"path": str(sc_path) if sc_path else None, "exists": sc_exists}


def _validate_annotations(
    errors: List[str], warnings: List[str]
) -> Dict[str, Dict[str, Any]]:
    """验证平台注释文件."""
    status = {}
    outputs_exist = all(
        (L2_OUTPUT_DIR / f).exists() for f in REQUIRED_BULK_OUTPUTS[L2_OUTPUT_DIR]
    )
    for key in REQUIRED_ANNOTATIONS:
        path = _CFG["annotations"].get(key)
        exists = path is not None and path.exists()
        status[key] = {"path": str(path) if path else None, "exists": exists}
        if not exists:
            msg = f"缺失平台注释 {key}: {path}"
            if outputs_exist:
                warnings.append(f"{msg} (已有 L2 输出, 可跳过)")
            else:
                errors.append(msg)
    return status


def _validate_r_environment(errors: List[str]) -> Dict[str, Any]:
    """验证 R 运行环境."""
    r_ok = RSCRIPT.exists()
    if not r_ok:
        errors.append(f"Rscript 不存在: {RSCRIPT}")

    r_script_path = PROJECT_ROOT / REQUIRED_R_SCRIPT
    r_script_ok = r_script_path.exists()
    if not r_script_ok:
        errors.append(f"缺失 R WGCNA 脚本: {r_script_path}")

    return {
        "rscript": {"path": str(RSCRIPT), "exists": r_ok},
        "r_wgcna_script": {"path": str(r_script_path), "exists": r_script_ok},
    }


def validate_inputs() -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
    """对两个 L2 模块的输入执行全面验证.

    返回: (passed, errors, warnings, details)
    """
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {
        "l1_outputs": _validate_l1_outputs(errors),
        "bulk_datasets": _validate_bulk_datasets(errors),
        "single_cell_dataset": _validate_single_cell(errors),
        "annotations": _validate_annotations(errors, warnings),
    }
    details.update(_validate_r_environment(errors))

    passed = len(errors) == 0
    details["input_validation_passed"] = passed
    details["input_validation_errors"] = errors
    details["input_validation_warnings"] = warnings
    return passed, errors, warnings, details


# ---------------------------------------------------------------------------
# 2. 运行 L2 模块
# ---------------------------------------------------------------------------
def _run_python_script(script_name: str, timeout: int = 7200) -> Tuple[bool, str]:
    """通过子进程运行指定 Python 脚本, 隔离执行环境."""
    script_path = PROJECT_ROOT / script_name
    if not script_path.exists():
        return False, f"脚本不存在: {script_path}"

    logger.info(f"启动 L2 模块: {script_name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    combined = result.stdout + "\n" + result.stderr
    success = result.returncode == 0
    if success:
        logger.info(f"  {script_name} 完成 (returncode=0)")
    else:
        logger.error(f"  {script_name} 失败 (returncode={result.returncode})")
    return success, combined


def run_l2_modules(skip_if_valid: bool = True) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """运行两个 L2 模块并返回状态与日志."""
    status: Dict[str, Any] = {}
    logs: Dict[str, str] = {}

    # 检查是否已有有效输出 (可选跳过)
    if skip_if_valid:
        ok, _, out_details = validate_outputs()
        if ok:
            logger.info("检测到已有有效 L2 输出, 跳过重新运行")
            status["bulk"] = {"skipped": True, "reason": "existing_valid_outputs"}
            status["single_cell"] = {"skipped": True, "reason": "existing_valid_outputs"}
            return status, logs

    # 2.1 bulk WGCNA 模块
    bulk_ok, bulk_log = _run_python_script("L2_ferroptosis_vs_isp_wgcna.py", timeout=7200)
    status["bulk"] = {"success": bulk_ok, "log_lines": len(bulk_log.splitlines())}
    logs["bulk"] = bulk_log
    if not bulk_ok:
        logger.error("L2 bulk 模块运行失败, 跳过后续联合分析")
        return status, logs

    # 2.2 单细胞模块
    sc_ok, sc_log = _run_python_script("module2_sc.py", timeout=7200)
    status["single_cell"] = {"success": sc_ok, "log_lines": len(sc_log.splitlines())}
    logs["single_cell"] = sc_log
    if not sc_ok:
        logger.error("L2 单细胞模块运行失败")

    return status, logs


# ---------------------------------------------------------------------------
# 3. 输出结果质量检查
# ---------------------------------------------------------------------------
REQUIRED_BULK_OUTPUTS = {
    L2_OUTPUT_DIR: [
        "L2_ACSL4_expression_data.csv",
        "L2_ACSL4_expression_summary.csv",
        "L2_GPX4_trend_data.csv",
        "L2_cross_species_overlap.csv",
    ],
    WGCNA_OUTPUT_DIR: [
        "L2_WGCNA_summary.csv",
        "GSE16561_module_assignment.csv",
        "GSE16561_ACSL4_module_genes.csv",
        "GSE37587_module_assignment.csv",
        "GSE37587_ACSL4_module_genes.csv",
        "GSE97537_module_assignment.csv",
        "GSE97537_ACSL4_module_genes.csv",
        "GSE61616_module_assignment.csv",
        "GSE104036_module_assignment.csv",
    ],
}

REQUIRED_SC_OUTPUTS = {
    MODULE2_OUT_DIR: [
        "cell_type_DEG_MCAO_vs_Sham.csv",
        "LIANA_all_LR_pairs.csv",
        "iron_aging_significance.csv",
        "metabolism_correlations.csv",
    ],
    MODULE2_PLOT_DIR: [
        "cell_type_score_summary.csv",
        "consensus_senescence_detail.csv",
        "bubble_data.csv",
    ],
    MODULE3_IFACE_DIR: [
        "cell_type_DEG_significant.csv",
        "celltype_express_edges.csv",
        "pseudobulk_celltype_condition.csv",
        "pseudobulk_raw_counts.csv",
        "iron_aging_LR_pairs.csv",
        "lr_edges.csv",
    ],
}

OUTPUT_FORMAT_RULES = {
    "L2_ACSL4_expression_data.csv": {"min_rows": 1, "required_cols": ["dataset", "gene", "group", "expression"]},
    "L2_ACSL4_expression_summary.csv": {
        "min_rows": 1,
        "required_cols": ["dataset", "gene", "mean_expr_case", "mean_expr_control"],
    },
    "L2_GPX4_trend_data.csv": {"min_rows": 1, "required_cols": ["dataset", "gpx4_trend"]},
    "L2_cross_species_overlap.csv": {"min_rows": 1, "required_cols": ["dataset", "module", "isp_overlap"]},
    "L2_WGCNA_summary.csv": {"min_rows": 1, "required_cols": ["dataset", "module", "n_genes"]},
    "cell_type_DEG_MCAO_vs_Sham.csv": {"min_rows": 1, "required_cols": ["gene", "cell_type", "avg_log2FC"]},
    "iron_aging_significance.csv": {"min_rows": 1, "required_cols": ["cell_type", "score_type", "pvalue"]},
    "cell_type_DEG_significant.csv": {"min_rows": 1, "required_cols": ["gene", "cell_type", "avg_log2FC", "p_val_adj"]},
    "celltype_express_edges.csv": {"min_rows": 1, "required_cols": ["celltype", "gene", "mean_expr"]},
}


def _validate_output_file(path: Path, rules: Dict[str, Any]) -> Tuple[bool, str]:
    if not path.exists():
        return False, f"缺失: {path.name}"
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return False, f"读取失败 {path.name}: {e}"

    missing_cols = [c for c in rules.get("required_cols", []) if c not in df.columns]
    if missing_cols:
        return False, f"{path.name} 缺失列 {missing_cols}"

    min_rows = rules.get("min_rows", 1)
    if len(df) < min_rows:
        return False, f"{path.name} 记录数不足: {len(df)} < {min_rows}"

    # 数值列基本检查: 无全 NaN
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].notna().sum() == 0:
                return False, f"{path.name} 列 {col} 全为 NaN"

    return True, f"OK: {path.name} ({len(df)} rows, {len(df.columns)} cols)"


def validate_outputs() -> Tuple[bool, List[str], Dict[str, Any]]:
    """对两个 L2 模块的输出执行全面质量检查."""
    errors: List[str] = []
    details: Dict[str, Any] = {}

    # 3.1 bulk 输出
    bulk_status = {}
    for out_dir, fnames in REQUIRED_BULK_OUTPUTS.items():
        for fname in fnames:
            path = out_dir / fname
            rules = OUTPUT_FORMAT_RULES.get(fname, {"min_rows": 1, "required_cols": []})
            ok, msg = _validate_output_file(path, rules)
            bulk_status[fname] = msg
            if not ok:
                errors.append(msg)
    details["bulk_outputs"] = bulk_status

    # 3.2 单细胞输出
    sc_status = {}
    for out_dir, fnames in REQUIRED_SC_OUTPUTS.items():
        for fname in fnames:
            path = out_dir / fname
            rules = OUTPUT_FORMAT_RULES.get(fname, {"min_rows": 1, "required_cols": []})
            ok, msg = _validate_output_file(path, rules)
            sc_status[fname] = msg
            if not ok:
                errors.append(msg)
    details["single_cell_outputs"] = sc_status

    passed = len(errors) == 0
    details["output_validation_passed"] = passed
    details["output_validation_errors"] = errors
    return passed, errors, details


# ---------------------------------------------------------------------------
# 4. 联合分析
# ---------------------------------------------------------------------------
def joint_analysis() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """整合 WGCNA ACSL4 模块基因与单细胞差异基因/铁衰老显著性."""
    summary: Dict[str, Any] = {}

    # 4.1 收集 WGCNA ACSL4 模块基因
    wgcna_genes: Dict[str, set] = {}
    for path in sorted(WGCNA_OUTPUT_DIR.glob("*_ACSL4_module_genes.csv")):
        dataset = path.stem.replace("_ACSL4_module_genes", "")
        df = pd.read_csv(path)
        if "gene" in df.columns:
            genes = set(df["gene"].dropna().astype(str).str.upper())
        elif "gene_symbol" in df.columns:
            genes = set(df["gene_symbol"].dropna().astype(str).str.upper())
        else:
            genes = set(df.iloc[:, 0].dropna().astype(str).str.upper())
        wgcna_genes[dataset] = genes

    all_wgcna_genes = set().union(*wgcna_genes.values()) if wgcna_genes else set()
    summary["wgcna_acsl4_modules"] = {
        ds: len(genes) for ds, genes in wgcna_genes.items()
    }
    summary["wgcna_acsl4_union_genes"] = len(all_wgcna_genes)

    # 4.2 单细胞 DEG 与显著性
    sc_deg_path = MODULE2_OUT_DIR / "cell_type_DEG_MCAO_vs_Sham.csv"
    sc_deg = pd.read_csv(sc_deg_path) if sc_deg_path.exists() else pd.DataFrame()

    sc_sig_path = MODULE2_OUT_DIR / "iron_aging_significance.csv"
    sc_sig = pd.read_csv(sc_sig_path) if sc_sig_path.exists() else pd.DataFrame()

    summary["single_cell_deg_records"] = len(sc_deg)
    summary["single_cell_significance_records"] = len(sc_sig)

    # 4.3 交集: WGCNA ACSL4 模块基因 ∩ 单细胞 DEG 基因
    if "gene" in sc_deg.columns:
        sc_deg_genes = set(sc_deg["gene"].dropna().astype(str).str.upper())
    else:
        sc_deg_genes = set()

    joint_genes = sorted(all_wgcna_genes & sc_deg_genes)
    summary["joint_acsl4_sc_deg_genes"] = len(joint_genes)

    # 4.4 构建联合基因表
    joint_rows = []
    for gene in joint_genes:
        in_datasets = [ds for ds, genes in wgcna_genes.items() if gene in genes]
        deg_subset = sc_deg[sc_deg["gene"].str.upper() == gene] if "gene" in sc_deg.columns else pd.DataFrame()
        for _, row in deg_subset.iterrows():
            joint_rows.append(
                {
                    "gene": gene,
                    "wgcna_datasets": ";".join(in_datasets),
                    "n_wgcna_datasets": len(in_datasets),
                    "cell_type": row.get("cell_type", "NA"),
                    "avg_log2FC": row.get("avg_log2FC", float("nan")),
                    "p_val_adj": row.get("p_val_adj", float("nan")),
                    "pct_expr_case": row.get("pct.1", float("nan")),
                    "pct_expr_control": row.get("pct.2", float("nan")),
                }
            )

    joint_df = pd.DataFrame(joint_rows)
    if not joint_df.empty:
        joint_df = joint_df.sort_values(
            by=["n_wgcna_datasets", "avg_log2FC"], ascending=[False, False]
        )

    # 4.5 核心铁死亡基因覆盖
    core_genes = {"ACSL4", "GPX4", "PTGS2", "HMOX1", "TFRC", "SLC7A11", "FTH1", "FTL"}
    summary["core_ferroptosis_in_wgcna"] = sorted(core_genes & all_wgcna_genes)
    summary["core_ferroptosis_in_sc_deg"] = sorted(core_genes & sc_deg_genes)
    summary["core_ferroptosis_in_joint"] = sorted(core_genes & set(joint_genes))

    return joint_df, summary


# ---------------------------------------------------------------------------
# 5. 主流程与报告输出
# ---------------------------------------------------------------------------
def save_report(
    input_details: Dict[str, Any],
    run_status: Dict[str, Any],
    output_details: Dict[str, Any],
    joint_summary: Dict[str, Any],
    joint_df: pd.DataFrame,
    logs: Dict[str, str],
    warnings: List[str],
) -> None:
    """保存标准化质量与联合分析报告."""
    report = {
        "project_root": str(PROJECT_ROOT),
        "warnings": warnings,
        "input_validation": input_details,
        "module_execution": run_status,
        "output_validation": output_details,
        "joint_analysis": joint_summary,
    }

    report_path = REPORT_DIR / "L2_integrated_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"联合分析报告已保存: {report_path}")

    # CSV 汇总
    summary_rows = []
    for category, items in {
        "l1_outputs": input_details.get("l1_outputs", {}),
        "bulk_datasets": input_details.get("bulk_datasets", {}),
        "single_cell_dataset": {"GSE174574": str(input_details.get("single_cell_dataset", {}).get("path", ""))},
        "bulk_outputs": output_details.get("bulk_outputs", {}),
        "single_cell_outputs": output_details.get("single_cell_outputs", {}),
    }.items():
        for name, value in items.items():
            summary_rows.append({"category": category, "item": name, "status": str(value)})

    summary_df = pd.DataFrame(summary_rows)
    summary_path = REPORT_DIR / "L2_integrated_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"汇总表已保存: {summary_path}")

    # 联合基因表
    joint_path = REPORT_DIR / "L2_joint_genes.csv"
    joint_df.to_csv(joint_path, index=False, encoding="utf-8-sig")
    logger.info(f"联合基因表已保存: {joint_path}")

    # 保留关键日志片段
    for module, log in logs.items():
        log_path = REPORT_DIR / f"L2_{module}_log.txt"
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(log)


def main(skip_if_valid: bool = True) -> int:
    """L2 双模块联合分析主入口."""
    logger.info("=" * 60)
    logger.info("L2 双模块联合分析与质量检查")
    logger.info("=" * 60)

    # Step 1: 输入验证
    input_ok, input_errors, input_warnings, input_details = validate_inputs()
    if input_warnings:
        logger.warning("输入验证警告:")
        for w in input_warnings:
            logger.warning(f"  - {w}")
    if not input_ok:
        logger.error("输入验证未通过:")
        for e in input_errors:
            logger.error(f"  - {e}")
        save_report(input_details, {}, {}, {}, pd.DataFrame(), {}, input_warnings)
        return 1
    logger.info("输入验证通过")

    # Step 2: 运行 L2 模块
    run_status, logs = run_l2_modules(skip_if_valid=skip_if_valid)

    # Step 3: 输出验证
    output_ok, output_errors, output_details = validate_outputs()
    if not output_ok:
        logger.error("输出验证未通过:")
        for e in output_errors:
            logger.error(f"  - {e}")

    # Step 4: 联合分析 (仅在输出有效时)
    if output_ok:
        joint_df, joint_summary = joint_analysis()
        logger.info(
            f"联合分析完成: WGCNA ACSL4 模块并集 {joint_summary['wgcna_acsl4_union_genes']} 个基因, "
            f"与单细胞 DEG 交集 {joint_summary['joint_acsl4_sc_deg_genes']} 个基因"
        )
    else:
        joint_df = pd.DataFrame()
        joint_summary = {"error": "输出验证未通过, 跳过联合分析"}

    # Step 5: 保存报告
    save_report(
        input_details, run_status, output_details, joint_summary, joint_df, logs,
        input_warnings
    )

    final_ok = input_ok and output_ok
    if final_ok:
        logger.info("=" * 60)
        logger.info("L2 双模块联合分析完成, 所有检查通过")
        logger.info("=" * 60)
        return 0
    else:
        logger.error("=" * 60)
        logger.error("L2 双模块联合分析完成, 但存在未通过项")
        logger.error("=" * 60)
        return 1


if __name__ == "__main__":
    skip_arg = "--skip-if-valid" in sys.argv
    sys.exit(main(skip_if_valid=skip_arg))
