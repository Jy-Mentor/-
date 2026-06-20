#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
improve_training_data.py — 训练数据补充与优化脚本
====================================================
基于 Web/GitHub 搜索收集的公开数据源，对当前项目的训练输入文件进行补充：
1. 从 GenAge 下载人类衰老基因，补充 disease_gene_associations.csv 的 Aging 条目。
2. 从 PanglaoDB GitHub 镜像下载细胞类型注释标记基因，补充 celltype_marker_genes.csv，
   重点覆盖 Neuron、Pericyte 等当前缺失的脑细胞类型。
3. 生成补充过程日志与数据质量诊断增量报告。

所有数据均来自真实外部文件；失败时会记录警告，不会静默补零或模拟数据。
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import traceback
import zipfile
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
L1_GENE_LEVEL = PROJECT_ROOT / "L1" / "l1_results" / "L1_gene_level_analysis.csv"
FERR_GENE_FILE = PROJECT_ROOT / "铁衰老基因.txt"
REPORT_DIR = PROJECT_ROOT / "L3_results"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

GENAGE_URL = "https://hagr.ageing-map.org/genes/human_genes.zip"
PANGLAODB_BASE = "https://raw.githubusercontent.com/oscar-franzen/PanglaoDB/master/data"


def safe_get(url: str, timeout: int = 60, headers: dict | None = None) -> requests.Response:
    """带简单重试的 GET 请求。"""
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            logger.warning(f"  下载 {url} 第 {attempt + 1}/3 次失败: {e}")
    raise last_err


def load_project_gene_set() -> set[str]:
    """加载项目当前已有的基因集合（铁衰老基因 + L1 基因级分析中的基因）。"""
    genes: set[str] = set()
    if FERR_GENE_FILE.exists():
        try:
            with open(FERR_GENE_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    g = line.strip()
                    if g:
                        genes.add(g.upper())
            logger.info(f"  从 {FERR_GENE_FILE.name} 加载 {len(genes)} 个铁衰老基因")
        except Exception:
            traceback.print_exc()
            logger.error("  读取铁衰老基因文件失败")

    if L1_GENE_LEVEL.exists():
        try:
            df = pd.read_csv(L1_GENE_LEVEL, usecols=["gene"])
            l1_genes = set(df["gene"].astype(str).str.strip().str.upper())
            logger.info(f"  从 {L1_GENE_LEVEL.name} 加载 {len(l1_genes)} 个 L1 基因")
            genes |= l1_genes
        except Exception:
            traceback.print_exc()
            logger.error("  读取 L1 基因级分析文件失败")

    # 额外从现有网络文件中收集基因名
    for net_file in [
        NETWORK_DIR / "compound_target_edges.csv",
        NETWORK_DIR / "disease_gene_associations.csv",
        NETWORK_DIR / "gene_coexp_edges.csv",
        NETWORK_DIR / "string_ppi_edges.csv",
        NETWORK_DIR / "trrust_tf_target.csv",
    ]:
        if not net_file.exists():
            continue
        try:
            df = pd.read_csv(net_file)
            for col in df.columns:
                if "gene" in col.lower() or "target" in col.lower():
                    genes |= set(df[col].astype(str).str.strip().str.upper())
        except Exception:
            traceback.print_exc()

    logger.info(f"  项目基因集合大小: {len(genes)}")
    return genes


def download_genage_human_genes() -> pd.DataFrame | None:
    """从 GenAge 下载人类衰老基因列表。"""
    logger.info("=== 下载 GenAge 人类衰老基因 ===")
    try:
        resp = safe_get(GENAGE_URL, timeout=120)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not names:
                logger.warning("  GenAge zip 中未找到 CSV 文件")
                return None
            with zf.open(names[0]) as fh:
                df = pd.read_csv(fh)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        logger.info(f"  GenAge: 下载 {len(df)} 条人类衰老基因")
        return df
    except Exception:
        traceback.print_exc()
        logger.error("  GenAge 下载失败")
        return None


def download_panglaodb_markers_from_github() -> pd.DataFrame | None:
    """从 PanglaoDB GitHub 镜像下载细胞类型注释标记基因。"""
    logger.info("=== 从 PanglaoDB GitHub 镜像下载细胞类型标记基因 ===")
    url = f"{PANGLAODB_BASE}/cell_type_annotations_markers.txt"
    try:
        resp = safe_get(url, timeout=180)
        # 文件无 header，逗号分隔：SRA, SRS, cluster, gene
        df = pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=["sra", "srs", "cluster", "gene"],
        )
        df["gene"] = df["gene"].astype(str).str.strip().str.upper()
        logger.info(f"  PanglaoDB markers: {len(df)} 行原始标记")
        return df
    except Exception:
        traceback.print_exc()
        logger.error("  PanglaoDB markers 下载失败")
        return None


def download_panglaodb_annotations() -> pd.DataFrame | None:
    """从 PanglaoDB GitHub 镜像下载细胞类型注释（cluster -> celltype 映射）。"""
    logger.info("=== 从 PanglaoDB GitHub 镜像下载细胞类型注释 ===")
    url = f"{PANGLAODB_BASE}/cell_type_annotations.txt"
    try:
        resp = safe_get(url, timeout=180)
        df = pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=["sra", "srs", "cluster", "celltype_raw", "p_value", "adj_p", "activity_score"],
        )
        df["celltype_raw"] = df["celltype_raw"].astype(str).str.strip()
        logger.info(f"  PanglaoDB annotations: {len(df)} 行")
        return df
    except Exception:
        traceback.print_exc()
        logger.error("  PanglaoDB annotations 下载失败")
        return None


def download_panglaodb_metadata() -> pd.DataFrame | None:
    """下载 PanglaoDB metadata 以判断样本组织来源。"""
    logger.info("=== 从 PanglaoDB GitHub 镜像下载样本 metadata ===")
    url = f"{PANGLAODB_BASE}/metadata.txt"
    try:
        resp = safe_get(url, timeout=120)
        df = pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=[
                "sra", "srs", "tissue", "protocol", "species",
                "instrument", "n_genes", "median_genes", "n_clusters",
                "is_tumor", "is_primary_adult", "is_cell_line",
            ],
        )
        logger.info(f"  PanglaoDB metadata: {len(df)} 个样本")
        return df
    except Exception:
        traceback.print_exc()
        logger.error("  PanglaoDB metadata 下载失败")
        return None


def download_panglaodb_celltype_desc() -> pd.DataFrame | None:
    """下载 PanglaoDB 细胞类型描述（用于规范命名）。"""
    logger.info("=== 从 PanglaoDB GitHub 镜像下载细胞类型描述 ===")
    url = f"{PANGLAODB_BASE}/cell_type_desc.txt"
    try:
        resp = safe_get(url, timeout=120)
        df = pd.read_csv(
            io.StringIO(resp.text),
            header=None,
            names=["celltype", "description", "synonyms"],
        )
        logger.info(f"  PanglaoDB cell_type_desc: {len(df)} 种细胞类型")
        return df
    except Exception:
        traceback.print_exc()
        logger.error("  PanglaoDB cell_type_desc 下载失败")
        return None


def normalize_celltype(name: str, desc_df: pd.DataFrame | None) -> str | None:
    """将 PanglaoDB 细胞类型名归一化为项目使用的 6 种脑细胞类型。"""
    name_lower = name.strip().lower()
    mapping = {
        "neuron": "Neuron",
        "neurons": "Neuron",
        "pyramidal cells": "Neuron",
        "gabaergic neurons": "Neuron",
        "glutaminergic neurons": "Neuron",
        "dopaminergic neurons": "Neuron",
        "cholinergic neurons": "Neuron",
        "serotonergic neurons": "Neuron",
        "noradrenergic neurons": "Neuron",
        "adrenergic neurons": "Neuron",
        "motor neurons": "Neuron",
        "interneurons": "Neuron",
        "purkinje neurons": "Neuron",
        "microglia": "Microglia",
        "astrocyte": "Astrocyte",
        "astrocytes": "Astrocyte",
        "bergmann glia": "Astrocyte",
        "oligodendrocyte": "Oligodendrocyte",
        "oligodendrocytes": "Oligodendrocyte",
        "oligodendrocyte progenitor cells": "Oligodendrocyte",
        "endothelial cells": "Endothelial",
        "endothelial cells (blood brain barrier)": "Endothelial",
        "pericytes": "Pericyte",
        "pericyte": "Pericyte",
        "vascular smooth muscle cells": "Pericyte",
        "smooth muscle cells": "Pericyte",
    }
    if name_lower in mapping:
        return mapping[name_lower]

    # 尝试同义词匹配
    if desc_df is not None and not desc_df.empty:
        for _, row in desc_df.iterrows():
            ct = str(row.get("celltype", "")).strip().lower()
            synonyms = str(row.get("synonyms", "")).strip().lower()
            if name_lower == ct or name_lower in synonyms.split(","):
                for key, val in mapping.items():
                    if key in ct or key in synonyms:
                        return val
    return None


def build_brain_celltype_markers(
    markers_df: pd.DataFrame,
    annotations_df: pd.DataFrame | None,
    metadata_df: pd.DataFrame | None,
    desc_df: pd.DataFrame | None,
    project_genes: set[str],
) -> pd.DataFrame:
    """从 PanglaoDB 标记基因中筛选脑/神经系统相关细胞类型的标记。"""
    logger.info("=== 构建脑细胞类型标记基因表 ===")

    if annotations_df is None or annotations_df.empty:
        logger.warning("  缺少 cluster->celltype 注释，无法构建细胞类型标记")
        return pd.DataFrame(columns=["celltype", "gene"])

    # 若 metadata 可用，先筛选脑/神经组织样本
    if metadata_df is not None and not metadata_df.empty:
        tissue = metadata_df["tissue"].astype(str).str.lower()
        brain_srs = set(
            metadata_df[
                tissue.str.contains("brain|neuro|cortex|hippocampus|cerebellum|spinal cord|ganglion")
            ]["srs"]
        )
        logger.info(f"  识别到 {len(brain_srs)} 个脑/神经相关样本 (SRS)")
        if brain_srs:
            markers_df = markers_df[markers_df["srs"].isin(brain_srs)].copy()
            annotations_df = annotations_df[annotations_df["srs"].isin(brain_srs)].copy()
            logger.info(f"  筛选后脑/神经样本 markers: {len(markers_df)} 行")

    # 合并注释，获取细胞类型名称
    merged = markers_df.merge(
        annotations_df[["sra", "srs", "cluster", "celltype_raw"]],
        on=["sra", "srs", "cluster"],
        how="left",
    )
    missing_annotation = merged["celltype_raw"].isna().sum()
    if missing_annotation:
        logger.warning(f"  {missing_annotation} 条标记无对应细胞类型注释，已丢弃")
    merged = merged.dropna(subset=["celltype_raw"]).copy()

    # 归一化细胞类型
    merged["celltype"] = merged["celltype_raw"].apply(lambda x: normalize_celltype(x, desc_df))
    merged = merged.dropna(subset=["celltype"]).copy()
    merged = merged[["celltype", "gene"]].drop_duplicates()

    # 只保留在项目基因集合中的标记（避免引入孤立节点）
    before = len(merged)
    merged = merged[merged["gene"].isin(project_genes)].copy()
    after = len(merged)
    logger.info(f"  细胞类型标记: 过滤前 {before} 条, 过滤后 {after} 条（在项目基因集合中）")

    if after < before:
        logger.warning(
            f"  有 {before - after} 条标记基因不在项目基因集合中，已丢弃；"
            "这些基因若后续纳入图节点需同步扩展基因特征。"
        )

    return merged.sort_values(["celltype", "gene"]).reset_index(drop=True)


def backup_file(path: Path) -> Path:
    """为现有网络文件创建带时间戳的备份。"""
    backup_path = path.with_suffix(f"{path.suffix}.backup")
    shutil.copy2(path, backup_path)
    logger.info(f"  已备份 {path.name} -> {backup_path.name}")
    return backup_path


def integrate_aging_genes(genage_df: pd.DataFrame | None, project_genes: set[str]) -> dict:
    """将 GenAge 衰老基因整合进 disease_gene_associations.csv。"""
    result = {"added": 0, "skipped_not_in_project": 0, "output_rows": 0}
    disease_file = NETWORK_DIR / "disease_gene_associations.csv"
    if not disease_file.exists():
        logger.error(f"  {disease_file} 不存在，跳过 Aging 整合")
        return result

    if genage_df is None or genage_df.empty:
        logger.warning("  GenAge 数据不可用，跳过 Aging 整合")
        return result

    backup_file(disease_file)
    df = pd.read_csv(disease_file)
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()

    # 移除现有 Aging 条目以便重新生成
    existing_non_aging = df[df["disease"] != "Aging"].copy()
    old_aging_count = (df["disease"] == "Aging").sum()
    logger.info(f"  现有 disease_gene_associations.csv: {len(df)} 行, 原 Aging {old_aging_count} 行")

    # 生成新的 Aging 条目，仅保留在项目基因集合中的基因
    aging_genes = set(genage_df["symbol"]) & project_genes
    result["skipped_not_in_project"] = len(set(genage_df["symbol"]) - project_genes)
    if result["skipped_not_in_project"]:
        logger.warning(
            f"  {result['skipped_not_in_project']} 个 GenAge 基因不在项目基因集合中，未加入 Aging 关联"
        )

    # 保留与 AD/CIRI 的重叠基因（一个基因可关联多种疾病，符合生物学事实）
    new_aging_genes = sorted(aging_genes)

    aging_rows = pd.DataFrame({"disease": "Aging", "gene": new_aging_genes})
    result["added"] = len(aging_rows)

    new_df = pd.concat([existing_non_aging, aging_rows], ignore_index=True)
    new_df = new_df.drop_duplicates(subset=["disease", "gene"]).sort_values(["disease", "gene"])
    new_df.to_csv(disease_file, index=False)
    result["output_rows"] = len(new_df)
    logger.info(
        f"  更新后 disease_gene_associations.csv: {len(new_df)} 行, "
        f"新增 Aging 基因 {result['added']} 个"
    )
    return result


def integrate_celltype_markers(markers_df: pd.DataFrame | None, project_genes: set[str]) -> dict:
    """将 PanglaoDB 脑细胞类型标记整合进 celltype_marker_genes.csv。"""
    result = {"added": 0, "output_rows": 0, "celltypes": set()}
    marker_file = NETWORK_DIR / "celltype_marker_genes.csv"
    if not marker_file.exists():
        logger.error(f"  {marker_file} 不存在，跳过 celltype 整合")
        return result

    if markers_df is None or markers_df.empty:
        logger.warning("  PanglaoDB 标记数据不可用，跳过 celltype 整合")
        return result

    backup_file(marker_file)
    df = pd.read_csv(marker_file)
    df["gene"] = df["gene"].astype(str).str.strip().str.upper()
    df["celltype"] = df["celltype"].astype(str).str.strip()
    logger.info(f"  现有 celltype_marker_genes.csv: {len(df)} 行, 细胞类型 {sorted(df['celltype'].unique())}")

    # 合并新标记
    combined = pd.concat([df, markers_df[["celltype", "gene"]]], ignore_index=True)
    combined = combined.drop_duplicates(subset=["celltype", "gene"]).sort_values(["celltype", "gene"])
    result["added"] = len(combined) - len(df)
    result["output_rows"] = len(combined)
    result["celltypes"] = set(combined["celltype"].unique())
    combined.to_csv(marker_file, index=False)
    logger.info(
        f"  更新后 celltype_marker_genes.csv: {len(combined)} 行, "
        f"新增 {result['added']} 条, 细胞类型 {sorted(result['celltypes'])}"
    )
    return result


def run_diagnostic() -> dict:
    """对更新后的关键网络文件做简单统计，返回增量报告。"""
    logger.info("=== 生成补充后数据诊断 ===")
    report: dict = {}
    for fname, key_cols in [
        ("disease_gene_associations.csv", ["disease", "gene"]),
        ("celltype_marker_genes.csv", ["celltype", "gene"]),
    ]:
        path = NETWORK_DIR / fname
        if not path.exists():
            report[fname] = {"error": "file not found"}
            continue
        df = pd.read_csv(path)
        report[fname] = {
            "rows": len(df),
            "unique_edges": len(df.drop_duplicates(subset=key_cols)),
            "unique_src": int(df[key_cols[0]].nunique()),
            "unique_dst": int(df[key_cols[1]].nunique()),
            "top_src": df[key_cols[0]].value_counts().head(5).to_dict(),
        }
    return report


def main() -> None:
    logger.info("开始训练数据补充与优化")

    # 1. 加载项目基因集合
    project_genes = load_project_gene_set()

    # 2. 下载外部数据
    genage_df = download_genage_human_genes()
    markers_df = download_panglaodb_markers_from_github()
    annotations_df = download_panglaodb_annotations()
    metadata_df = download_panglaodb_metadata()
    desc_df = download_panglaodb_celltype_desc()

    # 3. 构建脑细胞类型标记表
    brain_markers = pd.DataFrame()
    if markers_df is not None:
        brain_markers = build_brain_celltype_markers(
            markers_df, annotations_df, metadata_df, desc_df, project_genes
        )

    # 4. 整合到网络文件
    aging_result = integrate_aging_genes(genage_df, project_genes)
    marker_result = integrate_celltype_markers(brain_markers, project_genes)

    # 5. 生成增量报告
    diagnostic = run_diagnostic()

    report = {
        "data_sources": {
            "GenAge": {
                "url": GENAGE_URL,
                "downloaded_genes": len(genage_df) if genage_df is not None else 0,
            },
            "PanglaoDB_GitHub": {
                "markers_url": f"{PANGLAODB_BASE}/cell_type_annotations_markers.txt",
                "metadata_url": f"{PANGLAODB_BASE}/metadata.txt",
                "desc_url": f"{PANGLAODB_BASE}/cell_type_desc.txt",
                "brain_marker_rows": len(brain_markers),
            },
        },
        "integration": {
            "disease_gene_associations": aging_result,
            "celltype_marker_genes": marker_result,
        },
        "post_update_diagnostic": diagnostic,
    }

    report_path = REPORT_DIR / "training_data_improvement_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"补充报告已保存: {report_path}")

    logger.info("训练数据补充完成")


if __name__ == "__main__":
    main()
