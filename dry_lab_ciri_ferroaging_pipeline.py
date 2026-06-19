#!/usr/bin/env python3
"""干实验技术路线：铁衰老 × CIRI × GAT-HGT 综合分析流程.

本脚本基于项目已有数据执行以下分析:
1. CIRI 单细胞 DEG 中的铁衰老基因提取与统计
2. 铁衰老基因与 DisGeNET/OpenTargets CIRI 疾病基因取交集
3. 按细胞类型和通路类别构建铁衰老调控网络
4. 解析 GAT-HGT 输出 (Hub 基因、化合物靶向、细胞通讯)
5. 生成可视化图表与中间结果文件, 供 R 富集分析和报告使用
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODULE2_DIR = PROJECT_ROOT / "module2_results"
NETWORK_DIR = PROJECT_ROOT / "network_files"
L3_DIR = PROJECT_ROOT / "L3_results"
REPORT_DIR = PROJECT_ROOT / "dry_lab_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_ferroaging_genes() -> set:
    """加载铁衰老基因集 (来自 铁衰老基因.txt)."""
    fpath = PROJECT_ROOT / "铁衰老基因.txt"
    genes = set()
    if fpath.exists():
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                g = line.strip()
                if g and not g.startswith("#"):
                    genes.add(g.upper())
    logger.info(f"铁衰老基因集: {len(genes)} 个")
    return genes


def load_sc_deg(path: Path = None) -> pd.DataFrame:
    """加载 CIRI 单细胞 DEG (MCAO vs Sham)."""
    if path is None:
        path = MODULE2_DIR / "cell_type_DEG_MCAO_vs_Sham.csv"
    df = pd.read_csv(path)
    df["names"] = df["names"].astype(str).str.upper()
    # 统一列名
    if "logfoldchanges" in df.columns:
        df = df.rename(columns={"logfoldchanges": "log2FC"})
    if "pvals_adj" in df.columns:
        df = df.rename(columns={"pvals_adj": "padj"})
    logger.info(f"单细胞 DEG: {len(df)} 条记录, {df['cell_type'].nunique()} 种细胞类型")
    return df


def load_ciri_disease_genes() -> dict:
    """加载 DisGeNET 与 OpenTargets CIRI 疾病-基因关联."""
    result = {}
    for src, fname in [("DisGeNET", "disgenet_ciri_genes.csv"),
                       ("OpenTargets", "opentargets_ciri_genes.csv")]:
        fpath = NETWORK_DIR / fname
        if fpath.exists():
            df = pd.read_csv(fpath)
            gene_col = "gene_symbol" if "gene_symbol" in df.columns else "gene"
            genes = set(df[gene_col].dropna().astype(str).str.upper())
            result[src] = genes
            logger.info(f"{src} CIRI 基因: {len(genes)} 个")
        else:
            logger.warning(f"{src} CIRI 文件不存在: {fpath}")
            result[src] = set()
    return result


def extract_ferroaging_deg(deg_df: pd.DataFrame, ferro_genes: set) -> pd.DataFrame:
    """提取 DEG 中属于铁衰老基因集的条目."""
    hits = deg_df[deg_df["names"].isin(ferro_genes)].copy()
    hits["abs_log2FC"] = hits["log2FC"].abs()
    hits = hits.sort_values(["cell_type", "abs_log2FC"], ascending=[True, False])
    logger.info(f"铁衰老相关 DEG: {len(hits)} 条 ({hits['names'].nunique()} 个基因)")
    return hits


def summarize_by_cell_type(hits: pd.DataFrame) -> pd.DataFrame:
    """按细胞类型统计铁衰老 DEG."""
    summary = []
    for cell_type, group in hits.groupby("cell_type"):
        up = (group["log2FC"] > 0).sum()
        down = (group["log2FC"] < 0).sum()
        summary.append({
            "cell_type": cell_type,
            "n_deg": len(group),
            "n_up": up,
            "n_down": down,
            "mean_log2FC": group["log2FC"].mean(),
            "top_gene": group.loc[group["abs_log2FC"].idxmax(), "names"],
            "top_log2FC": group["abs_log2FC"].max(),
        })
    return pd.DataFrame(summary).sort_values("n_deg", ascending=False)


def build_ferroaging_ciri_network(
    hits: pd.DataFrame,
    ciri_genes: dict,
    ferro_genes: set,
) -> pd.DataFrame:
    """构建铁衰老-CIRI 关联网络: 基因-细胞类型-疾病来源."""
    rows = []
    all_ciri = set().union(*ciri_genes.values())
    for gene in sorted(ferro_genes & all_ciri):
        cell_types = sorted(hits[hits["names"] == gene]["cell_type"].unique())
        sources = [s for s, gset in ciri_genes.items() if gene in gset]
        rows.append({
            "gene": gene,
            "cell_types": ";".join(cell_types) if cell_types else "NA",
            "n_cell_types": len(cell_types),
            "disease_sources": ";".join(sources),
            "in_sc_deg": gene in set(hits["names"]),
        })
    df = pd.DataFrame(rows)
    logger.info(f"铁衰老 × CIRI 关联网络: {len(df)} 个节点")
    return df


def analyze_l3_results() -> dict:
    """解析 GAT-HGT 输出结果."""
    results = {}

    # Hub 基因排名
    hub_path = L3_DIR / "L3_hub_gene_ranking.csv"
    if hub_path.exists():
        hub_df = pd.read_csv(hub_path)
        results["hub"] = hub_df
        logger.info(f"Hub 基因排名: {len(hub_df)} 个")

    # 化合物靶向 ACSL4
    target_path = L3_DIR / "L3_compound_target_summary.csv"
    if target_path.exists():
        target_df = pd.read_csv(target_path)
        results["compound_target"] = target_df
        # 提取 BCP
        bcp_row = target_df[target_df["compound"].str.contains("BCP|Caryophyllene", case=False, na=False)]
        if not bcp_row.empty:
            results["bcp_acsl4_prob"] = bcp_row.iloc[0]["ACSL4_probability"]
        # 提取 VC
        vc_row = target_df[target_df["compound"].str.contains("VC|Vitamin C|Ascorb", case=False, na=False)]
        if not vc_row.empty:
            results["vc_acsl4_prob"] = vc_row.iloc[0]["ACSL4_probability"]

    # 细胞通讯流
    comm_path = L3_DIR / "L3_cell_communication_flow.csv"
    if comm_path.exists():
        comm_df = pd.read_csv(comm_path)
        results["communication"] = comm_df

    return results


def plot_cell_type_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    """绘制细胞类型铁衰老 DEG 统计图."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 上下调堆叠条形图
    summary_sorted = summary.sort_values("n_deg", ascending=True)
    axes[0].barh(summary_sorted["cell_type"], summary_sorted["n_up"],
                 color="#E74C3C", label="Up-regulated")
    axes[0].barh(summary_sorted["cell_type"], -summary_sorted["n_down"],
                 color="#3498DB", label="Down-regulated")
    axes[0].set_xlabel("Number of ferro-aging DEGs")
    axes[0].set_title("Ferro-aging DEGs by Cell Type (MCAO vs Sham)")
    axes[0].legend()
    axes[0].axvline(0, color="black", linewidth=0.8)

    # 平均 log2FC
    axes[1].bar(summary_sorted["cell_type"], summary_sorted["mean_log2FC"],
                color=["#E74C3C" if v > 0 else "#3498DB" for v in summary_sorted["mean_log2FC"]])
    axes[1].set_ylabel("Mean log2FC")
    axes[1].set_title("Mean log2FC of Ferro-aging DEGs")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    out_path = out_dir / "fig1_cell_type_ferroaging_deg.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved: {out_path}")


def plot_top_genes_heatmap(hits: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    """绘制 top 铁衰老基因热图."""
    # 每个细胞类型取 top N
    top_hits = hits.groupby("cell_type").apply(
        lambda x: x.nlargest(top_n, "abs_log2FC"), include_groups=False
    ).reset_index()

    pivot = top_hits.pivot_table(
        index="names", columns="cell_type", values="log2FC", aggfunc="mean"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-3, vmax=3)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Top Ferro-aging DEGs across Cell Types (log2FC)")
    plt.colorbar(im, ax=ax, label="log2FC")
    plt.tight_layout()
    out_path = out_dir / "fig2_top_ferroaging_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved: {out_path}")


def plot_gsea_results(out_dir: Path) -> None:
    """读取 GSEA 结果 CSV 并绘制点图."""
    for fname, title, prefix in [
        ("gsea_go_bp.csv", "GSEA GO BP", "go_bp"),
        ("gsea_kegg.csv", "GSEA KEGG", "kegg"),
        ("gsea_reactome.csv", "GSEA Reactome", "reactome"),
    ]:
        fpath = out_dir / fname
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        df = df.sort_values("p.adjust").head(15)
        df["-log10(padj)"] = -np.log10(df["p.adjust"].replace(0, 1e-300))
        df["color"] = df["NES"].apply(lambda x: "#E74C3C" if x > 0 else "#3498DB")

        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
        ax.scatter(df["NES"], range(len(df)), s=df["-log10(padj)"] * 30,
                   c=df["color"], alpha=0.7, edgecolors="black")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Description"], fontsize=8)
        ax.set_xlabel("Normalized Enrichment Score (NES)")
        ax.set_title(title)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        plt.tight_layout()
        out_path = out_dir / f"fig3_gsea_{prefix}_dotplot.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Figure saved: {out_path}")


def plot_hub_genes(out_dir: Path) -> None:
    """绘制 GAT-HGT Hub 基因前 20 条形图."""
    hub_path = L3_DIR / "L3_hub_gene_ranking.csv"
    if not hub_path.exists():
        return
    hub = pd.read_csv(hub_path).head(20).sort_values("hub_score")
    fig, ax = plt.subplots(figsize=(8, max(6, len(hub) * 0.35)))
    ax.barh(hub["gene"], hub["hub_score"], color="#9B59B6")
    ax.set_xlabel("Hub Score")
    ax.set_title("Top 20 Hub Genes in GAT-HGT Heterogeneous Graph")
    plt.tight_layout()
    out_path = out_dir / "fig4_hub_genes.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved: {out_path}")


def plot_acsl4_compounds(out_dir: Path) -> None:
    """绘制 ACSL4 候选化合物相似性条形图."""
    top_path = L3_DIR / "L3_top30_ACSL4_candidate_compounds.csv"
    if not top_path.exists():
        return
    df = pd.read_csv(top_path).head(15).sort_values("similarity_to_ACSL4")
    colors = ["#E74C3C" if "BCP" in c or "Caryophyllene" in c else "#3498DB" for c in df["compound"]]
    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.35)))
    ax.barh(df["compound"], df["similarity_to_ACSL4"], color=colors)
    ax.set_xlabel("Similarity to ACSL4 Embedding")
    ax.set_title("Top 15 ACSL4 Candidate Compounds")
    plt.tight_layout()
    out_path = out_dir / "fig5_acsl4_compounds.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Figure saved: {out_path}")


def export_for_r_enrichment(hits: pd.DataFrame, out_dir: Path) -> Path:
    """导出用于 R clusterProfiler 富集分析的基因列表."""
    gene_list = out_dir / "ferroaging_deg_gene_list.csv"
    hits[["names", "cell_type", "log2FC", "padj"]].drop_duplicates("names").to_csv(
        gene_list, index=False, encoding="utf-8-sig"
    )
    logger.info(f"富集分析基因列表: {gene_list}")
    return gene_list


def main() -> int:
    logger.info("=" * 60)
    logger.info("启动铁衰老 × CIRI 干实验分析流程")
    logger.info("=" * 60)

    # 1. 数据加载
    ferro_genes = load_ferroaging_genes()
    deg_df = load_sc_deg()
    ciri_genes = load_ciri_disease_genes()

    # 2. 铁衰老 × CIRI DEG 关联
    hits = extract_ferroaging_deg(deg_df, ferro_genes)
    summary = summarize_by_cell_type(hits)
    network_df = build_ferroaging_ciri_network(hits, ciri_genes, ferro_genes)

    # 3. 保存中间结果
    hits.to_csv(REPORT_DIR / "ferroaging_ciri_sc_deg.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(REPORT_DIR / "ferroaging_deg_by_cell_type.csv", index=False, encoding="utf-8-sig")
    network_df.to_csv(REPORT_DIR / "ferroaging_ciri_network.csv", index=False, encoding="utf-8-sig")

    # 4. GAT-HGT 结果解析
    l3_results = analyze_l3_results()
    if "hub" in l3_results:
        l3_results["hub"].head(30).to_csv(
            REPORT_DIR / "l3_hub_top30.csv", index=False, encoding="utf-8-sig"
        )
    if "compound_target" in l3_results:
        l3_results["compound_target"].head(30).to_csv(
            REPORT_DIR / "l3_compound_target_top30.csv", index=False, encoding="utf-8-sig"
        )

    # 5. 可视化
    plot_cell_type_summary(summary, REPORT_DIR)
    plot_top_genes_heatmap(hits, REPORT_DIR)
    plot_gsea_results(REPORT_DIR)
    plot_hub_genes(REPORT_DIR)
    plot_acsl4_compounds(REPORT_DIR)

    # 6. 导出 R 富集输入
    export_for_r_enrichment(hits, REPORT_DIR)

    # 7. 汇总信息
    logger.info("=" * 60)
    logger.info("分析摘要")
    logger.info("=" * 60)
    logger.info(f"铁衰老基因总数: {len(ferro_genes)}")
    logger.info(f"CIRI 单细胞 DEG 总数: {len(deg_df)}")
    logger.info(f"铁衰老相关 DEG: {len(hits)} ({hits['names'].nunique()} 基因)")
    logger.info(f"DisGeNET CIRI 基因: {len(ciri_genes.get('DisGeNET', set()))}")
    logger.info(f"OpenTargets CIRI 基因: {len(ciri_genes.get('OpenTargets', set()))}")
    logger.info(f"铁衰老 × CIRI 关联网络节点: {len(network_df)}")
    if "bcp_acsl4_prob" in l3_results:
        logger.info(f"BCP-ACSL4 预测概率: {l3_results['bcp_acsl4_prob']:.4f}")
    if "vc_acsl4_prob" in l3_results:
        logger.info(f"VC-ACSL4 预测概率: {l3_results['vc_acsl4_prob']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
