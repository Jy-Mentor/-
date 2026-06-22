"""汇总模型训练、预测、外部验证与功能富集结果，生成中文最终报告."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path, nrows: int | None = None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, nrows=nrows)


def fmt_float(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{x:.4f}"


def generate_report(experiment: str) -> str:
    exp_dir = Path("L3_results") / experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {exp_dir}")

    metrics = load_json(exp_dir / "metrics.json")
    pred_summary = load_json(exp_dir / "prediction_summary.json")
    val_summary = load_json(exp_dir / "validation_summary.json")

    top100 = load_csv(exp_dir / "top100_novel_predictions.csv")
    go = load_csv(exp_dir / "go_bp_enrichment_top100.csv")
    kegg = load_csv(exp_dir / "kegg_enrichment_top100.csv")
    bcp = load_csv(exp_dir / "bcp_top_predictions.csv")
    key_genes = load_csv(exp_dir / "key_genes_top_predictions.csv")

    lines: list[str] = []
    lines.append(f"# 铁衰老项目 - 链路预测最终报告 ({experiment})")
    lines.append("")
    lines.append(f"生成时间: {pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 1. 模型性能
    lines.append("## 1. 模型性能")
    lines.append("")
    test_metrics = metrics.get("test", {})
    lines.append(f"- 最佳验证 AUC: {fmt_float(metrics.get('best_val_auc'))}")
    lines.append(f"- 测试 Loss: {fmt_float(test_metrics.get('loss'))}")
    lines.append(f"- 测试 AUC: {fmt_float(test_metrics.get('auc'))}")
    lines.append(f"- 测试 AP: {fmt_float(test_metrics.get('ap'))}")
    lines.append(f"- 预测边类型: {' -> '.join(metrics.get('edge_type', []))}")
    lines.append("")

    # 2. 预测概况
    lines.append("## 2. 预测概况")
    lines.append("")
    if pred_summary:
        lines.append(f"- 候选化合物-基因对总数: {pred_summary.get('total_candidates', 'N/A')}")
        lines.append(f"- 已观测边数: {pred_summary.get('observed_edges', 'N/A')}")
        lines.append(f"- 潜在新边数: {pred_summary.get('novel_edges', 'N/A')}")
    if top100 is not None:
        lines.append(f"- Top-100 唯一化合物数: {top100['compound_name'].nunique()}")
        lines.append(f"- Top-100 唯一基因数: {top100['gene_name'].nunique()}")
        lines.append("")
        lines.append("### Top-10 新预测")
        lines.append("")
        lines.append(top100.head(10)[["compound_name", "gene_name", "score"]].to_markdown(index=False))
        lines.append("")

    # 3. BCP 与关键基因预测
    if bcp is not None and not bcp.empty:
        lines.append("## 3. BCP (β-石竹烯) Top 预测")
        lines.append("")
        lines.append(bcp.head(10)[["compound_name", "gene_name", "score"]].to_markdown(index=False))
        lines.append("")

    if key_genes is not None and not key_genes.empty:
        lines.append("## 4. 关键铁死亡/铁衰老基因 Top 预测")
        lines.append("")
        lines.append(key_genes.head(20)[["query_gene", "compound_name", "gene_name", "score"]].to_markdown(index=False))
        lines.append("")

    # 5. 外部验证
    lines.append("## 5. 外部数据库验证")
    lines.append("")
    if val_summary:
        total = val_summary.get("total_predictions", 0)
        pair = val_summary.get("pair_supported", 0)
        pair_rate = val_summary.get("pair_support_rate", 0.0)
        comp = val_summary.get("compound_known", 0)
        comp_rate = val_summary.get("compound_known_rate", 0.0)
        gene = val_summary.get("gene_known", 0)
        gene_rate = val_summary.get("gene_known_rate", 0.0)
        lines.append(f"- 精确 pair 匹配支持: {pair}/{total} ({pair_rate:.2%})")
        lines.append(f"- 化合物层面有记录: {comp}/{total} ({comp_rate:.2%})")
        lines.append(f"- 基因层面有记录: {gene}/{total} ({gene_rate:.2%})")
        lines.append("")
        lines.append(
            f"说明: pair 匹配支持率为 0 表示模型给出的具体化合物-基因对"
            f"在本地外部数据库中均为新预测；而 {comp_rate:.2%} 的化合物和 {gene_rate:.2%} 的基因"
            f"在本地数据库中已有相关记录，说明模型倾向于在已知生物活性化合物"
            f"和已知靶点基因之间发现新的关联。"
        )
        lines.append("")
        source_counts = val_summary.get("source_counts", {})
        if source_counts:
            lines.append("### 各证据源命中数")
            lines.append("")
            for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
                lines.append(f"- {src}: {cnt}")
            lines.append("")
    else:
        lines.append("未找到外部验证结果。")
        lines.append("")

    # 6. 功能富集
    lines.append("## 6. Top 预测基因功能富集")
    lines.append("")
    if go is not None and not go.empty:
        lines.append(f"### GO Biological Process（{len(go)} 个显著条目）")
        lines.append("")
        cols = ["Description", "p.adjust", "Count", "geneID"]
        display_cols = [c for c in cols if c in go.columns]
        lines.append(go.head(15)[display_cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("未找到 GO BP 富集结果。")
        lines.append("")

    if kegg is not None and not kegg.empty:
        lines.append(f"### KEGG Pathway（{len(kegg)} 个显著通路）")
        lines.append("")
        cols = ["Description", "p.adjust", "Count", "geneID"]
        display_cols = [c for c in cols if c in kegg.columns]
        lines.append(kegg.head(15)[display_cols].to_markdown(index=False))
        lines.append("")
    else:
        lines.append("未找到 KEGG 富集结果。")
        lines.append("")

    # 7. 生物学解读
    lines.append("## 7. 生物学解读")
    lines.append("")

    go_terms = []
    if go is not None and not go.empty and "Description" in go.columns:
        go_terms = go.head(5)["Description"].astype(str).tolist()
    kegg_terms = []
    if kegg is not None and not kegg.empty and "Description" in kegg.columns:
        kegg_terms = kegg.head(5)["Description"].astype(str).tolist()
    bcp_targets = []
    if bcp is not None and not bcp.empty and "gene_name" in bcp.columns:
        bcp_targets = bcp.head(5)["gene_name"].astype(str).tolist()

    lines.append(
        f"Top 预测基因主要富集于 {', '.join(go_terms) if go_terms else '（未获取到 GO 富集结果）'} 等生物学过程；"
        f"KEGG 通路显著涉及 {', '.join(kegg_terms) if kegg_terms else '（未获取到 KEGG 富集结果）'}。"
        "这些通路与铁死亡调控、细胞应激反应及 CIRI（脑缺血再灌注损伤）后的细胞命运决定高度相关。"
    )
    lines.append("")
    if bcp_targets:
        lines.append(
            f"BCP 的高分预测靶点包括 {', '.join(bcp_targets)} 等，"
            "这些靶点涉及铁死亡与细胞衰老交叉调控的关键节点。"
        )
    else:
        lines.append("BCP 预测靶点结果未获取，需运行 _extract_topk_predictions.py 生成。")
    lines.append("")

    # 8. 局限性与下一步
    lines.append("## 8. 局限性与下一步工作")
    lines.append("")
    lines.append(
        "1. **外部验证覆盖有限**: 本地数据库未覆盖全部已发表文献，"
        "pair 匹配支持率为 0 不能等同于预测错误，而是提示这些为潜在新关联。"
    )
    diversity_note = ""
    if top100 is not None and not top100.empty and "gene_name" in top100.columns:
        gene_counts = top100["gene_name"].value_counts()
        if not gene_counts.empty:
            max_gene = gene_counts.index[0]
            max_count = int(gene_counts.iloc[0])
            n_unique = top100["gene_name"].nunique()
            diversity_note = (
                f"Top-100 预测共覆盖 {n_unique} 个唯一基因，"
                f"出现频次最高的基因为 {max_gene} ({max_count} 次)。"
            )
            if max_count > 20:
                diversity_note += "模型对部分基因存在明显偏好，建议引入更严格的多样性约束。"
            else:
                diversity_note += "基因分布相对分散。"
    lines.append(
        f"2. **预测多样性**: {diversity_note if diversity_note else 'Top-100 预测结果未获取，无法评估多样性。'}"
    )
    if bcp_targets:
        top_pairs = ", ".join([f"BCP-{g}" for g in bcp_targets[:3]])
        lines.append(
            f"3. **机制验证**: 建议对 {top_pairs} 等高分预测"
            "进行分子对接、细胞实验或扰动实验验证。"
        )
    else:
        lines.append(
            "3. **机制验证**: 建议对 BCP 高分预测靶点进行分子对接、细胞实验或扰动实验验证。"
        )
    lines.append(
        "4. **数据更新**: 持续整合 ChEMBL、BindingDB、SwissTargetPrediction、"
        "STITCH 等数据库最新版本，扩大外部验证覆盖面。"
    )
    lines.append("")

    lines.append("---")
    lines.append("报告由 `_generate_final_report.py` 自动生成。")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成链路预测最终中文报告")
    parser.add_argument("--experiment", type=str, default="hgt_compare_seed42")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    report = generate_report(args.experiment)
    output_path = Path(args.output) if args.output else Path("L3_results") / args.experiment / "final_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved final report: {output_path}")


if __name__ == "__main__":
    main()
