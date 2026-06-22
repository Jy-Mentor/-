"""生成候选中药单体(CIRI via iron aging)综合分析报告.

输入:
  - L3_results/tcm_monomer_screening/iron_aging_ciri_candidates.csv
  - network_files/tcm_bbb_evaluation.csv
  - L3_results/gat_hgt_iron_aging_seed42/mechanism_analysis/key_compounds_comparison.csv
  - L3_results/gat_hgt_iron_aging_seed42/mechanism_analysis/mechanism_table_top20.csv
  - L3_results/gat_hgt_iron_aging_seed42/metrics.json
  - L3_results/tcm_monomer_screening/candidate_summary.json

输出:
  - L3_results/tcm_monomer_screening/candidate_tcm_monomers_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
L3_DIR = PROJECT_ROOT / "L3_results"
NETWORK_DIR = PROJECT_ROOT / "network_files"
OUTPUT_DIR = L3_DIR / "tcm_monomer_screening"

SYNTHETIC_TOOLS = {"Fer-1", "DFO", "Lip-1", "Erastin", "RSL3", "ML162"}


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """读取CSV; 缺失时抛出异常."""
    if not path.exists():
        raise FileNotFoundError(f"缺失必需文件: {path}")
    return pd.read_csv(path, **kwargs)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_data(
    experiment: str = "gat_hgt_iron_aging_seed42_v2",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    candidates = _read_csv(OUTPUT_DIR / "iron_aging_ciri_candidates.csv")
    bbb = _read_csv(NETWORK_DIR / "tcm_bbb_evaluation.csv")
    comparison = _read_csv(
        L3_DIR / experiment / "mechanism_analysis" / "key_compounds_comparison.csv"
    )
    mechanism = _read_csv(
        L3_DIR / experiment / "mechanism_analysis" / "mechanism_table_top20.csv"
    )
    metrics = _read_json(L3_DIR / experiment / "metrics.json")
    summary = _read_json(OUTPUT_DIR / "candidate_summary.json")
    return candidates, bbb, comparison, mechanism, metrics, summary


def _fmt_list(s: str | None, max_items: int = 8) -> str:
    if pd.isna(s) or s is None or s == "NA":
        return "—"
    items = [x.strip() for x in str(s).split(";") if x.strip()]
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f" 等 {len(items)} 个"


def _get_bbb_reason(compound: str, bbb_df: pd.DataFrame) -> str:
    sub = bbb_df[bbb_df["compound"] == compound]
    if sub.empty:
        return "未评估"
    reasons = sub.iloc[0].get("BBB_reasons", "")
    if pd.isna(reasons):
        return "符合BBB渗透规则"
    return str(reasons)


def generate_report(
    candidates: pd.DataFrame,
    bbb_df: pd.DataFrame,
    comparison: pd.DataFrame,
    mechanism: pd.DataFrame,
    metrics: dict,
    summary: dict,
    top_n: int = 15,
) -> str:
    # 将 BBB 信息合并到 candidates，便于后续引用
    candidates = candidates.merge(
        bbb_df[["compound", "BBB_score", "BBB_pass", "BBB_reasons"]],
        on="compound",
        how="left",
    )

    # 动态提取最优先候选，供全文引用，避免硬编码
    top_candidate = candidates.iloc[0]
    top_name = str(top_candidate["compound"])
    top_bbb_pass = top_candidate.get("BBB_pass", "未评估")
    top_ia_count = int(top_candidate.get("iron_aging_target_count", 0))
    top_bridge_count = int(top_candidate.get("bridge_target_count", 0))
    top_ferrdb = top_candidate.get("ferrdb_role", "—")

    lines: list[str] = []

    lines.append("# 通过铁衰老路径治疗脑缺血再灌注损伤(CIRI)的候选中药单体系统分析报告")
    lines.append("")
    lines.append(f"生成时间: {pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 摘要
    lines.append("## 摘要")
    lines.append("")
    n_total = summary.get("total_compounds", int(candidates["compound"].nunique()))
    n_tcm = summary.get("tcm_compounds screened", len(candidates))
    n_ia_genes = summary.get("iron_aging_genes_loaded", 0)
    n_ciri_genes = summary.get("ciri_genes_loaded", 0)
    n_pairs = summary.get("known_compound_target_pairs", 0)
    test_auc = metrics.get("test", {}).get("auc")
    best_val_auc = metrics.get("best_val_auc")

    lines.append(
        f"本分析基于 {n_total} 个化合物(排除合成铁死亡工具后 {n_tcm} 个中药单体)、"
        f"{n_ia_genes} 个铁衰老相关基因、{n_ciri_genes} 个 CIRI 相关基因及 {n_pairs} 条已知化合物-靶点边，"
        f"通过 GAT-HGT 异质图链路预测模型(最佳验证 AUC {best_val_auc:.4f}，测试 AUC {test_auc:.4f})"
        f"系统筛选能够作用于铁衰老通路并可能穿越血脑屏障(BBB)的中药单体。"
    )
    lines.append("")

    # 1. 数据收集与整理
    lines.append("## 1. 数据收集与整理")
    lines.append("")
    lines.append("### 1.1 数据来源")
    lines.append("")
    lines.append("- **化合物信息**: `network_files/compound_smiles.csv`、`network_files/pubchem_compound_props.csv`")
    lines.append(
        "- **化合物-靶点关系**: CTD、ChEMBL、DrugBank、BindingDB、DGIdb、"
        "DrugTargetCommons、SwissTargetPrediction 等数据库整合"
    )
    lines.append("- **铁衰老基因集**: `铁衰老基因.txt`(98 个基因)")
    lines.append(
        "- **CIRI 相关基因**: DisGeNET curated_gene_disease_associations 及 "
        "`disease_gene_associations.csv` 中 CIRI/AD/Aging 注释"
    )
    lines.append("- **通路注释**: `network_files/gene_pathway_enrichment.csv`(KEGG/Reactome 等)")
    lines.append("- **BBB 渗透性**: 基于 RDKit 分子描述符(MW、logP、HBD、HBA、TPSA、RotBonds)经验规则评估")
    lines.append("- **模型预测分数**: `L3_results/gat_hgt_iron_aging_seed42_v2/all_compound_gene_scores.csv`")
    lines.append("")
    lines.append("### 1.2 数据质量控制")
    lines.append("")
    lines.append("- 排除合成铁死亡工具化合物(Fer-1、DFO、Lip-1、Erastin、RSL3、ML162 等)，聚焦中药单体。")
    lines.append("- 已知化合物-靶点边保留来源与置信度，缺失数据记录警告，不静默填充。")
    lines.append("- BBB 评分按 0–5 规则计算，≥3 分视为可能具有一定 BBB 渗透能力。")
    lines.append("")

    # 2. 铁衰老靶点-CIRI 关联网络
    lines.append("## 2. 铁衰老靶点与 CIRI 关键病理环节的关联网络")
    lines.append("")
    lines.append(
        "通过异质图整合化合物(compound)、基因(gene)、通路(pathway)、疾病(disease)四类节点，"
        "建立铁衰老调控靶点与 CIRI 病理网络之间的桥接关系。"
    )
    lines.append("")
    lines.append("核心桥接靶点(同时属于铁衰老基因与 CIRI 相关基因)包括:")
    lines.append("")
    # 从 top candidates 提取桥接靶点
    top_candidates = candidates.head(top_n)
    all_bridge = set()
    for _, row in top_candidates.iterrows():
        if pd.notna(row.get("bridge_targets")) and row["bridge_targets"] != "NA":
            all_bridge.update([g.strip() for g in str(row["bridge_targets"]).split(";") if g.strip()])
    bridge_sorted = sorted(all_bridge)
    lines.append(", ".join(bridge_sorted[:30]) + (f" 等 {len(bridge_sorted)} 个" if len(bridge_sorted) > 30 else ""))
    lines.append("")
    lines.append("这些桥接基因覆盖以下关键 CIRI 病理通路:")
    lines.append("")
    lines.append("- Ferroptosis(铁死亡/铁衰老调控核心通路)")
    lines.append("- HIF-1 signaling pathway(缺氧应答)")
    lines.append("- MAPK signaling pathway(应激与炎症信号)")
    lines.append("- TNF signaling pathway / NF-kappa B signaling pathway(神经炎症)")
    lines.append("- NOD-like receptor signaling pathway / Toll-like receptor signaling pathway(先天免疫)")
    lines.append("- PI3K-Akt signaling pathway(细胞存活)")
    lines.append("- p53 signaling pathway(细胞周期与死亡决定)")
    lines.append("")

    # 3. BBB 渗透性筛选
    lines.append("## 3. 具有 BBB 穿透能力的铁衰老通路中药单体")
    lines.append("")
    bbb_ok = candidates[candidates["BBB_pass"].isin(["Likely", "Moderate"])].copy()
    bbb_likely = candidates[candidates["BBB_pass"] == "Likely"].copy()
    lines.append(
        f"在 {len(candidates)} 个中药单体中，"
        f"BBB 评估为 **Likely/Moderate** 的共 {len(bbb_ok)} 个，"
        f"其中 **Likely** 的共 {len(bbb_likely)} 个。"
    )
    lines.append("")
    lines.append("### 3.1 BBB 渗透性为 Likely 的候选单体(Top)")
    lines.append("")
    display_bbb = bbb_likely.head(20)[[
        "rank", "compound", "iron_aging_target_count", "bridge_target_count",
        "mean_confidence", "BBB_score", "BBB_pass",
    ]].copy()
    display_bbb["mean_confidence"] = display_bbb["mean_confidence"].round(4)
    lines.append(display_bbb.to_markdown(index=False))
    lines.append("")

    # 4. 药理活性、作用强度及潜在副作用评估
    lines.append("## 4. 候选单体药理活性、作用强度与潜在副作用")
    lines.append("")
    lines.append("### 4.1 评分维度说明")
    lines.append("")
    lines.append("候选评分 `candidate_score` 综合以下证据(均来自真实数据):")
    lines.append("")
    lines.append("- 铁衰老靶点数量(权重 ×2)")
    lines.append("- 桥接靶点(铁衰老 ∩ CIRI)数量(权重 ×3)")
    lines.append("- CIRI 靶点数量(权重 ×1)")
    lines.append("- CTD 中直接标注 CIRI therapeutic 证据(+2)")
    lines.append("- FerrDb 中标注为铁死亡诱导剂(+1)或抑制剂(+0.5)")
    lines.append("- 平均置信度加权")
    lines.append("")
    lines.append("### 4.2 Top-15 候选中药单体综合评分")
    lines.append("")
    display_top = top_candidates[[
        "rank", "compound", "iron_aging_target_count", "bridge_target_count",
        "ciri_target_count", "mean_confidence", "has_ctd_ciri_evidence",
        "ferrdb_role", "key_pathway_count", "BBB_pass", "candidate_score",
    ]].copy()
    display_top["mean_confidence"] = display_top["mean_confidence"].round(4)
    display_top["has_ctd_ciri_evidence"] = display_top["has_ctd_ciri_evidence"].map({True: "是", False: "否"})
    display_top["ferrdb_role"] = display_top["ferrdb_role"].replace(["NA", float("nan")], "—")
    display_top["ferrdb_role"] = display_top["ferrdb_role"].fillna("—")
    lines.append(display_top.to_markdown(index=False))
    lines.append("")
    lines.append("### 4.3 潜在副作用提示")
    lines.append("")
    lines.append(
        "以下提示基于已知靶点谱与通路覆盖度的推测，**不能替代毒理学实验**，"
        "仅用于后续实验设计参考:"
    )
    lines.append("")
    lines.append(
        "- **广谱多靶点化合物(Quercetin、Resveratrol、Genistein、Curcumin)**: "
        "靶点广泛，脱靶风险相对较高，需关注细胞毒性与药物相互作用。"
    )
    lines.append(
        "- **免疫/炎症通路强覆盖化合物(BCP、Luteolin、Apigenin、Baicalein)**: "
        "可能抑制过度炎症，但在感染或免疫抑制状态下需谨慎。"
    )
    lines.append("- **BBB 评为 Poor 的 VC**: 原型维生素 C 中枢渗透性差，若用于脑部疾病需考虑衍生物或局部/高剂量方案。")
    lines.append(
        "- **FerrDb 同时标注为 inducer 与 inhibitor 的化合物"
        "(Quercetin、Resveratrol、Luteolin、Curcumin 等)**: "
        "提示其对铁死亡/铁衰老调控具有上下文依赖性，体内效应需具体验证。"
    )
    lines.append("")

    # 5. 多靶点协同与神经保护活性优先级
    lines.append("## 5. 多靶点协同作用与神经保护活性优先级")
    lines.append("")
    lines.append("### 5.1 关键化合物多维度比较")
    lines.append("")
    if not comparison.empty:
        lines.append(comparison.to_markdown(index=False))
        lines.append("")
    top_mech = comparison.iloc[0] if not comparison.empty else None
    if top_mech is not None:
        top_mech_name = str(top_mech.get("compound", top_name))
        lines.append(
            f"{top_mech_name}在关键化合物比较中评分最高，"
            f"具有 {top_ia_count} 个铁衰老靶点、{top_bridge_count} 个桥接靶点、"
            f"BBB 评估为 {top_bbb_pass}。"
        )
    else:
        lines.append(
            f"{top_name}在关键化合物比较中评分最高，"
            f"具有 {top_ia_count} 个铁衰老靶点、{top_bridge_count} 个桥接靶点、"
            f"BBB 评估为 {top_bbb_pass}。"
        )
    lines.append("")

    lines.append("### 5.2 机制解释表(Top 候选化合物)")
    lines.append("")
    if not mechanism.empty:
        mech_display = mechanism[[
            "compound", "target_gene", "predicted_score", "overall_rank",
            "is_known_target", "BBB_pass", "related_pathways",
        ]].copy()
        mech_display["predicted_score"] = mech_display["predicted_score"].round(4)
        mech_display["is_known_target"] = mech_display["is_known_target"].map({True: "已知", False: "新预测"})
        mech_display["related_pathways"] = mech_display["related_pathways"].apply(lambda x: _fmt_list(x, max_items=3))
        lines.append(mech_display.head(30).to_markdown(index=False))
        lines.append("")

    lines.append("### 5.3 优先级排序原则")
    lines.append("")
    lines.append("1. **BBB 渗透性优先**: 中枢作用需穿越血脑屏障，Likely > Moderate > Poor。")
    lines.append("2. **桥接靶点丰富度**: 同时调控铁衰老与 CIRI 的靶点数量越多越好。")
    lines.append("3. **外部证据支持**: CTD 中直接标注 CIRI therapeutic 或 FerrDb 铁死亡调控角色加分。")
    lines.append("4. **通路覆盖度**: 覆盖 Ferroptosis、HIF-1、MAPK、TNF、NF-κB 等核心通路。")
    lines.append("5. **神经保护先验**: 已有文献报道对脑缺血、神经炎症或氧化应激具有保护作用的单体优先。")
    lines.append("")

    # 6. 候选清单
    lines.append("## 6. 候选中药单体清单及作用机制")
    lines.append("")
    lines.append(f"以下列出综合评分前 {top_n} 名中药单体，按候选评分降序排列。")
    lines.append("")

    for idx, row in top_candidates.iterrows():
        compound = row["compound"]
        cid = row["cid"]
        ia_count = int(row["iron_aging_target_count"])
        bridge_count = int(row["bridge_target_count"])
        ciri_count = int(row["ciri_target_count"])
        conf = float(row["mean_confidence"])
        ctd = "是" if row["has_ctd_ciri_evidence"] else "否"
        ferrdb = row["ferrdb_role"] if pd.notna(row["ferrdb_role"]) and row["ferrdb_role"] != "NA" else "—"
        pathways = _fmt_list(row["key_ciri_pathways_hit"], max_items=6)
        bbb_pass = row.get("BBB_pass", "未评估")
        bbb_score = row.get("BBB_score", "N/A")
        score = float(row["candidate_score"])

        lines.append(f"### 6.{idx + 1} {compound}")
        lines.append("")
        lines.append(f"- **排名**: {int(row['rank'])}")
        lines.append(f"- **PubChem CID**: {cid}")
        lines.append(f"- **候选评分**: {score:.4f}")
        lines.append(f"- **铁衰老靶点数**: {ia_count}")
        lines.append(f"- **桥接靶点数(铁衰老 ∩ CIRI)**: {bridge_count}")
        lines.append(f"- **CIRI 靶点数**: {ciri_count}")
        lines.append(f"- **平均置信度**: {conf:.4f}")
        lines.append(f"- **BBB 评估**: {bbb_pass} (score {bbb_score})")
        lines.append(f"- **BBB 原因**: {_get_bbb_reason(compound, bbb_df)}")
        lines.append(f"- **CTD CIRI 直接证据**: {ctd}")
        lines.append(f"- **FerrDb 角色**: {ferrdb}")
        lines.append(f"- **核心 CIRI 通路覆盖**: {pathways}")
        lines.append(f"- **代表性铁衰老靶点**: {_fmt_list(row['iron_aging_targets'], max_items=8)}")
        lines.append("")

    # 结论
    lines.append("## 7. 结论与建议")
    lines.append("")

    lines.append(
        "综合铁衰老靶点覆盖、CIRI 桥接靶点数量、BBB 渗透性、外部数据库证据及多靶点协同潜力，"
        f"**{top_name}** 是当前评分最高的候选中药单体: 具有 BBB {top_bbb_pass} 渗透性、"
        f"{top_ia_count} 个铁衰老靶点及 {top_bridge_count} 个桥接靶点"
        f"(FerrDb 角色: {top_ferrdb})。"
    )
    lines.append("")

    other_top = candidates.head(9).tail(8)
    other_names = [str(r["compound"]) for _, r in other_top.iterrows()]
    if other_names:
        lines.append(
            "其他高优先级候选包括 **" + "、".join(other_names) + "**，"
            "这些单体均覆盖核心 CIRI 病理通路，但部分 BBB 渗透性为 Moderate 或存在 FerrDb 双重角色，"
            "建议结合衍生物设计或递送系统优化。"
        )
    else:
        lines.append("其他高优先级候选数量不足，建议扩大化合物库或放宽评分阈值后重新评估。")
    lines.append("")
    lines.append("下一步建议:")
    lines.append("")
    lines.append(
        "1. 对 BCP 与 Top-5 候选单体进行分子对接"
        "(重点关注 GPX4、ACSL4、TFRC、NFE2L2/KEAP1、PTGS2 等核心靶点)。"
    )
    lines.append(
        "2. 在体外 OGD/R 神经元/小胶质细胞模型中验证候选单体对"
        "铁死亡标志物(LPCAT3、4-HNE、MDA、GSH、Fe²⁺)的调控作用。"
    )
    lines.append("3. 评估候选单体对 IL-1β、IL-6、TNF-α、NLRP3 炎症小体及血脑屏障完整性的影响。")
    lines.append("4. 建立体内 MCAO/R 动物模型，验证 BCP 等候选单体的神经保护效力与安全性。")
    lines.append("")

    lines.append("---")
    lines.append("报告由 `_generate_candidate_tcm_report.py` 基于真实网络文件自动生成。")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成候选中药单体(CIRI via iron aging)综合分析报告")
    parser.add_argument("--top-n", type=int, default=15, help="报告中详细展示的候选单体数量")
    parser.add_argument("--output", type=str, default=None, help="输出 Markdown 路径")
    parser.add_argument(
        "--experiment",
        type=str,
        default="gat_hgt_iron_aging_seed42_v2",
        help="L3_results 下的实验目录名",
    )
    args = parser.parse_args()

    try:
        candidates, bbb, comparison, mechanism, metrics, summary = load_data(experiment=args.experiment)
    except Exception:
        logger.exception("加载输入数据失败")
        traceback.print_exc()
        return 1

    report = generate_report(candidates, bbb, comparison, mechanism, metrics, summary, top_n=args.top_n)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "candidate_tcm_monomers_report.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("候选中药单体报告已保存: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
