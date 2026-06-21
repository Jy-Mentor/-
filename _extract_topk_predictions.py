"""提取训练好的链路预测模型在 compound->targets->gene 上的 top-k 预测.

输出包含：
- 所有候选 pair 的预测分数
- 排除训练/验证/测试已观测边后的新预测 top-k
- 关键化合物（如 BCP）和关键基因（如 ACSL4）的 top 关联
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from iron_aging.config import load_config
from iron_aging.data.graph_builder import HeteroGraphBuilder
from iron_aging.db.connection import get_engine, get_session_factory
from iron_aging.models import HeteroLinkPredictionModel


def load_model_and_graph(experiment_name: str, device: torch.device, encoder_type: str):
    """加载模型权重与异构图."""
    exp_dir = Path("L3_results") / experiment_name
    metrics_path = exp_dir / "metrics.json"
    model_path = exp_dir / "model.pt"

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    config = load_config("config.yaml")
    model_config = config.get("model", {})
    edge_type = tuple(model_config.get("edge_type", ["compound", "targets", "gene"]))

    engine = get_engine()
    session_factory = get_session_factory(engine)
    try:
        with session_factory() as session:
            builder = HeteroGraphBuilder(session)
            data = builder.build(use_cache=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        engine.dispose()

    if edge_type not in data.edge_types:
        raise ValueError(f"边类型 {edge_type} 不在图中")

    model = HeteroLinkPredictionModel.from_hetero_data(
        data,
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        out_dim=int(model_config.get("out_dim", 16)),
        encoder_type=encoder_type,
        num_heads=int(model_config.get("num_heads", 4)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.3)),
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    return model, data, edge_type, metrics


def compute_embeddings(model: HeteroLinkPredictionModel, data, device: torch.device):
    """在全图上计算节点嵌入."""
    with torch.no_grad():
        x_dict = {nt: data[nt].x.to(device) for nt in data.node_types}
        edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
        z_dict = model(x_dict, edge_index_dict)
    return {k: v.cpu() for k, v in z_dict.items()}


def score_pairs(
    model: HeteroLinkPredictionModel,
    z_dict: dict[str, torch.Tensor],
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    src_type: str,
    dst_type: str,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """批量计算源-目标节点对的预测分数."""
    scores: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(src_idx), batch_size):
            s = src_idx[i : i + batch_size]
            d = dst_idx[i : i + batch_size]
            src_emb = z_dict[src_type][s].to(device)
            dst_emb = z_dict[dst_type][d].to(device)
            batch_scores = model.predictor(src_emb, dst_emb).squeeze(-1)
            scores.append(batch_scores.cpu())
    return torch.cat(scores)


def _infer_encoder(experiment_name: str) -> str:
    """从实验名推断编码器类型."""
    lowered = experiment_name.lower()
    if "hgt" in lowered:
        return "hgt"
    if "gat" in lowered:
        return "gat"
    if "rgcn" in lowered:
        return "rgcn"
    return "rgcn"


def _get_names(data, node_type: str):
    """获取节点名称列表."""
    storage = data[node_type]
    names = getattr(storage, "names", None) or getattr(storage, "name", None)
    if names is None:
        return [f"{node_type}_{i}" for i in range(storage.num_nodes)]
    return list(names)


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 compound-target 链路预测 top-k 结果")
    parser.add_argument(
        "--experiment",
        type=str,
        default="hgt_compare_seed42",
        help="L3_results 下的实验目录名",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="编码器类型 (hgt/gat/rgcn); 默认从实验名推断",
    )
    args = parser.parse_args()

    experiment_name = args.experiment
    encoder_type = args.encoder or _infer_encoder(experiment_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, data, edge_type, metrics = load_model_and_graph(experiment_name, device, encoder_type)
    src_type, rel_type, dst_type = edge_type

    # 节点名称
    compound_names = _get_names(data, src_type)
    gene_names = _get_names(data, dst_type)

    z_dict = compute_embeddings(model, data, device)

    # 已观测边集合（训练/验证/测试全排除）
    observed_edges = set()
    for a, b in data[edge_type].edge_index.t().tolist():
        observed_edges.add((int(a), int(b)))

    # 生成所有候选 pair
    num_compounds = data[src_type].num_nodes
    num_genes = data[dst_type].num_nodes
    src_idx = torch.arange(num_compounds).repeat_interleave(num_genes)
    dst_idx = torch.arange(num_genes).repeat(num_compounds)

    scores = score_pairs(model, z_dict, src_idx, dst_idx, src_type, dst_type, device)

    # 构建 DataFrame
    df = pd.DataFrame({
        "compound_id": src_idx.numpy(),
        "gene_id": dst_idx.numpy(),
        "compound_name": [compound_names[i] for i in src_idx.numpy()],
        "gene_name": [gene_names[i] for i in dst_idx.numpy()],
        "score": scores.numpy(),
    })
    df["observed"] = df.apply(lambda r: (int(r["compound_id"]), int(r["gene_id"])) in observed_edges, axis=1)

    output_dir = Path("L3_results") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存全部候选分数（可用于后续分析）
    full_path = output_dir / "all_compound_gene_scores.csv"
    df.to_csv(full_path, index=False)
    print(f"Saved all scores: {full_path}")

    # 新预测 top-k
    novel_df = df[~df["observed"]].sort_values("score", ascending=False).reset_index(drop=True)
    top_k = 100
    top_novel = novel_df.head(top_k)
    top_novel_path = output_dir / f"top{top_k}_novel_predictions.csv"
    top_novel.to_csv(top_novel_path, index=False)
    print(f"Saved top-{top_k} novel predictions: {top_novel_path}")

    # 关键化合物 / 基因过滤
    key_compounds = ["BCP", "BETA-CARYOPHYLLENE", "β-CARYOPHYLLENE"]
    key_genes = ["ACSL4", "GPX4", "LPCAT3", "FTH1", "FTL", "TFRC", "SLC7A11", "NFE2L2", "KEAP1"]

    bcp_df = novel_df[novel_df["compound_name"].str.upper().isin([c.upper() for c in key_compounds])]
    bcp_top = bcp_df.head(50)
    bcp_path = output_dir / "bcp_top_predictions.csv"
    bcp_top.to_csv(bcp_path, index=False)
    print(f"Saved BCP top predictions: {bcp_path}")

    gene_top = []
    for gene in key_genes:
        sub = novel_df[novel_df["gene_name"].str.upper() == gene.upper()].head(10)
        sub["query_gene"] = gene
        gene_top.append(sub)
    if gene_top:
        gene_top_df = pd.concat(gene_top, ignore_index=True)
        gene_path = output_dir / "key_genes_top_predictions.csv"
        gene_top_df.to_csv(gene_path, index=False)
        print(f"Saved key genes top predictions: {gene_path}")

    # 摘要
    summary = {
        "experiment": experiment_name,
        "total_candidates": len(df),
        "observed_edges": int(df["observed"].sum()),
        "novel_edges": int((~df["observed"]).sum()),
        "top_10_novel": top_novel.head(10)[["compound_name", "gene_name", "score"]].to_dict("records"),
        "bcp_top_5": bcp_top.head(5)[["compound_name", "gene_name", "score"]].to_dict("records"),
    }
    summary_path = output_dir / "prediction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    print("\nTop 10 novel predictions:")
    print(top_novel.head(10)[["compound_name", "gene_name", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
