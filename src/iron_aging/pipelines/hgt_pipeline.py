"""HGT/GAT/RGCN 异质图链路预测 Pipeline.

使用 v4.0 分层组件完成端到端训练与评估：
- 数据层：HeteroGraphBuilder 从 DB 构建 HeteroData
- 模型层：HeteroLinkPredictionModel
- 训练层：HGTTrainer
- 输出：PipelineResult 含指标、模型路径

参考实践:
- SpotTarget (Zhu et al., WSDM 2024): 消息传递图严格排除 train/val/test 目标边.
- PLNLP (Wang et al., 2021): pairwise ranking loss.
- PyG: structured negative sampling.
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from iron_aging.data.graph_builder import HeteroGraphBuilder
from iron_aging.data.split import split_edges
from iron_aging.db.connection import get_engine, get_session_factory
from iron_aging.models import HeteroLinkPredictionModel
from iron_aging.pipelines.base import Pipeline, PipelineConfig, PipelineResult
from iron_aging.training.negative_sampling import (
    build_link_prediction_labels,
    negative_sample_edges,
    remove_leaked_edges,
    structured_negative_sampling,
)
from iron_aging.training.trainer import HGTTrainer

logger = logging.getLogger(__name__)

# Windows OpenMP 重复库规避
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _to_float(value: float | str | None, default: float) -> float:
    """将配置值安全转换为浮点数."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: int | str | None, default: int) -> int:
    """将配置值安全转换为整数."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _set_seed(seed: int) -> None:
    """固定随机种子以保证可复现性."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_link_prediction_data(
    data: HeteroData,
    edge_type: tuple[str, str, str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    neg_sampling_ratio: float = 1.0,
    negative_sampling: str = "structured",
    seed: int = 42,
    split_mode: str = "random",
) -> tuple[HeteroData, dict[str, list[tuple[int, int]]], int, int]:
    """为指定边类型构建链路预测训练数据.

    训练图会移除 train/val/test 全部目标正样本边, 避免 SpotTarget 论文指出的
    target-link inclusion 问题 (过拟合/分布偏移/隐式测试泄漏).

    Returns:
        train_data: 移除了所有监督目标边的训练图.
        splits: 各集合正负样本边.
        num_src: 源节点数.
        num_dst: 目标节点数.
    """
    src_type, rel_type, dst_type = edge_type
    num_src = data[src_type].num_nodes
    num_dst = data[dst_type].num_nodes

    pos_edges_full = data[edge_type].edge_index
    train_ei, val_ei, test_ei = split_edges(
        pos_edges_full,
        num_src=num_src,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        mode=split_mode,
        task="ct" if src_type == "compound" else "gp",
    )

    pos_train = set(
        (int(a), int(b)) for a, b in train_ei.t().tolist()
    )
    pos_val = set(
        (int(a), int(b)) for a, b in val_ei.t().tolist()
    )
    pos_test = set(
        (int(a), int(b)) for a, b in test_ei.t().tolist()
    )
    all_pos = pos_train | pos_val | pos_test

    rng = np.random.default_rng(seed)

    def _sample_neg(pos_set: set[tuple[int, int]], count: int) -> list[tuple[int, int]]:
        if negative_sampling == "structured":
            return structured_negative_sampling(
                num_src, num_dst, count, all_pos, rng
            )
        return negative_sample_edges(num_src, num_dst, count, all_pos, rng)

    train_neg = _sample_neg(pos_train, int(len(pos_train) * neg_sampling_ratio))
    val_neg = _sample_neg(pos_val, int(len(pos_val) * neg_sampling_ratio))
    test_neg = _sample_neg(pos_test, int(len(pos_test) * neg_sampling_ratio))

    train_edges, _ = build_link_prediction_labels(list(pos_train), train_neg)
    val_edges, _ = build_link_prediction_labels(list(pos_val), val_neg)
    test_edges, _ = build_link_prediction_labels(list(pos_test), test_neg)

    splits: dict[str, list[tuple[int, int]]] = {
        "train_pos_edges": list(pos_train),
        "train_neg_edges": train_neg,
        "val_pos_edges": list(pos_val),
        "val_neg_edges": val_neg,
        "test_pos_edges": list(pos_test),
        "test_neg_edges": test_neg,
    }

    # 构建训练图：移除所有目标正样本边 (SpotTarget)
    train_data = data.clone()
    leak_set = all_pos
    train_data[edge_type].edge_index = remove_leaked_edges(
        train_data[edge_type].edge_index, leak_set
    )
    logger.info(
        "构建训练图: 移除 %d 条目标边 (%s), 该类型剩余 %d 条消息传递边; 其他边类型仍参与消息传递",
        len(all_pos),
        "->".join(edge_type),
        train_data[edge_type].edge_index.shape[1],
    )
    return train_data, splits, num_src, num_dst


class HGTLinkPredictionPipeline(Pipeline):
    """HGT/GAT/RGCN 链路预测 Pipeline.

    支持从配置文件读取超参数，对指定异质边类型进行链路预测训练.
    """

    name = "hgt_link_prediction"

    def run(self, config: PipelineConfig) -> PipelineResult:
        """执行训练流程."""
        experiment_id = config.experiment_name or f"hgt_{uuid.uuid4().hex[:8]}"
        result = PipelineResult(experiment_id=experiment_id, status="failure")

        model_config = config.model_config
        training_config = config.training_config
        seed = config.seed
        use_cache = config.use_cache

        _set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("实验 %s 使用设备: %s", experiment_id, device)

        # 1. 从 DB 构建图
        engine = get_engine()
        session_factory = get_session_factory(engine)
        try:
            with session_factory() as session:
                builder = HeteroGraphBuilder(session)
                data = builder.build(use_cache=use_cache)
        except Exception:
            logger.exception("图构建失败")
            traceback.print_exc()
            result.errors = ["graph_build_failed"]
            return result
        finally:
            engine.dispose()

        # 2. 默认对 compound -> gene 靶点预测；可从配置覆盖
        edge_type = tuple(model_config.get("edge_type", ["compound", "targets", "gene"]))
        if len(edge_type) != 3 or edge_type not in data.edge_types:
            available = [list(et) for et in data.edge_types]
            logger.error("边类型 %s 不可用，可用: %s", edge_type, available)
            result.errors = [f"invalid_edge_type: {edge_type}"]
            result.metrics["available_edge_types"] = available
            return result

        # 3. 划分边
        train_ratio = _to_float(training_config.get("train_ratio"), 0.7)
        val_ratio = _to_float(training_config.get("val_ratio"), 0.15)
        neg_ratio = _to_float(training_config.get("neg_sampling_ratio"), 1.0)
        negative_sampling = str(training_config.get("negative_sampling", "structured")).lower()
        split_mode = str(training_config.get("split_mode", "random")).lower()
        try:
            train_data, splits, num_src, num_dst = _build_link_prediction_data(
                data,
                edge_type,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                neg_sampling_ratio=neg_ratio,
                negative_sampling=negative_sampling,
                seed=seed,
                split_mode=split_mode,
            )
        except Exception:
            logger.exception("边划分失败")
            traceback.print_exc()
            result.errors = ["edge_split_failed"]
            return result

        # 4. 构建模型
        hidden_dim = _to_int(model_config.get("hidden_dim"), 64)
        out_dim = _to_int(model_config.get("out_dim"), 16)
        num_heads = _to_int(model_config.get("num_heads"), 4)
        num_layers = _to_int(model_config.get("num_layers"), 2)
        dropout = _to_float(model_config.get("dropout"), 0.3)
        encoder_type = str(model_config.get("encoder_type", "hgt")).lower()

        model = HeteroLinkPredictionModel.from_hetero_data(
            train_data,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            encoder_type=encoder_type,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)
        logger.info(
            "使用编码器: %s, hidden_dim=%d, out_dim=%d, heads=%d, layers=%d",
            encoder_type,
            hidden_dim,
            out_dim,
            num_heads,
            num_layers,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=_to_float(training_config.get("learn_rate"), 0.001),
            weight_decay=_to_float(training_config.get("weight_decay"), 1e-5),
        )

        # 5. 训练
        epochs = _to_int(training_config.get("epochs"), 200)
        patience = _to_int(training_config.get("early_stop_patience"), 20)
        loss_type = str(training_config.get("loss_type", "bce")).lower()
        margin = _to_float(training_config.get("ranking_margin"), 1.0)
        trainer = HGTTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            early_stopping_patience=patience,
            edge_type=edge_type,
            loss_type=loss_type,
            neg_sampling_ratio=neg_ratio,
            margin=margin,
        )
        try:
            history = trainer.fit(
                train_data,
                splits["train_pos_edges"],
                splits["val_pos_edges"],
                splits["test_pos_edges"],
                splits["train_neg_edges"],
                splits["val_neg_edges"],
                splits["test_neg_edges"],
                num_src=num_src,
                num_dst=num_dst,
                epochs=epochs,
                seed=seed,
            )
        except Exception:
            logger.exception("训练失败")
            traceback.print_exc()
            result.errors = ["training_failed"]
            return result

        # 6. 测试集评估
        try:
            test_metrics = trainer.evaluate(
                train_data,
                splits["test_pos_edges"],
                splits["test_neg_edges"],
            )
        except Exception:
            logger.exception("测试评估失败")
            traceback.print_exc()
            result.errors = ["test_evaluation_failed"]
            return result

        # 7. 保存结果
        output_dir = Path("L3_results") / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.pt"
        metrics_path = output_dir / "metrics.json"
        torch.save(model.state_dict(), model_path)
        metrics = {
            "history": history,
            "test": test_metrics,
            "best_val_auc": trainer.best_val_metric,
            "edge_type": list(edge_type),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        logger.info("模型保存至 %s, 指标保存至 %s", model_path, metrics_path)

        result.status = "success"
        result.metrics = metrics
        result.model_path = model_path
        return result
