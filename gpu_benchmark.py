#!/usr/bin/env python3
"""GPU/CPU 训练性能对比基准 (module3_hgt 训练子集)."""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from module3_hgt import build_heterogeneous_graph, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def benchmark(device_str: str, epochs: int, graph_data: dict) -> float:
    """在指定设备上训练若干 epoch, 返回训练耗时 (秒)."""
    _set_seed(42)
    logger.info(f"[{device_str}] 开始训练 {epochs} epochs ...")
    start = time.perf_counter()
    train_model(
        graph_data,
        hidden_dim=64,
        epochs=epochs,
        learn_rate=0.001,
        device_str=device_str,
        config={
            "model": {
                "hidden_dim": 64,
                "gat_out_dim": 16,
                "hgt_out_dim": 16,
                "num_heads": 2,
                "dropout": 0.3,
                "gat_residual_weight": 0.3,
                "num_hgt_layers": 2,
                "use_vib": True,
                "use_self_distill": True,
            },
            "training": {
                "epochs": epochs,
                "learn_rate": 0.001,
                "weight_decay": 1e-5,
                "seed": 42,
                "eval_every": epochs + 1,
                "early_stop_patience": epochs + 1,
                "warmup_epochs": 0,
                "warmup_start_factor": 0.01,
                "ema_decay": 0.999,
                "vib_beta_max": 0.001,
                "vib_warmup_epochs": 0,
                "dropedge_p": 0.0,
                "curriculum": {
                    "start_ratio": 1.0,
                    "growth_rate": 0.0,
                    "patience_warmup": epochs + 1,
                    "ema_decay": 0.9,
                },
                "neg_sampling": {
                    "hard_neg_ratio": 0.3,
                    "hard_neg_refresh_interval": epochs + 1,
                    "easy_neg_ratio": 0.7,
                },
                "split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
            },
            "heco": {"temperature": 0.07, "projection_dim": 128, "view_mask_prob": 0.3},
            "gradnorm": {"alpha": 1.5, "lr_weight": 0.001},
            "self_distill": {"alpha": 0.05, "temperature": 3.0},
            "vib": {"beta": 0.001, "latent_dim_ratio": 0.5},
        },
    )
    elapsed = time.perf_counter() - start
    logger.info(f"[{device_str}] 训练完成: {elapsed:.2f}s")
    return elapsed


def main() -> int:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _set_seed(42)

    epochs = int(os.environ.get("BENCH_EPOCHS", "100"))
    logger.info(f"性能对比 epochs={epochs}")

    logger.info("构建异质图 (设备无关步骤, 仅执行一次) ...")
    graph_data = build_heterogeneous_graph()

    gpu_available = torch.cuda.is_available()
    results = {}

    # 先跑 GPU (若可用), 避免 CPU 后缓存/上下文切换影响
    if gpu_available:
        torch.cuda.synchronize()
        results["cuda"] = benchmark("cuda", epochs, graph_data)
        torch.cuda.synchronize()
    else:
        logger.warning("CUDA 不可用, 跳过 GPU 基准")

    results["cpu"] = benchmark("cpu", epochs, graph_data)

    logger.info("=" * 60)
    logger.info("性能对比结果")
    logger.info("=" * 60)
    for device_str, elapsed in results.items():
        logger.info(f"  {device_str}: {elapsed:.2f}s")
    if "cuda" in results and "cpu" in results:
        speedup = results["cpu"] / results["cuda"]
        logger.info(f"  GPU 加速比: {speedup:.2f}x")
        if speedup < 2.0:
            logger.warning("加速比低于 2x; 可能受图规模较小或 CPU/GPU 传输开销影响")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
