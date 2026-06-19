"""铁衰老项目训练层.

包含训练器、损失函数、负采样策略等模块.
"""

from iron_aging.training.losses import bce_with_logits_loss, vib_kl_loss
from iron_aging.training.negative_sampling import negative_sample_edges
from iron_aging.training.trainer import HGTTrainer

__all__ = [
    "HGTTrainer",
    "bce_with_logits_loss",
    "vib_kl_loss",
    "negative_sample_edges",
]
