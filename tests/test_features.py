"""特征构建模块测试."""

from __future__ import annotations

import numpy as np

from iron_aging.features.genes import GeneFeatureBuilder


def test_gene_feature_builder_shape():
    """GeneFeatureBuilder 应返回与输入基因列表等长的特征向量."""
    builder = GeneFeatureBuilder()
    genes = ["ACSL4", "GPX4", "TP53"]
    feats = builder.build(genes)
    assert set(feats.keys()) == set(genes)
    for g in genes:
        assert isinstance(feats[g], np.ndarray)
        assert feats[g].dtype == np.float32
        assert feats[g].ndim == 1
        assert feats[g].shape[0] > 0


def test_gene_feature_no_nan_inf():
    """所有基因特征应无 NaN/Inf."""
    builder = GeneFeatureBuilder()
    genes = ["ACSL4", "GPX4", "TP53"]
    feats = builder.build(genes)
    for g in genes:
        assert not np.isnan(feats[g]).any()
        assert not np.isinf(feats[g]).any()


def test_gene_feature_includes_celltype_and_lr():
    """特征应包含细胞类型与 LR 维度."""
    builder = GeneFeatureBuilder()
    genes = ["ACSL4", "GPX4", "TP53"]
    feats = builder.build(genes)
    dim = feats[genes[0]].shape[0]
    assert dim >= 9 + 2  # L1 + LR at minimum
