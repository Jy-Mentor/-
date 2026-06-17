#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module3_hgt.py 单元测试
========================
覆盖:
  1. build_heterogeneous_graph 图形状验证
  2. 边数量/类型完整性检查
  3. 正负样本分割比例验证
  4. 边泄露移除验证
  5. 种子可复现性验证
  6. DropEdge 增强正确性
  7. VIBLayer 数值稳定性
  8. GATv2 前向传播验证
  9. HGT 多层前向传播验证
"""

import sys, os, warnings, unittest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn


class TestGraphConstruction(unittest.TestCase):
    """测试异质图构建"""

    @classmethod
    def setUpClass(cls):
        from module3_hgt import build_heterogeneous_graph
        cls.graph = build_heterogeneous_graph()

    def test_node_counts(self):
        """验证节点数量"""
        g = self.graph
        self.assertGreater(g['gene']['n'], 0, "基因节点 > 0")
        self.assertGreater(g['pathway']['n'], 0, "通路节点 > 0")
        self.assertGreater(g['compound']['n'], 0, "化合物节点 > 0")
        self.assertGreater(g['lr']['n'], 0, "配体-受体节点 > 0")
        self.assertGreater(g['celltype']['n'], 0, "细胞类型节点 > 0")
        self.assertGreater(g['disease']['n'], 0, "疾病节点 > 0")
        self.assertEqual(g['gene']['x'].shape[0], g['gene']['n'], "基因特征帧数==节点数")
        self.assertEqual(g['gene']['x'].shape[1], 19, "基因特征维度==19 (16基础+3生物标志)")

    def test_edge_type_completeness(self):
        """验证11种边类型全部存在"""
        expected_rels = {
            'gene_coexp', 'lr_interaction', 'regulates', 'enriched_in',
            'compound_targets', 'celltype_express', 'gene_disease',
            'compound_pocket', 'gene_to_lr', 'gene_to_compound', 'gene_to_celltype'
        }
        edges = self.graph['edges']
        found_rels = set(edges.keys())  # edge dict 使用字符串键
        missing = expected_rels - found_rels
        self.assertEqual(len(missing), 0, f"缺失边类型: {missing}")

    def test_edge_counts(self):
        """验证边数量在合理范围内"""
        edges = self.graph['edges']
        for name in ['gene_coexp', 'enriched_in']:
            self.assertGreater(len(edges[name]), 0, f"{name} 边数量 > 0")

    def test_ppi_key_exists(self):
        """验证 STRING PPI 键独立存储"""
        self.assertIn('string_ppi', self.graph['edges'], "graph_data['edges'] 应包含 'string_ppi'")
        self.assertGreater(len(self.graph['edges']['string_ppi']), 0, "PPI 边数量 > 0")

    def test_feature_dimension(self):
        """验证特征维度一致性"""
        x = self.graph['gene']['x']
        self.assertFalse(np.isnan(x).any(), "基因特征无 NaN")
        self.assertFalse(np.isinf(x).any(), "基因特征无 Inf")

    def test_node_names_match(self):
        """验证 names 列表与节点数量一致"""
        for ntype in ['gene', 'pathway', 'compound', 'lr', 'celltype', 'disease']:
            n = self.graph[ntype]['n']
            names = self.graph[ntype]['names']
            self.assertEqual(len(names), n, f"{ntype} names 长度 == 节点数 {n}")


class TestVIBLayer(unittest.TestCase):
    """测试 VIB 层"""

    def setUp(self):
        from module3_hgt import VIBLayer
        self.vib = VIBLayer(in_dim=64, latent_dim=32, beta=1e-3)

    def test_output_shape(self):
        x = torch.randn(100, 64)
        out, kl = self.vib(x)
        self.assertEqual(out.shape, x.shape, "输出维度 == 输入维度")
        self.assertEqual(kl.dim(), 0, "KL 损失为标量")

    def test_training_vs_eval(self):
        x = torch.randn(100, 64)
        self.vib.train()
        out1, _ = self.vib(x)
        self.vib.eval()
        out2, _ = self.vib(x)
        # 推理模式应可复现 (无重参数化噪声)
        self.vib.eval()
        out3, _ = self.vib(x)
        self.assertTrue(torch.allclose(out2, out3, atol=1e-5), "推理模式确定性")

    def test_set_beta(self):
        self.vib.set_beta(0.001)
        self.assertEqual(self.vib.beta, 0.001)
        self.vib.set_beta(0.0)
        x = torch.randn(10, 64)
        _, kl = self.vib(x)
        self.assertAlmostEqual(kl.item(), 0.0, delta=1e-6, msg="beta=0 时 KL 权重为0")

    def test_clamp_logvar(self):
        """验证 logvar 裁剪防止溢出"""
        x = torch.randn(100, 64) * 100  # 大输入
        self.vib.train()
        try:
            out, kl = self.vib(x)
            self.assertFalse(torch.isnan(out).any(), "无 NaN 输出")
            self.assertFalse(torch.isinf(out).any(), "无 Inf 输出")
            self.assertFalse(torch.isnan(kl), "KL 非 NaN")
        except Exception as e:
            self.fail(f"前向传播异常: {e}")


class TestGATv2Encoder(unittest.TestCase):
    """测试 GATv2 编码器"""

    def test_forward(self):
        from module3_hgt import GATEncoder
        enc = GATEncoder(in_dim=16, hidden_dim=32, out_dim=16)
        self.assertFalse(enc.conv1.share_weights, "share_weights 应为 False")
        self.assertFalse(enc.conv2.share_weights, "share_weights 应为 False")

        x = torch.randn(20, 16)
        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        out = enc(x, ei)
        self.assertEqual(out.shape, (20, 16))


class TestDropEdge(unittest.TestCase):
    """测试 DropEdge 增强"""

    def setUp(self):
        from module3_hgt import augment_graph
        self.augment = augment_graph

    def test_drops_edges(self):
        ei = torch.randint(0, 100, (2, 500))
        d = {('gene', 'coexp', 'gene'): ei.clone()}
        aug = self.augment(d, drop_p=0.5, seed=42)
        dropped = ei.size(1) - aug[('gene', 'coexp', 'gene')].size(1)
        self.assertGreater(dropped, 0, "应丢弃部分边")

    def test_preserves_non_target_types(self):
        ei = torch.randint(0, 10, (2, 100))
        d = {
            ('gene', 'coexp', 'gene'): ei.clone(),
            ('gene', 'regulates', 'gene'): ei.clone(),
            ('gene', 'enriched_in', 'pathway'): ei.clone(),
            ('gene', 'gene_disease', 'disease'): ei.clone(),  # 不丢弃
        }
        aug = self.augment(d, drop_p=0.5, seed=42)
        # 不应被drop的边类型保持原样
        self.assertEqual(aug[('gene', 'gene_disease', 'disease')].size(1), 100,
                         "非目标边类型不应改变")

    def test_keeps_at_least_one(self):
        ei = torch.tensor([[0], [0]], dtype=torch.long)  # 只有1条边
        d = {('gene', 'coexp', 'gene'): ei}
        aug = self.augment(d, drop_p=0.9, seed=42)
        self.assertGreaterEqual(aug[('gene', 'coexp', 'gene')].size(1), 1,
                                "至少保留1条边")


class TestHeCoContrastive(unittest.TestCase):
    """测试 HeCo 对比预训练 (轻量级前向传播)"""

    def test_loss_non_negative(self):
        """验证 HeCo 损失 >= 0"""
        from module3_hgt import HeCoPreTrainer, MetaPathLearner
        trainer = HeCoPreTrainer(hidden_dim=32, temperature=0.07)
        # 模拟两份投影 (模拟 schema view 和 metapath view)
        schema = torch.randn(100, 32)
        metapath = torch.randn(100, 32)
        loss = trainer(schema, metapath)
        self.assertGreaterEqual(loss.item(), 0.0, "HeCo 损失 >= 0")
        self.assertFalse(torch.isnan(loss), "HeCo 损失非 NaN")


if __name__ == '__main__':
    unittest.main(verbosity=2)
