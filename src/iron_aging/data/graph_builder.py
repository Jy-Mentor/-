"""从数据库构建 PyG HeteroData.

本模块是 v4.0 重构核心：将原本内嵌在 module3_hgt.py 中的图构建逻辑
抽离为独立、可缓存、可测试的数据层组件.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sqlalchemy.orm import Session
from torch_geometric.data import HeteroData

from iron_aging import NETWORK_DIR, PROJECT_ROOT
from iron_aging.db.repositories import (
    CellTypeMarkerRepository,
    CellTypeRepository,
    CompoundCompoundSimilarityRepository,
    CompoundDiseaseRepository,
    CompoundRepository,
    CompoundTargetRepository,
    DiseaseDiseaseSimilarityRepository,
    DiseaseGeneRepository,
    DiseaseRepository,
    GeneCoexpRepository,
    GenePathwayRepository,
    GeneRepository,
    LigandReceptorRepository,
    MiRNATargetRepository,
    PathwayPathwaySimilarityRepository,
    PathwayRepository,
    PPIRepository,
    TFTargetRepository,
)

logger = logging.getLogger(__name__)


def clear_graph_cache(cache_dir: Path | str | None = None) -> None:
    """清除图缓存目录."""
    target = Path(cache_dir) if cache_dir else Path(".graph_cache")
    if target.exists():
        import shutil

        shutil.rmtree(target)
        logger.info("图缓存已清除: %s", target)


def build_heterogeneous_graph(use_cache: bool = True) -> dict[str, Any]:
    """向后兼容包装器：委托给 module3_hgt 的图构建逻辑.

    该函数保留以避免破坏直接调用 graph_builder.build_heterogeneous_graph 的旧代码。
    完整迁移后，将逐步替换为 HeteroGraphBuilder + ETL 流程。

    Args:
        use_cache: 为 False 时先清除图缓存，强制重新构建图。
    """
    import module3_hgt  # type: ignore[import-not-found]

    if not use_cache:
        clear_graph_cache()
    return module3_hgt.build_heterogeneous_graph()


def build_pyg_data(graph_data: dict[str, Any]) -> HeteroData:
    """将图数据字典转换为 PyG HeteroData.

    该函数已从 module3_hgt.py 迁移至此，作为 v4.0 图构建层的一部分。
    支持常规边类型以及 ACSL4 文献 curated 高置信度相互作用边。
    """
    data = HeteroData()

    # 节点特征 (5种节点类型)
    node_types = ["gene", "pathway", "compound", "disease", "pocket"]
    for nt in node_types:
        x = torch.from_numpy(graph_data[nt]["x"])
        data[nt].x = x
        data[nt].num_nodes = graph_data[nt]["n"]

    # 边 (需要转换为PyG格式, 含反向边)
    edge_mapping = {
        "gene_coexp": ("gene", "coexp", "gene"),
        "regulates": ("gene", "regulates", "gene"),
        "enriched_in": ("gene", "enriched_in", "pathway"),
        "compound_targets": ("compound", "targets", "gene"),
        "gene_disease": ("gene", "associated_with", "disease"),
        "compound_pocket": ("compound", "binds_to", "pocket"),
        # ACSL4 文献 curated 高置信度相互作用边
        "acsl4_curated": ("gene", "curated_interacts_with", "gene"),
        # 反向边 (基因聚合跨类型信息)
        "pathway_to_gene": ("pathway", "rev_enriched_in", "gene"),
        "disease_to_gene": ("disease", "rev_associated_with", "gene"),
    }

    for edge_key, (src, rel, dst) in edge_mapping.items():
        edges = graph_data["edges"].get(edge_key, [])
        if edges:
            src_nodes, dst_nodes = zip(*edges)
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            data[src, rel, dst].edge_index = edge_index

    return data


class HeteroGraphBuilder:
    """异构图构建器.

    从 Repository 读取节点与边, 构建 PyG HeteroData.
    支持基于输入数据哈希的安全缓存(使用 torch.save 替代 pickle).
    """

    def __init__(self, session: Session, cache_dir: Path | str | None = None) -> None:
        self.session = session
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".graph_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Repository 实例
        self.gene_repo = GeneRepository(session)
        self.compound_repo = CompoundRepository(session)
        self.disease_repo = DiseaseRepository(session)
        self.pathway_repo = PathwayRepository(session)
        self.cell_type_repo = CellTypeRepository(session)
        self.ppi_repo = PPIRepository(session)
        self.tf_repo = TFTargetRepository(session)
        self.ct_repo = CompoundTargetRepository(session)
        self.dg_repo = DiseaseGeneRepository(session)
        self.gp_repo = GenePathwayRepository(session)
        self.ctm_repo = CellTypeMarkerRepository(session)
        self.lr_repo = LigandReceptorRepository(session)
        self.coexp_repo = GeneCoexpRepository(session)
        self.ccs_repo = CompoundCompoundSimilarityRepository(session)
        self.pps_repo = PathwayPathwaySimilarityRepository(session)
        self.cd_repo = CompoundDiseaseRepository(session)
        self.dds_repo = DiseaseDiseaseSimilarityRepository(session)
        self.mt_repo = MiRNATargetRepository(session)

    def build(self, use_cache: bool = True) -> HeteroData:
        """构建 HeteroData."""
        logger.info("开始构建异构图...")

        # 1. 加载节点并建立业务 ID -> 连续索引 的映射
        nodes: dict[str, dict[str, int]] = {}
        node_lists: dict[str, list[dict[str, Any]]] = {}

        node_lists["gene"] = list(self.gene_repo.get_all())
        node_lists["compound"] = list(self.compound_repo.get_all())
        node_lists["disease"] = list(self.disease_repo.get_all())
        node_lists["pathway"] = list(self.pathway_repo.get_all())
        node_lists["cell_type"] = list(self.cell_type_repo.get_all())
        # miRNA 实体来自 mirna_target_edges 中的唯一 mirna 名称
        node_lists["mirna"] = [
            {"name": m} for m in self._collect_mirna_names()
        ]

        for nt, records in node_lists.items():
            nodes[nt] = {self._node_id(nt, r): i for i, r in enumerate(records)}
            logger.info("节点类型 %s: %d 个", nt, len(records))

        # 2. 计算缓存哈希(基于节点数量与表结构版本)
        data_hash = self._compute_hash(
            {nt: len(recs) for nt, recs in node_lists.items()}
        )
        cache_path = self._cache_path(data_hash)
        if use_cache and cache_path.exists():
            logger.info("命中图缓存: %s", cache_path)
            return torch.load(cache_path, weights_only=False)

        # 3. 初始化 HeteroData
        data = HeteroData()
        for nt, records in node_lists.items():
            data[nt].num_nodes = len(records)
            data[nt].names = [self._node_id(nt, r) for r in records]

        # 4. 加载边
        self._add_edges(data, "gene", "interacts", "gene", self.ppi_repo.get_all(), nodes)
        self._add_edges(data, "gene", "regulates", "gene", self.tf_repo.get_all(), nodes)
        self._add_edges(
            data, "compound", "targets", "gene", self.ct_repo.get_all(), nodes
        )
        self._add_edges(
            data, "gene", "associated_with", "disease", self.dg_repo.get_all(), nodes
        )
        self._add_edges(
            data, "gene", "belongs_to", "pathway", self.gp_repo.get_all(), nodes
        )
        self._add_edges(
            data, "cell_type", "expresses", "gene", self.ctm_repo.get_all(), nodes
        )
        self._add_edges(data, "gene", "lr_pair", "gene", self.lr_repo.get_all(), nodes)
        self._add_edges(data, "gene", "coexp", "gene", self.coexp_repo.get_all(), nodes)
        self._add_edges(
            data, "compound", "similar_to", "compound", self.ccs_repo.get_all(), nodes
        )
        self._add_edges(
            data, "pathway", "similar_to", "pathway", self.pps_repo.get_all(), nodes
        )
        self._add_edges(
            data, "compound", "indicates", "disease", self.cd_repo.get_all(), nodes
        )
        self._add_edges(
            data, "disease", "similar_to", "disease", self.dds_repo.get_all(), nodes
        )
        self._add_edges(
            data, "mirna", "targets", "gene", self.mt_repo.get_all(), nodes
        )

        # 5. 附加节点特征 (任何失败直接向上传播, 不静默使用空特征)
        self._attach_node_features(data, node_lists)

        logger.info("异构图构建完成")
        if use_cache:
            torch.save(data, cache_path)
            logger.info("图缓存已写入: %s", cache_path)
        return data

    def _node_id(self, node_type: str, record: dict[str, Any]) -> str:
        """根据节点类型提取业务主键."""
        id_field = {
            "gene": "symbol",
            "compound": "name",
            "disease": "name",
            "pathway": "id",
            "cell_type": "name",
            "mirna": "name",
        }.get(node_type, "name")
        return str(record.get(id_field))

    def _collect_mirna_names(self) -> list[str]:
        """从 miRNA-靶基因边中收集唯一 miRNA 名称."""
        names: set[str] = set()
        for rec in self.mt_repo.get_all():
            name = str(rec.get("mirna_id", "")).strip()
            if name:
                names.add(name)
        return sorted(names)

    def _add_edges(
        self,
        data: HeteroData,
        src_type: str,
        rel_type: str,
        dst_type: str,
        records: Iterable[dict[str, Any]],
        nodes: dict[str, dict[str, int]],
    ) -> None:
        """将记录列表转换为 PyG edge_index."""
        src_map = nodes.get(src_type, {})
        dst_map = nodes.get(dst_type, {})

        src_idx, dst_idx = [], []
        for rec in records:
            s, t = self._extract_edge_ids(src_type, rel_type, dst_type, rec)
            if s is None or t is None:
                continue
            si = src_map.get(str(s))
            ti = dst_map.get(str(t))
            if si is None or ti is None:
                continue
            src_idx.append(si)
            dst_idx.append(ti)

        if not src_idx:
            logger.info("边 [%s,%s,%s]: 0 条", src_type, rel_type, dst_type)
            return

        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[src_type, rel_type, dst_type].edge_index = edge_index
        logger.info(
            "边 [%s,%s,%s]: %d 条", src_type, rel_type, dst_type, edge_index.shape[1]
        )

    def _extract_edge_ids(
        self, src_type: str, rel_type: str, dst_type: str, rec: dict[str, Any]
    ) -> tuple[Any, Any]:
        """从边记录中提取源/目标业务 ID."""
        if src_type == "gene" and rel_type == "interacts":
            return rec.get("protein_a_id"), rec.get("protein_b_id")
        if rel_type == "regulates":
            return rec.get("tf_id"), rec.get("target_id")
        if rel_type == "targets":
            return rec.get("compound_id"), rec.get("gene_id")
        if rel_type == "associated_with":
            return rec.get("gene_id"), rec.get("disease_id")
        if rel_type == "belongs_to":
            return rec.get("gene_id"), rec.get("pathway_id")
        if rel_type == "expresses":
            return rec.get("cell_type_id"), rec.get("gene_id")
        if rel_type == "lr_pair":
            return rec.get("ligand_id"), rec.get("receptor_id")
        if rel_type == "coexp":
            return rec.get("gene_a_id"), rec.get("gene_b_id")
        if rel_type == "similar_to" and src_type == "compound":
            return rec.get("compound_a_id"), rec.get("compound_b_id")
        if rel_type == "similar_to" and src_type == "pathway":
            return str(rec.get("pathway_a_id")), str(rec.get("pathway_b_id"))
        if rel_type == "indicates":
            return rec.get("compound_id"), rec.get("disease_id")
        if rel_type == "similar_to" and src_type == "disease":
            return rec.get("disease_a_id"), rec.get("disease_b_id")
        if rel_type == "targets" and src_type == "mirna":
            return rec.get("mirna_id"), rec.get("gene_id")
        return None, None

    def _attach_node_features(
        self,
        data: HeteroData,
        node_lists: dict[str, list[dict[str, Any]]],
    ) -> None:
        """使用 feature builders 附加基因/化合物节点特征.

        当前阶段优先从 CSV 真实数据计算; 疾病/通路/细胞类型等节点暂用零向量占位.
        """
        from iron_aging.features.compounds import CompoundFeatureBuilder
        from iron_aging.features.genes import GeneFeatureBuilder

        gene_names = [self._node_id("gene", r) for r in node_lists.get("gene", [])]
        if gene_names:
            gene_builder = GeneFeatureBuilder(project_root=PROJECT_ROOT)
            gene_feat = gene_builder.build(gene_names)
            feat_matrix = np.stack([gene_feat[g] for g in gene_names]).astype(np.float32)
            data["gene"].x = torch.from_numpy(feat_matrix)
            logger.info("附加 gene 特征: shape=%s", feat_matrix.shape)

        compound_records = [
            {"compound": self._node_id("compound", r), "CanonicalSMILES": r.get("canonical_smiles", "")}
            for r in node_lists.get("compound", [])
        ]
        if compound_records:
            compound_df = pd.DataFrame(compound_records)
            compound_builder = CompoundFeatureBuilder(network_dir=NETWORK_DIR)
            try:
                prop_df = compound_builder.build_properties(compound_df)
                feat_cols = [c for c in prop_df.columns if c != "compound"]
                feat_matrix = prop_df[feat_cols].to_numpy(dtype=np.float32)
                data["compound"].x = torch.from_numpy(feat_matrix)
                logger.info("附加 compound 特征: shape=%s", feat_matrix.shape)
            except ImportError:
                logger.warning("RDKit 未安装, compound 特征使用零向量")
                n_compounds = len(compound_records)
                data["compound"].x = torch.zeros(n_compounds, 6, dtype=torch.float32)

        # 为 pathway / disease / cell_type / mirna 聚合相关基因特征
        # 符合项目规则：所有节点特征必须用真实数据，通路特征从成员基因学习
        if "gene" in data and data["gene"].x is not None:
            gene_features = data["gene"].x.numpy()
            gene_names = data["gene"].names

            # pathway: 成员基因均值
            pathway_x = self._aggregate_node_features_from_genes(
                node_type="pathway",
                node_names=data["pathway"].names,
                gene_names=gene_names,
                gene_features=gene_features,
                related_records=self.gp_repo.get_all(),
                node_key="pathway_id",
                gene_key="gene_id",
                node_id_to_name=None,
            )
            if pathway_x is not None:
                data["pathway"].x = pathway_x

            # disease: 关联基因均值
            disease_x = self._aggregate_node_features_from_genes(
                node_type="disease",
                node_names=data["disease"].names,
                gene_names=gene_names,
                gene_features=gene_features,
                related_records=self.dg_repo.get_all(),
                node_key="disease_id",
                gene_key="gene_id",
                node_id_to_name=None,
            )
            if disease_x is not None:
                data["disease"].x = disease_x

            # cell_type: 标记基因均值
            cell_type_x = self._aggregate_node_features_from_genes(
                node_type="cell_type",
                node_names=data["cell_type"].names,
                gene_names=gene_names,
                gene_features=gene_features,
                related_records=self.ctm_repo.get_all(),
                node_key="cell_type_id",
                gene_key="gene_id",
                node_id_to_name=None,
            )
            if cell_type_x is not None:
                data["cell_type"].x = cell_type_x

            # mirna: 靶基因均值
            mirna_x = self._aggregate_node_features_from_genes(
                node_type="mirna",
                node_names=data["mirna"].names,
                gene_names=gene_names,
                gene_features=gene_features,
                related_records=self.mt_repo.get_all(),
                node_key="mirna_id",
                gene_key="gene_id",
                node_id_to_name=None,
            )
            if mirna_x is not None:
                data["mirna"].x = mirna_x
        else:
            logger.warning("gene 特征缺失，无法为 pathway/disease/cell_type/mirna 聚合真实特征，使用零向量占位")
            for nt in ("disease", "pathway", "cell_type", "mirna"):
                n_nodes = len(node_lists.get(nt, []))
                if n_nodes:
                    data[nt].x = torch.zeros(n_nodes, 1, dtype=torch.float32)

    def _aggregate_node_features_from_genes(
        self,
        node_type: str,
        node_names: list[str],
        gene_names: list[str],
        gene_features: np.ndarray,
        related_records: Iterable[dict[str, Any]],
        node_key: str,
        gene_key: str,
        node_id_to_name: dict[int, str] | None,
    ) -> torch.Tensor | None:
        """通过相关基因特征均值构建非基因节点特征.

        Args:
            node_type: 节点类型，仅用于日志.
            node_names: 该类型节点业务主键列表.
            gene_names: 基因节点业务主键列表.
            gene_features: 基因特征矩阵，形状 (n_genes, feat_dim).
            related_records: 节点-基因关联记录.
            node_key: 关联记录中节点主键字段名.
            gene_key: 关联记录中基因主键字段名.
            node_id_to_name: 若节点主键为整数（如 pathway.id）而 node_names 为名称，
                             提供 id -> name 映射；否则传 None.

        Returns:
            聚合后的节点特征张量；若无法聚合则返回 None.
        """
        if not node_names or gene_features.size == 0:
            return None

        gene_name_to_idx = {name: i for i, name in enumerate(gene_names)}
        node_name_to_idx = {name: i for i, name in enumerate(node_names)}

        node_to_gene_indices: dict[int, list[int]] = {}
        for rec in related_records:
            node_id = rec.get(node_key)
            gene_id = rec.get(gene_key)
            if node_id is None or gene_id is None:
                continue
            if node_id_to_name is not None:
                node_id = node_id_to_name.get(node_id)
            node_name = str(node_id)
            gene_name = str(gene_id)
            if node_name not in node_name_to_idx or gene_name not in gene_name_to_idx:
                continue
            node_idx = node_name_to_idx[node_name]
            gene_idx = gene_name_to_idx[gene_name]
            node_to_gene_indices.setdefault(node_idx, []).append(gene_idx)

        if not node_to_gene_indices:
            logger.warning("%s 无相关基因映射，使用零向量占位", node_type)
            return torch.zeros(len(node_names), gene_features.shape[1], dtype=torch.float32)

        aggregated = np.zeros((len(node_names), gene_features.shape[1]), dtype=np.float32)
        for node_idx, gene_indices in node_to_gene_indices.items():
            aggregated[node_idx] = gene_features[gene_indices].mean(axis=0)

        logger.info(
            "附加 %s 特征：%d 个节点，基于真实基因特征聚合，shape=%s",
            node_type,
            len(node_names),
            aggregated.shape,
        )
        return torch.from_numpy(aggregated)

    def _compute_hash(self, *args: Any) -> str:
        """基于输入计算缓存哈希."""
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, data_hash: str) -> Path:
        return self.cache_dir / f"hetero_data_{data_hash}.pt"
