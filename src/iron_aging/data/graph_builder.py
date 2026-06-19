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

import torch
from sqlalchemy.orm import Session
from torch_geometric.data import HeteroData

from iron_aging.db.repositories import (
    CellTypeMarkerRepository,
    CellTypeRepository,
    CompoundRepository,
    CompoundTargetRepository,
    DiseaseGeneRepository,
    DiseaseRepository,
    GeneCoexpRepository,
    GenePathwayRepository,
    GeneRepository,
    LigandReceptorRepository,
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


def build_heterogeneous_graph(use_cache: bool = True) -> dict[str, Any]:  # noqa: ARG001
    """向后兼容包装器：委托给 module3_hgt 的图构建逻辑."""
    import module3_hgt  # type: ignore[import-not-found]

    return module3_hgt.build_heterogeneous_graph()


def build_pyg_data(graph_data: dict[str, Any]) -> HeteroData:
    """向后兼容包装器：委托给 module3_hgt 的 PyG 转换逻辑."""
    import module3_hgt  # type: ignore[import-not-found]

    return module3_hgt.build_pyg_data(graph_data)


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
        }.get(node_type, "name")
        return str(record.get(id_field))

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
        return None, None

    def _compute_hash(self, *args: Any) -> str:
        """基于输入计算缓存哈希."""
        content = json.dumps(args, sort_keys=True, default=str)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, data_hash: str) -> Path:
        return self.cache_dir / f"hetero_data_{data_hash}.pt"
