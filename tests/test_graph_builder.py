"""测试异构图构建与缓存."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from iron_aging.data.graph_builder import HeteroGraphBuilder  # noqa: E402
from iron_aging.db.connection import get_engine, get_session_factory, init_db  # noqa: E402
from iron_aging.db.repositories import (  # noqa: E402
    CompoundRepository,
    CompoundTargetRepository,
    GeneRepository,
)


@pytest.fixture(scope="function")
def session() -> Session:
    """内存 SQLite 会话, 含基础种子数据."""
    engine = get_engine(url="sqlite:///:memory:")
    init_db(engine)
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as sess:
        gene_repo = GeneRepository(sess)
        gene_repo.bulk_upsert(
            [
                {"symbol": "ACSL4", "name": "Acyl-CoA synthetase long-chain family member 4"},
                {"symbol": "GPX4", "name": "Glutathione peroxidase 4"},
                {"symbol": "TLR4", "name": "Toll-like receptor 4"},
            ]
        )
        compound_repo = CompoundRepository(sess)
        compound_repo.bulk_upsert(
            [
                {
                    "name": "BCP",
                    "canonical_smiles": "C/C/1=C\\CCC(=C)[C@H]2CC([C@@H]2CC1)(C)C",
                },
                {"name": "Fer-1", "canonical_smiles": "CCOC(=O)C1=CC(=C(C=C1)NC2CCCCC2)N"},
            ]
        )
        ct_repo = CompoundTargetRepository(sess)
        ct_repo.bulk_upsert(
            [
                {"compound_id": "BCP", "gene_id": "ACSL4", "source": "test"},
                {"compound_id": "Fer-1", "gene_id": "GPX4", "source": "test"},
            ]
        )
        sess.commit()
        yield sess
    engine.dispose()


@pytest.fixture(scope="function")
def builder(session: Session, tmp_path: Path) -> HeteroGraphBuilder:
    """带独立缓存目录的构建器."""
    return HeteroGraphBuilder(session=session, cache_dir=tmp_path / "graph_cache")


def test_build_returns_heterodata(builder: HeteroGraphBuilder) -> None:
    """验证 build 返回 PyG HeteroData 且包含核心节点类型."""
    data = builder.build(use_cache=False)

    assert hasattr(data, "node_types")
    assert "gene" in data.node_types
    assert "compound" in data.node_types
    assert data["gene"].num_nodes == 3
    assert data["compound"].num_nodes == 2


def test_build_populates_edges(builder: HeteroGraphBuilder) -> None:
    """验证 compound_targets 边被正确加载."""
    data = builder.build(use_cache=False)

    edge_type = ("compound", "targets", "gene")
    assert edge_type in data.edge_types
    edge_index = data[edge_type].edge_index
    assert edge_index.shape[1] == 2


def test_graph_build_cache_speedup(builder: HeteroGraphBuilder) -> None:
    """验证缓存能显著加速第二次图构建."""
    start1 = time.time()
    data1 = builder.build(use_cache=True)
    elapsed1 = time.time() - start1

    start2 = time.time()
    data2 = builder.build(use_cache=True)
    elapsed2 = time.time() - start2

    assert elapsed2 < elapsed1
    assert data1["gene"].num_nodes == data2["gene"].num_nodes
    assert data1["compound"].num_nodes == data2["compound"].num_nodes
