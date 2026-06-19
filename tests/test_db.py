"""数据库层单元测试."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from iron_aging.db.connection import get_engine, get_session_factory, init_db
from iron_aging.db.models import Compound, Gene
from iron_aging.db.repositories import CompoundRepository, GeneRepository


@pytest.fixture(scope="function")
def engine():
    """内存 SQLite 引擎."""
    engine = get_engine(url="sqlite:///:memory:")
    yield engine
    engine.dispose()


@pytest.fixture(scope="function")
def session(engine) -> Session:
    """初始化表并返回会话."""
    init_db(engine)
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as sess:
        yield sess


def test_init_db_creates_tables(engine):
    """init_db 应创建所有表."""
    init_db(engine)
    # 简单验证：可以查询
    SessionFactory = get_session_factory(engine)
    with SessionFactory() as sess:
        assert sess.query(Gene).count() == 0
        assert sess.query(Compound).count() == 0


def test_gene_repository_bulk_upsert(session: Session):
    """GeneRepository 应支持批量 upsert."""
    repo = GeneRepository(session)
    count = repo.bulk_upsert(
        [
            {"symbol": "ACSL4", "name": "Acyl-CoA synthetase long-chain family member 4"},
            {"symbol": "GPX4", "name": "Glutathione peroxidase 4"},
        ]
    )
    assert count == 2
    all_genes = list(repo.get_all())
    assert len(all_genes) == 2
    assert repo.get_by_id(symbol="ACSL4")["symbol"] == "ACSL4"


def test_compound_repository_bulk_upsert(session: Session):
    """CompoundRepository 应支持批量 upsert."""
    repo = CompoundRepository(session)
    count = repo.bulk_upsert(
        [
            {"name": "BCP", "canonical_smiles": "C/C/1=C\\CCC(=C)[C@H]2CC([C@@H]2CC1)(C)C"},
            {"name": "Fer-1", "canonical_smiles": "CCOC(=O)C1=CC(=C(C=C1)NC2CCCCC2)N"},
        ]
    )
    assert count == 2
    bcp = repo.get_by_id(name="BCP")
    assert bcp is not None
    assert bcp["name"] == "BCP"
    assert bcp["canonical_smiles"].startswith("C/C/1")


def test_bulk_upsert_updates_existing(session: Session):
    """bulk_upsert 对已有记录应更新."""
    repo = GeneRepository(session)
    repo.bulk_upsert([{"symbol": "ACSL4", "name": "Old name"}])
    repo.bulk_upsert([{"symbol": "ACSL4", "name": "New name"}])
    gene = repo.get_by_id(symbol="ACSL4")
    assert gene is not None
    assert gene["name"] == "New name"
