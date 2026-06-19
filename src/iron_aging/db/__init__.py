"""数据库层：ORM 模型、连接管理与 Repository 接口."""

from iron_aging.db.connection import get_engine, get_session_factory, init_db
from iron_aging.db.models import Base
from iron_aging.db.repositories import (
    CellTypeMarkerRepository,
    CellTypeRepository,
    CompoundRepository,
    CompoundTargetRepository,
    DatasetRepository,
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

__all__ = [
    "Base",
    "get_engine",
    "get_session_factory",
    "init_db",
    "GeneRepository",
    "CompoundRepository",
    "DiseaseRepository",
    "PathwayRepository",
    "CellTypeRepository",
    "DatasetRepository",
    "PPIRepository",
    "TFTargetRepository",
    "CompoundTargetRepository",
    "DiseaseGeneRepository",
    "GenePathwayRepository",
    "CellTypeMarkerRepository",
    "LigandReceptorRepository",
    "GeneCoexpRepository",
]
