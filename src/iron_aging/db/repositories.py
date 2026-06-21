"""Repository 模式封装.

提供统一的数据访问接口, 隔离 ORM 细节与业务逻辑.
所有 Repository 接收 SQLAlchemy Session, 不自行管理事务.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from iron_aging.db import models


class BaseRepository(ABC):
    """Repository 抽象基类."""

    def __init__(self, session: Session) -> None:
        self.session = session

    @abstractmethod
    def bulk_upsert(self, records: list[dict[str, Any]]) -> int:
        """批量插入或更新, 返回写入行数."""

    @abstractmethod
    def get_all(self) -> Iterable[dict[str, Any]]:
        """返回全部记录的字典序列."""


class _SimpleRepository(BaseRepository):
    """通用 Repository, 通过 model_class 与主键字段名自动适配."""

    model_class: type[models.Base]
    pk_fields: Sequence[str]

    def bulk_upsert(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        # 过滤掉空记录并补齐主键
        cleaned: list[dict[str, Any]] = []
        for rec in records:
            if not rec:
                continue
            rec = {k: v for k, v in rec.items() if k in self.model_class.__table__.columns}
            if not all(rec.get(pk) not in (None, "") for pk in self.pk_fields):
                continue
            cleaned.append(rec)
        if not cleaned:
            return 0

        dialect = self.session.bind.dialect.name if self.session.bind else "sqlite"
        if dialect == "postgresql":
            insert_cls = pg_insert
        else:
            insert_cls = sqlite_insert

        stmt = insert_cls(self.model_class).values(cleaned)
        update_dict = {c.name: c for c in stmt.excluded if c.name not in self.pk_fields}
        if update_dict:
            stmt = stmt.on_conflict_do_update(index_elements=list(self.pk_fields), set_=update_dict)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=list(self.pk_fields))
        result = self.session.execute(stmt)
        return result.rowcount if result.rowcount is not None else len(cleaned)

    def get_all(self) -> Iterable[dict[str, Any]]:
        for row in self.session.execute(select(self.model_class)).scalars():
            yield self._to_dict(row)

    def get_by_id(self, **kwargs: Any) -> dict[str, Any] | None:
        stmt = select(self.model_class)
        for key, value in kwargs.items():
            stmt = stmt.where(getattr(self.model_class, key) == value)
        row = self.session.execute(stmt).scalar_one_or_none()
        return self._to_dict(row) if row else None

    def _to_dict(self, obj: models.Base) -> dict[str, Any]:
        return {
            col.name: getattr(obj, col.name)
            for col in obj.__table__.columns  # type: ignore[attr-defined]
        }


class GeneRepository(_SimpleRepository):
    model_class = models.Gene
    pk_fields = ("symbol",)


class CompoundRepository(_SimpleRepository):
    model_class = models.Compound
    pk_fields = ("name",)


class DiseaseRepository(_SimpleRepository):
    model_class = models.Disease
    pk_fields = ("name",)


class PathwayRepository(_SimpleRepository):
    model_class = models.Pathway
    pk_fields = ("id",)

    def bulk_upsert(self, records: list[dict[str, Any]]) -> int:
        # Pathway 使用自增主键, 按 (name, source) 去重; external_id 仅在存在时参与查询
        if not records:
            return 0
        count = 0
        for rec in records:
            name = rec.get("name")
            source = rec.get("source")
            if not name or not source:
                continue
            stmt = select(models.Pathway).where(
                models.Pathway.name == name,
                models.Pathway.source == source,
            )
            if rec.get("external_id") is not None:
                stmt = stmt.where(models.Pathway.external_id == rec["external_id"])
            existing = self.session.execute(stmt).scalar_one_or_none()
            if existing:
                for key, value in rec.items():
                    if key != "id":
                        setattr(existing, key, value)
            else:
                self.session.add(models.Pathway(**rec))
                count += 1
        return count


class CellTypeRepository(_SimpleRepository):
    model_class = models.CellType
    pk_fields = ("name",)


class DatasetRepository(_SimpleRepository):
    model_class = models.Dataset
    pk_fields = ("accession",)


class PPIRepository(_SimpleRepository):
    model_class = models.PPIEdge
    pk_fields = ("protein_a_id", "protein_b_id", "source")


class TFTargetRepository(_SimpleRepository):
    model_class = models.TFTargetEdge
    pk_fields = ("tf_id", "target_id", "source")


class CompoundTargetRepository(_SimpleRepository):
    model_class = models.CompoundTargetEdge
    pk_fields = ("compound_id", "gene_id", "source")


class DiseaseGeneRepository(_SimpleRepository):
    model_class = models.DiseaseGeneEdge
    pk_fields = ("disease_id", "gene_id", "source")


class GenePathwayRepository(_SimpleRepository):
    model_class = models.GenePathwayEdge
    pk_fields = ("gene_id", "pathway_id", "source")


class CellTypeMarkerRepository(_SimpleRepository):
    model_class = models.CellTypeMarkerEdge
    pk_fields = ("cell_type_id", "gene_id", "source")


class LigandReceptorRepository(_SimpleRepository):
    model_class = models.LigandReceptorEdge
    pk_fields = ("ligand_id", "receptor_id", "source")


class GeneCoexpRepository(_SimpleRepository):
    model_class = models.GeneCoexpEdge
    pk_fields = ("gene_a_id", "gene_b_id", "source")


class CompoundCompoundSimilarityRepository(_SimpleRepository):
    model_class = models.CompoundCompoundSimilarityEdge
    pk_fields = ("compound_a_id", "compound_b_id", "source")


class PathwayPathwaySimilarityRepository(_SimpleRepository):
    model_class = models.PathwayPathwaySimilarityEdge
    pk_fields = ("pathway_a_id", "pathway_b_id", "source")


class CompoundDiseaseRepository(_SimpleRepository):
    model_class = models.CompoundDiseaseEdge
    pk_fields = ("compound_id", "disease_id", "source")


class DiseaseDiseaseSimilarityRepository(_SimpleRepository):
    model_class = models.DiseaseDiseaseSimilarityEdge
    pk_fields = ("disease_a_id", "disease_b_id", "source")


class MiRNARepository(_SimpleRepository):
    model_class = models.MiRNA
    pk_fields = ("name",)


class MiRNATargetRepository(_SimpleRepository):
    model_class = models.MiRNATargetEdge
    pk_fields = ("mirna_id", "gene_id", "source")


class DifferentialExpressionRepository(_SimpleRepository):
    model_class = models.DifferentialExpression
    pk_fields = ("dataset_id", "gene_id")


class CompoundPropertyRepository(_SimpleRepository):
    model_class = models.CompoundProperty
    pk_fields = ("compound_id", "property_name")


class CompoundFingerprintRepository(_SimpleRepository):
    model_class = models.CompoundFingerprint
    pk_fields = ("compound_id", "fingerprint_type")


class CompoundEmbeddingRepository(_SimpleRepository):
    model_class = models.CompoundEmbedding
    pk_fields = ("compound_id", "model_name")
