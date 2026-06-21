"""SQLAlchemy ORM 模型.

设计原则:
- 实体表使用业务稳定键作为主键(gene.symbol, compound.name 等), 减少迁移复杂度.
- 边表使用复合主键 (source_id, target_id, source), 天然去重.
- 向量/JSON 数据使用通用 JSON 类型, 兼容 PostgreSQL 与 SQLite.
- 时间戳由 ETL/应用层写入, 不依赖数据库触发器.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """ORM 基类."""

    pass


# ---------------------------------------------------------------------------
# 实体表
# ---------------------------------------------------------------------------


class Gene(Base):
    """基因实体."""

    __tablename__ = "genes"

    symbol: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str | None] = mapped_column(String(255))
    organism: Mapped[str | None] = mapped_column(String(64), default="Mus musculus")
    synonyms: Mapped[str | None] = mapped_column(Text)  # 逗号分隔
    ncbi_gene_id: Mapped[int | None] = mapped_column(Integer)
    ensembl_id: Mapped[str | None] = mapped_column(String(32))
    uniprot_id: Mapped[str | None] = mapped_column(String(32))
    source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    # Relationships (仅用于 ORM 导航, 不强加载)
    ppi_edges_a: Mapped[list["PPIEdge"]] = relationship(
        "PPIEdge",
        foreign_keys="PPIEdge.protein_a_id",
        back_populates="protein_a",
        overlaps="ppi_edges_b",
    )
    compound_targets: Mapped[list["CompoundTargetEdge"]] = relationship(
        "CompoundTargetEdge", back_populates="gene"
    )


class Compound(Base):
    """化合物实体."""

    __tablename__ = "compounds"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    canonical_smiles: Mapped[str | None] = mapped_column(Text)
    pubchem_cid: Mapped[int | None] = mapped_column(Integer)
    chembl_id: Mapped[str | None] = mapped_column(String(32))
    inchi_key: Mapped[str | None] = mapped_column(String(64))
    iupac_name: Mapped[str | None] = mapped_column(Text)
    molecular_formula: Mapped[str | None] = mapped_column(String(64))
    source_organism: Mapped[str | None] = mapped_column(String(255))
    source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    targets: Mapped[list["CompoundTargetEdge"]] = relationship(
        "CompoundTargetEdge", back_populates="compound"
    )


class Disease(Base):
    """疾病实体."""

    __tablename__ = "diseases"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    doid: Mapped[str | None] = mapped_column(String(64))
    mesh_id: Mapped[str | None] = mapped_column(String(64))
    mondo_id: Mapped[str | None] = mapped_column(String(64))
    umls_cui: Mapped[str | None] = mapped_column(String(16))
    efo_id: Mapped[str | None] = mapped_column(String(32))
    source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())


class Pathway(Base):
    """通路实体."""

    __tablename__ = "pathways"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(64))  # KEGG / Reactome / GO
    external_id: Mapped[str | None] = mapped_column(String(64))
    organism: Mapped[str | None] = mapped_column(String(64), default="Mus musculus")
    description: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("source", "external_id", name="uix_pathway_source_ext"),)


class CellType(Base):
    """细胞类型实体."""

    __tablename__ = "cell_types"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    tissue: Mapped[str | None] = mapped_column(String(64), default="brain")
    species: Mapped[str | None] = mapped_column(String(64), default="mouse")
    source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())


class Dataset(Base):
    """GEO / 实验数据集."""

    __tablename__ = "datasets"

    accession: Mapped[str] = mapped_column(String(32), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(512))
    organism: Mapped[str | None] = mapped_column(String(64))
    platform: Mapped[str | None] = mapped_column(String(64))
    condition: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())


# ---------------------------------------------------------------------------
# 边表
# ---------------------------------------------------------------------------


class PPIEdge(Base):
    """蛋白-蛋白相互作用边 (STRING 等)."""

    __tablename__ = "ppi_edges"

    protein_a_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    protein_b_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    score: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    download_date: Mapped[str | None] = mapped_column(String(16))

    protein_a: Mapped["Gene"] = relationship(
        "Gene", foreign_keys=[protein_a_id], back_populates="ppi_edges_a"
    )

    __table_args__ = (
        # 无向边按字典序去重
        UniqueConstraint(
            "protein_a_id",
            "protein_b_id",
            "source",
            name="uix_ppi_edge",
        ),
    )


class TFTargetEdge(Base):
    """转录因子-靶基因调控边 (TRRUST 等)."""

    __tablename__ = "tf_target_edges"

    tf_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    target_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    download_date: Mapped[str | None] = mapped_column(String(16))


class CompoundTargetEdge(Base):
    """化合物-靶点生物活性边 (ChEMBL / curated)."""

    __tablename__ = "compound_target_edges"

    compound_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("compounds.name"), primary_key=True
    )
    gene_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    target_chembl_id: Mapped[str | None] = mapped_column(String(32))
    molecule_chembl_id: Mapped[str | None] = mapped_column(String(32))
    activity_type: Mapped[str | None] = mapped_column(String(16))
    standard_value: Mapped[float | None] = mapped_column(Float)
    standard_unit: Mapped[str | None] = mapped_column(String(16))
    pchembl_value: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_level: Mapped[str | None] = mapped_column(String(16))
    download_date: Mapped[str | None] = mapped_column(String(16))

    compound: Mapped["Compound"] = relationship("Compound", back_populates="targets")
    gene: Mapped["Gene"] = relationship("Gene", back_populates="compound_targets")


class DiseaseGeneEdge(Base):
    """疾病-基因关联边 (DisGeNET / OpenTargets)."""

    __tablename__ = "disease_gene_edges"

    disease_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("diseases.name"), primary_key=True
    )
    gene_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    score: Mapped[float | None] = mapped_column(Float)
    disease_original_id: Mapped[str | None] = mapped_column(String(64))
    disease_original_name: Mapped[str | None] = mapped_column(String(255))
    target_name: Mapped[str | None] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    download_date: Mapped[str | None] = mapped_column(String(16))


class GenePathwayEdge(Base):
    """基因-通路归属边."""

    __tablename__ = "gene_pathway_edges"

    gene_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    pathway_id: Mapped[int] = mapped_column(Integer, ForeignKey("pathways.id"), primary_key=True)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    download_date: Mapped[str | None] = mapped_column(String(16))


class CellTypeMarkerEdge(Base):
    """细胞类型标记基因边."""

    __tablename__ = "cell_type_marker_edges"

    cell_type_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("cell_types.name"), primary_key=True
    )
    gene_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    marker_type: Mapped[str | None] = mapped_column(String(32), default="marker")
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    download_date: Mapped[str | None] = mapped_column(String(16))


class LigandReceptorEdge(Base):
    """配体-受体互作边."""

    __tablename__ = "ligand_receptor_edges"

    ligand_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    receptor_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("genes.symbol"), primary_key=True
    )
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    download_date: Mapped[str | None] = mapped_column(String(16))


class GeneCoexpEdge(Base):
    """基因共表达边 (WGCNA)."""

    __tablename__ = "gene_coexp_edges"

    gene_a_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    gene_b_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    score: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    download_date: Mapped[str | None] = mapped_column(String(16))


class CompoundCompoundSimilarityEdge(Base):
    """化合物-化合物化学相似性边 (RDKit Tanimoto)."""

    __tablename__ = "compound_compound_similarity_edges"

    compound_a_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("compounds.name"), primary_key=True
    )
    compound_b_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("compounds.name"), primary_key=True
    )
    similarity: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_level: Mapped[str | None] = mapped_column(String(16))
    download_date: Mapped[str | None] = mapped_column(String(16))


class PathwayPathwaySimilarityEdge(Base):
    """通路-通路基因集 Jaccard 相似性边."""

    __tablename__ = "pathway_pathway_similarity_edges"

    pathway_a_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pathways.id"), primary_key=True
    )
    pathway_b_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("pathways.id"), primary_key=True
    )
    jaccard: Mapped[float] = mapped_column(Float)
    intersection_size: Mapped[int | None] = mapped_column(Integer)
    union_size: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(64), primary_key=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_level: Mapped[str | None] = mapped_column(String(16))
    download_date: Mapped[str | None] = mapped_column(String(16))


# ---------------------------------------------------------------------------
# 特征与实验表
# ---------------------------------------------------------------------------


class DifferentialExpression(Base):
    """差异表达结果."""

    __tablename__ = "differential_expressions"

    dataset_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("datasets.accession"), primary_key=True
    )
    gene_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"), primary_key=True)
    log2fc: Mapped[float | None] = mapped_column(Float)
    padj: Mapped[float | None] = mapped_column(Float)
    stat: Mapped[float | None] = mapped_column(Float)
    contrast: Mapped[str | None] = mapped_column(String(128))
    source: Mapped[str | None] = mapped_column(String(64))


class CompoundProperty(Base):
    """化合物物化性质 (RDKit / PubChem)."""

    __tablename__ = "compound_properties"

    compound_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("compounds.name"), primary_key=True
    )
    property_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    property_value: Mapped[float] = mapped_column(Float)
    property_unit: Mapped[str | None] = mapped_column(String(16))
    source: Mapped[str | None] = mapped_column(String(64))


class CompoundFingerprint(Base):
    """化合物指纹向量."""

    __tablename__ = "compound_fingerprints"

    compound_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("compounds.name"), primary_key=True
    )
    fingerprint_type: Mapped[str] = mapped_column(String(32), primary_key=True)
    dimension: Mapped[int | None] = mapped_column(Integer)
    fingerprint_vector: Mapped[list[float]] = mapped_column(JSON)
    source: Mapped[str | None] = mapped_column(String(64))


class CompoundEmbedding(Base):
    """化合物预训练嵌入 (AttentiveFP 等)."""

    __tablename__ = "compound_embeddings"

    compound_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("compounds.name"), primary_key=True
    )
    model_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding_vector: Mapped[list[float]] = mapped_column(JSON)
    source: Mapped[str | None] = mapped_column(String(64))


class Pocket(Base):
    """蛋白口袋结构特征 (ACSL4 等)."""

    __tablename__ = "pockets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    gene_id: Mapped[str] = mapped_column(String(64), ForeignKey("genes.symbol"))
    pdb_id: Mapped[str | None] = mapped_column(String(16))
    pocket_volume: Mapped[float | None] = mapped_column(Float)
    n_pocket_residues: Mapped[int | None] = mapped_column(Integer)
    avg_hydropathy: Mapped[float | None] = mapped_column(Float)
    n_hbd: Mapped[int | None] = mapped_column(Integer)
    n_hba: Mapped[int | None] = mapped_column(Integer)
    n_aromatic: Mapped[int | None] = mapped_column(Integer)
    n_pos_charged: Mapped[int | None] = mapped_column(Integer)
    n_neg_charged: Mapped[int | None] = mapped_column(Integer)
    n_unique_residues: Mapped[int | None] = mapped_column(Integer)
    avg_bfactor_plddt: Mapped[float | None] = mapped_column(Float)
    backbone_ratio: Mapped[float | None] = mapped_column(Float)
    n_backbone_atoms: Mapped[int | None] = mapped_column(Integer)
    n_sidechain_atoms: Mapped[int | None] = mapped_column(Integer)
    n_literature_hotspots_in_pocket: Mapped[int | None] = mapped_column(Integer)
    n_literature_hotspots: Mapped[int | None] = mapped_column(Integer)
    key_residues: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    source: Mapped[str | None] = mapped_column(String(64))
    cutoff_angstrom: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())

    __table_args__ = (UniqueConstraint("gene_id", "pdb_id", name="uix_pocket_gene_pdb"),)


class Experiment(Base):
    """训练实验元数据."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="running")
    started_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[DateTime | None] = mapped_column(DateTime)
    best_epoch: Mapped[int | None] = mapped_column(Integer)
    best_metric_value: Mapped[float | None] = mapped_column(Float)
    best_metric_name: Mapped[str | None] = mapped_column(String(64))


class ExperimentMetric(Base):
    """实验指标时序记录."""

    __tablename__ = "experiment_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), ForeignKey("experiments.id"))
    epoch: Mapped[int | None] = mapped_column(Integer)
    metric_name: Mapped[str] = mapped_column(String(64))
    metric_value: Mapped[float] = mapped_column(Float)
    split: Mapped[str | None] = mapped_column(String(16))
    logged_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())


class ModelCheckpoint(Base):
    """模型检查点."""

    __tablename__ = "model_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), ForeignKey("experiments.id"))
    epoch: Mapped[int | None] = mapped_column(Integer)
    path: Mapped[str] = mapped_column(String(512))
    is_best: Mapped[bool] = mapped_column(default=False)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[DateTime | None] = mapped_column(DateTime, server_default=func.now())
