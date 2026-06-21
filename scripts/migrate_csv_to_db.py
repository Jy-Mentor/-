"""将 network_files/*.csv 迁移到 PostgreSQL/SQLite.

迁移顺序：
1. 实体：genes, compounds, diseases, pathways, cell_types, datasets
2. 边：ppi, tf_target, compound_target, disease_gene, gene_pathway,
      cell_type_marker, ligand_receptor, gene_coexp
3. 特征：compound_properties, compound_embeddings, pockets
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

# 允许从项目根目录导入 src
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iron_aging import NETWORK_DIR
from iron_aging.db.connection import get_engine, get_session_factory
from iron_aging.db.models import Base
from iron_aging.db.repositories import (
    CellTypeMarkerRepository,
    CellTypeRepository,
    CompoundCompoundSimilarityRepository,
    CompoundEmbeddingRepository,
    CompoundPropertyRepository,
    CompoundRepository,
    CompoundTargetRepository,
    DatasetRepository,
    DiseaseGeneRepository,
    DiseaseRepository,
    GeneCoexpRepository,
    GenePathwayRepository,
    GeneRepository,
    LigandReceptorRepository,
    PathwayPathwaySimilarityRepository,
    PathwayRepository,
    PPIRepository,
    TFTargetRepository,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CSVMigrator:
    """CSV 迁移器."""

    def __init__(self, network_dir: Path, session: Session) -> None:
        self.network_dir = network_dir
        self.session = session
        self.stats: dict[str, int] = {}

        # Repository 实例
        self.gene_repo = GeneRepository(session)
        self.compound_repo = CompoundRepository(session)
        self.disease_repo = DiseaseRepository(session)
        self.pathway_repo = PathwayRepository(session)
        self.cell_type_repo = CellTypeRepository(session)
        self.dataset_repo = DatasetRepository(session)
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
        self.prop_repo = CompoundPropertyRepository(session)
        self.emb_repo = CompoundEmbeddingRepository(session)

    def _read_csv(self, name: str) -> pd.DataFrame | None:
        path = self.network_dir / name
        if not path.exists():
            logger.warning("CSV 不存在, 跳过: %s", path)
            return None
        try:
            return pd.read_csv(path, dtype=str)
        except Exception as e:
            logger.error("读取 CSV 失败 %s: %s", path, e)
            traceback.print_exc()
            return None

    def _safe_float(self, value: Any, default: float | None = None) -> float | None:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    def migrate_genes(self) -> int:
        df = self._read_csv("string_ppi_edges.csv")
        if df is None:
            return 0
        symbols = set(df["protein_A"].dropna().unique()) | set(df["protein_B"].dropna().unique())
        records = [{"symbol": str(s).strip().upper(), "source": "STRING_PPI"} for s in symbols if s]
        count = self.gene_repo.bulk_upsert(records)
        self.stats["genes"] = count
        logger.info("迁移 genes: %d", count)
        return count

    def migrate_compounds(self) -> int:
        df = self._read_csv("compound_smiles.csv")
        if df is None:
            return 0
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            name = str(row.get("compound", "")).strip()
            if not name:
                continue
            cid = self._safe_float(row.get("cid"))
            records.append(
                {
                    "name": name,
                    "canonical_smiles": str(row.get("CanonicalSMILES", "")) or None,
                    "pubchem_cid": int(cid) if cid is not None else None,
                    "source": "compound_smiles.csv",
                }
            )
        count = self.compound_repo.bulk_upsert(records)
        self.stats["compounds"] = count
        logger.info("迁移 compounds: %d", count)
        return count

    def migrate_diseases(self) -> int:
        df = self._read_csv("disease_gene_associations.csv")
        if df is None:
            return 0
        names = df["disease"].dropna().unique()
        records = [{"name": str(n).strip()} for n in names if n]
        count = self.disease_repo.bulk_upsert(records)
        self.stats["diseases"] = count
        logger.info("迁移 diseases: %d", count)
        return count

    def migrate_pathways(self) -> int:
        df = self._read_csv("kegg_pathway_genes.csv")
        if df is None:
            return 0
        grouped = df.groupby(["pathway", "kegg_id"]).first().reset_index()
        records: list[dict[str, Any]] = []
        for _, row in grouped.iterrows():
            records.append(
                {
                    "name": str(row["pathway"]).strip(),
                    "source": "KEGG_REST",
                    "external_id": str(row["kegg_id"]).strip() if pd.notna(row.get("kegg_id")) else None,
                }
            )
        count = self.pathway_repo.bulk_upsert(records)
        self.stats["pathways"] = count
        logger.info("迁移 pathways: %d", count)
        return count

    def migrate_cell_types(self) -> int:
        df = self._read_csv("celltype_marker_genes.csv")
        if df is None:
            return 0
        names = df["celltype"].dropna().unique()
        records = [{"name": str(n).strip(), "tissue": "brain", "species": "mouse"} for n in names if n]
        count = self.cell_type_repo.bulk_upsert(records)
        self.stats["cell_types"] = count
        logger.info("迁移 cell_types: %d", count)
        return count

    def migrate_datasets(self) -> int:
        # 当前项目未提供 dataset CSV, 创建占位数据集以支持后续 DE 迁移
        records = [
            {"accession": "GSE16561", "organism": "Homo sapiens", "source": "placeholder"},
            {"accession": "GSE37587", "organism": "Homo sapiens", "source": "placeholder"},
            {"accession": "GSE61616", "organism": "Homo sapiens", "source": "placeholder"},
        ]
        count = self.dataset_repo.bulk_upsert(records)
        self.stats["datasets"] = count
        logger.info("迁移 datasets (占位): %d", count)
        return count

    def migrate_ppi_edges(self) -> int:
        df = self._read_csv("string_ppi_edges.csv")
        if df is None:
            return 0
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            a = str(row.get("protein_A", "")).strip().upper()
            b = str(row.get("protein_B", "")).strip().upper()
            score = self._safe_float(row.get("score"))
            if a and b and score is not None:
                records.append({
                    "protein_a_id": a,
                    "protein_b_id": b,
                    "score": score,
                    "source": str(row.get("source", "STRING")).strip() or "STRING",
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                })
        count = self.ppi_repo.bulk_upsert(records)
        self.stats["ppi_edges"] = count
        logger.info("迁移 ppi_edges: %d", count)
        return count

    def migrate_tf_edges(self) -> int:
        df = self._read_csv("trrust_tf_target.csv")
        if df is None:
            return 0
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            tf = str(row.get("tf", "")).strip().upper()
            target = str(row.get("target", "")).strip().upper()
            if tf and target:
                records.append({
                    "tf_id": tf,
                    "target_id": target,
                    "source": str(row.get("source", "TRRUST")).strip() or "TRRUST",
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                })
        count = self.tf_repo.bulk_upsert(records)
        self.stats["tf_target_edges"] = count
        logger.info("迁移 tf_target_edges: %d", count)
        return count

    def migrate_compound_target_edges(self) -> int:
        total = 0
        file_list = [
            ("curated", "compound_target_edges.csv"),
            ("ChEMBL", "chembl_compound_targets.csv"),
        ]
        for source_name, filename in file_list:
            df = self._read_csv(filename)
            if df is None:
                continue
            records: list[dict[str, Any]] = []
            seen: set[tuple[str, str]] = set()
            for _, row in df.iterrows():
                compound = str(row.get("compound", "")).strip()
                gene = str(row.get("gene", "")).strip().upper()
                if not compound or not gene:
                    continue
                # ChEMBL 中同一 target_chembl_id 常有多个基因别名, 按 (compound, target_chembl_id) 去重
                target_chembl = ""
                if "target_chembl_id" in row and pd.notna(row.get("target_chembl_id")):
                    target_chembl = str(row["target_chembl_id"]).strip()
                dedup_key = (compound, target_chembl or gene)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                rec: dict[str, Any] = {
                    "compound_id": compound,
                    "gene_id": gene,
                    "source": str(row.get("source", source_name)).strip() or source_name,
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                }
                if target_chembl:
                    rec["target_chembl_id"] = target_chembl
                if "molecule_chembl_id" in row:
                    val = row.get("molecule_chembl_id")
                    rec["molecule_chembl_id"] = str(val) if pd.notna(val) else None
                if "standard_type" in row:
                    val = row.get("standard_type")
                    rec["activity_type"] = str(val) if pd.notna(val) else None
                if "standard_value" in row:
                    rec["standard_value"] = self._safe_float(row.get("standard_value"))
                if "standard_units" in row:
                    val = row.get("standard_units")
                    rec["standard_unit"] = str(val) if pd.notna(val) else None
                if "pchembl_value" in row:
                    rec["pchembl_value"] = self._safe_float(row.get("pchembl_value"))
                if "confidence_level" in row:
                    val = row.get("confidence_level")
                    rec["confidence_level"] = str(val) if pd.notna(val) else None
                records.append(rec)
            count = self.ct_repo.bulk_upsert(records)
            total += count
            logger.info("迁移 compound_target_edges (%s): %d", source_name, count)
        self.stats["compound_target_edges"] = total
        return total

    def migrate_disease_gene_edges(self) -> int:
        total = 0
        for filename, source_name, disease_col in [
            ("disgenet_ciri_genes.csv", "DisGeNET", "disease"),
            ("opentargets_ciri_genes.csv", "OpenTargets", "disease"),
            ("disease_gene_associations.csv", "curated", "disease"),
        ]:
            df = self._read_csv(filename)
            if df is None:
                continue
            records: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                disease = str(row.get(disease_col, "")).strip()
                gene = str(row.get("gene", "")).strip().upper()
                if not disease or not gene:
                    continue
                rec = {
                    "disease_id": disease,
                    "gene_id": gene,
                    "source": str(row.get("source", source_name)).strip() or source_name,
                    "score": self._safe_float(row.get("score")),
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                }
                if "disease_id" in row and pd.notna(row.get("disease_id")):
                    rec["disease_original_id"] = str(row["disease_id"])
                if "disease_name" in row and pd.notna(row.get("disease_name")):
                    rec["disease_original_name"] = str(row["disease_name"])
                if "target_name" in row and pd.notna(row.get("target_name")):
                    rec["target_name"] = str(row["target_name"])
                records.append(rec)
            count = self.dg_repo.bulk_upsert(records)
            total += count
            logger.info("迁移 disease_gene_edges (%s): %d", source_name, count)
        self.stats["disease_gene_edges"] = total
        return total

    def migrate_gene_pathway_edges(self) -> int:
        total = 0
        # 来源1: KEGG REST 的 mmu ID -> pathway 映射 (暂缺 symbol 映射, 仅建立 pathway)
        df_kegg = self._read_csv("kegg_pathway_genes.csv")
        if df_kegg is not None:
            for _, row in df_kegg.iterrows():
                pathway_name = str(row.get("pathway", "")).strip()
                external_id = str(row.get("kegg_id", "")).strip() if pd.notna(row.get("kegg_id")) else None
                if not pathway_name:
                    continue
                self.pathway_repo.bulk_upsert(
                    [{"name": pathway_name, "source": "KEGG_REST", "external_id": external_id}]
                )

        # 来源2: 基因 symbol -> pathway 富集结果 (可直接建立边)
        df_enrich = self._read_csv("gene_pathway_enrichment.csv")
        if df_enrich is not None:
            pathway_map = {
                (p["name"], p["source"]): p["id"]
                for p in self.pathway_repo.get_all()
            }
            records: list[dict[str, Any]] = []
            seen: set[tuple[str, int]] = set()
            for _, row in df_enrich.iterrows():
                gene = str(row.get("gene", "")).strip().upper()
                pathway_name = str(row.get("pathway", "")).strip()
                source = str(row.get("source", "enrichment")).strip()
                if not gene or not pathway_name:
                    continue
                # 若 pathway 不存在则创建
                pathway_id = pathway_map.get((pathway_name, source))
                if pathway_id is None:
                    self.pathway_repo.bulk_upsert(
                        [{"name": pathway_name, "source": source}]
                    )
                    # 重新加载映射
                    pathway_map = {
                        (p["name"], p["source"]): p["id"]
                        for p in self.pathway_repo.get_all()
                    }
                    pathway_id = pathway_map.get((pathway_name, source))
                if pathway_id is None:
                    continue
                key = (gene, pathway_id)
                if key in seen:
                    continue
                seen.add(key)
                records.append({
                    "gene_id": gene,
                    "pathway_id": pathway_id,
                    "source": source,
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                })
            count = self.gp_repo.bulk_upsert(records)
            total += count
            logger.info("迁移 gene_pathway_edges (enrichment): %d", count)
        self.stats["gene_pathway_edges"] = total
        return total

    def migrate_cell_type_marker_edges(self) -> int:
        df = self._read_csv("celltype_marker_genes.csv")
        if df is None:
            return 0
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            ct = str(row.get("celltype", "")).strip()
            gene = str(row.get("gene", "")).strip().upper()
            if ct and gene:
                records.append({
                    "cell_type_id": ct,
                    "gene_id": gene,
                    "source": (
                        str(row.get("source", "celltype_marker_genes.csv")).strip()
                        or "celltype_marker_genes.csv"
                    ),
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                })
        count = self.ctm_repo.bulk_upsert(records)
        self.stats["cell_type_marker_edges"] = count
        logger.info("迁移 cell_type_marker_edges: %d", count)
        return count

    def migrate_ligand_receptor_edges(self) -> int:
        df = self._read_csv("ligand_receptor_pairs.csv")
        if df is None:
            return 0
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            ligand = str(row.get("ligand", "")).strip().upper()
            receptor = str(row.get("receptor", "")).strip().upper()
            if ligand and receptor:
                records.append({
                    "ligand_id": ligand,
                    "receptor_id": receptor,
                    "source": str(row.get("source", "curated")).strip() or "curated",
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                })
        count = self.lr_repo.bulk_upsert(records)
        self.stats["ligand_receptor_edges"] = count
        logger.info("迁移 ligand_receptor_edges: %d", count)
        return count

    def migrate_gene_coexp_edges(self) -> int:
        df = self._read_csv("gene_coexp_edges.csv")
        if df is None:
            return 0
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            a = str(row.get("gene_A", "")).strip().upper()
            b = str(row.get("gene_B", "")).strip().upper()
            score = self._safe_float(row.get("score"))
            if a and b and score is not None:
                records.append({
                    "gene_a_id": a,
                    "gene_b_id": b,
                    "score": score,
                    "source": str(row.get("source", "WGCNA")).strip() or "WGCNA",
                    "confidence": self._safe_float(row.get("confidence")),
                    "download_date": str(row.get("download_date", "")).strip() or None,
                })
        count = self.coexp_repo.bulk_upsert(records)
        self.stats["gene_coexp_edges"] = count
        logger.info("迁移 gene_coexp_edges: %d", count)
        return count

    def migrate_compound_compound_similarity_edges(self) -> int:
        df = self._read_csv("compound_compound_similarity_edges.csv")
        if df is None:
            return 0
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            a = str(row.get("compound_A", "")).strip()
            b = str(row.get("compound_B", "")).strip()
            similarity = self._safe_float(row.get("similarity"))
            if a and b and similarity is not None:
                records.append({
                    "compound_a_id": a,
                    "compound_b_id": b,
                    "similarity": similarity,
                    "source": str(row.get("source", "RDKit_Morgan_Tanimoto")).strip() or "RDKit_Morgan_Tanimoto",
                    "confidence": self._safe_float(row.get("confidence")),
                    "confidence_level": str(row.get("confidence_level", "")).strip() or None,
                    "download_date": str(row.get("download_date", "")).strip() or None,
                })
        count = self.ccs_repo.bulk_upsert(records)
        self.stats["compound_compound_similarity_edges"] = count
        logger.info("迁移 compound_compound_similarity_edges: %d", count)
        return count

    def migrate_pathway_pathway_similarity_edges(self) -> int:
        df = self._read_csv("pathway_pathway_similarity_edges.csv")
        if df is None:
            return 0
        # 需要把 pathway 名称映射到 pathways.id
        pathway_map = {
            (p["name"], p["source"]): p["id"]
            for p in self.pathway_repo.get_all()
        }
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            name_a = str(row.get("pathway_A", "")).strip()
            name_b = str(row.get("pathway_B", "")).strip()
            jaccard = self._safe_float(row.get("jaccard"))
            if not name_a or not name_b or jaccard is None:
                continue
            # 优先匹配 KEGG_REST 来源, 其次任意来源
            pid_a = pathway_map.get((name_a, "KEGG_REST"))
            pid_b = pathway_map.get((name_b, "KEGG_REST"))
            if pid_a is None or pid_b is None:
                logger.debug(" pathway 名称未映射到 ID: %s / %s", name_a, name_b)
                continue
            records.append({
                "pathway_a_id": pid_a,
                "pathway_b_id": pid_b,
                "jaccard": jaccard,
                "intersection_size": self._safe_int(row.get("intersection_size")),
                "union_size": self._safe_int(row.get("union_size")),
                "source": str(row.get("source", "KEGG_pathway_Jaccard")).strip() or "KEGG_pathway_Jaccard",
                "confidence": self._safe_float(row.get("confidence")),
                "confidence_level": str(row.get("confidence_level", "")).strip() or None,
                "download_date": str(row.get("download_date", "")).strip() or None,
            })
        count = self.pps_repo.bulk_upsert(records)
        self.stats["pathway_pathway_similarity_edges"] = count
        logger.info("迁移 pathway_pathway_similarity_edges: %d", count)
        return count

    def _safe_int(self, value: Any, default: int | None = None) -> int | None:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return default
        try:
            return int(float(value))
        except ValueError:
            return default

    def migrate_compound_embeddings(self) -> int:
        df = self._read_csv("compound_attentivefp_embeddings.csv")
        if df is None:
            return 0
        emb_cols = [c for c in df.columns if c.startswith("emb_")]
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            compound = str(row.get("compound", "")).strip()
            if not compound or not emb_cols:
                continue
            vector = [self._safe_float(row.get(c), 0.0) for c in emb_cols]
            records.append(
                {
                    "compound_id": compound,
                    "model_name": "AttentiveFP",
                    "embedding_vector": vector,
                    "source": "compound_attentivefp_embeddings.csv",
                }
            )
        count = self.emb_repo.bulk_upsert(records)
        self.stats["compound_embeddings"] = count
        logger.info("迁移 compound_embeddings: %d", count)
        return count

    def migrate_compound_properties(self) -> int:
        df = self._read_csv("pubchem_compound_props.csv")
        if df is None:
            return 0
        # 选择数值列
        prop_cols = ["MW", "LogP", "HBD", "HBA", "TPSA", "RotB"]
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            compound = str(row.get("compound", "")).strip()
            if not compound:
                continue
            for col in prop_cols:
                value = self._safe_float(row.get(col))
                if value is not None:
                    records.append(
                        {
                            "compound_id": compound,
                            "property_name": col,
                            "property_value": value,
                            "source": "pubchem_compound_props.csv",
                        }
                    )
        count = self.prop_repo.bulk_upsert(records)
        self.stats["compound_properties"] = count
        logger.info("迁移 compound_properties: %d", count)
        return count

    def run(self) -> dict[str, int]:
        """执行全部迁移."""
        logger.info("开始 CSV -> DB 迁移: %s", self.network_dir)
        # 实体
        self.migrate_genes()
        self.migrate_compounds()
        self.migrate_diseases()
        self.migrate_pathways()
        self.migrate_cell_types()
        self.migrate_datasets()
        self.session.flush()

        # 边
        self.migrate_ppi_edges()
        self.migrate_tf_edges()
        self.migrate_compound_target_edges()
        self.migrate_disease_gene_edges()
        self.migrate_gene_pathway_edges()
        self.migrate_cell_type_marker_edges()
        self.migrate_ligand_receptor_edges()
        self.migrate_gene_coexp_edges()
        self.migrate_compound_compound_similarity_edges()
        self.migrate_pathway_pathway_similarity_edges()
        self.session.flush()

        # 特征
        self.migrate_compound_embeddings()
        self.migrate_compound_properties()
        self.session.flush()

        self.session.commit()
        logger.info("迁移完成. 统计:\n%s", json.dumps(self.stats, indent=2, ensure_ascii=False))
        return self.stats


def main() -> int:
    parser = argparse.ArgumentParser(description="将 network_files CSV 迁移到数据库")
    parser.add_argument(
        "--network-dir",
        type=str,
        default=None,
        help="network_files 目录路径, 默认使用项目根目录下的 network_files",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="数据库 URL, 默认读取 config.yaml 或环境变量 IRON_DATABASE_URL",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="迁移前先创建所有表",
    )
    args = parser.parse_args()

    network_dir = Path(args.network_dir) if args.network_dir else NETWORK_DIR
    engine = get_engine(url=args.url)
    if args.init:
        Base.metadata.create_all(engine)
        logger.info("已创建数据库表")

    SessionFactory = get_session_factory(engine)
    with SessionFactory() as session:
        migrator = CSVMigrator(network_dir, session)
        try:
            migrator.run()
        except Exception as e:
            session.rollback()
            logger.error("迁移失败: %s", e)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
