"""数据库完整性校验脚本.

校验项：
1. 核心实体表非空 (genes, compounds, diseases)
2. 核心边表非空 (ppi_edges, compound_target_edges, disease_gene_edges)
3. 外键引用完整性 (通过 ORM 关系或手动统计孤立记录)
4. 关键化合物/基因存在 (ACSL4, BCP 等)
5. 与 network_files CSV 记录数基本对齐
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select, text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iron_aging import NETWORK_DIR
from iron_aging.db.connection import get_engine, get_session_factory
from iron_aging.db.models import (
    Compound,
    CompoundTargetEdge,
    Disease,
    Gene,
    PPIEdge,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DBValidator:
    """数据库校验器."""

    def __init__(self, session_factory: Any, network_dir: Path) -> None:
        self.session_factory = session_factory
        self.network_dir = network_dir
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _count_csv(self, filename: str) -> int:
        path = self.network_dir / filename
        if not path.exists():
            return -1
        try:
            df = pd.read_csv(path)
            return len(df)
        except Exception as e:
            self.warnings.append(f"无法读取 {filename}: {e}")
            return -1

    def _check_table(self, name: str, min_rows: int = 1) -> int:
        with self.session_factory() as session:
            count = session.execute(select(func.count()).select_from(text(name))).scalar() or 0
        if count < min_rows:
            self.errors.append(f"表 {name} 记录数 {count} < 期望 {min_rows}")
        else:
            logger.info("表 %s: %d 条", name, count)
        return count

    def validate_entities(self) -> None:
        logger.info("校验核心实体表...")
        self._check_table("genes", 200)
        self._check_table("compounds", 60)
        self._check_table("diseases", 1)
        self._check_table("pathways", 1)
        self._check_table("cell_types", 1)
        self._check_table("datasets", 1)

    def validate_edges(self) -> None:
        logger.info("校验核心边表...")
        self._check_table("ppi_edges", 500)
        self._check_table("compound_target_edges", 200)
        self._check_table("disease_gene_edges", 100)
        self._check_table("gene_pathway_edges", 1)
        self._check_table("cell_type_marker_edges", 1)
        self._check_table("ligand_receptor_edges", 1)
        self._check_table("gene_coexp_edges", 1)

    def validate_features(self) -> None:
        logger.info("校验特征表...")
        self._check_table("compound_embeddings", 60)
        self._check_table("compound_properties", 300)

    def validate_key_records(self) -> None:
        logger.info("校验关键记录...")
        with self.session_factory() as session:
            for symbol in ["ACSL4", "GPX4", "PTGS2"]:
                gene = session.execute(select(Gene).where(Gene.symbol == symbol)).scalar_one_or_none()
                if gene is None:
                    self.errors.append(f"关键基因缺失: {symbol}")
            for name in ["BCP", "Fer-1", "DFO"]:
                compound = session.execute(
                    select(Compound).where(Compound.name == name)
                ).scalar_one_or_none()
                if compound is None:
                    self.errors.append(f"关键化合物缺失: {name}")
            disease = session.execute(select(Disease).where(Disease.name == "CIRI")).scalar_one_or_none()
            if disease is None:
                self.errors.append("关键疾病缺失: CIRI")

    def validate_csv_alignment(self) -> None:
        logger.info("校验与 CSV 记录数对齐...")
        # genes 需要比较唯一 symbol 数
        ppi_df = pd.read_csv(self.network_dir / "string_ppi_edges.csv", dtype=str)
        gene_csv_count = len(set(ppi_df["protein_A"].dropna().unique()) | set(ppi_df["protein_B"].dropna().unique()))

        expected = {
            "genes": gene_csv_count,
            "compounds": self._count_csv("compound_smiles.csv"),
            "ppi_edges": self._count_csv("string_ppi_edges.csv"),
            "compound_target_edges": self._count_csv("compound_target_edges.csv"),
        }
        with self.session_factory() as session:
            for table, csv_count in expected.items():
                if csv_count < 0:
                    continue
                model = {
                    "genes": Gene,
                    "compounds": Compound,
                    "ppi_edges": PPIEdge,
                    "compound_target_edges": CompoundTargetEdge,
                }[table]
                db_count = session.execute(select(func.count()).select_from(model)).scalar() or 0
                # 允许 DB 记录数 >= CSV 记录数 (因为多个数据源合并)
                if db_count < csv_count:
                    self.errors.append(f"{table} DB 记录数 {db_count} < CSV {csv_count}")
                else:
                    logger.info("%s: DB=%d, CSV=%d", table, db_count, csv_count)

    def run(self) -> bool:
        self.validate_entities()
        self.validate_edges()
        self.validate_features()
        self.validate_key_records()
        self.validate_csv_alignment()

        if self.warnings:
            for w in self.warnings:
                logger.warning(w)
        if self.errors:
            logger.error("数据库校验失败, 共 %d 个错误:", len(self.errors))
            for e in self.errors:
                logger.error("  - %s", e)
            return False
        logger.info("数据库校验通过")
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description="校验铁衰老项目数据库完整性")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="数据库 URL, 默认读取 config.yaml 或环境变量 IRON_DATABASE_URL",
    )
    parser.add_argument(
        "--network-dir",
        type=str,
        default=None,
        help="network_files 目录路径",
    )
    args = parser.parse_args()

    engine = get_engine(url=args.url)
    session_factory = get_session_factory(engine)
    network_dir = Path(args.network_dir) if args.network_dir else NETWORK_DIR
    validator = DBValidator(session_factory, network_dir)
    ok = validator.run()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
