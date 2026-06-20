"""合并 curated + ChEMBL + STITCH 化合物-靶点数据，更新 network_files/compound_target_edges.csv.

输入:
    network_files/compound_target_edges_curated.csv (curated)
    network_files/chembl_compound_targets.csv       (ChEMBL)
    external_data/stitch/9606.protein_chemical.links.v5.0.tsv.gz
    external_data/stitch/9606.protein.info.v12.0.txt.gz
    external_data/stitch/chemical.aliases.v5.0.tsv.gz
    network_files/compound_smiles.csv               (用于 CID 映射)

输出:
    network_files/compound_target_edges.csv         (合并后)
    external_data/stitch/stitch_compound_targets.csv
    external_data/stitch/stitch_download_metadata.json
"""

from __future__ import annotations

import gzip
import json
import logging
import traceback
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
EXTERNAL_DIR = PROJECT_ROOT / "external_data" / "stitch"

CURATED_CSV = NETWORK_DIR / "compound_target_edges_curated.csv"
CHEMBL_CSV = NETWORK_DIR / "chembl_compound_targets.csv"
COMPOUND_SMILES_CSV = NETWORK_DIR / "compound_smiles.csv"
STITCH_LINKS_GZ = EXTERNAL_DIR / "9606.protein_chemical.links.v5.0.tsv.gz"
STITCH_INFO_GZ = EXTERNAL_DIR / "9606.protein.info.v12.0.txt.gz"
STITCH_ALIASES_GZ = EXTERNAL_DIR / "chemical.aliases.v5.0.tsv.gz"
OUTPUT_CSV = NETWORK_DIR / "compound_target_edges.csv"
STITCH_OUTPUT_CSV = EXTERNAL_DIR / "stitch_compound_targets.csv"
METADATA_JSON = EXTERNAL_DIR / "stitch_download_metadata.json"

STITCH_SCORE_THRESHOLD = 400


def load_compound_cid_map(csv_path: Path) -> dict[str, str]:
    """加载化合物名称到 PubChem CID 的映射."""
    df = pd.read_csv(csv_path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        cid = str(row.get("cid", "")).strip()
        if name and cid:
            mapping[name] = cid
    logger.info("加载 %d 个化合物的 CID 映射", len(mapping))
    return mapping


def load_cid_to_stitch_flat(aliases_path: Path, compound_to_cid: dict[str, str]) -> dict[str, str]:
    """从 STITCH chemical.aliases 构建 PubChem CID -> flat_chemical_id 映射.

    优先匹配 alias == f"CID{cid}"，其次 alias == cid；同时按化合物名称做补充匹配。
    """
    cid_to_flat: dict[str, str] = {}
    target_aliases: dict[str, str] = {}
    for name, cid in compound_to_cid.items():
        target_aliases[f"CID{cid}"] = cid
        target_aliases[cid] = cid
        target_aliases[name.lower()] = cid

    if not aliases_path.exists():
        logger.warning("STITCH aliases 文件不存在: %s", aliases_path)
        return cid_to_flat

    matched = 0
    with gzip.open(aliases_path, "rt", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            flat_chemical = parts[0].strip()
            alias = parts[2].strip()
            cid = target_aliases.get(alias)
            if cid is None:
                cid = target_aliases.get(alias.lower())
            if cid is not None and cid not in cid_to_flat:
                cid_to_flat[cid] = flat_chemical
                matched += 1

    logger.info("STITCH aliases 映射: %d 个 CID -> flat_chemical", len(cid_to_flat))
    return cid_to_flat


def load_stitch_protein_to_gene(info_path: Path) -> dict[str, str]:
    """从 STRING protein.info 构建 protein_id -> gene_symbol 映射."""
    gene_map: dict[str, str] = {}
    with gzip.open(info_path, "rt", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                protein_id = parts[0].strip()
                gene_symbol = parts[1].strip().upper()
                if protein_id and gene_symbol:
                    gene_map[protein_id] = gene_symbol
    logger.info("加载 %d 个 protein -> gene 映射", len(gene_map))
    return gene_map


def load_stitch_targets(
    links_path: Path,
    protein_to_gene: dict[str, str],
    cid_to_compound: dict[str, str],
    cid_to_flat: dict[str, str],
    score_threshold: int = STITCH_SCORE_THRESHOLD,
) -> list[dict]:
    """从 STITCH 文件中提取项目化合物的靶点."""
    results: list[dict] = []
    stats = {"total_rows": 0, "mapped_compounds": 0, "mapped_genes": 0, "filtered": 0}

    if not links_path.exists():
        logger.warning("STITCH links 文件不存在: %s", links_path)
        return results, stats

    target_flats = set(cid_to_flat.values())
    with gzip.open(links_path, "rt", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            stats["total_rows"] += 1
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            chemical = parts[0].strip()
            protein = parts[1].strip()
            try:
                score = int(parts[2].strip())
            except ValueError:
                continue

            if score < score_threshold:
                continue

            if chemical not in target_flats:
                continue
            stats["mapped_compounds"] += 1

            gene = protein_to_gene.get(protein)
            if not gene:
                continue
            stats["mapped_genes"] += 1

            # 将 flat_chemical 反查回化合物名称
            compound: str | None = None
            for cid, flat in cid_to_flat.items():
                if flat == chemical:
                    compound = cid_to_compound.get(cid)
                    if compound:
                        break
            if compound is None:
                continue

            results.append(
                {
                    "compound": compound,
                    "gene": gene,
                    "stitch_score": score,
                    "protein_id": protein,
                    "source": "STITCH",
                }
            )

    logger.info(
        "STITCH: total=%d, 映射到化合物=%d, 映射到基因=%d",
        stats["total_rows"],
        stats["mapped_compounds"],
        stats["mapped_genes"],
    )
    return results, stats


def main() -> int:
    try:
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

        # 1. 加载化合物 CID 映射
        compound_to_cid = load_compound_cid_map(COMPOUND_SMILES_CSV)
        cid_to_compound = {cid: name for name, cid in compound_to_cid.items()}

        # 2. 加载 curated
        curated_records: list[dict] = []
        if CURATED_CSV.exists():
            df = pd.read_csv(CURATED_CSV)
            for _, row in df.iterrows():
                curated_records.append(
                    {
                        "compound": str(row["compound"]).strip(),
                        "gene": str(row["gene"]).strip().upper(),
                        "source": str(row.get("source", "curated")).strip(),
                    }
                )
            logger.info("curated 记录: %d", len(curated_records))
        else:
            logger.error("curated 文件不存在: %s", CURATED_CSV)
            return 1

        # 3. 加载 ChEMBL
        chembl_records: list[dict] = []
        if CHEMBL_CSV.exists():
            df = pd.read_csv(CHEMBL_CSV)
            for _, row in df.iterrows():
                chembl_records.append(
                    {
                        "compound": str(row["compound"]).strip(),
                        "gene": str(row["gene"]).strip().upper(),
                        "source": "ChEMBL",
                    }
                )
            logger.info("ChEMBL 记录: %d", len(chembl_records))
        else:
            logger.warning("ChEMBL 文件不存在: %s", CHEMBL_CSV)

        # 4. 加载 STITCH
        cid_to_flat = load_cid_to_stitch_flat(STITCH_ALIASES_GZ, compound_to_cid)
        protein_to_gene = load_stitch_protein_to_gene(STITCH_INFO_GZ)
        stitch_records, stitch_stats = load_stitch_targets(
            STITCH_LINKS_GZ, protein_to_gene, cid_to_compound, cid_to_flat
        )

        # 保存 STITCH 原始结果
        if stitch_records:
            pd.DataFrame(stitch_records).drop_duplicates(subset=["compound", "gene"]).to_csv(
                STITCH_OUTPUT_CSV, index=False
            )
            logger.info("已保存 STITCH 结果: %s", STITCH_OUTPUT_CSV)

        # 5. 合并并去重：compound-gene 唯一，source 合并为列表
        edge_map: dict[tuple[str, str], set[str]] = {}
        for rec in curated_records + chembl_records + stitch_records:
            key = (rec["compound"], rec["gene"])
            edge_map.setdefault(key, set()).add(rec.get("source", "curated"))

        merged = [
            {"compound": comp, "gene": gene, "source": "|".join(sorted(sources))}
            for (comp, gene), sources in edge_map.items()
        ]
        logger.info("合并后唯一 compound-target 边: %d", len(merged))

        merged_df = pd.DataFrame(merged).sort_values(["compound", "gene"])
        # 备份原输出文件
        if OUTPUT_CSV.exists():
            backup_path = OUTPUT_CSV.with_suffix(".csv.bak")
            backup_path.write_text(OUTPUT_CSV.read_text(encoding="utf-8"), encoding="utf-8")
            logger.info("已备份原输出文件: %s", backup_path)
        merged_df.to_csv(OUTPUT_CSV, index=False)
        logger.info("已写入合并结果: %s", OUTPUT_CSV)

        # 6. 元数据
        metadata = {
            "download_date": pd.Timestamp.now().isoformat(),
            "stitch_score_threshold": STITCH_SCORE_THRESHOLD,
            "stats": {
                "curated": len(curated_records),
                "chembl": len(chembl_records),
                "stitch": len(stitch_records),
                "merged_unique": len(merged),
            },
            "stitch_stats": stitch_stats,
        }
        METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("已写入元数据: %s", METADATA_JSON)

        return 0
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
