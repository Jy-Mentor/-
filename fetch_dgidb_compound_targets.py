"""从 DGIdb 5.0 GraphQL API 获取化合物-靶点相互作用.

仅保留药物名称与项目化合物列表匹配（或已知同义词映射）的边，
用于补充 compound-target 边来源。
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
OUT_CSV = NETWORK_DIR / "dgidb_compound_targets.csv"
METADATA_JSON = PROJECT_ROOT / "external_data" / "dgidb_metadata.json"

DGIDB_GRAPHQL = "https://dgidb.org/api/graphql"
BATCH_SIZE = 25
SLEEP_SECONDS = 0.5

# 化合物同义词映射: DGIdb drug name -> 项目化合物名
COMPOUND_SYNONYMS: dict[str, str] = {
    "BETA-CARYOPHYLLENE": "BCP",
    "CARYOPHYLLENE": "BCP",
    "ASCORBIC ACID": "VC",
    "VITAMIN C": "VC",
    "FERROSTATIN-1": "Fer-1",
    "DEFEROXAMINE": "DFO",
    "LIPROXSTATIN-1": "Lip-1",
    "CURCUMIN": "Curcumin",
    "QUERCETIN": "Quercetin",
    "NARINGENIN": "Naringenin",
    "KAEMPFEROL": "Kaempferol",
    "LUTEOLIN": "Luteolin",
    "APIGENIN": "Apigenin",
    "GENISTEIN": "Genistein",
    "RESVERATROL": "Resveratrol",
    "BAICALEIN": "Baicalein",
    "WOGONIN": "Wogonin",
    "CHRYSIN": "Chrysin",
    "HESPERETIN": "Hesperetin",
    "DAIDZEIN": "Daidzein",
    "FORMONONETIN": "Formononetin",
    "OROXYLIN A": "Oroxylin A",
    "SCUTELLAREIN": "Scutellarein",
    "TANGERETIN": "Tangeretin",
    "NOBILETIN": "Nobiletin",
    "FISETIN": "Fisetin",
    "GALANGIN": "Galangin",
    "PHLORETIN": "Phloretin",
    "CATECHIN": "Catechin",
    "EPICATECHIN": "Epicatechin",
    "TETRAMETHYLPYRAZINE": "Tetramethylpyrazine",
    "TANSHINONE IIA": "Tanshinone IIA",
    "HONOKIOL": "Honokiol",
    "MAGNOLOL": "Magnolol",
    "SCHISANDRIN": "Schisandrin",
    "SCHISANDRIN B": "Schisandrin B",
    "DIHYDROARTEMISININ": "Dihydroartemisinin",
    "ARTEMISININ": "Artemisinin",
    "SHIKONIN": "Shikonin",
    "PLUMBAGIN": "Plumbagin",
    "EMODIN": "Emodin",
    "ALOE EMODIN": "Aloe emodin",
    "PHYSCION": "Physcion",
    "CHRYSOPHANOL": "Chrysophanol",
    "SALIDROSIDE": "Salidroside",
    "TYROSOL": "Tyrosol",
    "HYDROXYTYROSOL": "Hydroxytyrosol",
    "SULFORAPHANE": "Sulforaphane",
    "DIMETHYL FUMARATE": "Dimethyl fumarate",
    "EDARAVONE": "Edaravone",
    "IDEBENONE": "Idebenone",
    "ALPHA-LIPOIC ACID": "Alpha-lipoic acid",
    "LIPOIC ACID": "Alpha-lipoic acid",
    "MELATONIN": "Melatonin",
    "HUPERZINE A": "Huperzine A",
    "N-ACETYLCYSTEINE": "N-acetylcysteine",
    "CINNAMALDEHYDE": "Cinnamaldehyde",
    "CINNAMIC ACID": "Cinnamic_acid",
    "EUCALYPTOL": "Eucalyptol",
    "BORNEOL": "Borneol",
    "PINOCEMBRIN": "Pinocembrin",
    "DANSHENSU": "Danshensu",
}

# 明确排除的近似名称：避免将衍生物误匹配到项目中的母体化合物
BLOCKLIST_PREFIXES: tuple[str, ...] = (
    "EPIGALLOCATECHIN",
    "EPICATECHIN GALLATE",
    "GALLOCATECHIN",
    "THEAFLAVIN",
)


def load_core_genes() -> list[str]:
    """从铁衰老基因文件加载 98 个核心基因."""
    gene_file = PROJECT_ROOT / "铁衰老基因.txt"
    if not gene_file.exists():
        raise FileNotFoundError(f"铁衰老基因文件不存在: {gene_file}")
    with gene_file.open(encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]


def load_project_compounds() -> set[str]:
    """从 compound_smiles.csv 加载项目化合物名."""
    smiles_file = NETWORK_DIR / "compound_smiles.csv"
    if not smiles_file.exists():
        raise FileNotFoundError(f"化合物 SMILES 文件不存在: {smiles_file}")
    df = pd.read_csv(smiles_file)
    return set(df["compound"].astype(str).str.strip())


def normalize_compound_name(drug_name: str, project_compounds: set[str]) -> str | None:
    """将 DGIdb drug name 归一化为项目化合物名."""
    key = drug_name.strip().upper()

    # 明确排除近似衍生物，防止 EGCG 等被误归为 Catechin
    if any(key.startswith(prefix) for prefix in BLOCKLIST_PREFIXES):
        return None

    # 直接同义词映射
    if key in COMPOUND_SYNONYMS:
        return COMPOUND_SYNONYMS[key]

    # 项目化合物名直接匹配（忽略大小写、下划线/空格差异）
    for pc in project_compounds:
        pc_norm = pc.upper().replace("_", " ").replace("-", " ").strip()
        if key == pc_norm:
            return pc

    # 子串匹配: 项目化合物名是 drug name 的一部分（如 "CARYOPHYLLENE" 包含于 "BETA-CARYOPHYLLENE"）
    for pc in project_compounds:
        pc_norm = pc.upper().replace("_", " ").replace("-", " ").strip()
        if pc_norm and (pc_norm in key or key in pc_norm):
            # 避免过短名称误匹配
            if len(pc_norm) >= 4:
                return pc

    return None


def build_gene_query(genes: list[str]) -> str:
    """构建 GraphQL 查询."""
    gene_list = ", ".join(f'"{g}"' for g in genes)
    return f"""
{{
  genes(names: [{gene_list}]) {{
    nodes {{
      name
      interactions {{
        drug {{ name }}
        interactionTypes {{ type }}
        interactionScore
        sources {{
          sourceDbName
        }}
      }}
    }}
  }}
}}
"""


def fetch_interactions(genes: list[str]) -> dict[str, Any]:
    """向 DGIdb GraphQL 发送一次查询."""
    query = build_gene_query(genes)
    resp = requests.post(
        DGIDB_GRAPHQL,
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def confidence_level(score: float) -> str:
    if score >= 0.8:
        return "high"
    if score >= 0.5:
        return "medium"
    return "low"


def main() -> None:
    core_genes = load_core_genes()
    project_compounds = load_project_compounds()
    logger.info("核心基因: %d 个，项目化合物: %d 个", len(core_genes), len(project_compounds))

    all_records: list[dict[str, Any]] = []
    unmatched_drugs: set[str] = set()
    errors: list[str] = []

    for i in range(0, len(core_genes), BATCH_SIZE):
        batch = core_genes[i : i + BATCH_SIZE]
        logger.info("查询 DGIdb: 基因 %d-%d / %d", i + 1, i + len(batch), len(core_genes))
        try:
            data = fetch_interactions(batch)
            nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
            for node in nodes:
                gene = node.get("name", "").upper()
                for interaction in node.get("interactions", []):
                    drug = interaction.get("drug", {}).get("name", "")
                    score = interaction.get("interactionScore")
                    types = [t.get("type", "") for t in interaction.get("interactionTypes", [])]
                    sources = [s.get("sourceDbName", "") for s in interaction.get("sources", [])]

                    compound = normalize_compound_name(drug, project_compounds)
                    if compound is None:
                        unmatched_drugs.add(drug)
                        continue

                    if score is None:
                        raw_score = 0.4
                    else:
                        raw_score = float(score)

                    # DGIdb interactionScore 无上界，以 5.0 为参考上限截断到 [0,1]
                    confidence = min(max(raw_score, 0.0), 1.0)

                    all_records.append(
                        {
                            "compound": compound,
                            "gene": gene,
                            "interaction_type": "|".join(t for t in types if t) or "unknown",
                            "dgidb_drug_name": drug,
                            "dgidb_score": round(raw_score, 6),
                            "dgidb_sources": "|".join(s for s in sources if s) or "DGIdb",
                            "source": "DGIdb",
                            "confidence": round(confidence, 4),
                            "confidence_level": confidence_level(confidence),
                            "download_date": time.strftime("%Y-%m-%d"),
                        }
                    )
        except Exception:
            traceback.print_exc()
            errors.append(f"batch_{i}")
            continue

        time.sleep(SLEEP_SECONDS)

    if not all_records:
        logger.warning("DGIdb 未返回任何与项目化合物匹配的相互作用")

    # 按 compound,gene 去重，保留最高 score
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in all_records:
        key = (rec["compound"], rec["gene"])
        existing = dedup.get(key)
        if existing is None or rec["confidence"] > existing["confidence"]:
            dedup[key] = rec

    df = pd.DataFrame(list(dedup.values()))
    if not df.empty:
        df = df.sort_values(["compound", "gene"]).reset_index(drop=True)
        df.to_csv(OUT_CSV, index=False)
        logger.info("已保存 DGIdb 匹配边: %s (%d 条)", OUT_CSV, len(df))
    else:
        OUT_CSV.write_text("compound,gene,interaction_type,dgidb_drug_name,dgidb_score,dgidb_sources,source,confidence,confidence_level,download_date\n")
        logger.warning("无匹配记录，写入空 CSV: %s", OUT_CSV)

    # 元数据
    metadata = {
        "query_date": time.strftime("%Y-%m-%d"),
        "api_endpoint": DGIDB_GRAPHQL,
        "genes_queried": len(core_genes),
        "compounds_in_project": len(project_compounds),
        "matched_records": len(df) if not df.empty else 0,
        "unmatched_drug_count": len(unmatched_drugs),
        "sample_unmatched_drugs": sorted(list(unmatched_drugs))[:20],
        "errors": errors,
    }
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_JSON.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    logger.info("已保存元数据: %s", METADATA_JSON)


if __name__ == "__main__":
    main()
