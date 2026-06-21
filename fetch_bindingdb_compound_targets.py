"""从 BindingDB REST API 获取化合物-靶点亲和力数据。

参考:
- BindingDB: https://www.bindingdb.org/
- REST endpoint: https://bindingdb.org/rest/getTargetByCompound
- mygene.info: https://mygene.info/

输入: network_files/compound_smiles.csv
输出: network_files/bindingdb_compound_targets.csv
"""

from __future__ import annotations

import json
import logging
import re
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
NETWORK_DIR = BASE_DIR / "network_files"
EXTERNAL_DIR = BASE_DIR / "external_data"
INPUT_CSV = NETWORK_DIR / "compound_smiles.csv"
OUTPUT_CSV = NETWORK_DIR / "bindingdb_compound_targets.csv"
METADATA_JSON = EXTERNAL_DIR / "bindingdb_download_metadata.json"
GENE_LIST_PATH = BASE_DIR / "铁衰老基因.txt"

BINDINGDB_URL = "https://bindingdb.org/rest/getTargetByCompound"
MYGENE_URL = "https://mygene.info/v3/query"

# 亲和力阈值 (nM); 仅保留 <= 100 μM 的记录
MAX_AFFINITY_NM = 100_000.0


def load_gene_set(path: Path) -> set[str]:
    genes: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(f"核心基因集文件不存在: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


def load_compounds(csv_path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        smiles = str(row.get("CanonicalSMILES", "")).strip()
        if name and smiles:
            records.append({"compound": name, "smiles": smiles})
    return records


def parse_numeric_affinity(value: Any) -> float | None:
    """从 BindingDB 亲和力字符串中提取数值."""
    if value is None:
        return None
    s = str(value).strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def affinity_confidence(value_nm: float, affinity_type: str) -> float:
    """根据亲和力数值与类型映射置信度."""
    if value_nm <= 0 or pd.isna(value_nm):
        return 0.35
    aff = str(affinity_type).strip().upper()
    if aff in {"KI", "KD", "IC50"}:
        if value_nm <= 100.0:
            return 0.90
        if value_nm <= 1_000.0:
            return 0.75
        if value_nm <= 10_000.0:
            return 0.55
        if value_nm <= 100_000.0:
            return 0.40
        return 0.30
    if aff == "EC50":
        if value_nm <= 100.0:
            return 0.80
        if value_nm <= 1_000.0:
            return 0.60
        if value_nm <= 10_000.0:
            return 0.45
        if value_nm <= 100_000.0:
            return 0.35
        return 0.25
    # 其他类型保守赋值
    if value_nm <= 10_000.0:
        return 0.50
    return 0.35


def map_target_to_gene(
    target_name: str,
    session: requests.Session,
    cache: dict[str, str | None],
) -> str | None:
    """通过 mygene.info 将 BindingDB 靶点名称解析为 HGNC 基因符号."""
    if target_name in cache:
        return cache[target_name]

    # 清理特殊字符,避免 mygene 查询语法错误
    q = re.sub(r"[\(\)/\",:;]+", " ", target_name)
    q = re.sub(r"\s+", " ", q).strip()

    try:
        resp = session.get(
            MYGENE_URL,
            params={
                "q": q,
                "species": "human",
                "fields": "symbol,name,alias",
                "size": 3,
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            symbol = str(hits[0].get("symbol", "")).strip()
            if symbol:
                cache[target_name] = symbol.upper()
                return cache[target_name]
    except Exception:
        logger.debug("mygene mapping failed for '%s'", target_name)
        traceback.print_exc()

    cache[target_name] = None
    return None


def fetch_bindingdb_targets(
    compound: str,
    smiles: str,
    session: requests.Session,
) -> list[dict[str, Any]]:
    """查询 BindingDB 并返回原始亲和力记录."""
    records: list[dict[str, Any]] = []
    try:
        resp = session.get(
            BINDINGDB_URL,
            params={"smiles": smiles},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            logger.warning("%s: BindingDB 未找到记录", compound)
        else:
            logger.warning("%s: BindingDB 查询失败 %s", compound, e)
        return records
    except Exception:
        logger.warning("%s: BindingDB 请求异常", compound)
        traceback.print_exc()
        return records

    response = data.get("getLindsByUniprotResponse", {})
    affinities = response.get("bdb.affinities", [])
    if not affinities:
        logger.info("%s: BindingDB 返回 0 条亲和力记录", compound)
        return records

    for entry in affinities:
        target = str(entry.get("bdb.target", "")).strip()
        species = str(entry.get("bdb.species", "")).strip()
        aff_type = str(entry.get("bdb.affinity_type", "")).strip()
        aff_raw = entry.get("bdb.affinity", "")
        aff_value = parse_numeric_affinity(aff_raw)

        if not target or not aff_type:
            continue
        # 仅保留人类靶点
        if species.lower() != "human":
            continue

        records.append(
            {
                "compound": compound,
                "target_name": target,
                "affinity_type": aff_type.upper(),
                "affinity_value": aff_value,
                "affinity_raw": str(aff_raw).strip(),
                "species": species,
            }
        )
    return records


def main() -> int:
    core_genes = load_gene_set(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    compounds = load_compounds(INPUT_CSV)
    logger.info("加载 %d 个化合物 SMILES", len(compounds))

    bdb_session = requests.Session()
    bdb_session.headers.update({"Accept": "application/json"})
    mg_session = requests.Session()

    target_to_gene: dict[str, str | None] = {}
    raw_records: list[dict[str, Any]] = []

    for idx, comp in enumerate(compounds, 1):
        name = comp["compound"]
        smiles = comp["smiles"]
        logger.info("[%d/%d] BindingDB: %s", idx, len(compounds), name)

        try:
            entries = fetch_bindingdb_targets(name, smiles, bdb_session)
            for entry in entries:
                gene = map_target_to_gene(entry["target_name"], mg_session, target_to_gene)
                if gene is None:
                    continue
                entry["gene"] = gene
                raw_records.append(entry)
        except Exception:
            logger.warning("%s: 处理异常", name)
            traceback.print_exc()

        time.sleep(0.3)

    if not raw_records:
        logger.warning("未从 BindingDB 获取到任何记录")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "target_name",
                "affinity_type",
                "affinity_value",
                "affinity_raw",
                "confidence",
                "confidence_level",
                "source",
                "download_date",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        return 0

    df = pd.DataFrame(raw_records)
    df["confidence"] = df.apply(
        lambda row: affinity_confidence(
            row["affinity_value"] if pd.notna(row["affinity_value"]) else float("nan"),
            row["affinity_type"],
        ),
        axis=1,
    )
    df["confidence_level"] = df["confidence"].apply(
        lambda x: "high" if x >= 0.80 else ("medium" if x >= 0.60 else "low")
    )

    # 过滤: 必须有基因符号且在核心基因集中,亲和力可解析且 <= 100 μM
    df = df[df["gene"].isin(core_genes)]
    df = df[df["affinity_value"].notna()]
    df = df[df["affinity_value"] <= MAX_AFFINITY_NM]

    if df.empty:
        logger.warning("过滤后无有效记录")
        df.to_csv(OUTPUT_CSV, index=False)
        return 0

    # 同一 compound-gene 保留最佳(最低)亲和力
    df = df.sort_values("affinity_value", ascending=True)
    df = df.drop_duplicates(subset=["compound", "gene"], keep="first")

    df["source"] = "BindingDB"
    df["download_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    out_cols = [
        "compound",
        "gene",
        "target_name",
        "affinity_type",
        "affinity_value",
        "affinity_raw",
        "confidence",
        "confidence_level",
        "source",
        "download_date",
    ]
    df = df[out_cols].sort_values(["compound", "gene"]).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("已保存 %d 条 BindingDB compound-target 边到 %s", len(df), OUTPUT_CSV)

    metadata = {
        "source": "BindingDB",
        "url": BINDINGDB_URL,
        "download_date": pd.Timestamp.now().isoformat(),
        "n_compounds_queried": len(compounds),
        "n_records_raw": len(raw_records),
        "n_records_filtered": len(df),
        "unique_compounds": int(df["compound"].nunique()),
        "unique_genes": int(df["gene"].nunique()),
        "target_name_to_gene_mapping": {
            k: v for k, v in target_to_gene.items() if v is not None
        },
    }
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已保存元数据: %s", METADATA_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
