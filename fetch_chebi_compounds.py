"""从 ChEBI 下载化合物本体注释并映射到项目化合物。

ChEBI (Chemical Entities of Biological Interest) 是 EMBL-EBI 维护的
小分子本体数据库,提供稳定的 ChEBI ID、名称、结构标识(InChIKey/SMILES)、
分子式与质量等。

参考:
- ChEBI: https://www.ebi.ac.uk/chebi/
- Hastings J, et al. Nucleic Acids Res. 2016;44(D1):D1204-D1210.

输入:
    network_files/compound_smiles.csv

输出:
    network_files/chebi_compound_annotations.csv
    external_data/chebi_download_metadata.json
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
NETWORK_DIR = BASE_DIR / "network_files"
EXTERNAL_DIR = BASE_DIR / "external_data"
INPUT_CSV = NETWORK_DIR / "compound_smiles.csv"
OUTPUT_CSV = NETWORK_DIR / "chebi_compound_annotations.csv"
METADATA_JSON = EXTERNAL_DIR / "chebi_download_metadata.json"

CHEBI_SDF_URL = "https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi_lite_3_stars.sdf.gz"


def normalize_name(name: str) -> str:
    """标准化化合物名称,用于名称匹配."""
    s = str(name).lower()
    s = re.sub(r"[-_/\s]+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def load_compounds(csv_path: Path) -> pd.DataFrame:
    """加载项目化合物 SMILES 表."""
    df = pd.read_csv(csv_path)
    df["compound"] = df["compound"].astype(str).str.strip()
    df["CanonicalSMILES"] = df["CanonicalSMILES"].astype(str).str.strip()
    return df


def compute_inchikey14(smiles: str) -> str | None:
    """从 SMILES 计算 InChIKey 前 14 位(连接层)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        ik = Chem.MolToInchiKey(mol)
        return ik[:14] if ik else None
    except Exception:
        return None


def mol_descriptors(mol: Any) -> dict[str, Any]:
    """从 RDKit mol 计算结构描述符;失败时记录警告并返回 None 字段."""
    out: dict[str, Any] = {
        "smiles": None,
        "inchikey": None,
        "formula": None,
        "exact_mass": None,
        "molecular_weight": None,
        "charge": None,
    }
    if mol is None:
        return out

    _desc_calls = [
        ("smiles", lambda m: Chem.MolToSmiles(m)),
        ("inchikey", lambda m: Chem.MolToInchiKey(m)),
        ("formula", lambda m: rdMolDescriptors.CalcMolFormula(m)),
        ("exact_mass", lambda m: round(Descriptors.ExactMolWt(m), 6)),
        ("molecular_weight", lambda m: round(Descriptors.MolWt(m), 6)),
        ("charge", lambda m: m.GetFormalCharge()),
    ]
    for key, fn in _desc_calls:
        try:
            out[key] = fn(mol)
        except Exception:
            logger.warning("RDKit 描述符 %s 计算失败", key, exc_info=True)
    return out


def match_score(project_name: str, chebi_name: str) -> int:
    """名称匹配得分:精确匹配最高; 仅对较长项目名允许包含匹配,避免短缩写误配."""
    p = normalize_name(project_name)
    c = normalize_name(chebi_name)
    if p == c:
        return 2
    # 短缩写(如 DFO/BCP/VC)容易成为长串的子串,仅当项目名长度>=5时才用包含匹配
    if len(p) >= 5 and (p in c or c in p):
        return 1
    return 0


def download_sdf(url: str, session: requests.Session) -> bytes:
    """下载 gzipped SDF 文件."""
    logger.info("下载 ChEBI SDF: %s", url)
    resp = session.get(url, timeout=180)
    resp.raise_for_status()
    logger.info("下载完成: %d bytes", len(resp.content))
    return resp.content


def parse_chebi_records(sdf_bytes: bytes) -> list[dict[str, Any]]:
    """解析 ChEBI lite SDF,返回记录列表."""
    records: list[dict[str, Any]] = []
    with gzip.GzipFile(fileobj=io.BytesIO(sdf_bytes)) as gz:
        supplier = Chem.ForwardSDMolSupplier(gz)
        for mol in supplier:
            if mol is None:
                continue
            try:
                chebi_id = str(mol.GetProp("ChEBI ID")).strip()
                chebi_name = str(mol.GetProp("ChEBI NAME")).strip()
                star = str(mol.GetProp("STAR")).strip()
            except Exception:
                continue
            desc = mol_descriptors(mol)
            records.append(
                {
                    "chebi_id": chebi_id,
                    "chebi_name": chebi_name,
                    "chebi_star": int(star) if star.isdigit() else None,
                    "smiles": desc["smiles"],
                    "inchikey": desc["inchikey"],
                    "inchikey14": desc["inchikey"][:14] if desc["inchikey"] else None,
                    "formula": desc["formula"],
                    "exact_mass": desc["exact_mass"],
                    "molecular_weight": desc["molecular_weight"],
                    "charge": desc["charge"],
                }
            )
    logger.info("ChEBI SDF 解析记录数: %d", len(records))
    return records


def build_project_inchikeys(compounds_df: pd.DataFrame) -> dict[str, str]:
    """计算项目化合物的 InChIKey 连接层."""
    mapping: dict[str, str] = {}
    for _, row in compounds_df.iterrows():
        compound = str(row["compound"]).strip()
        smiles = str(row["CanonicalSMILES"]).strip()
        ik = compute_inchikey14(smiles)
        if ik:
            mapping[compound] = ik
        else:
            logger.warning("无法从 SMILES 计算 InChIKey: %s", compound)
    return mapping


def map_compounds_to_chebi(
    compounds_df: pd.DataFrame,
    chebi_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """将项目化合物映射到 ChEBI 记录,返回 {compound: record} 与未匹配集合."""
    project_ik = build_project_inchikeys(compounds_df)
    project_names = set(compounds_df["compound"].astype(str).str.strip())

    name_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in project_names}
    ik_candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in project_names}

    for rec in chebi_records:
        chebi_name = rec["chebi_name"]
        ik14 = rec["inchikey14"]
        for compound in project_names:
            score = match_score(compound, chebi_name)
            if score:
                # 精确名称为 3, 包含匹配为 1
                priority = 3 if score == 2 else 1
                name_candidates[compound].append({**rec, "name_score": score, "match_priority": priority})
            if ik14 and project_ik.get(compound) == ik14:
                ik_candidates[compound].append({**rec, "match_priority": 2})

    chosen: dict[str, dict[str, Any]] = {}
    for compound in project_names:
        candidates = name_candidates[compound] + ik_candidates[compound]
        if not candidates:
            continue
        # 优先:名称精确匹配(3) > InChIKey(2) > 名称包含(1); 然后高 star; 然后 ID 稳定排序
        def sort_key(item: dict[str, Any]) -> tuple[int, int, str]:
            priority = item.get("match_priority", 0)
            star = item.get("chebi_star") or 0
            return (-priority, -star, str(item.get("chebi_id", "")))

        candidates = sorted(candidates, key=sort_key)
        best = candidates[0]
        priority = best.get("match_priority", 0)
        if priority == 3:
            method = "name_exact"
        elif best.get("name_score", 0) == 1:
            method = "name_similar"
        elif priority == 2:
            method = "inchikey"
        else:
            method = "unknown"
        chosen[compound] = {**best, "match_method": method}

    unmatched = project_names - set(chosen.keys())
    return chosen, unmatched


def main() -> int:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"输入文件不存在: {INPUT_CSV}")

    compounds_df = load_compounds(INPUT_CSV)
    logger.info("项目化合物: %d 个", len(compounds_df))

    session = requests.Session()
    sdf_bytes = download_sdf(CHEBI_SDF_URL, session)
    chebi_records = parse_chebi_records(sdf_bytes)

    if not chebi_records:
        logger.warning("ChEBI SDF 无有效记录, 输出空文件")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "compound",
                "chebi_id",
                "chebi_name",
                "chebi_star",
                "smiles",
                "inchikey",
                "formula",
                "exact_mass",
                "molecular_weight",
                "charge",
                "match_method",
                "source",
                "download_date",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        return 0

    chosen, unmatched = map_compounds_to_chebi(compounds_df, chebi_records)
    if unmatched:
        logger.warning("未匹配到 ChEBI 的化合物: %s", sorted(unmatched))

    rows: list[dict[str, Any]] = []
    for _, row in compounds_df.iterrows():
        compound = str(row["compound"]).strip()
        rec = chosen.get(compound)
        if rec is None:
            rows.append(
                {
                    "compound": compound,
                    "chebi_id": None,
                    "chebi_name": None,
                    "chebi_star": None,
                    "smiles": None,
                    "inchikey": None,
                    "formula": None,
                    "exact_mass": None,
                    "molecular_weight": None,
                    "charge": None,
                    "match_method": None,
                    "source": "ChEBI",
                    "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                }
            )
            continue
        rows.append(
            {
                "compound": compound,
                "chebi_id": rec["chebi_id"],
                "chebi_name": rec["chebi_name"],
                "chebi_star": rec["chebi_star"],
                "smiles": rec["smiles"],
                "inchikey": rec["inchikey"],
                "formula": rec["formula"],
                "exact_mass": rec["exact_mass"],
                "molecular_weight": rec["molecular_weight"],
                "charge": rec["charge"],
                "match_method": rec["match_method"],
                "source": "ChEBI",
                "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("compound").reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("已保存 %d 条 ChEBI 化合物注释到 %s", len(out_df), OUTPUT_CSV)

    metadata = {
        "source": "ChEBI",
        "subsource": "chebi_lite_3_stars.sdf.gz",
        "url": CHEBI_SDF_URL,
        "download_date": pd.Timestamp.now().isoformat(),
        "n_project_compounds": int(len(compounds_df)),
        "n_chebi_records_parsed": len(chebi_records),
        "n_matched": int(out_df["chebi_id"].notna().sum()),
        "n_unmatched": int(out_df["chebi_id"].isna().sum()),
        "unmatched_compounds": sorted(unmatched),
        "citation": (
            "Hastings J, Owen G, Dekker A, et al. ChEBI in 2016: Improved services and an "
            "expanding collection of metabolites. Nucleic Acids Res. 2016;44(D1):D1204-D1210."
        ),
    }
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已保存元数据: %s", METADATA_JSON)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
