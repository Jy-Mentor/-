"""从 DrugTargetCommons (DTC) 下载化合物-靶点 bioactivity 数据。

DrugTargetCommons 官方完整数据下载地址:
    https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv

参考:
- DrugTargetCommons: https://drugtargetcommons.fimm.fi/
- Tanoli Z, Alam Z, Vähä-Koskela M, et al. Drug Target Commons 2.0:
  a community platform for systematic analysis of drug–target interaction profiles.
  Database (Oxford). 2018;2018:bay083. doi:10.1093/database/bay083
- DeepPurpose dataset.py 也使用该 URL 作为 DTC 数据源。

输入:
    network_files/compound_smiles.csv
    铁衰老基因.txt

输出:
    network_files/drugtargetcommons_compound_targets.csv
    external_data/drugtargetcommons_download_metadata.json

注意:
    若 DTC 服务器暂时不可达, 脚本会写出空 schema 文件并记录失败原因,
    不会伪造数据。待网络恢复后可直接重新运行本脚本。
"""

from __future__ import annotations

import json
import logging
import re
import traceback
from io import StringIO
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
OUTPUT_CSV = NETWORK_DIR / "drugtargetcommons_compound_targets.csv"
METADATA_JSON = EXTERNAL_DIR / "drugtargetcommons_download_metadata.json"
GENE_LIST_PATH = BASE_DIR / "铁衰老基因.txt"

DTC_URL = "https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv"
MYGENE_URL = "https://mygene.info/v3/query"

# 亲和力阈值: 仅保留 <= 100 μM 的活性记录
MAX_VALUE_NM = 100_000.0
# 仅读取前 N 行做列名探测, 避免全表加载
SCHEMA_PREVIEW_ROWS = 5


def load_gene_set(path: Path) -> set[str]:
    """加载核心铁衰老基因集."""
    genes: set[str] = set()
    if not path.exists():
        raise FileNotFoundError(f"核心基因集文件不存在: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


def normalize_name(name: str) -> str:
    """标准化化合物名称用于匹配."""
    s = str(name).lower()
    s = re.sub(r"[-_/\s]+", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def load_compound_names(csv_path: Path) -> dict[str, str]:
    """读取项目化合物名称,返回 {normalized_name: original_name}."""
    df = pd.read_csv(csv_path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        if name:
            mapping[normalize_name(name)] = name
    return mapping


def load_compound_inchikeys(csv_path: Path) -> dict[str, str]:
    """读取化合物 InChIKey 前 14 位,返回 {ik14: original_name}."""
    df = pd.read_csv(csv_path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        smiles = str(row.get("CanonicalSMILES", "")).strip()
        if not name or not smiles:
            continue
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            ik = Chem.MolToInchiKey(mol)[:14]
            mapping[ik] = name
        except Exception:
            logger.debug("无法为 %s 生成 InChIKey", name)
    return mapping


def detect_columns(df: pd.DataFrame) -> dict[str, str]:
    """根据实际列名探测关键字段,返回 {逻辑名: 实际列名}."""
    cols = {c.lower(): c for c in df.columns}
    mapping: dict[str, str] = {}

    # 化合物名称
    for key in ["compound_name", "drug_name", "molecule_name", "name", "compound"]:
        if key in cols:
            mapping["compound"] = cols[key]
            break

    # 化合物 InChIKey
    for key in ["inchi_key", "inchikey", "standard_inchi_key"]:
        if key in cols:
            mapping["inchikey"] = cols[key]
            break

    # SMILES
    for key in ["smiles", "canonical_smiles", "compound_smiles"]:
        if key in cols:
            mapping["smiles"] = cols[key]
            break

    # 靶点基因符号
    for key in ["gene_names", "gene_symbol", "gene", "target_gene"]:
        if key in cols:
            mapping["gene"] = cols[key]
            break

    # 靶点名称(用于 mygene 映射)
    for key in ["target_name", "target", "protein_name", "protein"]:
        if key in cols:
            mapping["target_name"] = cols[key]
            break

    # 活性类型
    for key in ["standard_type", "activity_type", "assay_type", "type"]:
        if key in cols:
            mapping["activity_type"] = cols[key]
            break

    # 活性数值
    for key in ["standard_value", "activity_value", "value", "affinity"]:
        if key in cols:
            mapping["activity_value"] = cols[key]
            break

    # 单位
    for key in ["standard_units", "units", "unit"]:
        if key in cols:
            mapping["units"] = cols[key]
            break

    # 活性注释(active/inactive)
    for key in ["activity_comment", "activity", "comment", "activity_flag"]:
        if key in cols:
            mapping["activity_comment"] = cols[key]
            break

    return mapping


def parse_numeric(value: Any) -> float | None:
    """从字符串中提取数值."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def normalize_units(units: str) -> str:
    """统一单位为 nM."""
    u = str(units).strip().lower()
    if u in {"nm", "nanomolar", "n"}:
        return "nM"
    if u in {"um", "micromolar", "μm", "μ m", "u m"}:
        return "uM"
    if u in {"mm", "millimolar"}:
        return "mM"
    return u


def to_nm(value: float, units: str) -> float | None:
    """将数值转换为 nM."""
    u = normalize_units(units)
    if u == "nM":
        return value
    if u == "uM":
        return value * 1_000.0
    if u == "mM":
        return value * 1_000_000.0
    return None


def confidence_from_activity(value_nm: float | None, activity_type: str, comment: str) -> float:
    """根据活性数值/类型/注释计算置信度."""
    act = str(activity_type).strip().upper()
    cmt = str(comment).strip().lower()

    if cmt in {"inactive", "not active", "no activity"}:
        return 0.20

    if value_nm is None or value_nm <= 0:
        # 无数值但注释为 active 时给中等置信度
        if cmt in {"active", "active inhibitor", "active agonist"}:
            return 0.55
        return 0.30

    if act in {"KI", "KD", "IC50", "EC50"}:
        if value_nm <= 10.0:
            return 0.90
        if value_nm <= 100.0:
            return 0.80
        if value_nm <= 1_000.0:
            return 0.65
        if value_nm <= 10_000.0:
            return 0.50
        if value_nm <= 100_000.0:
            return 0.35
        return 0.25

    # 其他活性类型保守处理
    if value_nm <= 10_000.0:
        return 0.50
    return 0.35


def confidence_level(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def map_target_to_gene(target_name: str, session: requests.Session, cache: dict[str, str | None]) -> str | None:
    """通过 mygene.info 将靶点名称解析为 HGNC 基因符号."""
    if target_name in cache:
        return cache[target_name]

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
        hits = resp.json().get("hits", [])
        if hits:
            symbol = str(hits[0].get("symbol", "")).strip()
            if symbol:
                cache[target_name] = symbol.upper()
                return cache[target_name]
    except Exception:
        logger.debug("mygene 映射失败: %s", target_name)
        traceback.print_exc()

    cache[target_name] = None
    return None


def parse_gene_list(value: Any) -> list[str]:
    """从可能包含多个基因符号的字段中提取基因符号."""
    if value is None or pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    # 常见分隔符: 分号、逗号、竖线、空格
    genes = re.split(r"[;|,|\s]+", s)
    return [g.strip().upper() for g in genes if g.strip()]


def download_dtc_data(url: str, session: requests.Session, timeout: int = 120) -> pd.DataFrame:
    """流式下载 DTC CSV 并返回 DataFrame."""
    logger.info("开始下载 DTC 数据: %s", url)
    with session.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            chunks.append(chunk)
            total += len(chunk)
            if total % (50 * 1024 * 1024) == 0:
                logger.info("已下载 %.1f MB", total / (1024 * 1024))
        logger.info("下载完成, 总大小 %.1f MB", total / (1024 * 1024))
        content = b"".join(chunks).decode("utf-8", errors="replace")
        return pd.read_csv(StringIO(content), low_memory=False)


def process_dtc(
    df: pd.DataFrame,
    project_compounds: dict[str, str],
    project_inchikeys: dict[str, str],
    core_genes: set[str],
    session: requests.Session,
) -> pd.DataFrame:
    """解析 DTC DataFrame, 过滤项目化合物与核心基因."""
    cols = detect_columns(df)
    logger.info("探测到的列映射: %s", cols)

    required = {"compound"}
    missing = required - set(cols.keys())
    if missing:
        raise ValueError(f"DTC 文件缺少关键列,无法定位化合物: {missing}")

    if "gene" not in cols and "target_name" not in cols:
        raise ValueError("DTC 文件缺少基因/靶点列,无法定位靶点")

    # 化合物匹配
    compound_col = cols["compound"]
    df["_norm_compound"] = df[compound_col].astype(str).str.strip().apply(normalize_name)
    df["_project_compound"] = df["_norm_compound"].map(project_compounds)

    # 若存在 InChIKey, 用结构匹配补充
    if "inchikey" in cols:
        df["_ik14"] = df[cols["inchikey"]].astype(str).str.strip().str.upper().str[:14]
        ik_match = df["_ik14"].map(project_inchikeys)
        df["_project_compound"] = df["_project_compound"].fillna(ik_match)

    matched = df[df["_project_compound"].notna()].copy()
    logger.info("化合物匹配到项目: %d / %d", len(matched), len(df))

    if matched.empty:
        return pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "activity_type",
                "activity_value",
                "activity_value_nm",
                "units",
                "activity_comment",
                "confidence",
                "confidence_level",
                "source",
                "download_date",
            ]
        )

    records: list[dict[str, Any]] = []
    target_cache: dict[str, str | None] = {}

    gene_col = cols.get("gene")
    target_name_col = cols.get("target_name")
    act_type_col = cols.get("activity_type")
    act_value_col = cols.get("activity_value")
    units_col = cols.get("units")
    comment_col = cols.get("activity_comment")

    for _, row in matched.iterrows():
        compound = str(row["_project_compound"])

        genes: set[str] = set()
        if gene_col:
            genes.update(parse_gene_list(row[gene_col]))
        if target_name_col and not genes:
            gene = map_target_to_gene(str(row[target_name_col]), session, target_cache)
            if gene:
                genes.add(gene)

        genes = {g for g in genes if g in core_genes}
        if not genes:
            continue

        act_type = str(row[act_type_col]).strip() if act_type_col else ""
        value = parse_numeric(row[act_value_col]) if act_value_col else None
        units = str(row[units_col]).strip() if units_col else ""
        comment = str(row[comment_col]).strip() if comment_col else ""

        value_nm = to_nm(value, units) if value is not None else None
        if value_nm is not None and value_nm > MAX_VALUE_NM:
            continue

        conf = confidence_from_activity(value_nm, act_type, comment)

        for gene in genes:
            records.append(
                {
                    "compound": compound,
                    "gene": gene,
                    "activity_type": act_type,
                    "activity_value": value,
                    "activity_value_nm": value_nm,
                    "units": units,
                    "activity_comment": comment,
                    "confidence": round(conf, 3),
                    "confidence_level": confidence_level(conf),
                    "source": "DrugTargetCommons",
                    "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "activity_type",
                "activity_value",
                "activity_value_nm",
                "units",
                "activity_comment",
                "confidence",
                "confidence_level",
                "source",
                "download_date",
            ]
        )

    out = pd.DataFrame(records)
    # 同一 compound-gene 保留最高置信度
    out = out.sort_values("confidence", ascending=False)
    out = out.drop_duplicates(subset=["compound", "gene"], keep="first")
    out = out.sort_values(["compound", "gene"]).reset_index(drop=True)
    return out


def write_empty_output() -> None:
    """写出空 schema 文件."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "compound",
            "gene",
            "activity_type",
            "activity_value",
            "activity_value_nm",
            "units",
            "activity_comment",
            "confidence",
            "confidence_level",
            "source",
            "download_date",
        ]
    ).to_csv(OUTPUT_CSV, index=False)
    logger.warning("已写出空 schema 文件: %s", OUTPUT_CSV)


def main() -> int:
    core_genes = load_gene_set(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    project_compounds = load_compound_names(INPUT_CSV)
    logger.info("项目化合物: %d 个", len(project_compounds))

    project_inchikeys = load_compound_inchikeys(INPUT_CSV)
    logger.info("项目化合物 InChIKey(ik14): %d 个", len(project_inchikeys))

    session = requests.Session()
    try:
        df = download_dtc_data(DTC_URL, session)
        logger.info("DTC 原始记录数: %d", len(df))
    except requests.exceptions.ConnectTimeout:
        logger.error("DTC 服务器连接超时 (%s), 当前网络无法访问。未生成数据。", DTC_URL)
        traceback.print_exc()
        write_empty_output()
        _write_metadata(0, connect_timeout=True)
        return 0
    except requests.exceptions.ConnectionError:
        logger.error("DTC 服务器连接失败 (%s), 当前网络无法访问。未生成数据。", DTC_URL)
        traceback.print_exc()
        write_empty_output()
        _write_metadata(0, connection_error=True)
        return 0
    except Exception:
        logger.error("下载或解析 DTC 数据时发生异常。未生成数据。")
        traceback.print_exc()
        write_empty_output()
        _write_metadata(0, other_error=True)
        return 0

    try:
        out = process_dtc(df, project_compounds, project_inchikeys, core_genes, session)
    except Exception:
        logger.error("处理 DTC 数据时发生异常。未生成数据。")
        traceback.print_exc()
        write_empty_output()
        _write_metadata(len(df), parse_error=True)
        return 0

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    logger.info("已保存 %d 条 DTC compound-target 边到 %s", len(out), OUTPUT_CSV)

    _write_metadata(
        len(df),
        n_filtered=len(out),
        unique_compounds=int(out["compound"].nunique()) if not out.empty else 0,
        unique_genes=int(out["gene"].nunique()) if not out.empty else 0,
    )
    return 0


def _write_metadata(
    n_raw: int,
    n_filtered: int = 0,
    unique_compounds: int = 0,
    unique_genes: int = 0,
    connect_timeout: bool = False,
    connection_error: bool = False,
    other_error: bool = False,
    parse_error: bool = False,
) -> None:
    metadata: dict[str, Any] = {
        "source": "DrugTargetCommons",
        "url": DTC_URL,
        "download_date": pd.Timestamp.now().isoformat(),
        "n_records_raw": n_raw,
        "n_records_filtered": n_filtered,
        "unique_compounds": unique_compounds,
        "unique_genes": unique_genes,
        "connect_timeout": connect_timeout,
        "connection_error": connection_error,
        "other_error": other_error,
        "parse_error": parse_error,
        "citation": (
            "Tanoli Z, Alam Z, Vähä-Koskela M, et al. Drug Target Commons 2.0: "
            "a community platform for systematic analysis of drug-target interaction profiles. "
            "Database (Oxford). 2018;2018:bay083."
        ),
    }
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已保存元数据: %s", METADATA_JSON)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
