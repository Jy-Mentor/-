"""下载并清洗 CTD (Comparative Toxicogenomics Database) chemical-disease 关联数据.

参考:
- CTD 数据下载: http://ctdbase.org/downloads/
- CTD chemical-disease associations TSV:
  http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz

输入:
    network_files/compound_smiles.csv
    network_files/graph_node_config.yaml

输出:
    network_files/ctd_compound_disease.csv
    external_data/ctd/ctd_compound_disease_metadata.json

规则:
1. 仅保留 DirectEvidence 标记为 "marker/mechanism" 或 "therapeutic" 的记录.
2. 仅保留项目化合物列表中的化合物.
3. 仅保留 graph_node_config.yaml 中配置的 disease 节点 (CIRI/AD/Aging).
4. 同 compound-disease 保留最高置信度.
5. 记录 source/confidence/confidence_level/download_date.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "network_files" / "compound_smiles.csv"
NODE_CONFIG = BASE_DIR / "network_files" / "graph_node_config.yaml"
OUTPUT_CSV = BASE_DIR / "network_files" / "ctd_compound_disease.csv"
METADATA_JSON = BASE_DIR / "external_data" / "ctd" / "ctd_compound_disease_metadata.json"
CTD_URL = "http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"

# 疾病名称 -> 项目内部 disease 节点映射
DISEASE_NAME_MAP: Dict[str, str] = {
    "cerebral ischemia": "CIRI",
    "brain ischemia": "CIRI",
    "reperfusion injury": "CIRI",
    "cerebral infarction": "CIRI",
    "stroke": "CIRI",
    "alzheimer disease": "AD",
    "alzheimer's disease": "AD",
    "aging": "Aging",
    "cellular senescence": "Aging",
}

# 化合物名称到 CTD 常见名称/同义词的映射
COMPOUND_SYNONYMS: Dict[str, List[str]] = {
    "BCP": ["beta-caryophyllene", "caryophyllene", "(-)-beta-caryophyllene"],
    "VC": ["ascorbic acid", "vitamin c", "l-ascorbic acid"],
    "Fer-1": ["ferrostatin-1", "ferrostatin 1"],
    "DFO": ["deferoxamine", "desferrioxamine"],
    "Lip-1": ["liproxstatin-1", "liproxstatin 1"],
    "Cinnamic_acid": ["cinnamic acid", "trans-cinnamic acid"],
}


def load_compound_names(path: Path) -> Dict[str, str]:
    """读取项目化合物列表, 建立 'ctd_lower_name' -> 'project_name' 映射."""
    df = pd.read_csv(path)
    name_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        project_name = str(row.get("compound", "")).strip()
        if not project_name:
            continue
        name_map[project_name.lower().replace("_", " ")] = project_name
        for synonym in COMPOUND_SYNONYMS.get(project_name, []):
            name_map[synonym.lower()] = project_name
    return name_map


def load_disease_nodes(config_path: Path) -> Set[str]:
    """从 graph_node_config.yaml 加载疾病节点列表."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    diseases = cfg.get("diseases", [])
    return {str(d).strip() for d in diseases if str(d).strip()}


def download_ctd_file(url: str, dest: Path, chunk_size: int = 8192) -> int:
    """流式下载 CTD TSV.gz, 返回下载字节数."""
    logger.info("开始下载 CTD 数据: %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    total += len(chunk)
    logger.info("下载完成: %s (%.2f MB)", dest, total / 1024 / 1024)
    return total


def decompress_gz(src: Path, dst: Path) -> None:
    """解压 gz 文件."""
    logger.info("解压 %s -> %s", src, dst)
    with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def assign_confidence(direct_evidence: str, score: float | None) -> tuple[float, str]:
    """根据 DirectEvidence 与 score 分配置信度."""
    evidence = str(direct_evidence).strip().lower()
    if evidence == "therapeutic":
        return 0.90, "high"
    if evidence == "marker/mechanism":
        return 0.75, "high"
    if evidence:
        return 0.50, "medium"
    return 0.30, "low"


CTD_COLUMNS = [
    "ChemicalName",
    "ChemicalID",
    "CasRN",
    "DiseaseName",
    "DiseaseID",
    "DirectEvidence",
    "InferenceGeneSymbol",
    "InferenceScore",
    "OmimIDs",
    "PubMedIDs",
]


def process_ctd(
    tsv_path: Path,
    name_map: Dict[str, str],
    disease_nodes: Set[str],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """分块读取 CTD TSV, 过滤并转换为项目边记录.

    使用 pandas 向量化过滤替代逐行 Python 循环, 大幅提升大文件处理速度.
    """
    logger.info("开始处理 CTD chemical-disease 文件: %s", tsv_path)

    # 反转 name_map: project_name -> set(匹配关键词)
    project_compound_terms: Dict[str, Set[str]] = {}
    for key, project_name in name_map.items():
        project_compound_terms.setdefault(project_name, set()).add(key)

    # 反转 disease map: project_disease -> set(匹配关键词)
    project_disease_terms: Dict[str, Set[str]] = {}
    for key, project_disease in DISEASE_NAME_MAP.items():
        if project_disease in disease_nodes:
            project_disease_terms.setdefault(project_disease, set()).add(key)

    all_records: List[Dict] = []
    total_rows = 0

    for chunk in pd.read_csv(
        tsv_path,
        sep="\t",
        comment="#",
        header=None,
        names=CTD_COLUMNS,
        low_memory=False,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    ):
        total_rows += len(chunk)
        # 仅保留有直接证据的记录
        chunk = chunk[chunk["DirectEvidence"].str.strip() != ""]
        if chunk.empty:
            continue

        for project_compound, terms in project_compound_terms.items():
            comp_mask = pd.Series(False, index=chunk.index)
            for term in terms:
                comp_mask = comp_mask | chunk["ChemicalName"].str.contains(
                    term, case=False, na=False, regex=False
                )
            if not comp_mask.any():
                continue
            comp_chunk = chunk[comp_mask]

            for project_disease, dis_terms in project_disease_terms.items():
                dis_mask = pd.Series(False, index=comp_chunk.index)
                for term in dis_terms:
                    dis_mask = dis_mask | comp_chunk["DiseaseName"].str.contains(
                        term, case=False, na=False, regex=False
                    )
                if not dis_mask.any():
                    continue

                matched = comp_chunk[dis_mask]
                for _, row in matched.iterrows():
                    direct_evidence = str(row["DirectEvidence"]).strip()
                    score = None
                    try:
                        score = float(row.get("InferenceScore", ""))
                    except (TypeError, ValueError):
                        score = None
                    conf, level = assign_confidence(direct_evidence, score)
                    all_records.append(
                        {
                            "compound": project_compound,
                            "disease": project_disease,
                            "direct_evidence": direct_evidence,
                            "inference_score": score,
                            "source": "CTD",
                            "confidence": round(conf, 4),
                            "confidence_level": level,
                            "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                        }
                    )

    logger.info("CTD 总记录数: %d, 原始命中记录数 (去重前): %d", total_rows, len(all_records))

    if not all_records:
        return pd.DataFrame(
            columns=[
                "compound",
                "disease",
                "direct_evidence",
                "inference_score",
                "source",
                "confidence",
                "confidence_level",
                "download_date",
            ]
        )

    df = pd.DataFrame(all_records)
    df = df.sort_values(
        ["compound", "disease", "confidence", "direct_evidence"],
        ascending=[True, True, False, True],
    )
    df = df.drop_duplicates(subset=["compound", "disease"], keep="first")
    df = df.sort_values(["compound", "disease"]).reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, metadata: dict, output_csv: Path, metadata_json: Path) -> None:
    """写入 CSV 与元数据."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("已写入 CTD compound-disease 清洗后边文件: %s (%d 条)", output_csv, len(df))

    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已写入元数据: %s", metadata_json)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and clean CTD chemical-disease associations")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--config", type=Path, default=NODE_CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--metadata", type=Path, default=METADATA_JSON)
    parser.add_argument("--url", type=str, default=CTD_URL)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("输入化合物文件不存在: %s", args.input)
        return 1
    if not args.config.exists():
        logger.error("节点配置文件不存在: %s", args.config)
        return 1

    name_map = load_compound_names(args.input)
    disease_nodes = load_disease_nodes(args.config)
    logger.info("项目化合物: %d 个, 疾病节点: %s", len(name_map), sorted(disease_nodes))

    raw_gz = args.metadata.parent / "CTD_chemicals_diseases.tsv.gz"
    raw_tsv = args.metadata.parent / "CTD_chemicals_diseases.tsv"

    try:
        if not raw_gz.exists() or not args.skip_download:
            download_ctd_file(args.url, raw_gz)
        else:
            logger.info("本地 CTD 压缩文件已存在, 跳过下载")

        if not raw_tsv.exists():
            decompress_gz(raw_gz, raw_tsv)
        else:
            logger.info("本地 CTD TSV 已存在, 跳过解压")

        df = process_ctd(raw_tsv, name_map, disease_nodes)

        metadata = {
            "source": "CTD",
            "source_url": "http://ctdbase.org/",
            "download_url": args.url,
            "download_date": pd.Timestamp.now().isoformat(),
            "raw_gz": str(raw_gz),
            "raw_tsv": str(raw_tsv),
            "stats": {
                "project_compounds": len(name_map),
                "disease_nodes": sorted(disease_nodes),
                "filtered_edges": int(len(df)),
                "unique_compounds": int(df["compound"].nunique()) if not df.empty else 0,
                "unique_diseases": int(df["disease"].nunique()) if not df.empty else 0,
            },
            "confidence_distribution": (
                df["confidence_level"].value_counts().to_dict() if not df.empty else {}
            ),
            "direct_evidence_distribution": (
                df["direct_evidence"].value_counts().to_dict() if not df.empty else {}
            ),
        }

        write_outputs(df, metadata, args.output, args.metadata)

    except Exception:
        logger.error("CTD compound-disease 数据处理失败")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
