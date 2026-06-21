"""下载并清洗 CTD (Comparative Toxicogenomics Database) chemical-gene 交互数据.

参考:
- CTD 数据下载: http://ctdbase.org/downloads/
- CTD chemical-gene interactions TSV:
  http://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz

输入:
    network_files/compound_smiles.csv
    铁衰老基因.txt

输出:
    network_files/ctd_compound_targets.csv
    external_data/ctd/ctd_download_metadata.json

规则:
1. 仅保留人类 (OrganismID == 9606) 记录.
2. 仅保留 gene 在 98 铁衰老核心基因集内的记录.
3. 仅保留项目化合物列表中化合物的交互 (通过名称/同义词匹配).
4. 按 interaction action 分配置信度, 同 compound-gene 保留最高置信度.
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
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "network_files" / "compound_smiles.csv"
GENE_LIST_PATH = BASE_DIR / "铁衰老基因.txt"
OUTPUT_CSV = BASE_DIR / "network_files" / "ctd_compound_targets.csv"
METADATA_JSON = BASE_DIR / "external_data" / "ctd" / "ctd_download_metadata.json"
CTD_URL = "http://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz"

# 化合物名称到 CTD 常见名称/同义词的映射
# key 为项目内部名称, value 为用于匹配 CTD ChemicalName 的候选名
COMPOUND_SYNONYMS: Dict[str, List[str]] = {
    "BCP": ["beta-caryophyllene", "caryophyllene", "(-)-beta-caryophyllene"],
    "VC": ["ascorbic acid", "vitamin c", "l-ascorbic acid"],
    "Fer-1": ["ferrostatin-1", "ferrostatin 1"],
    "DFO": ["deferoxamine", "desferrioxamine"],
    "Lip-1": ["liproxstatin-1", "liproxstatin 1"],
    "Cinnamic_acid": ["cinnamic acid", "trans-cinnamic acid"],
}

# 需要跳过的无意义 interaction actions
SKIP_ACTION_SUBSTRINGS = (
    "no interaction",
    "not specified",
)


def load_compound_names(path: Path) -> Dict[str, str]:
    """读取项目化合物列表, 建立 'ctd_lower_name' -> 'project_name' 映射."""
    df = pd.read_csv(path)
    name_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        project_name = str(row.get("compound", "")).strip()
        if not project_name:
            continue
        # 项目名本身
        name_map[project_name.lower().replace("_", " ")] = project_name
        # 同义词
        for synonym in COMPOUND_SYNONYMS.get(project_name, []):
            name_map[synonym.lower()] = project_name
    return name_map


def load_gene_set(path: Path) -> Set[str]:
    """读取铁衰老核心基因集."""
    genes: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


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


def parse_action(action_str: str) -> List[str]:
    """CTD InteractionActions 字段形如 'increases^expression|decreases^phosphorylation'."""
    if pd.isna(action_str):
        return []
    actions = []
    for part in str(action_str).split("|"):
        part = part.strip().lower()
        if not part or any(skip in part for skip in SKIP_ACTION_SUBSTRINGS):
            continue
        actions.append(part)
    return actions


def assign_confidence(action: str) -> Tuple[float, str]:
    """根据 action 类型分配置信度."""
    direct_modulators = ("agonist", "antagonist", "inhibitor", "activator", "binder")
    direct_actions = (
        "binding",
        "activity",
        "phosphorylation",
        "ubiquitination",
        "acetylation",
        "methylation",
        "glycosylation",
        "sumoylation",
        "hydroxylation",
        "oxidation",
        "cleavage",
        "folding",
        "secretion",
        "stability",
        "localization",
        "splicing",
    )
    expression_actions = ("expression",)

    a = action.lower()
    # 直接结合/活性调控最高
    if any(term in a for term in direct_modulators) or any(term in a for term in direct_actions):
        return 0.85, "high"
    # 表达调控中等
    if any(term in a for term in expression_actions):
        return 0.65, "medium"
    # 其他有记录但证据较弱
    return 0.45, "low"


def _row_to_records(row, name_map: Dict[str, str], core_genes: Set[str]) -> List[Dict]:
    """将一条 CTD 记录转换为零条或多条项目边记录."""
    chemical_name = str(row.get("ChemicalName", "")).strip()
    gene = str(row.get("GeneSymbol", "")).strip().upper()
    organism_id = str(row.get("OrganismID", "")).strip()
    interaction = str(row.get("Interaction", "")).strip()
    actions_raw = str(row.get("InteractionActions", "")).strip()

    if not chemical_name or not gene:
        return []
    if organism_id != "9606":
        return []
    if gene not in core_genes:
        return []

    key = chemical_name.lower()
    project_name = name_map.get(key)
    if project_name is None:
        return []

    actions = parse_action(actions_raw)
    if not actions:
        # 无具体 action 时仍保留一条记录, 但置信度低
        actions = ["unspecified"]

    records = []
    seen_actions = set()
    for action in actions:
        if action in seen_actions:
            continue
        seen_actions.add(action)
        conf, level = assign_confidence(action)
        records.append(
            {
                "compound": project_name,
                "gene": gene,
                "interaction": interaction,
                "action": action,
                "source": "CTD",
                "confidence": round(conf, 4),
                "confidence_level": level,
                "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            }
        )
    return records


CTD_COLUMNS = [
    "ChemicalName",
    "ChemicalID",
    "CasRN",
    "GeneSymbol",
    "GeneID",
    "GeneForms",
    "Organism",
    "OrganismID",
    "Interaction",
    "InteractionActions",
    "PubMedIDs",
]


def process_ctd(
    tsv_path: Path,
    name_map: Dict[str, str],
    core_genes: Set[str],
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """分块读取 CTD TSV, 过滤并转换为项目边记录."""
    logger.info("开始处理 CTD 文件: %s", tsv_path)
    total_matched = 0
    all_records: List[Dict] = []

    # 预编译可匹配名称集合, 用于 pandas 快速过滤
    ctd_names = set(name_map.keys())

    for i, chunk in enumerate(
        pd.read_csv(
            tsv_path,
            sep="\t",
            comment="#",
            header=None,
            names=CTD_COLUMNS,
            low_memory=False,
            chunksize=chunksize,
            dtype=str,
            keep_default_na=False,
        )
    ):
        # 快速过滤: 人类、核心基因、候选化合物名
        if "OrganismID" in chunk.columns:
            chunk = chunk[chunk["OrganismID"].astype(str).str.strip() == "9606"]
        if "GeneSymbol" in chunk.columns:
            chunk = chunk[chunk["GeneSymbol"].str.upper().isin(core_genes)]
        if "ChemicalName" in chunk.columns:
            chunk = chunk[chunk["ChemicalName"].str.strip().str.lower().isin(ctd_names)]

        if chunk.empty:
            if (i + 1) % 5 == 0:
                logger.info("已处理 %d 个 chunk, 尚未命中", i + 1)
            continue

        for _, row in chunk.iterrows():
            records = _row_to_records(row, name_map, core_genes)
            if records:
                all_records.extend(records)
                total_matched += 1

        if (i + 1) % 5 == 0:
            logger.info("已处理 %d 个 chunk, 当前命中记录 %d 条", i + 1, total_matched)

    logger.info("CTD 原始命中记录数 (去重前): %d", total_matched)

    if not all_records:
        return pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "interaction",
                "action",
                "source",
                "confidence",
                "confidence_level",
                "download_date",
            ]
        )

    df = pd.DataFrame(all_records)
    # 同 compound-gene 保留最高置信度; 如并列保留 action 按字母序第一条
    df = df.sort_values(["compound", "gene", "confidence", "action"], ascending=[True, True, False, True])
    df = df.drop_duplicates(subset=["compound", "gene"], keep="first")
    df = df.sort_values(["compound", "gene"]).reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, metadata: dict, output_csv: Path, metadata_json: Path) -> None:
    """写入 CSV 与元数据."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("已写入 CTD 清洗后边文件: %s (%d 条)", output_csv, len(df))

    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已写入元数据: %s", metadata_json)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and clean CTD chemical-gene interactions")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--genes", type=Path, default=GENE_LIST_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--metadata", type=Path, default=METADATA_JSON)
    parser.add_argument("--url", type=str, default=CTD_URL)
    parser.add_argument("--skip-download", action="store_true", help="如果已存在本地 TSV 则跳过下载")
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("输入化合物文件不存在: %s", args.input)
        return 1
    if not args.genes.exists():
        logger.error("输入基因列表不存在: %s", args.genes)
        return 1

    name_map = load_compound_names(args.input)
    core_genes = load_gene_set(args.genes)
    logger.info("项目化合物: %d 个, 核心基因: %d 个", len(name_map), len(core_genes))

    raw_gz = args.metadata.parent / "CTD_chem_gene_ixns.tsv.gz"
    raw_tsv = args.metadata.parent / "CTD_chem_gene_ixns.tsv"

    try:
        if not raw_gz.exists() or not args.skip_download:
            download_ctd_file(args.url, raw_gz)
        else:
            logger.info("本地 CTD 压缩文件已存在, 跳过下载")

        if not raw_tsv.exists():
            decompress_gz(raw_gz, raw_tsv)
        else:
            logger.info("本地 CTD TSV 已存在, 跳过解压")

        df = process_ctd(raw_tsv, name_map, core_genes)

        metadata = {
            "source": "CTD",
            "source_url": "http://ctdbase.org/",
            "download_url": args.url,
            "download_date": pd.Timestamp.now().isoformat(),
            "raw_gz": str(raw_gz),
            "raw_tsv": str(raw_tsv),
            "stats": {
                "project_compounds": len(name_map),
                "core_genes": len(core_genes),
                "filtered_edges": int(len(df)),
                "unique_compounds": int(df["compound"].nunique()) if not df.empty else 0,
                "unique_genes": int(df["gene"].nunique()) if not df.empty else 0,
            },
            "confidence_distribution": (
                df["confidence_level"].value_counts().to_dict() if not df.empty else {}
            ),
            "action_distribution": (
                df["action"].value_counts().head(20).to_dict() if not df.empty else {}
            ),
        }

        write_outputs(df, metadata, args.output, args.metadata)

    except Exception:
        logger.error("CTD 数据处理失败")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
