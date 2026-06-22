"""从 dhimmel/drugbank GitHub 镜像获取化合物-靶点关系。

DrugBank 官方完整数据需注册许可。本脚本使用 Daniel Himmelstein 维护的
公开预处理镜像 (https://github.com/dhimmel/drugbank, gh-pages 分支),
该镜像基于 DrugBank XML 提取了 drugbank.tsv 与 proteins.tsv。

参考:
- dhimmel/drugbank: https://github.com/dhimmel/drugbank
- DrugBank: https://go.drugbank.com/
- Wishart DS, et al. Nucleic Acids Res. 2018;46(D1):D1074-D1082.

输入:
    network_files/compound_smiles.csv
    铁衰老基因.txt

输出:
    network_files/drugbank_compound_targets.csv
    external_data/drugbank_download_metadata.json
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
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
OUTPUT_CSV = NETWORK_DIR / "drugbank_compound_targets.csv"
METADATA_JSON = EXTERNAL_DIR / "drugbank_download_metadata.json"
GENE_LIST_PATH = BASE_DIR / "铁衰老基因.txt"

DRUGBANK_REPO_OWNER = "dhimmel"
DRUGBANK_REPO = "drugbank"
DRUGBANK_REPO_REF = "gh-pages"
DRUGBANK_DRUGS_PATH = "data/drugbank.tsv"
DRUGBANK_PROTEINS_PATH = "data/proteins.tsv"
MYGENE_URL = "https://mygene.info/v3/query"


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


def _download_github_file_bytes(owner: str, repo: str, path: str, ref: str | None = None) -> bytes:
    """通过 GitHub Contents API 下载文件并返回原始 bytes.

    用于替代 raw.githubusercontent.com, 后者在中国大陆网络环境下
    经常出现连接重置/超时(错误 10054)。GitHub Contents API 通过
    api.github.com 返回 base64 编码内容, 稳定性更好。
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "User-Agent": "fetch_drugbank_compound_targets.py",
        "Accept": "application/vnd.github.v3+json",
    }
    params = {"ref": ref} if ref else {}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            if "content" not in payload:
                raise ValueError(f"GitHub API 响应缺少 content: {list(payload.keys())}")
            return base64.b64decode(payload["content"])
        except Exception:
            traceback.print_exc()
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def load_compound_names(csv_path: Path) -> dict[str, str]:
    """读取项目化合物名称,返回 {normalized_name: original_name}."""
    df = pd.read_csv(csv_path)
    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        if name:
            mapping[normalize_name(name)] = name
    return mapping


def download_tsv(path: str, session: requests.Session) -> pd.DataFrame:
    """通过 GitHub Contents API 下载 TSV 并返回 DataFrame."""
    logger.info("通过 GitHub Contents API 下载 %s", path)
    raw_bytes = _download_github_file_bytes(
        DRUGBANK_REPO_OWNER, DRUGBANK_REPO, path, ref=DRUGBANK_REPO_REF
    )
    return pd.read_csv(StringIO(raw_bytes.decode("utf-8")), sep="\t", low_memory=False)


def map_entrez_to_symbols(
    entrez_ids: set[str],
    session: requests.Session,
) -> dict[str, str]:
    """通过 mygene.info 批量将 Entrez Gene ID 映射到 HGNC 基因符号."""
    mapping: dict[str, str] = {}
    ids = sorted(entrez_ids)
    if not ids:
        return mapping

    # 过滤非数字 ID
    numeric_ids = [e for e in ids if str(e).isdigit()]
    logger.info("mygene.info 映射 %d 个 Entrez ID", len(numeric_ids))

    batch_size = 1000
    for i in range(0, len(numeric_ids), batch_size):
        batch = numeric_ids[i : i + batch_size]
        try:
            resp = session.post(
                MYGENE_URL,
                data={
                    "q": ",".join(batch),
                    "scopes": "entrezgene",
                    "species": "human",
                    "fields": "symbol",
                },
                timeout=60,
            )
            resp.raise_for_status()
            for item in resp.json():
                eid = str(item.get("query", "")).strip()
                symbol = str(item.get("symbol", "")).strip()
                if symbol and eid:
                    mapping[eid] = symbol.upper()
        except Exception:
            logger.warning("mygene.info 批量映射失败 (batch %d-%d)", i, i + batch_size)
            traceback.print_exc()

    logger.info("成功映射 %d/%d 个 Entrez ID 到基因符号", len(mapping), len(numeric_ids))
    return mapping


def build_compound_mapping(drugs_df: pd.DataFrame, project_compounds: dict[str, str]) -> dict[str, str]:
    """将 DrugBank 药物名称映射到项目化合物名称."""
    drugbank_to_project: dict[str, str] = {}
    for _, row in drugs_df.iterrows():
        db_id = str(row.get("drugbank_id", "")).strip()
        name = str(row.get("name", "")).strip()
        if not db_id or not name:
            continue
        norm = normalize_name(name)
        if norm in project_compounds:
            drugbank_to_project[db_id] = project_compounds[norm]
            continue
        # 尝试同义词字段(若存在)
        synonyms = str(row.get("synonyms", "")).strip()
        if synonyms:
            for syn in re.split(r"[|;]", synonyms):
                syn_norm = normalize_name(syn)
                if syn_norm in project_compounds:
                    drugbank_to_project[db_id] = project_compounds[syn_norm]
                    break
    return drugbank_to_project


def confidence_score(known_action: str, category: str) -> float:
    """根据 known_action 与 category 计算置信度."""
    action = str(known_action).strip().lower()
    cat = str(category).strip().lower()

    base = 0.65
    if action == "yes":
        base = 0.85
    elif action == "no":
        base = 0.50

    if cat == "target":
        return base
    if cat in {"enzyme", "transporter", "carrier"}:
        return base * 0.85
    return base * 0.75


def confidence_level(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def main() -> int:
    core_genes = load_gene_set(GENE_LIST_PATH)
    logger.info("核心铁衰老基因集: %d 个", len(core_genes))

    project_compounds = load_compound_names(INPUT_CSV)
    logger.info("项目化合物: %d 个", len(project_compounds))

    session = requests.Session()

    drugs_df = download_tsv(DRUGBANK_DRUGS_PATH, session)
    proteins_df = download_tsv(DRUGBANK_PROTEINS_PATH, session)
    logger.info(
        "DrugBank 药物: %d, 蛋白记录: %d", len(drugs_df), len(proteins_df)
    )

    drugbank_to_project = build_compound_mapping(drugs_df, project_compounds)
    logger.info("DrugBank 药物匹配到项目化合物: %d 个", len(drugbank_to_project))

    # 仅保留匹配到项目化合物的蛋白记录
    matched = proteins_df[
        proteins_df["drugbank_id"].isin(set(drugbank_to_project.keys()))
    ].copy()
    logger.info("匹配到项目化合物的蛋白记录: %d 条", len(matched))

    if matched.empty:
        logger.warning("未找到任何 DrugBank 匹配记录")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "drugbank_id",
                "category",
                "known_action",
                "actions",
                "confidence",
                "confidence_level",
                "source",
                "download_date",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        return 0

    # 仅保留人类蛋白
    matched = matched[matched["organism"].astype(str).str.lower() == "human"]
    logger.info("人类蛋白记录: %d 条", len(matched))

    # 提取唯一 Entrez ID 并映射到基因符号
    entrez_ids = set(matched["entrez_gene_id"].dropna().astype(int).astype(str))
    entrez_to_symbol = map_entrez_to_symbols(entrez_ids, session)

    records: list[dict[str, Any]] = []
    for _, row in matched.iterrows():
        db_id = str(row.get("drugbank_id", "")).strip()
        compound = drugbank_to_project.get(db_id)
        if not compound:
            continue

        eid = str(int(row["entrez_gene_id"])) if pd.notna(row["entrez_gene_id"]) else ""
        gene = entrez_to_symbol.get(eid)
        if not gene or gene not in core_genes:
            continue

        category = str(row.get("category", "")).strip()
        known_action = str(row.get("known_action", "")).strip()
        actions = str(row.get("actions", "")).strip()
        if actions.lower() == "nan":
            actions = ""
        conf = confidence_score(known_action, category)

        records.append(
            {
                "compound": compound,
                "gene": gene,
                "drugbank_id": db_id,
                "category": category,
                "known_action": known_action,
                "actions": actions,
                "confidence": round(conf, 3),
                "confidence_level": confidence_level(conf),
                "source": "DrugBank_dhimmel_mirror",
                "download_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            }
        )

    if not records:
        logger.warning("过滤后无有效 compound-target 记录")
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "drugbank_id",
                "category",
                "known_action",
                "actions",
                "confidence",
                "confidence_level",
                "source",
                "download_date",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        return 0

    df = pd.DataFrame(records)
    df = df.sort_values(["compound", "gene", "confidence"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["compound", "gene"], keep="first")
    df = df.reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("已保存 %d 条 DrugBank compound-target 边到 %s", len(df), OUTPUT_CSV)

    metadata = {
        "source": "DrugBank (via dhimmel/drugbank GitHub mirror)",
        "original_source": "DrugBank",
        "mirror_url": "https://github.com/dhimmel/drugbank",
        "download_date": pd.Timestamp.now().isoformat(),
        "n_drugbank_drugs": int(len(drugs_df)),
        "n_drugbank_proteins": int(len(proteins_df)),
        "matched_compounds": int(len(drugbank_to_project)),
        "n_records_raw": int(len(matched)),
        "n_records_filtered": int(len(df)),
        "unique_compounds": int(df["compound"].nunique()),
        "unique_genes": int(df["gene"].nunique()),
        "citation": (
            "Wishart DS, Feunang YD, Guo AC, et al. DrugBank 5.0: a major update to the "
            "DrugBank database for 2018. Nucleic Acids Res. 2018;46(D1):D1074-D1082."
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
