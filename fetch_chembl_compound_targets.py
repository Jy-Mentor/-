"""从 ChEMBL 数据库下载化合物-靶点生物活性数据。

参考:
- chembl/chembl_webresource_client: https://github.com/chembl/chembl_webresource_client
- ChEMBL Data Web Services: https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services
- Nowotka et al., Expert Opin Drug Discov 2017: https://doi.org/10.1080/17460441.2017.1339032

输入: network_files/compound_smiles.csv
输出: network_files/chembl_compound_targets.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
import warnings
from pathlib import Path

import pandas as pd

# 兼容直接运行
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _path in (_SRC_DIR, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "network_files" / "compound_smiles.csv"
OUTPUT_CSV = BASE_DIR / "network_files" / "chembl_compound_targets.csv"
METADATA_JSON = BASE_DIR / "external_data" / "chembl_download_metadata.json"

# 活性过滤阈值: pChEMBL >= 5 等价于 <= 10 μM
PChEMBL_THRESHOLD = 5.0
# 若无 pChEMBL, 对 IC50/Ki/Kd 要求 standard_value <= 10000 nM
STANDARD_VALUE_THRESHOLD_NM = 10_000.0
# 支持的 standard_type
ALLOWED_STANDARD_TYPES = {"IC50", "Ki", "Kd", "EC50", "Potency"}


def load_compounds(csv_path: Path) -> list[dict]:
    """读取化合物列表 (name, cid, CanonicalSMILES)."""
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        cid = str(row.get("cid", "")).strip()
        smiles = str(row.get("CanonicalSMILES", "")).strip()
        if name and smiles:
            records.append({"compound": name, "cid": cid, "smiles": smiles})
    return records


def resolve_molecule(client, smiles: str, name: str) -> str | None:
    """通过 SMILES 相似性搜索或名称搜索解析 ChEMBL molecule ID."""
    # 1. 尝试 canonical SMILES 精确匹配 (connectivity)
    try:
        res = client.molecule.filter(
            molecule_structures__canonical_smiles__connectivity=smiles
        ).only(["molecule_chembl_id", "pref_name"])
        items = list(res)
        if items:
            return items[0]["molecule_chembl_id"]
    except Exception:
        logger.debug(f"  {name}: SMILES connectivity search failed")
        traceback.print_exc()

    # 2. 尝试名称精确匹配
    if name:
        try:
            res = client.molecule.filter(pref_name__iexact=name.replace("_", " ")).only(
                ["molecule_chembl_id", "pref_name"]
            )
            items = list(res)
            if items:
                return items[0]["molecule_chembl_id"]
        except Exception:
            logger.debug(f"  {name}: name search failed")

    # 3. 尝试子结构/相似性搜索 (fallback, 较宽松)
    try:
        res = client.similarity.filter(smiles=smiles, similarity=95).only(
            ["molecule_chembl_id", "pref_name"]
        )
        items = list(res)
        if items:
            return items[0]["molecule_chembl_id"]
    except Exception:
        logger.debug(f"  {name}: similarity search failed")

    return None


def fetch_activities(client, molecule_chembl_id: str, max_pages: int = 100) -> list[dict]:
    """获取某分子的 bioactivity 记录,优先过滤有活性(pChEMBL>=阈值)的记录."""
    fields = [
        "activity_id",
        "target_chembl_id",
        "target_pref_name",
        "standard_type",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_chembl_id",
    ]

    activities = []

    # 策略1: 优先查询已有 pChEMBL 且 >= 阈值的记录 (数据量最小)
    try:
        res = client.activity.filter(
            molecule_chembl_id=molecule_chembl_id,
            pchembl_value__gte=PChEMBL_THRESHOLD,
        ).only(fields)
        activities.extend(list(res)[: max_pages * 20])
    except Exception as e:
        logger.debug(f"  pchembl filter query failed for {molecule_chembl_id}: {e}")

    # 策略2: 对无 pChEMBL 但 standard_value <= 阈值 (nM) 的记录做补充
    try:
        res = client.activity.filter(
            molecule_chembl_id=molecule_chembl_id,
            pchembl_value__isnull=True,
            standard_value__lte=STANDARD_VALUE_THRESHOLD_NM,
            standard_units__iexact="nM",
        ).only(fields)
        activities.extend(list(res)[: max_pages * 20])
    except Exception as e:
        logger.debug(f"  standard_value filter query failed for {molecule_chembl_id}: {e}")

    # 兜底: 若过滤查询均失败,返回少量原始记录 (防止漏检)
    if not activities:
        try:
            res = client.activity.filter(molecule_chembl_id=molecule_chembl_id).only(fields)
            activities = list(res)[: max_pages * 20]
        except Exception as e:
            logger.warning(f"  activity query failed for {molecule_chembl_id}: {e}")
            traceback.print_exc()
            return []

    return activities


def fetch_target_genes(client, target_chembl_id: str) -> list[str]:
    """将 target_chembl_id 解析为 HGNC 基因符号列表."""
    genes = set()
    try:
        target = client.target.get(target_chembl_id)
        target_type = target.get("target_type", "")

        # 单蛋白靶点
        if "SINGLE PROTEIN" in target_type.upper():
            for comp in target.get("target_components", []):
                for syn in comp.get("target_component_synonyms", []):
                    if syn.get("syn_type", "").upper() == "GENE_SYMBOL":
                        gene = syn.get("component_synonym", "").strip().upper()
                        if gene:
                            genes.add(gene)
                # component_description 有时就是蛋白名, 尝试提取大写基因符号
                desc = comp.get("component_description", "")
                if desc:
                    # 取括号内或纯大写 token 作为候选基因符号
                    for token in desc.replace(",", " ").replace("(", " ").replace(")", " ").split():
                        token = token.strip().upper()
                        if token.isalpha() and 2 <= len(token) <= 15:
                            genes.add(token)

        # 蛋白家族/复合物: 也收集 component 基因符号
        for comp in target.get("target_components", []):
            for syn in comp.get("target_component_synonyms", []):
                if syn.get("syn_type", "").upper() in {"GENE_SYMBOL", "UNIPROT"}:
                    val = syn.get("component_synonym", "").strip().upper()
                    if val:
                        genes.add(val)

    except Exception as e:
        logger.debug(f"  target get failed for {target_chembl_id}: {e}")

    return sorted(genes)


def passes_filter(record: dict) -> bool:
    """判断一条 activity 记录是否满足活性阈值."""
    std_type = str(record.get("standard_type", "") or "").strip().upper()
    if std_type not in {t.upper() for t in ALLOWED_STANDARD_TYPES}:
        return False

    pchembl = record.get("pchembl_value")
    if pchembl is not None:
        try:
            if float(pchembl) >= PChEMBL_THRESHOLD:
                return True
        except (TypeError, ValueError):
            pass

    std_value = record.get("standard_value")
    std_units = str(record.get("standard_units", "") or "").strip().upper()
    if std_value is not None and std_units in {"NM", "NANOMOLAR", "N"}:
        try:
            if float(std_value) <= STANDARD_VALUE_THRESHOLD_NM:
                return True
        except (TypeError, ValueError):
            pass

    return False


def fetch_compound_targets(
    client,
    compounds: list[dict],
    sleep_seconds: float = 0.2,
) -> tuple[list[dict], dict]:
    """批量获取化合物-靶点数据."""
    results = []
    stats = {
        "n_compounds": len(compounds),
        "resolved": 0,
        "with_targets": 0,
        "total_activities": 0,
        "filtered_activities": 0,
        "errors": [],
    }

    for idx, comp in enumerate(compounds, 1):
        name = comp["compound"]
        smiles = comp["smiles"]
        logger.info(f"[{idx}/{len(compounds)}] Querying ChEMBL: {name}")

        try:
            mol_id = resolve_molecule(client, smiles, name)
            if mol_id is None:
                logger.warning(f"  {name}: 未在 ChEMBL 找到对应分子")
                stats["errors"].append(f"{name}: molecule not found")
                continue
            stats["resolved"] += 1
            logger.info(f"  -> molecule_chembl_id={mol_id}")

            activities = fetch_activities(client, mol_id)
            stats["total_activities"] += len(activities)

            seen_targets = set()
            for act in activities:
                if not passes_filter(act):
                    continue
                stats["filtered_activities"] += 1

                target_id = act.get("target_chembl_id")
                if not target_id or target_id in seen_targets:
                    continue
                seen_targets.add(target_id)

                genes = fetch_target_genes(client, target_id)
                if not genes:
                    # 尝试用 target_pref_name 兜底
                    pref_name = act.get("target_pref_name", "")
                    if pref_name:
                        genes = [pref_name.strip().upper()]

                for gene in genes:
                    results.append(
                        {
                            "compound": name,
                            "gene": gene,
                            "target_chembl_id": target_id,
                            "molecule_chembl_id": mol_id,
                            "standard_type": act.get("standard_type"),
                            "standard_value": act.get("standard_value"),
                            "standard_units": act.get("standard_units"),
                            "pchembl_value": act.get("pchembl_value"),
                            "source": "ChEMBL",
                        }
                    )

            if seen_targets:
                stats["with_targets"] += 1
                logger.info(f"  -> {len(seen_targets)} targets, {len(genes) if genes else 0} genes")
            else:
                logger.info("  -> no active targets found")

        except Exception as e:
            err_msg = f"{name}: {e}"
            logger.error(err_msg)
            traceback.print_exc()
            stats["errors"].append(err_msg)

        time.sleep(sleep_seconds)

    return results, stats


def write_outputs(
    results: list[dict],
    stats: dict,
    output_csv: Path,
    metadata_json: Path,
) -> None:
    """写入 CSV 和元数据."""
    if results:
        df = pd.DataFrame(results)
        # 去重: 同一 compound-gene 保留最佳 pchembl
        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = df.sort_values("pchembl_value", ascending=False)
        df = df.drop_duplicates(subset=["compound", "gene"], keep="first")
        df = df.sort_values(["compound", "gene"])
        df.to_csv(output_csv, index=False)
        logger.info(f"已写入 {output_csv}: {len(df)} 条 compound-target 边")
    else:
        output_csv.write_text("compound,gene,target_chembl_id,molecule_chembl_id,standard_type,standard_value,standard_units,pchembl_value,source\n")
        logger.warning("未获取到任何 ChEMBL 靶点数据，写入空文件")

    # 元数据
    metadata_json.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "ChEMBL",
        "source_url": "https://www.ebi.ac.uk/chembl/",
        "client": "chembl_webresource_client",
        "client_url": "https://github.com/chembl/chembl_webresource_client",
        "download_date": pd.Timestamp.now().isoformat(),
        "filters": {
            "standard_types": sorted(ALLOWED_STANDARD_TYPES),
            "pchembl_threshold": PChEMBL_THRESHOLD,
            "standard_value_threshold_nM": STANDARD_VALUE_THRESHOLD_NM,
        },
        "stats": {
            k: v for k, v in stats.items() if k != "errors"
        },
        "errors": stats.get("errors", []),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"已写入元数据: {metadata_json}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch compound-target data from ChEMBL")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--metadata", type=Path, default=METADATA_JSON)
    parser.add_argument("--sleep", type=float, default=0.2, help="API 请求间隔(秒)")
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error(f"输入文件不存在: {args.input}")
        return 1

    compounds = load_compounds(args.input)
    logger.info(f"读取到 {len(compounds)} 个化合物")

    from chembl_webresource_client.new_client import new_client

    client = new_client  # 在 v0.10.9 中 new_client 已是 NewClient 实例
    results, stats = fetch_compound_targets(client, compounds, sleep_seconds=args.sleep)
    write_outputs(results, stats, args.output, args.metadata)

    logger.info("=" * 60)
    logger.info(f"解析成功分子: {stats['resolved']}/{stats['n_compounds']}")
    logger.info(f"有活性靶点分子: {stats['with_targets']}/{stats['n_compounds']}")
    logger.info(f"原始 activity 记录: {stats['total_activities']}")
    logger.info(f"过滤后记录: {stats['filtered_activities']}")
    logger.info(f"错误数: {len(stats['errors'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
