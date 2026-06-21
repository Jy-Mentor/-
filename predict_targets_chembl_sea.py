"""基于 ChEMBL 的 SEA-like 配体相似性靶点预测。

参考开源实现:
- FastTargetPred (ChEMBL 配体相似性靶点预测): https://github.com/ludovicchaput/FastTargetPred
- SEA (Similarity Ensemble Approach): https://sea.bkslab.org/

方法:
1. 对项目中每个化合物, 调用 ChEMBL similarity API 检索结构相似的已知活性分子。
2. 对相似分子, 提取其 pChEMBL >= 5 的 bioactivity 记录。
3. 将 target_chembl_id 解析为 HGNC 基因符号。
4. 仅保留落在 98 个铁衰老核心基因集内的预测边。
5. 用结构相似度作为置信度, 排除已在 compound_target_edges.csv 中的已知边。

输入:
    network_files/compound_smiles.csv
    铁衰老基因.txt
    network_files/compound_target_edges.csv (用于排除已知边)

输出:
    network_files/target_predictions_chembl_sea.csv
    external_data/target_prediction_sea_metadata.json
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
GENE_LIST_PATH = BASE_DIR / "铁衰老基因.txt"
KNOWN_EDGES_CSV = BASE_DIR / "network_files" / "compound_target_edges.csv"
OUTPUT_CSV = BASE_DIR / "network_files" / "target_predictions_chembl_sea.csv"
METADATA_JSON = BASE_DIR / "external_data" / "target_prediction_sea_metadata.json"
CACHE_DIR = BASE_DIR / "external_data" / "cache"
TARGET_CACHE_JSON = CACHE_DIR / "chembl_target_gene_cache.json"
PROGRESS_CSV = CACHE_DIR / "target_predictions_chembl_sea_progress.csv"

PChEMBL_THRESHOLD = 5.0
SIMILARITY_THRESHOLD = 40  # ChEMBL similarity API 阈值 (0-100)
SLEEP_SECONDS = 0.1
MAX_SIMILAR_COMPOUNDS = 50  # 每个化合物最多处理的相似分子数


def load_compounds(csv_path: Path) -> list[dict]:
    """读取项目化合物列表."""
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        smiles = str(row.get("CanonicalSMILES", "")).strip()
        if name and smiles:
            records.append({"compound": name, "smiles": smiles})
    return records


def load_gene_set(path: Path) -> set[str]:
    """读取铁衰老核心基因集."""
    genes: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            g = line.strip().upper()
            if g:
                genes.add(g)
    return genes


def load_known_edges(path: Path) -> set[tuple[str, str]]:
    """读取已知 compound-gene 边, 用于排除."""
    pairs: set[tuple[str, str]] = set()
    if not path.exists():
        return pairs
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        compound = str(row.get("compound", "")).strip()
        gene = str(row.get("gene", "")).strip().upper()
        if compound and gene:
            pairs.add((compound, gene))
    return pairs


def load_target_cache(path: Path) -> dict[str, list[str]]:
    """加载 target_chembl_id -> gene symbols 的持久缓存."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if isinstance(v, list)}
        except Exception:
            logger.warning("target cache 加载失败, 将重建")
    return {}


def save_target_cache(path: Path, cache: dict[str, list[str]]) -> None:
    """保存 target_chembl_id -> gene symbols 缓存."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def load_progress(path: Path) -> dict[str, list[dict]]:
    """加载已处理化合物的进度."""
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        progress: dict[str, list[dict]] = {}
        for _, row in df.iterrows():
            comp = str(row.get("compound", "")).strip()
            if not comp:
                continue
            progress.setdefault(comp, []).append(row.to_dict())
        return progress
    except Exception:
        logger.warning("progress 加载失败, 将从头开始")
        return {}


def append_progress(path: Path, predictions: list[dict]) -> None:
    """追加写入进度文件."""
    if not predictions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(predictions)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def fetch_similar_compounds(client, smiles: str, threshold: int = SIMILARITY_THRESHOLD) -> list[dict]:
    """调用 ChEMBL similarity API 获取相似化合物."""
    try:
        sim = client.similarity
        res = sim.filter(smiles=smiles, similarity=threshold).only(
            ["molecule_chembl_id", "similarity"]
        )
        items = list(res)
        return items
    except Exception as e:
        logger.warning("ChEMBL similarity search failed for %s: %s", smiles[:30], e)
        traceback.print_exc()
        return []


def fetch_activities(client, molecule_chembl_id: str) -> list[dict]:
    """获取某分子的活性记录, 仅保留 pChEMBL >= 阈值的记录."""
    try:
        activity = client.activity
        res = activity.filter(
            molecule_chembl_id=molecule_chembl_id,
            pchembl_value__gte=PChEMBL_THRESHOLD,
        ).only(["target_chembl_id", "pchembl_value", "standard_type"])
        return list(res)
    except Exception as e:
        logger.debug("activity query failed for %s: %s", molecule_chembl_id, e)
        return []


def fetch_target_genes(
    client, target_chembl_id: str, cache: dict[str, list[str]]
) -> list[str]:
    """将 target_chembl_id 解析为 HGNC 基因符号列表, 使用缓存避免重复请求."""
    if target_chembl_id in cache:
        return cache[target_chembl_id]

    genes: set[str] = set()
    try:
        target = client.target.get(target_chembl_id)
        for comp in target.get("target_components", []):
            for syn in comp.get("target_component_synonyms", []):
                if syn.get("syn_type", "").upper() == "GENE_SYMBOL":
                    gene = syn.get("component_synonym", "").strip().upper()
                    if gene:
                        genes.add(gene)
    except Exception as e:
        logger.debug("target get failed for %s: %s", target_chembl_id, e)

    result = sorted(genes)
    cache[target_chembl_id] = result
    return result


def confidence_level_from_score(score: float) -> str:
    """根据置信度分数划分等级."""
    if score >= 0.80:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"


def predict_targets_for_compound(
    client,
    compound_name: str,
    smiles: str,
    core_genes: set[str],
    known_pairs: set[tuple[str, str]],
    target_cache: dict[str, list[str]],
) -> list[dict]:
    """为一个化合物生成 SEA-like 靶点预测."""
    predictions: dict[tuple[str, str], dict] = {}

    similar = fetch_similar_compounds(client, smiles)
    if not similar:
        return []

    # 按相似度降序, 仅处理 top-N 相似分子以控制 API 调用量
    similar = sorted(
        similar,
        key=lambda x: float(x.get("similarity", 0)) if x.get("similarity") is not None else 0.0,
        reverse=True,
    )[:MAX_SIMILAR_COMPOUNDS]

    for sim_item in similar:
        mid = sim_item.get("molecule_chembl_id")
        sim_str = sim_item.get("similarity")
        if not mid or sim_str is None:
            continue
        try:
            sim_score = float(sim_str) / 100.0
        except (TypeError, ValueError):
            continue

        activities = fetch_activities(client, mid)
        time.sleep(SLEEP_SECONDS)

        for act in activities:
            pchembl = act.get("pchembl_value")
            if pchembl is None:
                continue
            try:
                pchembl_val = float(pchembl)
            except (TypeError, ValueError):
                continue
            if pchembl_val < PChEMBL_THRESHOLD:
                continue

            target_id = act.get("target_chembl_id")
            if not target_id:
                continue

            genes = fetch_target_genes(client, target_id, target_cache)
            time.sleep(SLEEP_SECONDS)

            for gene in genes:
                if gene not in core_genes:
                    continue
                if (compound_name, gene) in known_pairs:
                    continue

                # 综合置信度: 结构相似度 * pChEMBL 饱和度 (pChEMBL 7 视为饱和)
                pchembl_factor = min(pchembl_val / 7.0, 1.0)
                confidence = sim_score * pchembl_factor

                key = (compound_name, gene)
                existing = predictions.get(key)
                if existing is None or confidence > existing["confidence"]:
                    predictions[key] = {
                        "compound": compound_name,
                        "gene": gene,
                        "prediction_method": "ChEMBL_SEA_like",
                        "similarity": round(sim_score, 4),
                        "pchembl_value": round(pchembl_val, 2),
                        "nearest_chembl_id": mid,
                        "target_chembl_id": target_id,
                        "standard_type": act.get("standard_type"),
                        "source": "ChEMBL_similarity_SEA",
                        "confidence": round(confidence, 4),
                        "confidence_level": confidence_level_from_score(confidence),
                    }

    return list(predictions.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChEMBL SEA-like target prediction")
    parser.add_argument(
        "--similarity-threshold",
        type=int,
        default=SIMILARITY_THRESHOLD,
        help="ChEMBL similarity threshold (0-100)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=SLEEP_SECONDS,
        help="Sleep seconds between ChEMBL API calls",
    )
    args = parser.parse_args(argv)

    from chembl_webresource_client.new_client import new_client

    client = new_client

    compounds = load_compounds(INPUT_CSV)
    core_genes = load_gene_set(GENE_LIST_PATH)
    known_pairs = load_known_edges(KNOWN_EDGES_CSV)
    target_cache = load_target_cache(TARGET_CACHE_JSON)
    progress = load_progress(PROGRESS_CSV)

    logger.info("化合物数量: %d", len(compounds))
    logger.info("铁衰老核心基因数量: %d", len(core_genes))
    logger.info("已知 compound-target 边数量: %d", len(known_pairs))
    logger.info("target 缓存命中: %d", len(target_cache))
    logger.info("已处理化合物(可恢复): %d", len(progress))

    all_predictions: list[dict] = []
    stats = {
        "n_compounds": len(compounds),
        "n_core_genes": len(core_genes),
        "n_known_pairs": len(known_pairs),
        "similarity_threshold": args.similarity_threshold,
        "pchembl_threshold": PChEMBL_THRESHOLD,
        "predictions_by_compound": {},
        "errors": [],
    }

    for idx, comp in enumerate(compounds, 1):
        name = comp["compound"]
        smiles = comp["smiles"]

        if name in progress:
            logger.info("[%d/%d] 跳过已处理: %s", idx, len(compounds), name)
            preds = progress[name]
            all_predictions.extend(preds)
            stats["predictions_by_compound"][name] = len(preds)
            continue

        logger.info("[%d/%d] 预测靶点: %s", idx, len(compounds), name)
        try:
            preds = predict_targets_for_compound(
                client, name, smiles, core_genes, known_pairs, target_cache
            )
            # 仅保留每个化合物 top-10 预测, 避免低质量边泛滥
            preds = sorted(preds, key=lambda x: x["confidence"], reverse=True)[:10]
            append_progress(PROGRESS_CSV, preds)
            progress[name] = preds
            all_predictions.extend(preds)
            stats["predictions_by_compound"][name] = len(preds)
            logger.info("  -> %d 条新预测", len(preds))
        except Exception as e:
            err_msg = f"{name}: {e}"
            logger.error(err_msg)
            traceback.print_exc()
            stats["errors"].append(err_msg)
        time.sleep(args.sleep)
        # 每处理 5 个化合物持久化一次 target cache
        if idx % 5 == 0:
            save_target_cache(TARGET_CACHE_JSON, target_cache)

    save_target_cache(TARGET_CACHE_JSON, target_cache)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)

    if all_predictions:
        df = pd.DataFrame(all_predictions)
        df = df.sort_values(["compound", "confidence"], ascending=[True, False])
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info("已写入 %s: %d 条预测边", OUTPUT_CSV, len(df))
    else:
        logger.warning("未生成任何预测边")
        pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "prediction_method",
                "similarity",
                "pchembl_value",
                "nearest_chembl_id",
                "target_chembl_id",
                "standard_type",
                "source",
                "confidence",
                "confidence_level",
            ]
        ).to_csv(OUTPUT_CSV, index=False)

    level_counts = {}
    if all_predictions:
        for pred in all_predictions:
            level_counts[pred["confidence_level"]] = level_counts.get(pred["confidence_level"], 0) + 1

    metadata = {
        "source": "ChEMBL_similarity_SEA",
        "reference_implementation": "FastTargetPred (https://github.com/ludovicchaput/FastTargetPred)",
        "method": "ligand-based target inference via ChEMBL similarity API",
        "download_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_compounds": len(compounds),
        "input_genes": len(core_genes),
        "similarity_threshold": args.similarity_threshold,
        "pchembl_threshold": PChEMBL_THRESHOLD,
        "n_predictions": len(all_predictions),
        "confidence_level_counts": level_counts,
        "output_file": str(OUTPUT_CSV),
        "errors": stats["errors"],
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("元数据已写入 %s", METADATA_JSON)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
