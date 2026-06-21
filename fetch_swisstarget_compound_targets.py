"""从 SwissTargetPrediction 抓取化合物靶点预测, 仅保留铁衰老核心基因.

 SwissTargetPrediction 没有官方批量 API. 本脚本模拟浏览器表单提交:
   1. GET  https://www.swisstargetprediction.ch/index.php 获取会话.
   2. POST https://www.swisstargetprediction.ch/predict.php
      参数: organism=Homo_sapiens, smiles=<SMILES>, ioi=2
   3. 从响应中提取 result.php?job=<JOB_ID>&organism=...
   4. GET 结果页并解析 DataTables 表格.
   5. 仅保留 98 个铁衰老核心基因内的预测, 并排除已知的 compound-target 边.

 输出:
   network_files/swisstarget_compound_targets.csv
   external_data/swisstarget_download_metadata.json

 参考:
   - SwissTargetPrediction: Daina et al., Nucleic Acids Res. (2019)
   - 网站: https://www.swisstargetprediction.ch
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import re
import time
import traceback
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = PROJECT_ROOT / "network_files"
EXTERNAL_DIR = PROJECT_ROOT / "external_data"
GENE_LIST_PATH = PROJECT_ROOT / "铁衰老基因.txt"

INPUT_CSV = NETWORK_DIR / "compound_smiles.csv"
KNOWN_EDGES_CSV = NETWORK_DIR / "compound_target_edges.csv"
OUTPUT_CSV = NETWORK_DIR / "swisstarget_compound_targets.csv"
METADATA_JSON = EXTERNAL_DIR / "swisstarget_download_metadata.json"
PROGRESS_CSV = EXTERNAL_DIR / "cache" / "swisstarget_progress.csv"
PROCESSED_CSV = EXTERNAL_DIR / "cache" / "swisstarget_processed_compounds.csv"

BASE_URL = "https://www.swisstargetprediction.ch"
INDEX_URL = f"{BASE_URL}/index.php"
PREDICT_URL = f"{BASE_URL}/predict.php"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": BASE_URL,
    "Referer": INDEX_URL,
}

DEFAULT_SLEEP_BETWEEN = 5.0
DEFAULT_RESULT_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 3
DEFAULT_RESULT_POLL_MAX = 12


def load_gene_set(path: Path) -> set[str]:
    """读取铁衰老核心基因集."""
    genes: set[str] = set()
    if not path.exists():
        logger.error("基因集文件不存在: %s", path)
        return genes
    for line in path.read_text(encoding="utf-8").splitlines():
        g = line.strip().upper()
        if g:
            genes.add(g)
    return genes


def clean_smiles_for_submission(smiles: str) -> str:
    """去除 SMILES 中的盐/溶剂片段, 保留最长片段, 提高 SwissTargetPrediction 兼容性.

    SwissTargetPrediction 对含 '.' 的混合物/盐形式 SMILES 可能拒绝提交.
    取最长片段作为母体化合物进行预测.
    """
    fragments = [f.strip() for f in smiles.split(".") if f.strip()]
    if not fragments:
        return smiles
    # 选择原子数最多的片段(更可能是母体)
    return max(fragments, key=lambda f: len(f))


def load_compounds(path: Path) -> list[dict]:
    """读取项目化合物 SMILES 列表."""
    if not path.exists():
        logger.error("化合物 SMILES 文件不存在: %s", path)
        return []
    df = pd.read_csv(path)
    records: list[dict] = []
    for _, row in df.iterrows():
        name = str(row.get("compound", "")).strip()
        smiles = str(row.get("CanonicalSMILES", "")).strip()
        if name and smiles:
            records.append({"compound": name, "smiles": clean_smiles_for_submission(smiles)})
    return records


def load_known_pairs(path: Path) -> set[tuple[str, str]]:
    """读取已知 compound-target 边, 避免重复预测."""
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


def load_progress(path: Path, processed_path: Path) -> set[str]:
    """加载已处理化合物名称集合(包括有预测和无预测的化合物)."""
    processed: set[str] = set()
    for p in (path, processed_path):
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            processed.update(str(x).strip() for x in df["compound"].dropna().unique())
        except Exception:
            logger.warning("progress 加载失败: %s", p)
    return processed


def append_progress(path: Path, predictions: list[dict]) -> None:
    """追加写入进度文件."""
    if not predictions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(predictions)
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def mark_processed(path: Path, compound_name: str) -> None:
    """记录化合物已被处理(即使无预测结果)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{"compound": compound_name, "processed_at": pd.Timestamp.now().isoformat()}])
    header = not path.exists()
    df.to_csv(path, mode="a", index=False, header=header)


def create_session() -> requests.Session:
    """创建带浏览器头的请求会话."""
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def extract_job_url(text: str) -> str | None:
    """从 predict.php 响应中提取结果页 URL."""
    m = re.search(
        r'location\.replace\(["\'](https://www\.swisstargetprediction\.ch/result\.php\?[^"\']+)["\']\)',
        text,
    )
    if m:
        return m.group(1)
    m = re.search(r'result\.php\?job=(\d+)&organism=([^"\'\s]+)', text)
    if m:
        return f"{BASE_URL}/result.php?job={m.group(1)}&organism={m.group(2)}"
    return None


class SwissTargetSubmissionError(Exception):
    """SwissTargetPrediction 提交或结果获取失败."""


def submit_prediction(
    session: requests.Session,
    smiles: str,
    result_timeout: int = DEFAULT_RESULT_TIMEOUT,
    poll_max: int = DEFAULT_RESULT_POLL_MAX,
) -> str:
    """提交预测任务并返回结果页 HTML. 失败时抛出异常以便重试."""
    payload = {"organism": "Homo_sapiens", "smiles": smiles, "ioi": "2"}
    try:
        r = session.post(PREDICT_URL, data=payload, timeout=result_timeout, allow_redirects=False)
    except requests.RequestException as e:
        raise SwissTargetSubmissionError(f"POST predict.php 请求失败: {e}") from e

    logger.debug("POST predict status: %d, len: %d", r.status_code, len(r.text))

    if r.status_code not in (200, 302):
        raise SwissTargetSubmissionError(f"predict.php 返回非预期状态码: {r.status_code}")

    if "will not submitted" in r.text or "Please contact the SwissTargetPrediction team" in r.text:
        raise SwissTargetSubmissionError(
            "SwissTargetPrediction 服务器拒绝提交(可能触发反爬虫/速率限制), "
            "页面提示: 'Your job will not submitted. Please contact the SwissTargetPrediction team.'"
        )

    job_url = extract_job_url(r.text)
    if not job_url:
        # 保存响应以便调试
        debug_path = EXTERNAL_DIR / "cache" / f"swisstarget_debug_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.html"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(r.text, encoding="utf-8")
        raise SwissTargetSubmissionError(f"未从 predict.php 响应中提取到 job URL, 已保存调试文件: {debug_path}")

    # 轮询结果页, 等待计算完成
    last_html = ""
    for attempt in range(poll_max):
        time.sleep(min(2 + attempt, 10))
        try:
            res = session.get(job_url, timeout=result_timeout)
        except requests.RequestException as e:
            raise SwissTargetSubmissionError(f"GET result.php 请求失败: {e}") from e
        last_html = res.text
        if "Target</th>" in res.text and "Common name</th>" in res.text:
            return last_html
        if "Please be patient" in res.text or "submitted" in res.text.lower():
            logger.info("  结果尚未就绪, 继续轮询 (%d/%d)", attempt + 1, poll_max)
            continue
        # 如果页面内容足够长且不再变化, 可能已完成但未检测到表格
        if attempt > 0 and len(res.text) > 50000:
            return last_html

    raise SwissTargetSubmissionError("结果页轮询超时")


def parse_result_table(html: str) -> pd.DataFrame:
    """解析 SwissTargetPrediction 结果表格."""
    tables = pd.read_html(io.StringIO(html))
    for table in tables:
        cols = [str(c) for c in table.columns]
        if "Target" in cols and "Common name" in cols:
            return table.copy()
    raise ValueError("未在结果页中找到目标预测表格")


def confidence_level_from_probability(prob: float) -> str:
    """根据 SwissTargetPrediction Probability 划分置信度等级."""
    if prob >= 0.50:
        return "high"
    if prob >= 0.30:
        return "medium"
    return "low"


def predict_targets_for_compound(
    session: requests.Session,
    compound_name: str,
    smiles: str,
    core_genes: set[str],
    known_pairs: set[tuple[str, str]],
    probability_threshold: float = 0.05,
    top_k: int | None = None,
) -> list[dict]:
    """为一个化合物抓取并解析靶点预测."""
    html = submit_prediction(session, smiles)
    if html is None:
        return []

    df = parse_result_table(html)
    predictions: list[dict] = []

    for _, row in df.iterrows():
        gene = str(row.get("Common name", "")).strip().upper()
        if not gene or gene not in core_genes:
            continue
        if (compound_name, gene) in known_pairs:
            continue

        prob_raw = row.get("Probability*", "")
        try:
            probability = float(prob_raw)
        except (TypeError, ValueError):
            logger.warning("  %s -> %s 概率解析失败: %s", compound_name, gene, prob_raw)
            continue

        if probability < probability_threshold:
            continue

        predictions.append({
            "compound": compound_name,
            "gene": gene,
            "target_name": str(row.get("Target", "")).strip(),
            "uniprot_id": str(row.get("Uniprot ID", "")).strip(),
            "chembl_id": str(row.get("ChEMBL ID", "")).strip(),
            "target_class": str(row.get("Target Class", "")).strip(),
            "probability": round(probability, 6),
            "known_actives": str(row.get("Known actives (3D/2D)", "")).strip(),
            "source": "SwissTargetPrediction",
            "confidence": round(probability, 6),
            "confidence_level": confidence_level_from_probability(probability),
        })

    predictions = sorted(predictions, key=lambda x: x["probability"], reverse=True)
    if top_k:
        predictions = predictions[:top_k]
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch SwissTargetPrediction compound targets")
    parser.add_argument(
        "--probability-threshold",
        type=float,
        default=0.05,
        help="Minimum SwissTargetPrediction probability to keep (default: 0.05)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep top-k predictions per compound (default: all above threshold)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_BETWEEN,
        help="Seconds between compound submissions (default: 5.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retries per compound (default: 3)",
    )
    args = parser.parse_args(argv)

    core_genes = load_gene_set(GENE_LIST_PATH)
    compounds = load_compounds(INPUT_CSV)
    known_pairs = load_known_pairs(KNOWN_EDGES_CSV)
    processed = load_progress(PROGRESS_CSV, PROCESSED_CSV)

    # 若 progress 文件已有历史记录但 processed 文件不存在, 用 progress 初始化 processed
    if PROGRESS_CSV.exists() and not PROCESSED_CSV.exists():
        try:
            prog_df = pd.read_csv(PROGRESS_CSV)
            for comp_name in prog_df["compound"].dropna().unique():
                mark_processed(PROCESSED_CSV, str(comp_name).strip())
            processed = load_progress(PROGRESS_CSV, PROCESSED_CSV)
        except Exception:
            logger.warning("初始化 processed 文件失败")

    logger.info("核心铁衰老基因: %d", len(core_genes))
    logger.info("待处理化合物: %d", len(compounds))
    logger.info("已知 compound-target 边: %d", len(known_pairs))
    logger.info("已处理化合物: %d", len(processed))

    all_predictions: list[dict] = []
    errors: list[str] = []

    for idx, comp in enumerate(compounds, 1):
        name = comp["compound"]
        smiles = comp["smiles"]

        if name in processed:
            logger.info("[%d/%d] 跳过已处理: %s", idx, len(compounds), name)
            # 从进度文件读取该化合物的预测并加入 all_predictions
            try:
                prog_df = pd.read_csv(PROGRESS_CSV)
                prog_df = prog_df[prog_df["compound"] == name]
                all_predictions.extend(prog_df.to_dict("records"))
            except Exception:
                logger.warning("  读取 %s 进度失败", name)
            continue

        logger.info("[%d/%d] 提交 SwissTargetPrediction: %s", idx, len(compounds), name)
        preds: list[dict] = []
        # 每次重试使用新 session, 避免被服务器限制
        for attempt in range(args.max_retries + 1):
            session = create_session()
            try:
                preds = predict_targets_for_compound(
                    session,
                    name,
                    smiles,
                    core_genes,
                    known_pairs,
                    probability_threshold=args.probability_threshold,
                    top_k=args.top_k,
                )
                break
            except Exception as e:
                err_msg = f"{name} attempt {attempt + 1}: {e}"
                logger.error(err_msg)
                traceback.print_exc()
                if attempt == args.max_retries:
                    errors.append(err_msg)
                else:
                    # 指数退避, 每次重试增加等待时间
                    time.sleep(min(30, 5 * (attempt + 1) ** 2))

        # 无论是否有预测结果, 都标记为已处理, 避免重复提交
        mark_processed(PROCESSED_CSV, name)
        if preds:
            logger.info("  -> 保留 %d 条预测", len(preds))
            append_progress(PROGRESS_CSV, preds)
            all_predictions.extend(preds)
        else:
            logger.info("  -> 无新预测")

        #  polite delay between submissions
        if idx < len(compounds):
            time.sleep(args.sleep)

    # 去重: 同一 compound-gene 保留最高 probability
    if all_predictions:
        df_all = pd.DataFrame(all_predictions)
        df_all = df_all.sort_values("probability", ascending=False)
        df_all = df_all.drop_duplicates(subset=["compound", "gene"], keep="first")
        df_all = df_all.sort_values(["compound", "probability"], ascending=[True, False])
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(OUTPUT_CSV, index=False)
        logger.info("已写入 %s: %d 条预测边", OUTPUT_CSV, len(df_all))
    else:
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=[
                "compound",
                "gene",
                "target_name",
                "uniprot_id",
                "chembl_id",
                "target_class",
                "probability",
                "known_actives",
                "source",
                "confidence",
                "confidence_level",
            ]
        ).to_csv(OUTPUT_CSV, index=False)
        logger.warning("未生成任何预测边")

    metadata = {
        "source": "SwissTargetPrediction",
        "url": BASE_URL,
        "reference": "Daina et al., Nucleic Acids Res. (2019)",
        "download_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "organism": "Homo_sapiens",
        "input_compounds": len(compounds),
        "input_genes": len(core_genes),
        "probability_threshold": args.probability_threshold,
        "top_k": args.top_k,
        "n_predictions": len(all_predictions),
        "unique_predictions": (
            len(pd.DataFrame(all_predictions).drop_duplicates(["compound", "gene"]))
            if all_predictions else 0
        ),
        "output_file": str(OUTPUT_CSV),
        "errors": errors,
    }
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("元数据已写入 %s", METADATA_JSON)

    if errors:
        logger.warning("处理过程中发生 %d 个错误", len(errors))
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
