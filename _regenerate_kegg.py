"""重新生成 kegg_pathway_genes.csv（人类通路）.

特性:
- 并发下载 KEGG 通路基因映射（带速率限制）
- 单条通路缓存，支持断点续传
- 下载失败或进程被中断时自动恢复旧文件
- 命令行可配置 worker 数、超时、缓存目录

用法:
    python _regenerate_kegg.py
    python _regenerate_kegg.py --workers 4 --timeout 60 --cache-dir .cache/kegg
"""

import argparse
import csv
import logging
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from download_external_data import _load_db_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("network_files")
KEGG_FILE = OUT_DIR / "kegg_pathway_genes.csv"
BACKUP_FILE = OUT_DIR / "kegg_pathway_genes.csv.bak"

# KEGG API 速率限制: 每秒不超过 3 次请求
RATE_LIMIT_LOCK = threading.Lock()
MIN_INTERVAL = 0.35
_last_request_time = 0.0


def _rate_limited_request(session: requests.Session, url: str, timeout: int) -> str:
    """带全局速率限制的 GET 请求."""
    global _last_request_time  # noqa: PLW0603

    with RATE_LIMIT_LOCK:
        elapsed = time.time() - _last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        _last_request_time = time.time()

    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _fetch_pathway(
    session: requests.Session,
    kegg_id: str,
    pathway_name: str,
    cache_dir: Path,
    timeout: int,
    max_retries: int = 3,
) -> list[dict]:
    """下载单个 KEGG 通路的基因映射，优先读取缓存."""
    cache_file = cache_dir / f"{kegg_id}.txt"
    url = f"https://rest.kegg.jp/link/hsa/{kegg_id}"

    if cache_file.exists():
        text = cache_file.read_text(encoding="utf-8")
        logger.info("[缓存] %s (%s)", kegg_id, pathway_name)
    else:
        text = None
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                text = _rate_limited_request(session, url, timeout)
                cache_file.write_text(text, encoding="utf-8")
                logger.info("[下载] %s (%s)", kegg_id, pathway_name)
                break
            except Exception as e:
                last_error = e
                logger.warning("[重试 %d/%d] %s 下载失败: %s", attempt + 1, max_retries, kegg_id, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        if text is None:
            raise RuntimeError(f"{kegg_id} 下载失败: {last_error}") from last_error

    rows: list[dict] = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            gene_id = parts[1].strip()
            rows.append({
                "pathway": pathway_name,
                "gene_id": gene_id,
                "kegg_id": kegg_id,
                "source": "KEGG_REST",
            })
    return rows


def _download_all(
    pathways: dict[str, str],
    cache_dir: Path,
    workers: int,
    timeout: int,
) -> list[dict]:
    """并发下载所有通路基因映射."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    })

    rows: list[dict] = []
    failed: list[tuple[str, str, str]] = []
    total = len(pathways)
    completed = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_info = {
            executor.submit(
                _fetch_pathway, session, kegg_id, pathway_name, cache_dir, timeout
            ): (kegg_id, pathway_name)
            for kegg_id, pathway_name in pathways.items()
        }
        for future in as_completed(future_to_info):
            kegg_id, pathway_name = future_to_info[future]
            completed += 1
            try:
                pathway_rows = future.result()
                rows.extend(pathway_rows)
                logger.info("[%d/%d] %s 完成: %d 条基因", completed, total, kegg_id, len(pathway_rows))
            except Exception as e:
                failed.append((kegg_id, pathway_name, str(e)))
                logger.error("[%d/%d] %s 失败: %s", completed, total, kegg_id, e)

    if failed:
        logger.warning("共有 %d 条通路下载失败", len(failed))
        for kegg_id, pathway_name, err in failed:
            logger.warning("  - %s (%s): %s", kegg_id, pathway_name, err)

    return rows


def _write_csv(rows: list[dict]) -> Path:
    """将结果写入 CSV."""
    fieldnames = ["pathway", "gene_id", "kegg_id", "source"]
    with open(KEGG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return KEGG_FILE


def _restore_backup() -> None:
    """下载失败或中断时从备份恢复旧文件."""
    if BACKUP_FILE.exists():
        if KEGG_FILE.exists():
            KEGG_FILE.unlink()
        BACKUP_FILE.rename(KEGG_FILE)
        logger.info("已恢复旧文件: %s", KEGG_FILE)


def _on_interrupt(signum, frame) -> None:  # noqa: ARG001
    """捕获 SIGINT/SIGTERM，确保备份被恢复后再退出."""
    logger.warning("收到中断信号，正在恢复旧文件...")
    _restore_backup()
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="重新生成 KEGG 通路基因映射")
    parser.add_argument("--workers", type=int, default=4, help="并发下载线程数 (默认 4)")
    parser.add_argument("--timeout", type=int, default=60, help="单次请求超时秒数 (默认 60)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache") / "kegg",
        help="单条通路缓存目录 (默认 .cache/kegg)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="不备份旧文件（危险，默认会备份）",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _on_interrupt)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_interrupt)

    logger.info("重新生成 KEGG 人类通路基因映射...")
    logger.info("配置: workers=%d, timeout=%ds, cache=%s", args.workers, args.timeout, args.cache_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    db_config = _load_db_config()
    pathways = db_config.get("kegg_pathways", {})
    if not pathways:
        logger.error("external_db_config.yaml 中未找到 kegg_pathways 配置")
        sys.exit(1)
    logger.info("待下载通路数: %d", len(pathways))

    # 先备份旧文件，避免下载失败导致数据丢失
    if KEGG_FILE.exists() and not args.no_backup:
        if BACKUP_FILE.exists():
            BACKUP_FILE.unlink()
        KEGG_FILE.rename(BACKUP_FILE)
        logger.info("已备份旧文件: %s", BACKUP_FILE)

    try:
        rows = _download_all(pathways, args.cache_dir, args.workers, args.timeout)
        if not rows:
            raise RuntimeError("没有成功下载任何通路基因数据")

        _write_csv(rows)
        n_pathways = len({r["pathway"] for r in rows})
        logger.info(
            "完成: %s (%d 行, %d 通路)",
            KEGG_FILE, len(rows), n_pathways,
        )

        # 成功后删除备份
        if BACKUP_FILE.exists():
            BACKUP_FILE.unlink()
            logger.info("已删除备份: %s", BACKUP_FILE)
    except Exception:
        logger.error("重新生成 KEGG 通路基因映射失败，正在恢复旧文件...")
        traceback.print_exc()
        if not args.no_backup:
            _restore_backup()
        sys.exit(1)


if __name__ == "__main__":
    main()
