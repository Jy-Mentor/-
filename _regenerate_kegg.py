"""重新生成 kegg_pathway_genes.csv（人类通路）.

下载失败时会自动恢复旧文件，避免数据丢失.
"""
import logging
import sys
import traceback
from pathlib import Path

from download_external_data import download_kegg_pathway_genes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("network_files")
KEGG_FILE = OUT_DIR / "kegg_pathway_genes.csv"
BACKUP_FILE = OUT_DIR / "kegg_pathway_genes.csv.bak"


def _restore_backup() -> None:
    """下载失败时从备份恢复旧文件."""
    if BACKUP_FILE.exists():
        if KEGG_FILE.exists():
            KEGG_FILE.unlink()
        BACKUP_FILE.rename(KEGG_FILE)
        logger.info("已恢复旧文件: %s", KEGG_FILE)


if __name__ == "__main__":
    logger.info("重新生成 KEGG 人类通路基因映射...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 先备份旧文件，避免下载失败导致数据丢失
    if KEGG_FILE.exists():
        if BACKUP_FILE.exists():
            BACKUP_FILE.unlink()
        KEGG_FILE.rename(BACKUP_FILE)
        logger.info("已备份旧文件: %s", BACKUP_FILE)

    try:
        path = download_kegg_pathway_genes()
        if Path(path).exists() and Path(path).stat().st_size > 0:
            logger.info("完成: %s", path)
            # 成功后删除备份
            if BACKUP_FILE.exists():
                BACKUP_FILE.unlink()
                logger.info("已删除备份: %s", BACKUP_FILE)
        else:
            raise RuntimeError(f"下载结果为空或文件不存在: {path}")
    except Exception:
        logger.error("重新生成 KEGG 通路基因映射失败，正在恢复旧文件...")
        traceback.print_exc()
        _restore_backup()
        sys.exit(1)
