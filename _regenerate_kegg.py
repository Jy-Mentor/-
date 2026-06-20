"""重新生成 kegg_pathway_genes.csv（人类通路）."""
import logging

from download_external_data import download_kegg_pathway_genes

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("重新生成 KEGG 人类通路基因映射...")
    path = download_kegg_pathway_genes()
    logger.info("完成: %s", path)
