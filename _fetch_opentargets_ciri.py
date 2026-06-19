"""从 OpenTargets GraphQL API 获取 CIRI / 卒中相关疾病-基因关联。

OpenTargets Platform: https://platform.opentargets.org/
API: https://api.platform.opentargets.org/api/v4/graphql
特点: 公开 API, 无需注册/认证, 整合 GWAS/ClinVar/表达/文献等多源证据。

目标疾病 (EFO):
  - 卒中 (stroke): EFO_0000712
  - 缺血性卒中 (ischemic stroke): EFO_0001645 (MONDO_0005098)
  - 脑缺血 (cerebral ischemia): EFO_0000227
  - 再灌注损伤 (reperfusion injury): EFO_0004262
  - 缺氧缺血性脑病 (hypoxic-ischemic encephalopathy): EFO_0009502

输出:
  - network_files/opentargets_ciri_genes.csv (追加到 disease_gene_associations.csv)
"""
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "network_files"
OUT_DIR.mkdir(exist_ok=True)

OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# CIRI 相关疾病 ID (经 OpenTargets search API 校验)
DISEASE_MAP = {
    "CIRI_stroke": "EFO_0000712",
    "CIRI_ischemic_stroke": "HP_0002140",
    "CIRI_cerebrovascular_disorder": "EFO_0003763",
}

DISEASE_QUERY = """
query DiseaseAssociations($diseaseId: String!, $size: Int!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets(page: { index: 0, size: $size }) {
      rows {
        target {
          id
          approvedSymbol
          approvedName
        }
        score
      }
    }
  }
}
"""


def fetch_disease_genes(disease_id: str, size: int = 1000):
    """获取某疾病在 OpenTargets 中的关联基因。"""
    payload = {
        "query": DISEASE_QUERY,
        "variables": {"diseaseId": disease_id, "size": size},
    }
    try:
        r = requests.post(OT_URL, json=payload, timeout=60)
        if r.status_code != 200:
            logger.warning(f"  {disease_id} HTTP {r.status_code}: {r.text[:200]}")
            return []
        data = r.json()
        if data is None or "data" not in data:
            logger.warning(f"  {disease_id}: 返回空数据")
            return []
        if data["data"] is None or data["data"].get("disease") is None:
            logger.warning(f"  {disease_id}: 疾病未找到或响应异常, data={str(data)[:200]}")
            return []
        rows = data["data"]["disease"]["associatedTargets"]["rows"]
        return rows
    except Exception as e:
        logger.warning(f"  {disease_id} 查询失败: {e}")
        return []


def main(score_threshold: float = 0.1):
    all_records = []
    for label, efo_id in DISEASE_MAP.items():
        logger.info(f"查询 OpenTargets: {label} ({efo_id})")
        rows = fetch_disease_genes(efo_id, size=1000)
        logger.info(f"  返回 {len(rows)} 条关联")
        for row in rows:
            score = row.get("score", 0.0)
            if score < score_threshold:
                continue
            target = row["target"]
            sym = target.get("approvedSymbol")
            if not sym:
                continue
            all_records.append({
                "disease": "CIRI",
                "gene": sym.upper(),
                "score": score,
                "efo_id": efo_id,
                "disease_name": label,
                "target_name": target.get("approvedName", ""),
            })
        time.sleep(1.0)  # 礼貌请求间隔

    if not all_records:
        logger.warning("未获取到 OpenTargets 数据")
        return

    df = pd.DataFrame(all_records)
    # 合并同一基因的最高分
    df = df.sort_values("score", ascending=False).drop_duplicates(subset=["gene"], keep="first")
    df = df.sort_values("score", ascending=False)

    out_csv = OUT_DIR / "opentargets_ciri_genes.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"已保存: {out_csv} ({len(df)} genes, score>={score_threshold})")

    # 输出 top 20
    logger.info("Top 20 CIRI genes by OpenTargets score:")
    for _, row in df.head(20).iterrows():
        logger.info(f"  {row['gene']}: {row['score']:.3f} ({row['disease_name']})")


if __name__ == "__main__":
    main(score_threshold=0.1)
