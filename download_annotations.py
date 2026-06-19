#!/usr/bin/env python3
"""
全基因组基因注释下载 (并行优化版)
数据源: mygene.info API
基因列表: L1_genome_wide_de.csv → 22,777个唯一基因
输出:
  - data/go_terms.tsv
  - data/kegg_pathways.tsv
  - data/interpro_domains.tsv
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import mygene
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / ".annotation_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 200       # 每批基因数
MAX_WORKERS = 4        # 并行 workers

def load_gene_list():
    l1_file = Path(__file__).parent / "L3" / "L1_genome_wide_de.csv"
    l1 = pd.read_csv(l1_file)
    genes = l1["gene"].unique().tolist()
    print(f"L1全基因组基因数: {len(genes)}")
    return genes

def load_done_genes():
    f = CACHE_DIR / "done_genes.txt"
    if f.exists():
        with open(f) as fh:
            return set(line.strip() for line in fh if line.strip())
    return set()

def save_done_genes(batch):
    with open(CACHE_DIR / "done_genes.txt", "a") as f:
        for g in batch:
            f.write(g + "\n")

def process_batch(batch):
    """查询一批基因的注释, 返回 (go_set, kegg_set, ipr_set)"""
    mg = mygene.MyGeneInfo()
    try:
        results = mg.querymany(
            batch, scopes="symbol",
            fields="symbol,go,pathway.kegg,interpro",
            species="human", returnall=True, verbose=False
        )
    except Exception as e:
        print(f"  批次出错: {e}")
        return None

    go_set = set()
    kegg_set = set()
    ipr_set = set()

    for item in results.get("out", []):
        gene = item.get("query", "") or item.get("symbol", "")
        if not gene:
            continue

        go_info = item.get("go", {})
        for cat in ["BP", "CC", "MF"]:
            for entry in go_info.get(cat, []):
                tid = entry if isinstance(entry, str) else entry.get("id", "")
                if tid:
                    go_set.add((gene, tid))

        kegg_info = item.get("pathway", {}).get("kegg", {})
        if isinstance(kegg_info, dict):
            for pw_id in kegg_info:
                kegg_set.add((gene, pw_id))
        elif isinstance(kegg_info, str):
            kegg_set.add((gene, kegg_info))

        ipr_info = item.get("interpro", [])
        if isinstance(ipr_info, list):
            for ipr in ipr_info:
                ipr_id = ipr.get("id", "") if isinstance(ipr, dict) else ""
                if ipr_id:
                    ipr_set.add((gene, ipr_id))

    return go_set, kegg_set, ipr_set

def merge_and_save(all_go, all_kegg, all_ipr):
    pd.DataFrame([(g, t) for g, t in all_go], columns=["GeneSymbol", "GO_term"]).drop_duplicates().to_csv(
        OUTPUT_DIR / "go_terms.tsv", sep="\t", index=False)
    pd.DataFrame([(g, t) for g, t in all_kegg], columns=["GeneSymbol", "Pathway"]).drop_duplicates().to_csv(
        OUTPUT_DIR / "kegg_pathways.tsv", sep="\t", index=False)
    pd.DataFrame([(g, t) for g, t in all_ipr], columns=["GeneSymbol", "Domain"]).drop_duplicates().to_csv(
        OUTPUT_DIR / "interpro_domains.tsv", sep="\t", index=False)
    print(f"    保存: GO={len(all_go)}条, KEGG={len(all_kegg)}条, IPR={len(all_ipr)}条")

def main():
    print("=" * 60)
    print("全基因组基因注释下载 (并行优化)")
    print("=" * 60)

    genes = load_gene_list()
    done_genes = load_done_genes()
    remaining = [g for g in genes if g not in done_genes]
    print(f"已完成: {len(done_genes)}, 剩余: {len(remaining)}")

    # 加载已有结果
    all_go = set()
    all_kegg = set()
    all_ipr = set()
    for fn, cols, key_set in [
        ("go_terms.tsv", ["GeneSymbol", "GO_term"], all_go),
        ("kegg_pathways.tsv", ["GeneSymbol", "Pathway"], all_kegg),
        ("interpro_domains.tsv", ["GeneSymbol", "Domain"], all_ipr),
    ]:
        fp = OUTPUT_DIR / fn
        if fp.exists():
            df = pd.read_csv(fp, sep="\t")
            for _, r in df.iterrows():
                key_set.add((r[cols[0]], r[cols[1]]))

    if not remaining:
        print("全部基因已打！")
        merge_and_save(all_go, all_kegg, all_ipr)
        return

    # 分批并行处理
    batches = [remaining[i:i+BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    print(f"共 {len(batches)} 批, 每批{BATCH_SIZE}个, {MAX_WORKERS}线程并行\n")

    t0 = time.time()
    completed = 0

    for i in range(0, len(batches), MAX_WORKERS):
        chunk = batches[i:i+MAX_WORKERS]
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            futures = {pool.submit(process_batch, b): b for b in chunk}
            for fut in as_completed(futures):
                batch = futures[fut]
                result = fut.result()
                completed += 1
                pct = completed / len(batches) * 100
                elapsed = time.time() - t0

                if result is None:
                    print(f"[{completed}/{len(batches)}] ({pct:.0f}%) 批失败, {elapsed:.0f}s")
                else:
                    go_s, kegg_s, ipr_s = result
                    all_go.update(go_s)
                    all_kegg.update(kegg_s)
                    all_ipr.update(ipr_s)
                    save_done_genes(batch)
                    print(f"[{completed}/{len(batches)}] ({pct:.0f}%) GO+{len(go_s)} KEGG+{len(kegg_s)} IPR+{len(ipr_s)} | {elapsed:.0f}s")

        # 每10批保存一次
        if (i // MAX_WORKERS) % 10 == 0:
            merge_and_save(all_go, all_kegg, all_ipr)

    # 最终保存
    print(f"\n总耗时: {time.time()-t0:.0f}s")
    merge_and_save(all_go, all_kegg, all_ipr)

    print("\n=== 最终结果 ===")
    print(f"  GO terms: {len(all_go)}")
    print(f"  KEGG: {len(all_kegg)}")
    print(f"  InterPro: {len(all_ipr)}")
    print("  文件: data/go_terms.tsv, data/kegg_pathways.tsv, data/interpro_domains.tsv")

if __name__ == "__main__":
    main()
