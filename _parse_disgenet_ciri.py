"""
Parse DisGeNET curated gene-disease associations from GitHub mirror.

Source: https://github.com/dhimmel/disgenet (DisGeNET v3.0, May 2015)
License: Open Database License

This script reads the downloaded curated_gene_disease_associations.txt.gz
and extracts CIRI / AD / Aging related genes, generating:
  - network_files/disgenet_disease_genes.csv
  - network_files/disgenet_ciri_genes.csv
"""

import csv
import gzip
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).parent
DISGENET_DIR = BASE_DIR / "external_data" / "disgenet"
OUT_DIR = BASE_DIR / "network_files"
CONFIG_FILE = OUT_DIR / "external_db_config.yaml"


def _load_db_config():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_disgenet_curated(score_threshold: float = 0.0):
    """Parse DisGeNET curated associations and map to target diseases."""
    in_file = DISGENET_DIR / "curated_gene_disease_associations.txt.gz"
    if not in_file.exists():
        raise FileNotFoundError(f"DisGeNET curated file not found: {in_file}")

    db_config = _load_db_config()
    target_diseases = db_config.get('diseases', {})

    disease_hits = {d: set() for d in target_diseases}
    rows = []
    ciri_rows = []

    with gzip.open(in_file, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gene = row.get('geneSymbol', '').strip().upper()
            disease_name = row.get('diseaseName', '').strip().lower()
            disease_id = row.get('diseaseId', '').strip()
            try:
                score = float(row.get('score', 0.0))
            except ValueError:
                score = 0.0

            if not gene or not disease_name:
                continue
            if score < score_threshold:
                continue

            for target_disease, info in target_diseases.items():
                for kw in info.get('keywords', []):
                    if kw.lower() in disease_name:
                        if gene not in disease_hits[target_disease]:
                            disease_hits[target_disease].add(gene)
                            rows.append({
                                'disease': target_disease,
                                'gene': gene,
                                'disease_name': disease_name,
                                'disease_id': disease_id,
                                'score': score,
                                'source': 'DisGeNET_curated_github_mirror',
                            })
                            if target_disease == 'CIRI':
                                ciri_rows.append({
                                    'disease': target_disease,
                                    'gene': gene,
                                    'score': score,
                                    'disease_id': disease_id,
                                    'disease_name': disease_name,
                                    'target_name': row.get('geneName', ''),
                                })
                        break

    # Save disgenet_disease_genes.csv
    out_file = OUT_DIR / "disgenet_disease_genes.csv"
    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['disease', 'gene', 'disease_name', 'disease_id', 'score', 'source'])
        writer.writeheader()
        writer.writerows(rows)

    # Save disgenet_ciri_genes.csv
    ciri_file = OUT_DIR / "disgenet_ciri_genes.csv"
    with open(ciri_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['disease', 'gene', 'score', 'disease_id', 'disease_name', 'target_name'])
        writer.writeheader()
        writer.writerows(ciri_rows)

    print(f"DisGeNET curated parsed: {len(rows)} associations")
    for d, genes in disease_hits.items():
        print(f"  {d}: {len(genes)} genes")
    print(f"Saved: {out_file}")
    print(f"Saved: {ciri_file}")

    return out_file, ciri_file


if __name__ == "__main__":
    parse_disgenet_curated(score_threshold=0.0)
