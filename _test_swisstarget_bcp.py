"""Quick test: fetch SwissTargetPrediction for BCP only."""
from __future__ import annotations

import traceback
from pathlib import Path

import requests

from fetch_swisstarget_compound_targets import (
    HEADERS,
    load_compounds,
    load_gene_set,
    parse_result_table,
    submit_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    core_genes = load_gene_set(PROJECT_ROOT / "铁衰老基因.txt")
    compounds = load_compounds(PROJECT_ROOT / "network_files" / "compound_smiles.csv")

    bcp = next(c for c in compounds if c["compound"] == "BCP")
    print("Testing BCP:", bcp["smiles"])
    print("Core genes sample:", sorted(list(core_genes))[:20])

    session = requests.Session()
    session.headers.update(HEADERS)
    html = submit_prediction(session, bcp["smiles"])
    if html is None:
        print("No HTML returned")
        return 1

    df = parse_result_table(html)
    print(f"Result table shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Top 20 predictions:")
    print(df.head(20)[["Target", "Common name", "Probability*"]].to_string(index=False))

    # Check which genes are in core set
    matched = df[df["Common name"].str.upper().isin(core_genes)]
    print(f"\nMatched core genes: {len(matched)}")
    if not matched.empty:
        print(matched[["Target", "Common name", "Probability*"]].head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
