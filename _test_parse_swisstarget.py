"""Parse SwissTargetPrediction result HTML."""
from __future__ import annotations

import re
import traceback

import pandas as pd

HTML_PATH = "_swisstarget_result.html"


def parse_result_html(path: str) -> pd.DataFrame | None:
    tables = pd.read_html(path)
    print("tables found:", len(tables))
    for i, t in enumerate(tables):
        print(f"\n=== table {i} shape {t.shape} ===")
        print(t.head(3))
    # The target prediction table usually has columns with 'Target', 'Common name', 'Uniprot ID'
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("target" in c for c in cols) and any("common" in c for c in cols):
            return t
    return None


def parse_with_beautifulsoup(path: str) -> list[dict]:
    from bs4 import BeautifulSoup

    text = open(path, encoding="utf-8").read()
    soup = BeautifulSoup(text, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if len(tds) < 7:
            continue
        cols = [td.get_text(strip=True) for td in tds]
        if "Target" in cols[0] or "target" in cols[0].lower():
            continue
        # Probability is in a span with opacity 0.5
        prob_span = tds[5].find("span", style=re.compile(r"opacity"))
        probability = prob_span.get_text(strip=True) if prob_span else cols[5]
        rows.append({
            "target_name": cols[0],
            "gene_symbol": cols[1],
            "uniprot_id": cols[2],
            "chembl_id": cols[3],
            "target_class": cols[4],
            "probability": probability,
            "known_actives": cols[6],
        })
    return rows


def main() -> int:
    df = parse_result_html(HTML_PATH)
    if df is not None:
        print("\n=== parsed table ===")
        print(df.head())
        print("shape:", df.shape)

    rows = parse_with_beautifulsoup(HTML_PATH)
    print("\n=== beautifulsoup rows ===")
    print("count:", len(rows))
    for r in rows[:5]:
        print(r)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
