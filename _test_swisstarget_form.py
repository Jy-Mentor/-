"""Test SwissTargetPrediction full submission + result retrieval."""
from __future__ import annotations

import re
import time
import traceback

import requests

BASE = "https://www.swisstargetprediction.ch"
INDEX = f"{BASE}/index.php"
PREDICT = f"{BASE}/predict.php"
SMILES = "CC(=O)Nc1ccc(O)c(C)c1"

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
    "Origin": BASE,
    "Referer": INDEX,
}


def extract_job_url(text: str) -> str | None:
    m = re.search(r'location\.replace\(["\'](https://www\.swisstargetprediction\.ch/result\.php\?[^"\']+)["\']\)', text)
    if m:
        return m.group(1)
    m = re.search(r'result\.php\?job=(\d+)&organism=([^"\'\s]+)', text)
    if m:
        return f"{BASE}/result.php?job={m.group(1)}&organism={m.group(2)}"
    return None


def main() -> int:
    s = requests.Session()
    s.headers.update(HEADERS)
    r1 = s.get(INDEX, timeout=30)
    print("GET index:", r1.status_code)

    payload = {"organism": "Homo_sapiens", "smiles": SMILES, "ioi": "2"}
    r2 = s.post(PREDICT, data=payload, timeout=180, allow_redirects=False)
    print("POST predict:", r2.status_code, "len:", len(r2.text))

    job_url = extract_job_url(r2.text)
    print("job_url:", job_url)

    if job_url:
        # Wait a bit for computation
        time.sleep(3)
        r3 = s.get(job_url, timeout=120)
        print("GET result:", r3.status_code, "len:", len(r3.text))
        with open("_swisstarget_result.html", "w", encoding="utf-8") as f:
            f.write(r3.text)
        print("saved to _swisstarget_result.html")

        # Search for result table patterns
        for pat in [r"<table[^>]*>", r"Probability", r"commonName", r"UniProt", r"Target", r"Similarity"]:
            matches = re.findall(pat, r3.text, re.I)
            if matches:
                print(f"  '{pat}': {len(matches)} matches")

        # Print text around result keywords
        text = re.sub(r"<[^>]+>", " ", r3.text)
        for keyword in ["probability", "common name", "uniprot", "target name"]:
            idx = text.lower().find(keyword)
            if idx != -1:
                print(f"  context '{keyword}':", text[idx-80:idx+120].replace("\n", " "))
    else:
        print("No job URL found")
        with open("_swisstarget_predict_response.html", "w", encoding="utf-8") as f:
            f.write(r2.text)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
