"""
Fetch real PubChem data for BBB-permeable, ferroptosis-related TCM monomers
and append them to compound_smiles.csv / tcm_monomers_pubchem.csv.

Selection criteria (based on literature + RDKit rules):
- BBB-permeable: MW<400, logP 1-4, HBD<=3, HBA<=8, TPSA<=90
- Ferroptosis-related: reported modulation of GPX4, Nrf2, HO-1, lipid peroxidation, iron metabolism
- Source: traditional Chinese herbs or commonly used TCM prescriptions

New compounds:
  Baicalein   (CID 5281605) - Scutellaria baicalensis, flavone, Nrf2/GPX4 activator
  Wogonin     (CID 5281703) - Scutellaria baicalensis, flavone, neuroprotective
  Luteolin    (CID 5280445) - widespread flavone, antioxidant/anti-ferroptosis
  Kaempferol  (CID 5280863) - widespread flavonol, antioxidant
  Hesperetin  (CID 72281)   - Citrus peel flavanone, anti-inflammatory
  Resveratrol (CID 445154)  - Polygonum cuspidatum, sirtuin activator, anti-ferroptosis
"""

import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
NETWORK_DIR = BASE_DIR / "network_files"

NEW_COMPOUNDS = {
    "Baicalein": 5281605,
    "Wogonin": 5281703,
    "Luteolin": 5280445,
    "Kaempferol": 5280863,
    "Hesperetin": 72281,
    "Resveratrol": 445154,
}

NEW_TCM_INFO = {
    "Baicalein": {
        "source": "Scutellaria baicalensis (Huangqin)",
        "note": "flavone aglycone, strong Nrf2/GPX4-mediated anti-ferroptosis, BBB-permeable",
    },
    "Wogonin": {
        "source": "Scutellaria baicalensis (Huangqin)",
        "note": "flavone, neuroprotective, anti-inflammatory, BBB-permeable",
    },
    "Luteolin": {
        "source": "Artemisia argyi (Aiye) / multiple herbs",
        "note": "flavone, antioxidant, inhibits lipid peroxidation and ferroptosis",
    },
    "Kaempferol": {
        "source": "Artemisia argyi (Aiye) / multiple herbs",
        "note": "flavonol antioxidant, modulates ferroptosis markers",
    },
    "Hesperetin": {
        "source": "Citrus reticulata (Chenpi)",
        "note": "flavanone, neuroprotective, anti-inflammatory, BBB-permeable",
    },
    "Resveratrol": {
        "source": "Polygonum cuspidatum (Huzhang)",
        "note": "polyphenol, sirtuin activator, anti-ferroptosis via Nrf2/HO-1",
    },
}


def _safe_request(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1.0)


def fetch_pubchem_properties(cid: int) -> dict:
    """Fetch CanonicalSMILES, MolecularFormula, MolecularWeight, XLogP, HBD, HBA, TPSA, RotatableBondCount."""
    props = "CanonicalSMILES,MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA,RotatableBondCount"
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{props}/JSON"
    data = json.loads(_safe_request(url))
    return data["PropertyTable"]["Properties"][0]


def main():
    smiles_rows = []
    tcm_rows = []
    for name, cid in NEW_COMPOUNDS.items():
        print(f"Fetching {name} (CID {cid})...")
        props = fetch_pubchem_properties(cid)
        info = NEW_TCM_INFO[name]
        smiles_rows.append({
            "compound": name,
            "cid": cid,
            "CanonicalSMILES": props.get("CanonicalSMILES", props.get("ConnectivitySMILES", "")),
        })
        tcm_rows.append({
            "compound": name,
            "cid": cid,
            "source": info["source"],
            "note": info["note"],
            "SMILES": props.get("CanonicalSMILES", props.get("ConnectivitySMILES", "")),
            "ConnectivitySMILES": props.get("ConnectivitySMILES", ""),
            "MolecularFormula": props.get("MolecularFormula", ""),
            "MolecularWeight": props.get("MolecularWeight", ""),
            "XLogP": props.get("XLogP", ""),
            "HBondDonorCount": props.get("HBondDonorCount", ""),
            "HBondAcceptorCount": props.get("HBondAcceptorCount", ""),
            "TPSA": props.get("TPSA", ""),
            "RotatableBondCount": props.get("RotatableBondCount", ""),
        })
        time.sleep(0.5)

    # Append to compound_smiles.csv
    smiles_file = NETWORK_DIR / "compound_smiles.csv"
    smiles_df = pd.read_csv(smiles_file)
    new_smiles_df = pd.DataFrame(smiles_rows)
    smiles_df = pd.concat([smiles_df, new_smiles_df], ignore_index=True)
    smiles_df.to_csv(smiles_file, index=False)
    print(f"Updated {smiles_file}: {len(smiles_df)} compounds")

    # Append to tcm_monomers_pubchem.csv
    tcm_file = NETWORK_DIR / "tcm_monomers_pubchem.csv"
    tcm_df = pd.read_csv(tcm_file)
    new_tcm_df = pd.DataFrame(tcm_rows)
    tcm_df = pd.concat([tcm_df, new_tcm_df], ignore_index=True)
    tcm_df.to_csv(tcm_file, index=False)
    print(f"Updated {tcm_file}: {len(tcm_df)} TCM monomers")


if __name__ == "__main__":
    main()
