"""
Append PubChem properties for newly added TCM monomers to pubchem_compound_props.csv.
Properties are sourced from real PubChem PUG-REST (already fetched in _add_bbb_ferroptosis_tcm.py).
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
NETWORK_DIR = BASE_DIR / "network_files"

# Read the new TCM monomers (last 6 rows added)
tcm_df = pd.read_csv(NETWORK_DIR / "tcm_monomers_pubchem.csv")
new_names = ["Baicalein", "Wogonin", "Luteolin", "Kaempferol", "Hesperetin", "Resveratrol"]
new_df = tcm_df[tcm_df["compound"].isin(new_names)].copy()

# Map columns to pubchem_compound_props.csv format
new_props = pd.DataFrame({
    "compound": new_df["compound"],
    "cid": new_df["cid"],
    "MW": new_df["MolecularWeight"],
    "LogP": new_df["XLogP"],
    "HBD": new_df["HBondDonorCount"],
    "HBA": new_df["HBondAcceptorCount"],
    "TPSA": new_df["TPSA"],
    "RotB": new_df["RotatableBondCount"],
    "SMILES": new_df["SMILES"],
    "IUPACName": "",  # not fetched in this batch
    "MolecularFormula": new_df["MolecularFormula"],
    "source": "PubChem_PUG_REST",
})

props_df = pd.read_csv(NETWORK_DIR / "pubchem_compound_props.csv")
props_df = pd.concat([props_df, new_props], ignore_index=True)
props_df.to_csv(NETWORK_DIR / "pubchem_compound_props.csv", index=False)
print(f"Updated {NETWORK_DIR / 'pubchem_compound_props.csv'}: {len(props_df)} compounds")
print(new_props.to_string(index=False))
