"""Restore compound-related CSVs to the original 20-compound state before re-expansion."""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent
NETWORK_DIR = BASE_DIR / "network_files"
FP_DIR = BASE_DIR / "L4" / "药物指纹"

ORIGINAL_20 = [
    "BCP", "VC", "Fer-1", "DFO", "Lip-1",
    "Erastin", "RSL3", "ML162",
    "Cinnamaldehyde", "Cinnamic_acid", "Quercetin", "Naringenin", "Eucalyptol", "Borneol",
    "Baicalein", "Wogonin", "Luteolin", "Kaempferol", "Hesperetin", "Resveratrol"
]

# 1. compound_smiles.csv
smiles_df = pd.read_csv(NETWORK_DIR / "compound_smiles.csv")
smiles_df = smiles_df[smiles_df["compound"].isin(ORIGINAL_20)]
smiles_df.to_csv(NETWORK_DIR / "compound_smiles.csv", index=False)
print(f"Restored compound_smiles.csv: {len(smiles_df)} compounds")

# 2. pubchem_compound_props.csv
props_df = pd.read_csv(NETWORK_DIR / "pubchem_compound_props.csv")
name_col = "compound" if "compound" in props_df.columns else "Compound"
props_df = props_df[props_df[name_col].isin(ORIGINAL_20)]
props_df.to_csv(NETWORK_DIR / "pubchem_compound_props.csv", index=False)
print(f"Restored pubchem_compound_props.csv: {len(props_df)} compounds")

# 3. tcm_monomers_pubchem.csv: keep first 13 rows (header + 12 original TCM monomers)
tcm_df = pd.read_csv(NETWORK_DIR / "tcm_monomers_pubchem.csv")
tcm_df = tcm_df.iloc[:13]
tcm_df.to_csv(NETWORK_DIR / "tcm_monomers_pubchem.csv", index=False)
print(f"Restored tcm_monomers_pubchem.csv: {len(tcm_df)} rows")

# 4. drug_descriptors.csv
desc_df = pd.read_csv(FP_DIR / "drug_descriptors.csv")
desc_df = desc_df[desc_df["Compound"].isin(ORIGINAL_20)]
desc_df.to_csv(FP_DIR / "drug_descriptors.csv", index=False)
print(f"Restored drug_descriptors.csv: {len(desc_df)} compounds")

# 5. fingerprint CSVs
fp_configs = {
    "morgan": (FP_DIR / "drug_fingerprints_morgan.csv", 2048),
    "atompair": (FP_DIR / "drug_fingerprints_atompair.csv", 2048),
    "maccs": (FP_DIR / "drug_fingerprints_maccs.csv", 167),
    "rdkit": (FP_DIR / "drug_fingerprints_rdkit.csv", 2048),
}
for fp_name, (fp_file, n_bits) in fp_configs.items():
    fp_df = pd.read_csv(fp_file)
    fp_df = fp_df[fp_df["Compound"].isin(ORIGINAL_20)]
    fp_df.to_csv(fp_file, index=False)
    print(f"Restored {fp_file.name}: {len(fp_df)} compounds")
