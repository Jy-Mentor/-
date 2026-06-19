"""
Generate real RDKit descriptors and fingerprints for newly added TCM monomers
and append them to the L4/药物指纹/ CSV files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, RDKFingerprint, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect

BASE_DIR = Path(__file__).parent
FP_DIR = BASE_DIR / "L4" / "药物指纹"
NETWORK_DIR = BASE_DIR / "network_files"

NEW_COMPOUNDS = ["Baicalein", "Wogonin", "Luteolin", "Kaempferol", "Hesperetin", "Resveratrol"]


def _compute_fingerprint(smiles: str, fp_type: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        size = 167 if fp_type == "maccs" else n_bits
        return np.zeros(size, dtype=np.int8)
    if fp_type == "morgan":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    elif fp_type == "atompair":
        fp = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    elif fp_type == "maccs":
        fp = MACCSkeys.GenMACCSKeys(mol)
        n_bits = 167
    elif fp_type == "rdkit":
        fp = RDKFingerprint(mol, fpSize=n_bits)
    else:
        return np.zeros(n_bits, dtype=np.int8)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _compute_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "NumAtoms": mol.GetNumAtoms(),
    }


def main():
    smiles_df = pd.read_csv(NETWORK_DIR / "compound_smiles.csv")
    smiles_map = dict(zip(smiles_df["compound"], smiles_df["CanonicalSMILES"]))

    # 1. Update drug_descriptors.csv
    desc_file = FP_DIR / "drug_descriptors.csv"
    desc_df = pd.read_csv(desc_file)
    desc_rows = []
    for comp in NEW_COMPOUNDS:
        smi = smiles_map.get(comp, "")
        d = _compute_descriptors(smi)
        if not d:
            continue
        desc_rows.append({
            "Compound": comp,
            "Category": "TCM单体（BBB/铁死亡补充）",
            "SMILES": smi,
            **d,
        })
    if desc_rows:
        desc_df = pd.concat([desc_df, pd.DataFrame(desc_rows)], ignore_index=True)
        desc_df.to_csv(desc_file, index=False)
        print(f"Updated {desc_file}: {len(desc_df)} compounds")

    # 2. Update fingerprint CSVs
    fp_configs = {
        "morgan": (FP_DIR / "drug_fingerprints_morgan.csv", 2048),
        "atompair": (FP_DIR / "drug_fingerprints_atompair.csv", 2048),
        "maccs": (FP_DIR / "drug_fingerprints_maccs.csv", 167),
        "rdkit": (FP_DIR / "drug_fingerprints_rdkit.csv", 2048),
    }
    for fp_name, (fp_file, n_bits) in fp_configs.items():
        fp_df = pd.read_csv(fp_file)
        cols = [f"{fp_name.capitalize()}_{i}" for i in range(n_bits)]
        rows = []
        for comp in NEW_COMPOUNDS:
            smi = smiles_map.get(comp, "")
            arr = _compute_fingerprint(smi, fp_name, n_bits)
            row = {"Compound": comp}
            for c, v in zip(cols, arr):
                row[c] = int(v)
            rows.append(row)
        if rows:
            fp_df = pd.concat([fp_df, pd.DataFrame(rows)], ignore_index=True)
            fp_df.to_csv(fp_file, index=False)
            print(f"Updated {fp_file}: {len(fp_df)} compounds")


if __name__ == "__main__":
    main()
