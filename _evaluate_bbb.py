"""
Evaluate Blood-Brain Barrier (BBB) permeability of current compounds
using RDKit molecular descriptors and empirical rules.

Selection logic (practical decisions for this project):
- BBB is the first gate for TCM monomers; only BBB-Likely/Moderate compounds are kept.
- "Moderate" flavonoids (Quercetin, Luteolin, Kaempferol) are retained because they have
  documented neuroprotection / anti-ferroptosis effects in vivo, suggesting partial CNS
  penetration despite higher TPSA/HBD (network-pharmacology screening stage).
- Strictly BBB-poor compounds (Saikosaponin_A, Paeoniflorin, Baicalin, Chlorogenic_acid)
  have already been removed.

References / GitHub repos consulted:
- Lipinski et al. (1997) general ADME rules
- Pardridge WM. (2005) The blood-brain barrier: bottleneck in brain drug development.
  NeuroRx 2(1):3-14. doi:10.1602/neurorx.2.1.3
- Wager TT et al. (2010) Central nervous system multiparameter optimization desirability:
  application in analgesic drug discovery. ACS Chem Neurosci 1(6):435-449.
- B3DB benchmark: github.com/theochem/B3DB
- BBB-Permeability-Prediction: github.com/ActisDa/BBB-Permeability-Prediction
- ML BBB classifiers: github.com/orvelte/BBB_permeability, github.com/ScriptsOfLife/SOL_BBB
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

BASE_DIR = Path(__file__).parent
compound_file = BASE_DIR / "network_files" / "compound_smiles.csv"
out_file = BASE_DIR / "network_files" / "tcm_bbb_evaluation.csv"


def evaluate_bbb(smiles: str):
    """Return BBB descriptors and a simple pass/fail score."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Empirical BBB rules (combined from multiple literature rules)
    score = 0
    reasons = []
    if mw <= 400:
        score += 1
    else:
        reasons.append(f"MW={mw:.1f}>400")
    if 1 <= logp <= 4:
        score += 1
    else:
        reasons.append(f"logP={logp:.2f} outside [1,4]")
    if hbd <= 3:
        score += 1
    else:
        reasons.append(f"HBD={hbd}>3")
    if hba <= 8:
        score += 1
    else:
        reasons.append(f"HBA={hba}>8")
    if tpsa <= 90:
        score += 1
    else:
        reasons.append(f"TPSA={tpsa:.1f}>90")

    # Overall judgment
    if score >= 4:
        bbb_pass = "Likely"
    elif score >= 3:
        bbb_pass = "Moderate"
    else:
        bbb_pass = "Poor"

    return {
        "MW": mw,
        "LogP": logp,
        "HBD": hbd,
        "HBA": hba,
        "TPSA": tpsa,
        "RotBonds": rb,
        "BBB_score": score,
        "BBB_pass": bbb_pass,
        "BBB_reasons": "; ".join(reasons) if reasons else "",
    }


def main():
    df = pd.read_csv(compound_file)
    results = []
    for _, row in df.iterrows():
        comp = row["compound"]
        smi = row["CanonicalSMILES"]
        res = evaluate_bbb(smi)
        if res is None:
            continue
        res["compound"] = comp
        res["CanonicalSMILES"] = smi
        results.append(res)

    out_df = pd.DataFrame(results)
    out_df = out_df[
        ["compound", "CanonicalSMILES", "MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds", "BBB_score", "BBB_pass", "BBB_reasons"]
    ]
    out_df = out_df.sort_values(by=["BBB_pass", "BBB_score"], ascending=[False, False])
    out_df.to_csv(out_file, index=False)
    print(out_df.to_string(index=False))
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
