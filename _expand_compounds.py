"""
Expand the compound set to 50+ BBB-permeable, ferroptosis/neuroprotection-related
molecules. Resolves names to PubChem CIDs via PUG-REST, fetches real SMILES/properties,
computes RDKit descriptors/fingerprints, evaluates BBB permeability, and updates all
project files.

Selection criteria:
- BBB permeable (Likely or Moderate by RDKit descriptors + literature rules)
- Ferroptosis / neuroprotection / antioxidant / Nrf2-Keap1 pathway evidence
- Natural products / TCM monomers prioritized; a few known CNS drugs included as
  positive controls.

The original 20 compounds (graph_node_config.yaml) are kept as the core validation set.
This script regenerates the compound-related CSVs from scratch to avoid duplicates and
inconsistent columns introduced by earlier partial expansions.

References:
- PubChem PUG-REST: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
- MoleculeNet BBBP benchmark: Wu et al. 2018; DeepChem S3 mirror
- B3DB benchmark: github.com/theochem/B3DB
- BBB-Permeability-Prediction: github.com/ActisDa/BBB-Permeability-Prediction
"""

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, RDKFingerprint
from rdkit.Chem.rdMolDescriptors import (
    CalcNumHBA,
    CalcNumHBD,
    CalcNumRings,
    CalcNumRotatableBonds,
    CalcTPSA,
    GetHashedAtomPairFingerprintAsBitVect,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
NETWORK_DIR = BASE_DIR / "network_files"
FP_DIR = BASE_DIR / "L4" / "药物指纹"
CONFIG_FILE = NETWORK_DIR / "graph_node_config.yaml"

# ---- 原始 20 个核心化合物（保留作为验证集） ----
CORE_COMPOUND_INFO = {
    "BCP": ("Cannabis sativa / synthetic", "ferroptosis modulator (CB2 agonist)", "Likely"),
    "VC": ("Camellia sinensis / dietary", "antioxidant", "Poor"),
    "Fer-1": ("Synthetic", "ferroptosis inhibitor", "Likely"),
    "DFO": ("Synthetic", "iron chelator", "Poor"),
    "Lip-1": ("Synthetic", "ferroptosis inhibitor", "Likely"),
    "Erastin": ("Synthetic", "ferroptosis inducer (system xc-)", "Moderate"),
    "RSL3": ("Synthetic", "GPX4 inhibitor", "Likely"),
    "ML162": ("Synthetic", "GPX4 covalent inhibitor", "Moderate"),
    "Cinnamaldehyde": ("Cinnamomum cassia (Guizhi)", "anti-inflammatory, neuroprotective", "Likely"),
    "Cinnamic_acid": ("Cinnamomum cassia (Guizhi)", "phenolic acid", "Likely"),
    "Quercetin": ("Artemisia argyi (Aiye)", "Nrf2/HO-1 anti-ferroptosis", "Moderate"),
    "Naringenin": ("Artemisia argyi (Aiye)", "antioxidant", "Likely"),
    "Eucalyptol": ("Artemisia argyi (Aiye) essential oil", "volatile oil, anti-inflammatory", "Likely"),
    "Borneol": ("Artemisia argyi (Aiye)", "BBB penetration enhancer", "Likely"),
    "Baicalein": ("Scutellaria baicalensis (Huangqin)", "Nrf2/GPX4 anti-ferroptosis", "Likely"),
    "Wogonin": ("Scutellaria baicalensis (Huangqin)", "neuroprotective, anti-inflammatory", "Likely"),
    "Luteolin": ("Artemisia argyi (Aiye) / multiple herbs", "anti-ferroptosis, antioxidant", "Moderate"),
    "Kaempferol": ("Artemisia argyi (Aiye) / multiple herbs", "antioxidant, modulates ferroptosis", "Moderate"),
    "Hesperetin": ("Citrus reticulata (Chenpi)", "neuroprotective, anti-inflammatory", "Likely"),
    "Resveratrol": ("Polygonum cuspidatum (Huzhang)", "Nrf2/HO-1 anti-ferroptosis", "Likely"),
}

# ---- 新增化合物：name -> (source_herb, mechanism_note, bbb_tier) ----
# CID 由 PubChem name search 自动解析；不再硬编码 CID。
NEW_COMPOUNDS = {
    # 多酚/黄酮类 TCM 单体
    "Curcumin": ("Curcuma longa (Jianghuang)", "Nrf2/HO-1 anti-ferroptosis", "Likely"),
    "Ferulic acid": ("Angelica sinensis / Cinnamomum", "antioxidant, neuroprotective", "Likely"),
    "Pinocembrin": ("Pinus / Propolis", "anti-inflammatory, BBB protective", "Likely"),
    "Chrysin": ("Passiflora / Oroxylum", "antioxidant", "Likely"),
    "Apigenin": ("Apium / Artemisia", "Nrf2 anti-oxidative stress", "Moderate"),
    "Genistein": ("Glycine max / Pueraria", "Nrf2/HO-1, anti-ferroptosis", "Moderate"),
    "Daidzein": ("Glycine max / Pueraria", "antioxidant, estrogenic", "Likely"),
    "Formononetin": ("Astragalus / Trifolium", "neuroprotective", "Likely"),
    "Oroxylin A": ("Scutellaria baicalensis", "anti-inflammatory, BBB", "Likely"),
    "Scutellarein": ("Scutellaria baicalensis", "antioxidant", "Moderate"),
    "Tangeretin": ("Citrus reticulata (Chenpi)", "neuroprotective", "Likely"),
    "Nobiletin": ("Citrus reticulata (Chenpi)", "neuroprotective, anti-inflammatory", "Likely"),
    "Fisetin": ("Rhus / Cotinus", "senolytic, Nrf2, neuroprotective", "Moderate"),
    "Galangin": ("Alpinia officinarum (Gaoliangjiang)", "antioxidant", "Moderate"),
    "Phloretin": ("Malus pumila (Pingguo)", "anti-oxidative stress", "Moderate"),
    "Catechin": ("Camellia sinensis (Cha)", "antioxidant, metal chelation", "Moderate"),
    "Epicatechin": ("Camellia sinensis (Cha)", "antioxidant", "Moderate"),
    # 萜类/醌类/挥发油 TCM 单体
    "Tetramethylpyrazine": ("Ligusticum chuanxiong (Chuanxiong)", "anti-ischemic, BBB", "Likely"),
    "Tanshinone IIA": ("Salvia miltiorrhiza (Danshen)", "anti-ferroptosis, anti-apoptotic", "Likely"),
    "Danshensu": ("Salvia miltiorrhiza (Danshen)", "antioxidant, anti-ischemic", "Likely"),
    "Honokiol": ("Magnolia officinalis (Houpo)", "anti-inflammatory, BBB", "Likely"),
    "Magnolol": ("Magnolia officinalis (Houpo)", "anti-inflammatory, antioxidant", "Likely"),
    "Schisandrin": ("Schisandra chinensis (Wuweizi)", "hepatoprotective, antioxidant", "Likely"),
    "Schisandrin B": ("Schisandra chinensis (Wuweizi)", "anti-oxidative stress", "Likely"),
    "Dihydroartemisinin": ("Artemisia annua (Qinghao)", "anti-malarial, ferroptosis inducer", "Likely"),
    "Artemisinin": ("Artemisia annua (Qinghao)", "endoperoxide, ferroptosis-related", "Likely"),
    "Shikonin": ("Lithospermum erythrorhizon (Zicao)", "ferroptosis inducer / anti-tumor", "Moderate"),
    "Plumbagin": ("Plumbago zeylanica (Baihua Dan)", "ferroptosis inducer / anti-cancer", "Likely"),
    # 蒽醌类（大黄/何首乌）
    "Emodin": ("Rheum palmatum (Dahuang)", "anti-inflammatory, antioxidant", "Moderate"),
    "Aloe emodin": ("Aloe vera / Rheum", "anti-cancer, anti-oxidant", "Moderate"),
    "Physcion": ("Rheum palmatum", "anti-inflammatory", "Likely"),
    "Chrysophanol": ("Rheum palmatum", "anti-inflammatory", "Likely"),
    # 苯乙醇苷/木脂素
    "Salidroside": ("Rhodiola rosea (Hongjingtian)", "anti-hypoxia, neuroprotective", "Moderate"),
    "Tyrosol": ("Olea europaea / Rhodiola", "antioxidant", "Likely"),
    "Hydroxytyrosol": ("Olea europaea (Ganlan)", "potent antioxidant", "Likely"),
    # 含硫/异硫氰酸酯
    "Sulforaphane": ("Brassica oleracea / Wasabia", "Nrf2 activator, anti-ferroptosis", "Likely"),
    "Dimethyl fumarate": ("Synthetic / Fumaria", "Nrf2 activator, MS drug", "Likely"),
    # 已知 CNS 神经保护剂/抗氧化剂（作为阳性对照）
    "Edaravone": ("Synthetic", "radical scavenger, stroke drug", "Likely"),
    "Idebenone": ("Synthetic (CoQ10 analog)", "mitochondrial antioxidant", "Likely"),
    "Alpha-lipoic acid": ("Synthetic / endogenous", "antioxidant, mitochondrial", "Likely"),
    "Melatonin": ("Pineal / endogenous", "antioxidant, mitochondrial", "Likely"),
    "Huperzine A": ("Huperzia serrata (QianCengTa)", "AChE inhibitor, neuroprotective", "Likely"),
    "N-acetylcysteine": ("Synthetic / precursor", "GSH precursor, antioxidant", "Likely"),
}


def _pubchem_request(url: str, retries: int = 3) -> dict:
    """Make a PubChem PUG-REST request with exponential backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            last_err = e
            logger.warning(f"PubChem request failed (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2 ** attempt)
    raise last_err


def _resolve_cid(name: str) -> int:
    """Resolve a compound name to the first CID via PubChem PUG-REST."""
    encoded = urllib.parse.quote(name)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/cids/JSON"
    data = _pubchem_request(url)
    return int(data["IdentifierList"]["CID"][0])


def _pubchem_batch_properties(cids: list) -> dict:
    """Fetch SMILES, formula, weight, logP, HBD, HBA, TPSA, rotatable bonds."""
    cid_str = ",".join(str(c) for c in cids)
    props = (
        "IsomericSMILES,CanonicalSMILES,SMILES,IUPACName,MolecularFormula,"
        "MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,TPSA,RotatableBondCount"
    )
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/{props}/JSON"
    data = _pubchem_request(url)
    props = data["PropertyTable"]["Properties"]
    return {p["CID"]: p for p in props}


def _compute_fingerprint(
    smiles: str, fp_type: str, n_bits: int = 2048, compound_name: str | None = None
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        size = 167 if fp_type == "maccs" else n_bits
        logger.warning(
            "化合物 %s 的 SMILES 解析失败, %s 指纹使用零向量 (SMILES=%s)",
            compound_name or "UNKNOWN",
            fp_type,
            smiles[:50],
        )
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
        logger.warning(
            "化合物 %s 遇到未知指纹类型 %s, 使用零向量 (SMILES=%s)",
            compound_name or "UNKNOWN",
            fp_type,
            smiles[:50],
        )
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
        "HBA": CalcNumHBA(mol),
        "HBD": CalcNumHBD(mol),
        "RotBonds": CalcNumRotatableBonds(mol),
        "TPSA": CalcTPSA(mol),
        "RingCount": CalcNumRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "NumAtoms": mol.GetNumAtoms(),
    }


def _evaluate_bbb(smiles: str):
    """BBB permeability scoring using RDKit descriptors + empirical rules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = CalcNumHBD(mol)
    hba = CalcNumHBA(mol)
    tpsa = CalcTPSA(mol)
    rb = CalcNumRotatableBonds(mol)

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


def _load_existing_smiles():
    """Load the existing core compound SMILES file."""
    smiles_file = NETWORK_DIR / "compound_smiles.csv"
    records = []
    if smiles_file.exists():
        df = pd.read_csv(smiles_file)
        for _, row in df.iterrows():
            comp = str(row.get("compound", "")).strip()
            cid = row.get("cid")
            smi = str(row.get("CanonicalSMILES", "") or row.get("SMILES", "")).strip()
            if comp and smi:
                records.append({"compound": comp, "cid": int(cid) if pd.notna(cid) else None, "smiles": smi})
    return records


def main():
    # 1. Build complete compound list
    existing = _load_existing_smiles()
    existing_names = {r["compound"] for r in existing}

    records = []
    for r in existing:
        comp = r["compound"]
        source, mechanism, bbb = CORE_COMPOUND_INFO.get(
            comp, ("", "", "Likely")
        )
        records.append({
            "compound": comp,
            "cid": r["cid"],
            "smiles": r["smiles"],
            "source_herb": source,
            "mechanism": mechanism,
            "bbb_tier": bbb,
            "category": "核心验证集",
        })

    logger.info(f"Core compounds from compound_smiles.csv: {len(records)}")

    # Resolve and fetch new compounds
    new_records = []
    failed = []
    for name, (source, mechanism, bbb) in NEW_COMPOUNDS.items():
        if name in existing_names:
            logger.warning(f"Skipping {name}: already in core compound list")
            continue
        try:
            cid = _resolve_cid(name)
            new_records.append({
                "compound": name,
                "cid": cid,
                "smiles": "",
                "source_herb": source,
                "mechanism": mechanism,
                "bbb_tier": bbb,
                "category": "BBB/铁死亡扩展集",
            })
        except Exception as e:
            logger.error(f"Failed to resolve CID for {name}: {e}")
            failed.append(name)
        time.sleep(0.25)

    logger.info(f"Resolved {len(new_records)} new compounds; {len(failed)} failed")
    records.extend(new_records)

    # 2. Fetch PubChem properties in batches
    all_cids = [r["cid"] for r in records if r["cid"] is not None]
    props = {}
    for i in range(0, len(all_cids), 100):
        batch = all_cids[i:i + 100]
        try:
            batch_props = _pubchem_batch_properties(batch)
            props.update(batch_props)
            logger.info(f"Fetched properties for CIDs {i + 1}-{min(i + 100, len(all_cids))}")
        except Exception as e:
            logger.error(f"Batch property fetch failed: {e}")
        time.sleep(0.5)

    # Fill SMILES and properties
    for r in records:
        cid = r["cid"]
        p = props.get(cid, {})
        if not r["smiles"]:
            smi = p.get("IsomericSMILES") or p.get("CanonicalSMILES") or p.get("SMILES")
            r["smiles"] = str(smi).strip() if smi else ""
        r["prop"] = p

    # Drop compounds without SMILES
    valid_records = [r for r in records if r["smiles"]]
    if len(valid_records) < len(records):
        logger.warning(f"Dropped {len(records) - len(valid_records)} compounds with missing SMILES")
    records = valid_records

    logger.info(f"Total compounds after PubChem fetch: {len(records)}")

    # 3. Write compound_smiles.csv
    smiles_rows = []
    for r in records:
        smiles_rows.append({
            "compound": r["compound"],
            "cid": r["cid"],
            "CanonicalSMILES": r["smiles"],
        })
    pd.DataFrame(smiles_rows).to_csv(NETWORK_DIR / "compound_smiles.csv", index=False)
    logger.info(f"Wrote compound_smiles.csv: {len(smiles_rows)} compounds")

    # 4. Write pubchem_compound_props.csv (columns satisfy both _load_compound_props and _load_drug_fingerprints)
    prop_rows = []
    for r in records:
        p = r.get("prop", {})
        mw = p.get("MolecularWeight")
        logp = p.get("XLogP")
        hbd = p.get("HBondDonorCount")
        hba = p.get("HBondAcceptorCount")
        tpsa = p.get("TPSA")
        rotb = p.get("RotatableBondCount")
        row = {
            "compound": r["compound"],
            "cid": r["cid"],
            "MW": mw,
            "LogP": logp,
            "HBD": hbd,
            "HBA": hba,
            "TPSA": tpsa,
            "RotB": rotb,
            "SMILES": r["smiles"],
            "IUPACName": p.get("IUPACName", ""),
            "MolecularFormula": p.get("MolecularFormula", ""),
            "source": r["source_herb"],
            # Lowercase aliases used by _load_drug_fingerprints
            "molecular_weight": mw,
            "xlogp": logp,
            "hbd": hbd,
            "hba": hba,
            "tpsa": tpsa,
            "rotbonds": rotb,
        }
        prop_rows.append(row)
    pd.DataFrame(prop_rows).to_csv(NETWORK_DIR / "pubchem_compound_props.csv", index=False)
    logger.info(f"Wrote pubchem_compound_props.csv: {len(prop_rows)} compounds")

    # 5. Write tcm_monomers_pubchem.csv
    tcm_rows = []
    for r in records:
        p = r.get("prop", {})
        tcm_rows.append({
            "compound": r["compound"],
            "cid": r["cid"],
            "source": r["source_herb"],
            "note": r["mechanism"],
            "SMILES": r["smiles"],
            "ConnectivitySMILES": r["smiles"],
            "MolecularFormula": p.get("MolecularFormula", ""),
            "MolecularWeight": p.get("MolecularWeight", ""),
            "XLogP": p.get("XLogP", ""),
            "HBondDonorCount": p.get("HBondDonorCount", ""),
            "HBondAcceptorCount": p.get("HBondAcceptorCount", ""),
            "TPSA": p.get("TPSA", ""),
            "RotatableBondCount": p.get("RotatableBondCount", ""),
            "source_herb": r["source_herb"],
            "mechanism": r["mechanism"],
            "bbb_tier": r["bbb_tier"],
            "CanonicalSMILES": r["smiles"],
        })
    pd.DataFrame(tcm_rows).to_csv(NETWORK_DIR / "tcm_monomers_pubchem.csv", index=False)
    logger.info(f"Wrote tcm_monomers_pubchem.csv: {len(tcm_rows)} compounds")

    # 6. Write L4 drug_descriptors.csv (regenerate for all compounds)
    desc_rows = []
    for r in records:
        d = _compute_descriptors(r["smiles"])
        if not d:
            logger.warning(f"RDKit parse failure for {r['compound']}; using zero descriptors")
            d = {
                k: 0
                for k in [
                    "MolWt", "LogP", "HBA", "HBD", "RotBonds", "TPSA",
                    "RingCount", "HeavyAtomCount", "NumAtoms",
                ]
            }
        desc_rows.append({
            "Compound": r["compound"],
            "Category": r["category"],
            "SMILES": r["smiles"],
            **d,
        })
    pd.DataFrame(desc_rows).to_csv(FP_DIR / "drug_descriptors.csv", index=False)
    logger.info(f"Wrote drug_descriptors.csv: {len(desc_rows)} compounds")

    # 7. Write L4 fingerprint files (regenerate for all compounds)
    fp_configs = {
        "morgan": (FP_DIR / "drug_fingerprints_morgan.csv", 2048),
        "atompair": (FP_DIR / "drug_fingerprints_atompair.csv", 2048),
        "maccs": (FP_DIR / "drug_fingerprints_maccs.csv", 167),
        "rdkit": (FP_DIR / "drug_fingerprints_rdkit.csv", 2048),
    }
    for fp_name, (fp_file, n_bits) in fp_configs.items():
        cols = [f"{fp_name.capitalize()}_{i}" for i in range(n_bits)]
        rows = []
        for r in records:
            arr = _compute_fingerprint(r["smiles"], fp_name, n_bits, compound_name=r["compound"])
            fp_row = {"Compound": r["compound"]}
            for c, v in zip(cols, arr):
                fp_row[c] = int(v)
            rows.append(fp_row)
        pd.DataFrame(rows).to_csv(fp_file, index=False)
        logger.info(f"Wrote {fp_file.name}: {len(rows)} compounds")

    # 8. Evaluate BBB for all compounds
    bbb_rows = []
    for r in records:
        res = _evaluate_bbb(r["smiles"])
        if res is None:
            continue
        res["compound"] = r["compound"]
        res["CanonicalSMILES"] = r["smiles"]
        bbb_rows.append(res)
    bbb_df = pd.DataFrame(bbb_rows)
    if not bbb_df.empty:
        bbb_df = bbb_df[
            ["compound", "CanonicalSMILES", "MW", "LogP", "HBD", "HBA", "TPSA", "RotBonds",
             "BBB_score", "BBB_pass", "BBB_reasons"]
        ]
        bbb_df = bbb_df.sort_values(by=["BBB_pass", "BBB_score"], ascending=[False, False])
        bbb_df.to_csv(NETWORK_DIR / "tcm_bbb_evaluation.csv", index=False)
        logger.info(f"Wrote tcm_bbb_evaluation.csv: {len(bbb_df)} compounds")
        logger.info("BBB distribution:\n" + str(bbb_df["BBB_pass"].value_counts()))

    # 9. Update graph_node_config.yaml compounds list
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        yaml.preserve_quotes = True
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = yaml.load(f)
        cfg["compounds"] = [r["compound"] for r in records]
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        logger.info(f"Updated {CONFIG_FILE.name} compounds list: {len(cfg['compounds'])} compounds")
    except Exception as e:
        logger.warning(f"Could not update {CONFIG_FILE.name}: {e}")

    logger.info("Compound expansion complete.")


if __name__ == "__main__":
    main()
