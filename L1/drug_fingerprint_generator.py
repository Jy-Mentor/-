#!/usr/bin/env python3
"""
药物分子指纹生成脚本
===================================================
基于SMILES生成多种类型的分子指纹（Morgan、MACCS、RDKit拓扑指纹、Atom Pair），
并保存为CSV文件。

化合物列表：
  - Erastin（铁死亡诱导剂）
  - Fer-1（Ferrostatin-1，铁死亡抑制剂）
  - 槲皮素（Quercetin，铁死亡调节剂）
  - DFO（Deferoxamine，铁螯合剂）
  - 维生素C（Ascorbic Acid，抗氧化剂）

输出：
  - drug_fingerprints_morgan.csv    (Morgan/ECFP4, 2048-bit)
  - drug_fingerprints_maccs.csv     (MACCS Keys, 167-bit)
  - drug_fingerprints_rdkit.csv     (RDKit Topological, 2048-bit)
  - drug_fingerprints_atompair.csv  (Atom Pair)
  - drug_fingerprints_summary.csv   (汇总表)

用法：
  python drug_fingerprint_generator.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
)

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # L1/
OUTPUT_DIR = BASE_DIR.parent / "L4" / "药物指纹"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 化合物SMILES定义
# ---------------------------------------------------------------------------
COMPOUNDS = {
    "Erastin": {
        "smiles": "CCOC1=CC=CC=C1N2C(=O)C3=CC=CC=C3N=C2C(C)N4CCN(CC4)C(=O)COC5=CC=C(C=C5)Cl",
        "category": "铁死亡诱导剂",
        "mw": None,  # 分子量，自动计算
    },
    "Fer-1": {
        "smiles": "CCOC(=O)C1=CC(=C(C=C1)NC2CCCCC2)N",
        "category": "铁死亡抑制剂",
        "mw": None,
    },
    "Quercetin": {
        "smiles": "C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O",
        "category": "铁死亡调节剂",
        "mw": None,
    },
    "DFO": {
        "smiles": "CC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCNC(=O)CCC(=O)N(O)CCCCCN.CS(=O)(O)=O",
        "category": "铁螯合剂",
        "mw": None,
    },
    "Vitamin_C": {
        "smiles": "C([C@@H]([C@H]1[C@@H]([C@H]([C@@H](O1)O)O)O)O)O",
        "category": "抗氧化剂",
        "mw": None,
    },
    "BCP": {
        "smiles": "C/C/1=C\\CCC(=C)[C@H]2CC([C@@H]2CC1)(C)C",
        "category": "铁死亡调节剂（β-石竹烯）",
        "mw": None,
    },
}


def validate_smiles(smiles):
    """验证SMILES并返回RDKit分子对象"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效的SMILES: {smiles}")
    return mol


def generate_morgan_fingerprints(mol_dict):
    """生成Morgan指纹（ECFP4等效，radius=2, 2048-bit）"""
    fps = {}
    bit_count = 2048
    radius = 2

    # 使用新版MorganGenerator API
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bit_count)

    for name, info in mol_dict.items():
        mol = info["mol"]
        fp = gen.GetFingerprint(mol)
        fps[name] = np.array(fp, dtype=np.int8)

    # 构建DataFrame
    bit_columns = [f"Morgan_{i}" for i in range(bit_count)]
    df = pd.DataFrame(
        {name: fps[name] for name in mol_dict.keys()}, index=bit_columns
    ).T
    df.index.name = "Compound"
    return df


def generate_maccs_fingerprints(mol_dict):
    """生成MACCS Keys指纹（167-bit）"""
    fps = {}

    for name, info in mol_dict.items():
        mol = info["mol"]
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps[name] = np.array(fp, dtype=np.int8)

    bit_columns = [f"MACCS_{i}" for i in range(167)]
    df = pd.DataFrame(
        {name: fps[name] for name in mol_dict.keys()}, index=bit_columns
    ).T
    df.index.name = "Compound"
    return df


def generate_rdkit_fingerprints(mol_dict):
    """生成RDKit拓扑指纹（2048-bit）"""
    fps = {}
    bit_count = 2048

    for name, info in mol_dict.items():
        mol = info["mol"]
        fp = Chem.RDKFingerprint(mol, fpSize=bit_count)
        fps[name] = np.array(fp, dtype=np.int8)

    bit_columns = [f"RDKit_{i}" for i in range(bit_count)]
    df = pd.DataFrame(
        {name: fps[name] for name in mol_dict.keys()}, index=bit_columns
    ).T
    df.index.name = "Compound"
    return df


def generate_atompair_fingerprints(mol_dict):
    """生成Atom Pair指纹（Hashed, 2048-bit）"""
    fps = {}
    bit_count = 2048

    # 使用新版AtomPairGenerator API
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=bit_count)

    for name, info in mol_dict.items():
        mol = info["mol"]
        fp = gen.GetFingerprint(mol)
        fps[name] = np.array(fp, dtype=np.int8)

    bit_columns = [f"AtomPair_{i}" for i in range(bit_count)]
    df = pd.DataFrame(
        {name: fps[name] for name in mol_dict.keys()}, index=bit_columns
    ).T
    df.index.name = "Compound"
    return df


def compute_descriptors(mol_dict):
    """计算分子描述符"""
    records = []
    for name, info in mol_dict.items():
        mol = info["mol"]
        records.append(
            {
                "Compound": name,
                "Category": info["category"],
                "SMILES": info["smiles"],
                "MolWt": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "RotBonds": Descriptors.NumRotatableBonds(mol),
                "TPSA": Descriptors.TPSA(mol),
                "RingCount": Descriptors.RingCount(mol),
                "HeavyAtomCount": mol.GetNumHeavyAtoms(),
                "NumAtoms": mol.GetNumAtoms(),
            }
        )
    return pd.DataFrame(records)


def compute_tanimoto_similarity(mol_dict, fp_type="morgan"):
    """计算化合物之间的Tanimoto相似度"""
    names = list(mol_dict.keys())
    n = len(names)

    if fp_type == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_func = lambda m: gen.GetFingerprint(m)
    elif fp_type == "maccs":
        fp_func = MACCSkeys.GenMACCSKeys
    elif fp_type == "rdkit":
        fp_func = lambda m: Chem.RDKFingerprint(m, fpSize=2048)
    else:
        raise ValueError(f"未知指纹类型: {fp_type}")

    fps = {name: fp_func(mol_dict[name]["mol"]) for name in names}

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = DataStructs.TanimotoSimilarity(
                fps[names[i]], fps[names[j]]
            )

    df = pd.DataFrame(sim_matrix, index=names, columns=names)
    return df


def main():
    print("=" * 60)
    print("  药物分子指纹生成器")
    print("=" * 60)

    # 验证SMILES并创建分子对象
    mol_dict = {}
    for name, info in COMPOUNDS.items():
        try:
            mol = validate_smiles(info["smiles"])
            mol_dict[name] = {
                "mol": mol,
                "smiles": info["smiles"],
                "category": info["category"],
            }
            print(f"  [OK] {name} - {info['category']}")
        except ValueError as e:
            print(f"  [FAIL] {name}: {e}")
            sys.exit(1)

    print(f"\n共加载 {len(mol_dict)} 个化合物\n")

    # -------------------------------------------------------------------
    # 1. Morgan指纹 (ECFP4, 2048-bit)
    # -------------------------------------------------------------------
    print("[1/5] 生成 Morgan 指纹 (ECFP4, 2048-bit)...")
    df_morgan = generate_morgan_fingerprints(mol_dict)
    output_path = OUTPUT_DIR / "drug_fingerprints_morgan.csv"
    df_morgan.to_csv(output_path)
    print(f"  -> 已保存: {output_path}  [{df_morgan.shape[0]} x {df_morgan.shape[1]}]")

    # -------------------------------------------------------------------
    # 2. MACCS Keys指纹 (167-bit)
    # -------------------------------------------------------------------
    print("[2/5] 生成 MACCS Keys 指纹 (167-bit)...")
    df_maccs = generate_maccs_fingerprints(mol_dict)
    output_path = OUTPUT_DIR / "drug_fingerprints_maccs.csv"
    df_maccs.to_csv(output_path)
    print(f"  -> 已保存: {output_path}  [{df_maccs.shape[0]} x {df_maccs.shape[1]}]")

    # -------------------------------------------------------------------
    # 3. RDKit拓扑指纹 (2048-bit)
    # -------------------------------------------------------------------
    print("[3/5] 生成 RDKit 拓扑指纹 (2048-bit)...")
    df_rdkit = generate_rdkit_fingerprints(mol_dict)
    output_path = OUTPUT_DIR / "drug_fingerprints_rdkit.csv"
    df_rdkit.to_csv(output_path)
    print(f"  -> 已保存: {output_path}  [{df_rdkit.shape[0]} x {df_rdkit.shape[1]}]")

    # -------------------------------------------------------------------
    # 4. Atom Pair指纹
    # -------------------------------------------------------------------
    print("[4/5] 生成 Atom Pair 指纹...")
    df_atompair = generate_atompair_fingerprints(mol_dict)
    output_path = OUTPUT_DIR / "drug_fingerprints_atompair.csv"
    df_atompair.to_csv(output_path)
    print(
        f"  -> 已保存: {output_path}  [{df_atompair.shape[0]} x {df_atompair.shape[1]}]"
    )

    # -------------------------------------------------------------------
    # 5. 分子描述符 + Tanimoto相似度汇总
    # -------------------------------------------------------------------
    print("[5/5] 生成分子描述符与相似度汇总...")

    # 分子描述符
    df_desc = compute_descriptors(mol_dict)
    desc_path = OUTPUT_DIR / "drug_descriptors.csv"
    df_desc.to_csv(desc_path, index=False)
    print(f"  -> 描述符已保存: {desc_path}")

    # Tanimoto相似度矩阵（基于Morgan指纹）
    df_sim = compute_tanimoto_similarity(mol_dict, fp_type="morgan")
    sim_path = OUTPUT_DIR / "drug_tanimoto_similarity.csv"
    df_sim.to_csv(sim_path)
    print(f"  -> 相似度矩阵已保存: {sim_path}")

    # -------------------------------------------------------------------
    # 打印汇总
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  分子描述符汇总")
    print("=" * 60)
    print(df_desc.to_string(index=False))

    print("\n" + "=" * 60)
    print("  Tanimoto相似度矩阵 (Morgan指纹)")
    print("=" * 60)
    print(df_sim.round(4).to_string())

    print("\n" + "=" * 60)
    print("  所有文件已生成完毕！")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
