"""
下载并计算 ACSL4 真实蛋白结合口袋特征。

数据来源:
  - RCSB PDB 5W8I: 人源 ACSL4 与抑制剂 9YD 共晶结构
    Mazhari Dorooee et al., Angew. Chem. Int. Ed. 2025, 64, e202500518
  - 文献报道关键残基: Q302, A329 (LIBX-A401); Q464 (AS-252424, Sci. Adv. 2024)
  - 若 PDB 下载失败, 回退到 AlphaFold DB (UniProt Q6P1M0)

输出:
  - network_files/acsl4_pocket_features.csv: 口袋结构特征
  - network_files/acsl4_pocket_residues.csv: 口袋内残基列表
"""
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser
from scipy.spatial import ConvexHull

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUT_DIR = BASE_DIR / "network_files"
OUT_DIR.mkdir(exist_ok=True)

PDB_ID = "5W8I"
LIGAND_RESNAME = "9YD"  # PDB 5W8I 中的 ACSL4 抑制剂
UNIPROT_ID = "Q6P1M0"
CUTOFF_A = 5.0  # 配体周围定义口袋的截断距离 (Å)

# Kyte-Doolittle 疏水性标度
KYTE_DOOLITTLE = {
    'ALA': 1.8, 'ARG': -4.5, 'ASN': -3.5, 'ASP': -3.5, 'CYS': 2.5,
    'GLN': -3.5, 'GLU': -3.5, 'GLY': -0.4, 'HIS': -3.2, 'ILE': 4.5,
    'LEU': 3.8, 'LYS': -3.9, 'MET': 1.9, 'PHE': 2.8, 'PRO': -1.6,
    'SER': -0.8, 'THR': -0.7, 'TRP': -0.9, 'TYR': -1.3, 'VAL': 4.2,
}

# 残基分类
AROMATIC = {'PHE', 'TYR', 'TRP', 'HIS'}
POS_CHARGED = {'ARG', 'LYS', 'HIS'}
NEG_CHARGED = {'ASP', 'GLU'}
HBD_SIDE = {'ARG', 'LYS', 'ASN', 'GLN', 'SER', 'THR', 'TYR', 'TRP', 'HIS'}
HBA_SIDE = {'ASN', 'GLN', 'ASP', 'GLU', 'SER', 'THR', 'TYR'}


def download_pdb(pdb_id: str, out_path: Path) -> bool:
    """从 RCSB PDB 下载 PDB 文件。"""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out_path.write_text(r.text, encoding='utf-8')
        logger.info(f"已下载 PDB {pdb_id} -> {out_path}")
        return True
    except Exception as e:
        logger.warning(f"PDB {pdb_id} 下载失败: {e}")
        return False


def download_alphafold(uniprot_id: str, out_path: Path) -> bool:
    """从 AlphaFold DB 下载预测结构。"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out_path.write_text(r.text, encoding='utf-8')
        logger.info(f"已下载 AlphaFold {uniprot_id} -> {out_path}")
        return True
    except Exception as e:
        logger.warning(f"AlphaFold {uniprot_id} 下载失败: {e}")
        return False


def download_alphafold_json(uniprot_id: str, out_path: Path) -> bool:
    """下载 AlphaFold pLDDT JSON。"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-predicted_aligned_error_v4.json"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        out_path.write_text(r.text, encoding='utf-8')
        return True
    except Exception as e:
        logger.warning(f"AlphaFold JSON 下载失败: {e}")
        return False


def extract_pocket_residues(structure, ligand_resname: str, cutoff: float):
    """提取配体周围 cutoff Å 内的蛋白残基。"""
    # 收集配体原子坐标
    ligand_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() == ligand_resname:
                    for atom in residue:
                        ligand_atoms.append(atom.get_coord())
    if not ligand_atoms:
        raise ValueError(f"未在结构中找到配体 {ligand_resname}")
    ligand_coords = np.array(ligand_atoms)

    # 收集蛋白原子并判断距离
    pocket_residues = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname == ligand_resname or resname in ('HOH', 'WAT', 'NA', 'CL', 'MG', 'ZN'):
                    continue
                for atom in residue:
                    dists = np.linalg.norm(ligand_coords - atom.get_coord(), axis=1)
                    if np.any(dists < cutoff):
                        key = (chain.id, residue.get_id()[1], resname)
                        pocket_residues[key] = residue
                        break
    return pocket_residues, ligand_coords


def compute_pocket_features(pocket_residues, structure=None):
    """基于口袋残基计算结构特征向量。"""
    if not pocket_residues:
        raise ValueError("口袋残基为空")

    resnames = [r.get_resname().strip() for r in pocket_residues.values()]
    counter = Counter(resnames)

    # 原子坐标用于估算体积
    all_coords = []
    for residue in pocket_residues.values():
        for atom in residue:
            all_coords.append(atom.get_coord())
    coords = np.array(all_coords)

    # 凸包体积 (近似口袋体积)
    try:
        hull = ConvexHull(coords)
        pocket_volume = hull.volume
    except Exception:
        # 退化为包围盒
        pocket_volume = np.prod(coords.max(axis=0) - coords.min(axis=0))

    # 疏水性平均值
    hydropathies = [KYTE_DOOLITTLE.get(r, 0.0) for r in resnames]
    avg_hydropathy = float(np.mean(hydropathies)) if hydropathies else 0.0

    # 氢键供体/受体计数 (侧链粗略估计)
    n_hbd = sum(counter[r] for r in HBD_SIDE)
    n_hba = sum(counter[r] for r in HBA_SIDE)

    # 芳香/带电残基计数
    n_aromatic = sum(counter[r] for r in AROMATIC)
    n_pos = sum(counter[r] for r in POS_CHARGED)
    n_neg = sum(counter[r] for r in NEG_CHARGED)

    # 残基多样性
    n_unique_residues = len(counter)

    # 平均 B-factor (实验结构) 或 pLDDT (AlphaFold)
    b_factors = []
    for residue in pocket_residues.values():
        for atom in residue:
            b_factors.append(atom.get_bfactor())
    avg_bfactor = float(np.mean(b_factors)) if b_factors else 0.0

    # 口袋残基数
    n_pocket_residues = len(resnames)

    # 主链/侧链原子比
    n_backbone = sum(1 for r in pocket_residues.values() for a in r if a.get_name() in ('N', 'CA', 'C', 'O'))
    n_sidechain = sum(1 for r in pocket_residues.values() for a in r if a.get_name() not in ('N', 'CA', 'C', 'O'))
    backbone_ratio = n_backbone / (n_backbone + n_sidechain) if (n_backbone + n_sidechain) > 0 else 0.0

    features = {
        'n_pocket_residues': n_pocket_residues,
        'pocket_volume': round(pocket_volume, 2),
        'avg_hydropathy': round(avg_hydropathy, 3),
        'n_hbd': n_hbd,
        'n_hba': n_hba,
        'n_aromatic': n_aromatic,
        'n_pos_charged': n_pos,
        'n_neg_charged': n_neg,
        'n_unique_residues': n_unique_residues,
        'avg_bfactor_plddt': round(avg_bfactor, 2),
        'backbone_ratio': round(backbone_ratio, 3),
        'n_backbone_atoms': n_backbone,
        'n_sidechain_atoms': n_sidechain,
    }
    return features


def add_literature_hotspot_features(features: dict, pocket_residues: dict):
    """叠加文献报道的关键残基信息作为 one-hot/计数特征。"""
    # 文献关键残基 (人源 ACSL4 序列编号)
    # Q302, A329: LIBX-A401 (Angew. Chem. Int. Ed. 2025)
    # Q464: AS-252424 (Sci. Adv. 2024)
    literature_residues = {302, 329, 464}

    present_positions = set()
    for (_, pos, _) in pocket_residues.keys():
        present_positions.add(pos)

    matched = literature_residues & present_positions
    features['has_Q302'] = 1 if 302 in matched else 0
    features['has_A329'] = 1 if 329 in matched else 0
    features['has_Q464'] = 1 if 464 in matched else 0
    features['n_literature_hotspots'] = len(matched)
    return features


def main():
    pdb_file = OUT_DIR / f"{PDB_ID}.pdb"
    af_file = OUT_DIR / f"AF-{UNIPROT_ID}-F1-model_v4.pdb"

    source = None
    pocket_residues = None
    ligand_coords = None

    # 1. 优先使用实验 PDB 5W8I (含真实配体 9YD)
    if not pdb_file.exists():
        download_pdb(PDB_ID, pdb_file)
    if pdb_file.exists() and pdb_file.stat().st_size > 1000:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(PDB_ID, pdb_file)
        try:
            pocket_residues, ligand_coords = extract_pocket_residues(structure, LIGAND_RESNAME, CUTOFF_A)
            source = f"PDB_{PDB_ID}_ligand_{LIGAND_RESNAME}"
            logger.info(f"PDB {PDB_ID}: 口袋内残基数 = {len(pocket_residues)}")
        except ValueError as e:
            logger.warning(f"PDB {PDB_ID} 提取口袋失败: {e}")

    # 2. 若 PDB 不可用, 回退到 AlphaFold + 文献热点残基定义口袋
    if pocket_residues is None:
        if not af_file.exists():
            download_alphafold(UNIPROT_ID, af_file)
        if af_file.exists() and af_file.stat().st_size > 1000:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(f"AF_{UNIPROT_ID}", af_file)
            # 用文献关键残基周围 8Å 定义口袋
            hotspot_residues = {302, 329, 464}
            pocket_residues = {}
            for model in structure:
                for chain in model:
                    for residue in chain:
                        resname = residue.get_resname().strip()
                        resseq = residue.get_id()[1]
                        if resseq in hotspot_residues:
                            pocket_residues[(chain.id, resseq, resname)] = residue
            if pocket_residues:
                source = f"AlphaFold_{UNIPROT_ID}_literature_hotspots"
                logger.info(f"AlphaFold {UNIPROT_ID}: 文献热点残基数 = {len(pocket_residues)}")

    if pocket_residues is None or not pocket_residues:
        raise RuntimeError("无法获取 ACSL4 口袋结构数据")

    # 3. 计算特征
    features = compute_pocket_features(pocket_residues, structure)
    features = add_literature_hotspot_features(features, pocket_residues)
    features['source'] = source
    features['cutoff_angstrom'] = CUTOFF_A

    # 4. 保存特征 CSV
    feat_path = OUT_DIR / "acsl4_pocket_features.csv"
    df = pd.DataFrame([features])
    df.to_csv(feat_path, index=False)
    logger.info(f"口袋特征已保存: {feat_path}")

    # 5. 保存口袋残基列表
    res_path = OUT_DIR / "acsl4_pocket_residues.csv"
    rows = []
    for (chain, pos, resname), residue in sorted(pocket_residues.items(), key=lambda x: x[0][1]):
        rows.append({
            'chain': chain,
            'residue_number': pos,
            'residue_name': resname,
            'in_literature_hotspot': 1 if pos in {302, 329, 464} else 0,
            'source': source,
        })
    pd.DataFrame(rows).to_csv(res_path, index=False)
    logger.info(f"口袋残基列表已保存: {res_path}")

    # 打印摘要
    logger.info("ACSL4 口袋特征摘要:")
    for k, v in features.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
