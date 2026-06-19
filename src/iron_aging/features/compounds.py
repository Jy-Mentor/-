"""化合物特征生成.

负责从 SMILES / 数据库读取化合物, 生成：
- RDKit 物化性质
- Morgan / AtomPair / MACCS / RDKit 指纹
- AttentiveFP 预训练嵌入
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CompoundFeatureBuilder:
    """化合物特征构建器."""

    def __init__(self, network_dir: Path | str | None = None) -> None:
        from iron_aging import NETWORK_DIR

        self.network_dir = Path(network_dir) if network_dir else NETWORK_DIR

    def load_smiles(self) -> pd.DataFrame:
        """读取化合物 SMILES 表."""
        path = self.network_dir / "compound_smiles.csv"
        if not path.exists():
            raise FileNotFoundError(f"化合物 SMILES 文件缺失: {path}")
        return pd.read_csv(path)

    def build_properties(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """基于 RDKit 计算核心物化性质."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
        except ImportError as e:
            raise ImportError("RDKit 未安装, 无法计算化合物性质") from e

        if df is None:
            df = self.load_smiles()

        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            smiles = str(row.get("CanonicalSMILES", ""))
            name = str(row.get("compound", ""))
            if not smiles or not name:
                logger.warning("化合物 %s 缺少 SMILES, 跳过性质计算", name)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning("RDKit 无法解析 %s 的 SMILES: %s", name, smiles)
                continue
            records.append(
                {
                    "compound": name,
                    "MW": Descriptors.MolWt(mol),
                    "LogP": Descriptors.MolLogP(mol),
                    "HBD": Lipinski.NumHDonors(mol),
                    "HBA": Lipinski.NumHAcceptors(mol),
                    "TPSA": rdMolDescriptors.CalcTPSA(mol),
                    "RotB": Descriptors.NumRotatableBonds(mol),
                }
            )
        return pd.DataFrame(records)

    def build_fingerprints(self, df: pd.DataFrame | None = None) -> dict[str, dict[str, np.ndarray]]:
        """生成多种分子指纹.

        Returns:
            {compound_name: {fingerprint_type: np.ndarray}}
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
        except ImportError as e:
            raise ImportError("RDKit 未安装, 无法计算分子指纹") from e

        if df is None:
            df = self.load_smiles()

        result: dict[str, dict[str, np.ndarray]] = {}
        for _, row in df.iterrows():
            smiles = str(row.get("CanonicalSMILES", ""))
            name = str(row.get("compound", ""))
            if not smiles or not name:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            result[name] = {
                "morgan": np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)),
                "atompair": np.array(
                    rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=1024)
                ),
                "maccs": np.array(MACCSkeys.GenMACCSKeys(mol)),
                "rdkit": np.array(rdMolDescriptors.RDKFingerprint(mol)),
            }
        return result
