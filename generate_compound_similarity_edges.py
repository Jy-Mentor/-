"""生成分子-分子化学相似性边 (Tanimoto >= 0.7).

输入:
    network_files/compound_smiles.csv

输出:
    network_files/compound_compound_similarity_edges.csv
        compound_A, compound_B, similarity, source, confidence, confidence_level, download_date

方法:
    - 用 RDKit 计算 Morgan/ECFP4 指纹 (半径 2, 2048 bit).
    - 两两计算 Tanimoto 相似度.
    - 保留相似度 >= 0.7 的无向边 (A-B 与 B-A 去重, 无自环).
    - confidence = similarity, confidence_level 按 0.8/0.6 分档.

参考:
    - Morgan fingerprints: Rogers D, Hahn M. J Chem Inf Model, 2010.
    - Tanimoto coefficient for chemical similarity.
"""

from __future__ import annotations

import argparse
import json
import logging
import traceback
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "network_files" / "compound_smiles.csv"
OUTPUT_CSV = BASE_DIR / "network_files" / "compound_compound_similarity_edges.csv"
METADATA_JSON = BASE_DIR / "external_data" / "compound_similarity_metadata.json"

SIMILARITY_THRESHOLD = 0.7


def load_compounds(csv_path: Path) -> pd.DataFrame:
    """读取化合物 SMILES, 过滤无效分子."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["compound", "CanonicalSMILES"])
    df["compound"] = df["compound"].astype(str).str.strip()
    df["CanonicalSMILES"] = df["CanonicalSMILES"].astype(str).str.strip()
    return df


def compute_fingerprints(smiles_series: pd.Series):
    """计算 Morgan 指纹, 返回 (有效索引列表, 指纹列表)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    valid_indices = []
    fps = []
    for idx, smiles in smiles_series.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning("无法解析 SMILES, 跳过: %s", smiles[:60])
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        valid_indices.append(idx)
        fps.append(fp)
    return valid_indices, fps


def confidence_level(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def generate_edges(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """生成 compound-compound 相似性边."""
    from rdkit import DataStructs

    valid_indices, fps = compute_fingerprints(df["CanonicalSMILES"])
    n = len(valid_indices)
    logger.info("有效分子数: %d / %d", n, len(df))

    records = []
    names = df.iloc[valid_indices]["compound"].tolist()
    # 利用 DataStructs.BulkTanimotoSimilarity 逐行计算, 避免 O(n^2) 显式双重循环
    for i in range(n):
        name_a = names[i]
        similarities = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        for j in range(i + 1, n):
            sim = similarities[j]
            if sim >= threshold:
                records.append(
                    {
                        "compound_A": name_a,
                        "compound_B": names[j],
                        "similarity": round(sim, 4),
                        "source": "RDKit_Morgan_Tanimoto",
                        "confidence": round(sim, 4),
                        "confidence_level": confidence_level(sim),
                    }
                )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("similarity", ascending=False).reset_index(drop=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate compound-compound similarity edges")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help="Tanimoto similarity threshold",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not INPUT_CSV.exists():
        logger.error("输入文件不存在: %s", INPUT_CSV)
        return 1

    df = load_compounds(INPUT_CSV)
    logger.info("读取化合物: %d 个", len(df))

    edges = generate_edges(df, args.threshold)
    edges["download_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    METADATA_JSON.parent.mkdir(parents=True, exist_ok=True)

    if edges.empty:
        logger.warning("未生成任何 compound-compound 相似性边 (阈值 %.2f)", args.threshold)
        edges.to_csv(OUTPUT_CSV, index=False)
    else:
        edges.to_csv(OUTPUT_CSV, index=False)
        logger.info("已写入 %s: %d 条边", OUTPUT_CSV, len(edges))

    metadata = {
        "source": "RDKit_Morgan_Tanimoto",
        "method": "Morgan fingerprint (radius=2, nBits=2048) + Tanimoto similarity",
        "input_file": str(INPUT_CSV),
        "output_file": str(OUTPUT_CSV),
        "threshold": args.threshold,
        "n_compounds_input": len(df),
        "n_compounds_valid": len(df),
        "n_edges": len(edges),
        "download_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_distribution": edges["confidence_level"].value_counts().to_dict() if not edges.empty else {},
    }
    METADATA_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("已写入元数据: %s", METADATA_JSON)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
