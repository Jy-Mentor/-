"""基因特征生成.

负责从 L1 差异表达、细胞类型标记、配体-受体对等真实数据构建基因节点特征.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GeneFeatureBuilder:
    """基因特征构建器.

    输出特征维度:
        - 9 维 L1 全基因组差异表达统计
        - n_celltypes 维细胞类型标记 one-hot
        - 2 维配体/受体角色标记
        - (可选) 基因集成员身份标记
    """

    def __init__(
        self,
        project_root: Path | str | None = None,
        l1_de_path: Path | str | None = None,
        celltype_marker_path: Path | str | None = None,
        lr_pair_path: Path | str | None = None,
        gene_set_path: Path | str | None = None,
    ) -> None:
        from iron_aging import PROJECT_ROOT

        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.l1_de_path = Path(l1_de_path) if l1_de_path else self.project_root / "L3" / "L1_genome_wide_de.csv"
        self.celltype_marker_path = (
            Path(celltype_marker_path)
            if celltype_marker_path
            else self.project_root / "network_files" / "celltype_marker_genes.csv"
        )
        self.lr_pair_path = (
            Path(lr_pair_path)
            if lr_pair_path
            else self.project_root / "network_files" / "ligand_receptor_pairs.csv"
        )
        self.gene_set_path = (
            Path(gene_set_path)
            if gene_set_path
            else self.project_root / "铁衰老基因.txt"
        )

    def load_l1_features(self, genes: list[str]) -> dict[str, list[float]]:
        """从 L1 全基因组差异表达表计算 9 维统计特征."""
        features: dict[str, list[float]] = {g: [0.0] * 9 for g in genes}
        if not self.l1_de_path.exists():
            logger.error("L1 全基因组差异表达文件缺失: %s", self.l1_de_path)
            return features

        df = pd.read_csv(self.l1_de_path)
        required = {"gene", "log2FC", "pvalue", "padj", "dataset", "mean_case", "mean_control"}
        missing = required - set(df.columns)
        if missing:
            logger.error("L1 文件缺少必要字段: %s", missing)
            return features

        agg = (
            df.groupby("gene")
            .agg(
                mean_log2FC=("log2FC", "mean"),
                neg_log10_pval=("pvalue", lambda x: float(-np.log10(np.mean(x) + 1e-300))),
                n_datasets=("dataset", "nunique"),
                max_abs_log2FC=("log2FC", lambda x: float(np.max(np.abs(x)))),
                min_log2FC=("log2FC", "min"),
                std_log2FC=("log2FC", "std"),
                direction_consistency=("log2FC", lambda x: float(np.mean(x > 0))),
                mean_case=("mean_case", "mean"),
                mean_control=("mean_control", "mean"),
                min_padj=("padj", "min"),
            )
            .fillna(0)
        )
        raw_expression = (agg.pop("mean_case") + agg.pop("mean_control")) / 2
        agg["mean_expression"] = np.log2(raw_expression + 1)
        gene_dict = agg.to_dict("index")

        for gene in genes:
            if gene in gene_dict:
                d = gene_dict[gene]
                features[gene] = [
                    float(d["mean_log2FC"]),
                    float(d["neg_log10_pval"]),
                    float(d["n_datasets"]),
                    float(d["max_abs_log2FC"]),
                    float(d["min_log2FC"]),
                    float(d["std_log2FC"]),
                    float(d["direction_consistency"]),
                    float(d["mean_expression"]),
                    float(d["min_padj"]),
                ]

        n_nonzero = sum(1 for v in features.values() if v[0] != 0.0)
        logger.info(
            "L1 基因特征: %d 个基因有数据, 其中 %d 个目标基因非零",
            len(gene_dict),
            n_nonzero,
        )
        return features

    def load_celltype_features(self, genes: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
        """加载细胞类型标记 one-hot 特征."""
        celltype_map: dict[str, set[str]] = defaultdict(set)
        if self.celltype_marker_path.exists():
            df = pd.read_csv(self.celltype_marker_path)
            for _, row in df.iterrows():
                ct = str(row.get("celltype", "")).strip()
                g = str(row.get("gene", "")).strip().upper()
                if ct and g:
                    celltype_map[g].add(ct)
        else:
            logger.warning("细胞类型标记文件缺失: %s", self.celltype_marker_path)

        cell_types = sorted({ct for s in celltype_map.values() for ct in s})
        if not cell_types:
            cell_types = ["Neuron", "Microglia", "Astrocyte", "Oligodendrocyte", "Endothelial", "Pericyte"]

        ct_index = {ct: i for i, ct in enumerate(cell_types)}
        features: dict[str, np.ndarray] = {}
        for gene in genes:
            feat = np.zeros(len(cell_types), dtype=np.float32)
            for ct in celltype_map.get(gene, set()):
                if ct in ct_index:
                    feat[ct_index[ct]] = 1.0
            features[gene] = feat
        return features, cell_types

    def load_lr_features(self, genes: list[str]) -> dict[str, np.ndarray]:
        """加载配体/受体角色特征."""
        lr_role: dict[str, dict[str, bool]] = defaultdict(lambda: {"ligand": False, "receptor": False})
        if self.lr_pair_path.exists():
            df = pd.read_csv(self.lr_pair_path)
            for _, row in df.iterrows():
                ligand = str(row.get("ligand", "")).strip().upper()
                receptor = str(row.get("receptor", "")).strip().upper()
                if ligand:
                    lr_role[ligand]["ligand"] = True
                if receptor:
                    lr_role[receptor]["receptor"] = True
        else:
            logger.warning("配体-受体对文件缺失: %s", self.lr_pair_path)

        features: dict[str, np.ndarray] = {}
        for gene in genes:
            features[gene] = np.array(
                [
                    1.0 if lr_role.get(gene, {}).get("ligand", False) else 0.0,
                    1.0 if lr_role.get(gene, {}).get("receptor", False) else 0.0,
                ],
                dtype=np.float32,
            )
        return features

    def load_gene_set(self) -> set[str]:
        """加载铁衰老基因集."""
        gene_set: set[str] = set()
        if self.gene_set_path.exists():
            text = self.gene_set_path.read_text(encoding="utf-8")
            for line in text.splitlines():
                g = line.strip().upper()
                if g:
                    gene_set.add(g)
        else:
            logger.warning("基因集文件缺失: %s", self.gene_set_path)
        return gene_set

    def build(
        self,
        genes: list[str],
        include_l1: bool = True,
        include_celltype: bool = True,
        include_lr: bool = True,
        include_gene_set: bool = True,
    ) -> dict[str, np.ndarray]:
        """构建基因特征矩阵.

        Returns:
            gene -> feature_vector 的映射.
        """
        l1_feat = self.load_l1_features(genes) if include_l1 else {g: [] for g in genes}
        if include_celltype:
            ct_feat, cell_types = self.load_celltype_features(genes)
        else:
            ct_feat = {g: np.array([], dtype=np.float32) for g in genes}
            cell_types = []
        lr_feat = self.load_lr_features(genes) if include_lr else {g: np.array([], dtype=np.float32) for g in genes}
        gene_set = self.load_gene_set() if include_gene_set else set()

        result: dict[str, np.ndarray] = {}
        for gene in genes:
            parts: list[Any] = [np.array(l1_feat[gene], dtype=np.float32)]
            if include_gene_set:
                parts.append(np.array([1.0 if gene in gene_set else 0.0], dtype=np.float32))
            parts.extend([ct_feat[gene], lr_feat[gene]])
            result[gene] = np.concatenate(parts)

        dim = result[genes[0]].shape[0]
        logger.info(
            "基因特征维度: %d (L1=%d, gene_set=%d, celltype=%d, LR=%d)",
            dim,
            9 if include_l1 else 0,
            1 if include_gene_set else 0,
            len(cell_types),
            2 if include_lr else 0,
        )
        return result
