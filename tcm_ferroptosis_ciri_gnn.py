"""[EXPLORATORY / DEPRECATED] TCM monomer × ferroptosis × CIRI prediction module.

WARNING
-------
This script is kept for reproducibility of earlier exploratory analyses only.
It is NOT the official implementation for the iron-aging study.  The official
v4.0 pipeline is located at:

    src/iron_aging/apps/hgt_pipeline.py

The GAT-HGT fusion encoder originally developed here has been migrated to the
official pipeline (encoder_type="gat_hgt").  Key limitations of this script:

1. Evaluates on the full graph, so validation/test target edges participate in
   message passing.  This can inflate AUC/AP and is considered data leakage in
   link-prediction evaluation.  The official pipeline follows SpotTarget
   (Zhu et al., WSDM 2024) and removes all target edges from the message-passing
   graph during both training and evaluation.
2. Uses random reliable negative sampling, whereas the official pipeline uses
   structured negative sampling by default.
3. Uses stratified K-fold cross-validation with an 85/15 train/val split inside
   each fold; the official pipeline uses a single reproducible 70/15/15 split.
4. Some downstream scoring utilities mix BBB, activity and pathway heuristics
   with hard-coded weights; these are not part of the official model evaluation.

For any publication-grade result, please use src/iron_aging/apps/hgt_pipeline.py
with config.yaml and report metrics from the official pipeline only.

References
----------
- Hu et al. (2020) "Heterogeneous Graph Transformer", WWW.
- Velickovic et al. (2018) "Graph Attention Networks", ICLR.
- Brody et al. (2022) "How Attentive are Graph Attention Networks?", ICLR.
- Zhu et al. (2024) "SpotTarget", WSDM.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

# Suppress OpenMP duplicate-lib warning on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SRC_DIR = PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# 抑制已知第三方库弃用警告
from iron_aging.utils.warnings import suppress_known_library_warnings

suppress_known_library_warnings()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import HeteroData

# Reuse the vetted ablation framework for data construction, training and eval
import ablation_hgt_vs_gat as abl  # type: ignore[import-not-found]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_LOG_FILE = PROJECT_ROOT / "L3_results" / "tcm_ferroptosis_ciri_gnn.log"
_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_file_handler = logging.FileHandler(_LOG_FILE, mode="w", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(_file_handler)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = PROJECT_ROOT / "L3_results" / "tcm_ferroptosis_ciri_gnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_DIR = PROJECT_ROOT / "network_files"

# Ferroptosis-related gene set (curated from project L1 core set + literature)
FERROPTOSIS_GENES: set[str] = {
    # Core ferroptosis machinery
    "GPX4",
    "SLC7A11",
    "SLC3A2",
    "ACSL4",
    "ACSL3",
    "LPCAT3",
    "ALOX15",
    "ALOX12",
    "PTGS2",
    "FSP1",
    "GCH1",
    "DHODH",
    # Iron metabolism
    "TFRC",
    "FTH1",
    "FTL",
    "NCOA4",
    "DMT1",
    "SLC11A2",
    "SLC40A1",
    # Antioxidant / Nrf2
    "NFE2L2",
    "KEAP1",
    "HMOX1",
    "NQO1",
    "SOD1",
    "SOD2",
    "CAT",
    "GSS",
    "GSR",
    "TXNRD1",
    "MT1",
    "MT2",
    # Inflammation / stress
    "HMGB1",
    "SAT1",
    "VDAC2",
    "VAMP8",
    "SCP2",
    "PEBP1",
    "CS",
    "ATP5F1A",
    "MTOR",
    "PIK3CA",
    "AKT1",
    "STAT3",
    "NFKB1",
    "RELA",
    "NLRP3",
    "IL1B",
    "IL6",
    "TNF",
    "CXCL8",
    "HIF1A",
    "PPARG",
    "PPARA",
    "SIRT1",
    "SIRT3",
    "FOXO3",
    "BCL2",
    "BAX",
    "TP53",
    "CDKN1A",
}

# CIRI-relevant ferroptosis hub genes used for treatment scoring
CIRI_FERROPTOSIS_HUBS: set[str] = {
    "ACSL4",
    "GPX4",
    "SLC7A11",
    "TFRC",
    "FTH1",
    "PTGS2",
    "NFE2L2",
    "HMOX1",
    "TP53",
    "CDKN1A",
    "HIF1A",
    "TNF",
    "IL1B",
    "IL6",
    "NLRP3",
}


class GATHGTLP(abl.HGTLP):
    """GAT-HGT fusion encoder with learnable gating for link prediction.

    Inherits from HGTLP so that the existing training and evaluation loops
    in ablation_hgt_vs_gat can handle it via duck-typed HGT branch calls.
    The model runs an HGT encoder on the heterogeneous graph and a GATv2
    encoder on a homogeneous projection in parallel. Node embeddings are
    fused via a learnable gate and passed to the MLP link-prediction head.
    """

    def __init__(
        self,
        data: HeteroData,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(data, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=dropout)
        # GAT on the homogeneous projection
        self.gat = abl.GATLP(data, hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=dropout)

        # Project GAT output (hidden_dim * heads) to hidden_dim to match HGT
        self.gat_proj = nn.ModuleDict({
            nt: nn.Linear(self.gat.convs[-1].heads * hidden_dim, hidden_dim)
            for nt in data.node_types
        })

        # Learnable fusion gate per node type
        self.fusion_gate = nn.ModuleDict({nt: nn.Linear(hidden_dim * 2, 1) for nt in data.node_types})

        self.dropout = dropout
        self.node_type_offset = abl.GATLP._compute_offsets(data)

    def _to_homogeneous(self, data: HeteroData) -> torch.Tensor:
        """Convert heterogeneous edge_index_dict to a homogeneous edge_index."""
        edge_list: list[torch.Tensor] = []
        for (src_type, _, dst_type), edge_index in data.edge_index_dict.items():
            src_offset = self.node_type_offset[src_type]
            dst_offset = self.node_type_offset[dst_type]
            shifted = torch.stack([
                edge_index[0] + src_offset,
                edge_index[1] + dst_offset,
            ])
            edge_list.append(shifted)
        return torch.cat(edge_list, dim=1)

    def forward(
        self,
        data: HeteroData,
        edge_index_dict: dict | None = None,
        homogeneous_edge_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return fused node embeddings per node type."""
        hgt_z = super().forward(data, edge_index_dict)

        if homogeneous_edge_index is None:
            homogeneous_edge_index = self._to_homogeneous(data)
        gat_z_hom = self.gat(data, homogeneous_edge_index)

        fused: dict[str, torch.Tensor] = {}
        for nt in data.node_types:
            offset = self.node_type_offset[nt]
            n_nodes = data[nt].num_nodes
            gat_nt = gat_z_hom[offset : offset + n_nodes]
            gat_nt = self.gat_proj[nt](gat_nt)
            hgt_nt = hgt_z[nt]

            concat = torch.cat([hgt_nt, gat_nt], dim=-1)
            gate = torch.sigmoid(self.fusion_gate[nt](concat))
            fused[nt] = gate * hgt_nt + (1.0 - gate) * gat_nt
            fused[nt] = F.dropout(fused[nt], p=self.dropout, training=self.training)
        return fused


def set_seed(seed: int = SEED) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tcm_monomers() -> list[str]:
    """Load TCM monomer names from the project compound list."""
    path = NETWORK_DIR / "tcm_monomers_pubchem.csv"
    if not path.exists():
        logger.error("Missing TCM monomer list: %s", path)
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "compound" not in df.columns:
        raise ValueError("tcm_monomers_pubchem.csv must contain a 'compound' column")
    compounds = sorted({c.strip() for c in df["compound"].dropna().astype(str).tolist()})
    logger.info("Loaded %d TCM monomers", len(compounds))
    return compounds


def load_ferroptosis_genes_from_l1() -> set[str]:
    """Load ferroptosis/iron-aging gene set from L1 core gene set."""
    path = PROJECT_ROOT / "L1" / "core_gene_set.csv"
    if not path.exists():
        logger.warning("Missing L1 core gene set: %s; using curated list only", path)
        return set()
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        logger.warning("L1 core gene set has no 'gene' column")
        return set()
    genes = {g.strip().upper() for g in df["gene"].dropna().astype(str).tolist()}
    logger.info("Loaded %d genes from L1 core gene set", len(genes))
    return genes


def build_focused_subgraph_info(
    data: HeteroData, node_names: dict[str, list[str]], tcm_compounds: list[str]
) -> dict[str, Any]:
    """Build indices describing the focused TCM-ferroptosis-CIRI subgraph.

    The subgraph is defined by:
      - all TCM monomer compounds
      - ferroptosis-related genes (curated + L1)
      - their 1-hop neighbours in compound_target / PPI / pathway edges
      - pathways enriched in the selected genes
      - CIRI disease node(s)
    """
    ferro_genes = FERROPTOSIS_GENES | load_ferroptosis_genes_from_l1()

    compound2idx = {name: i for i, name in enumerate(node_names.get("compound", []))}
    gene2idx = {name: i for i, name in enumerate(node_names.get("gene", []))}
    pathway2idx = {name: i for i, name in enumerate(node_names.get("pathway", []))}
    disease2idx = {name: i for i, name in enumerate(node_names.get("disease", []))}

    tcm_indices = sorted({compound2idx[c] for c in tcm_compounds if c in compound2idx})
    if not tcm_indices:
        raise ValueError("No TCM monomers found in the graph")

    ferro_gene_indices = sorted({gene2idx[g] for g in ferro_genes if g in gene2idx})
    logger.info("Focused subgraph: %d TCM compounds, %d ferroptosis genes", len(tcm_indices), len(ferro_gene_indices))

    # 1-hop expansion via compound_target and PPI / pathway edges
    expanded_genes = set(ferro_gene_indices)
    expanded_pathways = set()

    key_ct = ("compound", "targets", "gene")
    if key_ct in data.edge_types:
        ei = data[key_ct].edge_index.cpu().numpy()
        for s, d in zip(ei[0], ei[1]):
            if s in tcm_indices:
                expanded_genes.add(int(d))

    key_ppi = ("gene", "interacts", "gene")
    if key_ppi not in data.edge_types:
        key_ppi = ("gene", "coexp", "gene")
    if key_ppi in data.edge_types:
        ei = data[key_ppi].edge_index.cpu().numpy()
        seed = set(ferro_gene_indices)
        for s, d in zip(ei[0], ei[1]):
            if int(s) in seed:
                expanded_genes.add(int(d))
            if int(d) in seed:
                expanded_genes.add(int(s))

    key_gp = ("gene", "enriched_in", "pathway")
    if key_gp not in data.edge_types:
        key_gp = ("gene", "belongs_to", "pathway")
    if key_gp in data.edge_types:
        ei = data[key_gp].edge_index.cpu().numpy()
        for s, d in zip(ei[0], ei[1]):
            if int(s) in expanded_genes:
                expanded_pathways.add(int(d))

    return {
        "compound_indices": tcm_indices,
        "ferroptosis_gene_indices": ferro_gene_indices,
        "expanded_gene_indices": sorted(expanded_genes),
        "pathway_indices": sorted(expanded_pathways),
        "compound2idx": compound2idx,
        "gene2idx": gene2idx,
        "pathway2idx": pathway2idx,
        "disease2idx": disease2idx,
        "ferroptosis_genes": ferro_genes,
    }


def load_pathway_enrichment() -> pd.DataFrame:
    """Load gene-pathway enrichment table."""
    path = NETWORK_DIR / "gene_pathway_enrichment.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pathway enrichment file: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df


def load_bbb_scores() -> dict[str, tuple[float | None, str | None]]:
    """Load blood-brain-barrier scores for TCM monomers."""
    path = NETWORK_DIR / "tcm_bbb_evaluation.csv"
    if not path.exists():
        logger.warning("Missing BBB evaluation file: %s", path)
        return {}
    df = pd.read_csv(path)
    if "compound" not in df.columns:
        logger.warning("BBB file missing 'compound' column")
        return {}
    score_col = "bbb_score" if "bbb_score" in df.columns else "BBB_score"
    pass_col = "bbb_pass" if "bbb_pass" in df.columns else "BBB_pass"
    out: dict[str, tuple[float | None, str | None]] = {}
    for _, row in df.iterrows():
        cmp = str(row["compound"]).strip()
        score = float(row[score_col]) if pd.notna(row.get(score_col)) else None
        p = str(row[pass_col]) if pd.notna(row.get(pass_col)) else None
        out[cmp] = (score, p)
    return out


def hyperparameter_search(
    data: HeteroData,
    node_names: dict[str, list[str]],
    task: str = "ct",
    n_trials: int = 8,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Random hyper-parameter search with stratified K-fold CV.

    Parameters
    ----------
    data, node_names
        Full heterogeneous graph and node name mappings.
    task
        Link-prediction task to optimise (default: compound-target).
    n_trials
        Number of random configurations to try.
    n_folds
        Number of CV folds per trial.

    Returns
    -------
    best_config and aggregated CV results per trial.
    """
    param_space = {
        "hidden_dim": [32, 64, 128],
        "num_layers": [1, 2, 3],
        "heads": [2, 4, 8],
        "dropout": [0.2, 0.3, 0.5],
        "lr": [5e-4, 1e-3, 2e-3],
        "weight_decay": [1e-6, 1e-5, 1e-4],
    }

    best_val_auc = -np.inf
    best_config: dict[str, Any] | None = None
    trial_results: list[dict[str, Any]] = []

    rng = np.random.default_rng(SEED)
    for trial in range(n_trials):
        config = {}
        for k, v in param_space.items():
            choice = rng.choice(v)
            config[k] = choice.item() if isinstance(choice, np.generic) else choice
        logger.info("HPO trial %d/%d: %s", trial + 1, n_trials, config)

        val_aucs: list[float] = []
        test_aucs: list[float] = []

        key, num_edges = abl.get_target_edge_index(data, task)
        src_type, _, dst_type = key
        edge_index = data[key].edge_index.cpu().numpy()
        seen: set[tuple[int, int]] = set()
        positives: list[tuple[int, int]] = []
        for i in range(edge_index.shape[1]):
            e = (int(edge_index[0, i]), int(edge_index[1, i]))
            if e not in seen:
                seen.add(e)
                positives.append(e)
        positives = np.array(positives)
        stratify = abl._stratify_labels(positives, data[src_type].num_nodes, task)  # type: ignore[attr-defined]

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED + trial)
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(positives, stratify)):
            fold_result = _run_fold_with_config(
                data, node_names, task, fold_idx, positives[train_val_idx], positives[test_idx], config
            )
            val_aucs.append(fold_result["val_auc"])
            test_aucs.append(fold_result["test_auc"])

        mean_val_auc = float(np.mean(val_aucs))
        mean_test_auc = float(np.mean(test_aucs))
        trial_results.append({
            "trial": trial,
            "config": config,
            "val_auc_mean": mean_val_auc,
            "val_auc_std": float(np.std(val_aucs)),
            "test_auc_mean": mean_test_auc,
            "test_auc_std": float(np.std(test_aucs)),
        })
        logger.info(
            "Trial %d val_auc=%.4f±%.4f test_auc=%.4f±%.4f",
            trial,
            mean_val_auc,
            np.std(val_aucs),
            mean_test_auc,
            np.std(test_aucs),
        )

        if mean_val_auc > best_val_auc:
            best_val_auc = mean_val_auc
            best_config = config.copy()

    if best_config is None:
        best_config = {k: v[0] for k, v in param_space.items()}

    return {"best_config": best_config, "trial_results": trial_results}


def _run_fold_with_config(
    data: HeteroData,
    node_names: dict[str, list[str]],
    task: str,
    fold_idx: int,
    train_val_pos: np.ndarray,
    test_pos: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    """Run a single CV fold with a given hyper-parameter configuration."""
    key, _ = abl.get_target_edge_index(data, task)
    src_type, _, dst_type = key
    rng = np.random.default_rng(SEED + fold_idx)

    # Split train_val into train / val
    n_train = int(len(train_val_pos) * 0.85)
    order = rng.permutation(len(train_val_pos))
    train_pos = train_val_pos[order[:n_train]]
    val_pos = train_val_pos[order[n_train:]]

    full_positives = np.vstack([train_pos, val_pos, test_pos])
    forbidden = set(map(tuple, full_positives.tolist()))
    n_neg_total = int(len(full_positives) * abl.NEGATIVE_RATIO)
    negatives = abl.sample_reliable_negatives(
        num_src=data[src_type].num_nodes,
        num_dst=data[dst_type].num_nodes,
        num_negatives=n_neg_total,
        forbidden=forbidden,
        exclude_self_loops=(task == "gp"),
        rng=rng,
    )
    neg_array = np.array(negatives)
    n_neg = len(neg_array)
    n_neg_train = int(n_neg * len(train_pos) / len(full_positives))
    n_neg_val = int(n_neg * len(val_pos) / len(full_positives))
    neg_order = rng.permutation(n_neg)
    neg_train = neg_array[neg_order[:n_neg_train]]
    neg_val = neg_array[neg_order[n_neg_train : n_neg_train + n_neg_val]]
    neg_test = neg_array[neg_order[n_neg_train + n_neg_val :]]

    def _to_tensor(edges_pos: np.ndarray, edges_neg: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.from_numpy(np.vstack([edges_pos, edges_neg]).T).long()
        labels = torch.cat([torch.ones(len(edges_pos)), torch.zeros(len(edges_neg))])
        return indices, labels

    train_idx, train_labels = _to_tensor(train_pos, neg_train)
    val_idx, val_labels = _to_tensor(val_pos, neg_val)
    test_idx, test_labels = _to_tensor(test_pos, neg_test)

    # Train graph: only training positive edges for the target relation
    edge_index = data[key].edge_index.cpu().numpy()
    pos_to_idx: dict[tuple[int, int], int] = {}
    seen_local: set[tuple[int, int]] = set()
    for i in range(edge_index.shape[1]):
        e = (int(edge_index[0, i]), int(edge_index[1, i]))
        if e not in seen_local:
            seen_local.add(e)
            pos_to_idx[e] = i
    train_pos_indices = torch.tensor(
        [pos_to_idx[tuple(e.tolist())] for e in train_pos], dtype=torch.long
    )
    train_data = abl.build_train_graph(data, key, train_pos_indices)

    offsets = abl.GATLP._compute_offsets(data)
    homogeneous_edge_index = abl.to_homogeneous_edge_index(data, offsets).to(DEVICE)

    model = GATHGTLP(
        data,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(DEVICE)

    # Temporarily override training hyperparameters
    original_lr = abl.LR
    original_wd = abl.WEIGHT_DECAY
    abl.LR = config["lr"]
    abl.WEIGHT_DECAY = config["weight_decay"]
    try:
        val_metrics, _ = abl.train_model(
            model,
            data.to(DEVICE),
            train_data.to(DEVICE),
            train_idx.to(DEVICE),
            train_labels.to(DEVICE),
            val_idx.to(DEVICE),
            val_labels.to(DEVICE),
            src_type,
            dst_type,
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
        )
        test_metrics = abl.evaluate(
            model,
            data.to(DEVICE),
            test_idx.to(DEVICE),
            test_labels.to(DEVICE),
            src_type,
            dst_type,
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
        )
    finally:
        abl.LR = original_lr
        abl.WEIGHT_DECAY = original_wd

    return {
        "val_auc": val_metrics["auc"],
        "test_auc": test_metrics["auc"],
        "val_idx": val_idx,
        "val_labels": val_labels,
        "src_type": src_type,
        "dst_type": dst_type,
    }


def train_final_model(
    data: HeteroData,
    node_names: dict[str, list[str]],
    config: dict[str, Any],
    task: str = "ct",
    n_folds: int = 5,
) -> tuple[nn.Module, dict[str, Any]]:
    """Train a final GAT-HGT model using the best HPO config with full CV.

    The model state is taken from the fold with the highest validation AUC.
    """
    key, num_edges = abl.get_target_edge_index(data, task)
    src_type, _, dst_type = key
    edge_index = data[key].edge_index.cpu().numpy()
    seen: set[tuple[int, int]] = set()
    positives: list[tuple[int, int]] = []
    for i in range(edge_index.shape[1]):
        e = (int(edge_index[0, i]), int(edge_index[1, i]))
        if e not in seen:
            seen.add(e)
            positives.append(e)
    positives = np.array(positives)
    stratify = abl._stratify_labels(positives, data[src_type].num_nodes, task)  # type: ignore[attr-defined]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    best_val_auc = -np.inf
    best_state: dict[str, Any] | None = None
    best_val_idx: torch.Tensor | None = None
    best_val_labels: torch.Tensor | None = None
    best_src_type: str | None = None
    best_dst_type: str | None = None
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(positives, stratify)):
        fold_result = _run_fold_with_config(
            data, node_names, task, fold_idx, positives[train_val_idx], positives[test_idx], config
        )
        fold_metrics.append(fold_result)
        logger.info(
            "Final CV fold %d: val_auc=%.4f test_auc=%.4f",
            fold_idx,
            fold_result["val_auc"],
            fold_result["test_auc"],
        )

        if fold_result["val_auc"] > best_val_auc:
            best_val_auc = fold_result["val_auc"]
            # Re-train this fold to capture state
            model, val_idx, val_labels, src_type, dst_type = _train_single_fold_model(
                data, node_names, task, fold_idx, positives[train_val_idx], positives[test_idx], config
            )
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_idx = val_idx
            best_val_labels = val_labels
            best_src_type = src_type
            best_dst_type = dst_type

    final_model = GATHGTLP(
        data,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(DEVICE)
    if best_state is not None:
        final_model.load_state_dict(best_state)

    aggregated = {
        "val_auc_mean": float(np.mean([f["val_auc"] for f in fold_metrics])),
        "val_auc_std": float(np.std([f["val_auc"] for f in fold_metrics])),
        "test_auc_mean": float(np.mean([f["test_auc"] for f in fold_metrics])),
        "test_auc_std": float(np.std([f["test_auc"] for f in fold_metrics])),
    }
    calibration_info = {
        "val_idx": best_val_idx,
        "val_labels": best_val_labels,
        "src_type": best_src_type,
        "dst_type": best_dst_type,
    }
    return final_model, aggregated, calibration_info


def _train_single_fold_model(
    data: HeteroData,
    node_names: dict[str, list[str]],
    task: str,
    fold_idx: int,
    train_val_pos: np.ndarray,
    test_pos: np.ndarray,
    config: dict[str, Any],
) -> nn.Module:
    """Train a single model instance for state capture."""
    key, _ = abl.get_target_edge_index(data, task)
    src_type, _, dst_type = key
    rng = np.random.default_rng(SEED + fold_idx)

    n_train = int(len(train_val_pos) * 0.85)
    order = rng.permutation(len(train_val_pos))
    train_pos = train_val_pos[order[:n_train]]
    val_pos = train_val_pos[order[n_train:]]

    full_positives = np.vstack([train_pos, val_pos, test_pos])
    forbidden = set(map(tuple, full_positives.tolist()))
    n_neg_total = int(len(full_positives) * abl.NEGATIVE_RATIO)
    negatives = abl.sample_reliable_negatives(
        num_src=data[src_type].num_nodes,
        num_dst=data[dst_type].num_nodes,
        num_negatives=n_neg_total,
        forbidden=forbidden,
        exclude_self_loops=(task == "gp"),
        rng=rng,
    )
    neg_array = np.array(negatives)
    n_neg = len(neg_array)
    n_neg_train = int(n_neg * len(train_pos) / len(full_positives))
    n_neg_val = int(n_neg * len(val_pos) / len(full_positives))
    neg_order = rng.permutation(n_neg)
    neg_train = neg_array[neg_order[:n_neg_train]]
    neg_val = neg_array[neg_order[n_neg_train : n_neg_train + n_neg_val]]

    def _to_tensor(edges_pos: np.ndarray, edges_neg: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.from_numpy(np.vstack([edges_pos, edges_neg]).T).long()
        labels = torch.cat([torch.ones(len(edges_pos)), torch.zeros(len(edges_neg))])
        return indices, labels

    train_idx, train_labels = _to_tensor(train_pos, neg_train)
    val_idx, val_labels = _to_tensor(val_pos, neg_val)

    edge_index = data[key].edge_index.cpu().numpy()
    pos_to_idx: dict[tuple[int, int], int] = {}
    seen_local: set[tuple[int, int]] = set()
    for i in range(edge_index.shape[1]):
        e = (int(edge_index[0, i]), int(edge_index[1, i]))
        if e not in seen_local:
            seen_local.add(e)
            pos_to_idx[e] = i
    train_pos_indices = torch.tensor(
        [pos_to_idx[tuple(e.tolist())] for e in train_pos], dtype=torch.long
    )
    train_data = abl.build_train_graph(data, key, train_pos_indices)

    offsets = abl.GATLP._compute_offsets(data)
    homogeneous_edge_index = abl.to_homogeneous_edge_index(data, offsets).to(DEVICE)

    model = GATHGTLP(
        data,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(DEVICE)

    original_lr = abl.LR
    original_wd = abl.WEIGHT_DECAY
    abl.LR = config["lr"]
    abl.WEIGHT_DECAY = config["weight_decay"]
    try:
        abl.train_model(
            model,
            data.to(DEVICE),
            train_data.to(DEVICE),
            train_idx.to(DEVICE),
            train_labels.to(DEVICE),
            val_idx.to(DEVICE),
            val_labels.to(DEVICE),
            src_type,
            dst_type,
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
        )
    finally:
        abl.LR = original_lr
        abl.WEIGHT_DECAY = original_wd
    return model, val_idx, val_labels, src_type, dst_type


def predict_compound_target_scores(
    model: nn.Module,
    data: HeteroData,
    compound_indices: list[int],
    gene_indices: list[int],
    offsets: dict[str, int] | None = None,
    homogeneous_edge_index: torch.Tensor | None = None,
    batch_size: int = 1024,
    temperature: float = 1.0,
) -> np.ndarray:
    """Predict compound-target scores for all TCM compound × ferroptosis gene pairs.

    Returns
    -------
    scores : np.ndarray, shape (n_compounds, n_genes)
        Sigmoid probabilities for each pair.
    """
    model.eval()
    if offsets is None:
        offsets = abl.GATLP._compute_offsets(data)
    if homogeneous_edge_index is None:
        homogeneous_edge_index = abl.to_homogeneous_edge_index(data, offsets)

    data_dev = data.to(DEVICE)
    homogeneous_edge_index = homogeneous_edge_index.to(DEVICE)

    n_compounds = len(compound_indices)
    n_genes = len(gene_indices)
    scores = np.zeros((n_compounds, n_genes), dtype=np.float32)

    with torch.no_grad():
        z_dict = model(
            data_dev,
            edge_index_dict=data_dev.edge_index_dict,
            homogeneous_edge_index=homogeneous_edge_index,
        )
        cmp_emb_all = z_dict["compound"]
        gene_emb_all = z_dict["gene"]

        for i, cidx in enumerate(compound_indices):
            src_emb = cmp_emb_all[cidx].unsqueeze(0)
            for j_start in range(0, n_genes, batch_size):
                j_end = min(j_start + batch_size, n_genes)
                dst_emb = gene_emb_all[gene_indices[j_start:j_end]]
                src_emb_batch = src_emb.expand(dst_emb.shape[0], -1)
                logits = model.predictor(src_emb_batch, dst_emb)
                scores[i, j_start:j_end] = torch.sigmoid(logits / temperature).cpu().numpy().flatten()
    return scores


def _collect_logits(
    model: nn.Module,
    data: HeteroData,
    edge_idx: torch.Tensor,
    src_type: str,
    dst_type: str,
    offsets: dict[str, int],
    homogeneous_edge_index: torch.Tensor,
) -> torch.Tensor:
    """Collect link-prediction logits for a set of candidate edges."""
    model.eval()
    data_dev = data.to(DEVICE)
    homogeneous_edge_index = homogeneous_edge_index.to(DEVICE)
    with torch.no_grad():
        z_dict = model(
            data_dev,
            edge_index_dict=data_dev.edge_index_dict,
            homogeneous_edge_index=homogeneous_edge_index,
        )
        src_emb = z_dict[src_type][edge_idx[0]]
        dst_emb = z_dict[dst_type][edge_idx[1]]
        return model.predictor(src_emb, dst_emb).squeeze(-1)


def expected_calibration_error(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10,
    temperature: float = 1.0,
) -> float:
    """Compute Expected Calibration Error (ECE) for link-prediction logits."""
    probs = torch.sigmoid(logits / temperature).cpu().numpy()
    labels_np = labels.cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (probs == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        acc = labels_np[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return float(ece / len(probs))


def calibrate_temperature(
    model: nn.Module,
    data: HeteroData,
    val_idx: torch.Tensor,
    val_labels: torch.Tensor,
    src_type: str,
    dst_type: str,
    offsets: dict[str, int],
    homogeneous_edge_index: torch.Tensor,
    n_grid: int = 50,
) -> tuple[float, float, float]:
    """Grid-search temperature scaling on validation logits.

    Returns
    -------
    best_temperature, ece_before, ece_after
    """
    logits = _collect_logits(
        model, data, val_idx, src_type, dst_type, offsets, homogeneous_edge_index
    )
    ece_before = expected_calibration_error(logits, val_labels)

    best_t = 1.0
    best_ece = ece_before
    for t in np.linspace(0.5, 3.0, n_grid):
        ece = expected_calibration_error(logits, val_labels, temperature=t)
        if ece < best_ece:
            best_ece = ece
            best_t = float(t)

    logger.info(
        "Temperature calibration: ECE before=%.4f, ECE after=%.4f, T=%.3f",
        ece_before,
        best_ece,
        best_t,
    )
    return best_t, ece_before, best_ece


def screen_active_ingredients(
    scores: np.ndarray,
    compound_names: list[str],
    gene_names: list[str],
    bbb_scores: dict[str, tuple[float | None, str | None]],
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """Screen TCM monomers for active ingredients targeting ferroptosis genes.

    Ranking considers:
      - max predicted probability across ferroptosis targets
      - number of targets above threshold
      - mean probability of top-5 targets
      - BBB permeability
    """
    records = []
    for i, cmp in enumerate(compound_names):
        row_scores = scores[i]
        above = row_scores >= score_threshold
        n_targets = int(above.sum())
        max_score = float(row_scores.max())
        top5_mean = float(np.sort(row_scores)[-5:].mean()) if len(row_scores) >= 5 else float(row_scores.mean())
        bbb_score, bbb_pass = bbb_scores.get(cmp, (None, None))

        records.append({
            "compound": cmp,
            "max_target_score": max_score,
            "n_targets_above_threshold": n_targets,
            "top5_mean_score": top5_mean,
            "bbb_score": bbb_score,
            "bbb_pass": bbb_pass,
        })

    df = pd.DataFrame(records)
    # Normalise ranking components
    df["max_target_score_norm"] = (df["max_target_score"] - df["max_target_score"].min()) / (
        df["max_target_score"].max() - df["max_target_score"].min() + 1e-9
    )
    df["n_targets_norm"] = (df["n_targets_above_threshold"] - df["n_targets_above_threshold"].min()) / (
        df["n_targets_above_threshold"].max() - df["n_targets_above_threshold"].min() + 1e-9
    )
    df["top5_mean_norm"] = (df["top5_mean_score"] - df["top5_mean_score"].min()) / (
        df["top5_mean_score"].max() - df["top5_mean_score"].min() + 1e-9
    )
    df["bbb_score_norm"] = 0.0
    if df["bbb_score"].notna().any():
        df.loc[df["bbb_score"].notna(), "bbb_score_norm"] = (
            df.loc[df["bbb_score"].notna(), "bbb_score"] - df["bbb_score"].min()
        ) / (df["bbb_score"].max() - df["bbb_score"].min() + 1e-9)

    df["activity_score"] = (
        0.35 * df["max_target_score_norm"]
        + 0.25 * df["n_targets_norm"]
        + 0.25 * df["top5_mean_norm"]
        + 0.15 * df["bbb_score_norm"]
    )
    df = df.sort_values("activity_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def analyze_mechanisms(
    scores: np.ndarray,
    compound_names: list[str],
    gene_names: list[str],
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """Analyse the predicted mechanism of each TCM monomer via pathway mapping."""
    try:
        pw_df = load_pathway_enrichment()
    except FileNotFoundError:
        logger.exception("Pathway enrichment file missing; cannot analyse mechanisms")
        raise

    gene_col = "gene" if "gene" in pw_df.columns else next((c for c in pw_df.columns if "gene" in c), None)
    pw_col = "pathway" if "pathway" in pw_df.columns else next((c for c in pw_df.columns if "pathway" in c), None)
    if gene_col is None or pw_col is None:
        raise ValueError("Pathway enrichment table lacks gene/pathway columns")

    records = []
    for i, cmp in enumerate(compound_names):
        active_genes = [gene_names[j] for j in range(len(gene_names)) if scores[i, j] >= score_threshold]
        if not active_genes:
            records.append({
                "compound": cmp,
                "n_active_targets": 0,
                "top_pathway": "",
                "pathway_count": 0,
                "mechanism_categories": "",
            })
            continue

        sub = pw_df[pw_df[gene_col].isin(active_genes)]
        pathway_counts = sub[pw_col].value_counts()
        top_pathway = pathway_counts.index[0] if not pathway_counts.empty else ""

        # Categorise mechanisms by gene function
        cats = set()
        active_set = {g.upper() for g in active_genes}
        if active_set & {"GPX4", "SLC7A11", "GSS", "GSR"}:
            cats.add("GSH/GPX4 axis")
        if active_set & {"TFRC", "FTH1", "FTL", "NCOA4", "SLC11A2", "SLC40A1"}:
            cats.add("iron metabolism")
        if active_set & {"NFE2L2", "KEAP1", "HMOX1", "NQO1"}:
            cats.add("Nrf2/ARE antioxidant")
        if active_set & {"ACSL4", "ACSL3", "LPCAT3", "ALOX15", "PTGS2"}:
            cats.add("lipid peroxidation")
        if active_set & {"SOD1", "SOD2", "CAT", "TXNRD1"}:
            cats.add("ROS scavenging")
        if active_set & {"IL1B", "IL6", "TNF", "CXCL8", "NLRP3", "NFKB1", "RELA"}:
            cats.add("anti-inflammatory")
        if active_set & {"TP53", "CDKN1A"}:
            cats.add("p53/CDKN1A stress")
        if active_set & {"HIF1A"}:
            cats.add("HIF-1 hypoxia response")

        records.append({
            "compound": cmp,
            "n_active_targets": len(active_genes),
            "active_targets": "; ".join(active_genes),
            "top_pathway": top_pathway,
            "pathway_count": int(pathway_counts.shape[0]),
            "top_pathway_n_genes": int(pathway_counts.iloc[0]) if not pathway_counts.empty else 0,
            "mechanism_categories": "; ".join(sorted(cats)),
        })

    return pd.DataFrame(records)


def evaluate_treatment_effects(
    activity_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
    scores: np.ndarray,
    compound_names: list[str],
    gene_names: list[str],
    bbb_scores: dict[str, tuple[float | None, str | None]],
) -> pd.DataFrame:
    """Evaluate CIRI treatment potential by combining activity, mechanism and BBB."""
    hub_indices = [j for j, g in enumerate(gene_names) if g in CIRI_FERROPTOSIS_HUBS]

    records = []
    for i, cmp in enumerate(compound_names):
        row_scores = scores[i]
        hub_score = float(row_scores[hub_indices].max()) if hub_indices else 0.0
        coverage = float(row_scores.mean())

        activity_row = activity_df[activity_df["compound"] == cmp]
        mechanism_row = mechanism_df[mechanism_df["compound"] == cmp]

        activity_score = float(activity_row["activity_score"].iloc[0]) if not activity_row.empty else 0.0
        n_active = int(mechanism_row["n_active_targets"].iloc[0]) if not mechanism_row.empty else 0
        n_categories = (
            len(str(mechanism_row["mechanism_categories"].iloc[0]).split("; "))
            if not mechanism_row.empty and mechanism_row["mechanism_categories"].iloc[0]
            else 0
        )
        bbb_score, bbb_pass = bbb_scores.get(cmp, (None, None))
        bbb_bonus = 0.1 if bbb_pass and str(bbb_pass).lower() in {"true", "yes", "likely", "pass"} else 0.0

        treatment_score = (
            0.35 * activity_score
            + 0.20 * hub_score
            + 0.15 * coverage
            + 0.15 * min(n_categories / 5.0, 1.0)
            + 0.10 * min(n_active / 10.0, 1.0)
            + bbb_bonus
        )

        level = "high" if treatment_score >= 0.6 else ("medium" if treatment_score >= 0.4 else "low")
        records.append({
            "compound": cmp,
            "activity_score": activity_score,
            "hub_target_score": hub_score,
            "mean_target_coverage": coverage,
            "n_active_targets": n_active,
            "n_mechanism_categories": n_categories,
            "bbb_pass": bbb_pass,
            "treatment_score": treatment_score,
            "potential_level": level,
        })

    df = pd.DataFrame(records).sort_values("treatment_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def build_focused_networkx_graph(
    scores: np.ndarray,
    compound_names: list[str],
    gene_names: list[str],
    score_threshold: float = 0.5,
    top_k_compounds: int = 15,
) -> nx.Graph:
    """Build a NetworkX graph of top compounds and their predicted targets."""
    g = nx.Graph()

    # Top compounds by max score
    max_scores = scores.max(axis=1)
    top_idx = np.argsort(-max_scores)[:top_k_compounds]

    for ci in top_idx:
        cmp = compound_names[ci]
        g.add_node(cmp, node_type="compound", score=float(max_scores[ci]))
        for gj, score in enumerate(scores[ci]):
            if score >= score_threshold:
                gene = gene_names[gj]
                g.add_node(gene, node_type="gene")
                g.add_edge(cmp, gene, weight=float(score), score=float(score))
    return g


def plot_network(
    g: nx.Graph,
    output_path: Path,
    figsize: tuple[int, int] = (14, 12),
) -> None:
    """Visualise the focused compound-target interaction network."""
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(g, seed=SEED, k=0.5, iterations=50)

    node_types = nx.get_node_attributes(g, "node_type")
    compound_nodes = [n for n, t in node_types.items() if t == "compound"]
    gene_nodes = [n for n, t in node_types.items() if t == "gene"]

    nx.draw_networkx_nodes(
        g, pos, nodelist=compound_nodes, node_color="coral", node_size=600, alpha=0.9, label="TCM monomer"
    )
    nx.draw_networkx_nodes(
        g, pos, nodelist=gene_nodes, node_color="skyblue", node_size=300, alpha=0.7, label="Ferroptosis target"
    )

    edges = g.edges(data=True)
    weights = [d["weight"] for _, _, d in edges]
    nx.draw_networkx_edges(g, pos, width=[1.5 * w for w in weights], alpha=0.5, edge_color="gray")

    nx.draw_networkx_labels(g, pos, font_size=8, font_family="sans-serif")
    plt.title("TCM monomer – ferroptosis target interaction network (GAT-HGT predictions)")
    plt.legend(loc="upper right")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Network plot saved: %s", output_path)


def plot_heatmap(
    scores: np.ndarray,
    compound_names: list[str],
    gene_names: list[str],
    output_path: Path,
    top_compounds: int = 20,
    top_genes: int = 25,
) -> None:
    """Plot compound-target prediction heatmap."""
    # Select top compounds and most-targeted genes for readability
    cmp_order = np.argsort(-scores.max(axis=1))[:top_compounds]
    gene_order = np.argsort(-scores.max(axis=0))[:top_genes]

    sub_scores = scores[np.ix_(cmp_order, gene_order)]
    df = pd.DataFrame(
        sub_scores,
        index=[compound_names[i] for i in cmp_order],
        columns=[gene_names[j] for j in gene_order],
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(df, cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Predicted probability"}, vmin=0, vmax=1)
    plt.title("TCM monomer – ferroptosis target prediction heatmap")
    plt.xlabel("Ferroptosis target gene")
    plt.ylabel("TCM monomer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Heatmap saved: %s", output_path)


def plot_top_compounds(
    treatment_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Bar plot of top-ranked TCM monomers by treatment score."""
    top = treatment_df.head(top_k).copy()
    plt.figure(figsize=(10, 8))
    colors = {"high": "#d62728", "medium": "#ff7f0e", "low": "#1f77b4"}
    sns.barplot(
        data=top,
        y="compound",
        x="treatment_score",
        palette=colors,
        hue="potential_level",
        hue_order=["high", "medium", "low"],
        legend=True,
    )
    plt.xlim(0, 1.2)
    plt.title(f"Top {top_k} TCM monomers for ferroptosis-related CIRI therapy")
    plt.xlabel("Treatment potential score")
    plt.ylabel("TCM monomer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Top compounds bar plot saved: %s", output_path)


def plot_mechanism_distribution(
    mechanism_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 15,
) -> None:
    """Stacked bar chart of mechanism categories for top compounds."""
    top = treatment_df.head(top_k)["compound"].tolist()
    sub = mechanism_df[mechanism_df["compound"].isin(top)].copy()

    all_categories = set()
    for cats in sub["mechanism_categories"].dropna():
        all_categories.update(str(cats).split("; "))
    all_categories.discard("")
    all_categories = sorted(all_categories)

    matrix = []
    for _, row in sub.iterrows():
        cats = set(str(row["mechanism_categories"]).split("; ")) if pd.notna(row["mechanism_categories"]) else set()
        cats.discard("")
        matrix.append([1 if c in cats else 0 for c in all_categories])
    plot_df = pd.DataFrame(matrix, index=sub["compound"], columns=all_categories)

    plot_df.plot(kind="barh", stacked=True, figsize=(10, 8), colormap="tab20")
    plt.title(f"Mechanism categories of top {top_k} TCM monomers")
    plt.xlabel("Mechanism category present")
    plt.ylabel("TCM monomer")
    plt.legend(title="Mechanism", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Mechanism distribution plot saved: %s", output_path)


def build_bcp_summary(
    scores: np.ndarray,
    compound_names: list[str],
    gene_names: list[str],
    activity_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    bbb_scores: dict[str, tuple[float | None, str | None]],
    score_threshold: float = 0.5,
    top_k_targets: int = 10,
) -> dict[str, Any]:
    """Build a focused interpretability summary for BCP (β-caryophyllene)."""
    if "BCP" not in compound_names:
        logger.warning("BCP not found in TCM monomer list")
        return {"error": "BCP not found in compound list"}

    idx = compound_names.index("BCP")
    row_scores = scores[idx]
    top_target_indices = np.argsort(-row_scores)[:top_k_targets]
    top_targets = [
        {"gene": gene_names[j], "score": float(row_scores[j])}
        for j in top_target_indices
    ]

    activity_row = activity_df[activity_df["compound"] == "BCP"]
    mechanism_row = mechanism_df[mechanism_df["compound"] == "BCP"]
    treatment_row = treatment_df[treatment_df["compound"] == "BCP"]

    bbb_score, bbb_pass = bbb_scores.get("BCP", (None, None))
    n_active = int(mechanism_row["n_active_targets"].iloc[0]) if not mechanism_row.empty else 0
    categories = (
        str(mechanism_row["mechanism_categories"].iloc[0]).split("; ")
        if not mechanism_row.empty and mechanism_row["mechanism_categories"].iloc[0]
        else []
    )

    return {
        "compound": "BCP",
        "activity_rank": int(activity_row["rank"].iloc[0]) if not activity_row.empty else None,
        "treatment_rank": int(treatment_row["rank"].iloc[0]) if not treatment_row.empty else None,
        "activity_score": float(activity_row["activity_score"].iloc[0]) if not activity_row.empty else None,
        "treatment_score": float(treatment_row["treatment_score"].iloc[0]) if not treatment_row.empty else None,
        "potential_level": str(treatment_row["potential_level"].iloc[0]) if not treatment_row.empty else None,
        "bbb_score": bbb_score,
        "bbb_pass": bbb_pass,
        "n_active_targets": n_active,
        "mechanism_categories": [c for c in categories if c],
        "top_predicted_targets": top_targets,
    }


def save_outputs(
    activity_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    cv_metrics: dict[str, Any],
    hpo_results: dict[str, Any],
    calibration: dict[str, Any],
    bcp_summary: dict[str, Any],
) -> None:
    """Save CSV/JSON outputs to L3_results/tcm_ferroptosis_ciri_gnn/."""
    activity_df.to_csv(OUTPUT_DIR / "active_ingredient_screening.csv", index=False, encoding="utf-8-sig")
    mechanism_df.to_csv(OUTPUT_DIR / "mechanism_analysis.csv", index=False, encoding="utf-8-sig")
    treatment_df.to_csv(OUTPUT_DIR / "treatment_effect_evaluation.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "cv_metrics": cv_metrics,
        "hpo_results": hpo_results,
        "calibration": calibration,
        "bcp_summary": bcp_summary,
    }
    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("All outputs saved to %s", OUTPUT_DIR)


def main() -> int:
    """Run the full TCM-ferroptosis-CIRI GNN prediction pipeline."""
    set_seed()
    try:
        logger.info("=" * 60)
        logger.info("TCM monomer × ferroptosis × CIRI GNN prediction module")
        logger.info("Device: %s", DEVICE)

        # 1. Load graph and focused subgraph info
        data, node_names = abl.build_data()
        tcm_compounds = load_tcm_monomers()
        subgraph_info = build_focused_subgraph_info(data, node_names, tcm_compounds)

        compound_names = [node_names["compound"][i] for i in subgraph_info["compound_indices"]]
        gene_names = [node_names["gene"][i] for i in subgraph_info["ferroptosis_gene_indices"]]

        # 2. Hyper-parameter optimisation
        logger.info("Starting hyper-parameter search...")
        hpo_results = hyperparameter_search(data, node_names, task="ct", n_trials=8, n_folds=3)
        best_config = hpo_results["best_config"]
        logger.info("Best HPO config: %s", best_config)

        # 3. Train final GAT-HGT model with best config
        logger.info("Training final GAT-HGT model...")
        final_model, cv_metrics, calibration_info = train_final_model(
            data, node_names, best_config, task="ct", n_folds=5
        )
        logger.info(
            "Final CV metrics: val_auc=%.4f±%.4f test_auc=%.4f±%.4f",
            cv_metrics["val_auc_mean"],
            cv_metrics["val_auc_std"],
            cv_metrics["test_auc_mean"],
            cv_metrics["test_auc_std"],
        )

        # 4. Calibrate predicted probabilities on the best-fold validation set
        offsets = abl.GATLP._compute_offsets(data)
        homogeneous_edge_index = abl.to_homogeneous_edge_index(data, offsets)
        temperature = 1.0
        ece_before = None
        ece_after = None
        if (
            calibration_info.get("val_idx") is not None
            and calibration_info.get("val_labels") is not None
        ):
            temperature, ece_before, ece_after = calibrate_temperature(
                final_model,
                data,
                calibration_info["val_idx"].to(DEVICE),
                calibration_info["val_labels"].to(DEVICE),
                calibration_info["src_type"],
                calibration_info["dst_type"],
                offsets,
                homogeneous_edge_index,
            )
        calibration = {
            "temperature": temperature,
            "ece_before": ece_before,
            "ece_after": ece_after,
        }

        # 5. Predict compound-target scores for focused subgraph
        scores = predict_compound_target_scores(
            final_model,
            data,
            subgraph_info["compound_indices"],
            subgraph_info["ferroptosis_gene_indices"],
            offsets=offsets,
            homogeneous_edge_index=homogeneous_edge_index,
            temperature=temperature,
        )

        # 6. Screening, mechanism analysis and treatment evaluation
        bbb_scores = load_bbb_scores()
        activity_df = screen_active_ingredients(scores, compound_names, gene_names, bbb_scores)
        mechanism_df = analyze_mechanisms(scores, compound_names, gene_names)
        treatment_df = evaluate_treatment_effects(
            activity_df, mechanism_df, scores, compound_names, gene_names, bbb_scores
        )

        # 7. BCP-focused interpretability summary
        bcp_summary = build_bcp_summary(
            scores, compound_names, gene_names, activity_df, mechanism_df, treatment_df, bbb_scores
        )

        # 8. Visualisations
        network_g = build_focused_networkx_graph(scores, compound_names, gene_names)
        plot_network(network_g, OUTPUT_DIR / "tcm_ferroptosis_network.png")
        plot_heatmap(scores, compound_names, gene_names, OUTPUT_DIR / "compound_target_heatmap.png")
        plot_top_compounds(treatment_df, OUTPUT_DIR / "top_tcm_compounds.png")
        plot_mechanism_distribution(mechanism_df, treatment_df, OUTPUT_DIR / "mechanism_distribution.png")

        # 9. Save outputs
        save_outputs(activity_df, mechanism_df, treatment_df, cv_metrics, hpo_results, calibration, bcp_summary)

        logger.info("Pipeline completed successfully. Outputs in %s", OUTPUT_DIR)
        return 0
    except Exception:
        logger.error("Pipeline failed")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
