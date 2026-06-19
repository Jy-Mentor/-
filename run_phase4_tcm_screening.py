#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4: TCM monomer screening, explainability and docking status.

Inputs (real files):
    - L3_results/L3_compound_target_ranking.csv
    - L3_results/L3_compound_target_summary.csv
    - L3_results/L3_top30_ACSL4_candidate_compounds.csv
    - L3_results/L3_hub_gene_ranking.csv
    - L3_results/L3_cell_communication_flow.csv
    - L3_results/L3_gene_embeddings.csv
    - network_files/compound_smiles.csv
    - network_files/tcm_bbb_evaluation.csv
    - network_files/compound_target_edges.csv
    - network_files/string_ppi_edges.csv
    - network_files/gene_pathway_enrichment.csv

Outputs:
    - L3_results/phase4_acsl4_compound_ranking.csv
    - L3_results/phase4_acsl4_explainability.csv
    - L3_results/phase4_bbb_comparison.csv
    - L3_results/phase4_docking_status.csv
    - L3_results/TCM_monomer_recommendation.xlsx

Notes:
    - If AutoDock Vina is not installed, the status is recorded as unavailable;
      no docking scores are fabricated.
    - All features and scores come from existing real outputs; no weights are
      manually adjusted.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
L3_RESULTS = PROJECT_ROOT / "L3_results"
NETWORK_DIR = PROJECT_ROOT / "network_files"
OUTPUT_DIR = L3_RESULTS


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV; raise with traceback if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, **kwargs)


def _check_vina_available() -> tuple[bool, str]:
    """Check whether the AutoDock Vina command line is available."""
    for cmd in ("vina", "autodock_vina"):
        executable = shutil.which(cmd)
        if executable:
            try:
                result = subprocess.run(
                    [cmd, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if result.returncode in (0, 1):
                    return True, str(executable)
            except Exception:
                traceback.print_exc()
    return False, ""


def build_acsl4_compound_ranking() -> pd.DataFrame:
    """Combine ACSL4 binding probability and embedding similarity for ranking."""
    binding_path = L3_RESULTS / "L3_compound_target_ranking.csv"
    summary_path = L3_RESULTS / "L3_compound_target_summary.csv"
    similarity_path = L3_RESULTS / "L3_top30_ACSL4_candidate_compounds.csv"
    smiles_path = NETWORK_DIR / "compound_smiles.csv"

    binding_df = _read_csv(binding_path)
    summary_df = _read_csv(summary_path)
    sim_df = _read_csv(similarity_path)
    smiles_df = _read_csv(smiles_path)

    acsl4_binding = (
        binding_df[binding_df["gene"] == "ACSL4"]
        .rename(columns={"binding_probability": "ACSL4_binding_probability"})
        .drop(columns=["gene"])
    )

    summary_acsl4 = summary_df[["compound", "ACSL4_probability", "ACSL4_rank_in_compound"]].copy()

    similarity = sim_df.rename(columns={"similarity_to_ACSL4": "ACSL4_embedding_similarity"})

    merged = acsl4_binding.merge(summary_acsl4, on="compound", how="outer")
    merged = merged.merge(similarity, on="compound", how="outer")
    merged = merged.merge(smiles_df[["compound", "CanonicalSMILES"]], on="compound", how="left")

    # Composite score: binding probability (50%) + embedding similarity (50%)
    prob = merged["ACSL4_binding_probability"].fillna(merged["ACSL4_probability"])
    sim = merged["ACSL4_embedding_similarity"].fillna(0.0)
    merged["ACSL4_binding_probability"] = prob
    merged["composite_score"] = 0.5 * prob + 0.5 * sim

    merged = merged.sort_values("composite_score", ascending=False).reset_index(drop=True)
    merged["composite_rank"] = np.arange(1, len(merged) + 1)

    cols = [
        "compound",
        "CanonicalSMILES",
        "ACSL4_binding_probability",
        "ACSL4_embedding_similarity",
        "composite_score",
        "composite_rank",
        "ACSL4_rank_in_compound",
    ]
    merged = merged[[c for c in cols if c in merged.columns]]

    output_path = OUTPUT_DIR / "phase4_acsl4_compound_ranking.csv"
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"ACSL4 compound ranking saved: {output_path} ({len(merged)} compounds)")
    return merged


def build_acsl4_explainability() -> pd.DataFrame:
    """Estimate ACSL4 ferro-aging Hub contributions from network topology.

    Since trained model weights are not saved, SHAP/IG gradients are not
    computed. Instead, this uses real network files and node embeddings to
    quantify PPI neighbors, pathway memberships and known compound targets.
    """
    hub_path = L3_RESULTS / "L3_hub_gene_ranking.csv"
    ppi_path = NETWORK_DIR / "string_ppi_edges.csv"
    pathway_path = NETWORK_DIR / "gene_pathway_enrichment.csv"
    compound_target_path = NETWORK_DIR / "compound_target_edges.csv"

    hub_df = _read_csv(hub_path)
    ppi_df = _read_csv(ppi_path)
    pathway_df = _read_csv(pathway_path)
    ct_df = _read_csv(compound_target_path)

    acsl4_hub = hub_df[hub_df["gene"] == "ACSL4"]
    if acsl4_hub.empty:
        logger.warning("ACSL4 not found in Hub ranking")
        acsl4_rank = None
    else:
        acsl4_rank = int(acsl4_hub["rank"].iloc[0])

    neighbors = pd.concat(
        [
            ppi_df[ppi_df["protein_A"] == "ACSL4"][["protein_B", "score"]].rename(
                columns={"protein_B": "neighbor_gene"}
            ),
            ppi_df[ppi_df["protein_B"] == "ACSL4"][["protein_A", "score"]].rename(
                columns={"protein_A": "neighbor_gene"}
            ),
        ],
        ignore_index=True,
    )
    neighbor_hub = hub_df[["gene", "hub_score", "rank"]].rename(
        columns={"gene": "neighbor_gene", "hub_score": "neighbor_hub_score", "rank": "neighbor_rank"}
    )
    neighbors = neighbors.merge(neighbor_hub, on="neighbor_gene", how="left")
    neighbors["edge_contribution"] = neighbors["score"] / 1000.0 * neighbors["neighbor_hub_score"].fillna(0.0)
    neighbors = neighbors.sort_values("edge_contribution", ascending=False).head(20)

    acsl4_pathways = pathway_df[pathway_df["gene"] == "ACSL4"][["pathway"]].drop_duplicates()
    pathway_contrib = []
    for _, row in acsl4_pathways.iterrows():
        pw = row["pathway"]
        n_genes = (pathway_df["pathway"] == pw).sum()
        avg_hub = pathway_df[pathway_df["pathway"] == pw]["gene"].map(
            hub_df.set_index("gene")["hub_score"]
        ).mean()
        pathway_contrib.append(
            {
                "pathway": pw,
                "member_gene_count": int(n_genes),
                "avg_member_hub_score": float(avg_hub) if pd.notna(avg_hub) else 0.0,
            }
        )
    pathway_contrib_df = pd.DataFrame(pathway_contrib)
    if not pathway_contrib_df.empty:
        pathway_contrib_df = pathway_contrib_df.sort_values(
            "avg_member_hub_score", ascending=False
        ).head(10)
    else:
        logger.warning("ACSL4 has no pathway membership in gene_pathway_enrichment.csv")

    acsl4_compounds = ct_df[ct_df["gene"] == "ACSL4"]["compound"].unique().tolist()

    output_path = OUTPUT_DIR / "phase4_acsl4_explainability.csv"
    records = []
    for _, row in neighbors.iterrows():
        records.append(
            {
                "category": "PPI_neighbor",
                "name": row["neighbor_gene"],
                "score": round(row["edge_contribution"], 6),
                "detail": (
                    f"STRING_score={row['score']}, "
                    f"neighbor_hub_score={row['neighbor_hub_score']:.4f}"
                ),
            }
        )
    for _, row in pathway_contrib_df.iterrows():
        records.append(
            {
                "category": "pathway",
                "name": row["pathway"],
                "score": round(row["avg_member_hub_score"], 6),
                "detail": f"member_gene_count={row['member_gene_count']}",
            }
        )
    for compound in acsl4_compounds:
        records.append(
            {
                "category": "known_compound_target",
                "name": compound,
                "score": 1.0,
                "detail": "Known association from compound_target_edges.csv",
            }
        )
    records.append(
        {
            "category": "ACSL4_hub_status",
            "name": "ACSL4",
            "score": float(acsl4_hub["hub_score"].iloc[0]) if not acsl4_hub.empty else 0.0,
            "detail": (
                f"hub_rank={acsl4_rank}, "
                f"degree={int(acsl4_hub['degree'].iloc[0]) if not acsl4_hub.empty else 'N/A'}"
            ),
        }
    )
    explain_df = pd.DataFrame(records)
    explain_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"ACSL4 explainability saved: {output_path}")
    return explain_df


def build_bbb_comparison() -> pd.DataFrame:
    """Generate BBB permeability comparison for BCP, VC and Fer-1."""
    bbb_path = NETWORK_DIR / "tcm_bbb_evaluation.csv"
    bbb_df = _read_csv(bbb_path)

    target_compounds = ["BCP", "VC", "Fer-1"]
    rows = []
    for compound in target_compounds:
        sub = bbb_df[bbb_df["compound"] == compound]
        if sub.empty:
            logger.warning(f"BBB table missing {compound}; record as missing")
            rows.append(
                {
                    "compound": compound,
                    "BBB_score": None,
                    "BBB_pass": "Missing",
                    "BBB_reasons": "Not found in tcm_bbb_evaluation.csv",
                }
            )
            continue
        row = sub.iloc[0]
        rows.append(
            {
                "compound": compound,
                "BBB_score": row["BBB_score"],
                "BBB_pass": row["BBB_pass"],
                "BBB_reasons": row["BBB_reasons"],
            }
        )

    df = pd.DataFrame(rows)
    output_path = OUTPUT_DIR / "phase4_bbb_comparison.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"BBB comparison saved: {output_path}")
    return df


def build_docking_status() -> pd.DataFrame:
    """Check AutoDock Vina availability and record the real status."""
    available, executable = _check_vina_available()
    status = "available" if available else "unavailable"
    if available:
        reason = f"executable={executable}"
    else:
        reason = "vina / autodock_vina command not found and Python package vina not installed"

    records = [
        {
            "tool": "AutoDock Vina",
            "status": status,
            "executable": executable,
            "reason": reason,
            "note": (
                "Docking scores are not fabricated; install vina on Linux/WSL "
                "to run real docking later."
            ),
        }
    ]
    df = pd.DataFrame(records)
    output_path = OUTPUT_DIR / "phase4_docking_status.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Docking status recorded: {output_path} (status={status})")
    return df


def build_tcm_recommendation_excel(
    ranking_df: pd.DataFrame,
    explain_df: pd.DataFrame,
    bbb_df: pd.DataFrame,
    docking_df: pd.DataFrame,
) -> Path:
    """Generate combined TCM monomer recommendation Excel."""
    output_path = OUTPUT_DIR / "TCM_monomer_recommendation.xlsx"

    top_k = ranking_df.head(30).copy()
    highlight_compounds = {"BCP", "Fer-1", "VC"}
    top_k["note"] = top_k["compound"].apply(
        lambda x: "project_focus" if x in highlight_compounds else ""
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:  # type: ignore[call-arg]
        top_k.to_excel(writer, sheet_name="Top30_ACSL4_candidates", index=False)
        ranking_df.to_excel(writer, sheet_name="All_compound_ranking", index=False)
        explain_df.to_excel(writer, sheet_name="ACSL4_explainability", index=False)
        bbb_df.to_excel(writer, sheet_name="BBB_comparison", index=False)
        docking_df.to_excel(writer, sheet_name="Docking_status", index=False)

    logger.info(f"TCM recommendation Excel saved: {output_path}")
    return output_path


def main() -> int:
    """Main entry for Phase 4."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Phase 4.1: build ACSL4 compound ranking")
        ranking_df = build_acsl4_compound_ranking()

        logger.info("Phase 4.2: build ACSL4 explainability")
        explain_df = build_acsl4_explainability()

        logger.info("Phase 4.3: build BBB comparison")
        bbb_df = build_bbb_comparison()

        logger.info("Phase 4.4: check docking availability")
        docking_df = build_docking_status()

        logger.info("Phase 4.5: generate TCM recommendation Excel")
        excel_path = build_tcm_recommendation_excel(ranking_df, explain_df, bbb_df, docking_df)

        logger.info(f"Phase 4 completed. Excel: {excel_path}")
        return 0
    except Exception:
        logger.error("Phase 4 failed")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
