#!/usr/bin/env python
"""
ingest_table1_sources.py — load the runs behind tab:ra_mean_adult into results.db.

Six rows of that table were produced by sweeps that wrote a result CSV and were
never loaded into the database. This script loads them, so that the whole table
rebuilds from results.db like every other table:

  row              columns      source CSV                                  stored as
  CoBP-RA          non-MST      marginalrf_combos_20260514_040431.csv,      MarginalRF_graphQI_entropyBP
                                marginalrf_combos_fill_20260521_182227.csv
  MultiHeadMLP     non-MST      joint_mlp_adult10k_20260529_084608.csv      JointMLP
  CondDDPM         non-MST      sweep_results_adult_20260305_211106.csv     TabDDPM
  CondRePaint      non-MST      sweep_results_adult_20260305_211106.csv     ConditionedRePaint
  Best chain       non-MST      marginalrf_chains_ensembles_20260529_190349.csv   CoBP-RA_HardChain
                   MST          marginalrf_chains_ensembles_20260911_192758.csv
  Best ensemble    non-MST      marginalrf_chains_ensembles_20260529_190349.csv   Ensemble_MRF_5
                   MST          marginalrf_chains_ensembles_20260911_221015.csv

All runs are Adult, 10k rows, QI1, five training samples. The MST columns of the
first four rows are already in the database.

Where the database already holds a different run under the same key, that run is
moved to `runs_superseded` and the one listed above takes its place. Runs flagged
'broken' are moved there too. Re-running the script changes nothing.

Usage:
    python experiment_scripts/historical/ingest_table1_sources.py
"""
import sys
from pathlib import Path

import pandas as pd

E = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(E))
from results_db import ResultsDB  # noqa: E402

NON_MST = ["RankSwap", "CellSuppression", "Synthpop", "TVAE", "CTGAN", "ARF", "TabDDPM", "AIM_eps1"]
MST = ["MST_eps0.1", "MST_eps1", "MST_eps10", "MST_eps100", "MST_eps1000"]
KEY = dict(dataset="adult", dataset_size=10000, qi="QI1", split="standard")

# (csv, label column, label in the CSV, generators to take, label in results.db)
SOURCES = [
    ("marginalrf_combos_20260514_040431.csv", "label", "MarginalRF_QIGraph_EntropyBP", NON_MST, "MarginalRF_graphQI_entropyBP"),
    ("marginalrf_combos_fill_20260521_182227.csv", "label", "MarginalRF_QIGraph_EntropyBP", NON_MST, "MarginalRF_graphQI_entropyBP"),
    ("joint_mlp_adult10k_20260529_084608.csv", "attack", "JointMLP", NON_MST, "JointMLP"),
    ("sweep_results_adult_20260305_211106.csv", "attack", "TabDDPM", NON_MST, "TabDDPM"),
    ("sweep_results_adult_20260305_211106.csv", "attack", "ConditionedRePaint", NON_MST, "ConditionedRePaint"),
    ("marginalrf_chains_ensembles_20260529_190349.csv", "label", "MarginalRF_HardChain", NON_MST, "CoBP-RA_HardChain"),
    ("marginalrf_chains_ensembles_20260911_192758.csv", "label", "CoBP-RA_HardChain", MST, "CoBP-RA_HardChain"),
    ("marginalrf_chains_ensembles_20260529_190349.csv", "label", "Ensemble_MRF_5", NON_MST, "Ensemble_MRF_5"),
    ("marginalrf_chains_ensembles_20260911_221015.csv", "label", "Ensemble_MRF_5", MST, "Ensemble_MRF_5"),
]


def supersede(conn, run_id):
    """Move one run and its per-feature scores to the *_superseded tables."""
    conn.execute("INSERT OR REPLACE INTO runs_superseded SELECT *, datetime('now') FROM runs WHERE run_id = ?", (run_id,))
    conn.execute("INSERT OR REPLACE INTO feature_scores_superseded SELECT * FROM feature_scores WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM feature_scores WHERE run_id = ?", (run_id,))
    conn.execute("DELETE FROM conflicts WHERE existing_run_id = ?", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def main():
    db = ResultsDB()
    conn = db._conn
    added = replaced = same = 0
    for csv, col, csv_label, gens, db_label in SOURCES:
        d = pd.read_csv(E / csv)
        d = d[(d[col] == csv_label) & (d.qi == "QI1") & d.sdg.isin(gens) & d.ra_mean.notna()]
        if "size" in d.columns:
            d = d[(d.dataset == "adult") & (d["size"] == 10000)]
        feats = [c for c in d.columns if c.startswith("RA_")]
        for row in d.to_dict("records"):
            key = (KEY["dataset"], KEY["dataset_size"], int(row["sample"]), KEY["qi"], row["sdg"], db_label, KEY["split"])
            old = conn.execute(
                "SELECT run_id, ra_mean FROM runs WHERE dataset=? AND dataset_size=? AND sample=? "
                "AND qi=? AND sdg_method=? AND attack_label=? AND split=?", key).fetchone()
            if old is not None:
                if old[1] is not None and abs(old[1] - row["ra_mean"]) < 1e-3:
                    same += 1
                    continue
                supersede(conn, old[0])
                replaced += 1
            else:
                added += 1
            scores = {c[3:]: float(row[c]) for c in feats if pd.notna(row[c])} or None
            db.insert_run(sample=int(row["sample"]), sdg_method=row["sdg"], attack_label=db_label,
                          ra_mean=float(row["ra_mean"]), feature_scores=scores, source_file=csv, **KEY)
    # Seven CondMST (bounded) runs on sample 4 carry the 'broken' flag of the W&B
    # group they were logged under. That flag marks runs whose score does not vary
    # with the generator; these vary (17.7 on ARF to 27.8 on Cell Suppression), and
    # the table uses them.
    cur = conn.execute(
        "UPDATE runs SET confidence='certain', confidence_notes=NULL "
        "WHERE dataset=? AND dataset_size=? AND qi=? AND split=? "
        "AND attack_label='PartialMSTBounded' AND confidence='broken'",
        (KEY["dataset"], KEY["dataset_size"], KEY["qi"], KEY["split"]))
    # Every other run still flagged 'broken' belongs to that W&B group and is read by
    # nothing. Move them out, so that `runs` holds only usable results.
    left = [r[0] for r in conn.execute("SELECT run_id FROM runs WHERE confidence='broken'")]
    for run_id in left:
        supersede(conn, run_id)
    conn.commit()
    db.close()
    print(f"added {added}, replaced {replaced}, already present {same}, "
          f"re-flagged {cur.rowcount}, moved out {len(left)}")


if __name__ == "__main__":
    main()
