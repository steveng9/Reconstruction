#!/usr/bin/env python
"""
archive_superseded_runs.py

Move rows marked confidence='superseded_encoding' out of `runs` into
`runs_superseded` (and their per-feature scores into `feature_scores_superseded`).

WHY MOVE RATHER THAN JUST RELABEL
---------------------------------
`runs` carries UNIQUE(dataset, dataset_size, sample, qi, sdg_method,
attack_label, split). ResultsDB.insert_run() treats a collision on that key as a
duplicate and *skips the insert*, keeping the existing row. So as long as the
stale float-binned rows sit in `runs`, every corrected re-run would be silently
discarded and the wrong numbers would remain — the exact failure mode we are
trying to eliminate. Moving them frees the key.

Nothing is lost: the archive tables keep the full row plus the reason and the
archival timestamp, so the old numbers stay auditable, they simply can no longer
be served to a query or block a corrected result.

USAGE
    python experiment_scripts/archive_superseded_runs.py --dry-run
    python experiment_scripts/archive_superseded_runs.py
"""

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("results.db")
MARK = "superseded_encoding"


def ensure_archive_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS runs_superseded (
        run_id INTEGER PRIMARY KEY,
        dataset TEXT, dataset_size INTEGER, sample INTEGER, qi TEXT,
        sdg_method TEXT, attack_label TEXT, split TEXT, ra_mean REAL,
        attack_params_json TEXT, sdg_params_json TEXT, source_file TEXT,
        wandb_run_id TEXT, wandb_group TEXT, wandb_project TEXT,
        ingested_at TEXT, confidence TEXT, confidence_notes TEXT,
        archived_at TEXT
    );
    CREATE TABLE IF NOT EXISTS feature_scores_superseded (
        score_id INTEGER PRIMARY KEY,
        run_id INTEGER, feature TEXT, ra_score REAL
    );
    """)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    ensure_archive_tables(conn)

    n_runs = conn.execute("SELECT COUNT(*) FROM runs WHERE confidence=?",
                          (MARK,)).fetchone()[0]
    n_feat = conn.execute(
        "SELECT COUNT(*) FROM feature_scores WHERE run_id IN "
        "(SELECT run_id FROM runs WHERE confidence=?)", (MARK,)).fetchone()[0]

    print(f"rows to archive : {n_runs} runs, {n_feat} feature scores")
    if args.dry_run:
        print("(dry run — nothing written)")
        return
    if not n_runs:
        print("nothing to do")
        return

    conn.execute("""
        INSERT OR REPLACE INTO runs_superseded
        SELECT run_id, dataset, dataset_size, sample, qi, sdg_method, attack_label,
               split, ra_mean, attack_params_json, sdg_params_json, source_file,
               wandb_run_id, wandb_group, wandb_project, ingested_at, confidence,
               confidence_notes, datetime('now')
        FROM runs WHERE confidence=?""", (MARK,))
    conn.execute("""
        INSERT OR REPLACE INTO feature_scores_superseded
        SELECT score_id, run_id, feature, ra_score FROM feature_scores
        WHERE run_id IN (SELECT run_id FROM runs WHERE confidence=?)""", (MARK,))
    conn.execute(
        "DELETE FROM feature_scores WHERE run_id IN "
        "(SELECT run_id FROM runs WHERE confidence=?)", (MARK,))
    conn.execute("DELETE FROM runs WHERE confidence=?", (MARK,))
    conn.commit()

    print(f"archived  : {conn.execute('SELECT COUNT(*) FROM runs_superseded').fetchone()[0]} runs")
    print(f"remaining : {conn.execute('SELECT COUNT(*) FROM runs').fetchone()[0]} runs in `runs`")
    print("\nThe UNIQUE key is now free for corrected re-runs.")
    conn.close()


if __name__ == "__main__":
    main()
