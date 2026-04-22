#!/usr/bin/env python
"""
results_db.py — Central SQLite store for reconstruction attack experiment results.

Every result (attack × SDG × dataset × sample × QI) lives in one place with
provenance tracking, confidence labels, and conflict logging.  LaTeX-table
scripts query this DB instead of enumerating CSVs.

DB location
-----------
experiment_scripts/results.db  (same directory as this module)

Tables
------
runs
    One row per unique (dataset, dataset_size, sample, qi, sdg_method,
    attack_label, split).  Holds the aggregate RA score.

feature_scores
    One row per (run_id, feature).  Per-feature RA scores.

conflicts
    When a new result matches an existing key but scores differ the
    incoming row is *not* inserted; the discrepancy is logged here instead.

Python API
----------
    from experiment_scripts.results_db import ResultsDB

    db = ResultsDB()

    # Insert one result
    db.insert_run(
        dataset="adult", dataset_size=1000, sample=0,
        qi="QI1", sdg_method="MST_eps1", attack_label="RandomForest",
        ra_mean=18.65,
        feature_scores={"income": 51.4, "occupation": 12.3},
        source_file="sweep_results_adult_20260401.csv",
        wandb_run_id="abc123",
    )

    # Query for a latex table
    df = db.query(dataset="adult", dataset_size=1000, qi="QI1")
    # → DataFrame: sample, sdg, attack, qi, ra_mean, RA_income, RA_occupation, …

Direct SQL (pandas)
-------------------
    import sqlite3, pandas as pd
    conn = sqlite3.connect("experiment_scripts/results.db")
    df = pd.read_sql(
        "SELECT * FROM runs WHERE dataset='adult' AND dataset_size=1000",
        conn,
    )
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
DEFAULT_DB_PATH = _SCRIPT_DIR / "results.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Experiment key (UNIQUE constraint below)
    dataset             TEXT    NOT NULL,  -- 'adult', 'nist_sbo', 'cdc_diabetes', …
    dataset_size        INTEGER NOT NULL,  -- 1000, 10000, 100000, …
    sample              INTEGER NOT NULL,  -- 0–9  (training-set sample index)
    qi                  TEXT    NOT NULL,  -- 'QI1', 'QI_large', 'QI_binary_SEX', …
    sdg_method          TEXT    NOT NULL,  -- display label: 'MST_eps1', 'RankSwap', …
    attack_label        TEXT    NOT NULL,  -- display label used in tables:
                                           --   'RandomForest', 'MarginalRF_mst_local_100', …
    split               TEXT    NOT NULL   -- 'standard'     → ordinary RA score
                            DEFAULT 'standard',
                                           -- 'train'        → RA on training set (mem. test)
                                           -- 'nontraining'  → RA on holdout   (mem. test)

    -- Aggregate score
    ra_mean             REAL,

    -- Full params (JSON, for reference / future filtering — not part of key)
    attack_params_json  TEXT,
    sdg_params_json     TEXT,

    -- Provenance
    source_file         TEXT,   -- CSV basename, or 'wandb_only'
    wandb_run_id        TEXT,   -- WandB run ID, e.g. 'abc123de'
    wandb_group         TEXT,   -- WandB group name
    wandb_project       TEXT,   -- WandB project name
    ingested_at         TEXT,   -- ISO-8601 UTC timestamp

    -- Data-integrity label
    -- 'certain'   → all dimensions explicitly verified (WandB or explicit CSV columns)
    -- 'uncertain' → at least one dimension unverified.  These should NOT appear in a
    --               clean DB; uncertain rows are written to pending_review.csv instead.
    confidence          TEXT    NOT NULL DEFAULT 'certain',
    confidence_notes    TEXT,

    UNIQUE(dataset, dataset_size, sample, qi, sdg_method, attack_label, split)
);

CREATE TABLE IF NOT EXISTS feature_scores (
    score_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    feature     TEXT    NOT NULL,   -- 'income', 'occupation', 'F1', …
    ra_score    REAL,
    UNIQUE(run_id, feature)
);

-- Logged when a new result matches an existing key but the scores differ.
-- The new result is NEVER written to runs; this table is the paper trail.
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    existing_run_id          INTEGER REFERENCES runs(run_id),
    new_source_file          TEXT,
    new_wandb_run_id         TEXT,
    new_ra_mean              REAL,
    new_feature_scores_json  TEXT,
    conflict_type            TEXT,  -- 'score_mismatch' | 'exact_duplicate'
    score_diff               REAL,  -- |new_ra_mean − existing ra_mean| (NaN if one is NULL)
    logged_at                TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_dataset_size
    ON runs(dataset, dataset_size);
CREATE INDEX IF NOT EXISTS idx_runs_key
    ON runs(dataset, dataset_size, qi, sdg_method, attack_label);
CREATE INDEX IF NOT EXISTS idx_feature_run
    ON feature_scores(run_id);
"""

# Score tolerance for deciding "exact duplicate" vs "mismatch"
_SCORE_TOL = 0.001


# ---------------------------------------------------------------------------
# ResultsDB
# ---------------------------------------------------------------------------

class ResultsDB:
    """Thin wrapper around the SQLite results database.

    Thread-safe for reads; use a single writer at a time (WAL mode enables
    concurrent reads while a write is in progress).

    Parameters
    ----------
    db_path
        Path to the SQLite file.  Defaults to ``experiment_scripts/results.db``.
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # concurrent readers
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert_run(
        self,
        *,
        dataset: str,
        dataset_size: int,
        sample: int,
        qi: str,
        sdg_method: str,
        attack_label: str,
        split: str = "standard",
        ra_mean: Optional[float],
        feature_scores: Optional[dict[str, float]] = None,
        attack_params: Optional[dict] = None,
        sdg_params: Optional[dict] = None,
        source_file: Optional[str] = None,
        wandb_run_id: Optional[str] = None,
        wandb_group: Optional[str] = None,
        wandb_project: Optional[str] = None,
        confidence: str = "certain",
        confidence_notes: Optional[str] = None,
    ) -> Optional[int]:
        """Insert one experiment result.  Returns the new run_id, or None if skipped.

        Duplicate handling
        ------------------
        If the UNIQUE key already exists:
        - Scores within ``_SCORE_TOL`` → logged as 'exact_duplicate', insertion skipped.
        - Scores differ → logged as 'score_mismatch', insertion skipped.

        In both cases the existing row is preserved unchanged.
        """
        now = datetime.now(timezone.utc).isoformat()
        try:
            cur = self._conn.execute(
                """
                INSERT INTO runs
                    (dataset, dataset_size, sample, qi, sdg_method, attack_label, split,
                     ra_mean, attack_params_json, sdg_params_json,
                     source_file, wandb_run_id, wandb_group, wandb_project,
                     ingested_at, confidence, confidence_notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    dataset, dataset_size, sample, qi, sdg_method, attack_label, split,
                    ra_mean,
                    json.dumps(attack_params) if attack_params else None,
                    json.dumps(sdg_params)    if sdg_params    else None,
                    source_file, wandb_run_id, wandb_group, wandb_project,
                    now, confidence, confidence_notes,
                ),
            )
            run_id = cur.lastrowid

            if feature_scores:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO feature_scores (run_id, feature, ra_score) "
                    "VALUES (?,?,?)",
                    [(run_id, feat, float(score) if score is not None else None)
                     for feat, score in feature_scores.items()],
                )

            self._conn.commit()
            return run_id

        except sqlite3.IntegrityError:
            existing = self._conn.execute(
                "SELECT run_id, ra_mean FROM runs "
                "WHERE dataset=? AND dataset_size=? AND sample=? AND qi=? "
                "AND sdg_method=? AND attack_label=? AND split=?",
                (dataset, dataset_size, sample, qi, sdg_method, attack_label, split),
            ).fetchone()

            if existing is None:
                return None  # shouldn't happen

            ex_id, ex_ra = existing["run_id"], existing["ra_mean"]

            if ex_ra is None and ra_mean is None:
                diff, conflict_type = 0.0, "exact_duplicate"
            elif ex_ra is None or ra_mean is None:
                diff, conflict_type = float("nan"), "score_mismatch"
            else:
                diff = abs(float(ra_mean) - float(ex_ra))
                conflict_type = "exact_duplicate" if diff < _SCORE_TOL else "score_mismatch"

            self._conn.execute(
                """
                INSERT INTO conflicts
                    (existing_run_id, new_source_file, new_wandb_run_id, new_ra_mean,
                     new_feature_scores_json, conflict_type, score_diff, logged_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    ex_id, source_file, wandb_run_id, ra_mean,
                    json.dumps(feature_scores) if feature_scores else None,
                    conflict_type, diff, now,
                ),
            )
            self._conn.commit()
            return None

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        dataset: Optional[str] = None,
        dataset_size: Optional[int] = None,
        qi: Optional[str] = None,
        split: str = "standard",
        confidence: Optional[str] = "certain",
        attacks: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame in the format expected by the LaTeX-table scripts.

        Columns: ``sample, sdg, attack, qi, ra_mean, RA_<feature>, ...``

        Parameters
        ----------
        dataset, dataset_size, qi
            Filter conditions.  ``None`` → no filter on that dimension.
        split
            Which score split to return (default: 'standard').
        confidence
            Only return rows with this confidence label (default: 'certain').
            Pass ``None`` to return all rows regardless of confidence.
        attacks
            Optional whitelist of attack labels.
        """
        conditions = ["r.split = ?"]
        params: list = [split]

        if dataset is not None:
            conditions.append("r.dataset = ?")
            params.append(dataset)
        if dataset_size is not None:
            conditions.append("r.dataset_size = ?")
            params.append(int(dataset_size))
        if qi is not None:
            conditions.append("r.qi = ?")
            params.append(qi)
        if confidence is not None:
            conditions.append("r.confidence = ?")
            params.append(confidence)
        if attacks:
            placeholders = ",".join("?" * len(attacks))
            conditions.append(f"r.attack_label IN ({placeholders})")
            params.extend(attacks)

        where = " AND ".join(conditions)

        runs_df = pd.read_sql_query(
            f"SELECT * FROM runs r WHERE {where}",
            self._conn,
            params=params,
        )

        if runs_df.empty:
            return runs_df.rename(columns={"sdg_method": "sdg", "attack_label": "attack"})

        # Fetch all matching feature scores in a single query
        run_ids = runs_df["run_id"].tolist()
        placeholders = ",".join("?" * len(run_ids))
        feat_df = pd.read_sql_query(
            f"SELECT run_id, feature, ra_score "
            f"FROM feature_scores WHERE run_id IN ({placeholders})",
            self._conn,
            params=run_ids,
        )

        if not feat_df.empty:
            feat_wide = feat_df.pivot_table(
                index="run_id", columns="feature", values="ra_score", aggfunc="first"
            ).reset_index()
            feat_wide.columns = ["run_id"] + [f"RA_{c}" for c in feat_wide.columns[1:]]
            result = runs_df.merge(feat_wide, on="run_id", how="left")
        else:
            result = runs_df.copy()

        # Rename to match the legacy CSV column names the LaTeX scripts expect
        result = result.rename(columns={"sdg_method": "sdg", "attack_label": "attack"})

        # Drop internal DB columns the LaTeX scripts don't need
        drop_cols = [
            "run_id", "split", "attack_params_json", "sdg_params_json",
            "source_file", "wandb_run_id", "wandb_group", "wandb_project",
            "ingested_at", "confidence", "confidence_notes", "dataset_size",
        ]
        result = result.drop(columns=[c for c in drop_cols if c in result.columns])

        return result

    def query_mem_test(
        self,
        dataset: Optional[str] = None,
        dataset_size: Optional[int] = None,
        qi: Optional[str] = None,
        confidence: Optional[str] = "certain",
    ) -> pd.DataFrame:
        """Return memorization-test rows with train_mean and nontrain_mean side by side."""
        train = self.query(
            dataset=dataset, dataset_size=dataset_size, qi=qi,
            split="train", confidence=confidence,
        ).rename(columns={"ra_mean": "train_mean"})

        nontrain = self.query(
            dataset=dataset, dataset_size=dataset_size, qi=qi,
            split="nontraining", confidence=confidence,
        ).rename(columns={"ra_mean": "nontrain_mean"})

        key = ["sample", "sdg", "attack", "qi"]
        if "dataset" in train.columns:
            key = ["dataset"] + key

        merged = train.merge(
            nontrain[key + ["nontrain_mean"]],
            on=key, how="outer",
        )
        if "train_mean" in merged and "nontrain_mean" in merged:
            merged["delta_mean"] = (
                merged["train_mean"].astype(float) - merged["nontrain_mean"].astype(float)
            ).round(4)
        return merged

    # ------------------------------------------------------------------
    # Inventory / diagnostics
    # ------------------------------------------------------------------

    def count_by_key(self) -> pd.DataFrame:
        """Count rows grouped by (dataset, dataset_size, qi, sdg_method, attack_label)."""
        return pd.read_sql_query(
            """
            SELECT dataset, dataset_size, qi, sdg_method, attack_label, split,
                   COUNT(*) AS n_runs
            FROM runs
            WHERE confidence = 'certain'
            GROUP BY dataset, dataset_size, qi, sdg_method, attack_label, split
            ORDER BY dataset, dataset_size, qi, sdg_method, attack_label
            """,
            self._conn,
        )

    def list_conflicts(self) -> pd.DataFrame:
        """Return all logged conflicts joined with the existing run's key."""
        return pd.read_sql_query(
            """
            SELECT c.*,
                   r.dataset, r.dataset_size, r.sample, r.qi,
                   r.sdg_method, r.attack_label, r.ra_mean AS existing_ra_mean
            FROM conflicts c
            JOIN runs r ON c.existing_run_id = r.run_id
            ORDER BY c.logged_at DESC
            """,
            self._conn,
        )

    def summary(self) -> None:
        """Print a quick human-readable summary to stdout."""
        n_runs = self._conn.execute(
            "SELECT COUNT(*) FROM runs WHERE confidence='certain'"
        ).fetchone()[0]
        n_uncertain = self._conn.execute(
            "SELECT COUNT(*) FROM runs WHERE confidence!='certain'"
        ).fetchone()[0]
        n_feat = self._conn.execute("SELECT COUNT(*) FROM feature_scores").fetchone()[0]
        n_conflicts = self._conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]

        datasets = self._conn.execute(
            "SELECT DISTINCT dataset, dataset_size FROM runs WHERE confidence='certain' "
            "ORDER BY dataset, dataset_size"
        ).fetchall()

        print(f"results.db at {self.db_path}")
        print(f"  Certain runs : {n_runs:,}")
        print(f"  Uncertain    : {n_uncertain:,}  (should be 0 in a clean DB)")
        print(f"  Feature scores: {n_feat:,}")
        print(f"  Conflicts    : {n_conflicts:,}")
        print(f"  Datasets     :")
        for ds, sz in datasets:
            n = self._conn.execute(
                "SELECT COUNT(*) FROM runs WHERE dataset=? AND dataset_size=? "
                "AND confidence='certain'",
                (ds, sz),
            ).fetchone()[0]
            print(f"    {ds} / size {sz:,} → {n:,} runs")
