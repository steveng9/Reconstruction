#!/usr/bin/env python
"""
One-time ingestion: california memorization CSVs → results.db

Reads the local summary CSVs written by run_california_memorization.py and
inserts each row as TWO splits (train + nontraining) into results.db.

Usage:
    python experiment_scripts/ingest_california_memorization.py
    python experiment_scripts/ingest_california_memorization.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from results_db import ResultsDB

DATASET      = "california"
DATASET_SIZE = 1000
QI           = "QI_large"
WANDB_GROUP  = "california-memorization-QI_large"
WANDB_PROJECT = "tabular-reconstruction-attacks"

# The two CSVs to ingest (skip the small smoke-test ones — their rows are
# already covered by the full-sweep CSV).
CSVS = [
    SCRIPT_DIR / "california_memorization_20260524_215418.csv",  # Mean/LR/RF/LGB  (280 rows)
    SCRIPT_DIR / "california_memorization_20260524_220141.csv",  # BayesianRidge    (70 rows)
]


def _parse_sdg_params(sdg_label: str) -> dict:
    """'MST_eps1' → {'epsilon': 1.0},  'ARF' → {}"""
    m = re.search(r"_eps([\d.]+)$", sdg_label)
    return {"epsilon": float(m.group(1))} if m else {}


def ingest_csv(db: ResultsDB, csv_path: Path, dry_run: bool = False) -> dict[str, int]:
    df = pd.read_csv(csv_path)
    # Drop any error rows
    df = df[df["error"].isna() | (df["error"] == "")]

    feature_cols_train  = [c for c in df.columns if c.startswith("RA_train_")]
    feature_cols_nontrain = [c for c in df.columns if c.startswith("RA_nontraining_")]
    features = [c.replace("RA_train_", "") for c in feature_cols_train]

    counts = {"inserted": 0, "duplicate": 0, "skipped_error": 0}

    for _, row in df.iterrows():
        sdg_label  = str(row["sdg"])
        sdg_params = _parse_sdg_params(sdg_label)
        attack     = str(row["attack"])
        sample     = int(row["sample"])
        source     = csv_path.name

        for split, mean_col, feat_prefix in [
            ("train",        "train_mean",   "RA_train_"),
            ("nontraining",  "nontrain_mean","RA_nontraining_"),
        ]:
            ra_mean = float(row[mean_col]) if pd.notna(row.get(mean_col)) else None
            feat_scores = {
                feat: float(row[f"{feat_prefix}{feat}"])
                for feat in features
                if f"{feat_prefix}{feat}" in row and pd.notna(row.get(f"{feat_prefix}{feat}"))
            }

            if dry_run:
                print(f"  [DRY] {sample=} {sdg_label} {attack} split={split} ra={ra_mean:.4f}")
                counts["inserted"] += 1
                continue

            run_id = db.insert_run(
                dataset=DATASET,
                dataset_size=DATASET_SIZE,
                sample=sample,
                qi=QI,
                sdg_method=sdg_label,
                attack_label=attack,
                split=split,
                ra_mean=ra_mean,
                feature_scores=feat_scores,
                sdg_params=sdg_params,
                source_file=source,
                wandb_group=WANDB_GROUP,
                wandb_project=WANDB_PROJECT,
                confidence="certain",
            )
            if run_id is not None:
                counts["inserted"] += 1
            else:
                counts["duplicate"] += 1

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db = ResultsDB()
    total = {"inserted": 0, "duplicate": 0, "skipped_error": 0}

    for csv_path in CSVS:
        print(f"\n{csv_path.name}")
        counts = ingest_csv(db, csv_path, dry_run=args.dry_run)
        for k, v in counts.items():
            total[k] += v
        print(f"  inserted={counts['inserted']}  duplicate={counts['duplicate']}")

    print(f"\nTotal: inserted={total['inserted']}  duplicate={total['duplicate']}")

    if not args.dry_run:
        print()
        db.summary()

    db.close()


if __name__ == "__main__":
    main()
