#!/usr/bin/env python
"""
insert_new_dp_sweep_to_db.py

Insert results from run_new_dp_epsilon_sweep.py's output CSV into results.db,
following the same dual-row convention already used for MST/AIM memorization
sweeps in this DB:

  - split='standard'    ra_mean=ra_mean_train, feature_scores from RA_<feat> cols
                         (this is what wandb_to_latex_epsilon_sweep.py / ResultsDB.query()
                         reads by default for the per-feature epsilon-sweep tables)
  - split='train'       ra_mean=ra_mean_train      (aggregate only, no feature detail)
  - split='nontraining' ra_mean=ra_mean_nontraining (aggregate only, no feature detail)

Usage:
    conda activate recon_
    python experiment_scripts/insert_new_dp_sweep_to_db.py experiment_scripts/new_dp_epsilon_sweep_TIMESTAMP.csv
    python experiment_scripts/insert_new_dp_sweep_to_db.py --dry-run experiment_scripts/new_dp_epsilon_sweep_TIMESTAMP.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from results_db import ResultsDB

def _feat_scores(row: pd.Series) -> dict:
    return {
        c[3:]: float(row[c])
        for c in row.index
        if c.startswith("RA_") and pd.notna(row[c]) and isinstance(row[c], (int, float, np.floating, np.integer))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dataset-size", type=int, default=10_000)
    parser.add_argument("--dataset", type=str, default="adult")
    parser.add_argument("--wandb-group", type=str, default="new-dp-epsilon-sweep-adult-10k")
    args = parser.parse_args()
    DATASET = args.dataset
    DATASET_SIZE = args.dataset_size
    WANDB_GROUP = args.wandb_group

    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    df = df[df["ra_mean_train"].notna()]  # drop failed jobs
    print(f"Loaded {len(df)} successful rows from {csv_path}")

    db = ResultsDB()
    counts = {"standard": [0, 0], "train": [0, 0], "nontraining": [0, 0]}  # [inserted, skipped]

    for _, row in df.iterrows():
        sample = int(str(row["sample"]).replace("sample_", ""))
        feat_scores = _feat_scores(row)

        if args.dry_run:
            continue

        rid = db.insert_run(
            dataset=DATASET, dataset_size=DATASET_SIZE, sample=sample,
            qi=row["qi"], sdg_method=row["sdg"], attack_label=row["label"],
            split="standard", ra_mean=float(row["ra_mean_train"]),
            feature_scores=feat_scores if feat_scores else None,
            source_file=csv_path.name, wandb_group=WANDB_GROUP,
        )
        counts["standard"][0 if rid is not None else 1] += 1

        rid = db.insert_run(
            dataset=DATASET, dataset_size=DATASET_SIZE, sample=sample,
            qi=row["qi"], sdg_method=row["sdg"], attack_label=row["label"],
            split="train", ra_mean=float(row["ra_mean_train"]),
            source_file=csv_path.name, wandb_group=WANDB_GROUP,
        )
        counts["train"][0 if rid is not None else 1] += 1

        rid = db.insert_run(
            dataset=DATASET, dataset_size=DATASET_SIZE, sample=sample,
            qi=row["qi"], sdg_method=row["sdg"], attack_label=row["label"],
            split="nontraining", ra_mean=float(row["ra_mean_nontraining"]),
            source_file=csv_path.name, wandb_group=WANDB_GROUP,
        )
        counts["nontraining"][0 if rid is not None else 1] += 1

    db.close()

    if args.dry_run:
        print(f"[dry-run] would insert up to {len(df) * 3} rows (3 per job: standard/train/nontraining).")
    else:
        for split, (ins, skip) in counts.items():
            print(f"  split={split:12s} inserted={ins:4d}  skipped_dupe_or_conflict={skip:4d}")


if __name__ == "__main__":
    main()
