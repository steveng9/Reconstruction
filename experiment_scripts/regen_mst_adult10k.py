#!/usr/bin/env python
"""
regen_mst_adult10k.py

Regenerate MST synthetic data for adult 10k, all 5 samples, 5 epsilon values
[0.1, 1, 10, 100, 1000], using bin_continuous_as_ordinal=True throughout.

ROOT CAUSE BEING FIXED
----------------------
The original MST synth files for epsilon in {0.1, 0.3, 3, 30, 300} were
generated WITHOUT bin_continuous_as_ordinal, so continuous features (age,
fnlwgt, etc.) came back as float bin-midpoints (e.g. 22.475) instead of
original integer values (e.g. 18, 23, ...).  ML attacks trained on the
real integer-valued data then scored near-baseline against the float synth
because no synthetic record's continuous value ever exactly matches a real
training record's value.

This script:
  1. Overwrites MST_eps{X}/synth.csv for ALL five epsilons × all 5 samples
     with bin_continuous_as_ordinal=True (epsilon={1,10,100,1000} were
     already correct; re-generating verifies consistency).
  2. Produces no WandB output — pure data regeneration.

Run (from Reconstruction/ or anywhere):
    conda activate recon_
    python experiment_scripts/regen_mst_adult10k.py
    python experiment_scripts/regen_mst_adult10k.py --dry-run   # list jobs
    python experiment_scripts/regen_mst_adult10k.py --workers 5

Called by: run_mst_regen_pipeline.sh  (Phase 1)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DATA_ROOT  = Path("/home/golobs/data/reconstruction_data/adult/size_10000")
META_PATH  = Path("/home/golobs/data/reconstruction_data/adult/meta.json")
N_SAMPLES  = 5
EPSILONS   = [0.1, 1.0, 10.0, 100.0, 1000.0]
N_WORKERS  = 5   # one worker per sample works well (MST is CPU-bound)


@dataclass
class RegenJob:
    sample_idx: int
    epsilon: float

    @property
    def sample_dir(self) -> Path:
        return DATA_ROOT / f"sample_{self.sample_idx:02d}"

    @property
    def synth_dir(self) -> Path:
        return self.sample_dir / f"MST_eps{self.epsilon:g}"

    @property
    def synth_path(self) -> Path:
        return self.synth_dir / "synth.csv"

    @property
    def train_path(self) -> Path:
        return self.sample_dir / "train.csv"

    @property
    def label(self) -> str:
        return f"sample_{self.sample_idx:02d}/MST_eps{self.epsilon:g}"


def _run_regen(job: RegenJob) -> dict:
    sys.argv = sys.argv[:1]
    import pandas as pd
    sys.path.insert(0, "/home/golobs/Reconstruction")
    from sdg import get_sdg

    with open(META_PATH) as f:
        meta = json.load(f)

    train_df = pd.read_csv(job.train_path)
    generate = get_sdg("MST")

    t0 = datetime.now()
    try:
        synth_df = generate(train_df, meta,
                            epsilon=job.epsilon,
                            bin_continuous_as_ordinal=True)
        elapsed = (datetime.now() - t0).seconds

        job.synth_dir.mkdir(parents=True, exist_ok=True)
        synth_df.to_csv(job.synth_path, index=False)

        # Quick sanity: check age column is integer-valued
        age_vals = synth_df["age"].dropna().unique()
        is_int = all(float(v) == int(float(v)) for v in age_vals[:10])
        flag = "✓ integer-encoded" if is_int else "⚠ float-encoded (unexpected!)"

        return {"label": job.label, "elapsed": elapsed, "rows": len(synth_df),
                "flag": flag, "error": None}
    except Exception as e:
        return {"label": job.label, "elapsed": None, "rows": None,
                "flag": None, "error": str(e) + "\n" + traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate MST adult 10k synth with bin_continuous_as_ordinal=True")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    jobs = [RegenJob(s, e) for s in range(N_SAMPLES) for e in EPSILONS]
    print(f"Regen jobs: {len(jobs)}  (5 samples × 5 epsilons)")
    print(f"Workers   : {args.workers}")
    print(f"Data root : {DATA_ROOT}\n")
    for j in jobs:
        exists = "EXISTS" if j.synth_path.exists() else "MISSING"
        print(f"  {j.label:<40} [{exists}]")
    print()

    if args.dry_run:
        print("[dry-run] no jobs executed.")
        return

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = {ex.submit(_run_regen, j): j for j in jobs}
        done = 0
        errors = 0
        for fut in as_completed(futures):
            done += 1
            result = fut.result()
            if result["error"]:
                errors += 1
                print(f"  [{done:2d}/{len(jobs)}] ✗ {result['label']}")
                print(f"           ERROR: {result['error'][:200]}", flush=True)
            else:
                print(f"  [{done:2d}/{len(jobs)}] {result['label']:<40} "
                      f"{result['elapsed']:3d}s  {result['rows']} rows  "
                      f"{result['flag']}", flush=True)

    print(f"\nDone. {done - errors}/{done} succeeded.")
    if errors:
        print(f"WARNING: {errors} jobs FAILED — check output above.")
    else:
        print("All synth.csv files regenerated with integer encoding. ✓")


if __name__ == "__main__":
    main()
