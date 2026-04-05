#!/usr/bin/env python
"""
CDC Diabetes reconstruction-attack sweep for Chapter 4 (dataset comparison).

Runs two tiers of jobs in one combined sweep:

  Tier 1 — cdc_diabetes/size_1000 / QI1 / all attacks
    Full suite: Mode, Random, KNN, NaiveBayes, LR, RF, LightGBM, MLP,
                TabDDPM (retrain=True), ConditionedRePaint (retrain=False)
    15 SDG methods, 5 samples  →  750 jobs

  Tier 2 — cdc_diabetes/size_100000 / QI1 / CPU-only attacks
    Mode, Random, NaiveBayes, LR, RF, LightGBM, MLP
    (KNN and diffusion omitted — O(n) and O(GPU) impractical at 100k)
    14 SDG methods, 5 samples  →  490 jobs

Total: ~1240 jobs.

Estimated wall-clock time with --workers 4 (recommended):
  CPU attacks finish in ~3–5 hours.
  GPU (TabDDPM + CR) dominates at ~12–15 hours on 2 × RTX 6000 Ada.
  Comfortably within a single overnight run.

Usage:
    conda activate recon_
    python experiment_scripts/run_cdc_sweep.py                  # full run, 4 workers
    python experiment_scripts/run_cdc_sweep.py --dry-run         # count jobs, no execution
    python experiment_scripts/run_cdc_sweep.py --workers 6       # override parallelism
    python experiment_scripts/run_cdc_sweep.py --size 1000       # only 1k tier
    python experiment_scripts/run_cdc_sweep.py --size 100000     # only 100k tier
    python experiment_scripts/run_cdc_sweep.py --attack RF       # one attack across both tiers
    python experiment_scripts/run_cdc_sweep.py --sdg MST         # one SDG method
    python experiment_scripts/run_cdc_sweep.py --serial          # single-process, Ctrl-C-killable
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_NAME  = "cdc_diabetes"
DATASET_TYPE  = "categorical"
DATA_ROOT_1K  = "/home/golobs/data/reconstruction_data/cdc_diabetes/size_1000"
DATA_ROOT_100K = "/home/golobs/data/reconstruction_data/cdc_diabetes/size_100000"

SAMPLE_RANGE  = list(range(5))   # sample_00 through sample_04

QI_VARIANT = "QI1"

# ── SDG methods available on disk ─────────────────────────────────────────────
# 1k: AIM at eps 1, 3, 10 (no 0.3/30); MST at 0.1, 1, 10, 100, 1000 (5 variants)
SDG_METHODS_1K = [
    ("MST",            {"epsilon": 0.1}),
    ("MST",            {"epsilon": 1.0}),
    ("MST",            {"epsilon": 10.0}),
    ("MST",            {"epsilon": 100.0}),
    ("MST",            {"epsilon": 1000.0}),
    ("AIM",            {"epsilon": 1.0}),
    ("AIM",            {"epsilon": 3.0}),
    ("AIM",            {"epsilon": 10.0}),
    ("TVAE",           {}),
    ("CTGAN",          {}),
    ("ARF",            {}),
    ("TabDDPM",        {}),
    ("Synthpop",       {}),
    ("CellSuppression",{}),
    ("RankSwap",       {}),
]

# 100k: same but AIM eps3 not generated; TabDDPM (as SDG) does exist
SDG_METHODS_100K = [
    ("MST",            {"epsilon": 0.1}),
    ("MST",            {"epsilon": 1.0}),
    ("MST",            {"epsilon": 10.0}),
    ("MST",            {"epsilon": 100.0}),
    ("MST",            {"epsilon": 1000.0}),
    ("AIM",            {"epsilon": 1.0}),
    ("AIM",            {"epsilon": 10.0}),
    ("TVAE",           {}),
    ("CTGAN",          {}),
    ("ARF",            {}),
    ("TabDDPM",        {}),
    ("Synthpop",       {}),
    ("CellSuppression",{}),
    ("RankSwap",       {}),
]

# ── Attack configs ─────────────────────────────────────────────────────────────

# Full suite for 1k — CPU classifiers + diffusion
ATTACK_CONFIGS_1K = [
    # Baselines
    ("Mode",         {}),
    ("Random",       {}),
    # ML classifiers
    ("KNN",          {}),
    ("NaiveBayes",   {}),
    ("LogisticRegression", {}),
    ("RandomForest", {}),
    ("LightGBM",     {}),
    ("MLP",          {}),
    # Diffusion
    ("TabDDPM",      {"retrain": True}),
]

# CPU-only for 100k:
#   - KNN excluded: O(n) lookup against 100k synth is impractical
#   - LightGBM excluded: uses thousands-of-% CPU and chokes parallel workers
#   - Diffusion excluded: GPU training at 100k scale is too slow for this sweep
#   - MLP: scaled down — default batch=264/epochs=500 → 190k steps at 100k rows;
#     batch=4096/epochs=100 gives ~2.4k steps (same order as 1k default), trains fast on GPU
ATTACK_CONFIGS_100K = [
    ("Mode",         {}),
    ("Random",       {}),
    ("NaiveBayes",   {}),
    ("LogisticRegression", {}),
    ("RandomForest", {}),
    ("MLP",          {"batch_size": 4096, "epochs": 100, "patience": 10}),
]

N_WORKERS     = 4
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "cdc-sweep-1"
WANDB_TAGS    = [DATASET_NAME, "chapter4", "production"]


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx:    int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict
    dataset_size:  int          # 1000 or 100000
    data_root:     str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{self.data_root}/sample_{self.sample_idx:02d}"

    @property
    def run_name(self) -> str:
        return f"cdc_{self.dataset_size}__s{self.sample_idx:02d}__{self.sdg_label}__{self.attack_method}"


def generate_jobs() -> list[Job]:
    jobs = []
    # Tier 1: 1k, full attack suite
    for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params) in itertools.product(
        SAMPLE_RANGE, SDG_METHODS_1K, ATTACK_CONFIGS_1K
    ):
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=sdg_params,
            attack_method=attack_method,
            attack_params=dict(attack_params),
            dataset_size=1_000,
            data_root=DATA_ROOT_1K,
        ))
    # Tier 2: 100k, CPU-only attacks
    for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params) in itertools.product(
        SAMPLE_RANGE, SDG_METHODS_100K, ATTACK_CONFIGS_100K
    ):
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=sdg_params,
            attack_method=attack_method,
            attack_params=dict(attack_params),
            dataset_size=100_000,
            data_root=DATA_ROOT_100K,
        ))
    return jobs


# ── Worker function (runs in a subprocess) ────────────────────────────────────

def _worker_setup_paths():
    """Called once per worker process to configure sys.path."""
    for p in [
        "/home/golobs/MIA_on_diffusion/",
        "/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM",
        "/home/golobs/recon-synth",
        "/home/golobs/recon-synth/attacks",
        "/home/golobs/recon-synth/attacks/solvers",
    ]:
        if p not in sys.path:
            sys.path.append(p)
    reconstruction = "/home/golobs/Reconstruction"
    if reconstruction in sys.path:
        sys.path.remove(reconstruction)
    sys.path.insert(0, reconstruction)


def run_job(job: Job) -> dict[str, Any]:
    """Execute one experiment job in a worker subprocess."""
    _worker_setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    cfg = {
        "dataset": {
            "name": DATASET_NAME,
            "dir":  job.sample_dir,
            "size": job.dataset_size,
            "type": DATASET_TYPE,
        },
        "QI":          QI_VARIANT,
        "data_type":   DATASET_TYPE,
        "sdg_method":  job.sdg_method,
        "sdg_params":  job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": {"enabled": False},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            job.attack_method: {
                **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
                **job.attack_params,
            },
        },
    }

    prepared = _prepare_config(cfg)

    effective_attack_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_params,
    }

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config={
            "sample_idx":    job.sample_idx,
            "sdg_method":    job.sdg_method,
            "sdg_params":    job.sdg_params,
            "attack_method": job.attack_method,
            "attack_params": effective_attack_params,
            "qi":            QI_VARIANT,
            "dataset":       DATASET_NAME,
            "dataset_name":  DATASET_NAME,
            "size":          job.dataset_size,
        },
        tags=WANDB_TAGS + [f"size_{job.dataset_size}"],
        group=WANDB_GROUP,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, _ = load_data(prepared)
        recon  = _run_attack(prepared, synth, train, qi, hidden_features)
        scores = _score_reconstruction(train, recon, hidden_features, DATASET_TYPE)

        metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
        ra_mean = round(float(np.mean(scores)), 4)
        metrics["RA_mean"] = ra_mean
        wandb.log(metrics)

        feat_scores = {k: v for k, v in metrics.items() if k.startswith("RA_") and k != "RA_mean"}
        return {
            "sample": job.sample_idx, "sdg": job.sdg_label,
            "attack": job.attack_method, "qi": QI_VARIANT,
            "size": job.dataset_size,
            "ra_mean": ra_mean, "error": None,
            **feat_scores,
        }

    except Exception as exc:
        wandb.log({"error": str(exc)})
        raise

    finally:
        wandb.finish()


# ── Summary helpers ────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    base_keys = ["size", "sample", "sdg", "attack", "qi", "ra_mean", "error"]
    feat_keys = sorted({k for r in rows for k in r if k.startswith("RA_") and k != "RA_mean"})
    keys = base_keys + feat_keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict]):
    successes = [r for r in rows if r.get("error") is None]
    failures  = [r for r in rows if r.get("error") is not None]

    print(f"\n{'='*70}")
    print(f"  CDC SWEEP COMPLETE")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*70}")

    from collections import defaultdict
    import numpy as np

    for size in [1_000, 100_000]:
        size_rows = [r for r in successes if r.get("size") == size]
        if not size_rows:
            continue
        print(f"\n  --- size={size} ---")
        groups: dict[tuple, list] = defaultdict(list)
        for r in size_rows:
            key = (r["attack"], r["qi"])
            if r["ra_mean"] is not None:
                groups[key].append(r["ra_mean"])
        print(f"  {'Attack':<25}  {'Mean RA (avg over samples+SDG)':>32}")
        print(f"  {'-'*60}")
        for (attack, qi), vals in sorted(groups.items()):
            print(f"  {attack:<25}  {np.mean(vals):>32.4f}")

    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CDC Diabetes RA sweep (1k + 100k).")
    parser.add_argument("--dry-run",      action="store_true", help="Print job list and exit.")
    parser.add_argument("--serial",       action="store_true", help="Run sequentially in main process.")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--size",         type=int, default=None, choices=[1000, 100000],
                        help="Restrict to one tier only.")
    parser.add_argument("--sample",       type=int, default=None)
    parser.add_argument("--sdg",          type=str, default=None)
    parser.add_argument("--attack",       type=str, default=None)
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "cdc_progress.log"),
                        metavar="FILE")
    args = parser.parse_args()

    all_jobs = generate_jobs()

    if args.size is not None:
        all_jobs = [j for j in all_jobs if j.dataset_size == args.size]
    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs if j.sdg_label == args.sdg or j.sdg_method == args.sdg]
    if args.attack is not None:
        all_jobs = [j for j in all_jobs if j.attack_method == args.attack]

    n_1k  = sum(1 for j in all_jobs if j.dataset_size == 1_000)
    n_100k = sum(1 for j in all_jobs if j.dataset_size == 100_000)

    progress_log = open(args.progress_log, "w", buffering=1) if args.progress_log else None

    header = (
        f"{'='*70}\n"
        f"  CDC Diabetes sweep: {DATASET_NAME} / {QI_VARIANT}\n"
        f"  Jobs: {n_1k} × size_1k  +  {n_100k} × size_100k  =  {len(all_jobs)} total\n"
        f"  Workers: {args.workers}  |  WandB group: {WANDB_GROUP}\n"
        f"{'='*70}\n"
    )
    print(header, end="")
    if progress_log:
        progress_log.write(header)

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>5d}]  {j.run_name}")
        n_gpu = sum(1 for j in all_jobs if j.attack_method in ("TabDDPM", "ConditionedRePaint"))
        n_cpu = len(all_jobs) - n_gpu
        print(f"\n{len(all_jobs)} jobs total  ({n_cpu} CPU,  {n_gpu} GPU/diffusion)")
        return

    # Verify all sample directories exist
    missing = list(dict.fromkeys(
        j.sample_dir for j in all_jobs if not Path(j.sample_dir).exists()
    ))
    if missing:
        print("ERROR: missing sample directories:")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    start_time = time.time()
    results: list[dict] = []
    n_done = 0
    n_fail = 0

    def _handle_result(job, result_or_exc):
        nonlocal n_done, n_fail
        n_done += 1
        elapsed = time.time() - start_time
        eta_str = ""
        if n_done > 1:
            rate    = n_done / elapsed
            eta_s   = (len(all_jobs) - n_done) / rate
            eta_str = f"  ETA {eta_s/60:.0f}m"
        width = len(str(len(all_jobs)))
        if isinstance(result_or_exc, Exception):
            n_fail += 1
            line = f"  [{n_done:>{width}}/{len(all_jobs)}]  FAILED  {job.run_name}:  {result_or_exc}"
            results.append({
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": QI_VARIANT,
                "size": job.dataset_size, "ra_mean": None, "error": str(result_or_exc),
            })
        else:
            val     = result_or_exc.get("ra_mean")
            val_str = f"{val:.4f}" if val is not None else "N/A"
            line = (
                f"  [{n_done:>{width}}/{len(all_jobs)}]"
                f"  {job.run_name:<65}  RA={val_str}{eta_str}"
            )
            results.append(result_or_exc)
        print(line)
        if progress_log:
            progress_log.write(line + "\n")

    if args.serial:
        _worker_setup_paths()
        sys.argv = sys.argv[:1]
        for job in all_jobs:
            try:
                _handle_result(job, run_job(job))
            except Exception as exc:
                _handle_result(job, exc)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            future_to_job = {pool.submit(run_job, job): job for job in all_jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    _handle_result(job, future.result())
                except Exception as exc:
                    _handle_result(job, exc)

    total_min = (time.time() - start_time) / 60
    print(f"\nAll {len(all_jobs)} jobs finished in {total_min:.1f} min.")

    _print_summary(results)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"cdc_sweep_results_{ts}.csv"
    _save_summary_csv(results, csv_path)

    if progress_log:
        progress_log.close()


if __name__ == "__main__":
    main()
