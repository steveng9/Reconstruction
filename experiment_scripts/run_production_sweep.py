#!/usr/bin/env python
"""
Production experiment sweep: all samples × all SDG methods × all attacks.

Runs the fundamental RA experiments for the paper:
  - 5 disjoint training samples
  - All 9 SDG methods (MST and AIM at multiple epsilon values)
  - All categorical attacks (comment out as needed)
  - Up to N_WORKERS experiments in parallel

Configuration is at the top of this file — update DATASET_NAME, DATA_ROOT,
QI_VARIANTS, SDG_METHODS, etc. for each dataset.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_production_sweep.py
    python experiment_scripts/run_production_sweep.py --dry-run     # print job list only
    python experiment_scripts/run_production_sweep.py --workers 4   # override parallelism

Results logged to WandB under WANDB_PROJECT, group WANDB_GROUP.
A summary CSV is written to the script's directory on completion.
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Configuration ─────────────────────────────────────────────────────────────

# DATASET_BASE: the directory name under reconstruction_data/ (always "nist_arizona_data")
# DATASET_NAME: the key used for QI / hidden-feature lookup in get_data.py — set this to
#   "nist_arizona_25feat", "nist_arizona_50feat", or "nist_arizona_data" to select the
#   right QI definitions and hidden-feature set for each feature-subset experiment.
# N_FEATURES:  None = full 98-col data; 25 or 50 = feature-subset data (adds "_Nfeat"
#   suffix to the size directory, matching what generate_synth.py creates).
#DATASET_BASE  = "nist_arizona_data"
#DATASET_NAME  = "nist_arizona_25feat"   # QI lookup key — change with N_FEATURES
DATASET_BASE  = "adult"
DATASET_NAME  = "adult"   # QI lookup key — change with N_FEATURES
DATASET_SIZE  = 10_000
#N_FEATURES    = 25                      # None | 25 | 50
N_FEATURES    = None                      # None | 25 | 50
DATA_ROOT     = (
    f"/home/golobs/data/reconstruction_data/{DATASET_BASE}/size_{DATASET_SIZE}"
    + (f"_{N_FEATURES}feat" if N_FEATURES is not None else "")
)
SAMPLE_RANGE  = range(5)          # sample_00 through sample_04
DATASET_TYPE  = "categorical"     # "categorical" or "continuous"

# Set HOLDOUT_SAMPLE_IDX to use a fixed sample as holdout for memorization test.
# Set to None to disable memorization test entirely.
HOLDOUT_SAMPLE_IDX = None

QI_VARIANTS = ["QI1"]   # "QI2" only defined for nist_arizona_data (98-col full data)

# SDG methods: (method_name, sdg_params)
SDG_METHODS = [
    ("RankSwap",       {}),
    #("MST",            {"epsilon": 0.1}),
    ("MST",            {"epsilon": 1.0}),
    ("MST",            {"epsilon": 10.0}),
    ("MST",            {"epsilon": 100.0}),
    ("MST",            {"epsilon": 1000.0}),
    ("AIM",            {"epsilon": 1.0}),
    #("AIM",            {"epsilon": 10.0}),
    #("AIM",            {"epsilon": 100.0}),
    ("TVAE",           {}),
    ("CTGAN",          {}),
    ("ARF",            {}),
    ("TabDDPM",        {}),
    ("Synthpop",       {}),
    ("CellSuppression", {}),
]

# (attack_method, method_specific_params)
ATTACK_CONFIGS = [
    # Baselines
    ("Mode",                    {}),
    ("Random",                  {}),
    #("Mean",                    {}),
    ("MeasureDeid",             {}),
    # ML classifiers
    ("KNN",                     {}),
    ("NaiveBayes",              {}),
    ("LogisticRegression",      {}),
    ("SVM",                     {}),   # O(n²–n³) — impractical at n=10k
    ("RandomForest",            {}),
    ("LightGBM",                {}),
    # Neural networks
    ("MLP",                     {}),
    ("Attention",               {}),
    #("AttentionAutoregressive", {}),
    # SOTA (requires Gurobi academic licence)
    # ("LinearReconstruction",    {}),
    # Partial diffusion (agnostic — work on categorical and continuous)
    # TabDDPM and ConditionedRePaint share the same artifact dir and trained model
    # (identical QI-conditioned training; differ only in sampling). Whichever runs
    # second will find model_ckpt.pkl already present and skip retraining automatically.
    # Pass retrain=True to force retraining from scratch.
    ("TabDDPM",            {"retrain": False}),   # QI-conditioned + TabDDPM sampling
    # ("RePaint",           {"retrain": False}),   # standard training + RePaint sampling
    ("ConditionedRePaint", {"retrain": False}),   # QI-conditioned + RePaint sampling
]

# ATTACK_PARAM_DEFAULTS is imported from attack_defaults.py (repo root) at the top of this file.
# Explicitly passed params in ATTACK_CONFIGS override those defaults; the merged result
# is logged to WandB so every run records the full effective parameter set.

N_WORKERS     = 8
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "main attack sweep 1"
WANDB_TAGS    = [DATASET_NAME, f"size_{DATASET_SIZE}", "production"]


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx:   int
    sdg_method:   str
    sdg_params:   dict
    attack_method: str
    attack_params: dict
    qi:           str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{DATA_ROOT}/sample_{self.sample_idx:02d}"

    @property
    def run_name(self) -> str:
        return f"s{self.sample_idx:02d}__{self.sdg_label}__{self.attack_method}__{self.qi}"


def generate_jobs() -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params), qi in itertools.product(
        SAMPLE_RANGE, SDG_METHODS, ATTACK_CONFIGS, QI_VARIANTS
    ):
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=sdg_params,
            attack_method=attack_method,
            attack_params=dict(attack_params),
            qi=qi,
        ))
    return jobs


# ── Worker function (runs in a subprocess) ────────────────────────────────────

def _worker_setup_paths():
    """Called once per worker process to configure sys.path."""
    # recon-synth defines its own 'attacks' package that shadows Reconstruction's.
    # Append those paths so they're available but don't take precedence.
    for p in [
        "/home/golobs/MIA_on_diffusion/",
        "/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM",
        "/home/golobs/recon-synth",
        "/home/golobs/recon-synth/attacks",
        "/home/golobs/recon-synth/attacks/solvers",
    ]:
        if p not in sys.path:
            sys.path.append(p)
    # Reconstruction must be at index 0 so its 'attacks' package wins over recon-synth's.
    # Remove first in case the spawned worker already added it elsewhere on startup.
    reconstruction = "/home/golobs/Reconstruction"
    if reconstruction in sys.path:
        sys.path.remove(reconstruction)
    sys.path.insert(0, reconstruction)


def run_job(job: Job) -> dict[str, Any]:
    """
    Execute one experiment job. Imports happen inside the worker process.
    Returns a result dict; raises on fatal error.
    """
    _worker_setup_paths()

    # master_experiment_script.py calls parse_args() at module level.
    # Clear sys.argv before importing it so its parser doesn't choke on
    # arguments meant for this script (e.g. --workers, --dry-run).
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    sample_dir = job.sample_dir

    cfg = {
        "dataset": {
            "name": DATASET_NAME,
            "dir":  sample_dir,
            "size": DATASET_SIZE,
            "type": DATASET_TYPE,
        },
        "QI":          job.qi,
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

    if HOLDOUT_SAMPLE_IDX is not None:
        holdout_dir = f"{DATA_ROOT}/sample_{HOLDOUT_SAMPLE_IDX:02d}"
        cfg["memorization_test"] = {
            "enabled": True,
            "holdout_dir": holdout_dir,
        }

    prepared = _prepare_config(cfg)

    # Effective params: same merge already applied to cfg above, reused for WandB logging.
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
            "qi":            job.qi,
            "dataset":       DATASET_NAME,
            "size":          DATASET_SIZE,
        },
        tags=WANDB_TAGS,
        group=WANDB_GROUP,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        if HOLDOUT_SAMPLE_IDX is not None and holdout is not None:
            recon_train   = _run_attack(prepared, synth, train,   qi, hidden_features)
            recon_holdout = _run_attack(prepared, synth, holdout, qi, hidden_features)

            train_scores   = _score_reconstruction(train,   recon_train,   hidden_features, DATASET_TYPE)
            holdout_scores = _score_reconstruction(holdout, recon_holdout, hidden_features, DATASET_TYPE)

            metrics = {}
            for feat, ts, hs in zip(hidden_features, train_scores, holdout_scores):
                metrics[f"RA_train_{feat}"]       = ts
                metrics[f"RA_nontraining_{feat}"] = hs
                metrics[f"RA_delta_{feat}"]       = round(ts - hs, 4)

            train_mean   = round(float(np.mean(train_scores)), 4)
            holdout_mean = round(float(np.mean(holdout_scores)), 4)
            metrics["RA_train_mean"]       = train_mean
            metrics["RA_nontraining_mean"] = holdout_mean
            metrics["RA_delta_mean"]       = round(train_mean - holdout_mean, 4)
            wandb.log(metrics)

            result = {
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": job.qi,
                "train_mean": train_mean, "nontrain_mean": holdout_mean,
                "delta_mean": metrics["RA_delta_mean"],
                "ra_mean": None, "error": None,
            }

        else:
            recon  = _run_attack(prepared, synth, train, qi, hidden_features)
            scores = _score_reconstruction(train, recon, hidden_features, DATASET_TYPE)

            metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
            ra_mean = round(float(np.mean(scores)), 4)
            metrics["RA_mean"] = ra_mean
            wandb.log(metrics)

            result = {
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": job.qi,
                "ra_mean": ra_mean,
                "train_mean": None, "nontrain_mean": None, "delta_mean": None,
                "error": None,
            }

        return result

    except Exception as exc:
        wandb.log({"error": str(exc)})
        raise

    finally:
        wandb.finish()


# ── Summary helpers ────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = ["sample", "sdg", "attack", "qi",
            "ra_mean", "train_mean", "nontrain_mean", "delta_mean", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict]):
    successes = [r for r in rows if r["error"] is None]
    failures  = [r for r in rows if r["error"] is not None]

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — {DATASET_NAME}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*70}")

    if not successes:
        return

    # Group by (attack, qi) and show mean RA across SDG methods + samples
    from collections import defaultdict
    import numpy as np

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["attack"], r["qi"])
        val = r["ra_mean"] if r["ra_mean"] is not None else r["train_mean"]
        if val is not None:
            groups[key].append(val)

    print(f"\n  {'Attack':<25}  {'QI':<6}  {'Mean RA (avg over samples+SDG)':>32}")
    print(f"  {'-'*65}")
    for (attack, qi), vals in sorted(groups.items()):
        mean_val = round(float(np.mean(vals)), 4)
        print(f"  {attack:<25}  {qi:<6}  {mean_val:>32.4f}")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Production RA sweep.")
    parser.add_argument("--dry-run",      action="store_true", help="Print job list and exit.")
    parser.add_argument("--serial",       action="store_true", help="Run jobs sequentially in the main process (useful for testing).")
    parser.add_argument("--workers",      type=int, default=N_WORKERS, help="Number of parallel workers.")
    parser.add_argument("--sample",       type=int, default=None, help="Run only this sample index.")
    parser.add_argument("--sdg",          type=str, default=None, help="Run only this SDG method (e.g. 'MST').")
    parser.add_argument("--attack",       type=str, default=None, help="Run only this attack method.")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "progress.log"),
                        metavar="FILE",
                        help="Write one progress line per completed job to FILE (no wandb noise). "
                             "Use 'tail -f FILE' to monitor a background run.")
    args = parser.parse_args()

    all_jobs = generate_jobs()

    # Optional filtering
    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs if j.sdg_method == args.sdg]
    if args.attack is not None:
        all_jobs = [j for j in all_jobs if j.attack_method == args.attack]

    progress_log = open(args.progress_log, "w", buffering=1) if args.progress_log else None

    header = (
        f"{'='*70}\n"
        f"  Production sweep: {DATASET_NAME}\n"
        f"  Jobs total:  {len(all_jobs)}\n"
        f"  Workers:     {args.workers}\n"
        f"  WandB group: {WANDB_GROUP}\n"
        f"{'='*70}\n"
    )
    print(header, end="")
    if progress_log:
        progress_log.write(header)

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    # Verify sample directories exist before dispatching
    missing = []
    for job in all_jobs:
        p = Path(job.sample_dir)
        if not p.exists():
            missing.append(job.sample_dir)
    missing = list(dict.fromkeys(missing))  # deduplicate
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
            line = (
                f"  [{n_done:>{width}}/{len(all_jobs)}]"
                f"  FAILED  {job.run_name}:  {result_or_exc}"
            )
            results.append({
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": job.qi,
                "ra_mean": None, "train_mean": None, "nontrain_mean": None,
            })
        else:
            val     = result_or_exc["ra_mean"] if result_or_exc["ra_mean"] is not None else result_or_exc["train_mean"]
            val_str = f"{val:.4f}" if val is not None else "N/A"
            line = (
                f"  [{n_done:>{width}}/{len(all_jobs)}]"
                f"  {job.run_name:<60}  RA={val_str}{eta_str}"
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

    script_dir = Path(__file__).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = script_dir / f"sweep_results_{DATASET_NAME}_{ts}.csv"
    _save_summary_csv(results, csv_path)

    if progress_log:
        progress_log.close()


if __name__ == "__main__":
    main()
