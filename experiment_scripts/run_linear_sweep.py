#!/usr/bin/env python
"""
Comprehensive sweep comparing LinearReconstruction against ML classifiers
in the single-binary-hidden-feature setting.

LinearReconstruction (LP + Gurobi) is designed specifically for reconstructing
a single binary feature given synthetic data and all other features as QI.
This script focuses exclusively on that setting for a fair comparison.

Binary targets per dataset:
  adult:       income (QI_linear), sex (QI_binary_sex)
  cdc_diabetes: Diabetes_binary (QI_linear), HighBP (QI_binary_HighBP),
                Stroke (QI_binary_Stroke)
  arizona:     FARM (QI_binary_FARM), URBAN (QI_binary_URBAN)
    NOTE: FARM/URBAN use IPUMS {1,2} encoding, not {0,1}. Verify that
    LinearReconstruction handles this before interpreting those results.

Memorization test is always run (round-robin holdout: sample N uses
sample (N+1)%5 as holdout, so all 5 samples contribute).

Usage:
    conda activate recon_
    python experiment_scripts/run_linear_sweep.py --dataset adult --size 10000
    python experiment_scripts/run_linear_sweep.py --dataset cdc_diabetes --size 1000
    python experiment_scripts/run_linear_sweep.py --dataset arizona --size 10000
    python experiment_scripts/run_linear_sweep.py --dataset adult --size 10000 --dry-run
    python experiment_scripts/run_linear_sweep.py --dataset adult --size 10000 --workers 8
    python experiment_scripts/run_linear_sweep.py --dataset adult --size 10000 --attack LinearReconstruction
    python experiment_scripts/run_linear_sweep.py --dataset adult --size 10000 --qi QI_linear
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Dataset configurations ────────────────────────────────────────────────────
# base:       directory name under reconstruction_data/
# name:       QI/hidden-feature lookup key in get_data.py
# data_type:  "categorical" for all three (all integer-coded)
# n_features: None | 25 | 50  (adds "_Nfeat" suffix to size dir if set)
# qi_variants: list of QI keys to sweep — each has exactly one binary hidden feature

DATASET_CONFIGS: dict[str, dict] = {
    "adult": {
        "base":        "adult",
        "name":        "adult",
        "data_type":   "categorical",
        "n_features":  None,
        "qi_variants": [
                 "QI_linear", "QI_binary_sex"],
                 #"QI_linear_lowcard", "QI_binary_sex_lowcard"],
    },
    "cdc_diabetes": {
        "base":        "cdc_diabetes",
        "name":        "cdc_diabetes",
        "data_type":   "categorical",
        "n_features":  None,
        "qi_variants": ["QI_linear", "QI_binary_HighBP", "QI_binary_Stroke"],
    },
    "arizona": {
        "base":        "nist_arizona_data",
        "name":        "nist_arizona_25feat",
        "data_type":   "categorical",
        "n_features":  25,
        "qi_variants": ["QI_binary_SEX_lowcard"],
    },
}

# ── SDG methods ───────────────────────────────────────────────────────────────
SDG_METHODS = [
    ("MST",      {"epsilon": 1.0}),
    ("MST",      {"epsilon": 10.0}),
    ("MST",      {"epsilon": 100.0}),
    ("MST",      {"epsilon": 1000.0}),
    ("AIM",    {"epsilon": 1.0}),
    ("ARF",    {}),
    ("TVAE",    {}),
    ("CTGAN",    {}),
    ("RankSwap",    {}),
    ("CellSuppression",    {}),
    ("Synthpop", {}),
    ("TabDDPM",  {}),
]

# ── Attack configs ────────────────────────────────────────────────────────────
# All are "categorical" registry lookups (LinearReconstruction is in
# ATTACK_REGISTRY["categorical"] so data_type="categorical" works for all).
ATTACK_CONFIGS = [
    #("Random",        {}),
    ("NaiveBayes",        {}),
    ("KNN",        {}),
    #("RandomForest",        {}),
    #("LogisticRegression",  {}),
    #("LightGBM",            {}),
    #("MLP",                 {}),
    #("LinearReconstruction", {}),
]

SAMPLE_RANGE  = list(range(5))   # samples 00–04; holdout is (idx+1)%5

WANDB_PROJECT = "tabular-reconstruction-attacks"
N_WORKERS_DEFAULT = 12


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    # Dataset
    dataset_key:  str   # "adult" | "cdc_diabetes" | "arizona"
    dataset_name: str   # QI lookup key (e.g. "nist_arizona_25feat")
    dataset_base: str   # dir name (e.g. "nist_arizona_data")
    dataset_size: int
    dataset_type: str
    data_root:    str   # full path to size dir
    # Experiment
    sample_idx:    int
    holdout_idx:   int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict
    qi:            str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{self.data_root}/sample_{self.sample_idx:02d}"

    @property
    def holdout_dir(self) -> str:
        return f"{self.data_root}/sample_{self.holdout_idx:02d}"

    @property
    def run_name(self) -> str:
        return (f"s{self.sample_idx:02d}__{self.sdg_label}"
                f"__{self.attack_method}__{self.qi}")

    @property
    def wandb_group(self) -> str:
        return f"linear-sweep-{self.dataset_key}-{self.dataset_size}"


def _data_root(cfg: dict, size: int) -> str:
    base = cfg["base"]
    n    = cfg["n_features"]
    suffix = f"_{n}feat" if n is not None else ""
    return f"/home/golobs/data/reconstruction_data/{base}/size_{size}{suffix}"


def generate_jobs(dataset_key: str, size: int) -> list[Job]:
    cfg = DATASET_CONFIGS[dataset_key]
    data_root = _data_root(cfg, size)

    jobs = []
    for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params), qi in itertools.product(
        SAMPLE_RANGE, SDG_METHODS, ATTACK_CONFIGS, cfg["qi_variants"]
    ):
        jobs.append(Job(
            dataset_key=dataset_key,
            dataset_name=cfg["name"],
            dataset_base=cfg["base"],
            dataset_size=size,
            dataset_type=cfg["data_type"],
            data_root=data_root,
            sample_idx=sample_idx,
            holdout_idx=(sample_idx + 1) % 5,
            sdg_method=sdg_method,
            sdg_params=dict(sdg_params),
            attack_method=attack_method,
            attack_params=dict(attack_params),
            qi=qi,
        ))
    return jobs


# ── Worker function (runs in a subprocess) ────────────────────────────────────

def _worker_setup_paths():
    """Configure sys.path for each worker process."""
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
    """Execute one experiment job. Runs in a worker subprocess."""
    _worker_setup_paths()
    sys.argv = sys.argv[:1]   # prevent master_experiment_script's argparse from choking

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    synth_path = Path(job.sample_dir) / job.sdg_label / "synth.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"synth.csv not found: {synth_path}")

    effective_attack_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_params,
    }

    cfg = {
        "dataset": {
            "name": job.dataset_name,
            "dir":  job.sample_dir,
            "size": job.dataset_size,
            "type": job.dataset_type,
        },
        "QI":          job.qi,
        "data_type":   job.dataset_type,
        "sdg_method":  job.sdg_method,
        "sdg_params":  job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": {
            "enabled":     True,
            "holdout_dir": job.holdout_dir,
        },
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            job.attack_method: dict(effective_attack_params),
        },
    }

    prepared = _prepare_config(cfg)

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config={
            "sample_idx":    job.sample_idx,
            "holdout_idx":   job.holdout_idx,
            "sdg_method":    job.sdg_method,
            "sdg_params":    job.sdg_params,
            "attack_method": job.attack_method,
            "attack_params": effective_attack_params,
            "qi":            job.qi,
            "dataset":       job.dataset_name,
            "dataset_key":   job.dataset_key,
            "size":          job.dataset_size,
        },
        tags=[job.dataset_key, job.dataset_name, f"size_{job.dataset_size}",
              "linear-sweep", job.attack_method, job.qi],
        group=job.wandb_group,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        # Always run with memorization test (round-robin holdout)
        recon_train   = _run_attack(prepared, synth, train,   qi, hidden_features)
        recon_holdout = _run_attack(prepared, synth, holdout, qi, hidden_features)

        train_scores   = _score_reconstruction(train,   recon_train,   hidden_features, job.dataset_type)
        holdout_scores = _score_reconstruction(holdout, recon_holdout, hidden_features, job.dataset_type)

        metrics = {}
        for feat, ts, hs in zip(hidden_features, train_scores, holdout_scores):
            metrics[f"RA_train_{feat}"]       = round(float(ts), 4)
            metrics[f"RA_nontraining_{feat}"] = round(float(hs), 4)
            metrics[f"RA_delta_{feat}"]       = round(float(ts - hs), 4)

        train_mean   = round(float(np.mean(train_scores)), 4)
        holdout_mean = round(float(np.mean(holdout_scores)), 4)
        metrics["RA_train_mean"]       = train_mean
        metrics["RA_nontraining_mean"] = holdout_mean
        metrics["RA_delta_mean"]       = round(train_mean - holdout_mean, 4)
        wandb.log(metrics)

        return {
            "sample":        job.sample_idx,
            "sdg":           job.sdg_label,
            "attack":        job.attack_method,
            "qi":            job.qi,
            "dataset":       job.dataset_key,
            "size":          job.dataset_size,
            "train_mean":    train_mean,
            "nontrain_mean": holdout_mean,
            "delta_mean":    metrics["RA_delta_mean"],
            "error":         None,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        wandb.log({"error": str(exc)})
        raise RuntimeError(f"{job.run_name}: {exc}\n{tb}") from exc

    finally:
        wandb.finish()


# ── Summary helpers ────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = ["dataset", "size", "qi", "attack", "sdg",
            "sample", "train_mean", "nontrain_mean", "delta_mean", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict], dataset_key: str, size: int):
    successes = [r for r in rows if r["error"] is None]
    failures  = [r for r in rows if r["error"] is not None]

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — {dataset_key}  size={size}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*70}")

    if not successes:
        return

    # Group by (qi, attack), average train_mean across SDG + samples
    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["qi"], r["attack"])
        if r["train_mean"] is not None:
            groups[key].append(r["train_mean"])

    print(f"\n  {'QI':<20}  {'Attack':<22}  {'Avg train RA (all SDG+samples)':>32}")
    print(f"  {'-'*76}")
    for (qi, attack), vals in sorted(groups.items()):
        mean_val = round(float(sum(vals) / len(vals)), 4)
        print(f"  {qi:<20}  {attack:<22}  {mean_val:>32.4f}")

    print(f"\n  --- Memorization delta (train - nontrain, avg) ---")
    delta_groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["qi"], r["attack"])
        if r["delta_mean"] is not None:
            delta_groups[key].append(r["delta_mean"])
    print(f"\n  {'QI':<20}  {'Attack':<22}  {'Avg delta (memorization)':>32}")
    print(f"  {'-'*76}")
    for (qi, attack), vals in sorted(delta_groups.items()):
        mean_val = round(float(sum(vals) / len(vals)), 4)
        print(f"  {qi:<20}  {attack:<22}  {mean_val:>32.4f}")

    print(f"{'='*70}\n")

    if failures:
        print(f"  Failed jobs ({len(failures)}):")
        for r in failures:
            print(f"    {r.get('run_name', r)}: {r['error']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LinearReconstruction vs ML classifiers — binary-feature sweep."
    )
    parser.add_argument("--dataset",  type=str, required=True,
                        choices=list(DATASET_CONFIGS),
                        help="Dataset to run: adult | cdc_diabetes | arizona")
    parser.add_argument("--size",     type=int, required=True,
                        help="Sample size (e.g. 1000 or 10000)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print job list and exit without running.")
    parser.add_argument("--serial",   action="store_true",
                        help="Run jobs sequentially in main process (easy Ctrl-C).")
    parser.add_argument("--workers",  type=int, default=N_WORKERS_DEFAULT,
                        help=f"Parallel workers (default {N_WORKERS_DEFAULT}).")
    parser.add_argument("--sample",   type=int, default=None,
                        help="Run only this sample index (0–4).")
    parser.add_argument("--sdg",      type=str, default=None,
                        help="Run only this SDG method name (e.g. 'MST').")
    parser.add_argument("--attack",   type=str, default=None,
                        help="Run only this attack (e.g. 'LinearReconstruction').")
    parser.add_argument("--qi",       type=str, default=None,
                        help="Run only this QI variant (e.g. 'QI_linear').")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "linear_sweep_progress.log"),
                        metavar="FILE",
                        help="Per-job progress log. Use 'tail -f FILE' to monitor.")
    args = parser.parse_args()

    all_jobs = generate_jobs(args.dataset, args.size)

    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs if j.sdg_method == args.sdg]
    if args.attack is not None:
        all_jobs = [j for j in all_jobs if j.attack_method == args.attack]
    if args.qi is not None:
        all_jobs = [j for j in all_jobs if j.qi == args.qi]

    wandb_group = all_jobs[0].wandb_group if all_jobs else f"linear-sweep-{args.dataset}-{args.size}"

    header = (
        f"{'='*70}\n"
        f"  Linear sweep: {args.dataset}  size={args.size}\n"
        f"  Jobs total:  {len(all_jobs)}\n"
        f"  Workers:     {args.workers}\n"
        f"  WandB group: {wandb_group}\n"
        f"  Memorization test: round-robin holdout (sample N → holdout (N+1)%5)\n"
        f"{'='*70}\n"
    )
    print(header, end="")

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}  holdout=sample_{j.holdout_idx:02d}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    # Verify sample + holdout directories exist before dispatching
    missing = []
    for job in all_jobs:
        for d in (job.sample_dir, job.holdout_dir):
            if not Path(d).exists():
                missing.append(d)
    missing = list(dict.fromkeys(missing))
    if missing:
        print("ERROR: missing sample/holdout directories:")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    Path(args.progress_log).parent.mkdir(parents=True, exist_ok=True)
    progress_log = open(args.progress_log, "w", buffering=1)
    progress_log.write(header)

    start_time = time.time()
    results: list[dict] = []

    def _handle_result(job: Job, result: dict | None, error: str | None):
        if error:
            row = {
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": job.qi,
                "dataset": job.dataset_key, "size": job.dataset_size,
                "train_mean": None, "nontrain_mean": None, "delta_mean": None,
                "error": error,
            }
            line = f"[FAIL]  {job.run_name}  {error[:120]}\n"
        else:
            row = result
            line = (f"[OK]  {job.run_name}"
                    f"  train={result['train_mean']:.4f}"
                    f"  nontrain={result['nontrain_mean']:.4f}"
                    f"  delta={result['delta_mean']:.4f}\n")
        results.append(row)
        print(line, end="")
        progress_log.write(line)

    if args.serial:
        for job in all_jobs:
            try:
                result = run_job(job)
                _handle_result(job, result, None)
            except Exception as exc:
                _handle_result(job, None, str(exc))
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            future_to_job = {pool.submit(run_job, job): job for job in all_jobs}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    _handle_result(job, result, None)
                except Exception as exc:
                    _handle_result(job, None, str(exc)[:300])

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed/60:.1f} min")
    progress_log.write(f"\nTotal time: {elapsed/60:.1f} min\n")
    progress_log.close()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"linear_sweep_{args.dataset}_{args.size}_{ts}.csv"
    _save_summary_csv(results, csv_path)
    _print_summary(results, args.dataset, args.size)


if __name__ == "__main__":
    main()
