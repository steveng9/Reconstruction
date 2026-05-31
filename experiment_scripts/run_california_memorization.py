#!/usr/bin/env python
"""
California Housing memorization-test sweep.

Runs reconstruction attacks on california/size_1000 with QI_large,
logging both training-set and holdout (non-member) scores for every
(sample × SDG method × attack) combination.

Attacks (continuous):
  Mean, LinearRegression, RandomForest, LightGBM

SDG methods: all that have synth.csv in california/size_1000/sample_{00-04}/

Memorization protocol (round-robin, 5 samples):
  sample N trains → holdout = sample (N+1)%5

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_california_memorization.py
    python experiment_scripts/run_california_memorization.py --dry-run
    python experiment_scripts/run_california_memorization.py --serial
    python experiment_scripts/run_california_memorization.py --workers 4
    python experiment_scripts/run_california_memorization.py --attack RandomForest
    python experiment_scripts/run_california_memorization.py --sdg MST
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
_SCRIPT_DIR = str(Path(__file__).parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(1, _SCRIPT_DIR)
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_NAME  = "california"
DATASET_BASE  = "california"
DATASET_SIZE  = 1_000
DATASET_TYPE  = "continuous"
QI_VARIANT    = "QI_large"

DATA_ROOT = f"/home/golobs/data/reconstruction_data/{DATASET_BASE}/size_{DATASET_SIZE}"

# Only samples 00-04 have SDG-generated synth.csv files.
SAMPLE_RANGE = list(range(5))   # [0, 1, 2, 3, 4]

# Round-robin holdout: sample N → holdout (N+1)%5
def _holdout_idx(sample_idx: int) -> int:
    return (sample_idx + 1) % len(SAMPLE_RANGE)


# All SDG methods present in every sample_00-04 directory.
SDG_METHODS = [
    ("MST",     {"epsilon": 0.1}),
    ("MST",     {"epsilon": 1.0}),
    ("MST",     {"epsilon": 10.0}),
    ("MST",     {"epsilon": 100.0}),
    ("MST",     {"epsilon": 1000.0}),
    ("AIM",     {"epsilon": 1.0}),
    ("AIM",     {"epsilon": 3.0}),
    ("AIM",     {"epsilon": 10.0}),
    ("ARF",     {}),
    ("CTGAN",   {}),
    ("TVAE",    {}),
    ("TabDDPM", {}),
    ("Synthpop",{}),
    ("RankSwap",{}),
]

# (attack_name, extra_params) — all resolved via ATTACK_REGISTRY["continuous"]
# Mean, LinearRegression, RandomForest, LightGBM already completed (280 jobs each).
ATTACK_CONFIGS = [
    ("BayesianRidge", {}),
]

N_WORKERS     = 8
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "california-memorization-QI_large"
WANDB_TAGS    = [DATASET_NAME, f"size_{DATASET_SIZE}", "memorization", QI_VARIANT]


# ── Job spec ───────────────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx:    int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{DATA_ROOT}/sample_{self.sample_idx:02d}"

    @property
    def holdout_dir(self) -> str:
        return f"{DATA_ROOT}/sample_{_holdout_idx(self.sample_idx):02d}"

    @property
    def run_name(self) -> str:
        return f"s{self.sample_idx:02d}__{self.sdg_label}__{self.attack_method}__{QI_VARIANT}"


def generate_jobs(
    sample_filter: int | None = None,
    sdg_filter: str | None = None,
    attack_filter: str | None = None,
) -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params) in itertools.product(
        SAMPLE_RANGE, SDG_METHODS, ATTACK_CONFIGS
    ):
        if sample_filter is not None and sample_idx != sample_filter:
            continue
        if sdg_filter is not None and sdg_method != sdg_filter:
            continue
        if attack_filter is not None and attack_method != attack_filter:
            continue
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=dict(sdg_params),
            attack_method=attack_method,
            attack_params=dict(attack_params),
        ))
    return jobs


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker_setup_paths():
    """Configure sys.path in each subprocess."""
    reconstruction = "/home/golobs/Reconstruction"
    if reconstruction in sys.path:
        sys.path.remove(reconstruction)
    sys.path.insert(0, reconstruction)
    for p in [
        "/home/golobs/MIA_on_diffusion/",
        "/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM",
        "/home/golobs/recon-synth",
        "/home/golobs/recon-synth/attacks",
        "/home/golobs/recon-synth/attacks/solvers",
    ]:
        if p not in sys.path:
            sys.path.append(p)


def run_job(job: Job) -> dict[str, Any]:
    _worker_setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    effective_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_params,
    }

    cfg = {
        "dataset": {
            "name": DATASET_NAME,
            "dir":  job.sample_dir,
            "size": DATASET_SIZE,
            "type": DATASET_TYPE,
        },
        "QI":           QI_VARIANT,
        "data_type":    DATASET_TYPE,
        "sdg_method":   job.sdg_method,
        "sdg_params":   job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": {
            "enabled":     True,
            "holdout_dir": job.holdout_dir,
        },
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            job.attack_method: dict(effective_params),
        },
    }

    prepared = _prepare_config(cfg)

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config={
            "sample_idx":    job.sample_idx,
            "sdg_method":    job.sdg_method,
            "sdg_params":    job.sdg_params,
            "attack_method": job.attack_method,
            "attack_params": effective_params,
            "qi":            QI_VARIANT,
            "dataset":       DATASET_NAME,
            "size":          DATASET_SIZE,
            "holdout_idx":   _holdout_idx(job.sample_idx),
        },
        tags=WANDB_TAGS,
        group=WANDB_GROUP,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        recon_train   = _run_attack(prepared, synth, train,   qi, hidden_features)
        recon_holdout = _run_attack(prepared, synth, holdout, qi, hidden_features)

        train_scores   = _score_reconstruction(train,   recon_train,   hidden_features, DATASET_TYPE)
        holdout_scores = _score_reconstruction(holdout, recon_holdout, hidden_features, DATASET_TYPE)

        metrics = {}
        for feat, ts, hs in zip(hidden_features, train_scores, holdout_scores):
            metrics[f"RA_train_{feat}"]       = ts
            metrics[f"RA_nontraining_{feat}"] = hs
            metrics[f"RA_delta_{feat}"]        = round(ts - hs, 4)

        train_mean   = round(float(np.mean(train_scores)),   4)
        holdout_mean = round(float(np.mean(holdout_scores)), 4)
        metrics["RA_train_mean"]       = train_mean
        metrics["RA_nontraining_mean"] = holdout_mean
        metrics["RA_delta_mean"]       = round(train_mean - holdout_mean, 4)
        wandb.log(metrics)

        return {
            "sample":        job.sample_idx,
            "sdg":           job.sdg_label,
            "attack":        job.attack_method,
            "train_mean":    train_mean,
            "nontrain_mean": holdout_mean,
            "delta_mean":    metrics["RA_delta_mean"],
            "error":         None,
            **{f"RA_train_{f}":       ts for f, ts in zip(hidden_features, train_scores)},
            **{f"RA_nontraining_{f}": hs for f, hs in zip(hidden_features, holdout_scores)},
            **{f"RA_delta_{f}":        round(ts - hs, 4) for f, ts, hs in zip(hidden_features, train_scores, holdout_scores)},
        }

    except Exception as exc:
        tb = traceback.format_exc()
        try:
            wandb.log({"error": str(exc)})
        except Exception:
            pass
        return {
            "sample": job.sample_idx, "sdg": job.sdg_label,
            "attack": job.attack_method,
            "train_mean": None, "nontrain_mean": None, "delta_mean": None,
            "error": str(exc),
        }

    finally:
        wandb.finish()


# ── Summary ────────────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    base_keys = ["sample", "sdg", "attack",
                 "train_mean", "nontrain_mean", "delta_mean", "error"]
    feat_keys = sorted({k for r in rows for k in r
                        if k.startswith("RA_") and not k.endswith("_mean")})
    keys = base_keys + feat_keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict]):
    import numpy as np
    from collections import defaultdict

    successes = [r for r in rows if r["error"] is None]
    failures  = [r for r in rows if r["error"] is not None]

    print(f"\n{'='*72}")
    print(f"  california memorization sweep — {WANDB_GROUP}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*72}")

    if failures:
        print("\n  FAILURES:")
        for r in failures:
            print(f"    s{r['sample']:02d} {r['sdg']:20s} {r['attack']:20s}  {r['error'][:60]}")

    if not successes:
        return

    groups: dict[str, list] = defaultdict(list)
    for r in successes:
        groups[r["attack"]].append((r["train_mean"], r["nontrain_mean"], r["delta_mean"]))

    print(f"\n  {'Attack':<20}  {'Train':>7}  {'Non-train':>10}  {'Delta':>7}")
    print(f"  {'-'*50}")
    for attack, vals in sorted(groups.items()):
        tr  = round(float(np.mean([v[0] for v in vals])), 4)
        nt  = round(float(np.mean([v[1] for v in vals])), 4)
        dt  = round(float(np.mean([v[2] for v in vals])), 4)
        print(f"  {attack:<20}  {tr:>7.4f}  {nt:>10.4f}  {dt:>7.4f}")
    print(f"{'='*72}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="California memorization-test sweep.")
    parser.add_argument("--dry-run",  action="store_true", help="Print job list and exit.")
    parser.add_argument("--serial",   action="store_true", help="Run jobs sequentially in main process.")
    parser.add_argument("--workers",  type=int, default=N_WORKERS, help="Parallel worker count.")
    parser.add_argument("--sample",   type=int, default=None, help="Run only this sample index (0-4).")
    parser.add_argument("--sdg",      type=str, default=None, help="Filter to one SDG method name.")
    parser.add_argument("--attack",   type=str, default=None, help="Filter to one attack name.")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "california_mem_progress.log"),
                        metavar="FILE",
                        help="Write one progress line per completed job to FILE.")
    args = parser.parse_args()

    jobs = generate_jobs(
        sample_filter=args.sample,
        sdg_filter=args.sdg,
        attack_filter=args.attack,
    )

    if args.dry_run:
        print(f"DRY RUN — {len(jobs)} jobs:")
        for j in jobs:
            print(f"  s{j.sample_idx:02d}  {j.sdg_label:<22}  {j.attack_method:<20}  holdout→s{_holdout_idx(j.sample_idx):02d}")
        return

    progress_log = Path(args.progress_log)
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(jobs)} jobs  |  workers={args.workers if not args.serial else 1}  |  dataset={DATASET_NAME}/{DATASET_SIZE}  |  QI={QI_VARIANT}")
    print(f"Progress log: {progress_log}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = Path(__file__).parent / f"california_memorization_{ts}.csv"

    rows: list[dict] = []

    def _handle_result(result: dict, job: Job):
        rows.append(result)
        status = "OK" if result["error"] is None else "FAIL"
        tr  = f"{result['train_mean']:.4f}"   if result["train_mean"]   is not None else "N/A"
        nt  = f"{result['nontrain_mean']:.4f}" if result["nontrain_mean"] is not None else "N/A"
        dt  = f"{result['delta_mean']:.4f}"    if result["delta_mean"]    is not None else "N/A"
        line = (
            f"[{status}] s{job.sample_idx:02d} {job.sdg_label:<22} "
            f"{job.attack_method:<20}  train={tr}  nt={nt}  delta={dt}"
        )
        print(line)
        with open(progress_log, "a") as f:
            f.write(line + "\n")

    if args.serial:
        for job in jobs:
            result = run_job(job)
            _handle_result(result, job)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "sample": job.sample_idx, "sdg": job.sdg_label,
                        "attack": job.attack_method,
                        "train_mean": None, "nontrain_mean": None, "delta_mean": None,
                        "error": str(exc),
                    }
                _handle_result(result, job)

    _print_summary(rows)
    _save_summary_csv(rows, out_csv)


if __name__ == "__main__":
    main()
