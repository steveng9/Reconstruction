#!/usr/bin/env python
"""
Fill-in sweep: run specific missing (SDG × attack) combinations.

Targets adult data, size=1000, all 5 samples.
Logs to the same WandB group as the main production sweep so results
appear in the same table.

Usage:
    conda activate recon_
    python experiment_scripts/run_fill_in_sweep.py
    python experiment_scripts/run_fill_in_sweep.py --dry-run
    python experiment_scripts/run_fill_in_sweep.py --workers 8
    python experiment_scripts/run_fill_in_sweep.py --serial    # main process, Ctrl-C friendly
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


# ── Dataset config ─────────────────────────────────────────────────────────────

DATASET_BASE  = "adult"
DATASET_NAME  = "adult"
DATASET_SIZE  = 1_000
N_FEATURES    = None
DATA_ROOT     = f"/home/golobs/data/reconstruction_data/{DATASET_BASE}/size_{DATASET_SIZE}"
SAMPLE_RANGE  = list(range(5))          # sample_00 through sample_04
DATASET_TYPE  = "categorical"
QI_VARIANTS   = ["QI1"]

WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "main attack sweep 1"   # keep consistent with production sweep
WANDB_TAGS    = [DATASET_NAME, f"size_{DATASET_SIZE}", "production", "fill-in"]

N_WORKERS     = 4


# ── Fill-in job specification ──────────────────────────────────────────────────
#
# Each entry is (sdg_method, sdg_params, attack_method, attack_params).
# All 5 samples × all QI_VARIANTS are crossed automatically.

FILL_IN_JOBS = [
    # ── AIM eps=10 — all missing attacks ──────────────────────────────────────
    #("AIM", {"epsilon": 10.0}, "Mode",                 {}),
    #("AIM", {"epsilon": 10.0}, "Random",               {}),
    #("AIM", {"epsilon": 10.0}, "MeasureDeid",          {}),
    #("AIM", {"epsilon": 10.0}, "KNN",                  {}),
    #("AIM", {"epsilon": 10.0}, "NaiveBayes",           {}),
    #("AIM", {"epsilon": 10.0}, "LogisticRegression",   {}),
    #("AIM", {"epsilon": 10.0}, "SVM",                  {}),
    #("AIM", {"epsilon": 10.0}, "RandomForest",         {}),
    #("AIM", {"epsilon": 10.0}, "LightGBM",             {}),
    #("AIM", {"epsilon": 10.0}, "MLP",                  {}),
    #("AIM", {"epsilon": 10.0}, "Attention",            {}),
    ("AIM", {"epsilon": 10.0}, "TabDDPM",              {"retrain": True}),
    ("AIM", {"epsilon": 10.0}, "TabDDPMWithMLP",       {"retrain": False, "retrain_mlp": True}),
    ("AIM", {"epsilon": 10.0}, "ConditionedRePaint",   {"retrain": False}),
    #("AIM", {"epsilon": 10.0}, "PartialMST",           {"retrain": False}),
    #("AIM", {"epsilon": 10.0}, "PartialMSTIndependent",{"retrain": False}),
    #("AIM", {"epsilon": 10.0}, "PartialMSTBounded",    {"retrain": False, "max_clique_size": 3}),
    # ── MST eps=0.1 — only logistic regression ────────────────────────────────
    #("MST", {"epsilon": 0.1},  "LogisticRegression",   {}),
]


# ── Job dataclass ──────────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx:    int
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
        return f"{DATA_ROOT}/sample_{self.sample_idx:02d}"

    @property
    def run_name(self) -> str:
        return f"s{self.sample_idx:02d}__{self.sdg_label}__{self.attack_method}__{self.qi}"


def generate_jobs() -> list[Job]:
    jobs = []
    for sample_idx, qi in itertools.product(SAMPLE_RANGE, QI_VARIANTS):
        for sdg_method, sdg_params, attack_method, attack_params in FILL_IN_JOBS:
            jobs.append(Job(
                sample_idx=sample_idx,
                sdg_method=sdg_method,
                sdg_params=dict(sdg_params),
                attack_method=attack_method,
                attack_params=dict(attack_params),
                qi=qi,
            ))
    return jobs


# ── Worker ─────────────────────────────────────────────────────────────────────

def _worker_setup_paths():
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
            "size": DATASET_SIZE,
            "type": DATASET_TYPE,
        },
        "QI":            job.qi,
        "data_type":     DATASET_TYPE,
        "sdg_method":    job.sdg_method,
        "sdg_params":    job.sdg_params or None,
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
            "qi":            job.qi,
            "dataset":       DATASET_NAME,
            "size":          DATASET_SIZE,
        },
        tags=WANDB_TAGS,
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
            "attack": job.attack_method, "qi": job.qi,
            "ra_mean": ra_mean, "error": None,
            **feat_scores,
        }

    except Exception as exc:
        wandb.log({"error": str(exc)})
        raise

    finally:
        wandb.finish()


# ── Summary ────────────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    base_keys = ["sample", "sdg", "attack", "qi", "ra_mean", "error"]
    feat_keys = sorted({k for r in rows for k in r if k.startswith("RA_") and k != "RA_mean"})
    keys = base_keys + feat_keys
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict]):
    from collections import defaultdict
    import numpy as np

    successes = [r for r in rows if r.get("error") is None]
    failures  = [r for r in rows if r.get("error") is not None]
    print(f"\n{'='*70}")
    print(f"  FILL-IN SWEEP COMPLETE — {DATASET_NAME} size={DATASET_SIZE}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*70}")
    if not successes:
        return
    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        if r.get("ra_mean") is not None:
            groups[(r["attack"], r["sdg"])].append(r["ra_mean"])
    print(f"\n  {'Attack':<30}  {'SDG':<18}  {'Mean RA':>10}")
    print(f"  {'-'*62}")
    for (attack, sdg), vals in sorted(groups.items()):
        print(f"  {attack:<30}  {sdg:<18}  {float(np.mean(vals)):>10.4f}")
    print(f"{'='*70}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fill-in sweep for missing (SDG, attack) pairs.")
    parser.add_argument("--dry-run",      action="store_true", help="Print job list and exit.")
    parser.add_argument("--serial",       action="store_true", help="Run sequentially in main process.")
    parser.add_argument("--workers",      type=int, default=N_WORKERS, help="Number of parallel workers.")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "fill_in_progress.log"),
                        metavar="FILE",
                        help="Write one line per completed job. Use 'tail -f FILE' to monitor.")
    args = parser.parse_args()

    all_jobs = generate_jobs()

    print(
        f"{'='*70}\n"
        f"  Fill-in sweep: {DATASET_NAME} size={DATASET_SIZE}\n"
        f"  Jobs total:  {len(all_jobs)}\n"
        f"  Workers:     {args.workers}\n"
        f"  WandB group: {WANDB_GROUP}\n"
        f"{'='*70}\n"
    )

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    missing = list(dict.fromkeys(
        j.sample_dir for j in all_jobs if not Path(j.sample_dir).exists()
    ))
    if missing:
        print("ERROR: missing sample directories:")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    progress_log = open(args.progress_log, "w", buffering=1) if args.progress_log else None

    start_time = time.time()
    results: list[dict] = []
    n_done = 0
    n_fail = 0
    width = len(str(len(all_jobs)))

    def _handle_result(job, result_or_exc):
        nonlocal n_done, n_fail
        n_done += 1
        elapsed = time.time() - start_time
        eta_str = ""
        if n_done > 1:
            rate    = n_done / elapsed
            eta_s   = (len(all_jobs) - n_done) / rate
            eta_str = f"  ETA {eta_s/60:.0f}m"
        if isinstance(result_or_exc, Exception):
            n_fail += 1
            line = f"  [{n_done:>{width}}/{len(all_jobs)}]  FAILED  {job.run_name}:  {result_or_exc}"
            results.append({"sample": job.sample_idx, "sdg": job.sdg_label,
                            "attack": job.attack_method, "qi": job.qi,
                            "ra_mean": None, "error": str(result_or_exc)})
        else:
            val = result_or_exc.get("ra_mean")
            val_str = f"{val:.4f}" if val is not None else "N/A"
            line = (f"  [{n_done:>{width}}/{len(all_jobs)}]"
                    f"  {job.run_name:<60}  RA={val_str}{eta_str}")
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"fill_in_results_{DATASET_NAME}_{ts}.csv"
    _save_summary_csv(results, csv_path)

    if progress_log:
        progress_log.close()


if __name__ == "__main__":
    main()
