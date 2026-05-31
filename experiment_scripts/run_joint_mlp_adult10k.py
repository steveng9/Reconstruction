#!/usr/bin/env python
"""
JointMLP sweep: adult 10k, all 5 samples, QI1, all requested SDG methods.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_joint_mlp_adult10k.py
    python experiment_scripts/run_joint_mlp_adult10k.py --dry-run
    python experiment_scripts/run_joint_mlp_adult10k.py --workers 4
    python experiment_scripts/run_joint_mlp_adult10k.py --serial   # Ctrl-C friendly
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
_EXPERIMENT_SCRIPTS_DIR = str(Path(__file__).parent)
if _EXPERIMENT_SCRIPTS_DIR not in sys.path:
    sys.path.insert(1, _EXPERIMENT_SCRIPTS_DIR)
from attack_defaults import ATTACK_PARAM_DEFAULTS
from results_db import ResultsDB


# ── Configuration ──────────────────────────────────────────────────────────────

DATASET_BASE  = "adult"
DATASET_NAME  = "adult"
DATASET_SIZE  = 10_000
N_FEATURES    = None
DATA_ROOT     = f"/home/golobs/data/reconstruction_data/{DATASET_BASE}/size_{DATASET_SIZE}"
SAMPLE_RANGE  = list(range(5))
DATASET_TYPE  = "categorical"

HOLDOUT_SAMPLE_IDX = None

QI_VARIANTS = ["QI1"]

SDG_METHODS = [
    ("RankSwap",        {}),
    ("CellSuppression", {}),
    ("Synthpop",        {}),
    ("MST",             {"epsilon": 0.1}),
    ("MST",             {"epsilon": 1.0}),
    ("MST",             {"epsilon": 10.0}),
    ("MST",             {"epsilon": 100.0}),
    ("MST",             {"epsilon": 1000.0}),
    ("AIM",             {"epsilon": 1.0}),
    ("TVAE",            {}),
    ("CTGAN",           {}),
    ("ARF",             {}),
    ("TabDDPM",         {}),
]

ATTACK_CONFIGS = [
    ("JointMLP", {}),
]

N_WORKERS     = 4
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "joint-mlp-adult-10k"
WANDB_TAGS    = [DATASET_NAME, f"size_{DATASET_SIZE}", "JointMLP"]


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx:    int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict
    qi:            str
    gpu_id:        int = 0   # physical GPU assigned to this job (round-robin at generation time)

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
            gpu_id=len(jobs) % 2,   # round-robin across GPU 0 and GPU 1
        ))
    return jobs


# ── Worker function ────────────────────────────────────────────────────────────

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
    # Set before any CUDA / torch import so PyTorch sees only the assigned GPU as "cuda:0".
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

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

    effective_attack_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_params,
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
            "ra_mean": ra_mean,
            "train_mean": None, "nontrain_mean": None, "delta_mean": None,
            "error": None,
            "_wandb_run_id": wandb.run.id,
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
    base_keys = ["sample", "sdg", "attack", "qi", "ra_mean", "error"]
    feat_keys = sorted({k for r in rows for k in r if k.startswith("RA_") and k != "RA_mean"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_keys + feat_keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict]):
    successes = [r for r in rows if r.get("error") is None]
    failures  = [r for r in rows if r.get("error") is not None]
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — {DATASET_NAME} (JointMLP)")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    if successes:
        import numpy as np
        vals = [r["ra_mean"] for r in successes if r["ra_mean"] is not None]
        if vals:
            print(f"  Mean RA across all jobs: {np.mean(vals):.4f}")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JointMLP sweep: adult 10k.")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--serial",       action="store_true")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--sample",       type=int, default=None)
    parser.add_argument("--sdg",          type=str, default=None)
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "joint_mlp_progress.log"),
                        metavar="FILE")
    args = parser.parse_args()

    all_jobs = generate_jobs()
    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs if j.sdg_label == args.sdg or j.sdg_method == args.sdg]

    os.makedirs(Path(args.progress_log).parent, exist_ok=True)
    progress_log = open(args.progress_log, "w", buffering=1)

    header = (
        f"{'='*70}\n"
        f"  JointMLP sweep: {DATASET_NAME} size={DATASET_SIZE}\n"
        f"  Jobs total: {len(all_jobs)}  Workers: {args.workers}\n"
        f"  WandB group: {WANDB_GROUP}\n"
        f"{'='*70}\n"
    )
    print(header, end="")
    progress_log.write(header)

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    missing = list(dict.fromkeys(
        job.sample_dir for job in all_jobs if not Path(job.sample_dir).exists()
    ))
    if missing:
        print("ERROR: missing sample directories:")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    start_time = time.time()
    results: list[dict] = []
    n_done = n_fail = 0
    _db = ResultsDB()

    def _handle_result(job, result_or_exc):
        nonlocal n_done, n_fail
        n_done += 1
        elapsed = time.time() - start_time
        eta_str = ""
        if n_done > 1:
            rate    = n_done / elapsed
            eta_s   = (len(all_jobs) - n_done) / rate
            eta_str = f"  ETA {eta_s/60:.0f}m"
        w = len(str(len(all_jobs)))
        if isinstance(result_or_exc, Exception):
            n_fail += 1
            line = f"  [{n_done:>{w}}/{len(all_jobs)}]  FAILED  {job.run_name}:  {result_or_exc}"
            results.append({"sample": job.sample_idx, "sdg": job.sdg_label,
                            "attack": job.attack_method, "qi": job.qi,
                            "ra_mean": None, "error": str(result_or_exc)})
        else:
            r = result_or_exc
            val_str = f"{r['ra_mean']:.4f}" if r["ra_mean"] is not None else "N/A"
            line = f"  [{n_done:>{w}}/{len(all_jobs)}]  {job.run_name:<60}  RA={val_str}{eta_str}"
            results.append(r)
            try:
                feat_scores = {k[3:]: float(v) for k, v in r.items()
                               if k.startswith("RA_") and k != "RA_mean" and v is not None}
                if r["ra_mean"] is not None:
                    _db.insert_run(
                        dataset=DATASET_NAME, dataset_size=DATASET_SIZE,
                        sample=job.sample_idx, qi=job.qi,
                        sdg_method=job.sdg_label, attack_label=job.attack_method,
                        split="standard", ra_mean=r["ra_mean"],
                        feature_scores=feat_scores or None,
                        sdg_params=job.sdg_params,
                        wandb_run_id=r.get("_wandb_run_id"), wandb_group=WANDB_GROUP,
                        wandb_project=WANDB_PROJECT,
                    )
            except Exception as db_exc:
                print(f"  WARNING: DB write failed for {job.run_name}: {db_exc}")
        print(line)
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
    csv_path = Path(__file__).parent / f"joint_mlp_adult10k_{ts}.csv"
    _save_summary_csv(results, csv_path)
    progress_log.close()


if __name__ == "__main__":
    main()
