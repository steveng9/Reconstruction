#!/usr/bin/env python
"""
Extended LinearReconstruction sweep: multi-valued and joint hidden features.

Experiments:
  1. Categorical  — LinearReconstructionCategorical vs RandomForest
                    hidden = race (5 categories), QI_race_lowcard
  2. Joint        — LinearReconstructionJoint vs RandomForest
                    hidden = [income, sex] (4 joint categories), QI_joint_income_sex_lowcard

In both experiments, RandomForest serves as the ML baseline on the same hidden
feature(s) / QI configuration.  For the joint experiment, RF predicts each
feature independently; the LP solves them jointly.  RA is averaged across all
hidden features for both attacks.

Usage:
    conda activate recon_
    python experiment_scripts/run_linear_extended.py --size 1000
    python experiment_scripts/run_linear_extended.py --size 1000 --dry-run
    python experiment_scripts/run_linear_extended.py --size 1000 --serial
    python experiment_scripts/run_linear_extended.py --size 1000 --workers 4
    python experiment_scripts/run_linear_extended.py --size 1000 --experiment categorical
    python experiment_scripts/run_linear_extended.py --size 1000 --attack RandomForest
    python experiment_scripts/run_linear_extended.py --size 1000 --sdg MST
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


# ── Dataset ───────────────────────────────────────────────────────────────────

DATASET_BASE = "adult"
DATASET_NAME = "adult"        # QI lookup key in get_data.py
DATASET_TYPE = "categorical"
N_FEATURES   = None           # no _Nfeat suffix


# ── SDG methods ───────────────────────────────────────────────────────────────

SDG_METHODS = [
    ("MST",      {"epsilon": 10.0}),
    ("MST",      {"epsilon": 1000.0}),
    #("AIM",      {"epsilon": 1.0}),
    ("TVAE",     {}),
    ("TabDDPM",  {}),
    #("ARF",     {}),
    ("Synthpop", {}),
]


# ── Experiment definitions ────────────────────────────────────────────────────
# Each experiment specifies a QI variant and the attack methods to compare.
# The hidden feature(s) are determined by the QI variant (minus_QIs lookup).

EXPERIMENTS: dict[str, dict] = {
    # Single multi-valued hidden feature (race, 5 categories)
    "categorical": {
        "qi":      "QI_race_lowcard",
        "attacks": [
            ("LinearReconstructionCategorical", {}),
            ("RandomForest",                    {}),
        ],
        "description": "hidden=race (C=5), LP categorical vs RF baseline",
    },
    # Two binary hidden features reconstructed simultaneously
    "joint": {
        "qi":      "QI_joint_income_sex_lowcard",
        "attacks": [
            ("LinearReconstructionJoint", {}),
            ("RandomForest",              {}),
        ],
        "description": "hidden=[income, sex] (C=4 joint), LP joint vs RF baseline",
    },
}

SAMPLE_RANGE      = list(range(4))   # samples 00–03; sample_04 is NO_HOLDOUT on 10k
WANDB_PROJECT     = "tabular-reconstruction-attacks"
N_WORKERS_DEFAULT = 8


# ── Job ───────────────────────────────────────────────────────────────────────

@dataclass
class Job:
    experiment:    str   # "categorical" | "joint"
    dataset_size:  int
    data_root:     str
    sample_idx:    int
    holdout_idx:   int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict
    qi:            str
    no_holdout:    bool = False

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
        return (f"{self.experiment}__s{self.sample_idx:02d}"
                f"__{self.sdg_label}__{self.attack_method}__{self.qi}")

    @property
    def wandb_group(self) -> str:
        return f"linear-extended-adult-{self.dataset_size}"


def _data_root(size: int) -> str:
    suffix = f"_{N_FEATURES}feat" if N_FEATURES is not None else ""
    return f"/home/golobs/data/reconstruction_data/{DATASET_BASE}/size_{size}{suffix}"


def generate_jobs(size: int, experiment_filter: str | None = None,
                  no_holdout: bool = False) -> list[Job]:
    data_root = _data_root(size)
    jobs = []
    exps = (
        {experiment_filter: EXPERIMENTS[experiment_filter]}
        if experiment_filter
        else EXPERIMENTS
    )
    for exp_name, exp_cfg in exps.items():
        for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params) in itertools.product(
            SAMPLE_RANGE, SDG_METHODS, exp_cfg["attacks"]
        ):
            jobs.append(Job(
                experiment=exp_name,
                dataset_size=size,
                data_root=data_root,
                sample_idx=sample_idx,
                holdout_idx=(sample_idx + 1) % 4,
                sdg_method=sdg_method,
                sdg_params=dict(sdg_params),
                attack_method=attack_method,
                attack_params=dict(attack_params),
                qi=exp_cfg["qi"],
                no_holdout=no_holdout,
            ))
    return jobs


# ── Worker ────────────────────────────────────────────────────────────────────

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

    synth_path = Path(job.sample_dir) / job.sdg_label / "synth.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"synth.csv not found: {synth_path}")

    effective_attack_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_params,
    }

    cfg = {
        "dataset": {
            "name": DATASET_NAME,
            "dir":  job.sample_dir,
            "size": job.dataset_size,
            "type": DATASET_TYPE,
        },
        "QI":          job.qi,
        "data_type":   DATASET_TYPE,
        "sdg_method":  job.sdg_method,
        "sdg_params":  job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": {
            "enabled":     not job.no_holdout,
            "holdout_dir": job.holdout_dir if not job.no_holdout else None,
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
            "experiment":    job.experiment,
            "sample_idx":    job.sample_idx,
            "holdout_idx":   job.holdout_idx,
            "sdg_method":    job.sdg_method,
            "sdg_params":    job.sdg_params,
            "attack_method": job.attack_method,
            "attack_params": effective_attack_params,
            "qi":            job.qi,
            "dataset":       DATASET_NAME,
            "size":          job.dataset_size,
        },
        tags=["adult", f"size_{job.dataset_size}", "linear-extended",
              job.experiment, job.attack_method, job.qi],
        group=job.wandb_group,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        recon_train = _run_attack(prepared, synth, train, qi, hidden_features)
        train_scores = _score_reconstruction(train, recon_train, hidden_features, DATASET_TYPE)
        train_mean = round(float(np.mean(train_scores)), 4)

        metrics = {}
        for feat, ts in zip(hidden_features, train_scores):
            metrics[f"RA_train_{feat}"] = round(float(ts), 4)
        metrics["RA_train_mean"] = train_mean

        holdout_mean = None
        if not job.no_holdout:
            recon_holdout = _run_attack(prepared, synth, holdout, qi, hidden_features)
            holdout_scores = _score_reconstruction(holdout, recon_holdout, hidden_features, DATASET_TYPE)
            holdout_mean = round(float(np.mean(holdout_scores)), 4)
            for feat, ts, hs in zip(hidden_features, train_scores, holdout_scores):
                metrics[f"RA_nontraining_{feat}"] = round(float(hs), 4)
                metrics[f"RA_delta_{feat}"]       = round(float(ts - hs), 4)
            metrics["RA_nontraining_mean"] = holdout_mean
            metrics["RA_delta_mean"]       = round(train_mean - holdout_mean, 4)

        wandb.log(metrics)

        return {
            "experiment":    job.experiment,
            "sample":        job.sample_idx,
            "sdg":           job.sdg_label,
            "attack":        job.attack_method,
            "qi":            job.qi,
            "size":          job.dataset_size,
            "train_mean":    train_mean,
            "nontrain_mean": holdout_mean,
            "delta_mean":    round(train_mean - holdout_mean, 4) if holdout_mean is not None else None,
            "error":         None,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        wandb.log({"error": str(exc)})
        raise RuntimeError(f"{job.run_name}: {exc}\n{tb}") from exc

    finally:
        wandb.finish()


# ── Summary ───────────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = ["experiment", "size", "qi", "attack", "sdg",
            "sample", "train_mean", "nontrain_mean", "delta_mean", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict], size: int):
    successes = [r for r in rows if r["error"] is None]
    failures  = [r for r in rows if r["error"] is not None]

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE — adult  size={size}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*70}")

    if not successes:
        return

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["experiment"], r["qi"], r["attack"])
        if r["train_mean"] is not None:
            groups[key].append(r["train_mean"])

    print(f"\n  {'Experiment':<14}  {'QI':<30}  {'Attack':<30}  {'Avg train RA':>12}")
    print(f"  {'-'*90}")
    for (exp, qi, attack), vals in sorted(groups.items()):
        mean_val = round(sum(vals) / len(vals), 4)
        print(f"  {exp:<14}  {qi:<30}  {attack:<30}  {mean_val:>12.4f}")

    if failures:
        print(f"\n  Failed jobs ({len(failures)}):")
        for r in failures[:10]:
            print(f"    {r.get('run_name', str(r))}: {r['error']}")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extended LP reconstruction: multi-valued and joint hidden features."
    )
    parser.add_argument("--size",       type=int, default=1000,
                        help="Sample size (default 1000).")
    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENTS),
                        help="Run only this experiment type.")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print job list and exit without running.")
    parser.add_argument("--serial",     action="store_true",
                        help="Run sequentially in main process (easy Ctrl-C).")
    parser.add_argument("--workers",    type=int, default=N_WORKERS_DEFAULT,
                        help=f"Parallel workers (default {N_WORKERS_DEFAULT}).")
    parser.add_argument("--sample",     type=int, default=None,
                        help="Run only this sample index (0–4).")
    parser.add_argument("--sdg",        type=str, default=None,
                        help="Run only this SDG method name.")
    parser.add_argument("--attack",     type=str, default=None,
                        help="Run only this attack name.")
    parser.add_argument("--no-holdout", action="store_true",
                        help="Skip memorization test (no holdout scoring).")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "linear_extended_progress.log"),
                        metavar="FILE")
    args = parser.parse_args()

    all_jobs = generate_jobs(args.size, args.experiment, no_holdout=args.no_holdout)

    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs if j.sdg_method == args.sdg]
    if args.attack is not None:
        all_jobs = [j for j in all_jobs if j.attack_method == args.attack]

    wandb_group = all_jobs[0].wandb_group if all_jobs else f"linear-extended-adult-{args.size}"

    exp_summary = "\n".join(
        f"    {name}: {cfg['description']}"
        for name, cfg in EXPERIMENTS.items()
        if args.experiment is None or name == args.experiment
    )
    header = (
        f"{'='*70}\n"
        f"  Linear extended sweep: adult  size={args.size}\n"
        f"  Experiments:\n{exp_summary}\n"
        f"  Jobs total:  {len(all_jobs)}\n"
        f"  Workers:     {args.workers}\n"
        f"  WandB group: {wandb_group}\n"
        f"{'='*70}\n"
    )
    print(header, end="")

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            holdout_str = "  no-holdout" if j.no_holdout else f"  holdout=sample_{j.holdout_idx:02d}"
            print(f"  [{i+1:>4d}]  {j.run_name}{holdout_str}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    missing = []
    for job in all_jobs:
        dirs = [job.sample_dir] if job.no_holdout else [job.sample_dir, job.holdout_dir]
        for d in dirs:
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
                "experiment": job.experiment, "sample": job.sample_idx,
                "sdg": job.sdg_label, "attack": job.attack_method,
                "qi": job.qi, "size": job.dataset_size,
                "train_mean": None, "nontrain_mean": None, "delta_mean": None,
                "error": error,
            }
            line = f"[FAIL]  {job.run_name}  {error[:120]}\n"
        else:
            row = result
            nt = result['nontrain_mean']
            dl = result['delta_mean']
            nt_str = f"{nt:.4f}" if nt is not None else "n/a"
            dl_str = f"{dl:.4f}" if dl is not None else "n/a"
            line = (f"[OK]  {job.run_name}"
                    f"  train={result['train_mean']:.4f}"
                    f"  nontrain={nt_str}"
                    f"  delta={dl_str}\n")
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
    csv_path = Path(__file__).parent / f"linear_extended_adult_{args.size}_{ts}.csv"
    _save_summary_csv(results, csv_path)
    _print_summary(results, args.size)


if __name__ == "__main__":
    main()
