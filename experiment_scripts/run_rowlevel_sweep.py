#!/usr/bin/env python
"""
Row-level RA sweep: saves per-record scores for later analysis (e.g. RA vs. equivalence class size).

Runs a subset of attacks on adult 1k and 10k, a handful of SDG methods, sample_00 only
(trial #1). Enables row_level_analysis so each job writes a CSV with per-record scores,
QI values, and true/reconstructed feature values.

Row-score CSVs land at:
    {sample_dir}/{sdg_dir}/row_scores/{attack}__{qi}__train_run0.csv

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_rowlevel_sweep.py
    python experiment_scripts/run_rowlevel_sweep.py --dry-run
    python experiment_scripts/run_rowlevel_sweep.py --workers 4
    python experiment_scripts/run_rowlevel_sweep.py --attack RandomForest --size 10000
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


# ── Configuration ──────────────────────────────────────────────────────────────

DATASETS = [
    {"base": "adult", "name": "adult", "size": 1_000},
    {"base": "adult", "name": "adult", "size": 10_000},
]

DATA_ROOT_TPL = "/home/golobs/data/reconstruction_data/{base}/size_{size}"

SAMPLE_RANGE  = [0]   # sample_00 only for trial #1
DATASET_TYPE  = "categorical"
QI_VARIANTS   = ["QI1"]

SDG_METHODS = [
    ("AIM",  {"epsilon": 1.0}),
    ("MST",  {"epsilon": 1.0}),
    ("MST",  {"epsilon": 10.0}),
    ("MST",  {"epsilon": 100.0}),
    ("MST",  {"epsilon": 1000.0}),
]

# (attack_method, extra_params_override)
ATTACK_CONFIGS = [
    ("Mode",                       {}),
    ("KNN",                        {}),
    ("NaiveBayes",                 {}),
    ("RandomForest",               {}),
    ("MarginalRF_graphQI_entropyBP", {}),
    ("MLP",                        {}),
    ("LightGBM",                   {}),
]

N_WORKERS     = 4
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "rowlevel-sweep-adult"
WANDB_TAGS    = ["adult", "rowlevel", "trial1"]


# ── Job specification ─────────────────────────────────────────────────────────

@dataclass
class Job:
    dataset_base:  str
    dataset_name:  str
    dataset_size:  int
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
    def data_root(self) -> str:
        return DATA_ROOT_TPL.format(base=self.dataset_base, size=self.dataset_size)

    @property
    def sample_dir(self) -> str:
        return f"{self.data_root}/sample_{self.sample_idx:02d}"

    @property
    def run_name(self) -> str:
        return (f"{self.dataset_name}_s{self.dataset_size}"
                f"_s{self.sample_idx:02d}__{self.sdg_label}"
                f"__{self.attack_method}__{self.qi}")


def generate_jobs(size_filter=None, attack_filter=None) -> list[Job]:
    jobs = []
    for ds_cfg, sample_idx, (sdg_method, sdg_params), (attack_method, attack_params), qi in itertools.product(
        DATASETS, SAMPLE_RANGE, SDG_METHODS, ATTACK_CONFIGS, QI_VARIANTS
    ):
        if size_filter   is not None and ds_cfg["size"]   != size_filter:   continue
        if attack_filter is not None and attack_method    != attack_filter:  continue
        jobs.append(Job(
            dataset_base=ds_cfg["base"],
            dataset_name=ds_cfg["name"],
            dataset_size=ds_cfg["size"],
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=sdg_params,
            attack_method=attack_method,
            attack_params=dict(attack_params),
            qi=qi,
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
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction, _save_row_scores

    sample_dir = job.sample_dir
    synth_path = Path(sample_dir) / job.sdg_label / "synth.csv"
    if not synth_path.exists():
        return {"error": f"no synth.csv: {synth_path}", "sample": job.sample_idx,
                "sdg": job.sdg_label, "attack": job.attack_method, "qi": job.qi,
                "dataset": job.dataset_name, "size": job.dataset_size, "ra_mean": None}

    cfg = {
        "dataset": {
            "name": job.dataset_name,
            "dir":  sample_dir,
            "size": job.dataset_size,
            "type": DATASET_TYPE,
        },
        "QI":            job.qi,
        "data_type":     DATASET_TYPE,
        "sdg_method":    job.sdg_method,
        "sdg_params":    job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": {"enabled": False},
        "row_level_analysis": {"enabled": True},
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
            "dataset":       job.dataset_name,
            "size":          job.dataset_size,
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

        # Save per-record scores CSV
        _save_row_scores(prepared, train, recon, hidden_features, qi,
                         DATASET_TYPE, run_id=0, is_training=True)

        feat_scores = {k: v for k, v in metrics.items() if k.startswith("RA_") and k != "RA_mean"}
        return {
            "dataset": job.dataset_name, "size": job.dataset_size,
            "sample": job.sample_idx, "sdg": job.sdg_label,
            "attack": job.attack_method, "qi": job.qi,
            "ra_mean": ra_mean, "error": None,
            "_wandb_run_id": wandb.run.id,
            **feat_scores,
        }

    except Exception as exc:
        wandb.log({"error": str(exc)})
        tb = traceback.format_exc()
        print(f"\nERROR {job.run_name}:\n{tb}", file=sys.stderr)
        return {"error": str(exc), "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": job.qi,
                "dataset": job.dataset_name, "size": job.dataset_size, "ra_mean": None}
    finally:
        wandb.finish()


# ── Progress logging ──────────────────────────────────────────────────────────

def _log_progress(path: Path, msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if path:
        with open(path, "a") as f:
            f.write(line + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    jobs = generate_jobs(
        size_filter=args.size,
        attack_filter=args.attack,
    )

    if args.dry_run:
        print(f"DRY RUN — {len(jobs)} jobs:")
        for j in jobs:
            print(f"  {j.run_name}")
        return

    progress_log = Path(args.progress_log) if args.progress_log else None
    if progress_log:
        progress_log.parent.mkdir(parents=True, exist_ok=True)

    n_workers = args.workers if not args.serial else 1
    print(f"Running {len(jobs)} jobs with {n_workers} worker(s)...\n")

    all_rows = []
    t0 = time.time()

    if args.serial:
        for i, job in enumerate(jobs, 1):
            _log_progress(progress_log, f"[{i}/{len(jobs)}] START {job.run_name}")
            try:
                result = run_job(job)
                status = f"ra={result.get('ra_mean')}" if not result.get("error") else f"ERROR: {result['error'][:60]}"
                _log_progress(progress_log, f"[{i}/{len(jobs)}] DONE  {job.run_name}  {status}")
            except Exception as exc:
                _log_progress(progress_log, f"[{i}/{len(jobs)}] FAIL  {job.run_name}  {exc}")
                result = {"error": str(exc), "sample": job.sample_idx, "sdg": job.sdg_label,
                          "attack": job.attack_method, "qi": job.qi,
                          "dataset": job.dataset_name, "size": job.dataset_size, "ra_mean": None}
            all_rows.append(result)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_job, job): job for job in jobs}
            for i, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    result = fut.result()
                    status = f"ra={result.get('ra_mean')}" if not result.get("error") else f"ERROR: {result['error'][:60]}"
                except Exception as exc:
                    result = {"error": str(exc), "sample": job.sample_idx, "sdg": job.sdg_label,
                              "attack": job.attack_method, "qi": job.qi,
                              "dataset": job.dataset_name, "size": job.dataset_size, "ra_mean": None}
                    status = f"EXCEPTION: {exc}"
                _log_progress(progress_log, f"[{i}/{len(jobs)}] {job.run_name}  {status}")
                all_rows.append(result)

    elapsed = time.time() - t0
    print(f"\nCompleted {len(all_rows)} jobs in {elapsed:.0f}s")

    # Save summary CSV
    if all_rows:
        base_keys = ["dataset", "size", "sample", "sdg", "attack", "qi", "ra_mean", "error"]
        feat_keys = sorted({k for r in all_rows for k in r if k.startswith("RA_") and k != "RA_mean"})
        keys = base_keys + feat_keys
        out_path = Path(__file__).parent / f"rowlevel_sweep_{datetime.now():%Y%m%d_%H%M%S}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Summary CSV → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--serial",       action="store_true", help="Run in main process (Ctrl-C killable)")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--size",         type=int, default=None, help="Filter to one dataset size (1000 or 10000)")
    parser.add_argument("--attack",       default=None, help="Filter to one attack method name")
    parser.add_argument("--progress-log", default="outfiles/rowlevel_progress.log", dest="progress_log")
    main(parser.parse_args())
