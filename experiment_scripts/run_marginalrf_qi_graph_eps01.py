#!/usr/bin/env python
"""
run_marginalrf_qi_graph_eps01.py

Runs the two QI-graph MarginalRF variants on MST_eps0.1, adult 10k, QI1,
across all 5 samples — filling the gap identified in the main paper table.

Two attack variants:
  MarginalRF_QIGraph:          qi_in_graph=True,  entropy_weighted=False
  MarginalRF_QIGraph_EntropyBP: qi_in_graph=True,  entropy_weighted=True

MST_eps0.1 was intentionally omitted from the original variants and combos sweeps
(those scripts only included eps 1, 10, etc.).

Usage:
    conda activate recon_
    python experiment_scripts/run_marginalrf_qi_graph_eps01.py
    python experiment_scripts/run_marginalrf_qi_graph_eps01.py --dry-run
    python experiment_scripts/run_marginalrf_qi_graph_eps01.py --workers 4
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import multiprocessing as mp

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_NAME  = "adult"
DATASET_SIZE  = 10_000
DATASET_TYPE  = "categorical"
DATA_ROOT     = f"/home/golobs/data/reconstruction_data/adult/size_{DATASET_SIZE}"
QI_VARIANTS   = ["QI1"]
SAMPLE_RANGE  = list(range(5))   # samples 00–04

# Only MST_eps0.1 (that's the missing column)
SDG_METHODS = [
    ("MST", {"epsilon": 0.1}),
]

_MRF = ATTACK_PARAM_DEFAULTS["MarginalRF"]

ATTACK_CONFIGS = [
    # ── Variant 1: QI nodes in graph, no entropy weighting ───────────────
    # Label used in the variants sweep files.
    (
        "MarginalRF_QIGraph",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": {**dict(_MRF),
                           "qi_in_graph":      True,
                           "entropy_weighted": False},
        },
    ),
    # ── Variant 2: QI graph + entropy-weighted BP ─────────────────────────
    # Same label used in the combos sweep files.
    (
        "MarginalRF_QIGraph_EntropyBP",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": {**dict(_MRF),
                           "qi_in_graph":      True,
                           "entropy_weighted": True},
        },
    ),
]

N_WORKERS     = 4
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "marginalrf-qi-graph-eps01-adult-10k"


# ── Job spec ──────────────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx:    int
    sdg_method:    str
    sdg_params:    dict
    attack_label:  str
    attack_method: str
    attack_params: dict
    qi:            str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{DATA_ROOT}/sample_{self.sample_idx:02d}"

    @property
    def run_name(self) -> str:
        return (
            f"{DATASET_NAME}__sz{DATASET_SIZE}__"
            f"s{self.sample_idx:02d}__{self.sdg_label}__"
            f"{self.attack_label}__{self.qi}"
        )


def generate_jobs() -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), (label, method, params), qi in itertools.product(
        SAMPLE_RANGE, SDG_METHODS, ATTACK_CONFIGS, QI_VARIANTS,
    ):
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=dict(sdg_params),
            attack_label=label,
            attack_method=method,
            attack_params=dict(params),
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

    sample_dir = job.sample_dir
    cfg = {
        "dataset": {"name": DATASET_NAME, "dir": sample_dir,
                    "size": DATASET_SIZE, "type": DATASET_TYPE},
        "QI":            job.qi,
        "data_type":     DATASET_TYPE,
        "sdg_method":    job.sdg_method,
        "sdg_params":    job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": {"enabled": False},
        "attack_params": job.attack_params,
    }
    prepared = _prepare_config(cfg)

    mrf_params = job.attack_params.get("MarginalRF", {})
    wandb_cfg = {
        "sample_idx":       job.sample_idx,
        "dataset":          DATASET_NAME,
        "size":             DATASET_SIZE,
        "sdg_method":       job.sdg_method,
        "sdg_params":       job.sdg_params,
        "attack_method":    job.attack_method,
        "attack_label":     job.attack_label,
        "qi":               job.qi,
        "entropy_weighted": mrf_params.get("entropy_weighted", False),
        "qi_in_graph":      mrf_params.get("qi_in_graph", False),
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=job.run_name,
        config=wandb_cfg,
        reinit=True,
    )
    try:
        train_df, synth_df, qi_features, hidden_features, _ = load_data(prepared)
        recon_df = _run_attack(prepared, synth_df, train_df, qi_features, hidden_features)
        scores_list = _score_reconstruction(train_df, recon_df, hidden_features, DATASET_TYPE)
        metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores_list)}
        ra_mean = round(float(np.mean(scores_list)), 4)
        wandb.log({"RA_mean": ra_mean, **metrics})

        result = {
            "dataset": DATASET_NAME, "size": DATASET_SIZE,
            "sample": f"sample_{job.sample_idx:02d}",
            "sdg": job.sdg_label, "attack": job.attack_method,
            "label": job.attack_label, "qi": job.qi,
            "ra_mean": ra_mean, "error": None,
        }
        result.update(metrics)
        return result

    except Exception as e:
        wandb.log({"error": str(e)})
        return {
            "dataset": DATASET_NAME, "size": DATASET_SIZE,
            "sample": f"sample_{job.sample_idx:02d}",
            "sdg": job.sdg_label, "attack": job.attack_method,
            "label": job.attack_label, "qi": job.qi,
            "ra_mean": None, "error": str(e),
        }
    finally:
        wandb.finish(quiet=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    jobs = generate_jobs()
    print(f"  Total jobs: {len(jobs)}")
    for j in jobs:
        print(f"    {j.run_name}")

    if args.dry_run:
        print("  [dry-run] no jobs executed.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"marginalrf_qi_graph_eps01_{ts}.csv"
    results = []

    def _write_row(row: dict):
        is_new = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    if args.serial:
        for job in jobs:
            result = run_job(job)
            results.append(result)
            _write_row(result)
            ra = result.get("ra_mean")
            err = result.get("error")
            print(f"  [{result['sdg']}] {result['label']:<40s} {result['qi']}  "
                  f"ra={ra:.3f}" if ra else f"  ERROR: {err}")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(run_job, j): j for j in jobs}
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                except Exception as e:
                    j = futures[fut]
                    result = {"sample": f"sample_{j.sample_idx:02d}",
                              "sdg": j.sdg_label, "label": j.attack_label,
                              "qi": j.qi, "ra_mean": None, "error": str(e)}
                results.append(result)
                _write_row(result)
                ra = result.get("ra_mean")
                err = result.get("error")
                print(f"  [{result['sdg']}] {result['label']:<40s} {result['qi']}  "
                      f"ra={ra:.3f}" if ra else f"  ERROR: {err}")

    # ── Summary ───────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(results)
    if "ra_mean" in df.columns and df["ra_mean"].notna().any():
        df_ok = df[df["ra_mean"].notna()]
        print("\n  ── Results Summary ──────────────────────────────────────────")
        pivot = df_ok.groupby(["label", "sdg"])["ra_mean"].mean()
        print(pivot.to_string())
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
