#!/usr/bin/env python
"""
run_mst_epsilon_rf_nb_sweep.py

RF + NaiveBayes on ALL MST epsilons for adult 10k, two QI variants,
all 5 samples, WITH memorization test.

Purpose: quantify the relationship between ε and attack success,
         for two QI settings (QI_large = "strong adversary" and
         QI_behavioral = "different composition, same size").

WandB group: "mst-epsilon-sweep-adult-10k"  (distinct from "main attack sweep 1")

Memorization test: for sample_k, holdout = sample_{(k+1)%5}.

Usage:
    conda activate recon_
    # Foreground (monitor live):
    python experiment_scripts/run_mst_epsilon_rf_nb_sweep.py --serial 2>&1 | tee experiment_scripts/outfiles/mst_eps_sweep.log

    # Background (detached — survives logout):
    nohup conda run -n recon_ python experiment_scripts/run_mst_epsilon_rf_nb_sweep.py \
        --workers 8 \
        >experiment_scripts/outfiles/mst_eps_sweep.log 2>&1 &
    echo "PID=$!"

    # Monitor:
    tail -f experiment_scripts/outfiles/mst_eps_sweep.log

Flags:
    --dry-run    list jobs, no execution
    --serial     run in main thread (Ctrl-C killable)
    --workers N  parallel workers  [default 8]
    --attack NAME  only this attack (RandomForest or NaiveBayes)
    --qi NAME      only this QI variant
    --sample N     only sample index N
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
DATA_ROOT     = Path("/home/golobs/data/reconstruction_data/adult/size_10000")
N_SAMPLES     = 4   # samples 00-03 only; sample_04 is marked NO_HOLDOUT
QI_VARIANTS   = ["QI_large", "QI_behavioral"]

# All 9 MST epsilons
SDG_METHODS   = [("MST", {"epsilon": eps}) for eps in
                 [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]]

ATTACK_CONFIGS = [
    (
        "RandomForest",
        "RandomForest",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "RandomForest": {**ATTACK_PARAM_DEFAULTS.get("RandomForest", {})},
        },
    ),
    (
        "NaiveBayes",
        "NaiveBayes",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "NaiveBayes": {**ATTACK_PARAM_DEFAULTS.get("NaiveBayes", {})},
        },
    ),
]

N_WORKERS     = 8
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "mst-epsilon-sweep-adult-10k"


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
        return str(DATA_ROOT / f"sample_{self.sample_idx:02d}")

    @property
    def holdout_dir(self) -> str:
        return str(DATA_ROOT / f"sample_{(self.sample_idx + 1) % N_SAMPLES:02d}")

    @property
    def run_name(self) -> str:
        return (
            f"{DATASET_NAME}__sz{DATASET_SIZE}__s{self.sample_idx:02d}__"
            f"{self.sdg_label}__{self.attack_label}__{self.qi}"
        )


def generate_jobs(
    attack_filter: str | None = None,
    qi_filter:     str | None = None,
    sample_filter: int | None = None,
) -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), (label, method, params), qi in itertools.product(
        range(N_SAMPLES), SDG_METHODS, ATTACK_CONFIGS, QI_VARIANTS,
    ):
        if attack_filter and label != attack_filter:
            continue
        if qi_filter and qi != qi_filter:
            continue
        if sample_filter is not None and sample_idx != sample_filter:
            continue
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

def _setup_paths():
    for p in [
        "/home/golobs/MIA_on_diffusion/",
        "/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM",
        "/home/golobs/recon-synth",
        "/home/golobs/recon-synth/attacks",
        "/home/golobs/recon-synth/attacks/solvers",
    ]:
        if p not in sys.path:
            sys.path.append(p)
    r = "/home/golobs/Reconstruction"
    if r in sys.path:
        sys.path.remove(r)
    sys.path.insert(0, r)


def run_job(job: Job) -> dict[str, Any]:
    _setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    # ── Build config (memorization test enabled) ──────────────────────────
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
        "memorization_test": {
            "enabled":     True,
            "holdout_dir": job.holdout_dir,
        },
        "attack_params": job.attack_params,
    }
    prepared = _prepare_config(cfg)

    wandb_cfg = {
        "sample_idx":    job.sample_idx,
        "dataset":       DATASET_NAME,
        "size":          DATASET_SIZE,
        "sdg_method":    job.sdg_method,
        "epsilon":       job.sdg_params.get("epsilon"),
        "attack_method": job.attack_method,
        "attack_label":  job.attack_label,
        "qi":            job.qi,
        "sweep_type":    "mst_epsilon_sweep",
    }

    run = wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=job.run_name,
        config=wandb_cfg,
        reinit=True,
    )
    try:
        train_df, synth_df, qi_features, hidden_features, holdout_df = load_data(prepared)

        # ── Training-target attack ─────────────────────────────────────────
        recon_df = _run_attack(prepared, synth_df, train_df,
                               qi_features, hidden_features)
        train_scores = _score_reconstruction(train_df, recon_df, hidden_features, DATASET_TYPE)
        train_metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, train_scores)}
        ra_mean_train = float(np.mean(train_scores))

        # ── Holdout-target attack (memorization test) ──────────────────────
        recon_nt = _run_attack(prepared, synth_df, holdout_df,
                               qi_features, hidden_features)
        nt_scores = _score_reconstruction(holdout_df, recon_nt, hidden_features, DATASET_TYPE)
        nt_metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, nt_scores)}
        ra_mean_nt = float(np.mean(nt_scores))
        delta = ra_mean_train - ra_mean_nt

        log = {
            "RA_mean_train":    ra_mean_train,
            "RA_mean_nontraining": ra_mean_nt,
            "RA_mean_delta":    delta,
            **{f"RA_train_{k}":    v for k, v in train_metrics.items()},
            **{f"RA_nontraining_{k}": v for k, v in nt_metrics.items()},
        }
        wandb.log(log)

        result = {
            "dataset":    DATASET_NAME,
            "size":       DATASET_SIZE,
            "sample":     f"sample_{job.sample_idx:02d}",
            "sdg":        job.sdg_label,
            "epsilon":    job.sdg_params.get("epsilon"),
            "attack":     job.attack_method,
            "label":      job.attack_label,
            "qi":         job.qi,
            "ra_mean_train":    ra_mean_train,
            "ra_mean_nontraining": ra_mean_nt,
            "ra_mean_delta":    delta,
            "error":      None,
        }
        return result

    except Exception as e:
        tb = traceback.format_exc()
        wandb.log({"error": str(e)})
        print(f"  ERROR in {job.run_name}: {e}\n{tb}", flush=True)
        return {
            "dataset": DATASET_NAME, "size": DATASET_SIZE,
            "sample": f"sample_{job.sample_idx:02d}",
            "sdg": job.sdg_label, "epsilon": job.sdg_params.get("epsilon"),
            "attack": job.attack_method, "label": job.attack_label,
            "qi": job.qi,
            "ra_mean_train": None, "ra_mean_nontraining": None,
            "ra_mean_delta": None, "error": str(e),
        }
    finally:
        wandb.finish(quiet=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RF + NaiveBayes sweep over all MST epsilons, adult 10k.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--serial",  action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--attack",  type=str, default=None)
    parser.add_argument("--qi",      type=str, default=None)
    parser.add_argument("--sample",  type=int, default=None)
    args = parser.parse_args()

    jobs = generate_jobs(attack_filter=args.attack,
                         qi_filter=args.qi,
                         sample_filter=args.sample)

    print(f"  WandB group : {WANDB_GROUP}")
    print(f"  Total jobs  : {len(jobs)}")
    print(f"  QI variants : {QI_VARIANTS}")
    print(f"  Epsilons    : {[j.sdg_params['epsilon'] for j in jobs[:9]]}")
    print(f"  Attacks     : {sorted({j.attack_label for j in jobs})}")
    print(f"  Samples     : 5 × (train + holdout memorization test)")

    if args.dry_run:
        for j in jobs:
            print(f"    {j.run_name}")
        print("  [dry-run] no jobs executed.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"mst_epsilon_sweep_{ts}.csv"
    print(f"  Output CSV  : {out_path}\n", flush=True)

    fields = ["dataset","size","sample","sdg","epsilon","attack","label","qi",
              "ra_mean_train","ra_mean_nontraining","ra_mean_delta","error"]
    rows_written = 0

    def _write_row(row: dict):
        nonlocal rows_written
        is_new = rows_written == 0
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow(row)
        rows_written += 1

    if args.serial:
        for job in jobs:
            result = run_job(job)
            _write_row(result)
            ra_t = result.get("ra_mean_train")
            ra_nt = result.get("ra_mean_nontraining")
            d = result.get("ra_mean_delta")
            err = result.get("error")
            if ra_t is not None:
                print(f"  [{result['sdg']:20s}] {result['label']:15s} {result['qi']:15s}  "
                      f"train={ra_t:.3f}  nt={ra_nt:.3f}  Δ={d:.3f}", flush=True)
            else:
                print(f"  [{result['sdg']:20s}] {result['label']:15s}  ERROR: {err}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(run_job, j): j for j in jobs}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    result = fut.result()
                except Exception as e:
                    j = futures[fut]
                    result = {
                        "dataset": DATASET_NAME, "size": DATASET_SIZE,
                        "sample": f"sample_{j.sample_idx:02d}",
                        "sdg": j.sdg_label, "epsilon": j.sdg_params.get("epsilon"),
                        "attack": j.attack_method, "label": j.attack_label,
                        "qi": j.qi,
                        "ra_mean_train": None, "ra_mean_nontraining": None,
                        "ra_mean_delta": None, "error": str(e),
                    }
                _write_row(result)
                ra_t = result.get("ra_mean_train")
                ra_nt = result.get("ra_mean_nontraining")
                d = result.get("ra_mean_delta")
                err = result.get("error")
                if ra_t is not None:
                    print(f"  [{done:3d}/{len(jobs)}] [{result['sdg']:20s}] "
                          f"{result['label']:15s} {result['qi']:15s}  "
                          f"train={ra_t:.3f}  nt={ra_nt:.3f}  Δ={d:.3f}", flush=True)
                else:
                    print(f"  [{done:3d}/{len(jobs)}] [{result['sdg']:20s}] "
                          f"{result['label']:15s}  ERROR: {err}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(out_path)
    df_ok = df[df["ra_mean_train"].notna()]
    if not df_ok.empty:
        print("\n  ── Mean RA_train by (attack, qi, epsilon) ──────────────────")
        pivot = df_ok.groupby(["label","qi","epsilon"])["ra_mean_train"].mean().unstack("epsilon")
        print(pivot.to_string())
        print("\n  ── Mean Δ (train - non-training) ───────────────────────────")
        delta_pivot = df_ok.groupby(["label","qi","epsilon"])["ra_mean_delta"].mean().unstack("epsilon")
        print(delta_pivot.to_string())
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
