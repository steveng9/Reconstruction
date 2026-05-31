#!/usr/bin/env python
"""
MarginalRF chaining + ensembling sweep.

Compares, on adult 10k (5 samples, QI1, 5 focused SDGs):

  MarginalRF            — baseline (no chaining, no ensembling)
  MarginalRF_HardChain  — hard chaining, mutual-info ordering
  MarginalRF_SoftChain  — soft chaining, mutual-info ordering (novel)
  Ensemble_MRF_LGB      — soft-vote: MarginalRF + LightGBM
  Ensemble_MRF_LGB_NB   — soft-vote: MarginalRF + LightGBM + NaiveBayes
  Ensemble_MRF_5        — soft-vote: MarginalRF + RF + LightGBM + MLP + KNN

SDGs chosen to span a quality range where we already have decisive baseline
numbers: MST_eps1 (low), MST_eps10 (mid), TVAE, Synthpop, TabDDPM (high).

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_marginalrf_chains_ensembles.py
    python experiment_scripts/run_marginalrf_chains_ensembles.py --dry-run
    python experiment_scripts/run_marginalrf_chains_ensembles.py --workers 8
    python experiment_scripts/run_marginalrf_chains_ensembles.py --attack MarginalRF_SoftChain
    python experiment_scripts/run_marginalrf_chains_ensembles.py --sdg TabDDPM
    python experiment_scripts/run_marginalrf_chains_ensembles.py --sample 0
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


# ── Dataset configuration ──────────────────────────────────────────────────────

DATASET_NAME  = "adult"
DATASET_SIZE  = 10_000
DATASET_TYPE  = "categorical"
DATA_ROOT     = f"/home/golobs/data/reconstruction_data/adult/size_{DATASET_SIZE}"
QI_VARIANTS   = ["QI1"]
SAMPLE_RANGE  = list(range(5))    # sample_00 through sample_04


# ── SDG methods (quality-range coverage) ──────────────────────────────────────

SDG_METHODS = [
    ("RankSwap",        {}),
    ("CellSuppression", {}),
    ("Synthpop",        {}),
    ("MST",      {"epsilon": 0.1}),
    ("MST",      {"epsilon": 1.0}),
    ("MST",      {"epsilon": 10.0}),
    ("MST",      {"epsilon": 100.0}),
    ("MST",      {"epsilon": 1000.0}),
    ("AIM",      {"epsilon": 1.0}),
    ("TVAE",     {}),
    ("CTGAN",    {}),
    ("ARF",      {}),
    ("TabDDPM",  {}),
]


# ── Attack configurations ──────────────────────────────────────────────────────
# Each entry: (label, attack_method, attack_params_overrides)
# attack_params_overrides contains chaining/ensembling sub-dicts plus any
# method-specific entries needed so _prepare_config finds them.

_MRF   = ATTACK_PARAM_DEFAULTS["MarginalRF"]
_RF    = ATTACK_PARAM_DEFAULTS["RandomForest"]
_LGB   = ATTACK_PARAM_DEFAULTS["LightGBM"]
_MLP   = ATTACK_PARAM_DEFAULTS["MLP"]
_KNN   = ATTACK_PARAM_DEFAULTS["KNN"]
_NB    = ATTACK_PARAM_DEFAULTS.get("NaiveBayes", {})

ATTACK_CONFIGS = [
    # ── Baseline (no chaining, no ensembling) ─────────────────────────────
    (
        "MarginalRF",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": dict(_MRF),
        },
    ),

    # ── Hard chaining (mutual-info ordering) ──────────────────────────────
    # At each step MarginalRF predicts one feature using QI + previously
    # hard-predicted features.  With a single remaining hidden feature the BP
    # has no pairs, so this is effectively RF hard-chaining.
    (
        "MarginalRF_HardChain",
        "MarginalRF",
        {
            "chaining": {
                "enabled":        True,
                "mode":           "hard",
                "order_strategy": "mutual_info",
                "log_intermediate": False,
            },
            "ensembling": {"enabled": False},
            "MarginalRF": dict(_MRF),
        },
    ),

    # ── Soft chaining (mutual-info ordering) ──────────────────────────────
    # At each step MarginalRF runs on ALL remaining features with an augmented
    # QI that appends predicted-proba vectors from earlier steps (one-hot on
    # synth, soft probas on targets).  BP still runs over remaining features.
    (
        "MarginalRF_SoftChain",
        "MarginalRF",
        {
            "chaining": {
                "enabled":        True,
                "mode":           "soft",
                "order_strategy": "mutual_info",
                "log_intermediate": False,
            },
            "ensembling": {"enabled": False},
            "MarginalRF": dict(_MRF),
        },
    ),

    # ── Ensemble: MarginalRF + LightGBM ───────────────────────────────────
    # Using hard_voting: LGB's predict_proba outputs max=1.0 for all rows
    # (extreme overconfidence), which would dominate soft voting even when
    # wrong.  Hard voting treats both models symmetrically.
    (
        "Ensemble_MRF_LGB",
        "MarginalRF",
        {
            "chaining": {"enabled": False},
            "ensembling": {
                "enabled":         True,
                "include_primary": True,
                "methods":         ["LightGBM"],
                "aggregation":     "hard_voting",
            },
            "MarginalRF": dict(_MRF),
            "LightGBM":   dict(_LGB),
        },
    ),

    # ── Ensemble: MarginalRF + LightGBM + NaiveBayes ─────────────────────
    (
        "Ensemble_MRF_LGB_NB",
        "MarginalRF",
        {
            "chaining": {"enabled": False},
            "ensembling": {
                "enabled":         True,
                "include_primary": True,
                "methods":         ["LightGBM", "NaiveBayes"],
                "aggregation":     "hard_voting",
            },
            "MarginalRF": dict(_MRF),
            "LightGBM":   dict(_LGB),
            "NaiveBayes": dict(_NB),
        },
    ),

    # ── Ensemble: MarginalRF + RF + LightGBM + MLP + KNN ─────────────────
    (
        "Ensemble_MRF_5",
        "MarginalRF",
        {
            "chaining": {"enabled": False},
            "ensembling": {
                "enabled":         True,
                "include_primary": True,
                "methods":         ["RandomForest", "LightGBM", "MLP", "KNN"],
                "aggregation":     "hard_voting",
            },
            "MarginalRF":   dict(_MRF),
            "RandomForest": dict(_RF),
            "LightGBM":     dict(_LGB),
            "MLP":          dict(_MLP),
            "KNN":          dict(_KNN),
        },
    ),
]

# Table fill-in (adult 10k, all 13 SDG columns): keep only the baseline plus the
# best chain (hard, mutual-info) and best ensemble (MRF+RF+LGB+MLP+KNN) configs.
_KEEP_LABELS = {"MarginalRF", "MarginalRF_HardChain", "Ensemble_MRF_5"}
ATTACK_CONFIGS = [c for c in ATTACK_CONFIGS if c[0] in _KEEP_LABELS]


# ── Misc ───────────────────────────────────────────────────────────────────────

N_WORKERS     = 4
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "marginalrf-chaining-ensembling"


# ── Job specification ──────────────────────────────────────────────────────────

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
            f"{DATASET_NAME}__"
            f"sz{DATASET_SIZE}__"
            f"s{self.sample_idx:02d}__"
            f"{self.sdg_label}__"
            f"{self.attack_label}__"
            f"{self.qi}"
        )


def generate_jobs(
    attack_filter: str | None = None,
    sdg_filter:    str | None = None,
    sample_filter: int | None = None,
) -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), (label, method, params), qi in itertools.product(
        SAMPLE_RANGE, SDG_METHODS, ATTACK_CONFIGS, QI_VARIANTS,
    ):
        if attack_filter and label != attack_filter:
            continue
        sdg_label = f"{sdg_method}_eps{sdg_params['epsilon']:g}" if sdg_params.get("epsilon") is not None else sdg_method
        if sdg_filter and sdg_label != sdg_filter and sdg_method != sdg_filter:
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
        "attack_params": job.attack_params,
    }

    prepared = _prepare_config(cfg)

    # WandB config: flatten attack_params to scalars/strings for readability
    ensembling_cfg = job.attack_params.get("ensembling", {})
    chaining_cfg   = job.attack_params.get("chaining", {})
    wandb_cfg = {
        "sample_idx":    job.sample_idx,
        "dataset":       DATASET_NAME,
        "size":          DATASET_SIZE,
        "sdg_method":    job.sdg_method,
        "sdg_params":    job.sdg_params,
        "attack_method": job.attack_method,
        "attack_label":  job.attack_label,
        "qi":            job.qi,
        "chaining_enabled":    chaining_cfg.get("enabled", False),
        "chaining_mode":       chaining_cfg.get("mode", "hard"),
        "chaining_order":      chaining_cfg.get("order_strategy", ""),
        "ensembling_enabled":  ensembling_cfg.get("enabled", False),
        "ensembling_methods":  ensembling_cfg.get("methods", []),
        "ensembling_agg":      ensembling_cfg.get("aggregation", ""),
    }

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config=wandb_cfg,
        tags=[DATASET_NAME, f"size_{DATASET_SIZE}", "chaining-ensembling", job.attack_label],
        group=WANDB_GROUP,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, _holdout = load_data(prepared)
        recon  = _run_attack(prepared, synth, train, qi, hidden_features)
        scores = _score_reconstruction(train, recon, hidden_features, DATASET_TYPE)

        metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
        ra_mean = round(float(np.mean(scores)), 4)
        metrics["RA_mean"] = ra_mean
        wandb.log(metrics)

        feat_scores = {k: v for k, v in metrics.items() if k.startswith("RA_") and k != "RA_mean"}
        return {
            "dataset": DATASET_NAME,
            "size":    DATASET_SIZE,
            "sample":  job.sample_idx,
            "sdg":     job.sdg_label,
            "attack":  job.attack_method,
            "label":   job.attack_label,
            "qi":      job.qi,
            "ra_mean": ra_mean,
            "error":   None,
            **feat_scores,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {job.run_name}:\n{tb}", flush=True)
        wandb.log({"error": str(exc)})
        raise

    finally:
        wandb.finish()


# ── Summary helpers ─────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    base_keys = ["dataset", "size", "sample", "sdg", "attack", "label", "qi", "ra_mean", "error"]
    feat_keys = sorted({k for r in rows for k in r if k.startswith("RA_") and k != "RA_mean"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_keys + feat_keys,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")


def _print_summary(rows: list[dict]):
    import numpy as np
    from collections import defaultdict

    successes = [r for r in rows if not r.get("error")]
    failures  = [r for r in rows if r.get("error")]

    print(f"\n{'='*80}")
    print(f"  {WANDB_GROUP}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*80}")

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["label"], r["sdg"])
        if r["ra_mean"] is not None:
            groups[key].append(r["ra_mean"])

    # Print table sorted by label then sdg
    print(f"\n  {'Label':<30}  {'SDG':<20}  {'N':>3}  {'Mean RA':>8}")
    print(f"  {'-'*68}")
    for (label, sdg), vals in sorted(groups.items()):
        print(f"  {label:<30}  {sdg:<20}  {len(vals):>3}  {float(np.mean(vals)):>8.4f}")
    print(f"{'='*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MarginalRF chaining + ensembling sweep.")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--serial",       action="store_true",
                        help="Run in main process (Ctrl-C killable).")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--attack",       type=str, default=None,
                        help="Restrict to one attack label (e.g. MarginalRF_SoftChain).")
    parser.add_argument("--sdg",          type=str, default=None)
    parser.add_argument("--sample",       type=int, default=None)
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" /
                                    "marginalrf_chains_ensembles.log"))
    args = parser.parse_args()

    all_jobs = generate_jobs(
        attack_filter=args.attack,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )

    header = (
        f"{'='*80}\n"
        f"  MarginalRF chaining + ensembling sweep\n"
        f"  Dataset:  {DATASET_NAME} {DATASET_SIZE}\n"
        f"  Configs:  {len(ATTACK_CONFIGS)} attack variants\n"
        f"  SDGs:     {len(SDG_METHODS)}\n"
        f"  Samples:  {len(SAMPLE_RANGE)}\n"
        f"  Jobs:     {len(all_jobs)}\n"
        f"  Workers:  {args.workers}\n"
        f"  WandB:    {WANDB_GROUP}\n"
        f"{'='*80}\n"
    )
    print(header, end="")

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    # Validate that sample dirs exist
    missing = []
    for job in all_jobs:
        p = Path(job.sample_dir)
        if not p.exists():
            missing.append(job.sample_dir)
    if missing := list(dict.fromkeys(missing)):
        print("ERROR: missing sample directories:")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    # Validate that synth.csv files exist
    missing_synth = []
    for job in all_jobs:
        synth_path = Path(job.sample_dir) / job.sdg_label / "synth.csv"
        if not synth_path.exists():
            missing_synth.append(str(synth_path))
    if missing_synth:
        missing_synth = list(dict.fromkeys(missing_synth))
        print(f"WARNING: {len(missing_synth)} synth.csv files missing (jobs will fail):")
        for p in missing_synth[:10]:
            print(f"  {p}")
        if len(missing_synth) > 10:
            print(f"  ... and {len(missing_synth) - 10} more")

    progress_log = open(args.progress_log, "w", buffering=1)
    progress_log.write(header)

    start_time = time.time()
    results: list[dict] = []
    n_done = n_fail = 0
    width = len(str(len(all_jobs)))

    def _handle(job, result_or_exc):
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
            results.append({
                "dataset": DATASET_NAME, "size": DATASET_SIZE,
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "label": job.attack_label,
                "qi": job.qi, "ra_mean": None, "error": str(result_or_exc),
            })
        else:
            val_str = f"{result_or_exc['ra_mean']:.4f}" if result_or_exc["ra_mean"] is not None else "N/A"
            line = (
                f"  [{n_done:>{width}}/{len(all_jobs)}]"
                f"  {job.run_name:<80}  RA={val_str}{eta_str}"
            )
            results.append(result_or_exc)

        print(line, flush=True)
        progress_log.write(line + "\n")

    if args.serial:
        _worker_setup_paths()
        sys.argv = sys.argv[:1]
        for job in all_jobs:
            try:
                _handle(job, run_job(job))
            except Exception as exc:
                _handle(job, exc)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_job, job): job for job in all_jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    _handle(job, future.result())
                except Exception as exc:
                    _handle(job, exc)

    total_min = (time.time() - start_time) / 60
    print(f"\nAll {len(all_jobs)} jobs finished in {total_min:.1f} min.")

    _print_summary(results)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"marginalrf_chains_ensembles_{ts}.csv"
    _save_csv(results, csv_path)

    progress_log.close()


if __name__ == "__main__":
    main()
