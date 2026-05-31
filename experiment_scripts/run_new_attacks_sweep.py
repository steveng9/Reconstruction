#!/usr/bin/env python
"""
MarginalRF sweep across all datasets and sizes.

Runs MarginalRF (default params: knn_k=100, graph_type="mst") on every dataset
and sample size where synth data is available, 5 samples each.  Baseline RF
results are assumed to already exist in WandB from earlier sweeps.

Datasets covered
----------------
  adult           10k   QI1   (9 MST + 3 AIM + 7 other SDGs)
  cdc_diabetes     1k   QI1   (5 MST + 3 AIM + 7 other SDGs)
  cdc_diabetes   100k   QI1   (5 MST + 1 AIM + 7 other SDGs)
  nist_sbo         1k   QI1   (5 MST        + 7 other SDGs — no AIM)
  nist_arizona   10k/25feat QI1 (5 MST + 2 AIM + 7 other SDGs)
  adult           20k   QI1   (9 MST + 4 AIM + 7 other SDGs) — submitted last

california is excluded (MarginalRF is categorical-only; california is continuous).

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_new_attacks_sweep.py
    python experiment_scripts/run_new_attacks_sweep.py --dry-run
    python experiment_scripts/run_new_attacks_sweep.py --workers 8
    python experiment_scripts/run_new_attacks_sweep.py --dataset adult
    python experiment_scripts/run_new_attacks_sweep.py --sdg Synthpop
    python experiment_scripts/run_new_attacks_sweep.py --sample 0
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Dataset configurations ─────────────────────────────────────────────────────
# Each dataset specifies its own sdg_methods — available SDGs differ by dataset
# and size.  Only methods with synth.csv present in sample_00 are listed.
# adult 20k is last so it doesn't starve earlier (faster) datasets of workers.

DATASET_CONFIGS = [
    {
        "base":        "adult",
        "name":        "adult",
        "size":        10_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/adult/size_10000",
        "sdg_methods": [
            ("MST",             {"epsilon": 0.1}),
            ("MST",             {"epsilon": 0.3}),
            ("MST",             {"epsilon": 1.0}),
            ("MST",             {"epsilon": 3.0}),
            ("MST",             {"epsilon": 10.0}),
            ("MST",             {"epsilon": 30.0}),
            ("MST",             {"epsilon": 100.0}),
            ("MST",             {"epsilon": 300.0}),
            ("MST",             {"epsilon": 1000.0}),
            ("AIM",             {"epsilon": 0.3}),
            ("AIM",             {"epsilon": 1.0}),
            ("AIM",             {"epsilon": 3.0}),
            # AIM eps=10 has no synth.csv for size_10000
            ("TVAE",            {}),
            ("CTGAN",           {}),
            ("ARF",             {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("RankSwap",        {}),
            ("CellSuppression", {}),
        ],
    },
    {
        "base":        "cdc_diabetes",
        "name":        "cdc_diabetes",
        "size":        1_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/cdc_diabetes/size_1000",
        "sdg_methods": [
            ("MST",             {"epsilon": 0.1}),
            ("MST",             {"epsilon": 1.0}),
            ("MST",             {"epsilon": 10.0}),
            ("MST",             {"epsilon": 100.0}),
            ("MST",             {"epsilon": 1000.0}),
            ("AIM",             {"epsilon": 1.0}),
            ("AIM",             {"epsilon": 3.0}),
            ("AIM",             {"epsilon": 10.0}),
            ("TVAE",            {}),
            ("CTGAN",           {}),
            ("ARF",             {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("RankSwap",        {}),
            ("CellSuppression", {}),
        ],
    },
    {
        "base":        "cdc_diabetes",
        "name":        "cdc_diabetes",
        "size":        100_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/cdc_diabetes/size_100000",
        "sdg_methods": [
            ("MST",             {"epsilon": 0.1}),
            ("MST",             {"epsilon": 1.0}),
            ("MST",             {"epsilon": 10.0}),
            ("MST",             {"epsilon": 100.0}),
            ("MST",             {"epsilon": 1000.0}),
            ("AIM",             {"epsilon": 1.0}),
            # AIM eps=10 has no synth.csv for size_100000
            ("TVAE",            {}),
            ("CTGAN",           {}),
            ("ARF",             {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("RankSwap",        {}),
            ("CellSuppression", {}),
        ],
    },
    {
        "base":        "nist_sbo",
        "name":        "nist_sbo",
        "size":        1_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/nist_sbo/size_1000",
        "sdg_methods": [
            ("MST",             {"epsilon": 0.1}),
            ("MST",             {"epsilon": 1.0}),
            ("MST",             {"epsilon": 10.0}),
            ("MST",             {"epsilon": 100.0}),
            ("MST",             {"epsilon": 1000.0}),
            # No AIM data generated for nist_sbo
            ("TVAE",            {}),
            ("CTGAN",           {}),
            ("ARF",             {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("RankSwap",        {}),
            ("CellSuppression", {}),
        ],
    },
    {
        "base":        "nist_arizona_data",
        "name":        "nist_arizona_25feat",
        "size":        10_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/nist_arizona_data/size_10000_25feat",
        "n_features":  25,
        "sdg_methods": [
            ("MST",             {"epsilon": 0.1}),
            ("MST",             {"epsilon": 1.0}),
            ("MST",             {"epsilon": 10.0}),
            ("MST",             {"epsilon": 100.0}),
            ("MST",             {"epsilon": 1000.0}),
            ("AIM",             {"epsilon": 1.0}),
            ("AIM",             {"epsilon": 3.0}),
            ("TVAE",            {}),
            ("CTGAN",           {}),
            ("ARF",             {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("RankSwap",        {}),
            ("CellSuppression", {}),
        ],
    },
    # adult 20k is last — more samples keep workers busy longer; run after smaller datasets
    {
        "base":        "adult",
        "name":        "adult",
        "size":        20_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/adult/size_20000",
        "sdg_methods": [
            ("MST",             {"epsilon": 0.1}),
            ("MST",             {"epsilon": 0.3}),
            ("MST",             {"epsilon": 1.0}),
            ("MST",             {"epsilon": 3.0}),
            ("MST",             {"epsilon": 10.0}),
            ("MST",             {"epsilon": 30.0}),
            ("MST",             {"epsilon": 100.0}),
            ("MST",             {"epsilon": 300.0}),
            ("MST",             {"epsilon": 1000.0}),
            ("AIM",             {"epsilon": 0.3}),
            ("AIM",             {"epsilon": 1.0}),
            ("AIM",             {"epsilon": 3.0}),
            ("AIM",             {"epsilon": 10.0}),
            ("TVAE",            {}),
            ("CTGAN",           {}),
            ("ARF",             {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("RankSwap",        {}),
            ("CellSuppression", {}),
        ],
    },
]

SAMPLE_RANGE = list(range(5))   # sample_00 through sample_04


# ── Attack configurations ──────────────────────────────────────────────────────
# Each entry: (attack_method, method_specific_params, display_label)
# display_label: leave "" to use attack_method directly.
#
# Using default params (knn_k=100, graph_type="mst") — validated as the best
# MarginalRF configuration from the adult 1k ablation.

ATTACK_CONFIGS = [
    ("MarginalRF", {}, ""),
]


# ── Misc configuration ─────────────────────────────────────────────────────────

N_WORKERS     = 8
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "marginalrf-all-datasets"


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    dataset_cfg:    dict
    sample_idx:     int
    sdg_method:     str
    sdg_params:     dict
    attack_method:  str
    attack_params:  dict
    attack_label:   str
    qi:             str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def effective_label(self) -> str:
        return self.attack_label or self.attack_method

    @property
    def dataset_name(self) -> str:
        return self.dataset_cfg["name"]

    @property
    def sample_dir(self) -> str:
        return f"{self.dataset_cfg['data_root']}/sample_{self.sample_idx:02d}"

    @property
    def run_name(self) -> str:
        return (
            f"{self.dataset_name}_sz{self.dataset_cfg['size']}__"
            f"s{self.sample_idx:02d}__"
            f"{self.sdg_label}__"
            f"{self.effective_label}__"
            f"{self.qi}"
        )


def generate_jobs(
    dataset_filter: str | None = None,
    size_filter:    int | None = None,
    attack_filter:  str | None = None,
    sdg_filter:     str | None = None,
    sample_filter:  int | None = None,
) -> list[Job]:
    jobs = []
    for ds_cfg in DATASET_CONFIGS:
        if dataset_filter and ds_cfg["name"] != dataset_filter and ds_cfg["base"] != dataset_filter:
            continue
        if size_filter is not None and ds_cfg["size"] != size_filter:
            continue
        sdg_methods = ds_cfg.get("sdg_methods")
        for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params, attack_label), qi in itertools.product(
            SAMPLE_RANGE,
            sdg_methods,
            ATTACK_CONFIGS,
            ds_cfg["qi_variants"],
        ):
            if attack_filter and attack_method != attack_filter and (attack_label or attack_method) != attack_filter:
                continue
            sdg_label = f"{sdg_method}_eps{sdg_params['epsilon']:g}" if sdg_params.get("epsilon") is not None else sdg_method
            if sdg_filter and sdg_label != sdg_filter and sdg_method != sdg_filter:
                continue
            if sample_filter is not None and sample_idx != sample_filter:
                continue
            jobs.append(Job(
                dataset_cfg=ds_cfg,
                sample_idx=sample_idx,
                sdg_method=sdg_method,
                sdg_params=dict(sdg_params),
                attack_method=attack_method,
                attack_params=dict(attack_params),
                attack_label=attack_label,
                qi=qi,
            ))
    return jobs


# ── Worker function ────────────────────────────────────────────────────────────

def _worker_setup_paths():
    """Configure sys.path so that local attacks/ wins over recon-synth's."""
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
    """Execute one experiment job in a worker process."""
    _worker_setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    ds         = job.dataset_cfg
    sample_dir = job.sample_dir

    cfg = {
        "dataset": {
            "name": ds["name"],
            "dir":  sample_dir,
            "size": ds["size"],
            "type": ds["type"],
        },
        "QI":          job.qi,
        "data_type":   ds["type"],
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
            "dataset":       ds["name"],
            "size":          ds["size"],
            "sdg_method":    job.sdg_method,
            "sdg_params":    job.sdg_params,
            "attack_method": job.attack_method,
            "attack_label":  job.effective_label,
            "attack_params": effective_attack_params,
            "qi":            job.qi,
        },
        tags=[ds["name"], f"size_{ds['size']}", "marginalrf", job.effective_label],
        group=WANDB_GROUP,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)
        recon  = _run_attack(prepared, synth, train, qi, hidden_features)
        scores = _score_reconstruction(train, recon, hidden_features, ds["type"])

        metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
        ra_mean = round(float(np.mean(scores)), 4)
        metrics["RA_mean"] = ra_mean
        wandb.log(metrics)

        feat_scores = {k: v for k, v in metrics.items() if k.startswith("RA_") and k != "RA_mean"}
        return {
            "dataset": ds["name"],
            "size":    ds["size"],
            "sample":  job.sample_idx,
            "sdg":     job.sdg_label,
            "attack":  job.attack_method,
            "label":   job.effective_label,
            "qi":      job.qi,
            "ra_mean": ra_mean,
            "error":   None,
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
    base_keys = ["dataset", "size", "sample", "sdg", "attack", "label", "qi", "ra_mean", "error"]
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

    print(f"\n{'='*80}")
    print(f"  SWEEP COMPLETE — {WANDB_GROUP}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*80}")

    if not successes:
        return

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["dataset"], r["size"], r.get("label") or r["attack"], r["qi"])
        if r["ra_mean"] is not None:
            groups[key].append(r["ra_mean"])

    print(f"\n  {'Dataset':<20}  {'Size':>7}  {'Attack':<14}  {'QI':<8}  {'Mean RA':>10}")
    print(f"  {'-'*66}")
    for (dataset, size, label, qi), vals in sorted(groups.items()):
        mean_val = round(float(np.mean(vals)), 4)
        print(f"  {dataset:<20}  {size:>7}  {label:<14}  {qi:<8}  {mean_val:>10.4f}")
    print(f"{'='*80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MarginalRF sweep across all datasets.")
    parser.add_argument("--dry-run",    action="store_true", help="Print job list and exit.")
    parser.add_argument("--serial",     action="store_true", help="Run sequentially in the main process.")
    parser.add_argument("--workers",    type=int,  default=N_WORKERS,  help="Parallel workers.")
    parser.add_argument("--dataset",    type=str,  default=None, help="Restrict to this dataset name or base (e.g. 'adult', 'cdc_diabetes').")
    parser.add_argument("--size",       type=int,  default=None, help="Restrict to this dataset size (e.g. 20000).")
    parser.add_argument("--sdg",        type=str,  default=None, help="Restrict to this SDG method/label.")
    parser.add_argument("--sample",     type=int,  default=None, help="Restrict to this sample index.")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "marginalrf_progress.log"),
                        metavar="FILE",
                        help="Write one progress line per completed job (tail -f to monitor).")
    args = parser.parse_args()

    all_jobs = generate_jobs(
        dataset_filter=args.dataset,
        size_filter=args.size,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )

    progress_log = open(args.progress_log, "w", buffering=1) if args.progress_log else None

    dataset_summary = ", ".join(
        f"{d['name']} {d['size']}" for d in DATASET_CONFIGS
    )
    header = (
        f"{'='*80}\n"
        f"  MarginalRF all-datasets sweep\n"
        f"  Datasets:    {dataset_summary}\n"
        f"  Jobs total:  {len(all_jobs)}\n"
        f"  Workers:     {args.workers}\n"
        f"  WandB group: {WANDB_GROUP}\n"
        f"{'='*80}\n"
    )
    print(header, end="")
    if progress_log:
        progress_log.write(header)

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    missing = []
    for job in all_jobs:
        p = Path(job.sample_dir)
        if not p.exists():
            missing.append(job.sample_dir)
    missing = list(dict.fromkeys(missing))
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
                "dataset": job.dataset_name,
                "size": job.dataset_cfg["size"],
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "label": job.effective_label, "qi": job.qi,
                "ra_mean": None, "error": str(result_or_exc),
            })
        else:
            val     = result_or_exc["ra_mean"]
            val_str = f"{val:.4f}" if val is not None else "N/A"
            line = (
                f"  [{n_done:>{width}}/{len(all_jobs)}]"
                f"  {job.run_name:<80}  RA={val_str}{eta_str}"
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
    csv_path = script_dir / f"marginalrf_results_{ts}.csv"
    _save_summary_csv(results, csv_path)

    if progress_log:
        progress_log.close()


if __name__ == "__main__":
    main()
