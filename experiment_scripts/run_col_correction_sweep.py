#!/usr/bin/env python
"""
Column-correction ablation sweep: MarginalRF with varying col_correction_alpha/mode.

Fixes the row-wise part to MST + global PMI (knn_k=None, graph_type="mst") and
ablates the new column marginal correction parameters:
  - alpha=0.0 (disabled, pure row-BP baseline)
  - alpha=0.5, mode="global"  (default)
  - alpha=1.0, mode="global"  (full correction)
  - alpha=0.5, mode="knn"     (local; knn_k=100 also enables local row-wise PMI)

Runs on adult 1k and adult 10k, all SDG methods, QI1, 5 samples.

NOTE: configs with col_mode="knn" use knn_k=100 to compute KNN indices (required
for local marginals).  This means those configs also use local PMI for row-level BP
(unlike the pure global configs).  Treat them as a combined local-mode baseline.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_col_correction_sweep.py
    python experiment_scripts/run_col_correction_sweep.py --dry-run
    python experiment_scripts/run_col_correction_sweep.py --workers 8
    python experiment_scripts/run_col_correction_sweep.py --dataset adult_1k
    python experiment_scripts/run_col_correction_sweep.py --size 1000
    python experiment_scripts/run_col_correction_sweep.py --sdg Synthpop
    python experiment_scripts/run_col_correction_sweep.py --attack MarginalRF_col_alpha0
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


# ── Dataset configurations ─────────────────────────────────────────────────────

DATASET_CONFIGS = [
    {
        "base":        "adult",
        "name":        "adult",
        "size":        1_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/adult/size_1000",
    },
    {
        "base":        "adult",
        "name":        "adult",
        "size":        10_000,
        "type":        "categorical",
        "qi_variants": ["QI1"],
        "data_root":   "/home/golobs/data/reconstruction_data/adult/size_10000",
    },
]

SAMPLE_RANGE = list(range(5))   # sample_00 through sample_04


# ── SDG methods ────────────────────────────────────────────────────────────────

SDG_METHODS = [
    ("MST",             {"epsilon": 0.1}),
    ("MST",             {"epsilon": 1.0}),
    ("MST",             {"epsilon": 10.0}),
    ("MST",             {"epsilon": 100.0}),
    ("MST",             {"epsilon": 1000.0}),
    ("AIM",             {"epsilon": 1.0}),
    ("AIM",             {"epsilon": 10.0}),
    ("TVAE",            {}),
    ("CTGAN",           {}),
    ("ARF",             {}),
    ("TabDDPM",         {}),
    ("Synthpop",        {}),
    ("RankSwap",        {}),
    ("CellSuppression", {}),
]


# ── Attack configurations ──────────────────────────────────────────────────────
# Each entry: (attack_method, method_specific_params, display_label)
#
# Row-wise params shared by all: graph_type="mst"
# knn_k=None  → global PMI for row-level BP (fast)
# knn_k=100   → local PMI for row-level BP AND enables local marginals for col correction

ATTACK_CONFIGS = [
    # Baseline: no column correction (pure MST + global BP)
    ("MarginalRF", {"knn_k": None, "graph_type": "mst", "col_correction_alpha": 0.0},
     "MarginalRF_col_alpha0"),

    # Global column correction, varying alpha
    ("MarginalRF", {"knn_k": None, "graph_type": "mst",
                    "col_correction_alpha": 0.5, "col_correction_mode": "global"},
     "MarginalRF_col_alpha0.5_global"),

    ("MarginalRF", {"knn_k": None, "graph_type": "mst",
                    "col_correction_alpha": 1.0, "col_correction_mode": "global"},
     "MarginalRF_col_alpha1_global"),

    # KNN (local) column correction — requires knn_k for local marginal computation
    # Note: knn_k=100 also enables local row-wise PMI.
    ("MarginalRF", {"knn_k": 100, "graph_type": "mst",
                    "col_correction_alpha": 0.5, "col_correction_mode": "knn"},
     "MarginalRF_col_alpha0.5_knn"),
]


# ── Misc configuration ─────────────────────────────────────────────────────────

N_WORKERS     = 8
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "col-correction-sweep"


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    dataset_cfg:   dict
    sample_idx:    int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict
    attack_label:  str
    qi:            str

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
            f"{self.dataset_name}__"
            f"sz{self.dataset_cfg['size']}__"
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
        if dataset_filter and ds_cfg["name"] != dataset_filter:
            continue
        if size_filter is not None and ds_cfg["size"] != size_filter:
            continue
        for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params, attack_label), qi in itertools.product(
            SAMPLE_RANGE,
            SDG_METHODS,
            ATTACK_CONFIGS,
            ds_cfg["qi_variants"],
        ):
            if attack_filter and attack_label != attack_filter and attack_method != attack_filter:
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
        tags=[ds["name"], f"size_{ds['size']}", "col-correction", job.effective_label],
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

    print(f"\n  {'Dataset':<8}  {'Size':>6}  {'Label':<32}  {'QI':<6}  {'Mean RA':>10}")
    print(f"  {'-'*70}")
    for (dataset, size, label, qi), vals in sorted(groups.items()):
        mean_val = round(float(np.mean(vals)), 4)
        print(f"  {dataset:<8}  {size:>6}  {label:<32}  {qi:<6}  {mean_val:>10.4f}")
    print(f"{'='*80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Column-correction ablation sweep for MarginalRF.")
    parser.add_argument("--dry-run",      action="store_true", help="Print job list and exit.")
    parser.add_argument("--serial",       action="store_true", help="Run sequentially in the main process.")
    parser.add_argument("--workers",      type=int, default=N_WORKERS, help="Parallel workers.")
    parser.add_argument("--dataset",      type=str, default=None, help="Restrict to dataset name (e.g. 'adult').")
    parser.add_argument("--size",         type=int, default=None, help="Restrict to dataset size (e.g. 1000 or 10000).")
    parser.add_argument("--attack",       type=str, default=None, help="Restrict to this attack label.")
    parser.add_argument("--sdg",          type=str, default=None, help="Restrict to this SDG method/label.")
    parser.add_argument("--sample",       type=int, default=None, help="Restrict to this sample index.")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "col_correction_progress.log"),
                        metavar="FILE",
                        help="Write one progress line per completed job.")
    args = parser.parse_args()

    all_jobs = generate_jobs(
        dataset_filter=args.dataset,
        size_filter=args.size,
        attack_filter=args.attack,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )

    progress_log = open(args.progress_log, "w", buffering=1) if args.progress_log else None

    header = (
        f"{'='*80}\n"
        f"  Column-correction ablation sweep\n"
        f"  Datasets:    adult 1k + adult 10k\n"
        f"  Configs:     {len(ATTACK_CONFIGS)} MarginalRF variants\n"
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
    csv_path = script_dir / f"col_correction_results_{ts}.csv"
    _save_summary_csv(results, csv_path)

    if progress_log:
        progress_log.close()


if __name__ == "__main__":
    main()
