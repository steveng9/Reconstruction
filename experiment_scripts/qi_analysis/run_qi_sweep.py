#!/usr/bin/env python
"""
QI-analysis sweep: investigates how adversary prior knowledge (quasi-identifier set)
shapes reconstruction risk. Section 4.3 of the paper.

Holds synthetic data fixed — only the attack-time QI definition changes.
No new SDG generation is required; all synth.csv files are pre-existing.

Sweep priority order (launch one at a time via the launch_*.sh scripts):
  1. adult QI_tiny        (10k) — bare demographic identity, 3 known features
  2. adult QI_large       (10k) — demographics + employment, 10 known features
  3. adult QI_behavioral  (10k) — employment-only prior, matched size to QI1
  4. cdc_diabetes QI_tiny        (1k) — minimal demographic, 4 known features
  5. cdc_diabetes QI_large       (1k) — demographic + health + lifestyle, 16 known features
  6. cdc_diabetes QI_behavioral  (1k) — lifestyle/behavior prior, matched size to QI1

Attack battery (reduced from production sweep — enough to characterise QI effects):
  Mode, Random, KNN, NaiveBayes, RandomForest, LightGBM, MLP

SDG methods are the full set actually on disk for each dataset (verified at time of writing):
  adult 10k   : 9 MST × ε + 4 AIM × ε + ARF/CTGAN/TVAE/TabDDPM/Synthpop/CellSuppression/RankSwap
  cdc_diabetes: 5 MST × ε + 3 AIM × ε + ARF/CTGAN/TVAE/TabDDPM/Synthpop/CellSuppression/RankSwap

Usage (from repo root, recon_ env active):
    python experiment_scripts/qi_analysis/run_qi_sweep.py --dataset adult --qi QI_tiny
    python experiment_scripts/qi_analysis/run_qi_sweep.py --dataset cdc_diabetes --qi QI_large
    python experiment_scripts/qi_analysis/run_qi_sweep.py --dry-run --dataset adult --qi QI_behavioral
    python experiment_scripts/qi_analysis/run_qi_sweep.py --serial --sample 0 --dataset adult --qi QI_tiny

CLI args:
    --dataset    adult | cdc_diabetes                        (required)
    --qi         QI_tiny | QI_large | QI_behavioral          (required)
    --workers    parallel worker count (default 8)
    --dry-run    print job list and exit without running
    --serial     run sequentially in the main process (useful for debugging)
    --sample N   restrict to a single sample index
    --sdg NAME   restrict to jobs matching this SDG label or method name
    --attack NAME  restrict to jobs for this attack method
    --progress-log FILE  write one progress line per completed job to FILE
                         (default: <repo_root>/outfiles/qi_<dataset>_<qi>.log)
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

# Ensure Reconstruction is always at the front of sys.path when this module is
# imported directly (e.g. from --serial mode without subprocess isolation).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Per-dataset configuration ──────────────────────────────────────────────────
#
# sdg_methods lists exactly the SDG subdirs that exist on disk (verified against
# /home/golobs/data/reconstruction_data/{dataset}/size_{N}/sample_0{0..4}/).
# If the data layout ever changes, update these lists to match.

DATASET_CONFIGS: dict[str, dict] = {
    "adult": {
        "data_root":   "/home/golobs/data/reconstruction_data/adult/size_10000",
        "dataset_size": 10_000,
        "data_type":   "categorical",
        "sample_range": list(reversed(range(5))),   # sample_04 → sample_00
        "valid_qi_variants": ["QI_tiny", "QI_large", "QI_behavioral"],
        "sdg_methods": [
            # ── Differentially private ─────────────────────────────────────
            ("MST", {"epsilon": 0.1}),
            #("MST", {"epsilon": 0.3}),
            ("MST", {"epsilon": 1.0}),
            #("MST", {"epsilon": 3.0}),
            ("MST", {"epsilon": 10.0}),
            #("MST", {"epsilon": 30.0}),
            ("MST", {"epsilon": 100.0}),
            #("MST", {"epsilon": 300.0}),
            ("MST", {"epsilon": 1000.0}),
            #("AIM", {"epsilon": 0.3}),
            ("AIM", {"epsilon": 1.0}),
            #("AIM", {"epsilon": 3.0}),
            ("AIM", {"epsilon": 10.0}),
            # ── Non-DP ────────────────────────────────────────────────────
            ("ARF",             {}),
            ("CTGAN",           {}),
            ("TVAE",            {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("CellSuppression", {}),
            ("RankSwap",        {}),
        ],
    },
    "cdc_diabetes": {
        "data_root":   "/home/golobs/data/reconstruction_data/cdc_diabetes/size_1000",
        "dataset_size": 1_000,
        "data_type":   "categorical",
        "sample_range": list(reversed(range(5))),   # sample_04 → sample_00
        "valid_qi_variants": ["QI_tiny", "QI_large", "QI_behavioral"],
        "sdg_methods": [
            # ── Differentially private ─────────────────────────────────────
            # NOTE: MST eps 0.3/3/30/300 not generated for cdc_diabetes 1k
            ("MST", {"epsilon": 0.1}),
            ("MST", {"epsilon": 1.0}),
            ("MST", {"epsilon": 10.0}),
            ("MST", {"epsilon": 100.0}),
            ("MST", {"epsilon": 1000.0}),
            # NOTE: AIM eps 0.3 not generated for cdc_diabetes 1k
            ("AIM", {"epsilon": 1.0}),
            ("AIM", {"epsilon": 3.0}),
            ("AIM", {"epsilon": 10.0}),
            # ── Non-DP ────────────────────────────────────────────────────
            ("ARF",             {}),
            ("CTGAN",           {}),
            ("TVAE",            {}),
            ("TabDDPM",         {}),
            ("Synthpop",        {}),
            ("CellSuppression", {}),
            ("RankSwap",        {}),
        ],
    },
}

# ── Attack battery for QI analysis ────────────────────────────────────────────
# Reduced relative to the full production sweep: covers baselines + the main
# ML attack tier. Diffusion/LP attacks omitted — too slow for a 6-sweep QI grid
# and already characterised in the main QI1 sweep.
ATTACK_CONFIGS = [
    # Baselines
    ("Mode",             {}),
    ("Random",           {}),
    # ML classifiers
    ("KNN",              {}),
    ("NaiveBayes",       {}),
    ("RandomForest",     {}),
    ("LightGBM",         {}),
    # Neural
    ("MLP",              {}),
]

# Default parallelism — fast attacks, so 8 workers is fine on this machine.
N_WORKERS_DEFAULT = 8

WANDB_PROJECT = "tabular-reconstruction-attacks"


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx:    int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict
    qi:            str
    dataset_name:  str
    data_root:     str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{self.data_root}/sample_{self.sample_idx:02d}"

    @property
    def run_name(self) -> str:
        return f"s{self.sample_idx:02d}__{self.sdg_label}__{self.attack_method}__{self.qi}"


def generate_jobs(dataset_name: str, qi_variant: str) -> list[Job]:
    cfg = DATASET_CONFIGS[dataset_name]
    jobs = []
    for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params) in itertools.product(
        cfg["sample_range"],
        cfg["sdg_methods"],
        ATTACK_CONFIGS,
    ):
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=dict(sdg_params),
            attack_method=attack_method,
            attack_params=dict(attack_params),
            qi=qi_variant,
            dataset_name=dataset_name,
            data_root=cfg["data_root"],
        ))
    return jobs


# ── Worker function (runs in a subprocess) ────────────────────────────────────

def _worker_setup_paths():
    """Configure sys.path once per worker process.

    recon-synth defines its own 'attacks' package that must NOT shadow the local
    Reconstruction/attacks package. Strategy: append recon-synth paths, then
    force-insert Reconstruction at index 0 so its packages win on all imports.
    """
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
    """Execute one experiment job in a worker subprocess.

    Returns a result dict on success; raises on fatal error (the caller logs
    the exception and continues to the next job).
    """
    _worker_setup_paths()

    # master_experiment_script calls parse_args() at module level — clear
    # sys.argv so its parser doesn't choke on our own CLI arguments.
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    dataset_cfg = DATASET_CONFIGS[job.dataset_name]
    dataset_size = dataset_cfg["dataset_size"]
    data_type    = dataset_cfg["data_type"]

    cfg = {
        "dataset": {
            "name": job.dataset_name,
            "dir":  job.sample_dir,
            "size": dataset_size,
            "type": data_type,
        },
        "QI":           job.qi,
        "data_type":    data_type,
        "sdg_method":   job.sdg_method,
        "sdg_params":   job.sdg_params or None,
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

    wandb_group = f"qi-analysis-{job.dataset_name}-{dataset_size}"
    wandb_tags  = [job.dataset_name, f"size_{dataset_size}", job.qi, "qi-analysis"]

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
            "size":          dataset_size,
        },
        tags=wandb_tags,
        group=wandb_group,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, _holdout = load_data(prepared)

        recon  = _run_attack(prepared, synth, train, qi, hidden_features)
        scores = _score_reconstruction(train, recon, hidden_features, data_type)

        metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
        ra_mean = round(float(np.mean(scores)), 4)
        metrics["RA_mean"] = ra_mean
        wandb.log(metrics)

        feat_scores = {k: v for k, v in metrics.items() if k.startswith("RA_") and k != "RA_mean"}
        return {
            "sample":  job.sample_idx,
            "sdg":     job.sdg_label,
            "attack":  job.attack_method,
            "qi":      job.qi,
            "ra_mean": ra_mean,
            "error":   None,
            **feat_scores,
        }

    except Exception:
        wandb.log({"error": traceback.format_exc()})
        raise

    finally:
        wandb.finish()


# ── Summary helpers ────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path) -> None:
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


def _print_summary(rows: list[dict], dataset_name: str, qi_variant: str) -> None:
    successes = [r for r in rows if r.get("error") is None]
    failures  = [r for r in rows if r.get("error") is not None]

    print(f"\n{'='*70}")
    print(f"  QI SWEEP COMPLETE — {dataset_name} / {qi_variant}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*70}")

    if not successes:
        return

    from collections import defaultdict
    import numpy as np

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        if r.get("ra_mean") is not None:
            groups[(r["attack"], r["qi"])].append(r["ra_mean"])

    print(f"\n  {'Attack':<25}  {'QI':<16}  {'Mean RA (avg over samples+SDG)':>32}")
    print(f"  {'-'*75}")
    for (attack, qi), vals in sorted(groups.items()):
        mean_val = round(float(np.mean(vals)), 4)
        print(f"  {attack:<25}  {qi:<16}  {mean_val:>32.4f}")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QI-analysis sweep: vary quasi-identifier set, hold synth fixed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",      required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset name (must match a key in DATASET_CONFIGS).")
    parser.add_argument("--qi",           required=True,
                        help="QI variant (e.g. QI_tiny, QI_large, QI_behavioral).")
    parser.add_argument("--workers",      type=int, default=N_WORKERS_DEFAULT,
                        help=f"Parallel worker count (default {N_WORKERS_DEFAULT}).")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print job list and exit without running anything.")
    parser.add_argument("--serial",       action="store_true",
                        help="Run jobs sequentially in the main process (for debugging).")
    parser.add_argument("--sample",       type=int, default=None,
                        help="Restrict to a single sample index.")
    parser.add_argument("--sdg",          type=str, default=None,
                        help="Restrict to jobs whose sdg_label or sdg_method matches NAME.")
    parser.add_argument("--attack",       type=str, default=None,
                        help="Restrict to jobs for this attack method name.")
    parser.add_argument("--progress-log", type=str, default=None, metavar="FILE",
                        help="Write one progress line per completed job to FILE. "
                             "Defaults to <repo_root>/outfiles/qi_<dataset>_<qi>.log.")
    args = parser.parse_args()

    dataset_name = args.dataset
    qi_variant   = args.qi
    dataset_cfg  = DATASET_CONFIGS[dataset_name]

    # Validate QI variant
    valid = dataset_cfg["valid_qi_variants"]
    if qi_variant not in valid:
        parser.error(
            f"--qi '{qi_variant}' is not valid for dataset '{dataset_name}'. "
            f"Valid options: {valid}"
        )

    # Default progress log path
    if args.progress_log is None:
        log_dir = _REPO_ROOT / "outfiles"
        log_dir.mkdir(parents=True, exist_ok=True)
        args.progress_log = str(log_dir / f"qi_{dataset_name}_{qi_variant}.log")

    all_jobs = generate_jobs(dataset_name, qi_variant)

    # Optional filtering
    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs if j.sdg_label == args.sdg or j.sdg_method == args.sdg]
    if args.attack is not None:
        all_jobs = [j for j in all_jobs if j.attack_method == args.attack]

    wandb_group = f"qi-analysis-{dataset_name}-{dataset_cfg['dataset_size']}"

    header = (
        f"{'='*70}\n"
        f"  QI sweep: {dataset_name} / {qi_variant}\n"
        f"  Dataset size:  {dataset_cfg['dataset_size']:,}\n"
        f"  Jobs total:    {len(all_jobs)}\n"
        f"  Workers:       {args.workers}\n"
        f"  WandB group:   {wandb_group}\n"
        f"  Progress log:  {args.progress_log}\n"
        f"{'='*70}\n"
    )
    print(header, end="")

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    # Verify all expected sample directories and SDG subdirs exist before dispatching.
    # This surfaces missing synth.csv early rather than mid-sweep.
    missing_dirs: list[str] = []
    for job in all_jobs:
        sdg_dir = Path(job.sample_dir) / job.sdg_label
        if not sdg_dir.exists():
            missing_dirs.append(str(sdg_dir))
    missing_dirs = list(dict.fromkeys(missing_dirs))  # deduplicate, preserve order
    if missing_dirs:
        print("ERROR: missing SDG directories (synth.csv not generated for these):")
        for d in missing_dirs:
            print(f"  {d}")
        sys.exit(1)

    progress_log = open(args.progress_log, "w", buffering=1)
    progress_log.write(header)

    start_time = time.time()
    results:  list[dict] = []
    n_done = 0
    n_fail = 0

    def _handle_result(job: Job, result_or_exc: Any) -> None:
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
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": job.qi,
                "ra_mean": None, "error": str(result_or_exc),
            })
        else:
            val_str = f"{result_or_exc['ra_mean']:.4f}" if result_or_exc["ra_mean"] is not None else "N/A"
            line = (
                f"  [{n_done:>{width}}/{len(all_jobs)}]"
                f"  {job.run_name:<64}  RA={val_str}{eta_str}"
            )
            results.append(result_or_exc)

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

    _print_summary(results, dataset_name, qi_variant)

    script_dir = Path(__file__).parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = script_dir / f"qi_{dataset_name}_{qi_variant}_{ts}.csv"
    _save_summary_csv(results, csv_path)

    progress_log.close()


if __name__ == "__main__":
    main()
