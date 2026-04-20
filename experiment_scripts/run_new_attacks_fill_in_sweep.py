#!/usr/bin/env python
"""
New-attacks fill-in sweep: TabPFN and MarginalRF variants on datasets not yet
covered by run_new_attacks_sweep.py.

Datasets
--------
  cdc_diabetes  100k   QI1
  nist_sbo        1k   QI1, QI_large

NOTE: california (continuous) is excluded — TabPFN and MarginalRF are
registered as categorical attacks only.  Add continuous variants to
attacks/__init__.py before including california here.

MarginalRF ablation modes
--------------------------
  knn_k=None   global (unconditional) PMI — fast, may double-count QI-mediated correlation
  knn_k=50     local PMI, small neighbourhood
  knn_k=100    local PMI, default (validated on adult QI1)
  knn_k=200    local PMI, large neighbourhood

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_new_attacks_fill_in_sweep.py
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --dry-run
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --workers 8
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --dataset cdc_diabetes
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --dataset nist_sbo
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --attack TabPFN
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --attack MarginalRF_mst_local_100
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --sdg Synthpop
    python experiment_scripts/run_new_attacks_fill_in_sweep.py --qi QI_large
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Dataset configurations ─────────────────────────────────────────────────────
# Each entry has its own sdg_methods list (available methods differ by dataset).

DATASET_CONFIGS = [
    #{
    #    "base":        "cdc_diabetes",
    #    "name":        "cdc_diabetes",
    #    "size":        100_000,
    #    "type":        "categorical",
    #    "qi_variants": ["QI1"],
    #    "data_root":   "/home/golobs/data/reconstruction_data/cdc_diabetes/size_100000",
    #    "sdg_methods": [
    #        ("MST",             {"epsilon": 0.1}),
    #        ("MST",             {"epsilon": 1.0}),
    #        ("MST",             {"epsilon": 10.0}),
    #        ("MST",             {"epsilon": 100.0}),
    #        ("MST",             {"epsilon": 1000.0}),
    #        #("AIM",             {"epsilon": 1.0}),
    #        #("AIM",             {"epsilon": 10.0}),
    #        ("TVAE",            {}),
    #        ("CTGAN",           {}),
    #        ("ARF",             {}),
    #        ("TabDDPM",         {}),
    #        ("Synthpop",        {}),
    #        ("RankSwap",        {}),
    #        ("CellSuppression", {}),
    #    ],
    #},
    # ── california excluded — TabPFN/MarginalRF are categorical-only ───────────
    # Uncomment and add continuous variants to attacks/__init__.py to enable.
    # {
    #     "base":        "california",
    #     "name":        "california",
    #     "size":        1_000,
    #     "type":        "continuous",
    #     "qi_variants": ["QI1"],
    #     "data_root":   "/home/golobs/data/reconstruction_data/california/size_1000",
    #     "sdg_methods": [
    #         ("MST",      {"epsilon": 0.1}),
    #         ("MST",      {"epsilon": 1.0}),
    #         ("MST",      {"epsilon": 10.0}),
    #         ("MST",      {"epsilon": 100.0}),
    #         ("MST",      {"epsilon": 1000.0}),
    #         ("AIM",      {"epsilon": 1.0}),
    #         ("AIM",      {"epsilon": 3.0}),
    #         ("AIM",      {"epsilon": 10.0}),
    #         ("TVAE",     {}),
    #         ("CTGAN",    {}),
    #         ("ARF",      {}),
    #         ("TabDDPM",  {}),
    #         ("Synthpop", {}),
    #         ("RankSwap", {}),
    #     ],
    # },
    {
        "base":        "nist_sbo",
        "name":        "nist_sbo",
        "size":        1_000,
        "type":        "categorical",
        "qi_variants": ["QI1", "QI_large"],
        "data_root":   "/home/golobs/data/reconstruction_data/nist_sbo/size_1000",
        "sdg_methods": [
            ("MST",             {"epsilon": 0.1}),
            ("MST",             {"epsilon": 1.0}),
            ("MST",             {"epsilon": 10.0}),
            ("MST",             {"epsilon": 100.0}),
            ("MST",             {"epsilon": 1000.0}),
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
# display_label: used in run_name and WandB to distinguish variants of the same
#   attack.  Leave "" to use attack_method directly.

ATTACK_CONFIGS = [
    # ── TabPFN (in-context learning) ──────────────────────────────────────────
    ("TabPFN",     {},                                                    ""),

    # ── MarginalRF: PMI mode (global vs local) × graph structure ──────────────
    #("MarginalRF", {"knn_k": None,  "graph_type": "mst"},                "MarginalRF_mst_global"),
    #("MarginalRF", {"knn_k": 50,    "graph_type": "mst"},                "MarginalRF_mst_local_50"),
    #("MarginalRF", {"knn_k": 100,   "graph_type": "mst"},                "MarginalRF_mst_local_100"),
    #("MarginalRF", {"knn_k": 200,   "graph_type": "mst"},                "MarginalRF_mst_local_200"),
    #("MarginalRF", {"knn_k": 500,   "graph_type": "mst"},                "MarginalRF_mst_local_500"),
    #("MarginalRF", {"knn_k": 100,   "graph_type": "complete"},           "MarginalRF_complete_local_100"),
    #("MarginalRF", {"knn_k": 100,   "graph_type": "topk"},               "MarginalRF_topk_local_100"),
    #("MarginalRF", {"knn_k": None,  "graph_type": "complete"},           "MarginalRF_complete_global"),
]


# ── Misc configuration ─────────────────────────────────────────────────────────

N_WORKERS     = 8
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "new-attacks-sweep-1k"   # same group as run_new_attacks_sweep.py


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    dataset_cfg:   dict
    sample_idx:    int
    sdg_method:    str
    sdg_params:    dict
    attack_method: str
    attack_params: dict
    attack_label:  str   # display name; "" → use attack_method
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
            f"s{self.sample_idx:02d}__"
            f"{self.sdg_label}__"
            f"{self.effective_label}__"
            f"{self.qi}"
        )


def generate_jobs(
    dataset_filter: str | None = None,
    attack_filter:  str | None = None,
    sdg_filter:     str | None = None,
    sample_filter:  int | None = None,
    qi_filter:      str | None = None,
) -> list[Job]:
    jobs = []
    for ds_cfg in DATASET_CONFIGS:
        if dataset_filter and ds_cfg["name"] != dataset_filter:
            continue
        for sample_idx, (sdg_method, sdg_params), (attack_method, attack_params, attack_label), qi in itertools.product(
            SAMPLE_RANGE,
            ds_cfg["sdg_methods"],
            ATTACK_CONFIGS,
            ds_cfg["qi_variants"],
        ):
            effective_label = attack_label or attack_method
            if attack_filter and attack_method != attack_filter and effective_label != attack_filter:
                continue
            sdg_label = f"{sdg_method}_eps{sdg_params['epsilon']:g}" if sdg_params.get("epsilon") is not None else sdg_method
            if sdg_filter and sdg_label != sdg_filter and sdg_method != sdg_filter:
                continue
            if sample_filter is not None and sample_idx != sample_filter:
                continue
            if qi_filter and qi != qi_filter:
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

    synth_path = Path(sample_dir) / job.sdg_label / "synth.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"synth.csv not found: {synth_path}")

    effective_attack_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_params,
    }

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
        tags=[ds["name"], f"size_{ds['size']}", "new-attacks", job.effective_label, job.qi],
        group=WANDB_GROUP,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, _ = load_data(prepared)
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
        key = (r["dataset"], r.get("label") or r["attack"], r["qi"])
        if r["ra_mean"] is not None:
            groups[key].append(r["ra_mean"])

    print(f"\n  {'Dataset':<16}  {'Attack / Label':<34}  {'QI':<10}  {'Mean RA':>10}")
    print(f"  {'-'*75}")
    for (dataset, label, qi), vals in sorted(groups.items()):
        mean_val = round(float(np.mean(vals)), 4)
        print(f"  {dataset:<16}  {label:<34}  {qi:<10}  {mean_val:>10.4f}")
    print(f"{'='*80}\n")

    if failures:
        print(f"  Failed jobs ({len(failures)}):")
        for r in failures:
            print(f"    {r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="New-attacks fill-in sweep: TabPFN + MarginalRF on cdc_diabetes 100k and nist_sbo 1k."
    )
    parser.add_argument("--dry-run",    action="store_true", help="Print job list and exit.")
    parser.add_argument("--serial",     action="store_true", help="Run sequentially in the main process.")
    parser.add_argument("--workers",    type=int,  default=N_WORKERS,  help="Parallel workers.")
    parser.add_argument("--dataset",    type=str,  default=None, help="Restrict to this dataset name.")
    parser.add_argument("--attack",     type=str,  default=None, help="Restrict to this attack method or label.")
    parser.add_argument("--sdg",        type=str,  default=None, help="Restrict to this SDG method/label.")
    parser.add_argument("--sample",     type=int,  default=None, help="Restrict to this sample index.")
    parser.add_argument("--qi",         type=str,  default=None, help="Restrict to this QI variant.")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "new_attacks_fill_in_progress.log"),
                        metavar="FILE",
                        help="Write one progress line per completed job (tail -f to monitor).")
    args = parser.parse_args()

    all_jobs = generate_jobs(
        dataset_filter=args.dataset,
        attack_filter=args.attack,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
        qi_filter=args.qi,
    )

    dataset_names = ", ".join(dict.fromkeys(j.dataset_name for j in all_jobs)) if all_jobs else "none"
    header = (
        f"{'='*80}\n"
        f"  New-attacks fill-in sweep\n"
        f"  Datasets:    {dataset_names}\n"
        f"  Jobs total:  {len(all_jobs)}\n"
        f"  Workers:     {args.workers}\n"
        f"  WandB group: {WANDB_GROUP}\n"
        f"{'='*80}\n"
    )
    print(header, end="")

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{len(all_jobs)} jobs total.")
        return

    # Verify sample directories exist before dispatching
    missing = list(dict.fromkeys(
        j.sample_dir for j in all_jobs if not Path(j.sample_dir).exists()
    ))
    if missing:
        print("ERROR: missing sample directories:")
        for d in missing:
            print(f"  {d}")
        sys.exit(1)

    Path(args.progress_log).parent.mkdir(parents=True, exist_ok=True)
    progress_log = open(args.progress_log, "w", buffering=1)
    progress_log.write(header)

    start_time = time.time()
    results: list[dict] = []
    n_done = 0
    n_fail = 0
    width  = len(str(len(all_jobs)))

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
                f"  {job.run_name:<75}  RA={val_str}{eta_str}"
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
    progress_log.write(f"\nAll {len(all_jobs)} jobs finished in {total_min:.1f} min.\n")
    progress_log.close()

    _print_summary(results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"new_attacks_fill_in_results_{ts}.csv"
    _save_summary_csv(results, csv_path)


if __name__ == "__main__":
    main()
