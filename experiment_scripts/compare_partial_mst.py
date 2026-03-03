#!/usr/bin/env python
"""
Compare PartialMST attack variants against baselines on nist_arizona_25feat.

Runs Mode, Random, RandomForest, PartialMST, and PartialMSTIndependent across
all available samples for one or more SDG methods, then prints a per-attack
summary table (mean RA averaged across samples and, optionally, SDG methods).

No WandB logging — results go to stdout and an optional CSV.

Usage (from repo root, conda activate recon_):
    python experiment_scripts/compare_partial_mst.py
    python experiment_scripts/compare_partial_mst.py --sdg MST_eps1 MST_eps10
    python experiment_scripts/compare_partial_mst.py --samples 0 1 2
    python experiment_scripts/compare_partial_mst.py --out results.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────

sys.path.insert(0, "/home/golobs/Reconstruction")
for p in [
    "/home/golobs/MIA_on_diffusion/",
    "/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM",
    "/home/golobs/recon-synth",
    "/home/golobs/recon-synth/attacks",
    "/home/golobs/recon-synth/attacks/solvers",
]:
    if p not in sys.path:
        sys.path.append(p)

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_BASE  = "nist_arizona_data"
DATASET_NAME  = "nist_arizona_25feat"
DATASET_SIZE  = 10_000
N_FEATURES    = 25
DATA_ROOT     = (
    f"/home/golobs/data/reconstruction_data/{DATASET_BASE}"
    f"/size_{DATASET_SIZE}_{N_FEATURES}feat"
)
DATASET_TYPE  = "categorical"
QI_VARIANT    = "QI3"

# SDG methods to test (must have synth.csv already generated)
N_WORKERS = 8

DEFAULT_SDG_METHODS = [
    "MST_eps0.1",
    "MST_eps1",
    "MST_eps10",
    "MST_eps100",
    "MST_eps1000",
    "AIM_eps1",
    "TVAE",
    "CTGAN",
    "ARF",
    "TabDDPM",
    "Synthpop",
    "RankSwap",
    "CellSuppression",
]

# Attacks to compare — (display_name, attack_method, extra_attack_params)
ATTACKS = [
    ("Random",             "Random",                {}),
    ("RandomForest",       "RandomForest",          {}),
    #("BoundedK4",          "PartialMSTBounded",     {"retrain": False, "max_clique_size": 4}),
    #("BoundedK5",          "PartialMSTBounded",     {"retrain": False, "max_clique_size": 5}),
    ("PartialMST",         "PartialMST",            {"retrain": False}),
    ("PartialMST-argmax",  "PartialMST",            {"retrain": False, "sample_mode": "argmax"}),
    #("PartialMST-top20",   "PartialMST",            {"retrain": False, "sample_mode": "top_pct", "top_pct": 20.0}),
    ("BoundedK3",          "PartialMSTBounded",     {"retrain": False, "max_clique_size": 3}),
    ("BoundedK3-top10%",    "PartialMSTBounded",     {"retrain": False, "max_clique_size": 3, "sample_mode": "top_pct", "top_pct": 10.0}),
    ("BoundedK3-top20%",    "PartialMSTBounded",     {"retrain": False, "max_clique_size": 3, "sample_mode": "top_pct", "top_pct": 20.0}),
    #("BoundedK3-top30%",    "PartialMSTBounded",     {"retrain": False, "max_clique_size": 3, "sample_mode": "top_pct", "top_pct": 30.0}),
    #("BoundedK3-top40%",    "PartialMSTBounded",     {"retrain": False, "max_clique_size": 3, "sample_mode": "top_pct", "top_pct": 40.0}),
    #("BoundedK3-top50%",    "PartialMSTBounded",     {"retrain": False, "max_clique_size": 3, "sample_mode": "top_pct", "top_pct": 50.0}),
    #("BoundedK3-top60%",    "PartialMSTBounded",     {"retrain": False, "max_clique_size": 3, "sample_mode": "top_pct", "top_pct": 60.0}),
    #("BoundedK3-argmax",   "PartialMSTBounded",     {"retrain": False, "sample_mode": "argmax", "max_clique_size": 3}),
    ("BoundedK3-argmax",   "PartialMSTBounded",     {"retrain": False, "sample_mode": "argmax", "max_clique_size": 3}),
    #("BoundedK4-argmax",   "PartialMSTBounded",     {"retrain": False, "sample_mode": "argmax", "max_clique_size": 4}),
    #("PartialMSTIndep",    "PartialMSTIndependent", {"retrain": False}),
    # New variants:
    #("BoundedK2",          "PartialMSTBounded",     {"retrain": False, "max_clique_size": 2}),
    #("Hub",                "PartialMSTHub",         {"retrain": False}),
]

RETRAIN_METHODS = {"PartialMST", "PartialMSTIndependent", "PartialMSTBounded", "PartialMSTHub"}

# ── Core run logic ────────────────────────────────────────────────────────────

def run_one(sample_idx: int, sdg_dirname: str, display_name: str,
            attack_method: str, extra_params: dict) -> dict:
    """Run a single (sample, SDG, attack) job and return a result dict."""
    # master_experiment_script calls parse_args() at module level — clear argv
    # before importing it so it doesn't choke on this script's arguments.
    sys.argv = sys.argv[:1]

    import numpy as np
    from attack_defaults import ATTACK_PARAM_DEFAULTS
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    sample_dir = f"{DATA_ROOT}/sample_{sample_idx:02d}"
    synth_path = Path(sample_dir) / sdg_dirname / "synth.csv"
    if not synth_path.exists():
        return {
            "sample": sample_idx, "sdg": sdg_dirname, "attack": display_name,
            "ra_mean": None, "scores": None, "error": f"missing {synth_path}",
        }

    # Reconstruct sdg_method and sdg_params from the dirname
    # dirname is like "MST_eps1", "AIM_eps10", "TVAE", "RankSwap", etc.
    if "_eps" in sdg_dirname:
        sdg_method, eps_str = sdg_dirname.rsplit("_eps", 1)
        sdg_params = {"epsilon": float(eps_str)}
    else:
        sdg_method = sdg_dirname
        sdg_params = {}

    cfg = {
        "dataset": {
            "name": DATASET_NAME,
            "dir":  sample_dir,
            "size": DATASET_SIZE,
            "type": DATASET_TYPE,
        },
        "QI":            QI_VARIANT,
        "data_type":     DATASET_TYPE,
        "sdg_method":    sdg_method,
        "sdg_params":    sdg_params,
        "attack_method": attack_method,
        "memorization_test": {"enabled": False},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            attack_method: {
                **ATTACK_PARAM_DEFAULTS.get(attack_method, {}),
                **extra_params,
            },
        },
    }

    prepared = _prepare_config(cfg)

    try:
        train, synth, qi, hidden_features, _ = load_data(prepared)
        recon  = _run_attack(prepared, synth, train, qi, hidden_features)
        scores = _score_reconstruction(train, recon, hidden_features, DATASET_TYPE)
        ra_mean = round(float(np.mean(scores)), 4)
        per_feat = {f: round(float(s), 4) for f, s in zip(hidden_features, scores)}
        return {
            "sample": sample_idx, "sdg": sdg_dirname, "attack": display_name,
            "ra_mean": ra_mean, "scores": per_feat, "error": None,
        }
    except Exception as exc:
        return {
            "sample": sample_idx, "sdg": sdg_dirname, "attack": display_name,
            "ra_mean": None, "scores": None, "error": str(exc),
        }


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_table(rows: list[dict], sdg_methods: list[str]):
    """Print mean RA table: rows = attacks, columns = SDG methods."""
    import numpy as np
    from collections import defaultdict

    attack_names = [a[0] for a in ATTACKS]

    # Accumulate: (attack, sdg) -> list of ra_mean
    cell: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r["error"] is None and r["ra_mean"] is not None:
            cell[(r["attack"], r["sdg"])].append(r["ra_mean"])

    # Column width
    col_w = 11
    name_w = 18

    header = f"{'Attack':<{name_w}}" + "".join(f"  {s[:col_w]:>{col_w}}" for s in sdg_methods)
    print("\n" + "=" * len(header))
    print(f"  RA comparison — {DATASET_NAME}  (mean over samples, higher = better attacker)")
    print("=" * len(header))
    print(f"  {header}")
    print(f"  {'-' * (len(header) - 2)}")

    for name in attack_names:
        row_str = f"  {name:<{name_w}}"
        for sdg in sdg_methods:
            vals = cell.get((name, sdg), [])
            if vals:
                row_str += f"  {np.mean(vals):>{col_w}.4f}"
            else:
                row_str += f"  {'—':>{col_w}}"
        print(row_str)

    print("=" * len(header) + "\n")


def _save_csv(rows: list[dict], path: Path):
    keys = ["sample", "sdg", "attack", "ra_mean", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare PartialMST vs baselines.")
    parser.add_argument(
        "--sdg", nargs="+", default=None, metavar="SDG_DIRNAME",
        help=(
            "SDG sub-directory names to test (default: all present in sample_00). "
            "E.g. --sdg MST_eps1 MST_eps10 AIM_eps1 TVAE"
        ),
    )
    parser.add_argument(
        "--samples", nargs="+", type=int, default=None, metavar="N",
        help="Sample indices to use (default: all found under DATA_ROOT).",
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE",
        help="Save raw results to this CSV path.",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retrain PartialMST checkpoints even if they exist.",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="Run jobs one at a time in the main process (Ctrl-C friendly).",
    )
    parser.add_argument(
        "--workers", type=int, default=N_WORKERS, metavar="N",
        help=f"Number of parallel workers (default: {N_WORKERS}; ignored with --serial).",
    )
    args = parser.parse_args()

    # Resolve sample indices
    if args.samples is not None:
        sample_indices = args.samples
    else:
        sample_indices = sorted(
            int(p.name.split("_")[1])
            for p in Path(DATA_ROOT).iterdir()
            if p.is_dir() and p.name.startswith("sample_")
        )

    # Resolve SDG methods: filter to those with synth.csv in sample_00
    if args.sdg is not None:
        sdg_methods = args.sdg
    else:
        first_sample = Path(DATA_ROOT) / f"sample_{sample_indices[0]:02d}"
        sdg_methods = [
            d for d in DEFAULT_SDG_METHODS
            if (first_sample / d / "synth.csv").exists()
        ]

    if not sdg_methods:
        print("No SDG methods found. Check DATA_ROOT or pass --sdg explicitly.")
        sys.exit(1)

    # Apply --retrain flag to PartialMST attacks

    attacks = []
    for name, method, params in ATTACKS:
        p = dict(params)
        if args.retrain and method in RETRAIN_METHODS:
            p["retrain"] = True
        attacks.append((name, method, p))

    # Flatten to a list of job tuples
    jobs = [
        (sample_idx, sdg_dirname, display_name, attack_method, extra_params)
        for sdg_dirname in sdg_methods
        for sample_idx in sample_indices
        for display_name, attack_method, extra_params in attacks
    ]
    total = len(jobs)

    mode_str = "serial" if args.serial else f"parallel ({args.workers} workers)"
    print(f"\nRunning {total} jobs [{mode_str}]:")
    print(f"  samples  : {sample_indices}")
    print(f"  SDG      : {sdg_methods}")
    print(f"  attacks  : {[a[0] for a in attacks]}\n")

    rows = []
    t0 = time.time()

    if args.serial:
        for done, (sample_idx, sdg_dirname, display_name, attack_method, extra_params) in enumerate(jobs):
            label = f"  [{done+1:3d}/{total}] s{sample_idx:02d}  {sdg_dirname:<14}  {display_name}"
            print(label, end="", flush=True)
            t1 = time.time()
            result = run_one(sample_idx, sdg_dirname, display_name, attack_method, extra_params)
            elapsed = time.time() - t1
            if result["error"]:
                print(f"  ERROR ({elapsed:.1f}s): {result['error'][:60]}")
            else:
                print(f"  RA={result['ra_mean']:.4f}  ({elapsed:.1f}s)")
            rows.append(result)
    else:
        ctx = mp.get_context("spawn")
        futures_map = {}
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            for job in jobs:
                f = pool.submit(run_one, *job)
                futures_map[f] = job
            done = 0
            for f in as_completed(futures_map):
                sample_idx, sdg_dirname, display_name, *_ = futures_map[f]
                done += 1
                try:
                    result = f.result()
                except Exception as exc:
                    result = {
                        "sample": sample_idx, "sdg": sdg_dirname,
                        "attack": display_name, "ra_mean": None,
                        "scores": None, "error": str(exc),
                    }
                label = f"  [{done:3d}/{total}] s{sample_idx:02d}  {sdg_dirname:<14}  {display_name}"
                if result["error"]:
                    print(f"{label}  ERROR: {result['error'][:60]}", flush=True)
                else:
                    print(f"{label}  RA={result['ra_mean']:.4f}", flush=True)
                rows.append(result)

    total_time = time.time() - t0
    print(f"\nFinished {total} jobs in {total_time:.1f}s")

    _print_table(rows, sdg_methods)

    if args.out:
        _save_csv(rows, Path(args.out))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = Path(__file__).parent / f"partial_mst_comparison_{ts}.csv"
        _save_csv(rows, default_out)


if __name__ == "__main__":
    main()
