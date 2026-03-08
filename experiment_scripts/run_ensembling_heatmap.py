#!/usr/bin/env python
"""
Ensembling heatmap sweep: every 2-attack pair on adult data.

Job unit: one (sample × SDG × QI) combination.
Each job runs all N attacks exactly once, then computes all N*(N-1)/2
ensemble aggregations from the already-held predictions — zero redundancy.

Logs to WandB:
  - N individual runs  (run_type="individual", attack_method=attack_name)
  - N*(N-1)/2 ensemble runs  (run_type="ensemble", attack_method="A+B")
All runs land in WANDB_GROUP so the heatmap can be built from a single query.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_ensembling_heatmap.py
    python experiment_scripts/run_ensembling_heatmap.py --dry-run
    python experiment_scripts/run_ensembling_heatmap.py --workers 4
    python experiment_scripts/run_ensembling_heatmap.py --sdg TVAE
    python experiment_scripts/run_ensembling_heatmap.py --sample 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_NAME  = "adult"
DATASET_SIZE  = 1_000
DATA_ROOT     = f"/home/golobs/data/reconstruction_data/adult/size_{DATASET_SIZE}"
SAMPLE_RANGE  = list(range(5))   # sample_00 through sample_04
DATASET_TYPE  = "categorical"

QI_VARIANTS = ["QI1"]

# Attacks included in the heatmap — all N*(N-1)/2 pairs are computed automatically.
# Excludes: SVM (O(n²) at n=10k), AttentionAutoregressive (slow),
#           LinearReconstruction (Gurobi), diffusion attacks (too slow for exhaustive pairs).
ATTACKS = [
    #"Mode",
    "Random",
    "KNN",
    "NaiveBayes",
    "LogisticRegression",
    "RandomForest",
    "LightGBM",
    "MLP",
    "PartialMSTBounded",
]

SDG_METHODS = [
    #("MST",      {"epsilon": 1.0}),
    ("MST",      {"epsilon": 10.0}),
    ("TVAE",     {}),
    #("CTGAN",    {}),
    ("Synthpop",    {}),
    #("ARF",    {}),
    ("TabDDPM",    {}),
]

AGGREGATION   = "soft_voting"   # uses predicted probability distributions where available

N_WORKERS     = 3
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = f"ensembling-heatmap-{DATASET_NAME}-{DATASET_SIZE}"
WANDB_TAGS    = [DATASET_NAME, f"size_{DATASET_SIZE}", "ensembling", "heatmap"]


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    """One job = one (sample × SDG × QI) combination.

    Inside run_job, ALL attacks in ATTACKS are run once each, and all
    N*(N-1)/2 ensemble pairs are computed from those cached results.
    """
    sample_idx: int
    sdg_method: str
    sdg_params: dict
    qi:         str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{DATA_ROOT}/sample_{self.sample_idx:02d}"

    @property
    def job_label(self) -> str:
        return f"s{self.sample_idx:02d}__{self.sdg_label}__{self.qi}"

    def run_name(self, attack: str) -> str:
        return f"s{self.sample_idx:02d}__{self.sdg_label}__{attack}__{self.qi}"


def generate_jobs() -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), qi in itertools.product(
        SAMPLE_RANGE, SDG_METHODS, QI_VARIANTS
    ):
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=dict(sdg_params),
            qi=qi,
        ))
    return jobs


# ── Worker function (runs in a subprocess) ────────────────────────────────────

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


def _make_single_cfg(job: Job, attack_name: str) -> dict:
    """Config for a single attack with no ensembling or chaining."""
    return {
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
        "attack_method": attack_name,
        "memorization_test": {"enabled": False},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            attack_name:  {**ATTACK_PARAM_DEFAULTS.get(attack_name, {})},
        },
    }


def _log_wandb_run(project, group, tags, run_name, wandb_cfg, metrics):
    import wandb
    wandb.init(project=project, group=group, tags=tags,
               name=run_name, config=wandb_cfg, reinit=True)
    wandb.log(metrics)
    wandb.finish()


def run_job(job: Job) -> list[dict[str, Any]]:
    """
    Run all attacks once, then compute all pairwise ensemble aggregations.

    Returns a flat list of result dicts — N individual rows followed by
    N*(N-1)/2 ensemble rows.
    """
    _worker_setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    from get_data import load_data
    from master_experiment_script import _prepare_config, _score_reconstruction
    from attacks import get_attack
    from enhancements.ensembling_wrapper import _aggregate_predictions

    # Load data once — shared across all attacks for this (sample, SDG, QI)
    base_prepared = _prepare_config(_make_single_cfg(job, ATTACKS[0]))
    train, synth, qi, hidden_features, _holdout = load_data(base_prepared)

    base_wandb_cfg = {
        "sample_idx": job.sample_idx,
        "sdg_method": job.sdg_method,
        "sdg_params": job.sdg_params,
        "qi":         job.qi,
        "dataset":    DATASET_NAME,
        "size":       DATASET_SIZE,
    }

    # ── Phase 1: run each attack once ─────────────────────────────────────────
    attack_results: dict[str, dict] = {}   # attack_name → {recon, probas, classes, scores, ra}

    for attack_name in ATTACKS:
        prepared = _prepare_config(_make_single_cfg(job, attack_name))
        try:
            attack_fn = get_attack(attack_name, DATASET_TYPE)
            recon, probas, classes = attack_fn(prepared, synth, train, qi, hidden_features)
            scores = _score_reconstruction(train, recon, hidden_features, DATASET_TYPE)
            ra = round(float(np.mean(scores)), 4)

            attack_results[attack_name] = {
                "recon": recon, "probas": probas, "classes": classes,
                "scores": scores, "ra": ra,
            }

            _log_wandb_run(
                WANDB_PROJECT, WANDB_GROUP, WANDB_TAGS,
                run_name  = job.run_name(attack_name),
                wandb_cfg = {**base_wandb_cfg, "attack_method": attack_name,
                             "run_type": "individual"},
                metrics   = {f"RA_{f}": s for f, s in zip(hidden_features, scores)}
                          | {"RA_mean": ra},
            )

        except Exception as exc:
            print(f"  WARNING: {attack_name} failed on {job.job_label}: {exc}")
            attack_results[attack_name] = None

    # ── Phase 2: ensemble every pair from cached results ──────────────────────
    results = []

    # Individual rows first
    for attack_name in ATTACKS:
        r = attack_results.get(attack_name)
        results.append({
            "type":    "individual",
            "sample":  job.sample_idx,
            "sdg":     job.sdg_label,
            "attack":  attack_name,
            "qi":      job.qi,
            "ra_mean": r["ra"] if r is not None else None,
            "error":   None if r is not None else "attack failed",
        })

    # Ensemble rows
    for attack_a, attack_b in itertools.combinations(ATTACKS, 2):
        r_a = attack_results.get(attack_a)
        r_b = attack_results.get(attack_b)
        pair_label = f"{attack_a}+{attack_b}"

        if r_a is None or r_b is None:
            results.append({
                "type": "ensemble", "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": pair_label, "qi": job.qi,
                "ra_mean": None, "error": "one or both attacks failed",
            })
            continue

        try:
            ens_recon = _aggregate_predictions(
                [r_a["recon"],   r_b["recon"]],
                [r_a["probas"],  r_b["probas"]],
                [r_a["classes"], r_b["classes"]],
                hidden_features, AGGREGATION, None, DATASET_TYPE,
            )
            ens_scores = _score_reconstruction(train, ens_recon, hidden_features, DATASET_TYPE)
            ra_ens = round(float(np.mean(ens_scores)), 4)

            _log_wandb_run(
                WANDB_PROJECT, WANDB_GROUP, WANDB_TAGS,
                run_name  = job.run_name(pair_label),
                wandb_cfg = {**base_wandb_cfg, "attack_method": pair_label,
                             "attack_a": attack_a, "attack_b": attack_b,
                             "aggregation": AGGREGATION, "run_type": "ensemble"},
                metrics   = {f"RA_{f}": s for f, s in zip(hidden_features, ens_scores)}
                          | {"RA_mean": ra_ens},
            )

            results.append({
                "type": "ensemble", "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": pair_label, "qi": job.qi,
                "ra_mean": ra_ens, "error": None,
            })

        except Exception as exc:
            results.append({
                "type": "ensemble", "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": pair_label, "qi": job.qi,
                "ra_mean": None, "error": str(exc),
            })

    return results


# ── Summary helpers ────────────────────────────────────────────────────────────

def _save_summary_csv(rows: list[dict], path: Path):
    if not rows:
        return
    keys = ["type", "sample", "sdg", "attack", "qi", "ra_mean", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary CSV saved to: {path}")


def _print_summary(rows: list[dict]):
    import numpy as np
    from collections import defaultdict

    successes = [r for r in rows if r.get("error") is None]
    failures  = [r for r in rows if r.get("error") is not None]

    print(f"\n{'='*70}")
    print(f"  HEATMAP SWEEP COMPLETE — {DATASET_NAME}")
    print(f"  {len(successes)} successful result rows  /  {len(failures)} errors")
    print(f"{'='*70}")

    if not successes:
        return

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        if r.get("ra_mean") is not None:
            groups[(r["type"], r["attack"])].append(r["ra_mean"])

    indiv = sorted([(k, v) for k, v in groups.items() if k[0] == "individual"],
                   key=lambda x: -float(np.mean(x[1])))
    ens   = sorted([(k, v) for k, v in groups.items() if k[0] == "ensemble"],
                   key=lambda x: -float(np.mean(x[1])))

    print(f"\n  Individual attacks (avg over samples × SDG):")
    print(f"  {'Attack':<30}  {'N':>4}  {'Mean RA':>8}")
    print(f"  {'-'*46}")
    for (_, attack), vals in indiv:
        print(f"  {attack:<30}  {len(vals):>4}  {float(np.mean(vals)):>8.4f}")

    print(f"\n  Top 20 ensemble pairs:")
    print(f"  {'Pair':<40}  {'N':>4}  {'Mean RA':>8}")
    print(f"  {'-'*56}")
    for (_, attack), vals in ens[:20]:
        print(f"  {attack:<40}  {len(vals):>4}  {float(np.mean(vals)):>8.4f}")
    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    n_attacks = len(ATTACKS)
    n_pairs   = n_attacks * (n_attacks - 1) // 2
    n_jobs    = len(SDG_METHODS) * len(SAMPLE_RANGE) * len(QI_VARIANTS)

    parser = argparse.ArgumentParser(description="Ensembling heatmap sweep.")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--serial",       action="store_true",
                        help="Run sequentially in the main process.")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--sample",       type=int, default=None,
                        help="Run only this sample index.")
    parser.add_argument("--sdg",          type=str, default=None,
                        help="Run only this SDG method name (e.g. 'MST').")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles"
                                    / "ensemble_heatmap_progress.log"),
                        metavar="FILE")
    args = parser.parse_args()

    all_jobs = generate_jobs()
    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs if j.sdg_method == args.sdg]

    progress_log = open(args.progress_log, "w", buffering=1) if args.progress_log else None

    header = (
        f"{'='*70}\n"
        f"  Ensembling heatmap: {DATASET_NAME}\n"
        f"  Attacks:      {n_attacks}  →  {n_pairs} pairs per job\n"
        f"  SDG methods:  {len(SDG_METHODS)}\n"
        f"  Samples:      {len(SAMPLE_RANGE)}\n"
        f"  Jobs:         {len(all_jobs)} of {n_jobs}  "
        f"(each runs {n_attacks} attacks + {n_pairs} ensembles)\n"
        f"  Workers:      {args.workers}\n"
        f"  WandB group:  {WANDB_GROUP}\n"
        f"{'='*70}\n"
    )
    print(header, end="")
    if progress_log:
        progress_log.write(header)

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>3d}]  {j.job_label}")
        print(f"\n{len(all_jobs)} jobs — {len(all_jobs) * (n_attacks + n_pairs)} WandB runs total.")
        return

    missing = list(dict.fromkeys(
        job.sample_dir for job in all_jobs if not Path(job.sample_dir).exists()
    ))
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
            eta_s   = (len(all_jobs) - n_done) / (n_done / elapsed)
            eta_str = f"  ETA {eta_s/60:.0f}m"
        width = len(str(len(all_jobs)))

        if isinstance(result_or_exc, Exception):
            n_fail += 1
            line = (f"  [{n_done:>{width}}/{len(all_jobs)}]"
                    f"  FAILED  {job.job_label}:  {result_or_exc}")
            results.append({
                "type": "ensemble", "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": "ALL", "qi": job.qi, "ra_mean": None,
                "error": str(result_or_exc),
            })
        else:
            results.extend(result_or_exc)
            indiv_rows = [r for r in result_or_exc
                          if r["type"] == "individual" and r["ra_mean"] is not None]
            ens_rows   = [r for r in result_or_exc
                          if r["type"] == "ensemble"   and r["ra_mean"] is not None]
            best_ind = max(indiv_rows, key=lambda r: r["ra_mean"], default=None)
            best_ens = max(ens_rows,   key=lambda r: r["ra_mean"], default=None)
            best_ind_str = f"{best_ind['attack']}={best_ind['ra_mean']:.4f}" if best_ind else "N/A"
            best_ens_str = f"{best_ens['attack']}={best_ens['ra_mean']:.4f}" if best_ens else "N/A"
            line = (f"  [{n_done:>{width}}/{len(all_jobs)}]  {job.job_label:<40}"
                    f"  best_ind={best_ind_str}  best_ens={best_ens_str}{eta_str}")

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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"ensembling_heatmap_results_{ts}.csv"
    _save_summary_csv(results, csv_path)

    if progress_log:
        progress_log.close()


if __name__ == "__main__":
    main()
