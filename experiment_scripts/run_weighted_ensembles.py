#!/usr/bin/env python
"""
Weighted soft-vote ensemble sweep.

Key innovation: each attack is run ONCE per (sample, SDG) job; its predictions
and probability distributions are cached in memory.  Multiple weighted ensemble
combinations are then computed from the cached results without re-running any
attack.  This eliminates all redundant computation.

Individual attacks logged (for reference):
  CoBP-RA, LightGBM, NaiveBayes, KNN, MLP

Ensemble configurations (soft voting with per-model weights):
  Ensemble_W_a: CoBP-RA(0.40), LightGBM(0.30), NaiveBayes(0.15), KNN(0.15)
  Ensemble_W_b: CoBP-RA(0.35), LightGBM(0.25), MLP(0.25),        KNN(0.15)
  Ensemble_W_c: CoBP-RA(0.40), MLP(0.35),       KNN(0.15),        NaiveBayes(0.10)

Soft voting aligns each model's class-probability array to a common class space,
applies per-model weights, sums, and argmaxes.  Models without real probas
(KNN) are one-hotted from their hard predictions.

Each (sample, SDG) job logs 8 separate WandB runs: 5 individual + 3 ensembles.

Dataset: adult 10k, QI1, samples 00–04, 5 SDGs.
Total: 5 SDGs × 5 samples = 25 jobs; each job logs 8 WandB runs (200 total runs).

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_weighted_ensembles.py
    python experiment_scripts/run_weighted_ensembles.py --dry-run
    python experiment_scripts/run_weighted_ensembles.py --workers 3
    python experiment_scripts/run_weighted_ensembles.py --sdg TabDDPM --sample 0
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import multiprocessing as mp

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Dataset configuration ──────────────────────────────────────────────────────

DATASET_NAME  = "adult"
DATASET_SIZE  = 10_000
DATASET_TYPE  = "categorical"
DATA_ROOT     = f"/home/golobs/data/reconstruction_data/adult/size_{DATASET_SIZE}"
QI_VARIANTS   = ["QI1"]
SAMPLE_RANGE  = list(range(5))


# ── SDG methods ────────────────────────────────────────────────────────────────

SDG_METHODS = [
    ("MST",      {"epsilon": 1.0}),
    ("MST",      {"epsilon": 10.0}),
    ("TVAE",     {}),
    ("Synthpop", {}),
    ("TabDDPM",  {}),
]


# ── Individual attacks to run (run once per job, results cached) ───────────────
# Each entry: (label, attack_method, attack_params_overrides)

INDIVIDUAL_ATTACKS = [
    ("CoBP-RA", "CoBP-RA", {
        "chaining":   {"enabled": False},
        "ensembling": {"enabled": False},
        "CoBP-RA": dict(ATTACK_PARAM_DEFAULTS["CoBP-RA"]),
    }),
    ("LightGBM", "LightGBM", {
        "chaining":   {"enabled": False},
        "ensembling": {"enabled": False},
        "LightGBM":   dict(ATTACK_PARAM_DEFAULTS["LightGBM"]),
    }),
    ("NaiveBayes", "NaiveBayes", {
        "chaining":   {"enabled": False},
        "ensembling": {"enabled": False},
    }),
    ("KNN", "KNN", {
        "chaining":   {"enabled": False},
        "ensembling": {"enabled": False},
        "KNN":        dict(ATTACK_PARAM_DEFAULTS["KNN"]),
    }),
    ("MLP", "MLP", {
        "chaining":   {"enabled": False},
        "ensembling": {"enabled": False},
        "MLP":        dict(ATTACK_PARAM_DEFAULTS["MLP"]),
    }),
]


# ── Ensemble configurations ────────────────────────────────────────────────────
# Each entry: (label, [(attack_label, weight), ...])
# Weights are automatically normalised to sum=1 inside _weighted_soft_vote.

ENSEMBLE_CONFIGS = [
    # (a) MRF-heavy with LGB + equal small-weight models
    ("Ensemble_W_a", [
        ("CoBP-RA", 0.40),
        ("LightGBM",   0.30),
        ("NaiveBayes", 0.15),
        ("KNN",        0.15),
    ]),
    # (b) MRF + LGB + MLP triangle
    ("Ensemble_W_b", [
        ("CoBP-RA", 0.35),
        ("LightGBM",   0.25),
        ("MLP",        0.25),
        ("KNN",        0.15),
    ]),
    # (c) MRF + MLP strong, KNN + NB minor
    ("Ensemble_W_c", [
        ("CoBP-RA", 0.40),
        ("MLP",        0.35),
        ("KNN",        0.15),
        ("NaiveBayes", 0.10),
    ]),
]


# ── Misc ───────────────────────────────────────────────────────────────────────

N_WORKERS     = 3
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "weighted-ensembles-adult-10k"


# ── Job specification ──────────────────────────────────────────────────────────

@dataclass
class Job:
    sample_idx: int
    sdg_method: str
    sdg_params: dict
    qi:         str

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def sample_dir(self) -> str:
        return f"{DATA_ROOT}/sample_{self.sample_idx:02d}"

    def run_name(self, attack_label: str) -> str:
        return (
            f"{DATASET_NAME}__"
            f"sz{DATASET_SIZE}__"
            f"s{self.sample_idx:02d}__"
            f"{self.sdg_label}__"
            f"{attack_label}__"
            f"{self.qi}"
        )


def generate_jobs(
    sdg_filter:    str | None = None,
    sample_filter: int | None = None,
) -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), qi in itertools.product(
        SAMPLE_RANGE, SDG_METHODS, QI_VARIANTS,
    ):
        sdg_label = (f"{sdg_method}_eps{sdg_params['epsilon']:g}"
                     if sdg_params.get("epsilon") is not None else sdg_method)
        if sdg_filter and sdg_label != sdg_filter and sdg_method != sdg_filter:
            continue
        if sample_filter is not None and sample_idx != sample_filter:
            continue
        jobs.append(Job(
            sample_idx=sample_idx,
            sdg_method=sdg_method,
            sdg_params=dict(sdg_params),
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


def _run_single_attack(atk_label, atk_method, atk_params, cfg_base, synth, train,
                        qi_cols, hidden_features, data_type):
    """Run one attack, return (recon_df, probas_list, classes_list)."""
    from master_experiment_script import _prepare_config
    from attacks import get_attack

    cfg = dict(cfg_base)
    cfg["attack_method"]  = atk_method
    cfg["attack_params"]  = dict(atk_params)
    prepared = _prepare_config(cfg)

    attack_fn = get_attack(atk_method, data_type)
    return attack_fn(prepared, synth, train, qi_cols, hidden_features)


def _weighted_soft_vote(attack_results, ensemble_members, hidden_features):
    """Compute a weighted soft-vote ensemble from cached attack results.

    attack_results: dict {label: (recon_df, probas_list, classes_list)}
    ensemble_members: [(label, weight), ...]
    Returns a reconstructed DataFrame.
    """
    import numpy as np
    import pandas as pd
    from enhancements.ensembling_wrapper import _soft_voting

    labels  = [lbl for lbl, _ in ensemble_members]
    raw_w   = np.array([w for _, w in ensemble_members], dtype=float)
    weights = raw_w / raw_w.sum()

    all_recons  = [attack_results[lbl][0] for lbl in labels]
    all_probas  = [attack_results[lbl][1] for lbl in labels]
    all_classes = [attack_results[lbl][2] for lbl in labels]

    result = all_recons[0].copy()
    for feat_idx, feat in enumerate(hidden_features):
        feat_preds = [r[feat] for r in all_recons]
        result[feat] = _soft_voting(feat_preds, all_probas, all_classes, feat_idx, weights)

    return result


def run_job(job: Job) -> list[dict[str, Any]]:
    """Run all individual attacks + ensembles for one (sample, SDG) pair.

    Logs one WandB run per attack/ensemble variant (8 total).
    Returns a list of result dicts.
    """
    _worker_setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config
    from scoring import calculate_reconstruction_score

    sample_dir = job.sample_dir

    cfg_base = {
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
        "attack_method": "CoBP-RA",        # placeholder, overridden per attack
        "memorization_test": {"enabled": False},
        "attack_params": {},
    }

    # Load data once — shared across all attacks in this job
    prepared_base = _prepare_config(dict(cfg_base))
    train, synth, qi_cols, hidden_features, _holdout = load_data(prepared_base)

    # ── Step 1: Run each individual attack and cache results ─────────────────
    attack_results: dict[str, tuple] = {}  # label → (recon_df, probas, classes)
    all_rows: list[dict] = []

    for atk_label, atk_method, atk_params in INDIVIDUAL_ATTACKS:
        run_name = job.run_name(atk_label)
        print(f"    [{atk_label}] {run_name}", flush=True)

        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "sample_idx":   job.sample_idx,
                "dataset":      DATASET_NAME,
                "size":         DATASET_SIZE,
                "sdg_method":   job.sdg_method,
                "sdg_params":   job.sdg_params,
                "attack_method": atk_method,
                "attack_label": atk_label,
                "qi":           job.qi,
                "run_type":     "individual",
            },
            tags=[DATASET_NAME, f"size_{DATASET_SIZE}", "weighted-ensembles",
                  "individual", atk_label],
            group=WANDB_GROUP,
            reinit=True,
        )

        try:
            recon, probas, classes = _run_single_attack(
                atk_label, atk_method, atk_params,
                cfg_base, synth, train, qi_cols, hidden_features, DATASET_TYPE,
            )
            attack_results[atk_label] = (recon, probas, classes)

            scores  = calculate_reconstruction_score(train, recon, hidden_features)
            ra_mean = round(float(np.mean(scores)), 4)
            metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
            metrics["RA_mean"] = ra_mean
            wandb.log(metrics)

            feat_scores = {k: v for k, v in metrics.items()
                          if k.startswith("RA_") and k != "RA_mean"}
            all_rows.append({
                "dataset":    DATASET_NAME,
                "size":       DATASET_SIZE,
                "sample":     job.sample_idx,
                "sdg":        job.sdg_label,
                "attack":     atk_method,
                "label":      atk_label,
                "run_type":   "individual",
                "qi":         job.qi,
                "ra_mean":    ra_mean,
                "error":      None,
                **feat_scores,
            })

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n[ERROR] {run_name}:\n{tb}", flush=True)
            wandb.log({"error": str(exc)})
            attack_results[atk_label] = None
            all_rows.append({
                "dataset": DATASET_NAME, "size": DATASET_SIZE,
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": atk_method, "label": atk_label,
                "run_type": "individual", "qi": job.qi,
                "ra_mean": None, "error": str(exc),
            })
        finally:
            wandb.finish()

    # ── Step 2: Compute weighted ensembles from cached results ───────────────
    for ens_label, members in ENSEMBLE_CONFIGS:
        run_name = job.run_name(ens_label)
        print(f"    [{ens_label}] {run_name}", flush=True)

        # Only proceed if all required attacks succeeded
        missing = [lbl for lbl, _ in members if attack_results.get(lbl) is None]

        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "sample_idx":   job.sample_idx,
                "dataset":      DATASET_NAME,
                "size":         DATASET_SIZE,
                "sdg_method":   job.sdg_method,
                "sdg_params":   job.sdg_params,
                "attack_method": "Ensemble",
                "attack_label": ens_label,
                "qi":           job.qi,
                "run_type":     "ensemble",
                "ensemble_members": [lbl for lbl, _ in members],
                "ensemble_weights": [w for _, w in members],
            },
            tags=[DATASET_NAME, f"size_{DATASET_SIZE}", "weighted-ensembles",
                  "ensemble", ens_label],
            group=WANDB_GROUP,
            reinit=True,
        )

        try:
            if missing:
                raise RuntimeError(f"Missing cached results for: {missing}")

            ens_recon = _weighted_soft_vote(attack_results, members, hidden_features)
            scores    = calculate_reconstruction_score(train, ens_recon, hidden_features)
            ra_mean   = round(float(np.mean(scores)), 4)
            metrics   = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
            metrics["RA_mean"] = ra_mean
            wandb.log(metrics)

            feat_scores = {k: v for k, v in metrics.items()
                          if k.startswith("RA_") and k != "RA_mean"}
            all_rows.append({
                "dataset":    DATASET_NAME,
                "size":       DATASET_SIZE,
                "sample":     job.sample_idx,
                "sdg":        job.sdg_label,
                "attack":     "Ensemble",
                "label":      ens_label,
                "run_type":   "ensemble",
                "qi":         job.qi,
                "ra_mean":    ra_mean,
                "error":      None,
                **feat_scores,
            })

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"\n[ERROR] {run_name}:\n{tb}", flush=True)
            wandb.log({"error": str(exc)})
            all_rows.append({
                "dataset": DATASET_NAME, "size": DATASET_SIZE,
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": "Ensemble", "label": ens_label,
                "run_type": "ensemble", "qi": job.qi,
                "ra_mean": None, "error": str(exc),
            })
        finally:
            wandb.finish()

    return all_rows


# ── Progress logging ────────────────────────────────────────────────────────────

def _job_summary_line(job: Job, rows: list[dict], elapsed: float, eta: float | None) -> str:
    """One-line summary for a completed job (shows RA for each variant)."""
    eta_str = f"ETA {int(eta // 60)}m" if eta is not None else "ETA --"
    ra_parts = []
    for r in rows:
        if r.get("ra_mean") is not None:
            ra_parts.append(f"{r['label']}={r['ra_mean']:.2f}")
    ra_str = "  ".join(ra_parts)
    job_id = f"{DATASET_NAME}__s{job.sample_idx:02d}__{job.sdg_label}"
    return f"  {job_id:<45}  {ra_str}  {eta_str}"


# ── Summary helpers ─────────────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    base_keys = ["dataset", "size", "sample", "sdg", "attack", "label",
                 "run_type", "qi", "ra_mean", "error"]
    feat_keys = sorted({k for r in rows for k in r if k.startswith("RA_") and k != "RA_mean"})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_keys + feat_keys,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")


def _print_summary(all_rows: list[dict]):
    import numpy as np
    from collections import defaultdict

    successes = [r for r in all_rows if not r.get("error")]
    failures  = [r for r in all_rows if r.get("error")]

    print(f"\n{'='*80}")
    print(f"  {WANDB_GROUP}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*80}")

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["run_type"], r["label"], r["sdg"])
        if r["ra_mean"] is not None:
            groups[key].append(r["ra_mean"])

    print(f"\n  {'Type':<12}  {'Label':<30}  {'SDG':<20}  {'N':>3}  {'Mean RA':>8}")
    print(f"  {'-'*77}")
    for (rtype, label, sdg), vals in sorted(groups.items()):
        print(f"  {rtype:<12}  {label:<30}  {sdg:<20}  {len(vals):>3}  "
              f"{float(np.mean(vals)):>8.4f}")
    print(f"{'='*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weighted soft-vote ensemble sweep.")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--serial",       action="store_true",
                        help="Run in main process (Ctrl-C killable).")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--sdg",          type=str, default=None)
    parser.add_argument("--sample",       type=int, default=None)
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" /
                                    "weighted_ensembles.log"))
    args = parser.parse_args()

    all_jobs = generate_jobs(
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )

    n_attacks   = len(INDIVIDUAL_ATTACKS)
    n_ensembles = len(ENSEMBLE_CONFIGS)
    n_runs      = len(all_jobs) * (n_attacks + n_ensembles)

    header = (
        f"{'='*80}\n"
        f"  Weighted soft-vote ensemble sweep\n"
        f"  Dataset:    {DATASET_NAME} {DATASET_SIZE}\n"
        f"  Attacks:    {n_attacks} individual ({', '.join(l for l,_,_ in INDIVIDUAL_ATTACKS)})\n"
        f"  Ensembles:  {n_ensembles} weighted combos\n"
        f"  SDGs:       {len(SDG_METHODS)}\n"
        f"  Samples:    {len(SAMPLE_RANGE)}\n"
        f"  Jobs:       {len(all_jobs)} (each logs {n_attacks + n_ensembles} WandB runs)\n"
        f"  WandB runs: {n_runs} total\n"
        f"  Workers:    {args.workers}\n"
        f"  WandB:      {WANDB_GROUP}\n"
        f"{'='*80}\n"
    )

    log_path = Path(args.progress_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(header)
        for j in all_jobs:
            print(f"  {j.run_name('(all variants)')}")
        return

    print(header)
    with open(log_path, "w") as f:
        f.write(header)

    if args.serial:
        all_results = []
        t0 = time.monotonic()
        for idx, job in enumerate(all_jobs):
            rows = run_job(job)
            elapsed = time.monotonic() - t0
            done = idx + 1
            eta = (elapsed / done) * (len(all_jobs) - done) if done > 0 else None
            line = _job_summary_line(job, rows, elapsed, eta)
            n = len(all_jobs)
            w = len(str(n))
            msg = f"  [{done:>{w}}/{n}]  {line}"
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
            all_results.extend(rows)
    else:
        ctx = mp.get_context("spawn")
        all_results = []
        futures = {}
        t0 = time.monotonic()
        n_total = len(all_jobs)
        width   = len(str(n_total))
        done    = 0

        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            for job in all_jobs:
                futures[executor.submit(run_job, job)] = job

            for future in as_completed(futures):
                job = futures[future]
                done += 1
                elapsed = time.monotonic() - t0
                eta = (elapsed / done) * (n_total - done) if done > 0 else None
                try:
                    rows = future.result()
                    line = _job_summary_line(job, rows, elapsed, eta)
                    all_results.extend(rows)
                except Exception as exc:
                    line = f"  {job.run_name('ERROR'):<80}  ERROR: {exc}"
                msg = f"  [{done:>{width}}/{n_total}]  {line}"
                print(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"weighted_ensembles_{ts}.csv"
    _save_csv(all_results, csv_path)
    _print_summary(all_results)


if __name__ == "__main__":
    main()
