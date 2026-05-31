#!/usr/bin/env python
"""
MarginalRF architectural variants sweep.

Tests three novel architectural changes to the MarginalRF attack, each
independently and in one combination, against the baseline:

  MarginalRF            — baseline (RF unary, no entropy weighting, hidden-only graph)
  MarginalRF_LGBUnary   — LightGBM replaces RandomForest as the per-feature unary model
  MarginalRF_EntropyBP  — entropy-weighted message passing (uncertain nodes dampen their
                          messages by 1 − H/H_max, preventing wrong confident propagation)
  MarginalRF_QIGraph    — QI features included as observed (near-delta) nodes in the
                          graphical model; BP propagates QI certainty to hidden neighbours
  MarginalRF_LGB_EntropyBP — combination: LGB unary + entropy weighting

These are architectural variants of MarginalRF itself — distinct from the
chaining/ensembling enhancements in run_marginalrf_chains_ensembles.py.

Dataset: adult 10k, QI1, samples 00–04, 5 focused SDGs.
Total: 5 configs × 5 SDGs × 5 samples = 125 jobs.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_marginalrf_variants.py
    python experiment_scripts/run_marginalrf_variants.py --dry-run
    python experiment_scripts/run_marginalrf_variants.py --workers 6
    python experiment_scripts/run_marginalrf_variants.py --attack MarginalRF_QIGraph
    python experiment_scripts/run_marginalrf_variants.py --sdg TabDDPM --sample 0
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
from dataclasses import dataclass
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
SAMPLE_RANGE  = list(range(5))    # sample_00 through sample_04


# ── SDG methods (quality-range coverage) ──────────────────────────────────────

SDG_METHODS = [
    ("MST",      {"epsilon": 1.0}),
    ("MST",      {"epsilon": 10.0}),
    ("TVAE",     {}),
    ("Synthpop", {}),
    ("TabDDPM",  {}),
]


# ── Attack configurations ──────────────────────────────────────────────────────
# Each entry: (label, attack_method, attack_params_overrides)
# All configs use attack_method="MarginalRF"; the label distinguishes them in
# WandB and results CSVs.  Variants differ only in their attack_params.

_MRF = ATTACK_PARAM_DEFAULTS["MarginalRF"]

ATTACK_CONFIGS = [
    # ── Baseline ──────────────────────────────────────────────────────────
    # All defaults: RF unary, no entropy weighting, hidden features only in graph.
    (
        "MarginalRF",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": dict(_MRF),
        },
    ),

    # ── Variant: NaiveBayes unary model ──────────────────────────────────
    # Replaces RandomForestClassifier with GaussianNB for per-feature posterior
    # estimation.  NB tends to be under-confident (too uniform) — the opposite
    # calibration problem from LightGBM (overconfident).  Comparing the two
    # tells us whether any deviation from RF calibration hurts, or only
    # overconfidence specifically.  NB is also essentially free to run.
    (
        "MarginalRF_NBUnary",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": {**dict(_MRF), "unary_model": "NaiveBayes"},
        },
    ),

    # ── Variant: entropy-weighted belief propagation ──────────────────────
    # Scales each outgoing BP message by the sender node's confidence weight
    # w = 1 − H(p) / H_max ∈ [0, 1].  Uncertain nodes contribute less to
    # their neighbours' beliefs; confident nodes pass messages at full strength.
    # Hypothesis: prevents wrong confident predictions from propagating and
    #             pulling other features' beliefs astray.
    (
        "MarginalRF_EntropyBP",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": {**dict(_MRF), "entropy_weighted": True},
        },
    ),

    # ── Variant: QI nodes as observed graph variables ─────────────────────
    # Eligible QI features (cardinality ≤ max_pair_cardinality) are added to
    # the graphical model as observed nodes with near-delta unaries at each
    # target's known QI value.  BP propagates this certainty to connected
    # hidden neighbours, adding QI-to-hidden dependency modelling on top of
    # the existing hidden-to-hidden correction.
    # Hypothesis: captures QI-hidden correlations that the RF unary misses
    #             (especially for features weakly predicted by QI alone).
    (
        "MarginalRF_QIGraph",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": {**dict(_MRF), "qi_in_graph": True},
        },
    ),

    # ── Combination: NaiveBayes unary + entropy-weighted BP ──────────────
    # Tests whether entropy weighting can compensate for NB's under-confidence
    # by boosting confident NB predictions and suppressing the noisy uniform ones.
    (
        "MarginalRF_NB_EntropyBP",
        "MarginalRF",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "MarginalRF": {**dict(_MRF),
                           "unary_model":      "NaiveBayes",
                           "entropy_weighted": True},
        },
    ),
]


# ── Misc ───────────────────────────────────────────────────────────────────────

N_WORKERS     = 4
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "marginalrf-variants-adult-10k"


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

    # Extract the resolved MarginalRF params for WandB logging
    mrf_params  = job.attack_params.get("MarginalRF", {})
    wandb_cfg = {
        "sample_idx":       job.sample_idx,
        "dataset":          DATASET_NAME,
        "size":             DATASET_SIZE,
        "sdg_method":       job.sdg_method,
        "sdg_params":       job.sdg_params,
        "attack_method":    job.attack_method,
        "attack_label":     job.attack_label,
        "qi":               job.qi,
        # Variant-distinguishing params — logged explicitly so WandB filters work
        "unary_model":      mrf_params.get("unary_model",      "RF"),
        "entropy_weighted": mrf_params.get("entropy_weighted", False),
        "qi_in_graph":      mrf_params.get("qi_in_graph",      False),
    }

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config=wandb_cfg,
        tags=[DATASET_NAME, f"size_{DATASET_SIZE}", "marginalrf-variants", job.attack_label],
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


# ── Progress logging ────────────────────────────────────────────────────────────

def _progress_line(job: Job, ra_mean: float, elapsed: float, eta: float | None) -> str:
    eta_str = f"ETA {int(eta // 60)}m" if eta is not None else "ETA --"
    return (
        f"  {job.run_name:<80}  "
        f"RA={ra_mean * 100:.4f}  {eta_str}"
    )


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

    print(f"\n  {'Label':<30}  {'SDG':<20}  {'N':>3}  {'Mean RA':>8}")
    print(f"  {'-'*68}")
    for (label, sdg), vals in sorted(groups.items()):
        print(f"  {label:<30}  {sdg:<20}  {len(vals):>3}  {float(np.mean(vals)):>8.4f}")
    print(f"{'='*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MarginalRF architectural variants sweep.")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--serial",       action="store_true",
                        help="Run in main process (Ctrl-C killable).")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--attack",       type=str, default=None,
                        help="Restrict to one attack label (e.g. MarginalRF_QIGraph).")
    parser.add_argument("--sdg",          type=str, default=None)
    parser.add_argument("--sample",       type=int, default=None)
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" /
                                    "marginalrf_variants.log"))
    args = parser.parse_args()

    all_jobs = generate_jobs(
        attack_filter=args.attack,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )

    header = (
        f"{'='*80}\n"
        f"  MarginalRF architectural variants sweep\n"
        f"  Dataset:  {DATASET_NAME} {DATASET_SIZE}\n"
        f"  Configs:  {len(ATTACK_CONFIGS)} variants\n"
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
    if missing:
        print("\n[WARN] Missing sample directories (jobs will fail):")
        for d in sorted(set(missing)):
            print(f"  {d}")

    os.makedirs(Path(args.progress_log).parent, exist_ok=True)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"marginalrf_variants_{ts}.csv"

    rows:      list[dict] = []
    n_done     = 0
    n_total    = len(all_jobs)
    t_start    = time.time()

    def _on_done(job: Job, result: dict):
        nonlocal n_done
        n_done += 1
        elapsed = time.time() - t_start
        eta     = (elapsed / n_done) * (n_total - n_done) if n_done > 0 else None
        ra      = result.get("ra_mean", 0.0) or 0.0
        line    = f"  [{n_done}/{n_total}]  {job.run_name:<80}  RA={ra:.4f}  ETA {int((eta or 0)//60)}m"
        print(line, flush=True)
        with open(args.progress_log, "a") as lf:
            lf.write(line + "\n")

    if args.serial:
        for job in all_jobs:
            result = run_job(job)
            rows.append(result)
            _on_done(job, result)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_job, job): job for job in all_jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    result = fut.result()
                    rows.append(result)
                    _on_done(job, result)
                except Exception as exc:
                    n_done += 1
                    err_line = f"  [{n_done}/{n_total}]  FAILED  {job.run_name}  {exc}"
                    print(err_line, flush=True)
                    rows.append({
                        "dataset": DATASET_NAME, "size": DATASET_SIZE,
                        "sample": job.sample_idx, "sdg": job.sdg_label,
                        "attack": job.attack_method, "label": job.attack_label,
                        "qi": job.qi, "ra_mean": None, "error": str(exc),
                    })

    _save_csv(rows, csv_path)
    _print_summary(rows)


if __name__ == "__main__":
    main()
