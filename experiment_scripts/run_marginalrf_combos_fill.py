#!/usr/bin/env python
"""
CoBP-RA combination variants sweep — FILL-IN RUN.

Fills the missing cells in Table 1 for the QI-graph CoBP-RA variants.
The original run_marginalrf_combos.py covered 5 SDGs:
    MST eps1, MST eps10, TVAE, Synthpop, TabDDPM

This script covers the 7 remaining SDGs that are currently "--" in the table:
    RankSwap, CellSuppression, MST eps100, MST eps1000, AIM eps1, CTGAN, ARF

3 attack configs × 7 SDGs × 5 samples = 105 jobs.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_marginalrf_combos_fill.py
    python experiment_scripts/run_marginalrf_combos_fill.py --dry-run
    python experiment_scripts/run_marginalrf_combos_fill.py --workers 7
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
SAMPLE_RANGE  = list(range(5))


# ── SDG methods — the 7 missing from the original combos run ──────────────────

SDG_METHODS = [
    ("RankSwap",        {}),
    ("CellSuppression", {}),
    ("MST",             {"epsilon": 100.0}),
    ("MST",             {"epsilon": 1000.0}),
    ("AIM",             {"epsilon": 1.0}),
    ("CTGAN",           {}),
    ("ARF",             {}),
]


# ── Attack configurations (same 3 as original combos run) ─────────────────────

_MRF = ATTACK_PARAM_DEFAULTS["CoBP-RA"]

ATTACK_CONFIGS = [
    (
        "CoBP-RA_QIGraph_EntropyBP",
        "CoBP-RA",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "CoBP-RA": {**dict(_MRF),
                           "qi_in_graph":      True,
                           "entropy_weighted": True},
        },
    ),
    (
        "CoBP-RA_QIGraph_AllQI",
        "CoBP-RA",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "CoBP-RA": {**dict(_MRF),
                           "qi_in_graph":       True,
                           "max_qi_cardinality": 10_000},
        },
    ),
    (
        "CoBP-RA_QIGraph_EntropyBP_AllQI",
        "CoBP-RA",
        {
            "chaining":   {"enabled": False},
            "ensembling": {"enabled": False},
            "CoBP-RA": {**dict(_MRF),
                           "qi_in_graph":       True,
                           "entropy_weighted":  True,
                           "max_qi_cardinality": 10_000},
        },
    ),
]


# ── Misc ───────────────────────────────────────────────────────────────────────

N_WORKERS     = 7
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "marginalrf-combos-adult-10k"   # same group — results merge in WandB


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

    cfg = {
        "dataset": {
            "name": DATASET_NAME,
            "dir":  job.sample_dir,
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

    mrf_params = job.attack_params.get("CoBP-RA", {})
    wandb_cfg = {
        "sample_idx":         job.sample_idx,
        "dataset":            DATASET_NAME,
        "size":               DATASET_SIZE,
        "sdg_method":         job.sdg_method,
        "sdg_params":         job.sdg_params,
        "attack_method":      job.attack_method,
        "attack_label":       job.attack_label,
        "qi":                 job.qi,
        "unary_model":        mrf_params.get("unary_model",       "RF"),
        "entropy_weighted":   mrf_params.get("entropy_weighted",  False),
        "qi_in_graph":        mrf_params.get("qi_in_graph",       False),
        "max_qi_cardinality": mrf_params.get("max_qi_cardinality", None),
    }

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config=wandb_cfg,
        tags=[DATASET_NAME, f"size_{DATASET_SIZE}", "marginalrf-combos", job.attack_label,
              "fill-in"],
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
        f"  {job.run_name:<85}  "
        f"RA={ra_mean:.4f}  {eta_str}"
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
    print(f"  {WANDB_GROUP}  [fill-in: {len(SDG_METHODS)} new SDGs]")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*80}")

    groups: dict[tuple, list] = defaultdict(list)
    for r in successes:
        key = (r["label"], r["sdg"])
        if r["ra_mean"] is not None:
            groups[key].append(r["ra_mean"])

    print(f"\n  {'Label':<45}  {'SDG':<20}  {'N':>3}  {'Mean RA':>8}")
    print(f"  {'-'*80}")
    for (label, sdg), vals in sorted(groups.items()):
        print(f"  {label:<45}  {sdg:<20}  {len(vals):>3}  {float(np.mean(vals)):>8.4f}")
    print(f"{'='*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CoBP-RA combos fill-in: 7 missing SDG methods.")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--serial",       action="store_true",
                        help="Run in main process (Ctrl-C killable).")
    parser.add_argument("--workers",      type=int, default=N_WORKERS)
    parser.add_argument("--attack",       type=str, default=None)
    parser.add_argument("--sdg",          type=str, default=None)
    parser.add_argument("--sample",       type=int, default=None)
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" /
                                    "marginalrf_combos_fill.log"))
    args = parser.parse_args()

    all_jobs = generate_jobs(
        attack_filter=args.attack,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )

    n_sdg    = len({j.sdg_label for j in all_jobs})
    n_attack = len({j.attack_label for j in all_jobs})
    n_sample = len({j.sample_idx  for j in all_jobs})

    header = (
        f"{'='*80}\n"
        f"  CoBP-RA combos fill-in sweep\n"
        f"  Dataset:  {DATASET_NAME} {DATASET_SIZE}\n"
        f"  Configs:  {n_attack} variants\n"
        f"  SDGs:     {n_sdg} (fill-in)\n"
        f"  Samples:  {n_sample}\n"
        f"  Jobs:     {len(all_jobs)}\n"
        f"  Workers:  {args.workers}\n"
        f"  WandB:    {WANDB_GROUP}\n"
        f"{'='*80}\n"
    )

    log_path = Path(args.progress_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(header)
        for j in all_jobs:
            print(f"  {j.run_name}")
        return

    print(header)
    with open(log_path, "w") as f:
        f.write(header)

    if args.serial:
        results = []
        t0 = time.monotonic()
        for idx, job in enumerate(all_jobs):
            r = run_job(job)
            elapsed = time.monotonic() - t0
            done = idx + 1
            eta = (elapsed / done) * (len(all_jobs) - done) if done > 0 else None
            line = _progress_line(job, r["ra_mean"], elapsed, eta)
            print(f"  [{done:>{len(str(len(all_jobs)))}}/{len(all_jobs)}]  {line}")
            with open(log_path, "a") as f:
                f.write(f"  [{done:>{len(str(len(all_jobs)))}}/{len(all_jobs)}]  {line}\n")
            results.append(r)
    else:
        ctx = mp.get_context("spawn")
        results = []
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
                    r = future.result()
                    line = _progress_line(job, r["ra_mean"], elapsed, eta)
                    results.append(r)
                except Exception as exc:
                    line = f"  {job.run_name:<85}  ERROR: {exc}"
                    results.append({"error": str(exc), "label": job.attack_label,
                                    "sdg": job.sdg_label, "sample": job.sample_idx,
                                    "ra_mean": None})
                msg = f"  [{done:>{width}}/{n_total}]  {line}"
                print(msg, flush=True)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"marginalrf_combos_fill_{ts}.csv"
    _save_csv(results, csv_path)
    _print_summary(results)


if __name__ == "__main__":
    main()
