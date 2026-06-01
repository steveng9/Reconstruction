#!/usr/bin/env python
"""
Memorization sweep: train vs holdout reconstruction accuracy across three datasets.

Attacks run:
  CoBP-RA_QIGraph_EntropyBP  — best CoBP-RA variant (qi_in_graph + entropy_weighted)
  RandomForest
  KNN
  NaiveBayes
  TabPFN

Datasets / QI variants:
  adult         10k  QI1       (9 MST eps + 3 AIM eps + 7 non-DP SDGs = 19 total)
  cdc_diabetes   1k  QI1       (5 MST eps + 3 AIM eps + 7 non-DP SDGs = 15 total)
  nist_sbo       1k  QI_large  (5 MST eps + 7 non-DP SDGs = 12 total, no AIM)

Samples: 00, 01, 02  (3 trials — all have valid holdouts)
Holdout: round-robin — sample N uses sample (N+1)%5 as holdout.

Per-feature memorization scores (RA_train_{feat}, RA_nontraining_{feat}, RA_delta_{feat})
are logged to WandB and included in the local summary CSV.

Estimated job count:
  adult 10k:      19 SDGs × 5 attacks × 3 samples = 285
  cdc_diabetes 1k: 15 SDGs × 5 attacks × 3 samples = 225
  nist_sbo 1k:     12 SDGs × 5 attacks × 3 samples = 180
  Total: 690 jobs

Usage:
    conda activate recon_
    python experiment_scripts/run_memorization_sweep.py [--dry-run] [--workers N]
    python experiment_scripts/run_memorization_sweep.py --dataset adult
    python experiment_scripts/run_memorization_sweep.py --attack RandomForest --sample 0

    # Detached background (survives logout):
    nohup conda run -n recon_ python experiment_scripts/run_memorization_sweep.py \\
        --workers 4 > outfiles/mem_sweep.log 2>&1 &

    # Monitor:
    tail -f outfiles/mem_sweep_progress.log
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

DATASET_CONFIGS = [
    {
        "name":      "adult",
        "base":      "adult",
        "size":      10_000,
        "type":      "categorical",
        "data_root": "/home/golobs/data/reconstruction_data/adult/size_10000",
        "qi":        "QI1",
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
        "name":      "cdc_diabetes",
        "base":      "cdc_diabetes",
        "size":      1_000,
        "type":      "categorical",
        "data_root": "/home/golobs/data/reconstruction_data/cdc_diabetes/size_1000",
        "qi":        "QI1",
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
        "name":      "nist_sbo",
        "base":      "nist_sbo",
        "size":      1_000,
        "type":      "categorical",
        "data_root": "/home/golobs/data/reconstruction_data/nist_sbo/size_1000",
        "qi":        "QI_large",
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

SAMPLE_RANGE = [0, 1, 2]   # 3 trials; all have valid holdouts ((0+1)%5=1, etc.)


# ── Attack configurations ──────────────────────────────────────────────────────
# Each entry: (label, attack_method, param_overrides)
# label distinguishes variants in WandB; attack_method is the registered name.

_MRF = ATTACK_PARAM_DEFAULTS["CoBP-RA"]

ATTACK_CONFIGS = [
    # Best CoBP-RA variant: QI nodes as observed graph variables + entropy-weighted BP
    (
        "CoBP-RA_QIGraph_EntropyBP",
        "CoBP-RA",
        {"qi_in_graph": True, "entropy_weighted": True},
    ),
    ("RandomForest", "RandomForest", {}),
    ("KNN",          "KNN",          {}),
    ("NaiveBayes",   "NaiveBayes",   {}),
]


# ── Misc ───────────────────────────────────────────────────────────────────────

N_WORKERS     = 4
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "mem-sweep-3datasets"


# ── Job dataclass ──────────────────────────────────────────────────────────────

@dataclass
class Job:
    dataset_cfg:    dict
    sample_idx:     int
    holdout_idx:    int
    sdg_method:     str
    sdg_params:     dict
    attack_label:   str
    attack_method:  str
    attack_overrides: dict

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def dataset_name(self) -> str:
        return self.dataset_cfg["name"]

    @property
    def data_root(self) -> str:
        return self.dataset_cfg["data_root"]

    @property
    def sample_dir(self) -> str:
        return f"{self.data_root}/sample_{self.sample_idx:02d}"

    @property
    def holdout_dir(self) -> str:
        return f"{self.data_root}/sample_{self.holdout_idx:02d}"

    @property
    def run_name(self) -> str:
        return (
            f"{self.dataset_name}_sz{self.dataset_cfg['size']}__"
            f"s{self.sample_idx:02d}__"
            f"{self.sdg_label}__"
            f"{self.attack_label}__"
            f"{self.dataset_cfg['qi']}"
        )


def generate_jobs(
    dataset_filter: str | None = None,
    attack_filter:  str | None = None,
    sdg_filter:     str | None = None,
    sample_filter:  int | None = None,
) -> list[Job]:
    jobs = []
    for ds_cfg in DATASET_CONFIGS:
        if dataset_filter and ds_cfg["name"] != dataset_filter and ds_cfg["base"] != dataset_filter:
            continue
        for sample_idx, (sdg_method, sdg_params), (label, method, overrides) in itertools.product(
            SAMPLE_RANGE,
            ds_cfg["sdg_methods"],
            ATTACK_CONFIGS,
        ):
            if attack_filter and label != attack_filter and method != attack_filter:
                continue
            sdg_label = f"{sdg_method}_eps{sdg_params['epsilon']:g}" if sdg_params.get("epsilon") is not None else sdg_method
            if sdg_filter and sdg_label != sdg_filter and sdg_method != sdg_filter:
                continue
            if sample_filter is not None and sample_idx != sample_filter:
                continue
            jobs.append(Job(
                dataset_cfg=ds_cfg,
                sample_idx=sample_idx,
                holdout_idx=(sample_idx + 1) % 5,
                sdg_method=sdg_method,
                sdg_params=dict(sdg_params),
                attack_label=label,
                attack_method=method,
                attack_overrides=dict(overrides),
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
    from get_data import load_data, _normalize_sbo_strings
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction

    synth_path = Path(job.sample_dir) / job.sdg_label / "synth.csv"
    if not synth_path.exists():
        raise FileNotFoundError(f"synth.csv not found: {synth_path}")

    effective_attack_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_overrides,
    }

    cfg = {
        "dataset": {
            "name": job.dataset_name,
            "dir":  job.sample_dir,
            "size": job.dataset_cfg["size"],
            "type": job.dataset_cfg["type"],
        },
        "QI":          job.dataset_cfg["qi"],
        "data_type":   job.dataset_cfg["type"],
        "sdg_method":  job.sdg_method,
        "sdg_params":  job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": {
            "enabled":     True,
            "holdout_dir": job.holdout_dir,
        },
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            job.attack_method: dict(effective_attack_params),
        },
    }

    prepared = _prepare_config(cfg)

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config={
            "dataset":       job.dataset_name,
            "size":          job.dataset_cfg["size"],
            "qi":            job.dataset_cfg["qi"],
            "sample_idx":    job.sample_idx,
            "holdout_idx":   job.holdout_idx,
            "sdg_method":    job.sdg_method,
            "sdg_params":    job.sdg_params,
            "attack_method": job.attack_method,
            "attack_label":  job.attack_label,
            "attack_params": effective_attack_params,
        },
        tags=[job.dataset_name, f"size_{job.dataset_cfg['size']}",
              "mem-sweep", job.attack_label, job.dataset_cfg["qi"]],
        group=WANDB_GROUP,
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        # nist_sbo: normalize holdout strings (load_data only normalizes train/synth)
        if job.dataset_name == "nist_sbo" and holdout is not None:
            holdout = _normalize_sbo_strings(holdout)

        recon_train   = _run_attack(prepared, synth, train,   qi, hidden_features)
        recon_holdout = _run_attack(prepared, synth, holdout, qi, hidden_features)

        ds_type = job.dataset_cfg["type"]
        train_scores   = _score_reconstruction(train,   recon_train,   hidden_features, ds_type)
        holdout_scores = _score_reconstruction(holdout, recon_holdout, hidden_features, ds_type)

        metrics: dict[str, float] = {}
        feat_scores: dict[str, float] = {}
        for feat, ts, hs in zip(hidden_features, train_scores, holdout_scores):
            ts_r = round(float(ts), 4)
            hs_r = round(float(hs), 4)
            metrics[f"RA_train_{feat}"]       = ts_r
            metrics[f"RA_nontraining_{feat}"] = hs_r
            metrics[f"RA_delta_{feat}"]       = round(float(ts - hs), 4)
            feat_scores[f"train_{feat}"]      = ts_r
            feat_scores[f"nontrain_{feat}"]   = hs_r

        train_mean   = round(float(np.mean(train_scores)),   4)
        holdout_mean = round(float(np.mean(holdout_scores)), 4)
        delta_mean   = round(train_mean - holdout_mean,      4)
        metrics["RA_train_mean"]       = train_mean
        metrics["RA_nontraining_mean"] = holdout_mean
        metrics["RA_delta_mean"]       = delta_mean
        wandb.log(metrics)

        return {
            "dataset":       job.dataset_name,
            "size":          job.dataset_cfg["size"],
            "qi":            job.dataset_cfg["qi"],
            "sample":        job.sample_idx,
            "holdout":       job.holdout_idx,
            "sdg":           job.sdg_label,
            "attack":        job.attack_label,
            "train_mean":    train_mean,
            "nontrain_mean": holdout_mean,
            "delta_mean":    delta_mean,
            "error":         None,
            **feat_scores,
        }

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {job.run_name}:\n{tb}", flush=True)
        wandb.log({"error": str(exc)})
        raise RuntimeError(f"{job.run_name}: {exc}") from exc

    finally:
        wandb.finish()


# ── CSV / summary helpers ──────────────────────────────────────────────────────

def _save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    base_keys = ["dataset", "size", "qi", "sample", "holdout",
                 "sdg", "attack", "train_mean", "nontrain_mean", "delta_mean", "error"]
    feat_keys = sorted(
        {k for r in rows for k in r
         if k.startswith("train_") or k.startswith("nontrain_")}
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_keys + feat_keys,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved → {path}")


def _print_summary(rows: list[dict]):
    from collections import defaultdict
    import numpy as np

    successes = [r for r in rows if not r.get("error")]
    failures  = [r for r in rows if r.get("error")]
    print(f"\n{'='*90}")
    print(f"  SWEEP COMPLETE — {WANDB_GROUP}")
    print(f"  {len(successes)} succeeded  /  {len(failures)} failed")
    print(f"{'='*90}")

    groups: dict[tuple, list[tuple]] = defaultdict(list)
    for r in successes:
        key = (r["dataset"], r["size"], r["qi"], r["attack"])
        if r["train_mean"] is not None and r["nontrain_mean"] is not None:
            groups[key].append((r["train_mean"], r["nontrain_mean"], r["delta_mean"]))

    print(f"\n  {'Dataset':<18}  {'Sz':>6}  {'QI':<10}  {'Attack':<30}  "
          f"{'Train':>7}  {'NT':>7}  {'Delta':>7}  N")
    print(f"  {'-'*97}")
    for (ds, sz, qi, atk), vals in sorted(groups.items()):
        tr  = float(np.mean([v[0] for v in vals]))
        nt  = float(np.mean([v[1] for v in vals]))
        dlta = float(np.mean([v[2] for v in vals]))
        print(f"  {ds:<18}  {sz:>6}  {qi:<10}  {atk:<30}  "
              f"{tr:>7.4f}  {nt:>7.4f}  {dlta:>7.4f}  {len(vals)}")
    print(f"{'='*90}\n")

    if failures:
        print(f"  Failed jobs ({len(failures)}):")
        for r in failures[:20]:
            print(f"    {r.get('attack','?')} / {r.get('sdg','?')} / s{r.get('sample','?')}: {r.get('error','')[:120]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Memorization sweep: train vs holdout RA across 3 datasets."
    )
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print job list and exit.")
    parser.add_argument("--serial",       action="store_true",
                        help="Run in main process (Ctrl-C killable, no spawn overhead).")
    parser.add_argument("--workers",      type=int, default=N_WORKERS,
                        help=f"Parallel worker processes (default {N_WORKERS}).")
    parser.add_argument("--dataset",      type=str, default=None,
                        help="Restrict to this dataset name (adult | cdc_diabetes | nist_sbo).")
    parser.add_argument("--attack",       type=str, default=None,
                        help="Restrict to this attack label (e.g. RandomForest).")
    parser.add_argument("--sdg",          type=str, default=None,
                        help="Restrict to this SDG label or method (e.g. MST_eps1).")
    parser.add_argument("--sample",       type=int, default=None,
                        help="Restrict to this sample index (0, 1, or 2).")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" / "mem_sweep_progress.log"),
                        metavar="FILE",
                        help="One progress line per completed job (tail -f to monitor).")
    args = parser.parse_args()

    all_jobs = generate_jobs(
        dataset_filter=args.dataset,
        attack_filter=args.attack,
        sdg_filter=args.sdg,
        sample_filter=args.sample,
    )

    n_total = len(all_jobs)
    header = (
        f"{'='*80}\n"
        f"  Memorization sweep — {WANDB_GROUP}\n"
        f"  Attacks:  {len(ATTACK_CONFIGS)}: {', '.join(a for a,_,_ in ATTACK_CONFIGS)}\n"
        f"  Datasets: adult 10k (QI1), cdc_diabetes 1k (QI1), nist_sbo 1k (QI_large)\n"
        f"  Samples:  {SAMPLE_RANGE}  (holdout = (sample+1)%%5)\n"
        f"  Jobs:     {n_total}\n"
        f"  Workers:  {args.workers}\n"
        f"  WandB:    {WANDB_PROJECT} / {WANDB_GROUP}\n"
        f"{'='*80}\n"
    )
    print(header, end="")

    if args.dry_run:
        for i, j in enumerate(all_jobs):
            print(f"  [{i+1:>4d}]  {j.run_name}")
        print(f"\n{n_total} jobs total.")
        return

    # Validate synth.csv presence upfront
    missing = []
    for job in all_jobs:
        p = Path(job.sample_dir) / job.sdg_label / "synth.csv"
        if not p.exists():
            missing.append(str(p))
    if missing:
        print(f"\n[WARN] {len(missing)} synth.csv files not found — those jobs will fail:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing)-10} more")

    os.makedirs(Path(args.progress_log).parent, exist_ok=True)
    progress_log = open(args.progress_log, "w", buffering=1)
    progress_log.write(header)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(__file__).parent / f"mem_sweep_results_{ts}.csv"

    rows:    list[dict] = []
    n_done   = 0
    n_fail   = 0
    t_start  = time.time()
    width    = len(str(n_total))

    def _on_done(job: Job, result_or_exc):
        nonlocal n_done, n_fail
        n_done += 1
        elapsed = time.time() - t_start
        eta_str = ""
        if n_done > 1:
            eta_s   = (elapsed / n_done) * (n_total - n_done)
            eta_str = f"  ETA {eta_s/60:.0f}m"

        if isinstance(result_or_exc, Exception):
            n_fail += 1
            rows.append({
                "dataset": job.dataset_name, "size": job.dataset_cfg["size"],
                "qi": job.dataset_cfg["qi"], "sample": job.sample_idx,
                "holdout": job.holdout_idx, "sdg": job.sdg_label,
                "attack": job.attack_label,
                "train_mean": None, "nontrain_mean": None, "delta_mean": None,
                "error": str(result_or_exc),
            })
            line = (f"  [{n_done:>{width}}/{n_total}]  FAILED"
                    f"  {job.run_name}:  {str(result_or_exc)[:80]}")
        else:
            rows.append(result_or_exc)
            tr  = result_or_exc["train_mean"]
            nt  = result_or_exc["nontrain_mean"]
            dlta = result_or_exc["delta_mean"]
            line = (
                f"  [{n_done:>{width}}/{n_total}]  {job.run_name:<80}"
                f"  Tr={tr:.4f}  NT={nt:.4f}  Δ={dlta:+.4f}{eta_str}"
            )

        print(line, flush=True)
        progress_log.write(line + "\n")

    if args.serial:
        _worker_setup_paths()
        sys.argv = sys.argv[:1]
        for job in all_jobs:
            try:
                _on_done(job, run_job(job))
            except Exception as exc:
                _on_done(job, exc)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            future_to_job = {pool.submit(run_job, job): job for job in all_jobs}
            for fut in as_completed(future_to_job):
                job = future_to_job[fut]
                try:
                    _on_done(job, fut.result())
                except Exception as exc:
                    _on_done(job, exc)

    progress_log.close()
    total_min = (time.time() - t_start) / 60
    print(f"\nAll {n_total} jobs finished in {total_min:.1f} min.")
    _save_csv(rows, csv_path)
    _print_summary(rows)


if __name__ == "__main__":
    main()
