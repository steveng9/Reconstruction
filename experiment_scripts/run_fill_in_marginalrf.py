#!/usr/bin/env python
"""
run_fill_in_marginalrf.py — fill in MarginalRF_graphQI_entropyBP (best variant) results.

Uses the DB to determine which jobs are already done and skip them.
Writes results incrementally (one row per job) to per-group CSVs AND inserts
each result into results.db after completion.

Groups (run in priority order A → E → B → C → D):
  Group A — CDC Diabetes 1k,     QI1,              15 SDGs, 5 samples
  Group E — California 1k,       QI_large,         14 SDGs, 5 samples  (+memorization test)
  Group B — CDC Diabetes 100k,   QI1,              13 SDGs, 5 samples
  Group C — NIST SBO 1k,         QI1 + QI_large,   12 SDGs, 5 samples
  Group D — Adult 10k,           QI_large + QI_beh, 9 MST ε, 4 samples

Attack labels stored in DB:
  Groups A–D: "MarginalRF_graphQI_entropyBP"  (qi_in_graph=True, entropy_weighted=True)
  Group  E:   "MarginalRF_continuous"          (quantile-discretized, knn_k=None)

Usage:
    conda activate recon_
    python experiment_scripts/run_fill_in_marginalrf.py
    python experiment_scripts/run_fill_in_marginalrf.py --dry-run
    python experiment_scripts/run_fill_in_marginalrf.py --workers 8
    python experiment_scripts/run_fill_in_marginalrf.py --group A

Overnight detached run:
    nohup conda run -n recon_ python experiment_scripts/run_fill_in_marginalrf.py \\
        --workers 8 >outfiles/marginalrf_fillin.log 2>&1 &
    echo "PID=$!"; tail -f outfiles/marginalrf_fillin.log
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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_ROOT     = Path("/home/golobs/data/reconstruction_data")
REPO_ROOT     = Path(__file__).resolve().parent.parent
DB_PATH       = Path(__file__).parent / "results.db"
WANDB_PROJECT = "tabular-reconstruction-attacks"
N_WORKERS     = 8

# ── SDG lists ─────────────────────────────────────────────────────────────────

_CDC_1K_SDGS = [
    ("MST",  {"epsilon": 0.1}),
    ("MST",  {"epsilon": 1.0}),
    ("MST",  {"epsilon": 10.0}),
    ("MST",  {"epsilon": 100.0}),
    ("MST",  {"epsilon": 1000.0}),
    ("AIM",  {"epsilon": 1.0}),
    ("AIM",  {"epsilon": 3.0}),
    ("AIM",  {"epsilon": 10.0}),
    ("TVAE", {}),
    ("CTGAN", {}),
    ("ARF",  {}),
    ("TabDDPM", {}),
    ("Synthpop", {}),
    ("CellSuppression", {}),
    ("RankSwap", {}),
]

# AIM_eps3 and AIM_eps10 not generated for CDC 100k
_CDC_100K_SDGS = [
    ("MST",  {"epsilon": 0.1}),
    ("MST",  {"epsilon": 1.0}),
    ("MST",  {"epsilon": 10.0}),
    ("MST",  {"epsilon": 100.0}),
    ("MST",  {"epsilon": 1000.0}),
    ("AIM",  {"epsilon": 1.0}),
    ("TVAE", {}),
    ("CTGAN", {}),
    ("ARF",  {}),
    ("TabDDPM", {}),
    ("Synthpop", {}),
    ("CellSuppression", {}),
    ("RankSwap", {}),
]

# No AIM for SBO
_SBO_SDGS = [
    ("MST",  {"epsilon": 0.1}),
    ("MST",  {"epsilon": 1.0}),
    ("MST",  {"epsilon": 10.0}),
    ("MST",  {"epsilon": 100.0}),
    ("MST",  {"epsilon": 1000.0}),
    ("TVAE", {}),
    ("CTGAN", {}),
    ("ARF",  {}),
    ("TabDDPM", {}),
    ("Synthpop", {}),
    ("RankSwap", {}),
    ("CellSuppression", {}),
]

_ADULT_MST_EPS_SDGS = [
    ("MST", {"epsilon": eps})
    for eps in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
]

# No CellSuppression for California
_CALI_SDGS = [
    ("MST",     {"epsilon": 0.1}),
    ("MST",     {"epsilon": 1.0}),
    ("MST",     {"epsilon": 10.0}),
    ("MST",     {"epsilon": 100.0}),
    ("MST",     {"epsilon": 1000.0}),
    ("AIM",     {"epsilon": 1.0}),
    ("AIM",     {"epsilon": 3.0}),
    ("AIM",     {"epsilon": 10.0}),
    ("ARF",     {}),
    ("CTGAN",   {}),
    ("TVAE",    {}),
    ("TabDDPM", {}),
    ("Synthpop", {}),
    ("RankSwap", {}),
]


# ── Attack params ─────────────────────────────────────────────────────────────

def _mrf_params() -> dict:
    """Best MarginalRF variant: QI nodes in graph + entropy-weighted BP."""
    return dict(ATTACK_PARAM_DEFAULTS["MarginalRF_graphQI_entropyBP"])


def _mrf_cont_params() -> dict:
    """MarginalRF_continuous defaults (knn_k=None for continuous QI)."""
    return dict(ATTACK_PARAM_DEFAULTS["MarginalRF_continuous"])


# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Job:
    group:        str
    dataset:      str       # QI lookup key (e.g. "cdc_diabetes", "california")
    dataset_base: str       # filesystem dir name (same as dataset here)
    dataset_size: int
    dataset_type: str       # "categorical" | "continuous"
    sample_idx:   int
    sdg_method:   str
    sdg_params:   dict
    attack_method: str      # ATTACK_REGISTRY key AND DB attack_label
    attack_params: dict
    qi:           str
    wandb_group:  str
    memorization: bool = False   # if True, run train + nontraining splits

    @property
    def sdg_label(self) -> str:
        eps = self.sdg_params.get("epsilon") or self.sdg_params.get("eps")
        return f"{self.sdg_method}_eps{eps:g}" if eps is not None else self.sdg_method

    @property
    def data_root(self) -> str:
        return str(DATA_ROOT / self.dataset_base / f"size_{self.dataset_size}")

    @property
    def sample_dir(self) -> str:
        return f"{self.data_root}/sample_{self.sample_idx:02d}"

    @property
    def holdout_dir(self) -> str | None:
        if not self.memorization:
            return None
        holdout_idx = (self.sample_idx + 1) % 5
        return f"{self.data_root}/sample_{holdout_idx:02d}"

    @property
    def run_name(self) -> str:
        return (f"{self.group}_{self.dataset}_{self.dataset_size}"
                f"__s{self.sample_idx:02d}__{self.sdg_label}"
                f"__{self.attack_method}__{self.qi}")


# ── DB-based skip set ─────────────────────────────────────────────────────────

def _load_done_set() -> set[tuple]:
    """
    Return set of (dataset, dataset_size, sample, qi, sdg_method, attack_label, split)
    tuples that are already in results.db.
    """
    import sqlite3
    if not DB_PATH.exists():
        return set()
    con = sqlite3.connect(str(DB_PATH))
    cur = con.execute(
        "SELECT dataset, dataset_size, sample, qi, sdg_method, attack_label, split FROM runs"
    )
    done = {tuple(row) for row in cur.fetchall()}
    con.close()
    return done


def _is_done(job: Job, done: set[tuple]) -> bool:
    """
    A job is done if its expected DB rows already exist.
    - Non-memorization: split='standard' row exists.
    - Memorization: BOTH split='train' AND split='nontraining' rows exist.
    """
    key = (job.dataset, job.dataset_size, job.sample_idx,
           job.qi, job.sdg_label, job.attack_method)
    if job.memorization:
        return ((*key, "train") in done and (*key, "nontraining") in done)
    return (*key, "standard") in done


# ── Job generation ────────────────────────────────────────────────────────────

def generate_jobs(group_filter: str | None = None) -> list[Job]:
    jobs: list[Job] = []

    def _add(group, dataset, dataset_base, size, dtype, sdgs, qi_variants,
             attack, params, samples, wandb_group, memorization=False):
        qi_list = qi_variants if isinstance(qi_variants, list) else [qi_variants]
        for sample_idx, (sdg_method, sdg_params), qi in itertools.product(
            range(samples), sdgs, qi_list
        ):
            jobs.append(Job(
                group=group,
                dataset=dataset,
                dataset_base=dataset_base,
                dataset_size=size,
                dataset_type=dtype,
                sample_idx=sample_idx,
                sdg_method=sdg_method,
                sdg_params=dict(sdg_params),
                attack_method=attack,
                attack_params=dict(params),
                qi=qi,
                wandb_group=wandb_group,
                memorization=memorization,
            ))

    # Priority order: A, E, B, C, D
    # ── Group A: CDC Diabetes 1k ──────────────────────────────────────────
    if group_filter in (None, "A"):
        _add("A", "cdc_diabetes", "cdc_diabetes", 1_000, "categorical",
             _CDC_1K_SDGS, "QI1",
             "MarginalRF_graphQI_entropyBP", _mrf_params(),
             samples=5, wandb_group="marginalrf-cdc-1k")

    # ── Group E: California 1k (continuous, WITH memorization) ───────────
    if group_filter in (None, "E"):
        _add("E", "california", "california", 1_000, "continuous",
             _CALI_SDGS, "QI_large",
             "MarginalRF_continuous", _mrf_cont_params(),
             samples=5, wandb_group="marginalrf-california-continuous-1k",
             memorization=True)

    # ── Group B: CDC Diabetes 100k ─────────────────────────────────────────
    if group_filter in (None, "B"):
        _add("B", "cdc_diabetes", "cdc_diabetes", 100_000, "categorical",
             _CDC_100K_SDGS, "QI1",
             "MarginalRF_graphQI_entropyBP", _mrf_params(),
             samples=5, wandb_group="marginalrf-cdc-100k")

    # ── Group C: NIST SBO 1k ──────────────────────────────────────────────
    if group_filter in (None, "C"):
        _add("C", "nist_sbo", "nist_sbo", 1_000, "categorical",
             _SBO_SDGS, ["QI1", "QI_large"],
             "MarginalRF_graphQI_entropyBP", _mrf_params(),
             samples=5, wandb_group="marginalrf-sbo-1k")

    # ── Group D: Adult 10k MST epsilon sweep ──────────────────────────────
    if group_filter in (None, "D"):
        _add("D", "adult", "adult", 10_000, "categorical",
             _ADULT_MST_EPS_SDGS, ["QI_large", "QI_behavioral"],
             "MarginalRF_graphQI_entropyBP", _mrf_params(),
             samples=4,
             wandb_group="marginalrf-mst-eps-sweep-adult-10k")

    return jobs


# ── Worker ────────────────────────────────────────────────────────────────────

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
    recon = "/home/golobs/Reconstruction"
    if recon in sys.path:
        sys.path.remove(recon)
    sys.path.insert(0, recon)


def run_job(job: Job) -> dict[str, Any]:
    """Run one job and insert results into results.db.  Returns result dict."""
    _worker_setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    import wandb
    from get_data import load_data
    from master_experiment_script import _prepare_config, _run_attack, _score_reconstruction
    from experiment_scripts.results_db import ResultsDB

    effective_params = {
        **ATTACK_PARAM_DEFAULTS.get(job.attack_method, {}),
        **job.attack_params,
    }

    mem_cfg = {"enabled": False}
    if job.holdout_dir is not None:
        mem_cfg = {"enabled": True, "holdout_dir": job.holdout_dir}

    cfg = {
        "dataset": {
            "name": job.dataset,
            "dir":  job.sample_dir,
            "size": job.dataset_size,
            "type": job.dataset_type,
        },
        "QI":          job.qi,
        "data_type":   job.dataset_type,
        "sdg_method":  job.sdg_method,
        "sdg_params":  job.sdg_params or None,
        "attack_method": job.attack_method,
        "memorization_test": mem_cfg,
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            job.attack_method: effective_params,
        },
    }

    prepared = _prepare_config(cfg)

    wandb.init(
        project=WANDB_PROJECT,
        name=job.run_name,
        config={
            "group":         job.group,
            "dataset":       job.dataset,
            "size":          job.dataset_size,
            "sample_idx":    job.sample_idx,
            "sdg_method":    job.sdg_method,
            "sdg_params":    job.sdg_params,
            "attack_method": job.attack_method,
            "attack_params": effective_params,
            "qi":            job.qi,
            "data_type":     job.dataset_type,
            "memorization":  job.memorization,
        },
        tags=[job.dataset, job.attack_method, f"group_{job.group}", "marginalrf-fillin"],
        group=job.wandb_group,
        reinit=True,
    )

    try:
        train, synth, qi_feats, hidden_features, holdout = load_data(prepared)

        # ── Standard / training reconstruction ────────────────────────────
        recon_train  = _run_attack(prepared, synth, train, qi_feats, hidden_features)
        train_scores = _score_reconstruction(train, recon_train, hidden_features, job.dataset_type)
        train_mean   = round(float(np.mean(train_scores)), 4)

        feat_scores_train = {f: round(float(s), 4)
                             for f, s in zip(hidden_features, train_scores)}

        result: dict[str, Any] = {
            "group": job.group, "dataset": job.dataset, "size": job.dataset_size,
            "sample": job.sample_idx, "sdg": job.sdg_label,
            "attack": job.attack_method, "qi": job.qi,
            "ra_mean": train_mean,
            "train_mean": None, "nontrain_mean": None, "delta_mean": None,
            "error": None,
            **{f"RA_{f}": s for f, s in feat_scores_train.items()},
        }

        db = ResultsDB(str(DB_PATH))

        if job.memorization:
            # Memorization mode: insert 'train' and 'nontraining' splits
            wandb.log({f"RA_train_{f}": s for f, s in feat_scores_train.items()})
            wandb.log({"RA_train_mean": train_mean})
            result["train_mean"] = train_mean

            db.insert_run(
                dataset=job.dataset, dataset_size=job.dataset_size,
                sample=job.sample_idx, qi=job.qi, sdg_method=job.sdg_label,
                attack_label=job.attack_method, split="train",
                ra_mean=train_mean, feature_scores=feat_scores_train,
                attack_params=effective_params, sdg_params=job.sdg_params,
                source_file="run_fill_in_marginalrf.py",
            )

            if holdout is not None:
                recon_nt    = _run_attack(prepared, synth, holdout, qi_feats, hidden_features)
                nt_scores   = _score_reconstruction(holdout, recon_nt, hidden_features, job.dataset_type)
                nt_mean     = round(float(np.mean(nt_scores)), 4)
                delta       = round(train_mean - nt_mean, 4)
                feat_scores_nt = {f: round(float(s), 4)
                                  for f, s in zip(hidden_features, nt_scores)}

                wandb.log({f"RA_nontraining_{f}": s for f, s in feat_scores_nt.items()})
                wandb.log({"RA_nontraining_mean": nt_mean, "RA_delta_mean": delta})
                result.update({"nontrain_mean": nt_mean, "delta_mean": delta,
                               **{f"RA_nt_{f}": s for f, s in feat_scores_nt.items()}})

                db.insert_run(
                    dataset=job.dataset, dataset_size=job.dataset_size,
                    sample=job.sample_idx, qi=job.qi, sdg_method=job.sdg_label,
                    attack_label=job.attack_method, split="nontraining",
                    ra_mean=nt_mean, feature_scores=feat_scores_nt,
                    attack_params=effective_params, sdg_params=job.sdg_params,
                    source_file="run_fill_in_marginalrf.py",
                )
        else:
            # Standard mode: insert single 'standard' split
            wandb.log({f"RA_{f}": s for f, s in feat_scores_train.items()})
            wandb.log({"RA_mean": train_mean})

            db.insert_run(
                dataset=job.dataset, dataset_size=job.dataset_size,
                sample=job.sample_idx, qi=job.qi, sdg_method=job.sdg_label,
                attack_label=job.attack_method, split="standard",
                ra_mean=train_mean, feature_scores=feat_scores_train,
                attack_params=effective_params, sdg_params=job.sdg_params,
                source_file="run_fill_in_marginalrf.py",
            )

        return result

    except Exception as exc:
        wandb.log({"error": str(exc)})
        raise
    finally:
        wandb.finish()


# ── Per-group incremental CSV writers ─────────────────────────────────────────

_CSV_BASE_KEYS = [
    "group", "dataset", "size", "sample", "sdg", "attack", "qi",
    "ra_mean", "train_mean", "nontrain_mean", "delta_mean", "error",
]


class GroupCSV:
    """Append-mode CSV writer for one experiment group."""

    def __init__(self, group: str, ts: str):
        fname = Path(__file__).parent / f"marginalrf_fillin_group{group}_{ts}.csv"
        self._path   = fname
        self._lock   = Lock()
        self._header_written = fname.exists() and fname.stat().st_size > 0

    def write(self, row: dict):
        with self._lock:
            feat_keys = sorted(k for k in row if k.startswith("RA_") and k not in ("RA_mean",))
            nt_keys   = sorted(k for k in row if k.startswith("RA_nt_"))
            fieldnames = _CSV_BASE_KEYS + feat_keys + nt_keys
            with open(self._path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore", restval="")
                if not self._header_written:
                    writer.writeheader()
                    self._header_written = True
                writer.writerow(row)

    @property
    def path(self) -> Path:
        return self._path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fill in MarginalRF_graphQI_entropyBP results for the manuscript."
    )
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print planned jobs and exit (skips DB check).")
    parser.add_argument("--serial",   action="store_true",
                        help="Run in main process (Ctrl-C killable).")
    parser.add_argument("--workers",  type=int, default=N_WORKERS)
    parser.add_argument("--group",    type=str, default=None, choices=list("ABCDE"),
                        help="Run only this group.")
    parser.add_argument("--sample",   type=int, default=None)
    parser.add_argument("--sdg",      type=str, default=None)
    parser.add_argument("--qi",       type=str, default=None)
    parser.add_argument("--force",    action="store_true",
                        help="Run all jobs even if already in DB.")
    parser.add_argument("--progress-log", type=str,
                        default=str(REPO_ROOT / "outfiles" / "marginalrf_fillin_progress.log"),
                        metavar="FILE")
    args = parser.parse_args()

    # ── Progress logger ────────────────────────────────────────────────────
    Path(args.progress_log).parent.mkdir(parents=True, exist_ok=True)
    _plog = open(args.progress_log, "w", buffering=1)

    def _log(msg: str):
        ts   = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        _plog.write(line + "\n")

    all_jobs = generate_jobs(args.group)

    if args.sample is not None:
        all_jobs = [j for j in all_jobs if j.sample_idx == args.sample]
    if args.sdg is not None:
        all_jobs = [j for j in all_jobs
                    if j.sdg_label == args.sdg or j.sdg_method == args.sdg]
    if args.qi is not None:
        all_jobs = [j for j in all_jobs if j.qi == args.qi]

    if args.dry_run:
        _log("DRY RUN — skipping DB check")
        for j in all_jobs:
            print(f"  [{j.group}] {j.dataset}/{j.dataset_size}"
                  f"  s{j.sample_idx:02d}  {j.sdg_label:<20}  {j.attack_method}  {j.qi}"
                  f"{'  +mem' if j.memorization else ''}")
        _log(f"Total jobs (before DB skip): {len(all_jobs)}")
        _plog.close()
        return

    # ── DB-based deduplication ────────────────────────────────────────────
    if args.force:
        pending_jobs = all_jobs
        _log(f"--force: skipping DB check, running all {len(all_jobs)} jobs")
    else:
        done = _load_done_set()
        pending_jobs = [j for j in all_jobs if not _is_done(j, done)]
        n_skipped = len(all_jobs) - len(pending_jobs)
        _log(f"DB check: {len(all_jobs)} total, {n_skipped} already done, "
             f"{len(pending_jobs)} to run")

    _log("=" * 70)
    _log("  run_fill_in_marginalrf.py")
    _log(f"  Jobs to run : {len(pending_jobs)}")
    _log(f"  Workers     : {args.workers}")
    _log(f"  Progress log: {args.progress_log}")
    _log(f"  Group filter: {args.group or 'all'}")
    _log("=" * 70)
    for grp, cnt in sorted(Counter(j.group for j in pending_jobs).items()):
        _log(f"  Group {grp}: {cnt} jobs")
    _log("")

    if not pending_jobs:
        _log("Nothing to do. All jobs already in DB.")
        _plog.close()
        return

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_by_group: dict[str, GroupCSV] = {}
    for grp in sorted({j.group for j in pending_jobs}):
        csv_by_group[grp] = GroupCSV(grp, ts)

    ok = fail = 0

    def _handle_result(job: Job, result: dict | None, err: str | None):
        nonlocal ok, fail
        if err:
            last_line = err.splitlines()[-1][:200]
            _log(f"FAIL  [{job.group}] {job.run_name}: {last_line}")
            csv_by_group[job.group].write({
                "group": job.group, "dataset": job.dataset, "size": job.dataset_size,
                "sample": job.sample_idx, "sdg": job.sdg_label,
                "attack": job.attack_method, "qi": job.qi,
                "ra_mean": None, "error": last_line,
            })
            fail += 1
        else:
            ra = result.get("ra_mean") or result.get("train_mean")
            _log(f"OK    [{job.group}] {job.run_name}  RA={ra:.4f}")
            csv_by_group[job.group].write(result)
            ok += 1

    if args.serial:
        for job in pending_jobs:
            try:
                result = run_job(job)
                _handle_result(job, result, None)
            except Exception:
                _handle_result(job, None, traceback.format_exc())
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_job, job): job for job in pending_jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    _handle_result(job, fut.result(), None)
                except Exception:
                    _handle_result(job, None, traceback.format_exc())

    # ── Summary ───────────────────────────────────────────────────────────
    _log("=" * 70)
    _log(f"  DONE — {ok} ok / {fail} failed")
    for grp, gcsv in sorted(csv_by_group.items()):
        _log(f"  Group {grp} CSV: {gcsv.path}")
    _log("=" * 70)
    _plog.close()


if __name__ == "__main__":
    main()
