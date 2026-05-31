#!/usr/bin/env python
"""
Save per-attack probability distributions to disk for ensemble grid search.

Runs each of 5 attacks once per (sample, SDG), saves pickles to:
  outfiles/probas/sample_{N:02d}/{sdg_label}/{attack_label}.pkl

Each pickle contains:
  {
    "probas":          list of np.ndarray (n_targets, n_classes) per hidden feature,
    "classes":         list of np.ndarray per hidden feature,
    "recon":           pd.DataFrame (reconstructed hidden features),
    "hidden_features": list[str],
    "ra_mean":         float,
    "qi":              str,
    "sdg_label":       str,
    "sample_idx":      int,
    "attack_label":    str,
  }

Existing pickles are skipped (safe to re-run after partial failure).
After this script completes, run ensemble_grid_search.py to sweep weight
combinations without re-running any attack.

Usage:
    conda activate recon_
    python experiment_scripts/save_attack_probas.py
    python experiment_scripts/save_attack_probas.py --workers 4
    python experiment_scripts/save_attack_probas.py --sdg TabDDPM --sample 0
    python experiment_scripts/save_attack_probas.py --force   # overwrite existing
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import multiprocessing as mp

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS


# ── Dataset configuration ──────────────────────────────────────────────────────

DATASET_NAME = "adult"
DATASET_SIZE = 10_000
DATASET_TYPE = "categorical"
DATA_ROOT    = f"/home/golobs/data/reconstruction_data/adult/size_{DATASET_SIZE}"
QI_VARIANTS  = ["QI1"]
SAMPLE_RANGE = list(range(5))
PROBAS_DIR   = Path(__file__).parent.parent / "outfiles" / "probas"


# ── SDG methods ────────────────────────────────────────────────────────────────

SDG_METHODS = [
    ("MST",      {"epsilon": 1.0}),
    ("MST",      {"epsilon": 10.0}),
    ("TVAE",     {}),
    ("Synthpop", {}),
    ("TabDDPM",  {}),
]


# ── Individual attacks ─────────────────────────────────────────────────────────

INDIVIDUAL_ATTACKS = [
    ("MarginalRF", "MarginalRF", {
        "chaining":   {"enabled": False},
        "ensembling": {"enabled": False},
        "MarginalRF": dict(ATTACK_PARAM_DEFAULTS["MarginalRF"]),
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

N_WORKERS = 3


# ── Job ────────────────────────────────────────────────────────────────────────

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

    @property
    def out_dir(self) -> Path:
        return PROBAS_DIR / f"sample_{self.sample_idx:02d}" / self.sdg_label


def generate_jobs(sdg_filter=None, sample_filter=None) -> list[Job]:
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


def run_job(job: Job, force: bool = False) -> dict[str, Any]:
    """Run all attacks for one (sample, SDG), pickle each result."""
    _worker_setup_paths()
    sys.argv = sys.argv[:1]

    import numpy as np
    from get_data import load_data
    from master_experiment_script import _prepare_config
    from scoring import calculate_reconstruction_score
    from attacks import get_attack

    cfg_base = {
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
        "attack_method": "MarginalRF",          # placeholder, overridden per attack
        "memorization_test": {"enabled": False},
        "attack_params": {},
    }

    prepared_base = _prepare_config(dict(cfg_base))
    train, synth, qi_cols, hidden_features, _holdout = load_data(prepared_base)

    out_dir = job.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for atk_label, atk_method, atk_params in INDIVIDUAL_ATTACKS:
        out_path = out_dir / f"{atk_label}.pkl"

        if out_path.exists() and not force:
            with open(out_path, "rb") as f:
                cached = pickle.load(f)
            results[atk_label] = cached["ra_mean"]
            print(f"    [skip] {atk_label}={cached['ra_mean']:.4f}  (cached)", flush=True)
            continue

        try:
            cfg = dict(cfg_base)
            cfg["attack_method"] = atk_method
            cfg["attack_params"] = dict(atk_params)
            prepared = _prepare_config(cfg)
            attack_fn = get_attack(atk_method, DATASET_TYPE)
            recon, probas, classes = attack_fn(prepared, synth, train, qi_cols, hidden_features)

            scores  = calculate_reconstruction_score(train, recon, hidden_features)
            ra_mean = float(np.mean(scores))

            payload = {
                "probas":          probas,
                "classes":         classes,
                "recon":           recon,
                "hidden_features": hidden_features,
                "ra_mean":         ra_mean,
                "qi":              job.qi,
                "sdg_label":       job.sdg_label,
                "sample_idx":      job.sample_idx,
                "attack_label":    atk_label,
            }
            with open(out_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            results[atk_label] = ra_mean
            print(f"    [{atk_label}] RA={ra_mean:.4f}  → saved", flush=True)

        except Exception:
            print(f"\n[ERROR] {job.sdg_label}/s{job.sample_idx:02d}/{atk_label}:\n"
                  f"{traceback.format_exc()}", flush=True)
            results[atk_label] = None

    return {"job": f"s{job.sample_idx:02d}/{job.sdg_label}", "results": results}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Save attack probability distributions to disk.")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--serial",   action="store_true",
                        help="Run in main process (Ctrl-C killable).")
    parser.add_argument("--workers",  type=int, default=N_WORKERS)
    parser.add_argument("--sdg",      type=str, default=None)
    parser.add_argument("--sample",   type=int, default=None)
    parser.add_argument("--force",    action="store_true",
                        help="Overwrite existing pickles.")
    parser.add_argument("--progress-log", type=str,
                        default=str(Path(__file__).parent.parent / "outfiles" /
                                    "save_probas.log"))
    args = parser.parse_args()

    all_jobs = generate_jobs(sdg_filter=args.sdg, sample_filter=args.sample)
    n_total  = len(all_jobs)

    header = (
        f"{'='*70}\n"
        f"  save_attack_probas\n"
        f"  Jobs:    {n_total} ({len(INDIVIDUAL_ATTACKS)} attacks each)\n"
        f"  Output:  {PROBAS_DIR}\n"
        f"  Workers: {args.workers}\n"
        f"  Force:   {args.force}\n"
        f"{'='*70}\n"
    )
    print(header)

    log_path = Path(args.progress_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for j in all_jobs:
            print(f"  s{j.sample_idx:02d}/{j.sdg_label}")
        return

    with open(log_path, "w") as f:
        f.write(header)

    def _fmt_result(result: dict, elapsed: float, done: int, eta: float | None) -> str:
        ra_parts = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in result["results"].items()
        )
        eta_str = f"ETA {int(eta // 60)}m" if eta is not None else ""
        w = len(str(n_total))
        return f"  [{done:>{w}}/{n_total}]  {result['job']:<32}  {ra_parts}  {eta_str}"

    if args.serial:
        t0 = time.monotonic()
        for idx, job in enumerate(all_jobs):
            result  = run_job(job, force=args.force)
            elapsed = time.monotonic() - t0
            done    = idx + 1
            eta     = (elapsed / done) * (n_total - done)
            msg     = _fmt_result(result, elapsed, done, eta)
            print(msg)
            with open(log_path, "a") as f:
                f.write(msg + "\n")
    else:
        ctx  = mp.get_context("spawn")
        t0   = time.monotonic()
        done = 0

        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
            futures = {executor.submit(run_job, job, args.force): job for job in all_jobs}
            for future in as_completed(futures):
                done   += 1
                elapsed = time.monotonic() - t0
                eta     = (elapsed / done) * (n_total - done) if done > 0 else None
                try:
                    result = future.result()
                    msg    = _fmt_result(result, elapsed, done, eta)
                except Exception as exc:
                    msg = f"  [{done}/{n_total}]  ERROR: {exc}"
                print(msg)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")

    print(f"\nDone. Pickles in {PROBAS_DIR}")
    print(f"Next: python experiment_scripts/ensemble_grid_search.py")


if __name__ == "__main__":
    main()
