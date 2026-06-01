#!/usr/bin/env python
"""
run_adult10k_mst_eps01_attacks.py

Re-run ALL Table 1 attacks on adult 10k, QI1, MST eps=0.1 — using the
freshly-regenerated (integer-encoded) synth.csv produced by
regen_mst_adult10k.py.

Why only MST eps=0.1?
  - eps=1,10,100,1000 were already generated with bin_continuous_as_ordinal=True
    and the attack results in the manuscript table are correct.
  - eps=0.1 was generated WITHOUT bin_continuous_as_ordinal, producing float
    bin-midpoints that suppressed Mode/Random baselines to 9.5 (instead of ~10.5)
    and depressed all ML attack scores.  After regen they should match the
    encoding of the other MST columns.

Attacks included (all Table 1 attacks except SVM, which is O(n²) at n=10k):
  Baselines:   Mode, Random, MeasureDeid
  ML:          KNN, NaiveBayes, LogisticRegression, RandomForest, LightGBM
  Neural:      MLP, ARFFormer
  Diffusion:   CondDDPM (retrain=True), CondRePaint (retrain=True),
               CondDDPMWithMLP (retrain=True)
  CondMST:  CondMST (retrain=True), CondMSTIndependent (retrain=True),
               CondMSTBounded k=3 (retrain=True)
  CoBP-RA:  CoBP-RA, CoBP-RA (QI graph), CoBP-RA (QI+entropy BP)

WandB group: "mst-corrected-adult-10k"   (distinct from "main attack sweep 1")

Run in background:
    nohup conda run -n recon_ python experiment_scripts/run_adult10k_mst_eps01_attacks.py \\
        --workers 8 \\
        >experiment_scripts/outfiles/mst_corrected_attacks.log 2>&1 &
    echo PID=$!

Monitor:
    tail -f experiment_scripts/outfiles/mst_corrected_attacks.log
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, "/home/golobs/Reconstruction")
from attack_defaults import ATTACK_PARAM_DEFAULTS

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_NAME  = "adult"
DATASET_SIZE  = 10_000
DATASET_TYPE  = "categorical"
DATA_ROOT     = Path("/home/golobs/data/reconstruction_data/adult/size_10000")
N_SAMPLES     = 5
QI_VARIANTS   = ["QI1"]

SDG_METHODS   = [("MST", {"epsilon": 0.1})]   # only the corrected epsilon

_MRF = ATTACK_PARAM_DEFAULTS.get("CoBP-RA", {})
_TAB = ATTACK_PARAM_DEFAULTS.get("CondDDPM", {})
_REP = ATTACK_PARAM_DEFAULTS.get("CondRePaint", {})
_MLP = ATTACK_PARAM_DEFAULTS.get("CondDDPMWithMLP", {})

ATTACK_CONFIGS = [
    # ── Baselines ──────────────────────────────────────────────────────────
    ("Mode",        "Mode",        {}),
    ("Random",      "Random",      {}),
    ("MeasureDeid", "MeasureDeid", {}),
    # ── ML classifiers ─────────────────────────────────────────────────────
    ("KNN",               "KNN",               {}),
    ("NaiveBayes",        "NaiveBayes",         {}),
    ("LogisticRegression","LogisticRegression", {}),
    # SVM excluded: O(n²–n³) too slow at n=10k
    ("RandomForest",      "RandomForest",       {}),
    ("LightGBM",          "LightGBM",           {}),
    # ── Neural ─────────────────────────────────────────────────────────────
    ("MLP",               "MLP",               {}),
    ("ARFFormer",         "ARFFormer",          {}),
    # ── Diffusion (retrain=True because underlying synth changed) ──────────
    # ORDERING: CondDDPM first so CondRePaint can reuse its checkpoint.
    # Both are set retrain=True to force fresh models on the new synth.
    ("CondDDPM",          "CondDDPM",          {**_TAB, "retrain": True}),
    ("CondRePaint","CondRePaint", {**_REP, "retrain": True}),
    ("CondDDPMWithMLP",    "CondDDPMWithMLP",     {**_MLP, "retrain": True}),
    # ── CondMST (retrain=True: model fit on synth, must be redone) ──────
    ("CondMST",
     "CondMST",
     {**ATTACK_PARAM_DEFAULTS.get("CondMST", {}), "retrain": True}),
    ("CondMST_Indep",
     "CondMSTIndependent",
     {**ATTACK_PARAM_DEFAULTS.get("CondMSTIndependent", {}), "retrain": True}),
    ("CondMSTBounded_k3",
     "CondMSTBounded",
     {**ATTACK_PARAM_DEFAULTS.get("CondMSTBounded", {}), "retrain": True, "max_clique_size": 3}),
    # ── CoBP-RA ──────────────────────────────────────────────────────────
    ("CoBP-RA",
     "CoBP-RA",
     {**_MRF, "qi_in_graph": False, "entropy_weighted": False}),
    ("CoBP-RA_QIGraph",
     "CoBP-RA",
     {**_MRF, "qi_in_graph": True, "entropy_weighted": False}),
    ("CoBP-RA_QIGraph_EntropyBP",
     "CoBP-RA",
     {**_MRF, "qi_in_graph": True, "entropy_weighted": True}),
]

N_WORKERS     = 8
WANDB_PROJECT = "tabular-reconstruction-attacks"
WANDB_GROUP   = "mst-corrected-adult-10k"


# ── Job spec ──────────────────────────────────────────────────────────────────

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
        return str(DATA_ROOT / f"sample_{self.sample_idx:02d}")

    @property
    def run_name(self) -> str:
        return (f"{DATASET_NAME}__sz{DATASET_SIZE}__s{self.sample_idx:02d}__"
                f"{self.sdg_label}__{self.attack_label}__{self.qi}")


def generate_jobs() -> list[Job]:
    jobs = []
    for sample_idx, (sdg_method, sdg_params), (label, method, params), qi in itertools.product(
        range(N_SAMPLES), SDG_METHODS, ATTACK_CONFIGS, QI_VARIANTS
    ):
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

def _setup_paths():
    for p in [
        "/home/golobs/MIA_on_diffusion/",
        "/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM",
        "/home/golobs/recon-synth",
        "/home/golobs/recon-synth/attacks",
        "/home/golobs/recon-synth/attacks/solvers",
    ]:
        if p not in sys.path:
            sys.path.append(p)
    r = "/home/golobs/Reconstruction"
    if r in sys.path:
        sys.path.remove(r)
    sys.path.insert(0, r)


def run_job(job: Job) -> dict[str, Any]:
    _setup_paths()
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
        "QI":            job.qi,
        "data_type":     DATASET_TYPE,
        "sdg_method":    job.sdg_method,
        "sdg_params":    job.sdg_params or None,
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
        group=WANDB_GROUP,
        name=job.run_name,
        config={
            "sample_idx":    job.sample_idx,
            "dataset":       DATASET_NAME,
            "size":          DATASET_SIZE,
            "sdg_method":    job.sdg_method,
            "epsilon":       job.sdg_params.get("epsilon"),
            "attack_method": job.attack_method,
            "attack_label":  job.attack_label,
            "qi":            job.qi,
        },
        reinit=True,
    )
    try:
        train_df, synth_df, qi_features, hidden_features, _ = load_data(prepared)
        recon_df = _run_attack(prepared, synth_df, train_df, qi_features, hidden_features)
        scores = _score_reconstruction(train_df, recon_df, hidden_features, DATASET_TYPE)
        metrics = {f"RA_{feat}": s for feat, s in zip(hidden_features, scores)}
        ra_mean = round(float(np.mean(scores)), 4)
        wandb.log({"RA_mean": ra_mean, **metrics})

        return {
            "sample": f"sample_{job.sample_idx:02d}",
            "sdg": job.sdg_label,
            "attack": job.attack_method,
            "label": job.attack_label,
            "qi": job.qi,
            "ra_mean": ra_mean,
            "error": None,
            **metrics,
        }
    except Exception as e:
        tb = traceback.format_exc()
        wandb.log({"error": str(e)})
        print(f"  ERROR in {job.run_name}: {e}\n{tb}", flush=True)
        return {
            "sample": f"sample_{job.sample_idx:02d}",
            "sdg": job.sdg_label,
            "attack": job.attack_method,
            "label": job.attack_label,
            "qi": job.qi,
            "ra_mean": None,
            "error": str(e),
        }
    finally:
        wandb.finish(quiet=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rerun all Table 1 attacks on MST eps=0.1 (corrected synth).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--serial",  action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--attack",  type=str, default=None,
                        help="Filter to one attack label")
    args = parser.parse_args()

    jobs = generate_jobs()
    if args.attack:
        jobs = [j for j in jobs if j.attack_label == args.attack]

    print(f"  WandB group  : {WANDB_GROUP}")
    print(f"  Total jobs   : {len(jobs)}")
    print(f"  Attacks      : {sorted({j.attack_label for j in jobs})}")
    print(f"  SDG          : MST_eps0.1 (corrected, bin_continuous_as_ordinal=True)")
    print(f"  NOTE: Diffusion attacks (CondDDPM/CondRePaint/CondDDPMWithMLP)")
    print(f"        train from scratch and take ~1-2 hrs per sample.")

    if args.dry_run:
        for j in jobs:
            print(f"    {j.run_name}")
        print("[dry-run] no jobs executed.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"mst_corrected_attacks_{ts}.csv"
    print(f"  Output CSV   : {out_path}\n", flush=True)

    rows_written = 0

    def _write_row(row: dict):
        nonlocal rows_written
        fields = ["sample", "sdg", "attack", "label", "qi", "ra_mean", "error"]
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if rows_written == 0:
                writer.writeheader()
            writer.writerow(row)
        rows_written += 1

    if args.serial:
        for job in jobs:
            result = run_job(job)
            _write_row(result)
            ra = result.get("ra_mean")
            err = result.get("error")
            if ra is not None:
                print(f"  {result['label']:40s} s{int(result['sample'][-2:]):02d}  "
                      f"RA={ra:.3f}", flush=True)
            else:
                print(f"  {result['label']:40s} ERROR: {err}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = {ex.submit(run_job, j): j for j in jobs}
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    result = fut.result()
                except Exception as e:
                    j = futures[fut]
                    result = {"sample": f"sample_{j.sample_idx:02d}",
                              "sdg": j.sdg_label, "attack": j.attack_method,
                              "label": j.attack_label, "qi": j.qi,
                              "ra_mean": None, "error": str(e)}
                _write_row(result)
                ra = result.get("ra_mean")
                err = result.get("error")
                if ra is not None:
                    print(f"  [{done:3d}/{len(jobs)}] {result['label']:40s}  "
                          f"s{int(result['sample'][-2:]):02d}  RA={ra:.3f}", flush=True)
                else:
                    print(f"  [{done:3d}/{len(jobs)}] {result['label']:40s}  "
                          f"ERROR: {err}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(out_path)
    ok = df[df["ra_mean"].notna()]
    if not ok.empty:
        print("\n  ── Mean RA by attack, averaged over 5 samples ─────────────")
        print(ok.groupby("label")["ra_mean"].mean().sort_values(ascending=False).to_string())
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
