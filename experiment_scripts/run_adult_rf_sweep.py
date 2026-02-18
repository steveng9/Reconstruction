#!/usr/bin/env python
"""
Quick sweep: RandomForest reconstruction attack (plain + chained) on all
adult size_10k/sample_00 SDG methods, with memorization test.

Intended for one-off results — not for final paper runs.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_adult_rf_sweep.py

Results logged to WandB under project "tabular-reconstruction-attacks",
group "adult_RF_sweep".
"""

import sys
import numpy as np

sys.path.insert(0, "/home/golobs/Reconstruction")
sys.path.append("/home/golobs/MIA_on_diffusion/")
sys.path.append("/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM")
sys.path.append("/home/golobs/recon-synth")
sys.path.append("/home/golobs/recon-synth/attacks")
sys.path.append("/home/golobs/recon-synth/attacks/solvers")

import wandb
from get_data import load_data
from master_experiment_script import _run_attack, _score_reconstruction, _prepare_config


def sdg_dirname(method, params=None):
    """Derive canonical SDG output directory name (mirrors sdg/__init__.py)."""
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    if eps is not None:
        return f"{method}_eps{eps:g}"
    return method


# ── Data paths ────────────────────────────────────────────────────────────────

DATA_ROOT  = "/home/golobs/data/reconstruction_data/adult/size_10000"
SAMPLE_DIR = f"{DATA_ROOT}/sample_00"
HOLDOUT_DIR = f"{DATA_ROOT}/sample_01"   # disjoint sample for memorization test


# ── SDG methods present in sample_00 ─────────────────────────────────────────
# (method_name, sdg_params) — params only need epsilon for dirname resolution;
# key_vars / swap_features not needed here since we're reading pre-generated synth.

SDG_METHODS = [
    ("TabDDPM",       {}),
    ("TVAE",          {}),
    ("CTGAN",         {}),
    ("ARF",           {}),
    ("MST",           {"epsilon": 1.0}),
    ("MST",           {"epsilon": 10.0}),
    ("MST",           {"epsilon": 100.0}),
    ("MST",           {"epsilon": 1000.0}),
    ("AIM",           {"epsilon": 1.0}),
    ("Synthpop",      {}),
    ("RankSwap",      {}),
    ("CellSuppression", {}),
]

RF_PARAMS = {
    "max_depth": 25,
    "num_estimators": 25,
}


# ── Config builder ────────────────────────────────────────────────────────────

def make_config(sdg_method, sdg_params, chaining: bool) -> dict:
    return {
        "dataset": {
            "name": "adult",
            "dir": SAMPLE_DIR,
            "size": 10000,
            "type": "categorical",
        },
        "QI": "QI1",
        "data_type": "categorical",
        "attack_method": "RandomForest",
        "sdg_method": sdg_method,
        "sdg_params": sdg_params or None,
        "memorization_test": {
            "enabled": True,
            "holdout_dir": HOLDOUT_DIR,
        },
        "attack_params": {
            # Enhancement sub-dicts (kept by _prepare_config logic below)
            "ensembling": {"enabled": False},
            "chaining": {
                "enabled": chaining,
                "order_strategy": "mutual_info",
                "log_intermediate": False,
                "random_seed": 42,
            },
            # Method-specific params (merged into attack_params by _prepare_config)
            "RandomForest": RF_PARAMS,
        },
    }


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(sdg_method: str, sdg_params: dict, chaining: bool):
    """Run one attack configuration. Returns summary dict, or None on error."""
    sdg_label = sdg_dirname(sdg_method, sdg_params)
    run_name = f"RF_{'chained' if chaining else 'plain'}_{sdg_label}"

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    cfg = make_config(sdg_method, sdg_params, chaining)
    prepared = _prepare_config(cfg)

    wandb.init(
        project="tabular-reconstruction-attacks",
        name=run_name,
        config=cfg,
        tags=["adult", "size_10000", "quick_results", "RF_sweep"],
        group="adult_RF_sweep",
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        print("  Reconstructing training targets...")
        reconstructed_train   = _run_attack(prepared, synth, train,   qi, hidden_features)
        train_scores          = _score_reconstruction(train,   reconstructed_train,   hidden_features, "categorical")

        print("  Reconstructing holdout (non-training) targets...")
        reconstructed_holdout = _run_attack(prepared, synth, holdout, qi, hidden_features)
        holdout_scores        = _score_reconstruction(holdout, reconstructed_holdout, hidden_features, "categorical")

        results = {}
        for feat, ts, hs in zip(hidden_features, train_scores, holdout_scores):
            results[f"RA_train_{feat}"]       = ts
            results[f"RA_nontraining_{feat}"] = hs
            results[f"RA_delta_{feat}"]       = round(ts - hs, 2)

        train_mean    = round(float(np.mean(train_scores)), 2)
        holdout_mean  = round(float(np.mean(holdout_scores)), 2)
        delta_mean    = round(train_mean - holdout_mean, 2)

        results["RA_train_mean"]       = train_mean
        results["RA_nontraining_mean"] = holdout_mean
        results["RA_delta_mean"]       = delta_mean

        print(f"\n  --- Results ---")
        print(f"  train_mean    = {train_mean:.3f}")
        print(f"  nontrain_mean = {holdout_mean:.3f}")
        print(f"  delta_mean    = {delta_mean:+.3f}")

        wandb.log(results)

        return {
            "sdg":      sdg_label,
            "chained":  chaining,
            "train":    train_mean,
            "nontrain": holdout_mean,
            "delta":    delta_mean,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        raise

    finally:
        wandb.finish()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(rows):
    """Print a formatted summary table of all sweep results."""
    if not rows:
        return

    # Column widths
    w_sdg     = max(len(r["sdg"]) for r in rows)
    w_sdg     = max(w_sdg, len("SDG method"))
    col_w     = 9   # width for numeric columns

    def fmt_row(sdg, chained, train, nontrain, delta):
        chain_str = "yes" if chained else "no"
        return (
            f"  {sdg:<{w_sdg}}  {chain_str:^7}  "
            f"{train:>{col_w}}  {nontrain:>{col_w}}  {delta:>{col_w}}"
        )

    sep = "  " + "-" * (w_sdg + 7 + col_w * 3 + 10)
    header = (
        f"  {'SDG method':<{w_sdg}}  {'chained':^7}  "
        f"{'train':>{col_w}}  {'nontrain':>{col_w}}  {'delta':>{col_w}}"
    )

    print(f"\n\n{'='*70}")
    print("  SWEEP SUMMARY  —  RF attack on adult size_10k/sample_00")
    print(f"  QI: age, sex, race, native-country, education, marital-status")
    print(f"  Hidden: workclass, fnlwgt, educ-num, occupation, relationship,")
    print(f"          capital-gain, capital-loss, hours-per-week, income")
    print(f"  Scores: rarity-weighted accuracy (higher = better attack)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    prev_sdg = None
    for r in rows:
        if prev_sdg and r["sdg"] != prev_sdg:
            print(sep)   # blank separator between SDG groups
        print(fmt_row(r["sdg"], r["chained"], r["train"], r["nontrain"], r["delta"]))
        prev_sdg = r["sdg"]

    print(sep)

    # Aggregate averages by chaining
    for chained in [False, True]:
        subset = [r for r in rows if r["chained"] == chained]
        if subset:
            avg_train    = round(float(np.mean([r["train"]    for r in subset])), 2)
            avg_nontrain = round(float(np.mean([r["nontrain"] for r in subset])), 2)
            avg_delta    = round(float(np.mean([r["delta"]    for r in subset])), 2)
            label = f"AVG ({'chained' if chained else 'plain'})"
            print(fmt_row(label, chained, avg_train, avg_nontrain, avg_delta))

    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total = len(SDG_METHODS) * 2   # plain + chained
    done = 0
    all_results = []

    for sdg_method, sdg_params in SDG_METHODS:
        for chaining in [False, True]:
            done += 1
            print(f"\n[{done}/{total}]", end="")
            summary = run_experiment(sdg_method, sdg_params, chaining)
            if summary:
                all_results.append(summary)

    print_summary_table(all_results)
    print(f"All {total} runs complete.")
