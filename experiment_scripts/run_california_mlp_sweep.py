#!/usr/bin/env python
"""
Quick sweep: MLP reconstruction attack + random baseline on all
california size_1k SDG methods, averaged over 5 training samples.

Intended for one-off results — not for final paper runs.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_california_mlp_sweep.py

Results logged to WandB under project "tabular-reconstruction-attacks",
group "california_MLP_sweep".

Scoring: RMSE normalized by the full dataset's feature range, so the
denominator is identical across all samples and folds.
Per-feature absolute RMSE is also printed at the end of each run.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/golobs/Reconstruction")
sys.path.append("/home/golobs/MIA_on_diffusion/")
sys.path.append("/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM")
sys.path.append("/home/golobs/recon-synth")
sys.path.append("/home/golobs/recon-synth/attacks")
sys.path.append("/home/golobs/recon-synth/attacks/solvers")

import wandb
from get_data import load_data
from master_experiment_script import _run_attack, _prepare_config


def sdg_dirname(method, params=None):
    """Derive canonical SDG output directory name (mirrors sdg/__init__.py)."""
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    if eps is not None:
        return f"{method}_eps{eps:g}"
    return method


# ── Data paths ────────────────────────────────────────────────────────────────

DATA_ROOT    = "/home/golobs/data/reconstruction_data/california/size_1000"
FULL_DATA_PATH = "/home/golobs/data/reconstruction_data/california/full_data.csv"

# 5 training samples paired with 5 non-overlapping holdout samples
TRAIN_SAMPLES   = [0, 1, 2, 3, 4]
HOLDOUT_SAMPLES = [5, 6, 7, 8, 9]   # paired: train[i] uses holdout[i]

QI_FEATURES     = ["Latitude", "Longitude", "HouseAge", "Population"]
HIDDEN_FEATURES = ["MedInc", "AveRooms", "AveBedrms", "AveOccup", "MedHouseVal"]


# ── Load full-dataset feature ranges for consistent normalization ─────────────

_full_df = pd.read_csv(FULL_DATA_PATH)
FULL_RANGES = {
    col: float(_full_df[col].max() - _full_df[col].min())
    for col in HIDDEN_FEATURES
}
print("Full-dataset ranges used for normalization:")
for col, r in FULL_RANGES.items():
    print(f"  {col}: {r:.4f}")
print()


# ── SDG methods present across all samples ───────────────────────────────────

SDG_METHODS = [
    ("TabDDPM",  {}),
    ("TVAE",     {}),
    ("CTGAN",    {}),
    ("ARF",      {}),
    ("MST",      {"epsilon": 1.0}),
    ("MST",      {"epsilon": 10.0}),
    ("MST",      {"epsilon": 100.0}),
    ("MST",      {"epsilon": 1000.0}),
    ("AIM",      {"epsilon": 1.0}),
    ("AIM",      {"epsilon": 10.0}),
    ("Synthpop", {}),
    ("RankSwap", {}),
]

ATTACK_METHODS = ["MLP", "Random"]

MLP_PARAMS = {
    "hidden_dims":   [128, 96, 64],
    "epochs":        250,
    "learning_rate": 0.0003,
    "batch_size":    264,
    "dropout_rate":  0.2,
    "patience":      50,
    "test_size":     0.2,
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def _per_feature_rmse(original, reconstructed):
    """Absolute RMSE per hidden feature (in the feature's own units)."""
    return {
        feat: float(np.sqrt(np.mean((original[feat].values - reconstructed[feat].values) ** 2)))
        for feat in HIDDEN_FEATURES
    }


def _mean_norm_rmse(per_feature_rmse: dict) -> float:
    """Mean RMSE normalized by full-dataset range — comparable across features."""
    return round(float(np.mean([
        per_feature_rmse[feat] / FULL_RANGES[feat]
        for feat in HIDDEN_FEATURES
    ])), 4)


# ── Config builder ────────────────────────────────────────────────────────────

def make_config(sdg_method, sdg_params, attack_method, sample_idx) -> dict:
    sample_dir  = f"{DATA_ROOT}/sample_{sample_idx:02d}"
    holdout_dir = f"{DATA_ROOT}/sample_{HOLDOUT_SAMPLES[sample_idx]:02d}"
    return {
        "dataset": {
            "name": "california",
            "dir":  sample_dir,
            "size": 1000,
            "type": "continuous",
        },
        "QI":           "QI1",
        "data_type":    "continuous",
        "attack_method": attack_method,
        "sdg_method":   sdg_method,
        "sdg_params":   sdg_params or None,
        "memorization_test": {
            "enabled":     True,
            "holdout_dir": holdout_dir,
        },
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            "MLP":    MLP_PARAMS,
            "Random": {},
        },
    }


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(sdg_method, sdg_params, attack_method, sample_idx):
    """Run one attack configuration on one sample. Returns summary dict or None."""
    sdg_label  = sdg_dirname(sdg_method, sdg_params)
    run_name   = f"{attack_method}_{sdg_label}_s{sample_idx:02d}"

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    cfg      = make_config(sdg_method, sdg_params, attack_method, sample_idx)
    prepared = _prepare_config(cfg)

    wandb.init(
        project="tabular-reconstruction-attacks",
        name=run_name,
        config=cfg,
        tags=["california", "size_1000", "quick_results", "MLP_sweep"],
        group="california_MLP_sweep",
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        print("  Reconstructing training targets...")
        recon_train   = _run_attack(prepared, synth, train,   qi, hidden_features)

        print("  Reconstructing holdout (non-training) targets...")
        recon_holdout = _run_attack(prepared, synth, holdout, qi, hidden_features)

        train_feat   = _per_feature_rmse(train,   recon_train)
        holdout_feat = _per_feature_rmse(holdout, recon_holdout)

        train_mean   = _mean_norm_rmse(train_feat)
        holdout_mean = _mean_norm_rmse(holdout_feat)
        delta        = round(holdout_mean - train_mean, 4)

        # Per-feature absolute RMSE printout
        print(f"\n  --- Per-feature absolute RMSE ---")
        print(f"  {'feature':<14}  {'train':>8}  {'holdout':>8}")
        for feat in HIDDEN_FEATURES:
            print(f"  {feat:<14}  {train_feat[feat]:>8.4f}  {holdout_feat[feat]:>8.4f}")

        print(f"\n  --- Summary (RMSE / full-data-range, lower = better attack) ---")
        print(f"  train_mean    = {train_mean:.4f}")
        print(f"  nontrain_mean = {holdout_mean:.4f}")
        print(f"  delta         = {delta:+.4f}  (holdout - train)")

        # Log to wandb
        results = {}
        for feat in HIDDEN_FEATURES:
            results[f"abs_rmse_train_{feat}"]    = round(train_feat[feat], 4)
            results[f"abs_rmse_nontrain_{feat}"] = round(holdout_feat[feat], 4)
        results["norm_rmse_train_mean"]    = train_mean
        results["norm_rmse_nontrain_mean"] = holdout_mean
        results["norm_rmse_delta_mean"]    = delta
        wandb.log(results)

        return {
            "sdg":      sdg_label,
            "attack":   attack_method,
            "sample":   sample_idx,
            "train":    train_mean,
            "nontrain": holdout_mean,
            "delta":    delta,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        raise

    finally:
        wandb.finish()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(rows):
    """Average across samples and print a formatted summary table."""
    if not rows:
        return

    # Aggregate: mean ± std across samples for each (sdg, attack) combo
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["sdg"], r["attack"])].append(r)

    # Build aggregated rows in original SDG order
    seen = []
    agg_rows = []
    for sdg_method, sdg_params in SDG_METHODS:
        sdg_label = sdg_dirname(sdg_method, sdg_params)
        for attack in ATTACK_METHODS:
            key = (sdg_label, attack)
            if key in groups and key not in seen:
                seen.append(key)
                samples = groups[key]
                trains    = [s["train"]    for s in samples]
                nontrains = [s["nontrain"] for s in samples]
                deltas    = [s["delta"]    for s in samples]
                agg_rows.append({
                    "sdg":    sdg_label,
                    "attack": attack,
                    "n":      len(samples),
                    "train":       round(float(np.mean(trains)),    4),
                    "train_std":   round(float(np.std(trains)),     4),
                    "nontrain":    round(float(np.mean(nontrains)), 4),
                    "nontrain_std":round(float(np.std(nontrains)),  4),
                    "delta":       round(float(np.mean(deltas)),    4),
                    "delta_std":   round(float(np.std(deltas)),     4),
                })

    if not agg_rows:
        return

    w_sdg  = max(len(r["sdg"])    for r in agg_rows)
    w_sdg  = max(w_sdg, len("SDG method"))
    w_atk  = max(len(r["attack"]) for r in agg_rows)
    w_atk  = max(w_atk, len("attack"))
    cw     = 13   # column width for "mean ± std"

    def fmt_val(mean, std):
        return f"{mean:.4f}±{std:.4f}"

    def fmt_row(sdg, attack, n, train, tr_std, nontrain, nt_std, delta, d_std):
        return (
            f"  {sdg:<{w_sdg}}  {attack:<{w_atk}}  {n:^3}  "
            f"{fmt_val(train, tr_std):>{cw}}  "
            f"{fmt_val(nontrain, nt_std):>{cw}}  "
            f"{fmt_val(delta, d_std):>{cw}}"
        )

    sep    = "  " + "-" * (w_sdg + w_atk + cw * 3 + 17)
    header = (
        f"  {'SDG method':<{w_sdg}}  {'attack':<{w_atk}}  {'n':^3}  "
        f"{'train (mean±std)':>{cw}}  "
        f"{'nontrain (mean±std)':>{cw}}  "
        f"{'delta (mean±std)':>{cw}}"
    )

    print(f"\n\n{'='*80}")
    print("  SWEEP SUMMARY  —  california size_1k, averaged over 5 samples")
    print(f"  QI: Latitude, Longitude, HouseAge, Population")
    print(f"  Hidden: MedInc, AveRooms, AveBedrms, AveOccup, MedHouseVal")
    print(f"  Metric: RMSE / full-dataset-range (lower = better attack)")
    print(f"  delta = holdout - train  (positive = memorization signal)")
    print(f"{'='*80}")
    print(header)
    print(sep)

    prev_sdg = None
    for r in agg_rows:
        if prev_sdg and r["sdg"] != prev_sdg:
            print(sep)
        print(fmt_row(
            r["sdg"], r["attack"], r["n"],
            r["train"],    r["train_std"],
            r["nontrain"], r["nontrain_std"],
            r["delta"],    r["delta_std"],
        ))
        prev_sdg = r["sdg"]

    print(sep)

    for attack in ATTACK_METHODS:
        subset = [r for r in agg_rows if r["attack"] == attack]
        if subset:
            print(fmt_row(
                f"AVG ({attack})", attack, sum(r["n"] for r in subset),
                round(float(np.mean([r["train"]    for r in subset])), 4),
                round(float(np.mean([r["train_std"] for r in subset])), 4),
                round(float(np.mean([r["nontrain"] for r in subset])), 4),
                round(float(np.mean([r["nontrain_std"] for r in subset])), 4),
                round(float(np.mean([r["delta"]    for r in subset])), 4),
                round(float(np.mean([r["delta_std"] for r in subset])), 4),
            ))

    print(f"{'='*80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    total = len(SDG_METHODS) * len(ATTACK_METHODS) * len(TRAIN_SAMPLES)
    done  = 0
    all_results = []

    for sdg_method, sdg_params in SDG_METHODS:
        for attack_method in ATTACK_METHODS:
            for sample_idx in TRAIN_SAMPLES:
                done += 1
                print(f"\n[{done}/{total}]", end="")
                summary = run_experiment(sdg_method, sdg_params, attack_method, sample_idx)
                if summary:
                    all_results.append(summary)

    print_summary_table(all_results)
    print(f"All {total} runs complete.")
