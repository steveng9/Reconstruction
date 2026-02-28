#!/usr/bin/env python
"""
RA Subgroup & Outlier Analysis.

For each SDG method, runs a reconstruction attack on training targets and
analyses how RA performance varies between:
  - QI-space outliers vs non-outliers (IsolationForest)
  - Subgroups defined by individual categorical QI feature values

Outputs:
  - Per-SDG terminal report with outlier/subgroup breakdowns
  - Per-feature outlier vs non-outlier comparison
  - Cross-SDG summary table
  - Optional row-level score CSVs for downstream analysis

Usage:
    python experiment_scripts/analyze_ra_subgroups.py
"""

import sys
import os

sys.path.insert(0, "/home/golobs/Reconstruction")
sys.path.append("/home/golobs/MIA_on_diffusion/")
sys.path.append("/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM")
sys.path.append("/home/golobs/recon-synth")
sys.path.append("/home/golobs/recon-synth/attacks")
sys.path.append("/home/golobs/recon-synth/attacks/solvers")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

from get_data import load_data, QIs, minus_QIs
from scoring import calculate_row_level_scores, compute_outlier_scores
from attacks import get_attack
from enhancements import apply_chaining, apply_ensembling
from master_experiment_script import _prepare_config


# ── Configuration ─────────────────────────────────────────────────────────────

DATASET     = "adult"
DATA_ROOT   = "/home/golobs/data/reconstruction_data/"
SAMPLE_SIZE = 10_000
SAMPLE_DIR  = f"{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_01"

SDG_METHODS = [
    ("TabDDPM",  {}),
    ("MST",      {"epsilon": 1000.0}),
    ("Synthpop", {}),
]

QI_NAME       = "QI1"
DATA_TYPE     = "categorical"   # "categorical" or "continuous"
ATTACK_METHOD = "RandomForest"
ATTACK_PARAMS = {"max_depth": 15, "num_estimators": 25}

# Outlier detection settings
OUTLIER_METHOD     = "isolation_forest"  # "isolation_forest" or "gower_knn"
OUTLIER_PERCENTILE = 90                  # top-N% of outlier scores flagged as outlier

# Categorical QI columns to use for subgroup breakdowns.
# Set to None to auto-detect all non-float QI columns.
SUBGROUP_COLS = None   # e.g. ["race", "sex", "education"]

# Min group size to include in subgroup report (avoids noisy tiny groups)
MIN_GROUP_SIZE = 30

# Save a CSV of row-level scores + outlier flags per SDG method?
SAVE_CSV   = True
OUTPUT_DIR = "/home/golobs/Reconstruction/analysis_outputs/ra_subgroups"

# ── Helpers ───────────────────────────────────────────────────────────────────

# For categorical RA: higher = better (attacker correct more often).
# For continuous RA: lower = better (smaller normalized error).
LOWER_IS_BETTER = (DATA_TYPE == "continuous")


def sdg_dirname(method, params=None):
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    return f"{method}_eps{eps:g}" if eps is not None else method


def make_config(sdg_method, sdg_params):
    return {
        "dataset": {
            "name": DATASET,
            "dir":  SAMPLE_DIR,
            "size": SAMPLE_SIZE,
            "type": DATA_TYPE,
        },
        "QI":            QI_NAME,
        "data_type":     DATA_TYPE,
        "attack_method": ATTACK_METHOD,
        "sdg_method":    sdg_method,
        "sdg_params":    sdg_params or None,
        "memorization_test": {"enabled": False},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            ATTACK_METHOD: ATTACK_PARAMS,
        },
    }


def _infer_subgroup_cols(train_df, qi_cols):
    """Auto-detect categorical QI columns (non-float64)."""
    return [c for c in qi_cols if train_df[c].dtype != np.float64]


def _lift_str(val, baseline):
    """
    Return a lift annotation relative to baseline, e.g. '▲+0.0312' or '↓-0.0211'.
    Direction labels assume higher RA_row_mean = better for the attacker (categorical).
    For continuous (lower = better) the caller should negate or override accordingly.
    """
    diff = val - baseline
    if not LOWER_IS_BETTER:
        arrow = "▲" if diff >  0.005 else ("↓" if diff < -0.005 else " ")
    else:
        arrow = "▲" if diff < -0.005 else ("↓" if diff >  0.005 else " ")
    return f"{arrow}{diff:+.4f}"


def _sig_stars(p):
    if np.isnan(p):     return ""
    if p < 0.001:       return "***"
    if p < 0.01:        return "**"
    if p < 0.05:        return "*"
    return "ns"


# ── Analysis functions ────────────────────────────────────────────────────────

def _outlier_analysis(row_scores, outlier_flags):
    """
    Compare RA_row_mean for QI-space outliers vs non-outliers.

    Returns dict: overall_mean, outlier_mean, non_outlier_mean,
                  n_outlier, n_non_outlier, mannwhitney_p
    """
    scores   = row_scores["RA_row_mean"].values
    out_mask = np.asarray(outlier_flags, dtype=bool)
    non_mask = ~out_mask

    res = {
        "overall_mean":     float(scores.mean()),
        "outlier_mean":     float(scores[out_mask].mean()) if out_mask.any() else float("nan"),
        "non_outlier_mean": float(scores[non_mask].mean()) if non_mask.any() else float("nan"),
        "n_outlier":        int(out_mask.sum()),
        "n_non_outlier":    int(non_mask.sum()),
        "mannwhitney_p":    float("nan"),
    }

    if out_mask.any() and non_mask.any():
        _, p = mannwhitneyu(scores[out_mask], scores[non_mask], alternative="two-sided")
        res["mannwhitney_p"] = float(p)

    return res


def _per_feature_outlier_analysis(row_scores, outlier_flags, hidden_features):
    """
    Per hidden-feature RA breakdown: outlier mean vs non-outlier mean.

    For categorical features RA_row_{feat} is 0/1 accuracy.
    For continuous features  RA_row_{feat} is normalized absolute error.
    """
    out_mask = np.asarray(outlier_flags, dtype=bool)
    non_mask = ~out_mask

    rows = []
    for feat in hidden_features:
        col = f"RA_row_{feat}"
        if col not in row_scores.columns:
            continue
        sc = row_scores[col].values
        om  = float(sc[out_mask].mean()) if out_mask.any() else float("nan")
        nm  = float(sc[non_mask].mean()) if non_mask.any() else float("nan")

        # Direction of diff: for categorical, positive diff means outliers reconstructed better.
        # For continuous, negative diff means outliers reconstructed better (lower error).
        rows.append({
            "feature":          feat,
            "outlier_mean":     om,
            "non_outlier_mean": nm,
            "diff":             om - nm,
        })

    return pd.DataFrame(rows)


def _subgroup_analysis(row_scores, train_df, subgroup_cols):
    """
    For each categorical QI column, compute mean RA_row_mean per value group.

    Returns dict: {col_name: DataFrame with [value, mean_ra, std_ra, n]}
    """
    scores  = row_scores["RA_row_mean"].values
    results = {}

    for col in subgroup_cols:
        if col not in train_df.columns:
            continue
        vals = train_df[col].reset_index(drop=True).values

        rows = []
        for v in np.unique(vals[~pd.isnull(vals)]):
            mask = vals == v
            if mask.sum() < MIN_GROUP_SIZE:
                continue
            grp = scores[mask]
            rows.append({
                "value":   v,
                "mean_ra": float(grp.mean()),
                "std_ra":  float(grp.std()),
                "n":       int(mask.sum()),
            })

        if rows:
            df = pd.DataFrame(rows).sort_values("mean_ra", ascending=LOWER_IS_BETTER)
            results[col] = df

    return results


# ── Terminal report ────────────────────────────────────────────────────────────

def _print_report(sdg_label, outlier_res, subgroup_res, per_feat_df):
    W = 62
    direction_note = "lower = better (less error)" if LOWER_IS_BETTER else "higher = better (more correct)"
    print(f"\n{'═'*W}")
    print(f"  RA Subgroup Analysis — {DATASET} / {sdg_label}")
    print(f"{'═'*W}")
    print(f"  Metric: RA_row_mean  ({direction_note})")

    # ── Outlier vs non-outlier ──────────────────────────────────────────────
    ov  = outlier_res["overall_mean"]
    om  = outlier_res["outlier_mean"]
    nm  = outlier_res["non_outlier_mean"]
    n_o = outlier_res["n_outlier"]
    n_n = outlier_res["n_non_outlier"]
    p   = outlier_res["mannwhitney_p"]

    print(f"\n  Overall RA_row_mean: {ov:.4f}")
    print(f"\n  QI-space outlier comparison")
    print(f"  (method={OUTLIER_METHOD}, flagging top-{OUTLIER_PERCENTILE}% as outliers)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"    Outliers     (n={n_o:>5}):  {om:.4f}  {_lift_str(om, ov)}")
    print(f"    Non-outliers (n={n_n:>5}):  {nm:.4f}  {_lift_str(nm, ov)}")
    if not np.isnan(p):
        print(f"    Mann-Whitney U p={p:.4f}  {_sig_stars(p)}")

    # ── Per-feature breakdown ───────────────────────────────────────────────
    if per_feat_df is not None and not per_feat_df.empty:
        print(f"\n  Per-feature breakdown  (outlier / non-outlier)")
        print(f"  {'Feature':<22}  {'Outlier':>8}  {'Non-Out':>8}  {'Diff':>8}")
        print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}")
        for _, row in per_feat_df.iterrows():
            feat = row["feature"]
            om_f = row["outlier_mean"]
            nm_f = row["non_outlier_mean"]
            diff = row["diff"]
            # Flag features where outliers fare noticeably differently
            if not LOWER_IS_BETTER:
                arrow = "▲" if diff > 0.01 else ("↓" if diff < -0.01 else " ")
            else:
                arrow = "▲" if diff < -0.01 else ("↓" if diff > 0.01 else " ")
            print(f"  {feat:<22}  {om_f:>8.4f}  {nm_f:>8.4f}  {arrow}{diff:>+7.4f}")

    # ── Subgroup breakdown ──────────────────────────────────────────────────
    if subgroup_res:
        print(f"\n  Subgroup breakdown by categorical QI feature:")
        for col, grp_df in subgroup_res.items():
            overall_col = grp_df["mean_ra"].mean()   # unweighted; use ov for consistency
            print(f"\n  [{col}]")
            print(f"  {'Value':<26}  {'n':>5}  {'RA mean':>8}  {'± std':>7}  {'Lift vs overall':>16}")
            print(f"  {'─'*26}  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*16}")
            for _, row in grp_df.iterrows():
                val  = str(row["value"])
                mean = row["mean_ra"]
                std  = row["std_ra"]
                n    = row["n"]
                lift = _lift_str(mean, ov)
                print(f"  {val:<26}  {n:>5}  {mean:>8.4f}  {std:>7.4f}  {lift:>16}")

    print(f"\n{'═'*W}\n")


# ── Summary table ─────────────────────────────────────────────────────────────

def _print_summary(all_results):
    W = 70
    print(f"\n{'═'*W}")
    print(f"  CROSS-SDG SUMMARY  ({DATASET}, {QI_NAME}, {ATTACK_METHOD})")
    direction_note = "lower=better" if LOWER_IS_BETTER else "higher=better"
    print(f"  RA_row_mean  ({direction_note})")
    print(f"{'═'*W}")

    col_w = max((len(r["sdg_label"]) for r in all_results), default=12)
    header = (f"  {'SDG Method':<{col_w}}  {'Overall':>9}  {'Outlier':>9}"
              f"  {'Non-Outl':>9}  {'Lift (O-NO)':>12}  {'p-val':>8}")
    print(f"\n{header}")
    print("  " + "─" * (col_w + 60))

    for r in all_results:
        label = r["sdg_label"]
        o = r["outlier_res"]
        if o is None:
            print(f"  {label:<{col_w}}  (outlier analysis failed)")
            continue

        ov   = o["overall_mean"]
        om   = o["outlier_mean"]
        nm   = o["non_outlier_mean"]
        p    = o["mannwhitney_p"]
        lift = om - nm

        if not LOWER_IS_BETTER:
            arrow = "▲" if lift >  0.005 else ("↓" if lift < -0.005 else " ")
        else:
            arrow = "▲" if lift < -0.005 else ("↓" if lift >  0.005 else " ")

        sig  = _sig_stars(p)
        p_s  = f"{p:.4f}" if not np.isnan(p) else "  N/A"
        print(f"  {label:<{col_w}}  {ov:>9.4f}  {om:>9.4f}  {nm:>9.4f}  {arrow}{lift:>+10.4f}  {p_s:>7} {sig}")

    print(f"{'═'*W}\n")


# ── Core per-SDG run ──────────────────────────────────────────────────────────

def run_analysis(sdg_method, sdg_params):
    sdg_label = sdg_dirname(sdg_method, sdg_params)
    print(f"\n  Loading data and running attack: {DATASET} / {sdg_label} ...")

    cfg      = make_config(sdg_method, sdg_params)
    prepared = _prepare_config(cfg)

    train, synth, qi, hidden_features, _ = load_data(cfg)

    # Run attack
    attack_fn = get_attack(ATTACK_METHOD, DATA_TYPE)
    attack_fn = apply_ensembling(attack_fn, prepared)
    reconstructed, _, _ = apply_chaining(attack_fn, prepared, synth, train, qi, hidden_features)

    # Row-level RA scores
    row_scores = calculate_row_level_scores(train, reconstructed, hidden_features, DATA_TYPE)

    # Outlier detection in QI space
    subgroup_cols = SUBGROUP_COLS if SUBGROUP_COLS is not None else _infer_subgroup_cols(train, qi)
    qi_cat = [c for c in qi if train[c].dtype != np.float64]
    qi_num = [c for c in qi if train[c].dtype == np.float64]

    has_outliers = False
    o_scores     = pd.Series(0.0,   index=train.index)
    o_flags      = pd.Series(False, index=train.index)
    try:
        o_scores, o_flags = compute_outlier_scores(
            train, qi, qi_cat, qi_num,
            method=OUTLIER_METHOD, percentile=OUTLIER_PERCENTILE,
        )
        has_outliers = True
    except Exception as e:
        print(f"  WARNING: outlier scoring failed ({e}), skipping outlier analysis.")

    # Analyses
    outlier_res = _outlier_analysis(row_scores, o_flags) if has_outliers else None
    per_feat_df = _per_feature_outlier_analysis(row_scores, o_flags, hidden_features) if has_outliers else None
    subgroup_res = _subgroup_analysis(row_scores, train.reset_index(drop=True), subgroup_cols)

    # Optionally save CSV
    if SAVE_CSV:
        out_dir = Path(OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        full_df = train[qi].copy().reset_index(drop=True)
        for feat in hidden_features:
            full_df[f"true_{feat}"]  = train[feat].reset_index(drop=True).values
            full_df[f"recon_{feat}"] = reconstructed[feat].reset_index(drop=True).values
        for col in row_scores.columns:
            full_df[col] = row_scores[col].reset_index(drop=True).values
        full_df["outlier_score"] = o_scores.reset_index(drop=True).values
        full_df["is_outlier"]    = o_flags.reset_index(drop=True).values

        out_path = out_dir / f"{DATASET}_{sdg_label}_{QI_NAME}.csv"
        full_df.to_csv(out_path, index=False)
        print(f"  Row scores saved → {out_path}")

    _print_report(sdg_label, outlier_res, subgroup_res, per_feat_df)

    return {
        "sdg_label":   sdg_label,
        "outlier_res": outlier_res,
        "subgroup_res": subgroup_res,
        "per_feat_df": per_feat_df,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nRA Subgroup & Outlier Analysis")
    print(f"  Dataset      : {DATASET}  /  size_{SAMPLE_SIZE}")
    print(f"  Sample dir   : {SAMPLE_DIR}")
    print(f"  Attack       : {ATTACK_METHOD}  (QI={QI_NAME})")
    print(f"  Outlier      : {OUTLIER_METHOD}  (top-{OUTLIER_PERCENTILE}%)")
    print(f"  SDG methods  : {[sdg_dirname(m, p) for m, p in SDG_METHODS]}")
    print(f"  Save CSV     : {SAVE_CSV}  →  {OUTPUT_DIR if SAVE_CSV else 'n/a'}")

    all_results = []
    for sdg_method, sdg_params in SDG_METHODS:
        sdg_label = sdg_dirname(sdg_method, sdg_params)
        try:
            result = run_analysis(sdg_method, sdg_params)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR for {sdg_label}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_results) > 1:
        _print_summary(all_results)

    print("Done.")
