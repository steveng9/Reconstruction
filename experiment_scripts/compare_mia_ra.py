#!/usr/bin/env python
"""
MIA vs RA-as-MIA Comparison Script.

Runs SynthDistance, NNDR, and RA-as-MIA on the same aligned target sample,
then reports:
  - AUC / Advantage / TPR@FPR0.1% for each method
  - Quadrant analysis: where methods agree/disagree on training records
  - Outlier intersection: do high-scoring records cluster in QI-space outliers?

Results are printed to the terminal and logged to WandB.

Usage:
    python experiment_scripts/compare_mia_ra.py
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
from scipy.stats import spearmanr
import wandb

from get_data import load_mia_data
from attacks import get_attack, ra_as_mia
from attacks.mia import synth_distance_mia, nndr_mia
from master_experiment_script import _prepare_config


# ── Configuration ─────────────────────────────────────────────────────────────

DATASET       = "adult"
DATA_ROOT     = "/home/golobs/data/reconstruction_data/"
SAMPLE_SIZE   = 10_000
SAMPLE_DIR    = f"{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_00"
HOLDOUT_DIR   = f"{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_01"

SDG_METHODS = [
    ("MST",    {"epsilon": 1.0}),
    ("MST",    {"epsilon": 10.0}),
    ("TVAE",   {}),
    ("TabDDPM",{}),
]

RA_METHOD = "RandomForest"
RA_PARAMS  = {"max_depth": 15, "num_estimators": 25}
QI         = "QI1"
DATA_TYPE  = "categorical"

MIA_METHODS  = ["SynthDistance", "NNDR"]
N_TARGETS    = None   # None = use full train + full holdout; int = sample N from each
SEED         = 42
OUTLIER_PCT  = 90     # top-N% = outlier
TOP_K_PCT    = 0.10   # top-10% of MIA scores for outlier intersection analysis


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        "QI":            QI,
        "data_type":     DATA_TYPE,
        "attack_method": RA_METHOD,
        "sdg_method":    sdg_method,
        "sdg_params":    sdg_params or None,
        "memorization_test": {
            "enabled":     True,
            "holdout_dir": HOLDOUT_DIR,
        },
        "mia_params": {
            "n_targets": N_TARGETS,
            "seed":      SEED,
        },
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            RA_METHOD:    RA_PARAMS,
        },
    }


def _infer_cat_num(df, cols):
    """Infer categorical/continuous split from dtypes (float64 → continuous)."""
    cat = [c for c in cols if df[c].dtype != np.float64]
    num = [c for c in cols if df[c].dtype == np.float64]
    return cat, num


# ── Analysis helpers ───────────────────────────────────────────────────────────

def _quadrant_analysis(mia_scores, ra_scores, labels, mia_name):
    """
    Quadrant analysis on training records only (labels == 1).

    Uses the median of the full sample (train + holdout) as the threshold
    for each score independently.

    Returns dict with counts and Spearman correlation.
    """
    mia_thresh = np.median(mia_scores)
    ra_thresh  = np.median(ra_scores)

    member_mask = labels == 1
    mia_m = mia_scores[member_mask]
    ra_m  = ra_scores[member_mask]

    both_high   = int(((mia_m >= mia_thresh) & (ra_m >= ra_thresh)).sum())
    mia_only    = int(((mia_m >= mia_thresh) & (ra_m <  ra_thresh)).sum())
    ra_only     = int(((mia_m <  mia_thresh) & (ra_m >= ra_thresh)).sum())
    both_low    = int(((mia_m <  mia_thresh) & (ra_m <  ra_thresh)).sum())

    corr, pval = spearmanr(mia_m, ra_m)

    return {
        "mia_name":   mia_name,
        "both_high":  both_high,
        "mia_only":   mia_only,
        "ra_only":    ra_only,
        "both_low":   both_low,
        "spearman_r": round(float(corr), 3),
        "spearman_p": round(float(pval), 4),
    }


def _outlier_intersection(score_dict, outlier_flags, top_k_pct):
    """
    For each method, find top-k% records and report fraction that are QI-outliers.

    Parameters
    ----------
    score_dict   : dict {method_name: np.ndarray of scores}
    outlier_flags: np.ndarray of bool (True = outlier in QI space)
    top_k_pct    : float, e.g. 0.10 for top-10%

    Returns
    -------
    list of dicts with keys: method, outlier_rate, baseline_rate, lift_pp
    """
    baseline_rate = float(outlier_flags.mean())
    results = []
    k = max(1, int(len(outlier_flags) * top_k_pct))
    for name, scores in score_dict.items():
        top_k_idx    = np.argsort(scores)[::-1][:k]
        top_k_outlier_rate = float(outlier_flags[top_k_idx].mean())
        results.append({
            "method":        name,
            "outlier_rate":  top_k_outlier_rate,
            "baseline_rate": baseline_rate,
            "lift_pp":       top_k_outlier_rate - baseline_rate,
        })
    return results, baseline_rate


# ── Core comparison ───────────────────────────────────────────────────────────

def run_comparison(sdg_method, sdg_params):
    sdg_label = sdg_dirname(sdg_method, sdg_params)
    print(f"\n{'═'*60}")
    print(f"  MIA vs RA-as-MIA — {DATASET} / {sdg_label}")
    print(f"{'═'*60}")

    cfg         = make_config(sdg_method, sdg_params)
    prepared_ra = _prepare_config(cfg)  # flattens attack params for RA

    # Load data
    train, synth, holdout, meta = load_mia_data(cfg)

    from get_data import QIs, minus_QIs
    qi              = QIs[DATASET][QI]
    hidden_features = minus_QIs[DATASET][QI]

    # ── MIA attacks ──────────────────────────────────────────────────────────
    all_scores  = {}
    all_metrics = {}

    print("\n  [SynthDistance]")
    sd_metrics, sd_scores, labels, all_targets = synth_distance_mia(
        cfg, synth, train, holdout, meta, return_raw=True
    )
    all_scores["SynthDistance"]  = sd_scores
    all_metrics.update({f"SynthDistance_{k}": v for k, v in sd_metrics.items()})

    print("\n  [NNDR]")
    nn_metrics, nn_scores, _, _ = nndr_mia(
        cfg, synth, train, holdout, meta, return_raw=True
    )
    all_scores["NNDR"]  = nn_scores
    all_metrics.update({f"NNDR_{k}": v for k, v in nn_metrics.items()})

    # ── RA-as-MIA ────────────────────────────────────────────────────────────
    print("\n  [RA-as-MIA]")
    attack_fn    = get_attack(RA_METHOD, DATA_TYPE)
    ra_metrics, ra_scores, _, _ = ra_as_mia(
        attack_fn, prepared_ra, synth, train, holdout, qi, hidden_features,
        n_targets=N_TARGETS, seed=SEED,
    )
    all_scores[f"RA-as-MIA ({RA_METHOD})"] = ra_scores
    all_metrics.update(ra_metrics)

    # ── Outlier scores in QI space ────────────────────────────────────────────
    from scoring import compute_outlier_scores
    qi_cat, qi_num = _infer_cat_num(all_targets, qi)
    try:
        o_scores, o_flags = compute_outlier_scores(
            all_targets, qi, qi_cat, qi_num,
            method='isolation_forest', percentile=OUTLIER_PCT
        )
        outlier_flags = o_flags.values
        has_outliers  = True
    except Exception as e:
        print(f"  WARNING: outlier scoring failed ({e}), skipping outlier analysis.")
        has_outliers = False

    # ── Quadrant analyses ─────────────────────────────────────────────────────
    quadrant_results = []
    for mia_name, mia_sc in [("SynthDistance", sd_scores), ("NNDR", nn_scores)]:
        q = _quadrant_analysis(mia_sc, ra_scores, labels, mia_name)
        quadrant_results.append(q)
        all_metrics.update({
            f"{mia_name}_vs_RA_both_high":  q["both_high"],
            f"{mia_name}_vs_RA_mia_only":   q["mia_only"],
            f"{mia_name}_vs_RA_ra_only":    q["ra_only"],
            f"{mia_name}_vs_RA_both_low":   q["both_low"],
            f"{mia_name}_vs_RA_spearman_r": q["spearman_r"],
        })

    # ── Outlier intersection ──────────────────────────────────────────────────
    oi_results   = []
    baseline_rate = None
    if has_outliers:
        oi_results, baseline_rate = _outlier_intersection(
            all_scores, outlier_flags, TOP_K_PCT
        )
        for r in oi_results:
            key = r["method"].replace(" ", "_").replace("(", "").replace(")", "")
            all_metrics[f"{key}_outlier_rate_top{int(TOP_K_PCT*100)}pct"] = round(r["outlier_rate"], 4)

    # ── Terminal report ───────────────────────────────────────────────────────
    _print_report(sdg_label, all_scores, sd_metrics, nn_metrics, ra_metrics,
                  quadrant_results, oi_results, baseline_rate)

    return all_metrics


# ── Terminal report ────────────────────────────────────────────────────────────

def _print_report(sdg_label, all_scores, sd_metrics, nn_metrics, ra_metrics,
                  quadrant_results, oi_results, baseline_rate):
    print(f"\n{'═'*60}")
    print(f"  MIA vs RA-as-MIA Comparison — {DATASET} / {sdg_label}")
    print(f"{'═'*60}")

    # Main metrics table
    header = f"  {'Method':<22}  {'AUC':>6}  {'Advantage':>9}  {'TPR@FPR0.1%':>11}"
    print(header)
    print("  " + "-" * 56)

    def row(name, m_dict, prefix="MIA"):
        auc  = m_dict.get(f"{prefix}_auc",            m_dict.get("RA_as_MIA_auc",            "?"))
        adv  = m_dict.get(f"{prefix}_advantage",      m_dict.get("RA_as_MIA_advantage",      "?"))
        tpr  = m_dict.get(f"{prefix}_tpr_at_fpr0001", m_dict.get("RA_as_MIA_tpr_at_fpr0001", "?"))
        return f"  {name:<22}  {auc:>6.3f}  {adv:>9.3f}  {tpr:>11.3f}"

    print(row("SynthDistance",        sd_metrics, "MIA"))
    print(row("NNDR",                 nn_metrics, "MIA"))
    print(row(f"RA-as-MIA ({RA_METHOD})", ra_metrics, "RA_as_MIA"))

    # Quadrant analysis
    print(f"\n  Quadrant Analysis (training records, median threshold):")
    for q in quadrant_results:
        print(f"    vs {q['mia_name']}:")
        print(f"      Both-high={q['both_high']}  MIA-only={q['mia_only']}"
              f"  RA-only={q['ra_only']}  Both-low={q['both_low']}")
        print(f"      Spearman r={q['spearman_r']:.3f}  (p={q['spearman_p']:.4f})")

    # Outlier intersection
    if oi_results:
        print(f"\n  Outlier Intersection (top {int(TOP_K_PCT*100)}% of each method's scores):")
        print(f"  {'Method':<26}  {'Outlier rate':>12}  {'vs baseline':>12}")
        bl_str = f"(baseline {baseline_rate*100:.1f}%)"
        print(f"  {'':26}  {'':12}  {bl_str:>12}")
        print("  " + "-" * 54)
        for r in oi_results:
            lift_str = f"{r['lift_pp']*100:+.1f}pp"
            print(f"  {r['method']:<26}  {r['outlier_rate']*100:>11.1f}%  {lift_str:>12}")

    print(f"{'═'*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nMIA vs RA-as-MIA Comparison")
    print(f"  Dataset   : {DATASET}  /  size_{SAMPLE_SIZE}")
    print(f"  Sample    : {SAMPLE_DIR}")
    print(f"  Holdout   : {HOLDOUT_DIR}")
    print(f"  SDG methods: {[sdg_dirname(m, p) for m, p in SDG_METHODS]}")
    print(f"  RA method : {RA_METHOD}")
    print(f"  N targets : {N_TARGETS}")

    wandb.init(
        project="tabular-reconstruction-attacks",
        name=f"compare_mia_ra_{DATASET}",
        config={
            "dataset": DATASET,
            "sample_dir": SAMPLE_DIR,
            "holdout_dir": HOLDOUT_DIR,
            "ra_method": RA_METHOD,
            "n_targets": N_TARGETS,
            "seed": SEED,
            "top_k_pct": TOP_K_PCT,
            "outlier_pct": OUTLIER_PCT,
        },
        tags=[DATASET, "mia_vs_ra", "comparison"],
        group=f"mia_ra_comparison_{DATASET}",
    )

    all_results = {}
    for sdg_method, sdg_params in SDG_METHODS:
        sdg_label = sdg_dirname(sdg_method, sdg_params)
        try:
            metrics = run_comparison(sdg_method, sdg_params)
            # Prefix with SDG label to distinguish methods in WandB
            prefixed = {f"{sdg_label}/{k}": v for k, v in metrics.items()}
            all_results.update(prefixed)
            wandb.log(prefixed)
        except Exception as e:
            print(f"\n  ERROR for {sdg_label}: {e}")
            import traceback
            traceback.print_exc()

    wandb.finish()
    print(f"\nAll comparisons complete.")
