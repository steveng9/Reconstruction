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
SAMPLE_DIR    = f"{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_01"
HOLDOUT_DIR   = f"{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_02"

SDG_METHODS = [
    ("TabDDPM",{}),
    #("MST",    {"epsilon": 1.0}),
    #("MST",    {"epsilon": 10.0}),
    ("MST",    {"epsilon": 1000.0}),
    #("TVAE",   {}),
    ("Synthpop",{}),
]

RA_METHOD = "RandomForest"
RA_PARAMS  = {"max_depth": 15, "num_estimators": 25}
QI         = "QI1"
DATA_TYPE  = "categorical"

# For RA-as-MIA scoring, restrict to genuinely categorical hidden features.
# QI1's full hidden set includes fnlwgt (census weight, ~unique per person),
# education-num (ordinal int), capital-gain/loss, and hours-per-week — all
# effectively continuous.  When treated as categorical their rarity weights
# dominate the score and swamp the real membership signal.
# Set to None to use the full hidden feature set from the QI definition.
RA_HIDDEN_OVERRIDE = ["workclass", "occupation", "relationship", "income"]

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


def _outlier_auc(score_dict, outlier_flags, labels):
    """
    For each method, compute AUC restricted to outlier records and non-outlier
    records separately.

    This catches cases where a method is highly effective against outliers even
    if their scores are not in the global top-k% — e.g. all outliers scored just
    above the median would go undetected by the top-k intersection analysis.

    Parameters
    ----------
    score_dict   : dict {method_name: np.ndarray of scores}
    outlier_flags: np.ndarray of bool
    labels       : np.ndarray of int (1 = member, 0 = non-member)

    Returns
    -------
    list of dicts with keys: method, auc_outlier, auc_non_outlier, n_outlier, n_non_outlier
    """
    from sklearn.metrics import roc_auc_score

    out_mask     = outlier_flags.astype(bool)
    non_out_mask = ~out_mask
    results = []

    for name, scores in score_dict.items():
        def _auc(mask):
            if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
                return float('nan')
            return float(roc_auc_score(labels[mask], scores[mask]))

        results.append({
            "method":        name,
            "auc_outlier":   _auc(out_mask),
            "auc_non_outlier": _auc(non_out_mask),
            "n_outlier":     int(out_mask.sum()),
            "n_non_outlier": int(non_out_mask.sum()),
        })

    return results


def _outlier_intersection(score_dict, outlier_flags, labels, top_k_pct):
    """
    For each method, find top-k% records ("predicted members") and report
    QI-outlier rates overall and split by TP vs FP.

    The key distinction:
      - TP outlier rate: attack is accurate AND targets unusual records → concerning
      - FP outlier rate: attack prefers outliers but is wrong about them → less concerning

    Parameters
    ----------
    score_dict   : dict {method_name: np.ndarray of scores}
    outlier_flags: np.ndarray of bool (True = outlier in QI space)
    labels       : np.ndarray of int (1 = true member, 0 = true non-member)
    top_k_pct    : float, e.g. 0.10 for top-10%

    Returns
    -------
    list of dicts, baseline_rate
    """
    baseline_rate = float(outlier_flags.mean())
    results = []
    k = max(1, int(len(outlier_flags) * top_k_pct))

    for name, scores in score_dict.items():
        top_k_mask = np.zeros(len(scores), dtype=bool)
        top_k_mask[np.argsort(scores)[::-1][:k]] = True

        overall_rate = float(outlier_flags[top_k_mask].mean())

        tp_mask = top_k_mask & (labels == 1)
        fp_mask = top_k_mask & (labels == 0)
        tp_rate = float(outlier_flags[tp_mask].mean()) if tp_mask.any() else float('nan')
        fp_rate = float(outlier_flags[fp_mask].mean()) if fp_mask.any() else float('nan')

        results.append({
            "method":       name,
            "overall_rate": overall_rate,
            "tp_rate":      tp_rate,
            "fp_rate":      fp_rate,
            "n_tp":         int(tp_mask.sum()),
            "n_fp":         int(fp_mask.sum()),
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
    # For RA-as-MIA, optionally restrict to a categorical-only subset of hidden features
    ra_hidden = RA_HIDDEN_OVERRIDE if RA_HIDDEN_OVERRIDE is not None else hidden_features

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

    # ── RA-as-MIA, two QI variants ───────────────────────────────────────────
    attack_fn = get_attack(RA_METHOD, DATA_TYPE)

    # Approach A: standard QI1 (6 features) → predict ra_hidden (4 features)
    print(f"\n  [RA-as-MIA / QI={QI}]")
    ra_a_metrics, ra_a_scores, _, _ = ra_as_mia(
        attack_fn, prepared_ra, synth, train, holdout, qi, ra_hidden,
        n_targets=N_TARGETS, seed=SEED,
    )
    ra_a_label = f"RA-as-MIA ({RA_METHOD}, {QI})"
    all_scores[ra_a_label] = ra_a_scores
    all_metrics.update({k.replace("RA_as_MIA_", f"RA_as_MIA_QI1_"): v
                        for k, v in ra_a_metrics.items()})

    # Approach B: wide QI = all features except ra_hidden → predict ra_hidden
    # The attack has more signal and is trained on exactly what it's scored on.
    qi_wide = [f for f in train.columns if f not in ra_hidden]
    print(f"\n  [RA-as-MIA / wide QI ({len(qi_wide)} features)]")
    ra_b_metrics, ra_b_scores, _, _ = ra_as_mia(
        attack_fn, prepared_ra, synth, train, holdout, qi_wide, ra_hidden,
        n_targets=N_TARGETS, seed=SEED,
    )
    ra_b_label = f"RA-as-MIA ({RA_METHOD}, wide QI)"
    all_scores[ra_b_label] = ra_b_scores
    all_metrics.update({k.replace("RA_as_MIA_", f"RA_as_MIA_wideQI_"): v
                        for k, v in ra_b_metrics.items()})

    # Use the wide-QI variant as the primary RA signal for quadrant/outlier analysis
    ra_scores  = ra_b_scores
    ra_metrics = ra_b_metrics

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

    # ── Outlier intersection & subgroup AUC ──────────────────────────────────
    oi_results    = []
    oa_results    = []
    baseline_rate = None
    if has_outliers:
        oi_results, baseline_rate = _outlier_intersection(
            all_scores, outlier_flags, labels, TOP_K_PCT
        )
        oa_results = _outlier_auc(all_scores, outlier_flags, labels)

        for r in oi_results:
            key = r["method"].replace(" ", "_").replace("(", "").replace(")", "")
            pct = int(TOP_K_PCT * 100)
            all_metrics[f"{key}_outlier_rate_top{pct}pct_overall"] = round(r["overall_rate"], 4)
            all_metrics[f"{key}_outlier_rate_top{pct}pct_tp"]      = round(r["tp_rate"], 4)
            all_metrics[f"{key}_outlier_rate_top{pct}pct_fp"]      = round(r["fp_rate"], 4)
        for r in oa_results:
            key = r["method"].replace(" ", "_").replace("(", "").replace(")", "")
            all_metrics[f"{key}_auc_outlier_subgroup"]     = round(r["auc_outlier"], 4)
            all_metrics[f"{key}_auc_non_outlier_subgroup"] = round(r["auc_non_outlier"], 4)

    # ── Terminal report ───────────────────────────────────────────────────────
    ra_variants = {ra_a_label: ra_a_metrics, ra_b_label: ra_b_metrics}
    _print_report(sdg_label, all_scores, sd_metrics, nn_metrics, ra_variants,
                  quadrant_results, oi_results, oa_results, baseline_rate)

    return all_metrics


# ── Terminal report ────────────────────────────────────────────────────────────

def _print_report(sdg_label, all_scores, sd_metrics, nn_metrics, ra_variants,
                  quadrant_results, oi_results, oa_results, baseline_rate):
    print(f"\n{'═'*60}")
    print(f"  MIA vs RA-as-MIA Comparison — {DATASET} / {sdg_label}")
    print(f"{'═'*60}")

    # Main metrics table
    w = max(22, max(len(n) for n in ra_variants))
    header = f"  {'Method':<{w}}  {'AUC':>6}  {'Advantage':>9}  {'TPR@FPR1%':>10}"
    print(header)
    print("  " + "-" * (w + 32))

    def fmt_row(name, auc, adv, tpr):
        return f"  {name:<{w}}  {auc:>6.3f}  {adv:>9.3f}  {tpr:>11.3f}"

    def mia_row(name, m):
        return fmt_row(name, m["MIA_auc"], m["MIA_advantage"], m["MIA_tpr_at_fpr001"])

    def ra_row(name, m):
        return fmt_row(name, m["RA_as_MIA_auc"], m["RA_as_MIA_advantage"],
                       m["RA_as_MIA_tpr_at_fpr001"])

    print(mia_row("SynthDistance", sd_metrics))
    print(mia_row("NNDR",          nn_metrics))
    for label, m in ra_variants.items():
        print(ra_row(label, m))

    # Quadrant analysis
    print(f"\n  Quadrant Analysis (training records, median threshold):")
    for q in quadrant_results:
        print(f"    vs {q['mia_name']}:")
        print(f"      Both-high={q['both_high']}  MIA-only={q['mia_only']}"
              f"  RA-only={q['ra_only']}  Both-low={q['both_low']}")
        print(f"      Spearman r={q['spearman_r']:.3f}  (p={q['spearman_p']:.4f})")

    # Outlier intersection
    if oi_results:
        k_pct = int(TOP_K_PCT * 100)
        print(f"\n  Outlier Intersection  (top {k_pct}% of scores = 'predicted member',"
              f"  baseline outlier rate: {baseline_rate*100:.1f}%)")
        print(f"  TP outlier rate = attack accurate AND targets unusual records  ← concerning if high")
        print(f"  FP outlier rate = attack prefers outliers but is wrong         ← less concerning")

        w = max(len(r["method"]) for r in oi_results)
        header = (f"  {'Method':<{w}}  {'Overall':>9}  "
                  f"{'TPs [concerning]':>18}  {'FPs [less concern]':>20}")
        print(f"\n{header}")
        print("  " + "-" * (w + 54))

        def fmt_rate(rate, n):
            if np.isnan(rate):
                return f"{'N/A':>8}       "
            lift = rate - baseline_rate
            arrow = "▲" if lift > 0.01 else ("↓" if lift < -0.01 else " ")
            return f"{rate*100:>7.1f}%  {arrow}{lift*100:>+5.1f}pp  (n={n})"

        for r in oi_results:
            overall_lift = r["overall_rate"] - baseline_rate
            overall_arrow = "▲" if overall_lift > 0.01 else ("↓" if overall_lift < -0.01 else " ")
            overall_str = f"{r['overall_rate']*100:>7.1f}%{overall_arrow}{overall_lift*100:>+5.1f}pp"
            tp_str = fmt_rate(r["tp_rate"], r["n_tp"])
            fp_str = fmt_rate(r["fp_rate"], r["n_fp"])
            print(f"  {r['method']:<{w}}  {overall_str}  {tp_str}  {fp_str}")

    # Subgroup AUC
    if oa_results:
        r0 = oa_results[0]
        print(f"\n  Subgroup AUC  (outliers: n={r0['n_outlier']},  non-outliers: n={r0['n_non_outlier']})")
        print(f"  Outlier AUC >> Non-outlier AUC → attack disproportionately effective on unusual records")
        w = max(len(r["method"]) for r in oa_results)
        print(f"\n  {'Method':<{w}}  {'Outlier AUC':>13}  {'Non-outlier AUC':>17}  {'Difference':>12}")
        print("  " + "-" * (w + 48))
        for r in oa_results:
            auc_o   = r["auc_outlier"]
            auc_no  = r["auc_non_outlier"]
            diff    = auc_o - auc_no if not (np.isnan(auc_o) or np.isnan(auc_no)) else float('nan')
            auc_o_s  = f"{auc_o:.3f}"  if not np.isnan(auc_o)  else "N/A"
            auc_no_s = f"{auc_no:.3f}" if not np.isnan(auc_no) else "N/A"
            diff_s   = f"{diff:+.3f}"  if not np.isnan(diff)   else "N/A"
            arrow    = "▲" if (not np.isnan(diff) and diff > 0.02) else \
                       ("↓" if (not np.isnan(diff) and diff < -0.02) else " ")
            print(f"  {r['method']:<{w}}  {auc_o_s:>13}  {auc_no_s:>17}  {arrow}{diff_s:>11}")

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
