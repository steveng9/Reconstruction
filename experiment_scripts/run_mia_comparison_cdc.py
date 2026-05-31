#!/usr/bin/env python
"""
MIA vs RA-as-MIA Comparison — cdc_diabetes (1k, sample_01 / sample_02).

Mirrors compare_mia_ra.py exactly; only the dataset configuration changes.

Run detached:
    nohup conda run -n recon_ python experiment_scripts/run_mia_comparison_cdc.py \
          > outfiles/mia_comparison_cdc.log 2>&1 &
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

DATASET       = "cdc_diabetes"
DATA_ROOT     = "/home/golobs/data/reconstruction_data/"
SAMPLE_SIZE   = 1_000
SAMPLE_DIR    = f"{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_01"
HOLDOUT_DIR   = f"{DATA_ROOT}{DATASET}/size_{SAMPLE_SIZE}/sample_02"

SDG_METHODS = [
    ("CellSuppression", {}),
    ("RankSwap",        {}),
    ("TabDDPM",         {}),
    ("Synthpop",        {}),
    ("MST",             {"epsilon": 1.0}),
    ("MST",             {"epsilon": 1000.0}),
]

RA_METHOD = "RandomForest"
RA_PARAMS  = {"max_depth": 15, "num_estimators": 25}
QI         = "QI1"
DATA_TYPE  = "categorical"

# Exclude continuous-ish features (BMI=62 vals, MentHlth=31, PhysHlth=31) from RA signal.
# All remaining hidden features are binary — clean categorical signal.
RA_HIDDEN_OVERRIDE = [
    "Diabetes_binary", "Stroke", "HeartDiseaseorAttack", "CholCheck",
    "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "DiffWalk",
]

MIA_METHODS  = ["SynthDistance", "NNDR"]
N_TARGETS    = None   # None = use full train + full holdout
SEED         = 42
OUTLIER_PCT  = 90
TOP_K_PCT    = 0.10


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
    cat = [c for c in cols if df[c].dtype != np.float64]
    num = [c for c in cols if df[c].dtype == np.float64]
    return cat, num


# ── Analysis helpers ───────────────────────────────────────────────────────────

def _quadrant_analysis(mia_scores, ra_scores, labels, mia_name):
    mia_thresh = np.median(mia_scores)
    ra_thresh  = np.median(ra_scores)
    member_mask = labels == 1
    mia_m = mia_scores[member_mask]
    ra_m  = ra_scores[member_mask]
    both_high   = int(((mia_m >= mia_thresh) & (ra_m >= ra_thresh)).sum())
    mia_only    = int(((mia_m >= mia_thresh) & (ra_m <  ra_thresh)).sum())
    ra_only     = int(((mia_m <  mia_thresh) & (ra_m >= ra_thresh)).sum())
    both_low    = int(((mia_m <  mia_thresh) & (ra_m <  ra_thresh)).sum())
    corr, pval  = spearmanr(mia_m, ra_m)
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
            "method":          name,
            "auc_outlier":     _auc(out_mask),
            "auc_non_outlier": _auc(non_out_mask),
            "n_outlier":       int(out_mask.sum()),
            "n_non_outlier":   int(non_out_mask.sum()),
        })
    return results


def _outlier_intersection(score_dict, outlier_flags, labels, top_k_pct):
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
    print(f"{'═'*60}", flush=True)

    cfg         = make_config(sdg_method, sdg_params)
    prepared_ra = _prepare_config(cfg)

    train, synth, holdout, meta = load_mia_data(cfg)

    from get_data import QIs, minus_QIs
    qi              = QIs[DATASET][QI]
    hidden_features = minus_QIs[DATASET][QI]
    ra_hidden = RA_HIDDEN_OVERRIDE if RA_HIDDEN_OVERRIDE is not None else hidden_features

    # ── MIA attacks ──────────────────────────────────────────────────────────
    all_scores  = {}
    all_metrics = {}

    print("\n  [SynthDistance]", flush=True)
    sd_metrics, sd_scores, labels, all_targets = synth_distance_mia(
        cfg, synth, train, holdout, meta, return_raw=True
    )
    all_scores["SynthDistance"]  = sd_scores
    all_metrics.update({f"SynthDistance_{k}": v for k, v in sd_metrics.items()})

    print("\n  [NNDR]", flush=True)
    nn_metrics, nn_scores, _, _ = nndr_mia(
        cfg, synth, train, holdout, meta, return_raw=True
    )
    all_scores["NNDR"]  = nn_scores
    all_metrics.update({f"NNDR_{k}": v for k, v in nn_metrics.items()})

    # ── RA-as-MIA, two QI variants ───────────────────────────────────────────
    attack_fn = get_attack(RA_METHOD, DATA_TYPE)

    print(f"\n  [RA-as-MIA / QI={QI}]", flush=True)
    ra_a_metrics, ra_a_scores, _, _ = ra_as_mia(
        attack_fn, prepared_ra, synth, train, holdout, qi, ra_hidden,
        n_targets=N_TARGETS, seed=SEED,
    )
    ra_a_label = f"RA-as-MIA ({RA_METHOD}, {QI})"
    all_scores[ra_a_label] = ra_a_scores
    all_metrics.update({k.replace("RA_as_MIA_", "RA_as_MIA_QI1_"): v
                        for k, v in ra_a_metrics.items()})

    qi_wide = [f for f in train.columns if f not in ra_hidden]
    print(f"\n  [RA-as-MIA / wide QI ({len(qi_wide)} features)]", flush=True)
    ra_b_metrics, ra_b_scores, _, _ = ra_as_mia(
        attack_fn, prepared_ra, synth, train, holdout, qi_wide, ra_hidden,
        n_targets=N_TARGETS, seed=SEED,
    )
    ra_b_label = f"RA-as-MIA ({RA_METHOD}, wide QI)"
    all_scores[ra_b_label] = ra_b_scores
    all_metrics.update({k.replace("RA_as_MIA_", "RA_as_MIA_wideQI_"): v
                        for k, v in ra_b_metrics.items()})

    ra_scores  = ra_b_scores
    ra_metrics = ra_b_metrics

    # ── Outlier scores ────────────────────────────────────────────────────────
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
        print(f"  WARNING: outlier scoring failed ({e}), skipping.", flush=True)
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
    oi_results  = []
    oa_results  = []
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

    _print_report(sdg_label, all_scores, sd_metrics, nn_metrics,
                  {ra_a_label: ra_a_metrics, ra_b_label: ra_b_metrics},
                  quadrant_results, oi_results, oa_results, baseline_rate)

    return all_metrics


# ── Terminal report ────────────────────────────────────────────────────────────

def _print_report(sdg_label, all_scores, sd_metrics, nn_metrics, ra_variants,
                  quadrant_results, oi_results, oa_results, baseline_rate):
    print(f"\n{'═'*60}")
    print(f"  MIA vs RA-as-MIA Comparison — {DATASET} / {sdg_label}")
    print(f"{'═'*60}")

    w = max(22, max(len(n) for n in ra_variants))
    header = f"  {'Method':<{w}}  {'AUC':>6}  {'Advantage':>9}  {'TPR@FPR1%':>10}"
    print(header)
    print("  " + "-" * (w + 32))

    def fmt_row(name, auc, adv, tpr):
        return f"  {name:<{w}}  {auc:>6.3f}  {adv:>9.3f}  {tpr:>11.3f}"

    print(fmt_row("SynthDistance", sd_metrics["MIA_auc"],
                  sd_metrics["MIA_advantage"], sd_metrics["MIA_tpr_at_fpr001"]))
    print(fmt_row("NNDR", nn_metrics["MIA_auc"],
                  nn_metrics["MIA_advantage"], nn_metrics["MIA_tpr_at_fpr001"]))
    for label, m in ra_variants.items():
        print(fmt_row(label, m["RA_as_MIA_auc"],
                      m["RA_as_MIA_advantage"], m["RA_as_MIA_tpr_at_fpr001"]))

    print(f"\n  Quadrant Analysis (training records, median threshold):")
    for q in quadrant_results:
        print(f"    vs {q['mia_name']}:")
        print(f"      Both-high={q['both_high']}  MIA-only={q['mia_only']}"
              f"  RA-only={q['ra_only']}  Both-low={q['both_low']}")
        print(f"      Spearman r={q['spearman_r']:.3f}  (p={q['spearman_p']:.4f})")

    if oi_results:
        k_pct = int(TOP_K_PCT * 100)
        print(f"\n  Outlier Intersection  (top {k_pct}% of scores, baseline: {baseline_rate*100:.1f}%)")
        w2 = max(len(r["method"]) for r in oi_results)
        for r in oi_results:
            overall_lift = r["overall_rate"] - baseline_rate
            arrow = "▲" if overall_lift > 0.01 else ("↓" if overall_lift < -0.01 else " ")
            print(f"  {r['method']:<{w2}}  overall={r['overall_rate']*100:.1f}%{arrow}{overall_lift*100:+.1f}pp"
                  f"  TP={r['tp_rate']*100:.1f}%  FP={r['fp_rate']*100:.1f}%")

    if oa_results:
        r0 = oa_results[0]
        print(f"\n  Subgroup AUC (outliers n={r0['n_outlier']}, non-outliers n={r0['n_non_outlier']})")
        w2 = max(len(r["method"]) for r in oa_results)
        for r in oa_results:
            diff = r["auc_outlier"] - r["auc_non_outlier"]
            print(f"  {r['method']:<{w2}}  outlier={r['auc_outlier']:.3f}  "
                  f"non-outlier={r['auc_non_outlier']:.3f}  diff={diff:+.3f}")

    print(f"{'═'*60}\n", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nMIA vs RA-as-MIA Comparison — {DATASET}")
    print(f"  Sample  : {SAMPLE_DIR}")
    print(f"  Holdout : {HOLDOUT_DIR}")
    print(f"  SDG     : {[sdg_dirname(m, p) for m, p in SDG_METHODS]}")
    print(f"  RA      : {RA_METHOD}")
    print(f"  RA hidden override: {RA_HIDDEN_OVERRIDE}")

    wandb.init(
        project="tabular-reconstruction-attacks",
        name=f"compare_mia_ra_{DATASET}",
        config={
            "dataset": DATASET,
            "sample_dir": SAMPLE_DIR,
            "holdout_dir": HOLDOUT_DIR,
            "ra_method": RA_METHOD,
            "ra_hidden_override": RA_HIDDEN_OVERRIDE,
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
            prefixed = {f"{sdg_label}/{k}": v for k, v in metrics.items()}
            all_results.update(prefixed)
            wandb.log(prefixed)
        except Exception as e:
            print(f"\n  ERROR for {sdg_label}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    wandb.finish()
    print(f"\nAll comparisons complete.")
