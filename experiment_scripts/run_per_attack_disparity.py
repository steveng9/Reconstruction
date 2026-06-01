#!/usr/bin/env python
"""
Per-attack disparity analysis.

Question (reviewer/\steven TODO): is the demographic / outlier disparity in
Table tab:disparate_impact a property of the *attack* or of the *SDG mechanism*?

We reuse analyze_ra_subgroups.run_analysis but sweep the ATTACK_METHOD across the
fast classifier attacks, for each high-/low-disparity SDG method, on the same
setting as the main disparate-impact table (adult 10k, QI1, sample_01, RF row).

For each (attack, SDG) we record:
  - mean row-level R_adv
  - outlier penalty  = outlier_mean / non_outlier_mean   (the "x" column)
  - per-race means (AI/AN, API, Black, White, Other)

Output: experiment_scripts/per_attack_disparity_<ts>.csv  + console summary.

Usage:
    conda activate recon_
    python experiment_scripts/run_per_attack_disparity.py
    python experiment_scripts/run_per_attack_disparity.py --attack KNN --sdg TabDDPM
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/golobs/Reconstruction")

# Parse our args BEFORE importing analyze_ra_subgroups, because that module
# imports master_experiment_script, which calls parse_args() at import time and
# would otherwise choke on our flags. We stash the parsed args and clear argv.
_ap = argparse.ArgumentParser()
_ap.add_argument("--attack", default=None, help="restrict to one attack")
_ap.add_argument("--sdg", default=None, help="restrict to one SDG label")
_ap.add_argument("--out", default=None)
_ARGS = _ap.parse_args()
sys.argv = sys.argv[:1]

import pandas as pd

import analyze_ra_subgroups as ars
from attack_defaults import ATTACK_PARAM_DEFAULTS

# Match the disparate-impact table exactly (adult 10k, QI1, sample_01).
ars.SAMPLE_SIZE = 10_000
ars.SAMPLE_DIR = f"{ars.DATA_ROOT}{ars.DATASET}/size_{ars.SAMPLE_SIZE}/sample_01"
ars.OUTLIER_PERCENTILE = 90
# Save per-record row-score CSVs (QI values + per-feature RA_row_* + is_outlier)
# so feature-type disparity can be analysed downstream.
ars.SAVE_CSV = True
ars.OUTPUT_DIR = "/home/golobs/Reconstruction/experiment_scripts/disparity_rowscores"
# Smaller floor so minority race groups (AI/AN ~110, API ~310) still report.
ars.MIN_GROUP_SIZE = 20

ATTACKS = [
    "Mode",
    "KNN",
    "NaiveBayes",
    "RandomForest",
    "CoBP-RA_graphQI_entropyBP",
    "MLP",
    "LightGBM",
]

# (label, method, params) — ALL main SDG methods, so we can see whether a given
# attack (e.g. CoBP-RA) exposes disparity under one generator but not another.
SDGS = [
    ("CellSuppression", "CellSuppression", {}),
    ("RankSwap",        "RankSwap",        {}),
    ("Synthpop",        "Synthpop",        {}),
    ("TabDDPM",         "TabDDPM",         {}),
    ("CTGAN",           "CTGAN",           {}),
    ("TVAE",            "TVAE",            {}),
    ("ARF",             "ARF",             {}),
    ("AIM_eps1",        "AIM",             {"epsilon": 1.0}),
    ("MST_eps1",        "MST",             {"epsilon": 1.0}),
    ("MST_eps1000",     "MST",             {"epsilon": 1000.0}),
]

# IPUMS/adult race codes are strings here ('White', 'Black', ...). Map to short labels.
RACE_ALIASES = {
    "Amer-Indian-Eskimo": "AI/AN",
    "Asian-Pac-Islander": "API",
    "Black": "Black",
    "White": "White",
    "Other": "Other",
}


def _race_means(subgroup_res):
    out = {}
    if not subgroup_res or "race" not in subgroup_res:
        return out
    for _, row in subgroup_res["race"].iterrows():
        label = RACE_ALIASES.get(str(row["value"]).strip(), str(row["value"]).strip())
        out[label] = float(row["mean_ra"])
    return out


def _sex_means(subgroup_res):
    out = {}
    if not subgroup_res or "sex" not in subgroup_res:
        return out
    for _, row in subgroup_res["sex"].iterrows():
        out[str(row["value"]).strip()] = float(row["mean_ra"])
    return out


def run_one(attack, sdg_label, sdg_method, sdg_params):
    ars.ATTACK_METHOD = attack
    ars.ATTACK_PARAMS = dict(ATTACK_PARAM_DEFAULTS.get(attack, {}))
    res = ars.run_analysis(sdg_method, sdg_params)

    o = res["outlier_res"] or {}
    om = o.get("outlier_mean", float("nan"))
    nm = o.get("non_outlier_mean", float("nan"))
    ov = o.get("overall_mean", float("nan"))
    penalty = (om / nm) if (nm and nm == nm and nm != 0) else float("nan")

    summary = {
        "sdg": sdg_label,
        "attack": attack,
        "mean_ra": ov,
        "outlier_mean": om,
        "non_outlier_mean": nm,
        "outlier_penalty": penalty,
        "mannwhitney_p": o.get("mannwhitney_p", float("nan")),
    }
    summary.update({f"race_{k}": v for k, v in _race_means(res["subgroup_res"]).items()})
    summary.update({f"sex_{k}": v for k, v in _sex_means(res["subgroup_res"]).items()})

    # Per-feature outlier vs non-outlier breakdown → lets us ask which feature
    # types (categorical vs continuous, high- vs low-cardinality) carry the gap.
    feat_rows = []
    pf = res["per_feat_df"]
    if pf is not None and not pf.empty:
        for _, r in pf.iterrows():
            feat_rows.append({
                "sdg": sdg_label,
                "attack": attack,
                "feature": r["feature"],
                "outlier_mean": r["outlier_mean"],
                "non_outlier_mean": r["non_outlier_mean"],
                "diff": r["diff"],
            })
    return summary, feat_rows


def main():
    args = _ARGS

    attacks = [args.attack] if args.attack else ATTACKS
    sdgs = [s for s in SDGS if (args.sdg is None or s[0] == args.sdg)]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_summary = args.out or f"experiment_scripts/per_attack_disparity_{ts}.csv"
    out_perfeat = out_summary.replace(".csv", "_perfeat.csv")
    Path(ars.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    rows = []
    feat_rows = []
    for sdg_label, sdg_method, sdg_params in sdgs:
        for attack in attacks:
            try:
                summary, feats = run_one(attack, sdg_label, sdg_method, sdg_params)
                rows.append(summary)
                feat_rows.extend(feats)
                # Incremental save so a crash mid-sweep still leaves usable data.
                pd.DataFrame(rows).to_csv(out_summary, index=False)
                pd.DataFrame(feat_rows).to_csv(out_perfeat, index=False)
            except Exception as e:
                print(f"\n  ERROR {attack} / {sdg_label}: {e}")
                import traceback
                traceback.print_exc()

    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("PER-ATTACK DISPARITY SUMMARY (adult 10k, QI1, sample_01)")
    print("=" * 78)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    cols = ["sdg", "attack", "mean_ra", "outlier_penalty"] + [c for c in df.columns if c.startswith("race_")]
    print(df[cols].to_string(index=False))
    print(f"\nSummary  → {out_summary}")
    print(f"Per-feat → {out_perfeat}")
    print(f"Row CSVs → {ars.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
