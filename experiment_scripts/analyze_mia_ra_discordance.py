#!/usr/bin/env python
"""Who do membership inference and RA-as-MIA disagree on?

Reproduces the quadrant split of compare_mia_ra.py (TabDDPM, Adult 10k,
sample_01 members / sample_02 holdout, RF RA-as-MIA with the wide QI) and
describes the records in each quadrant: QI-outlier rate, race and sex mix,
and how rare their hidden values are. No WandB; prints to stdout.

Usage:
    DISCORD_SDG=TabDDPM python experiment_scripts/analyze_mia_ra_discordance.py
"""
import os, sys, pathlib
sys.argv = sys.argv[:1]  # master_experiment_script parses argv on import
for _anc in pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        sys.path.insert(0, str(_anc))
        break

import numpy as np
import pandas as pd

import experiment_scripts.compare_mia_ra as C
from paths import DATA_ROOT
from get_data import load_mia_data, QIs, minus_QIs
from attacks import get_attack, ra_as_mia
from attacks.mia import synth_distance_mia, nndr_mia
from master_experiment_script import _prepare_config
from scoring import compute_outlier_scores

C.SAMPLE_DIR = f"{DATA_ROOT}/{C.DATASET}/size_{C.SAMPLE_SIZE}/sample_01"
C.HOLDOUT_DIR = f"{DATA_ROOT}/{C.DATASET}/size_{C.SAMPLE_SIZE}/sample_02"


def rarity(df, cols, ref):
    """Mean over cols of 1/freq(value) in ref, per row (higher = rarer values)."""
    out = np.zeros(len(df))
    for c in cols:
        freq = ref[c].value_counts(normalize=True)
        out += 1.0 / df[c].map(freq).fillna(freq.min()).values
    return out / len(cols)


def describe(name, mask, members, outlier, hid_rarity, qi_rarity, synth_dist):
    sub = members[mask]
    race = sub["race"].value_counts(normalize=True).reindex(
        members["race"].value_counts().index).fillna(0)
    print(f"  {name:<10} n={mask.sum():5d}  outlier={outlier[mask].mean()*100:5.1f}%"
          f"  hidden-rarity={np.median(hid_rarity[mask]):6.2f}"
          f"  QI-rarity={np.median(qi_rarity[mask]):6.2f}"
          f"  NN-dist={np.median(synth_dist[mask]):.3f}"
          f"  female={(sub['sex'].astype(str).str.contains('Female')).mean()*100:4.1f}%"
          f"  race=" + ", ".join(f"{k}:{v*100:.1f}" for k, v in race.items()))


def run(sdg):
    cfg = C.make_config(sdg, {})
    prepared = _prepare_config(cfg)
    train, synth, holdout, meta = load_mia_data(cfg)
    qi = QIs[C.DATASET][C.QI]
    ra_hidden = C.RA_HIDDEN_OVERRIDE

    sd_metrics, sd_scores, labels, all_targets = synth_distance_mia(
        cfg, synth, train, holdout, meta, return_raw=True)
    nn_metrics, nn_scores, _, _ = nndr_mia(cfg, synth, train, holdout, meta, return_raw=True)
    qi_wide = [f for f in train.columns if f not in ra_hidden]
    ra_metrics, ra_scores, _, _ = ra_as_mia(
        get_attack(C.RA_METHOD, C.DATA_TYPE), prepared, synth, train, holdout,
        qi_wide, ra_hidden, n_targets=C.N_TARGETS, seed=C.SEED)

    qi_cat, qi_num = C._infer_cat_num(all_targets, qi)
    _, o_flags = compute_outlier_scores(all_targets, qi, qi_cat, qi_num,
                                        method="isolation_forest", percentile=C.OUTLIER_PCT)
    outlier_all = o_flags.values.astype(bool)

    print(f"\n=== {sdg}: AUC SynthDistance={sd_metrics['MIA_auc']:.3f} "
          f"NNDR={nn_metrics['MIA_auc']:.3f} RA-as-MIA={ra_metrics['RA_as_MIA_auc']:.3f}")

    m = labels == 1
    members = all_targets[m].reset_index(drop=True)
    outlier = outlier_all[m]
    hid_r = rarity(members, ra_hidden, train)
    qi_r = rarity(members, qi_cat, train)
    for mia_name, mia_sc in [("SynthDistance", sd_scores), ("NNDR", nn_scores)]:
        q = C._quadrant_analysis(mia_sc, ra_scores, labels, mia_name)
        mt, rt = np.median(mia_sc), np.median(ra_scores)
        mh, rh = mia_sc[m] >= mt, ra_scores[m] >= rt
        print(f"\n vs {mia_name}: spearman={q['spearman_r']}  both_high={q['both_high']} "
              f"mia_only={q['mia_only']} ra_only={q['ra_only']} both_low={q['both_low']} "
              f"discordant={q['mia_only'] + q['ra_only']}")
        dist = -sd_scores[m]  # SynthDistance score is the negative NN distance
        for name, mask in [("both_high", mh & rh), ("mia_only", mh & ~rh),
                           ("ra_only", ~mh & rh), ("both_low", ~mh & ~rh), ("all", mh | ~mh)]:
            describe(name, mask, members, outlier, hid_r, qi_r, dist)


if __name__ == "__main__":
    for sdg in os.environ.get("DISCORD_SDG", "TabDDPM").split(","):
        run(sdg)
