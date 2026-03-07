#!/usr/bin/env python3
"""
experiment_scripts/run_chaining_analysis.py

Analyzes the effect of chaining on feature-level reconstruction accuracy
for RandomForest and MLP attacks on adult (categorical) and california
(continuous) datasets.

Chaining strategies evaluated:
  - "correlation":          predict most QI-correlated features first
  - "reverse_correlation":  predict least QI-correlated features first
  - "mutual_info":          predict highest MI-with-QI features first
  - "dynamic_mutual_info":  greedy: at each step pick remaining feature with
                            highest MI with the *current* known set (QI + prev predicted)
  - "random":               N_RANDOM_ORDERINGS random orderings per sample

For each deterministic strategy, three variants are run:
  - hard:   predicted class used directly as next feature (standard chaining)
  - soft:   predicted class probabilities (one-hot in training, RF probas in inference)
            — categorical data + RandomForest only; falls back to hard otherwise
  - oracle: TRUE hidden values used as conditioning — upper bound on chaining benefit

Deterministic strategies use an order computed once from sample_00's synth data
and applied identically to all samples, enabling meaningful cross-sample averaging.

For "random", results are presented anonymized by chain position (position k =
avg RA across whichever feature occupied position k across all runs).

Output:
  1. Console: per-feature RA tables + chain-position divergence tables
  2. CSV:     experiment_scripts/chaining_analysis/{dataset}/{attack}_*.csv

For categorical data, also computes modified RA:
  mod_RA = (RA/100 - 1/num_classes) / (1 - 1/num_classes)
  Normalizes random-chance baseline to 0, keeps perfect score at 1.
"""

# ── Path setup (must precede all project imports) ────────────────────────────
import sys
import os

RECON_ROOT = '/home/golobs/Reconstruction'
if RECON_ROOT not in sys.path:
    sys.path.insert(0, RECON_ROOT)

for _p in [
    '/home/golobs/MIA_on_diffusion/',
    '/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM',
    '/home/golobs/recon-synth',
    '/home/golobs/recon-synth/attacks',
    '/home/golobs/recon-synth/attacks/solvers',
]:
    if _p not in sys.path:
        sys.path.append(_p)

# Mock wandb before any module that imports it is loaded
from unittest.mock import MagicMock
sys.modules['wandb'] = MagicMock()

# ── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT = '/home/golobs/data/reconstruction_data'

DATASETS = {
    "adult": {
        "name":       "adult",
        "root":       f"{DATA_ROOT}/adult/size_1000",
        "data_type":  "categorical",
        "qi_variant": "QI1",
        "sdg_method": "TabDDPM",
    },
    #"california": {
    #    "name":       "california",
    #    "root":       f"{DATA_ROOT}/california/size_1000",
    #    "data_type":  "continuous",
    #    "qi_variant": "QI1",
    #    "sdg_method": "MST_eps1",
    #},
}

SAMPLES            = [2, 3, 4]
ATTACKS            = ["MLP"]
#DET_STRATEGIES     = ["correlation", "reverse_correlation", "mutual_info", "dynamic_mutual_info"]
DET_STRATEGIES     = ["dynamic_mutual_info"]
N_RANDOM_ORDERINGS = 2   # per sample; total random runs = len(SAMPLES) * N_RANDOM_ORDERINGS

# ── Chaining variant toggles ──────────────────────────────────────────────────
# Hard chaining always runs. These add extra variants per strategy.
ORACLE_CHAINING        = True    # upper bound: condition each step on TRUE previous values
SOFT_CHAINING          = True    # probabilistic: one-hot training, probas at inference
                                 # (categorical only; falls back to hard for unsupported attacks)
CONFIDENCE_GATED_CHAINING = True  # soft chaining variant: rows below threshold get uniform
                                   # conditioning (treated as unknown) instead of propagating
                                   # a low-confidence prediction. Requires SOFT_CHAINING=True.
CONFIDENCE_THRESHOLD   = 0.7    # max-prob threshold for confidence gating

OUT_DIR = os.path.join(RECON_ROOT, 'experiment_scripts', 'chaining_analysis')

# ── Imports ───────────────────────────────────────────────────────────────────
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from attacks import get_attack
from scoring import (
    calculate_reconstruction_score,
    calculate_continuous_vals_reconstruction_score,
)
from get_data import QIs, minus_QIs
from attack_defaults import ATTACK_PARAM_DEFAULTS
from enhancements.chaining_wrapper import (
    _order_by_correlation,
    _order_by_mutual_info,
    _dynamic_order,
    _run_soft_chained_sklearn,
    _make_soft_chain_clf,
)


def compute_deterministic_order(strategy, ref_synth, qi, hidden_features,
                                data_type="categorical"):
    """
    Compute a chain order from reference synth data (sample_00).
    Applied identically to all samples for consistent cross-sample averaging.
    """
    if strategy == "correlation":
        return _order_by_correlation(ref_synth, qi, hidden_features, ascending=False)
    elif strategy == "reverse_correlation":
        return _order_by_correlation(ref_synth, qi, hidden_features, ascending=True)
    elif strategy == "mutual_info":
        return _order_by_mutual_info(ref_synth, qi, hidden_features, data_type=data_type)
    elif strategy == "dynamic_mutual_info":
        return _dynamic_order(ref_synth, qi, hidden_features, data_type=data_type)
    else:
        raise ValueError(f"Unknown deterministic strategy: {strategy!r}")


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sample_data(dataset_cfg, sample_idx):
    """Load train.csv and synth.csv for a given sample index."""
    sdir  = Path(dataset_cfg["root"]) / f"sample_{sample_idx:02d}"
    train = pd.read_csv(sdir / "train.csv")
    synth = pd.read_csv(sdir / dataset_cfg["sdg_method"] / "synth.csv")
    ds_name = dataset_cfg["name"]
    qi_key  = dataset_cfg["qi_variant"]
    return train, synth, QIs[ds_name][qi_key], minus_QIs[ds_name][qi_key]


def make_cfg(dataset_cfg, attack_name, sample_idx):
    """Build a minimal cfg dict for direct attack function calls."""
    params     = dict(ATTACK_PARAM_DEFAULTS.get(attack_name, {}))
    sample_dir = str(Path(dataset_cfg["root"]) / f"sample_{sample_idx:02d}")
    return {
        "dataset":       {"name": dataset_cfg["name"], "type": dataset_cfg["data_type"],
                          "dir": sample_dir},
        "data_type":     dataset_cfg["data_type"],
        "attack_method": attack_name,
        "attack_params": params,
    }


# ── Attack runners ────────────────────────────────────────────────────────────

def run_attack(attack_fn, cfg, synth, train, qi, hidden_features):
    """Non-chained baseline. Returns reconstructed DataFrame."""
    recon, _, _ = attack_fn(cfg, synth, train, qi, hidden_features)
    return recon


def run_chained_attack(attack_fn, cfg, synth, train, qi, chain_order):
    """
    Hard chaining: at each step the hard predicted class is appended to the
    known feature set and used as conditioning for the next step.
    """
    reconstructed = train.copy()
    known = list(qi)
    for feature in chain_order:
        step, _, _ = attack_fn(cfg, synth, reconstructed[known], known, [feature])
        reconstructed[feature] = step[feature]
        known.append(feature)
    return reconstructed


def run_oracle_chained_attack(attack_fn, cfg, synth, train, qi, chain_order):
    """
    Oracle chaining: TRUE hidden feature values are used as conditioning at each
    step instead of predicted values.

    This is an upper bound — it shows how much chaining *could* help if all
    previous predictions were perfect. The returned DataFrame still contains
    the attack's *predictions* for each feature (not the true values), so
    scoring against train reveals the gain from perfect context.
    """
    reconstructed = train.copy()
    known = list(qi)
    for feature in chain_order:
        # train[known] has TRUE values for both QI and previously chained features
        step, _, _ = attack_fn(cfg, synth, train[known], known, [feature])
        reconstructed[feature] = step[feature]
        known.append(feature)
    return reconstructed


def run_soft_chained_attack(attack_name, cfg, synth, train, qi, chain_order,
                            confidence_threshold=None):
    """
    Soft/probabilistic chaining dispatcher.
    Delegates to _run_soft_chained_sklearn (from chaining_wrapper) for any
    supported categorical classifier (RF, MLP, LightGBM, LR, NaiveBayes, KNN).
    Falls back to hard chaining for continuous data or unsupported attacks.

    confidence_threshold: if set, passed to _run_soft_chained_sklearn to enable
      confidence gating (low-confidence rows get uniform conditioning for next step).
    """
    data_type = cfg["data_type"]
    if data_type == "continuous":
        attack_fn = get_attack(attack_name, data_type)
        return run_chained_attack(attack_fn, cfg, synth, train, qi, chain_order)
    clf_check = _make_soft_chain_clf(attack_name, cfg.get("attack_params", {}))
    if clf_check is None:
        print(f"      [WARN] Soft chaining not supported for {attack_name}; "
              f"falling back to hard chaining")
        attack_fn = get_attack(attack_name, data_type)
        return run_chained_attack(attack_fn, cfg, synth, train, qi, chain_order)
    return _run_soft_chained_sklearn(
        attack_name, cfg.get("attack_params", {}), synth, train, qi, chain_order,
        confidence_threshold=confidence_threshold)


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_features(train, reconstructed, hidden_features, data_type):
    """
    Return {feature: score} for all hidden features.
    Categorical: rarity-weighted accuracy in [0, 100].
    Continuous:  normalized RMSE in [0, 1] (lower = better).
    """
    if data_type == "categorical":
        scores = calculate_reconstruction_score(train, reconstructed, hidden_features)
        return dict(zip(hidden_features, scores))
    else:
        df = calculate_continuous_vals_reconstruction_score(train, reconstructed, hidden_features)
        return {f: df.loc[f, "normalized_rmse"] for f in hidden_features}


def modified_ra(ra_pct, num_classes):
    """
    Normalize categorical RA so random-chance baseline = 0, perfect = 1.
      ra_pct:      rarity-weighted accuracy in [0, 100]
      num_classes: number of distinct values for this feature
    """
    baseline = 1.0 / num_classes
    denom    = 1.0 - baseline
    return float('nan') if denom == 0 else (ra_pct / 100.0 - baseline) / denom


def compute_num_classes(train, features):
    """Return {feature: n_unique} computed from the training dataframe."""
    return {f: train[f].nunique() for f in features}


# ── Condition/label helpers ───────────────────────────────────────────────────

_STRATEGY_LABELS = {
    "correlation":          "Corr",
    "reverse_correlation":  "RevCorr",
    "mutual_info":          "MI",
    "dynamic_mutual_info":  "DynMI",
}


def _get_conditions(data_type):
    """
    Build the ordered list of condition keys and display labels for tables/CSVs.
    Varies by data_type: soft chaining is only defined for categorical data.
    """
    conds, labels = ["none"], ["No Chain"]
    for s in DET_STRATEGIES:
        lbl = _STRATEGY_LABELS.get(s, s)
        conds.append(s);                          labels.append(lbl)
        if SOFT_CHAINING and data_type == "categorical":
            conds.append(f"{s}_soft");            labels.append(f"{lbl}(soft)")
            if CONFIDENCE_GATED_CHAINING:
                conds.append(f"{s}_soft_gated");  labels.append(f"{lbl}(soft-gated)")
        if ORACLE_CHAINING:
            conds.append(f"{s}_oracle");          labels.append(f"{lbl}(oracle)")
    conds.append("random_avg");                   labels.append("Rand")
    if SOFT_CHAINING and data_type == "categorical":
        conds.append("random_soft_avg");          labels.append("Rand(soft)")
        if CONFIDENCE_GATED_CHAINING:
            conds.append("random_soft_gated_avg"); labels.append("Rand(soft-gated)")
    if ORACLE_CHAINING:
        conds.append("random_oracle_avg");        labels.append("Rand(oracle)")
    return conds, labels


# ── Main experiment driver ────────────────────────────────────────────────────

def run_dataset(dataset_key, dataset_cfg):
    """
    Run all attacks × chaining conditions for one dataset.

    Returns
    -------
    avg_scores      : {attack: {condition: {feature: avg_score}}}
    random_pos_avg  : {attack: {pos_idx: {"hard": float, "soft": float, "oracle": float}}}
    num_classes_map : {feature: int}  (categorical only, else {})
    det_orders      : {strategy: [feature, ...]}
    hidden_features : list[str]
    """
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_key}")
    print(f"{'='*70}")

    data_type = dataset_cfg["data_type"]

    # Load reference data (sample_00) to fix deterministic chain orders
    _, ref_synth, qi, hidden_features = load_sample_data(dataset_cfg, 0)
    n_feats = len(hidden_features)

    print(f"  QI ({len(qi)}): {qi}")
    print(f"  Hidden ({n_feats}): {hidden_features}")

    det_orders = {}
    for strategy in DET_STRATEGIES:
        det_orders[strategy] = compute_deterministic_order(
            strategy, ref_synth, qi, hidden_features, data_type=data_type)
        print(f"  {strategy} order: {det_orders[strategy]}")

    # ── Accumulators ──────────────────────────────────────────────────────────
    all_det_conds = ["none"]
    for s in DET_STRATEGIES:
        all_det_conds.append(s)
        if SOFT_CHAINING:
            all_det_conds.append(f"{s}_soft")
            if CONFIDENCE_GATED_CHAINING:
                all_det_conds.append(f"{s}_soft_gated")
        if ORACLE_CHAINING:
            all_det_conds.append(f"{s}_oracle")

    det_scores = {
        atk: {cond: defaultdict(list) for cond in all_det_conds}
        for atk in ATTACKS
    }
    rand_feat = {atk: {"hard": defaultdict(list), "soft": defaultdict(list),
                        "soft_gated": defaultdict(list), "oracle": defaultdict(list)}
                 for atk in ATTACKS}
    rand_pos  = {atk: {p: {"hard": [], "soft": [], "soft_gated": [], "oracle": []}
                        for p in range(n_feats)}
                 for atk in ATTACKS}
    all_num_classes = defaultdict(list)

    # ── Sample loop ───────────────────────────────────────────────────────────
    for sample_idx in SAMPLES:
        print(f"\n  --- Sample {sample_idx:02d} ---")
        train, synth, qi, hidden_features = load_sample_data(dataset_cfg, sample_idx)

        if data_type == "categorical":
            for f, n in compute_num_classes(train, hidden_features).items():
                all_num_classes[f].append(n)

        for attack_name in ATTACKS:
            print(f"    [{attack_name}]")
            cfg       = make_cfg(dataset_cfg, attack_name, sample_idx)
            attack_fn = get_attack(attack_name, data_type)

            # Non-chained baseline
            recon_nc  = run_attack(attack_fn, cfg, synth, train, qi, hidden_features)
            nc_scores = score_features(train, recon_nc, hidden_features, data_type)
            for f, s in nc_scores.items():
                det_scores[attack_name]["none"][f].append(s)
            print(f"      no chain: { {f: round(s,2) for f,s in nc_scores.items()} }")

            # Deterministic chaining strategies (hard + optional soft/oracle)
            for strategy, order in det_orders.items():
                recon_h  = run_chained_attack(attack_fn, cfg, synth, train, qi, order)
                h_scores = score_features(train, recon_h, hidden_features, data_type)
                for f, s in h_scores.items():
                    det_scores[attack_name][strategy][f].append(s)
                print(f"      {strategy} (hard):   "
                      f"{ {f: round(s,2) for f,s in h_scores.items()} }")

                if SOFT_CHAINING:
                    recon_s  = run_soft_chained_attack(
                        attack_name, cfg, synth, train, qi, order)
                    s_scores = score_features(train, recon_s, hidden_features, data_type)
                    for f, s in s_scores.items():
                        det_scores[attack_name][f"{strategy}_soft"][f].append(s)
                    print(f"      {strategy} (soft):   "
                          f"{ {f: round(s,2) for f,s in s_scores.items()} }")

                    if CONFIDENCE_GATED_CHAINING:
                        recon_sg  = run_soft_chained_attack(
                            attack_name, cfg, synth, train, qi, order,
                            confidence_threshold=CONFIDENCE_THRESHOLD)
                        sg_scores = score_features(train, recon_sg, hidden_features, data_type)
                        for f, s in sg_scores.items():
                            det_scores[attack_name][f"{strategy}_soft_gated"][f].append(s)
                        print(f"      {strategy} (soft-gated,τ={CONFIDENCE_THRESHOLD}): "
                              f"{ {f: round(s,2) for f,s in sg_scores.items()} }")

                if ORACLE_CHAINING:
                    recon_o  = run_oracle_chained_attack(
                        attack_fn, cfg, synth, train, qi, order)
                    o_scores = score_features(train, recon_o, hidden_features, data_type)
                    for f, s in o_scores.items():
                        det_scores[attack_name][f"{strategy}_oracle"][f].append(s)
                    print(f"      {strategy} (oracle): "
                          f"{ {f: round(s,2) for f,s in o_scores.items()} }")

            # Random chaining
            for rand_i in range(N_RANDOM_ORDERINGS):
                seed  = sample_idx * 1000 + rand_i
                rng   = random.Random(seed)
                order = hidden_features.copy()
                rng.shuffle(order)

                recon_rh = run_chained_attack(attack_fn, cfg, synth, train, qi, order)
                rh_scores = score_features(train, recon_rh, hidden_features, data_type)

                recon_rs = (run_soft_chained_attack(attack_name, cfg, synth, train, qi, order)
                            if SOFT_CHAINING else None)
                rs_scores = (score_features(train, recon_rs, hidden_features, data_type)
                             if SOFT_CHAINING else {})

                recon_rsg = (run_soft_chained_attack(
                                 attack_name, cfg, synth, train, qi, order,
                                 confidence_threshold=CONFIDENCE_THRESHOLD)
                             if SOFT_CHAINING and CONFIDENCE_GATED_CHAINING else None)
                rsg_scores = (score_features(train, recon_rsg, hidden_features, data_type)
                              if SOFT_CHAINING and CONFIDENCE_GATED_CHAINING else {})

                recon_ro = (run_oracle_chained_attack(attack_fn, cfg, synth, train, qi, order)
                            if ORACLE_CHAINING else None)
                ro_scores = (score_features(train, recon_ro, hidden_features, data_type)
                             if ORACLE_CHAINING else {})

                for pos_idx, feat in enumerate(order):
                    rand_feat[attack_name]["hard"][feat].append(rh_scores[feat])
                    rand_pos[attack_name][pos_idx]["hard"].append(rh_scores[feat])
                    if SOFT_CHAINING:
                        rand_feat[attack_name]["soft"][feat].append(rs_scores[feat])
                        rand_pos[attack_name][pos_idx]["soft"].append(rs_scores[feat])
                    if SOFT_CHAINING and CONFIDENCE_GATED_CHAINING:
                        rand_feat[attack_name]["soft_gated"][feat].append(rsg_scores[feat])
                        rand_pos[attack_name][pos_idx]["soft_gated"].append(rsg_scores[feat])
                    if ORACLE_CHAINING:
                        rand_feat[attack_name]["oracle"][feat].append(ro_scores[feat])
                        rand_pos[attack_name][pos_idx]["oracle"].append(ro_scores[feat])

            print(f"      random ({N_RANDOM_ORDERINGS} orderings): done")

    # ── Average across samples / runs ─────────────────────────────────────────
    num_classes_map = {f: round(np.mean(vs)) for f, vs in all_num_classes.items()}

    avg_scores = {}
    for atk in ATTACKS:
        avg_scores[atk] = {}
        for cond in all_det_conds:
            avg_scores[atk][cond] = {
                f: float(np.mean(vals))
                for f, vals in det_scores[atk][cond].items()
            }
        avg_scores[atk]["random_avg"] = {
            f: float(np.mean(v)) for f, v in rand_feat[atk]["hard"].items()
        }
        if SOFT_CHAINING:
            avg_scores[atk]["random_soft_avg"] = {
                f: float(np.mean(v)) for f, v in rand_feat[atk]["soft"].items()
            }
        if SOFT_CHAINING and CONFIDENCE_GATED_CHAINING:
            avg_scores[atk]["random_soft_gated_avg"] = {
                f: float(np.mean(v)) for f, v in rand_feat[atk]["soft_gated"].items()
            }
        if ORACLE_CHAINING:
            avg_scores[atk]["random_oracle_avg"] = {
                f: float(np.mean(v)) for f, v in rand_feat[atk]["oracle"].items()
            }

    random_pos_avg = {
        atk: {
            pos: {
                "hard":       float(np.mean(d["hard"])),
                "soft":       float(np.mean(d["soft"]))       if SOFT_CHAINING                        else float('nan'),
                "soft_gated": float(np.mean(d["soft_gated"])) if SOFT_CHAINING and CONFIDENCE_GATED_CHAINING else float('nan'),
                "oracle":     float(np.mean(d["oracle"]))     if ORACLE_CHAINING                      else float('nan'),
            }
            for pos, d in pos_dict.items()
        }
        for atk, pos_dict in rand_pos.items()
    }

    return avg_scores, random_pos_avg, num_classes_map, det_orders, hidden_features


# ── Presentation ──────────────────────────────────────────────────────────────

def print_per_feature_table(dataset_key, avg_scores, num_classes_map,
                             hidden_features, data_type):
    """
    Table 1: one row per feature, columns = all conditions (all variants).
    For categorical, also shows modified RA columns.
    """
    is_cat = data_type == "categorical"
    metric  = "RA (%)" if is_cat else "norm_RMSE (lower=better)"
    conds, cond_labels = _get_conditions(data_type)

    for attack_name in ATTACKS:
        scores = avg_scores[attack_name]
        print(f"\n{'='*90}")
        print(f"[{dataset_key.upper()}]  {attack_name}  —  Per-feature {metric}")
        print(f"  Averaged over {len(SAMPLES)} samples  |  SDG: {DATASETS[dataset_key]['sdg_method']}")
        print(f"{'='*90}")

        FW, CW = 20, 11
        hdr = f"{'Feature':<{FW}}"
        for lbl in cond_labels:
            hdr += f"  {lbl:>{CW}}"
        if is_cat:
            for lbl in cond_labels:
                hdr += f"  {'m_'+lbl:>{CW}}"
        print(hdr)
        print("-" * len(hdr))

        for feat in hidden_features:
            raw_vals = [scores.get(c, {}).get(feat, float('nan')) for c in conds]
            row = f"{feat:<{FW}}"
            for v in raw_vals:
                row += f"  {v:>{CW}.3f}"
            if is_cat:
                nc = num_classes_map.get(feat, 2)
                for v in raw_vals:
                    mv = modified_ra(v, nc) if not np.isnan(v) else float('nan')
                    row += f"  {mv:>{CW}.3f}"
            print(row)


def print_chain_position_table(dataset_key, avg_scores, random_pos_avg,
                                det_orders, num_classes_map, hidden_features,
                                data_type):
    """
    Table 2: chain-position divergence.
    Deterministic strategies: features in chain order, No-Chain / Hard / Soft / Oracle.
    Random strategy: anonymized positions, same columns averaged over all runs.
    """
    is_cat  = data_type == "categorical"
    n_feats = len(hidden_features)

    for attack_name in ATTACKS:
        scores = avg_scores[attack_name]

        for strategy in DET_STRATEGIES:
            order = det_orders[strategy]
            print(f"\n{'='*90}")
            print(f"[{dataset_key.upper()}]  {attack_name}  —  Chain divergence: {strategy}")
            print(f"  Order fixed from sample_00 synth; averaged over {len(SAMPLES)} samples")
            print(f"{'='*90}")

            FW, CW = 20, 11
            hdr = f"{'Pos':<4}  {'Feature':<{FW}}  {'No Chain':>{CW}}  {'Hard':>{CW}}"
            if SOFT_CHAINING and data_type == "categorical":
                hdr += f"  {'Soft':>{CW}}"
                if CONFIDENCE_GATED_CHAINING:
                    hdr += f"  {'Soft-Gated':>{CW}}"
            if ORACLE_CHAINING:
                hdr += f"  {'Oracle':>{CW}}"
            print(hdr)
            print("-" * len(hdr))

            for pos_idx, feat in enumerate(order):
                base   = scores["none"].get(feat, float('nan'))
                hard   = scores[strategy].get(feat, float('nan'))
                row    = f"{pos_idx+1:<4}  {feat:<{FW}}  {base:>{CW}.3f}  {hard:>{CW}.3f}"
                if SOFT_CHAINING and data_type == "categorical":
                    soft = scores.get(f"{strategy}_soft", {}).get(feat, float('nan'))
                    row += f"  {soft:>{CW}.3f}"
                    if CONFIDENCE_GATED_CHAINING:
                        sg = scores.get(f"{strategy}_soft_gated", {}).get(feat, float('nan'))
                        row += f"  {sg:>{CW}.3f}"
                if ORACLE_CHAINING:
                    oracle = scores.get(f"{strategy}_oracle", {}).get(feat, float('nan'))
                    row += f"  {oracle:>{CW}.3f}"
                print(row)

        # Random strategy (anonymized by position)
        pos_data = random_pos_avg[attack_name]
        n_total  = len(SAMPLES) * N_RANDOM_ORDERINGS
        print(f"\n{'='*90}")
        print(f"[{dataset_key.upper()}]  {attack_name}  —  Chain divergence: random")
        print(f"  {n_total} orderings ({N_RANDOM_ORDERINGS}/sample × {len(SAMPLES)} samples)")
        print(f"  Scores anonymized: averaged across whichever feature occupied each position")
        print(f"{'='*90}")

        CW  = 13
        hdr = f"{'Pos':<5}  {'Hard':>{CW}}"
        if SOFT_CHAINING and data_type == "categorical":
            hdr += f"  {'Soft':>{CW}}"
            if CONFIDENCE_GATED_CHAINING:
                hdr += f"  {'Soft-Gated':>{CW}}"
        if ORACLE_CHAINING:
            hdr += f"  {'Oracle':>{CW}}"
        print(hdr)
        print("-" * len(hdr))

        for pos_idx in range(n_feats):
            d   = pos_data[pos_idx]
            row = f"{pos_idx+1:<5}  {d['hard']:>{CW}.3f}"
            if SOFT_CHAINING and data_type == "categorical":
                row += f"  {d['soft']:>{CW}.3f}"
                if CONFIDENCE_GATED_CHAINING:
                    row += f"  {d['soft_gated']:>{CW}.3f}"
            if ORACLE_CHAINING:
                row += f"  {d['oracle']:>{CW}.3f}"
            print(row)


# ── CSV output ────────────────────────────────────────────────────────────────

def save_csvs(dataset_key, avg_scores, random_pos_avg, det_orders,
              num_classes_map, hidden_features, data_type, out_dir):
    """Write per-feature and random-position results to CSV files."""
    out_path = Path(out_dir) / dataset_key
    out_path.mkdir(parents=True, exist_ok=True)
    is_cat = data_type == "categorical"
    conds, _ = _get_conditions(data_type)

    for attack_name in ATTACKS:
        scores = avg_scores[attack_name]

        # Per-feature table
        rows = []
        for feat in hidden_features:
            row = {"feature": feat}
            nc  = num_classes_map.get(feat, 2) if is_cat else None
            for cond in conds:
                s = scores.get(cond, {}).get(feat, float('nan'))
                row[f"score_{cond}"] = round(s, 5)
                if is_cat and not np.isnan(s):
                    row[f"mod_{cond}"] = round(modified_ra(s, nc), 5)
            if is_cat:
                row["num_classes"] = nc
            rows.append(row)
        p = out_path / f"{attack_name}_per_feature.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        print(f"  Saved: {p}")

        # Per-strategy chain-position CSVs
        for strategy in DET_STRATEGIES:
            order  = det_orders[strategy]
            rows_c = []
            for pos_idx, feat in enumerate(order):
                nc = num_classes_map.get(feat, 2) if is_cat else None
                r  = {
                    "position": pos_idx + 1,
                    "feature":  feat,
                    "no_chain": round(scores["none"].get(feat, float('nan')), 5),
                    "hard":     round(scores[strategy].get(feat, float('nan')), 5),
                }
                if SOFT_CHAINING:
                    sv = scores.get(f"{strategy}_soft", {}).get(feat, float('nan'))
                    r["soft"] = round(sv, 5)
                if SOFT_CHAINING and CONFIDENCE_GATED_CHAINING:
                    sgv = scores.get(f"{strategy}_soft_gated", {}).get(feat, float('nan'))
                    r["soft_gated"] = round(sgv, 5)
                if ORACLE_CHAINING:
                    ov = scores.get(f"{strategy}_oracle", {}).get(feat, float('nan'))
                    r["oracle"] = round(ov, 5)
                    if is_cat and not np.isnan(ov):
                        r["mod_oracle"] = round(modified_ra(ov, nc), 5)
                rows_c.append(r)
            p_c = out_path / f"{attack_name}_chain_{strategy}.csv"
            pd.DataFrame(rows_c).to_csv(p_c, index=False)
            print(f"  Saved: {p_c}")

        # Random chain-position CSV
        rows_r = []
        for pos_idx in range(len(hidden_features)):
            d = random_pos_avg[attack_name][pos_idx]
            r = {"position": pos_idx + 1, "hard": round(d["hard"], 5)}
            if SOFT_CHAINING:
                r["soft"]   = round(d["soft"],   5)
            if SOFT_CHAINING and CONFIDENCE_GATED_CHAINING:
                r["soft_gated"] = round(d["soft_gated"], 5)
            if ORACLE_CHAINING:
                r["oracle"] = round(d["oracle"], 5)
            rows_r.append(r)
        p_r = out_path / f"{attack_name}_chain_random_positions.csv"
        pd.DataFrame(rows_r).to_csv(p_r, index=False)
        print(f"  Saved: {p_r}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_results = {}
    for dataset_key, dataset_cfg in DATASETS.items():
        all_results[dataset_key] = run_dataset(dataset_key, dataset_cfg)

    for dataset_key, dataset_cfg in DATASETS.items():
        avg_scores, random_pos_avg, num_classes_map, det_orders, hidden_features = \
            all_results[dataset_key]
        data_type = dataset_cfg["data_type"]

        print_per_feature_table(
            dataset_key, avg_scores, num_classes_map, hidden_features, data_type)
        print_chain_position_table(
            dataset_key, avg_scores, random_pos_avg, det_orders,
            num_classes_map, hidden_features, data_type)

        print(f"\n--- Saving CSVs for {dataset_key} ---")
        save_csvs(dataset_key, avg_scores, random_pos_avg, det_orders,
                  num_classes_map, hidden_features, data_type, OUT_DIR)

    print(f"\nDone. Results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
