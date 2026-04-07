#!/usr/bin/env python3
"""
experiment_scripts/run_gibbs_chaining_analysis.py

Tests Gibbs-style iterative refinement chaining against hard chaining and the
no-chain baseline, across attacks and SDG methods.

The Gibbs approach runs a standard hard chain for initialization (pass 0), then
iteratively re-predicts each hidden feature conditioning on QI + current
best-guess predictions for ALL other hidden features, repeating until convergence.

Conditions evaluated
--------------------
  no_chain   : standard attack with no chaining
  hard       : standard hard chain (= Gibbs pass 0 only)
  gibbs_K    : Gibbs chaining stopped after K passes (K in GIBBS_PASS_CHECKPOINTS)
  oracle     : hard chain with TRUE hidden values as conditioning — upper bound

Key design choices
------------------
  - Chain order fixed from sample_00's synth (dynamic_mutual_info strategy),
    applied identically to all samples for comparable cross-sample averaging.
  - Multiple SDG methods to test the interaction: chaining's benefit should scale
    with the fidelity of the joint synthetic distribution. MST only captures
    pairwise marginals — conditioning on a predicted feature adds little. TabDDPM
    and AIM preserve higher-order dependencies, so Gibbs should gain more there.
  - Per-pass convergence stats tracked and printed.

Output
------
  Console: per-feature RA tables + convergence tables per (attack, SDG) combo
  CSV:     experiment_scripts/chaining_analysis/gibbs/{dataset}/{attack}_{sdg}.csv
           experiment_scripts/chaining_analysis/gibbs/{dataset}/{attack}_{sdg}_convergence.csv
"""

# ── Path setup ────────────────────────────────────────────────────────────────
import sys, os

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

from unittest.mock import MagicMock
sys.modules['wandb'] = MagicMock()

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT = '/home/golobs/data/reconstruction_data'

DATASET = {
    "name":      "adult",
    "root":      f"{DATA_ROOT}/adult/size_1000",
    "data_type": "categorical",
    "qi_variant": "QI1",
}

# SDG method directory names to compare.
# MST (pairwise only) vs TabDDPM (deep generative, higher-order) is the key contrast.
SDG_METHODS = [
    "MST_eps10",
    "TabDDPM",
    "Synthpop",
]

ATTACKS = ["RandomForest", "MLP"]

# Samples to average over (use 0-4 for full 5-sample average; subset for speed)
SAMPLES = [0, 1, 2, 3, 4]

# Maximum Gibbs passes; also controls which intermediate checkpoints are scored.
MAX_PASSES = 10

# Score reconstruction at these pass counts (must be <= MAX_PASSES).
# "hard" = pass 0 only (standard hard chain); "gibbs_K" = after K Gibbs passes.
GIBBS_PASS_CHECKPOINTS = [1, 2, 5, 10]

# Stop early if < this fraction of predictions change in a full Gibbs sweep.
CONVERGENCE_TOL = 0.005

OUT_DIR = os.path.join(RECON_ROOT, 'experiment_scripts', 'chaining_analysis', 'gibbs')

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from attacks import get_attack
from scoring import calculate_reconstruction_score
from get_data import QIs, minus_QIs
from attack_defaults import ATTACK_PARAM_DEFAULTS
from enhancements.chaining_wrapper import (
    _dynamic_order,
    _make_soft_chain_clf,
    _run_gibbs_chained_sklearn,
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sample(sample_idx, sdg_method):
    sdir  = Path(DATASET["root"]) / f"sample_{sample_idx:02d}"
    train = pd.read_csv(sdir / "train.csv")
    synth = pd.read_csv(sdir / sdg_method / "synth.csv")
    ds    = DATASET["name"]
    qi_v  = DATASET["qi_variant"]
    return train, synth, QIs[ds][qi_v], minus_QIs[ds][qi_v]


def make_cfg(attack_name, sample_idx, sdg_method):
    params     = dict(ATTACK_PARAM_DEFAULTS.get(attack_name, {}))
    sample_dir = str(Path(DATASET["root"]) / f"sample_{sample_idx:02d}")
    return {
        "dataset":       {"name": DATASET["name"], "type": DATASET["data_type"],
                          "dir": sample_dir},
        "data_type":     DATASET["data_type"],
        "sdg_method":    sdg_method,
        "attack_method": attack_name,
        "attack_params": params,
    }


# ── Attack runners ─────────────────────────────────────────────────────────────

def run_no_chain(attack_fn, cfg, synth, train, qi, hidden_features):
    recon, _, _ = attack_fn(cfg, synth, train, qi, hidden_features)
    return recon


def run_oracle_chain(attack_fn, cfg, synth, train, qi, chain_order):
    """Hard chain using TRUE hidden values — upper bound."""
    reconstructed = train.copy()
    known = list(qi)
    for feat in chain_order:
        step, _, _ = attack_fn(cfg, synth, train[known], known, [feat])
        reconstructed[feat] = step[feat]
        known.append(feat)
    return reconstructed


def run_gibbs_at_checkpoints(attack_name, attack_params, synth, train, qi,
                              chain_order, checkpoints):
    """
    Run Gibbs chaining and return snapshots of `reconstructed` at each checkpoint.

    Returns
    -------
    snapshots : dict[int, pd.DataFrame]   checkpoint_pass → reconstructed df
    conv_stats: list[dict]                per-pass convergence info
    """
    from sklearn.preprocessing import OrdinalEncoder

    # ── Encode QI ──────────────────────────────────────────────────────────
    qi_enc     = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_synth_qi = qi_enc.fit_transform(synth[qi].astype(str))
    X_test_qi  = qi_enc.transform(train[qi].astype(str))

    feat_encoders = {}
    for feat in chain_order:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(synth[[feat]].astype(str))
        feat_encoders[feat] = enc

    def _enc_block(df, feats):
        if not feats:
            return None
        return np.hstack([feat_encoders[f].transform(df[[f]].astype(str)) for f in feats])

    # ── Pass 0: hard chain initialization ──────────────────────────────────
    reconstructed = train.copy()
    known_so_far  = []

    for feat in chain_order:
        X_s = np.hstack([X_synth_qi] + ([_enc_block(synth, known_so_far)] if known_so_far else []))
        X_t = np.hstack([X_test_qi]  + ([_enc_block(reconstructed, known_so_far)] if known_so_far else []))
        y   = synth[feat].astype(str)
        clf = _make_soft_chain_clf(attack_name, attack_params)
        clf.fit(X_s, y)
        preds = clf.classes_[np.argmax(clf.predict_proba(X_t), axis=1)]
        try:
            reconstructed[feat] = preds.astype(synth[feat].dtype)
        except (ValueError, TypeError):
            reconstructed[feat] = preds
        known_so_far.append(feat)

    # Snapshot after pass 0 (= hard chain result)
    snapshots = {0: reconstructed.copy()}

    # ── Pre-train Gibbs classifiers (once; reused across passes) ───────────
    gibbs_clfs = {}
    for feat in chain_order:
        other = [f for f in chain_order if f != feat]
        X_s   = np.hstack([X_synth_qi] + ([_enc_block(synth, other)] if other else []))
        y     = synth[feat].astype(str)
        clf   = _make_soft_chain_clf(attack_name, attack_params)
        clf.fit(X_s, y)
        gibbs_clfs[feat] = clf

    # ── Gibbs passes ────────────────────────────────────────────────────────
    conv_stats  = []
    n_targets   = len(train)
    max_ckpt    = max(checkpoints)

    for pass_idx in range(max_ckpt):
        n_changed = {}
        for feat in chain_order:
            prev  = reconstructed[feat].copy()
            other = [f for f in chain_order if f != feat]
            X_t   = np.hstack([X_test_qi] + ([_enc_block(reconstructed, other)] if other else []))
            clf   = gibbs_clfs[feat]
            preds = clf.classes_[np.argmax(clf.predict_proba(X_t), axis=1)]
            try:
                reconstructed[feat] = preds.astype(synth[feat].dtype)
            except (ValueError, TypeError):
                reconstructed[feat] = preds
            n_changed[feat] = int((reconstructed[feat].astype(str) != prev.astype(str)).sum())

        total   = n_targets * len(chain_order)
        frac    = sum(n_changed.values()) / total
        stat    = {"pass": pass_idx + 1, "frac_changed": round(frac, 6),
                   **{f: round(n_changed[f] / n_targets, 4) for f in chain_order}}
        conv_stats.append(stat)
        print(f"      Gibbs pass {pass_idx+1}/{max_ckpt}: {frac:.3%} changed")

        checkpoint_pass = pass_idx + 1
        if checkpoint_pass in checkpoints:
            snapshots[checkpoint_pass] = reconstructed.copy()

        if frac < CONVERGENCE_TOL:
            print(f"      Converged at pass {pass_idx+1}")
            # Fill remaining checkpoints with the converged result
            for ck in checkpoints:
                if ck > checkpoint_pass and ck not in snapshots:
                    snapshots[ck] = reconstructed.copy()
            break

    # Fill any remaining checkpoints (if convergence stopped early)
    for ck in checkpoints:
        if ck not in snapshots:
            snapshots[ck] = reconstructed.copy()

    return snapshots, conv_stats


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(train, reconstructed, hidden_features):
    scores = calculate_reconstruction_score(train, reconstructed, hidden_features)
    return dict(zip(hidden_features, scores))


def mean_score(feat_scores):
    return float(np.mean(list(feat_scores.values())))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load reference synth (sample_00, first SDG) once for chain order
    _, ref_synth, qi, hidden_features = load_sample(0, SDG_METHODS[0])
    chain_order = _dynamic_order(ref_synth, qi, hidden_features, data_type="categorical")
    print(f"Chain order (dynamic_MI from sample_00/{SDG_METHODS[0]}):")
    print(f"  {chain_order}\n")

    for sdg_method in SDG_METHODS:
        print(f"\n{'='*70}")
        print(f"SDG: {sdg_method}")
        print(f"{'='*70}")

        for attack_name in ATTACKS:
            print(f"\n  Attack: {attack_name}")
            print(f"  {'─'*50}")

            # Accumulators: condition → {feature: [scores across samples]}
            cond_keys = ["no_chain", "hard"] + [f"gibbs_{k}" for k in GIBBS_PASS_CHECKPOINTS] + ["oracle"]
            acc       = {c: defaultdict(list) for c in cond_keys}
            all_conv  = []   # list of conv_stats lists, one per sample

            for sample_idx in SAMPLES:
                print(f"\n    Sample {sample_idx:02d}")
                try:
                    train, synth, qi, hidden_features = load_sample(sample_idx, sdg_method)
                except FileNotFoundError as e:
                    print(f"      [SKIP] {e}")
                    continue

                cfg       = make_cfg(attack_name, sample_idx, sdg_method)
                attack_fn = get_attack(attack_name, DATASET["data_type"])

                # Check Gibbs is supported
                if _make_soft_chain_clf(attack_name, cfg["attack_params"]) is None:
                    print(f"      [SKIP] Gibbs not supported for {attack_name}")
                    break

                # No-chain baseline
                recon_nc = run_no_chain(attack_fn, cfg, synth, train, qi, hidden_features)
                for f, s in score(train, recon_nc, hidden_features).items():
                    acc["no_chain"][f].append(s)

                # Gibbs (includes hard chain = pass 0 snapshot)
                snapshots, conv_stats = run_gibbs_at_checkpoints(
                    attack_name, cfg["attack_params"], synth, train, qi,
                    chain_order, GIBBS_PASS_CHECKPOINTS,
                )
                all_conv.append(conv_stats)

                for f, s in score(train, snapshots[0], hidden_features).items():
                    acc["hard"][f].append(s)
                for ck in GIBBS_PASS_CHECKPOINTS:
                    for f, s in score(train, snapshots[ck], hidden_features).items():
                        acc[f"gibbs_{ck}"][f].append(s)

                # Oracle
                recon_oracle = run_oracle_chain(attack_fn, cfg, synth, train, qi, chain_order)
                for f, s in score(train, recon_oracle, hidden_features).items():
                    acc["oracle"][f].append(s)

            # ── Average across samples ────────────────────────────────────
            avg = {c: {f: float(np.mean(v)) for f, v in feat_dict.items()}
                   for c, feat_dict in acc.items() if feat_dict}

            # ── Print per-feature table ───────────────────────────────────
            valid_conds = [c for c in cond_keys if c in avg and avg[c]]
            FW, CW = 20, 10
            print(f"\n  [{sdg_method}]  {attack_name}  —  RA (%) averaged over {len(SAMPLES)} samples")
            hdr = f"  {'Feature':<{FW}}"
            for c in valid_conds:
                lbl = c.replace("_", " ")
                hdr += f"  {lbl:>{CW}}"
            print(hdr)
            print("  " + "─" * (FW + (CW + 2) * len(valid_conds)))

            for feat in hidden_features:
                row = f"  {feat:<{FW}}"
                for c in valid_conds:
                    v = avg.get(c, {}).get(feat, float('nan'))
                    row += f"  {v:>{CW}.2f}"
                print(row)

            # Mean row
            row = f"  {'MEAN':<{FW}}"
            for c in valid_conds:
                v = mean_score(avg[c]) if c in avg else float('nan')
                row += f"  {v:>{CW}.2f}"
            print("  " + "─" * (FW + (CW + 2) * len(valid_conds)))
            print(row)

            # ── Print convergence table ───────────────────────────────────
            if all_conv:
                # Average frac_changed per pass across samples
                max_passes_seen = max(len(cv) for cv in all_conv)
                print(f"\n  Convergence (avg frac predictions changed per Gibbs pass):")
                print(f"  {'Pass':<6}  {'frac_changed':>14}")
                for p in range(max_passes_seen):
                    fracs = [cv[p]["frac_changed"] for cv in all_conv if p < len(cv)]
                    avg_frac = float(np.mean(fracs)) if fracs else float('nan')
                    print(f"  {p+1:<6}  {avg_frac:>14.4%}")

            # ── Save CSV ──────────────────────────────────────────────────
            out_path = Path(OUT_DIR) / DATASET["name"]
            out_path.mkdir(parents=True, exist_ok=True)

            # Per-feature scores
            rows = []
            for feat in hidden_features:
                row = {"feature": feat}
                for c in valid_conds:
                    row[c] = round(avg.get(c, {}).get(feat, float('nan')), 4)
                rows.append(row)
            # Mean row
            mean_row = {"feature": "MEAN"}
            for c in valid_conds:
                mean_row[c] = round(mean_score(avg[c]) if c in avg else float('nan'), 4)
            rows.append(mean_row)
            p_feat = out_path / f"{attack_name}_{sdg_method}.csv"
            pd.DataFrame(rows).to_csv(p_feat, index=False)
            print(f"\n  Saved: {p_feat}")

            # Convergence stats
            if all_conv:
                conv_rows = []
                max_p = max(len(cv) for cv in all_conv)
                for p in range(max_p):
                    fracs = [cv[p]["frac_changed"] for cv in all_conv if p < len(cv)]
                    conv_rows.append({"pass": p + 1,
                                      "frac_changed_mean": round(float(np.mean(fracs)), 6),
                                      "frac_changed_std":  round(float(np.std(fracs)),  6)})
                p_conv = out_path / f"{attack_name}_{sdg_method}_convergence.csv"
                pd.DataFrame(conv_rows).to_csv(p_conv, index=False)
                print(f"  Saved: {p_conv}")

    print(f"\nDone. Results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
