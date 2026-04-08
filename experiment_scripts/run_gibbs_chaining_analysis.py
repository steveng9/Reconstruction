#!/usr/bin/env python3
"""
experiment_scripts/run_gibbs_chaining_analysis.py

Tests Gibbs-style iterative refinement chaining against the no-chain baseline,
comparing four Gibbs classifier configurations: RF, wide-RF, MLP, and an
RF+MLP+KNN ensemble.

Approach
--------
  Pass 0 (init):  Predict each hidden feature independently from QI only
                  (no cross-conditioning). Gives a clean, unbiased starting point.

  Pre-training:   For each hidden feature h_i, train one classifier (or ensemble)
                  on synth using QI + all *other* hidden features (true values).
                  Done once — training inputs never change across passes.

  Gibbs passes:   For each feature h_i, re-predict using QI + current best-guess
                  predictions for all other hidden features. Update in place.
                  Repeat until convergence or max passes.

Gibbs classifier configs compared
----------------------------------
  RF             : RandomForest (default params)
  RF_wide        : RandomForest (100 trees, unlimited depth)
  MLP            : MLPClassifier (default params)
  Ensemble       : soft vote of RF + MLP + KNN(k=5)

  NOTE: Ensemble and MLP are significantly slower due to neural-net training
  for each of the j hidden features per sample. Adjust SAMPLES or SDG_METHODS
  to control runtime.

Conditions in output
---------------------
  no_chain_{attack}   : standard attack, no chaining
  oracle_{attack}     : hard chain using TRUE hidden values — upper bound
  {clf}_init          : Gibbs pass 0 (QI-only independent predictions)
  {clf}_gibbs_{K}     : Gibbs result after K passes

Output
------
  Console: per-feature RA tables + convergence tables per (SDG, clf_config)
  CSV:     experiment_scripts/chaining_analysis/gibbs/{dataset}/{sdg}_{clf}.csv
           experiment_scripts/chaining_analysis/gibbs/{dataset}/{sdg}_{clf}_convergence.csv
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

SDG_METHODS = [
    "MST_eps10",
    "TabDDPM",
    "Synthpop",
]

# Attacks used only for no_chain and oracle baselines.
ATTACKS = ["RandomForest", "MLP"]

# Gibbs classifier configurations. Each is tested independently as the
# classifier for both Pass 0 (init) and all Gibbs passes.
GIBBS_CLF_CONFIGS = [
    {
        "name": "RF",
        "members": [
            {"attack": "RandomForest", "params": {}},
        ],
    },
    {
        "name": "RF_wide",
        "members": [
            {"attack": "RandomForest", "params": {"num_estimators": 100, "max_depth": 10}},
        ],
    },
    {
        "name": "MLP",
        "members": [
            {"attack": "MLP", "params": {}},
        ],
    },
    {
        "name": "Ensemble",  # RF + MLP + KNN(k=5) soft vote
        "members": [
            {"attack": "RandomForest", "params": {}},
            {"attack": "MLP",          "params": {}},
            {"attack": "KNN",          "params": {}},
        ],
    },
]

SAMPLES = [0, 1, 2, 3, 4]

# Score reconstruction at these Gibbs pass counts. 0 = init (QI-only), also
# recorded. Must be <= MAX_PASSES.
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
)


# ── Classifier helpers ────────────────────────────────────────────────────────

def _fit_clf(clf, attack_name, X_train, y_train):
    """Fit a classifier. MLP uses partial_fit loop for epoch tracking."""
    if attack_name == "MLP":
        classes  = np.unique(y_train)
        n_epochs = clf.max_iter
        for ep in range(n_epochs):
            clf.partial_fit(X_train, y_train, classes=classes)
    else:
        clf.fit(X_train, y_train)
    return clf


def _predict_labels(fitted_clfs, X_test):
    """
    Predict class labels from a list of (clf, attack_name) tuples.
    Single clf: argmax(proba). Multiple clfs: soft vote with class-space alignment.
    """
    if len(fitted_clfs) == 1:
        clf, _ = fitted_clfs[0]
        return clf.classes_[np.argmax(clf.predict_proba(X_test), axis=1)]

    # Soft vote across ensemble members
    all_classes  = sorted(set(c for clf, _ in fitted_clfs for c in clf.classes_))
    n_classes    = len(all_classes)
    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    summed       = np.zeros((len(X_test), n_classes))
    for clf, _ in fitted_clfs:
        proba = clf.predict_proba(X_test)
        for local_i, cls in enumerate(clf.classes_):
            summed[:, class_to_idx[cls]] += proba[:, local_i]
    return np.array(all_classes)[np.argmax(summed, axis=1)]


def _train_clf_list(clf_config, X_train, y_train):
    """Train all members of a clf_config. Returns list of (clf, attack_name)."""
    result = []
    for member in clf_config["members"]:
        clf = _make_soft_chain_clf(member["attack"], member["params"])
        clf = _fit_clf(clf, member["attack"], X_train, y_train)
        result.append((clf, member["attack"]))
    return result


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sample(sample_idx, sdg_method):
    sdir  = Path(DATASET["root"]) / f"sample_{sample_idx:02d}"
    train = pd.read_csv(sdir / "train.csv")
    synth = pd.read_csv(sdir / sdg_method / "synth.csv")
    ds    = DATASET["name"]
    qi_v  = DATASET["qi_variant"]
    return train, synth, QIs[ds][qi_v], minus_QIs[ds][qi_v]


def make_cfg(attack_name, sample_idx):
    params     = dict(ATTACK_PARAM_DEFAULTS.get(attack_name, {}))
    sample_dir = str(Path(DATASET["root"]) / f"sample_{sample_idx:02d}")
    return {
        "dataset":       {"name": DATASET["name"], "type": DATASET["data_type"],
                          "dir": sample_dir},
        "data_type":     DATASET["data_type"],
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


def run_gibbs_at_checkpoints(clf_config, synth, train, qi, chain_order, checkpoints):
    """
    Run Gibbs chaining with a given clf_config and return snapshots at each
    checkpoint pass (including pass 0 = QI-only init).

    Returns
    -------
    snapshots  : dict[int, pd.DataFrame]  checkpoint_pass → reconstructed df
    conv_stats : list[dict]               per-pass convergence info
    """
    from sklearn.preprocessing import OrdinalEncoder

    clf_name = clf_config["name"]

    # ── Encode QI ──────────────────────────────────────────────────────────
    qi_enc     = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_synth_qi = qi_enc.fit_transform(synth[qi].astype(str))
    X_test_qi  = qi_enc.transform(train[qi].astype(str))

    # ── Per-feature ordinal encoder for hidden feature conditioning ────────
    feat_encoders = {}
    for feat in chain_order:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        enc.fit(synth[[feat]].astype(str))
        feat_encoders[feat] = enc

    def _enc_block(df, feats):
        if not feats:
            return None
        return np.hstack([feat_encoders[f].transform(df[[f]].astype(str)) for f in feats])

    # ── Pass 0: independent QI-only predictions ────────────────────────────
    # Each hidden feature predicted from QI alone — no cross-conditioning.
    reconstructed = train.copy()
    print(f"      [{clf_name}] Pass 0: QI-only init ...")
    for feat in chain_order:
        y   = synth[feat].astype(str)
        clfs = _train_clf_list(clf_config, X_synth_qi, y)
        preds = _predict_labels(clfs, X_test_qi)
        try:
            reconstructed[feat] = preds.astype(synth[feat].dtype)
        except (ValueError, TypeError):
            reconstructed[feat] = preds

    snapshots = {0: reconstructed.copy()}
    print(f"      [{clf_name}] Pass 0 complete")

    # ── Pre-train Gibbs classifiers (once; reused across all passes) ───────
    # Each classifier: QI + all OTHER hidden features (true synth values).
    print(f"      [{clf_name}] Pre-training {len(chain_order)} Gibbs classifiers ...")
    gibbs_clfs = {}
    for feat in chain_order:
        other = [f for f in chain_order if f != feat]
        X_s   = np.hstack([X_synth_qi] + ([_enc_block(synth, other)] if other else []))
        y     = synth[feat].astype(str)
        gibbs_clfs[feat] = _train_clf_list(clf_config, X_s, y)
    print(f"      [{clf_name}] Pre-training complete")

    # ── Gibbs passes ────────────────────────────────────────────────────────
    conv_stats = []
    n_targets  = len(train)
    max_ckpt   = max(checkpoints)

    for pass_idx in range(max_ckpt):
        n_changed = {}
        for feat in chain_order:
            prev  = reconstructed[feat].copy()
            other = [f for f in chain_order if f != feat]
            X_t   = np.hstack([X_test_qi] + ([_enc_block(reconstructed, other)] if other else []))
            preds = _predict_labels(gibbs_clfs[feat], X_t)
            try:
                reconstructed[feat] = preds.astype(synth[feat].dtype)
            except (ValueError, TypeError):
                reconstructed[feat] = preds
            n_changed[feat] = int((reconstructed[feat].astype(str) != prev.astype(str)).sum())

        total = n_targets * len(chain_order)
        frac  = sum(n_changed.values()) / total
        stat  = {"pass": pass_idx + 1, "frac_changed": round(frac, 6),
                 **{f: round(n_changed[f] / n_targets, 4) for f in chain_order}}
        conv_stats.append(stat)
        print(f"      [{clf_name}] Gibbs pass {pass_idx+1}/{max_ckpt}: {frac:.3%} changed")

        ck = pass_idx + 1
        if ck in checkpoints:
            snapshots[ck] = reconstructed.copy()

        if frac < CONVERGENCE_TOL:
            print(f"      [{clf_name}] Converged at pass {pass_idx+1}")
            for remaining_ck in checkpoints:
                if remaining_ck not in snapshots:
                    snapshots[remaining_ck] = reconstructed.copy()
            break

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

    # Compute chain order once from sample_00 / first SDG (deterministic across runs)
    _, ref_synth, qi, hidden_features = load_sample(0, SDG_METHODS[0])
    chain_order = _dynamic_order(ref_synth, qi, hidden_features, data_type="categorical")
    print(f"Chain order (dynamic_MI from sample_00/{SDG_METHODS[0]}):")
    print(f"  {chain_order}\n")

    # Build all condition keys
    baseline_conds = [f"no_chain_{a}" for a in ATTACKS] + [f"oracle_{a}" for a in ATTACKS]
    gibbs_conds    = []
    for cfg in GIBBS_CLF_CONFIGS:
        gibbs_conds.append(f"{cfg['name']}_init")
        for k in GIBBS_PASS_CHECKPOINTS:
            gibbs_conds.append(f"{cfg['name']}_gibbs_{k}")
    all_conds = baseline_conds + gibbs_conds

    for sdg_method in SDG_METHODS:
        print(f"\n{'='*70}")
        print(f"SDG: {sdg_method}")
        print(f"{'='*70}")

        # Accumulators: condition → {feature: [scores across samples]}
        acc      = {c: defaultdict(list) for c in all_conds}
        all_conv = {cfg["name"]: [] for cfg in GIBBS_CLF_CONFIGS}

        for sample_idx in SAMPLES:
            print(f"\n  --- Sample {sample_idx:02d} ---")
            try:
                train, synth, qi, hidden_features = load_sample(sample_idx, sdg_method)
            except FileNotFoundError as e:
                print(f"    [SKIP] {e}")
                continue

            # ── Baselines: no_chain and oracle for each attack ─────────────
            for attack_name in ATTACKS:
                cfg       = make_cfg(attack_name, sample_idx)
                attack_fn = get_attack(attack_name, DATASET["data_type"])

                recon_nc = run_no_chain(attack_fn, cfg, synth, train, qi, hidden_features)
                for f, s in score(train, recon_nc, hidden_features).items():
                    acc[f"no_chain_{attack_name}"][f].append(s)

                recon_oracle = run_oracle_chain(attack_fn, cfg, synth, train, qi, chain_order)
                for f, s in score(train, recon_oracle, hidden_features).items():
                    acc[f"oracle_{attack_name}"][f].append(s)

                print(f"    no_chain_{attack_name}: mean={mean_score(score(train, recon_nc, hidden_features)):.2f}  "
                      f"oracle_{attack_name}: mean={mean_score(score(train, recon_oracle, hidden_features)):.2f}")

            # ── Gibbs experiments ──────────────────────────────────────────
            for clf_config in GIBBS_CLF_CONFIGS:
                clf_name = clf_config["name"]
                print(f"\n    [{clf_name}]")

                if _make_soft_chain_clf(clf_config["members"][0]["attack"],
                                        clf_config["members"][0]["params"]) is None:
                    print(f"      [SKIP] clf not supported")
                    continue

                snapshots, conv_stats = run_gibbs_at_checkpoints(
                    clf_config, synth, train, qi, chain_order, GIBBS_PASS_CHECKPOINTS,
                )
                all_conv[clf_name].append(conv_stats)

                for f, s in score(train, snapshots[0], hidden_features).items():
                    acc[f"{clf_name}_init"][f].append(s)
                for k in GIBBS_PASS_CHECKPOINTS:
                    for f, s in score(train, snapshots[k], hidden_features).items():
                        acc[f"{clf_name}_gibbs_{k}"][f].append(s)

                init_mean   = mean_score(score(train, snapshots[0], hidden_features))
                final_k     = GIBBS_PASS_CHECKPOINTS[-1]
                final_mean  = mean_score(score(train, snapshots[final_k], hidden_features))
                print(f"      init={init_mean:.2f}  gibbs_{final_k}={final_mean:.2f}")

        # ── Average across samples ─────────────────────────────────────────
        avg = {}
        for cond in all_conds:
            feat_dict = acc[cond]
            if feat_dict:
                avg[cond] = {f: float(np.mean(v)) for f, v in feat_dict.items()}

        valid_conds = [c for c in all_conds if c in avg and avg[c]]

        # ── Print per-feature table ────────────────────────────────────────
        FW, CW = 20, 10
        print(f"\n  [{sdg_method}]  RA (%) averaged over {len(SAMPLES)} samples")
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

        mean_row = f"  {'MEAN':<{FW}}"
        for c in valid_conds:
            v = mean_score(avg[c]) if c in avg else float('nan')
            mean_row += f"  {v:>{CW}.2f}"
        print("  " + "─" * (FW + (CW + 2) * len(valid_conds)))
        print(mean_row)

        # ── Print convergence tables ───────────────────────────────────────
        for clf_config in GIBBS_CLF_CONFIGS:
            clf_name  = clf_config["name"]
            conv_list = all_conv[clf_name]
            if not conv_list:
                continue
            max_p = max(len(cv) for cv in conv_list)
            print(f"\n  Convergence [{clf_name}] (avg frac changed per pass):")
            print(f"  {'Pass':<6}  {'frac_changed':>14}")
            for p in range(max_p):
                fracs    = [cv[p]["frac_changed"] for cv in conv_list if p < len(cv)]
                avg_frac = float(np.mean(fracs)) if fracs else float('nan')
                print(f"  {p+1:<6}  {avg_frac:>14.4%}")

        # ── Save CSVs ─────────────────────────────────────────────────────
        out_path = Path(OUT_DIR) / DATASET["name"]
        out_path.mkdir(parents=True, exist_ok=True)

        for clf_config in GIBBS_CLF_CONFIGS:
            clf_name = clf_config["name"]
            clf_conds = [f"{clf_name}_init"] + [f"{clf_name}_gibbs_{k}" for k in GIBBS_PASS_CHECKPOINTS]
            table_conds = baseline_conds + [c for c in clf_conds if c in avg]

            rows = []
            for feat in hidden_features:
                row = {"feature": feat}
                for c in table_conds:
                    row[c] = round(avg.get(c, {}).get(feat, float('nan')), 4)
                rows.append(row)
            mean_r = {"feature": "MEAN"}
            for c in table_conds:
                mean_r[c] = round(mean_score(avg[c]) if c in avg else float('nan'), 4)
            rows.append(mean_r)

            p_feat = out_path / f"{sdg_method}_{clf_name}.csv"
            pd.DataFrame(rows).to_csv(p_feat, index=False)
            print(f"\n  Saved: {p_feat}")

            # Convergence CSV
            conv_list = all_conv[clf_name]
            if conv_list:
                max_p     = max(len(cv) for cv in conv_list)
                conv_rows = []
                for p in range(max_p):
                    fracs = [cv[p]["frac_changed"] for cv in conv_list if p < len(cv)]
                    conv_rows.append({
                        "pass":               p + 1,
                        "frac_changed_mean":  round(float(np.mean(fracs)), 6),
                        "frac_changed_std":   round(float(np.std(fracs)),  6),
                    })
                p_conv = out_path / f"{sdg_method}_{clf_name}_convergence.csv"
                pd.DataFrame(conv_rows).to_csv(p_conv, index=False)
                print(f"  Saved: {p_conv}")

    print(f"\nDone. Results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
