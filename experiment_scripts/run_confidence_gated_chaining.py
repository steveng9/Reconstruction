#!/usr/bin/env python3
"""
experiment_scripts/run_confidence_gated_chaining.py

Last-ditch investigation: can confidence-gated chaining close the gap between
hard chaining and oracle chaining?

Background
----------
Prior results show:
  - Hard chaining barely helps (±1-2 pp vs. no-chain baseline)
  - Oracle chaining substantially helps (+10–25 pp for key features)
  - Soft chaining collapses to near-random for high-cardinality features
    (a probability vector over 8477 fnlwgt classes overwhelms the QI signal)

New approach: masked-hard chaining
  - High-confidence rows (max_prob >= τ): one-hot at predicted class
    → matches training-time conditioning; propagates a confident hard label
  - Low-confidence rows: uniform distribution (1/K per class)
    → signals "unknown", avoids injecting a likely-wrong label downstream

Compares against soft-gated chaining (existing approach, same threshold logic
but uses raw probability vector for confident rows instead of one-hot).

Variants tested
---------------
  baseline    No-chain (independent prediction per feature)
  hard        Hard chaining (always propagate argmax, even when uncertain)
  soft        Soft chaining (probability vector; existing, collapses for high-card)
  soft_gated  Soft-gated (probas for confident rows, uniform for uncertain rows)
  mh_gated    Masked-hard-gated (one-hot for confident rows, uniform for uncertain)
  oracle      Oracle chaining (upper bound: true values used as conditioning)

For soft_gated and mh_gated, a threshold sweep is run: τ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}.

Outputs
-------
  Console:  per-feature RA table + threshold sweep summary + confidence diagnostics
  CSV:      experiment_scripts/chaining_analysis/confidence_gated/results.csv
"""

# ── Path setup ────────────────────────────────────────────────────────────────
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

from unittest.mock import MagicMock
sys.modules['wandb'] = MagicMock()

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT  = '/home/golobs/data/reconstruction_data'
DATASET    = {
    "name":       "adult",
    "root":       f"{DATA_ROOT}/adult/size_1000",
    "data_type":  "categorical",
    "qi_variant": "QI1",
    "sdg_method": "TabDDPM",
}
ATTACK      = "RandomForest"
SAMPLES     = [0, 1, 2, 3, 4]
THRESHOLDS  = [0.5, 0.6, 0.7, 0.8, 0.9]   # confidence gating sweep
STRATEGY    = "mutual_info"                 # chain ordering strategy

OUT_DIR = os.path.join(RECON_ROOT, 'experiment_scripts', 'chaining_analysis',
                       'confidence_gated')

# ── Imports ───────────────────────────────────────────────────────────────────
import random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from attacks import get_attack
from scoring import calculate_reconstruction_score
from get_data import QIs, minus_QIs
from attack_defaults import ATTACK_PARAM_DEFAULTS
from enhancements.chaining_wrapper import (
    _order_by_mutual_info,
    _make_soft_chain_clf,
    _run_soft_chained_sklearn,
    _run_masked_hard_chained_sklearn,
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_sample(sample_idx):
    sdir  = Path(DATASET["root"]) / f"sample_{sample_idx:02d}"
    train = pd.read_csv(sdir / "train.csv")
    synth = pd.read_csv(sdir / DATASET["sdg_method"] / "synth.csv")
    qi    = QIs[DATASET["name"]][DATASET["qi_variant"]]
    hid   = minus_QIs[DATASET["name"]][DATASET["qi_variant"]]
    return train, synth, qi, hid


def make_cfg(sample_idx):
    params     = dict(ATTACK_PARAM_DEFAULTS.get(ATTACK, {}))
    sample_dir = str(Path(DATASET["root"]) / f"sample_{sample_idx:02d}")
    return {
        "dataset":       {"name": DATASET["name"], "type": DATASET["data_type"],
                          "dir": sample_dir},
        "data_type":     DATASET["data_type"],
        "attack_method": ATTACK,
        "attack_params": params,
    }


# ── Chaining runners ──────────────────────────────────────────────────────────

def run_no_chain(attack_fn, cfg, synth, train, qi, hidden):
    recon, _, _ = attack_fn(cfg, synth, train, qi, hidden)
    return recon


def run_hard_chain(attack_fn, cfg, synth, train, qi, order):
    recon  = train.copy()
    known  = list(qi)
    for feat in order:
        step, _, _ = attack_fn(cfg, synth, recon[known], known, [feat])
        recon[feat] = step[feat]
        known.append(feat)
    return recon


def run_oracle_chain(attack_fn, cfg, synth, train, qi, order):
    """Upper bound: true values used as conditioning at every step."""
    recon = train.copy()
    known = list(qi)
    for feat in order:
        step, _, _ = attack_fn(cfg, synth, train[known], known, [feat])
        recon[feat] = step[feat]
        known.append(feat)
    return recon


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_features(train, recon, hidden):
    """Rarity-weighted accuracy per feature, returned as dict."""
    scores = calculate_reconstruction_score(train, recon, hidden)
    return dict(zip(hidden, scores))


def modified_ra(ra_pct, n_classes):
    baseline = 1.0 / n_classes
    denom    = 1.0 - baseline
    if denom == 0:
        return float('nan')
    return (ra_pct / 100.0 - baseline) / denom


# ── Main experiment ───────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Fix chain order from sample_00 synth
    print(f"\nLoading sample_00 to fix chain order...")
    _, ref_synth, qi, hidden = load_sample(0)
    chain_order = _order_by_mutual_info(ref_synth, qi, hidden, data_type="categorical")
    print(f"  QI ({len(qi)}): {qi}")
    print(f"  Hidden ({len(hidden)}): {hidden}")
    print(f"  Chain order ({STRATEGY}): {chain_order}")

    attack_fn    = get_attack(ATTACK, "categorical")
    attack_parms = dict(ATTACK_PARAM_DEFAULTS.get(ATTACK, {}))

    # Accumulators: {condition_key: {feature: [scores across samples]}}
    VARIANT_KEYS = (
        ["baseline", "hard", "soft", "oracle"]
        + [f"soft_gated_{t}" for t in THRESHOLDS]
        + [f"mh_gated_{t}"   for t in THRESHOLDS]
    )
    acc = {k: defaultdict(list) for k in VARIANT_KEYS}

    # Per-threshold confidence stats: {threshold: {feature: [fracs across samples]}}
    conf_acc_sg = {t: defaultdict(list) for t in THRESHOLDS}   # soft-gated
    conf_acc_mh = {t: defaultdict(list) for t in THRESHOLDS}   # masked-hard-gated

    for sample_idx in SAMPLES:
        print(f"\n{'─'*60}")
        print(f"  Sample {sample_idx:02d}")
        print(f"{'─'*60}")
        train, synth, qi, hidden = load_sample(sample_idx)
        cfg = make_cfg(sample_idx)

        # ── Baseline ──────────────────────────────────────────────────────────
        print("  [baseline] no chaining...")
        recon_nc = run_no_chain(attack_fn, cfg, synth, train, qi, hidden)
        for f, s in score_features(train, recon_nc, hidden).items():
            acc["baseline"][f].append(s)

        # ── Hard chaining ─────────────────────────────────────────────────────
        print("  [hard] hard chaining...")
        recon_h = run_hard_chain(attack_fn, cfg, synth, train, qi, chain_order)
        for f, s in score_features(train, recon_h, hidden).items():
            acc["hard"][f].append(s)

        # ── Soft chaining ─────────────────────────────────────────────────────
        print("  [soft] soft chaining (no threshold)...")
        recon_s = _run_soft_chained_sklearn(
            ATTACK, attack_parms, synth, train, qi, chain_order,
            confidence_threshold=None)
        for f, s in score_features(train, recon_s, hidden).items():
            acc["soft"][f].append(s)

        # ── Oracle chaining ───────────────────────────────────────────────────
        print("  [oracle] oracle chaining (upper bound)...")
        recon_o = run_oracle_chain(attack_fn, cfg, synth, train, qi, chain_order)
        for f, s in score_features(train, recon_o, hidden).items():
            acc["oracle"][f].append(s)

        # ── Soft-gated threshold sweep ─────────────────────────────────────
        for tau in THRESHOLDS:
            print(f"  [soft_gated τ={tau}] soft-gated chaining...")
            recon_sg = _run_soft_chained_sklearn(
                ATTACK, attack_parms, synth, train, qi, chain_order,
                confidence_threshold=tau)
            for f, s in score_features(train, recon_sg, hidden).items():
                acc[f"soft_gated_{tau}"][f].append(s)
            # confidence stats not available from soft-gated (we'd need to
            # instrument it; approximate from masked-hard stats at same τ)

        # ── Masked-hard-gated threshold sweep ─────────────────────────────────
        for tau in THRESHOLDS:
            print(f"  [mh_gated τ={tau}] masked-hard-gated chaining...")
            recon_mh, c_stats = _run_masked_hard_chained_sklearn(
                ATTACK, attack_parms, synth, train, qi, chain_order,
                confidence_threshold=tau)
            for f, s in score_features(train, recon_mh, hidden).items():
                acc[f"mh_gated_{tau}"][f].append(s)
            for f, frac in c_stats.items():
                conf_acc_mh[tau][f].append(frac)

    # ── Average across samples ─────────────────────────────────────────────────
    avg = {k: {f: float(np.mean(vs)) for f, vs in fd.items()}
           for k, fd in acc.items()}
    avg_conf_mh = {
        t: {f: float(np.mean(vs)) for f, vs in fd.items()}
        for t, fd in conf_acc_mh.items()
    }

    num_classes = {f: train[f].nunique() for f in hidden}   # from last sample

    # ═══════════════════════════════════════════════════════════════════════════
    #  PRINT RESULTS
    # ═══════════════════════════════════════════════════════════════════════════

    print(f"\n\n{'='*100}")
    print(f"ADULT  |  {ATTACK}  |  chain order: {STRATEGY}  |  "
          f"{len(SAMPLES)} samples  |  SDG: {DATASET['sdg_method']}")
    print(f"{'='*100}")

    # ── Table 1: Per-feature results for all variants at τ=0.7 ────────────────
    TAU_SHOW = 0.7
    print(f"\n── Table 1: Per-feature RA (%) — representative threshold τ={TAU_SHOW} ──")
    cols = [
        ("No Chain",         "baseline"),
        ("Hard",             "hard"),
        ("Soft",             "soft"),
        (f"Soft-Gated {TAU_SHOW}", f"soft_gated_{TAU_SHOW}"),
        (f"MH-Gated  {TAU_SHOW}", f"mh_gated_{TAU_SHOW}"),
        ("Oracle",           "oracle"),
    ]
    FW, CW = 18, 11
    hdr = f"{'Feature':<{FW}}"
    for lbl, _ in cols:
        hdr += f"  {lbl:>{CW}}"
    hdr += f"  {'mod Baseline':>{CW}}  {'mod Oracle':>{CW}}  {'mod MH-Gated':>{CW}}"
    print(hdr)
    print("─" * len(hdr))

    for feat in chain_order:
        nc = num_classes[feat]
        vals = [avg[k].get(feat, float('nan')) for _, k in cols]
        row  = f"{feat:<{FW}}"
        for v in vals:
            row += f"  {v:>{CW}.1f}"
        mod_base   = modified_ra(avg["baseline"].get(feat, float('nan')), nc)
        mod_oracle = modified_ra(avg["oracle"].get(feat, float('nan')), nc)
        mod_mh     = modified_ra(avg[f"mh_gated_{TAU_SHOW}"].get(feat, float('nan')), nc)
        row += f"  {mod_base:>{CW}.3f}  {mod_oracle:>{CW}.3f}  {mod_mh:>{CW}.3f}"
        print(row)

    # Mean row
    feats = chain_order
    print("─" * len(hdr))
    mean_vals = [np.mean([avg[k].get(f, float('nan')) for f in feats]) for _, k in cols]
    row = f"{'MEAN':<{FW}}"
    for v in mean_vals:
        row += f"  {v:>{CW}.1f}"
    print(row)

    # ── Table 2: Threshold sweep — mean RA across all features ─────────────────
    print(f"\n── Table 2: Threshold sweep — mean RA (%) across all hidden features ──")
    TW = 12
    hdr2 = f"  {'Threshold':>{TW}}  {'Soft-Gated':>{TW}}  {'MH-Gated':>{TW}}"
    print(hdr2)
    print("─" * len(hdr2))
    base_mean   = np.mean([avg["baseline"].get(f, np.nan) for f in feats])
    hard_mean   = np.mean([avg["hard"].get(f, np.nan) for f in feats])
    soft_mean   = np.mean([avg["soft"].get(f, np.nan) for f in feats])
    oracle_mean = np.mean([avg["oracle"].get(f, np.nan) for f in feats])
    print(f"  {'baseline':>{TW}}  {base_mean:>{TW}.2f}  {'—':>{TW}}")
    print(f"  {'hard':>{TW}}  {hard_mean:>{TW}.2f}  {'—':>{TW}}")
    print(f"  {'soft':>{TW}}  {soft_mean:>{TW}.2f}  {'—':>{TW}}")
    for tau in THRESHOLDS:
        sg_mean = np.mean([avg[f"soft_gated_{tau}"].get(f, np.nan) for f in feats])
        mh_mean = np.mean([avg[f"mh_gated_{tau}"].get(f, np.nan) for f in feats])
        print(f"  {str(tau):>{TW}}  {sg_mean:>{TW}.2f}  {mh_mean:>{TW}.2f}")
    print(f"  {'oracle':>{TW}}  {oracle_mean:>{TW}.2f}  {'—':>{TW}}")

    # ── Table 3: Chain-position divergence at τ=0.7 ────────────────────────────
    print(f"\n── Table 3: Chain-position divergence (τ={TAU_SHOW}) ──")
    hdr3 = (f"  {'Pos':<4}  {'Feature':<{FW}}  {'No Chain':>{CW}}  {'Hard':>{CW}}  "
            f"{'MH-Gated':>{CW}}  {'Oracle':>{CW}}  "
            f"{'Δ Hard':>{CW}}  {'Δ MH-Gated':>{CW}}  {'Δ Oracle':>{CW}}")
    print(hdr3)
    print("─" * len(hdr3))
    for pos, feat in enumerate(chain_order):
        base  = avg["baseline"].get(feat, np.nan)
        hard  = avg["hard"].get(feat, np.nan)
        mh    = avg[f"mh_gated_{TAU_SHOW}"].get(feat, np.nan)
        orc   = avg["oracle"].get(feat, np.nan)
        d_h   = hard - base
        d_mh  = mh   - base
        d_o   = orc  - base
        print(f"  {pos+1:<4}  {feat:<{FW}}  {base:>{CW}.1f}  {hard:>{CW}.1f}  "
              f"{mh:>{CW}.1f}  {orc:>{CW}.1f}  "
              f"{d_h:>+{CW}.1f}  {d_mh:>+{CW}.1f}  {d_o:>+{CW}.1f}")

    # ── Table 4: Confidence diagnostics (masked-hard, τ sweep) ──────────────
    print(f"\n── Table 4: % of rows above confidence threshold (masked-hard, per feature) ──")
    TW2 = 8
    hdr4 = f"  {'Feature':<{FW}}" + "".join(f"  {str(t):>{TW2}}" for t in THRESHOLDS)
    print(hdr4)
    print("─" * len(hdr4))
    for feat in chain_order:
        nc  = num_classes[feat]
        row4 = f"  {feat:<{FW}}"
        for t in THRESHOLDS:
            pct = avg_conf_mh[t].get(feat, np.nan) * 100
            row4 += f"  {pct:>{TW2}.1f}%"
        row4 += f"  (n_classes={nc})"
        print(row4)

    # ── Summary: oracle gap analysis ───────────────────────────────────────────
    print(f"\n── Summary: does any variant close the oracle gap? ──")
    print(f"  Oracle gap = oracle_mean - baseline_mean = "
          f"{oracle_mean - base_mean:+.2f} pp")
    print(f"  Hard gain  = hard_mean - baseline_mean   = "
          f"{hard_mean - base_mean:+.2f} pp")
    for tau in THRESHOLDS:
        mh_mean = np.mean([avg[f"mh_gated_{tau}"].get(f, np.nan) for f in feats])
        gap_closed = (mh_mean - base_mean) / (oracle_mean - base_mean + 1e-9) * 100
        print(f"  MH-gated τ={tau}: mean={mh_mean:.2f}  Δ={mh_mean-base_mean:+.2f}  "
              f"oracle-gap closed={gap_closed:.1f}%")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    rows = []
    for feat in chain_order:
        nc  = num_classes[feat]
        row = {
            "feature":    feat,
            "n_classes":  nc,
            "baseline":   round(avg["baseline"].get(feat, np.nan), 3),
            "hard":       round(avg["hard"].get(feat, np.nan), 3),
            "soft":       round(avg["soft"].get(feat, np.nan), 3),
            "oracle":     round(avg["oracle"].get(feat, np.nan), 3),
            "mod_baseline": round(modified_ra(avg["baseline"].get(feat, np.nan), nc), 4),
            "mod_oracle":   round(modified_ra(avg["oracle"].get(feat, np.nan), nc), 4),
        }
        for tau in THRESHOLDS:
            sg_score = avg[f"soft_gated_{tau}"].get(feat, np.nan)
            mh_score = avg[f"mh_gated_{tau}"].get(feat, np.nan)
            conf_pct = avg_conf_mh[tau].get(feat, np.nan) * 100
            row[f"soft_gated_{tau}"]    = round(sg_score, 3)
            row[f"mh_gated_{tau}"]      = round(mh_score, 3)
            row[f"conf_pct_mh_{tau}"]   = round(conf_pct, 1)
            row[f"mod_mh_gated_{tau}"]  = round(modified_ra(mh_score, nc), 4)
        rows.append(row)

    csv_path = Path(OUT_DIR) / "results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    print(f"\nDone. View all results in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
