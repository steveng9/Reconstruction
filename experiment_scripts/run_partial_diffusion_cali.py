#!/usr/bin/env python
"""
Sweep: partialDiffusion (TabDDPM + RePaint) reconstruction attacks on california data.

Both attacks train a diffusion model on the synthetic data, then use it to
reconstruct hidden features. The key workflow for tuning RePaint is:

  1. First run (train diffusion models and reconstruct):
         python experiment_scripts/run_partial_diffusion_cali.py --retrain

  2. Subsequent runs (tune reconstruction params without retraining):
         python experiment_scripts/run_partial_diffusion_cali.py

Artifacts are saved under:
  {SAMPLE_DIR}/partial_tabddpm_artifacts/    (TabDDPM)
  {SAMPLE_DIR}/repaint_artifacts/            (RePaint)

NOTE: artifacts are keyed by sample dir, not SDG method. If you run multiple
SDG methods with --retrain, each overwrites the previous model for that sample.
Run one SDG at a time (comment others out in SDG_METHODS) to avoid this.

Results logged to WandB under project "tabular-reconstruction-attacks",
group "cali_partialDiffusion".
"""

import sys
import argparse
import numpy as np

# Parse our args BEFORE importing master_experiment_script, which runs its own
# argparse on import and would choke on unrecognised args.
_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--retrain", action="store_true", default=False,
    help=(
        "Retrain diffusion models from scratch and save artifacts. "
        "Omit to skip training and reuse previously saved artifacts."
    ),
)
_args, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining   # strip our arg before master import

sys.path.insert(0, "/home/golobs/Reconstruction")
sys.path.append("/home/golobs/MIA_on_diffusion/")
sys.path.append("/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM")
sys.path.append("/home/golobs/recon-synth")
sys.path.append("/home/golobs/recon-synth/attacks")
sys.path.append("/home/golobs/recon-synth/attacks/solvers")

import wandb
from get_data import load_data
from master_experiment_script import _run_attack, _score_reconstruction, _prepare_config


def sdg_dirname(method, params=None):
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    return f"{method}_eps{eps:g}" if eps is not None else method


# ── Data paths ────────────────────────────────────────────────────────────────

SAMPLE_SIZE      = 1_000
DATA_ROOT        = f"/home/golobs/data/reconstruction_data/california/size_{SAMPLE_SIZE}"
SAMPLE_DIR       = f"{DATA_ROOT}/sample_00"
HOLDOUT_DIR      = f"{DATA_ROOT}/sample_01"   # disjoint sample for memorization test
MEMORIZATION_TEST = True   # set True to also reconstruct holdout targets


# ── SDG methods ───────────────────────────────────────────────────────────────
# Only one SDG method uncommented at a time is recommended — artifacts are
# written to the same sample-level directory regardless of SDG, so running
# multiple methods with --retrain would overwrite each other's saved model.

SDG_METHODS = [
    ("TabDDPM",         {}),
    # ("TVAE",            {}),
    # ("CTGAN",           {}),
    # ("ARF",             {}),
    # ("MST",             {"epsilon": 1.0}),
    # ("MST",             {"epsilon": 10.0}),
    # ("MST",             {"epsilon": 100.0}),
    # ("Synthpop",        {}),
    # ("RankSwap",        {}),
    # ("CellSuppression", {}),
]


# ── Attack hyperparameters ────────────────────────────────────────────────────
# Tune these freely without --retrain; only model architecture / training params
# (num_epochs, hidden_dims, dropout, lr, batch_size, num_timesteps) require
# --retrain to take effect.

TABDDPM_PARAMS = {
    "retrain":       _args.retrain,
    "num_epochs":    50_000,
    "resamples":     50,
    "num_timesteps": 2000,
    "hidden_dims":   [512, 1024, 1024, 1024, 512],
    #"hidden_dims":   [1024, 4096, 4096, 4096, 1024],
    "dropout":       0.0,
    #"jump_fn":       "jump_max10",
    "batch_size":    2048,
    "lr":            0.0006,
}

REPAINT_PARAMS = {
    "retrain":       _args.retrain,
    "num_epochs":    200_000,
    "resamples":     50,
    #"jump":          10,
    "num_timesteps": 1000,
    "hidden_dims":   [512, 1024, 1024, 1024, 512],
    "batch_size":    2048,
    "dropout":       0.0,
    "jump_fn":       "jump_max50",
    "lr":            0.0006,
}

CONDITIONED_REPAINT_PARAMS = {
    "retrain":       _args.retrain,
    "num_epochs":    50_000,
    "resamples":     50,
    "num_timesteps": 1000,
    "hidden_dims":   [512, 1024, 1024, 1024, 512],
    "batch_size":    2048,
    "dropout":       0.0,
    "jump_fn":       "jump_max50",
    "lr":            0.0006,
}

ATTACKS = [
    ("TabDDPM",          TABDDPM_PARAMS),
    #("RePaint",            REPAINT_PARAMS),
    ("ConditionedRePaint", CONDITIONED_REPAINT_PARAMS),
]


# ── Config builder ────────────────────────────────────────────────────────────

def make_config(sdg_method, sdg_params, attack_method, attack_specific_params) -> dict:
    return {
        "dataset": {
            "name": "california",
            "dir":  SAMPLE_DIR,
            "size": SAMPLE_SIZE,
            "type": "continuous",
        },
        "QI":            "QI1",
        "data_type":     "agnostic",
        "attack_method": attack_method,
        "sdg_method":    sdg_method,
        "sdg_params":    sdg_params or None,
        "memorization_test": {
            "enabled":     MEMORIZATION_TEST,
            "holdout_dir": HOLDOUT_DIR,
        },
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            attack_method: attack_specific_params,
        },
    }


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(sdg_method, sdg_params, attack_method, attack_specific_params):
    sdg_label = sdg_dirname(sdg_method, sdg_params)
    retrain   = attack_specific_params.get("retrain", False)
    run_name  = f"pDiff_{attack_method}_{'retrain' if retrain else 'reuse'}_{sdg_label}"

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"  retrain={retrain}")
    print(f"{'='*60}")

    cfg      = make_config(sdg_method, sdg_params, attack_method, attack_specific_params)
    prepared = _prepare_config(cfg)

    wandb.init(
        project="tabular-reconstruction-attacks",
        name=run_name,
        config=cfg,
        tags=["california", "size_10000", "partialDiffusion", attack_method],
        group="cali_partialDiffusion",
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)

        print("  Reconstructing training targets...")
        recon_train  = _run_attack(prepared, synth, train, qi, hidden_features)
        train_scores = _score_reconstruction(train, recon_train, hidden_features, "continuous")
        train_mean   = round(float(np.mean(train_scores)), 2)

        results = {f"RA_train_{f}": s for f, s in zip(hidden_features, train_scores)}
        results["RA_train_mean"] = train_mean

        if MEMORIZATION_TEST:
            print("  Reconstructing holdout (non-training) targets...")
            recon_holdout  = _run_attack(prepared, synth, holdout, qi, hidden_features)
            holdout_scores = _score_reconstruction(holdout, recon_holdout, hidden_features, "continuous")
            holdout_mean   = round(float(np.mean(holdout_scores)), 2)
            delta_mean     = round(train_mean - holdout_mean, 2)

            for feat, ts, hs in zip(hidden_features, train_scores, holdout_scores):
                results[f"RA_nontraining_{feat}"] = hs
                results[f"RA_delta_{feat}"]       = round(ts - hs, 2)
            results["RA_nontraining_mean"] = holdout_mean
            results["RA_delta_mean"]       = delta_mean
        else:
            holdout_mean = None
            delta_mean   = None

        print(f"\n  --- Results ---")
        print(f"  train_mean    = {train_mean:.3f}")
        if MEMORIZATION_TEST:
            print(f"  nontrain_mean = {holdout_mean:.3f}")
            print(f"  delta_mean    = {delta_mean:+.3f}")

        wandb.log(results)

        return {
            "sdg":      sdg_label,
            "attack":   attack_method,
            "retrain":  retrain,
            "train":    train_mean,
            "nontrain": holdout_mean,
            "delta":    delta_mean,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        raise

    finally:
        wandb.finish()


# ── Random baseline ───────────────────────────────────────────────────────────

def run_random_baseline(sdg_method, sdg_params):
    """Run the Random attack once and return the mean train score."""
    cfg = {
        "dataset": {
            "name": "california",
            "dir":  SAMPLE_DIR,
            "size": SAMPLE_SIZE,
            "type": "continuous",
        },
        "QI":            "QI1",
        "data_type":     "continuous",
        "attack_method": "Random",
        "sdg_method":    sdg_method,
        "sdg_params":    sdg_params or None,
        "memorization_test": {"enabled": False, "holdout_dir": HOLDOUT_DIR},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            "Random": {},
        },
    }
    prepared = _prepare_config(cfg)
    train, synth, qi, hidden_features, _ = load_data(prepared)
    recon   = _run_attack(prepared, synth, train, qi, hidden_features)
    scores  = _score_reconstruction(train, recon, hidden_features, "continuous")
    return round(float(np.mean(scores)), 2)


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(rows):
    if not rows:
        return

    w_sdg = max(max(len(r["sdg"])    for r in rows), len("SDG"))
    w_atk = max(max(len(r["attack"]) for r in rows), len("Attack"))
    cw    = 9

    def fmt_val(v):
        return f"{v:.3f}" if v is not None else "N/A"

    def fmt_row(sdg, attack, retrain, train, nontrain, delta, baseline):
        rt_str = "yes" if retrain else "no"
        return (
            f"  {sdg:<{w_sdg}}  {attack:<{w_atk}}  {rt_str:^7}  "
            f"{fmt_val(train):>{cw}}  {fmt_val(nontrain):>{cw}}  {fmt_val(delta):>{cw}}  "
            f"{fmt_val(baseline):>{cw}}"
        )

    header = (
        f"  {'SDG':<{w_sdg}}  {'Attack':<{w_atk}}  {'retrain':^7}  "
        f"{'train':>{cw}}  {'nontrain':>{cw}}  {'delta':>{cw}}  {'random':>{cw}}"
    )
    sep = "  " + "-" * (w_sdg + w_atk + 7 + cw * 4 + 14)

    print(f"\n\n{'='*70}")
    print(f"  SWEEP SUMMARY  —  partialDiffusion on california size_{SAMPLE_SIZE}/sample_00")
    print(f"  QI1: age, sex, race, native-country, education, marital-status")
    print(f"  Hidden: workclass, fnlwgt, education-num, occupation, relationship,")
    print(f"          capital-gain, capital-loss, hours-per-week, income")
    print(f"  Metric: rarity-weighted accuracy (higher = better attack)")
    print(f"{'='*70}")
    print(header)
    print(sep)

    for r in rows:
        print(fmt_row(r["sdg"], r["attack"], r["retrain"],
                      r["train"], r["nontrain"], r["delta"], r.get("baseline")))

    print(sep)

    for atk in sorted({r["attack"] for r in rows}):
        subset = [r for r in rows if r["attack"] == atk]
        if len(subset) > 1:
            avg_t = round(float(np.mean([r["train"] for r in subset])), 2)
            avg_h = round(float(np.mean([r["nontrain"] for r in subset if r["nontrain"] is not None])), 2) if MEMORIZATION_TEST else None
            avg_d = round(float(np.mean([r["delta"]    for r in subset if r["delta"]    is not None])), 2) if MEMORIZATION_TEST else None
            avg_b = round(float(np.mean([r["baseline"] for r in subset if r.get("baseline") is not None])), 2)
            print(fmt_row("AVG", atk, False, avg_t, avg_h, avg_d, avg_b))

    print(f"{'='*70}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retrain = _args.retrain
    print(f"\npartialDiffusion sweep on california  |  retrain={retrain}")
    print(f"  SDG methods : {[sdg_dirname(m, p) for m, p in SDG_METHODS]}")
    print(f"  Attacks     : {[a for a, _ in ATTACKS]}")
    if not retrain:
        print("\n  NOTE: retrain=False — reusing saved model artifacts.")
        print("        Run with --retrain to (re)train diffusion models first.")

    total = len(SDG_METHODS) * len(ATTACKS)
    done  = 0
    all_results = []

    for sdg_method, sdg_params in SDG_METHODS:
        sdg_label = sdg_dirname(sdg_method, sdg_params)

        print(f"\n[baseline] Running Random baseline for {sdg_label}...")
        try:
            baseline_score = run_random_baseline(sdg_method, sdg_params)
            print(f"  Random baseline mean = {baseline_score:.3f}")
        except Exception as e:
            print(f"  WARNING: baseline failed ({e}), will show N/A")
            baseline_score = None

        for attack_method, attack_params in ATTACKS:
            done += 1
            print(f"\n[{done}/{total}]", end="")
            try:
                summary = run_experiment(sdg_method, sdg_params, attack_method, attack_params)
                if summary:
                    summary["baseline"] = baseline_score
                    all_results.append(summary)
            except Exception as e:
                print(f"  SKIPPED ({sdg_label} / {attack_method}): {e}")

    print_summary_table(all_results)
    print(f"All {done} run(s) complete.")
