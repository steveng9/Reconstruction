#!/usr/bin/env python
"""
TabDDPMWithMLP (and TabDDPMEnsemble) reconstruction attacks on adult 10k data.

Workflow:
  1. First run — train diffusion model + MLP stacker, then reconstruct:
         python experiment_scripts/run_tabddpm_mlp_adult.py --retrain

  2. Re-run inference without retraining (e.g. tweak stacking params):
         python experiment_scripts/run_tabddpm_mlp_adult.py

  3. Only retrain the MLP stacker (keep diffusion checkpoint):
         python experiment_scripts/run_tabddpm_mlp_adult.py --retrain-mlp

Artifacts are sample-level (not SDG-level). Run ONE SDG at a time with
--retrain to avoid overwriting another method's saved model.
"""

import sys
import argparse
import traceback
import numpy as np

_parser = argparse.ArgumentParser()
_parser.add_argument("--retrain",     action="store_true", default=False,
                     help="Retrain diffusion model and MLP stacker from scratch.")
_parser.add_argument("--retrain-mlp", action="store_true", default=False,
                     help="Retrain only the MLP stacker (reuse diffusion checkpoint).")
_parser.add_argument("--sdg", default=None,
                     help="Override SDG method name (must match dir name, e.g. MST_eps1).")
_args, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining

sys.path.insert(0, "/home/golobs/Reconstruction")
sys.path.append("/home/golobs/MIA_on_diffusion/")
sys.path.append("/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM")
sys.path.append("/home/golobs/recon-synth")
sys.path.append("/home/golobs/recon-synth/attacks")
sys.path.append("/home/golobs/recon-synth/attacks/solvers")

from get_data import load_data
from master_experiment_script import _run_attack, _score_reconstruction, _prepare_config


# ── Data ──────────────────────────────────────────────────────────────────────

SAMPLE_SIZE = 10_000
DATA_ROOT   = f"/home/golobs/data/reconstruction_data/adult/size_{SAMPLE_SIZE}"
SAMPLE_DIR  = f"{DATA_ROOT}/sample_00"

# ── SDG methods to sweep (run one at a time with --retrain) ───────────────────

SDG_METHODS = [
    #("CellSuppression",  {}),
    #("MST",    {"epsilon": 1.0}),
    #("MST",    {"epsilon": 10.0}),
    #("TVAE",   {}),
    ("RankSwap",  {}),
    # ("ARF",    {}),
    # ("TabDDPM", {}),
    # ("Synthpop", {}),
]

# ── Attack params ─────────────────────────────────────────────────────────────

SHARED_DIFFUSION = {
    "hidden_dims":   [512, 1024, 1024, 1024, 1024, 512],
    "dropout":       0.1,
    "batch_size":    4096,
    "lr":            0.0006,
    "num_epochs":    100_000,
    "num_timesteps": 1000,
    "resamples":     10,
    "jump_fn":       "jump_max10",
}

ATTACKS = [
    ("TabDDPMWithMLP", {
        **SHARED_DIFFUSION,
        "retrain":       _args.retrain,      # controls diffusion retraining
        "retrain_mlp":   _args.retrain_mlp,  # controls MLP-only retraining
        "stacking_frac": 0.5,
        "mlp_hidden_dims": [512, 512],
        "mlp_epochs":    500,
        "mlp_lr":        0.001,
    }),
    #("TabDDPMEnsemble", {
    #    **SHARED_DIFFUSION,
    #    "retrain":           _args.retrain,
    #    "n_diffusion_samples": 5,
    #}),
    #("TabDDPM", {
    #    **SHARED_DIFFUSION,
    #    "retrain": _args.retrain,
    #}),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def sdg_label(method, params):
    eps = (params or {}).get("epsilon")
    return f"{method}_eps{eps:g}" if eps is not None else method


def make_config(sdg_method, sdg_params, attack_method, attack_params):
    return {
        "dataset": {
            "name": "adult",
            "dir":  SAMPLE_DIR,
            "size": SAMPLE_SIZE,
            "type": "categorical",
        },
        "QI":            "QI1",
        "data_type":     "agnostic",
        "attack_method": attack_method,
        "sdg_method":    sdg_method,
        "sdg_params":    sdg_params or None,
        "memorization_test": {"enabled": False},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            attack_method: attack_params,
        },
    }


def run_one(sdg_method, sdg_params, attack_method, attack_params):
    label = sdg_label(sdg_method, sdg_params)
    print(f"\n{'─'*55}")
    print(f"  {attack_method}  ×  {label}")
    print(f"{'─'*55}")

    cfg      = make_config(sdg_method, sdg_params, attack_method, attack_params)
    prepared = _prepare_config(cfg)
    train, synth, qi, hidden_features, _ = load_data(prepared)

    recon  = _run_attack(prepared, synth, train, qi, hidden_features)
    scores = _score_reconstruction(train, recon, hidden_features, "categorical")
    mean   = round(float(np.mean(scores)), 3)

    for feat, s in zip(hidden_features, scores):
        print(f"    {feat:<20} {s:.3f}")
    print(f"    {'mean':<20} {mean:.3f}")
    return mean


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Override SDG list from --sdg flag
    if _args.sdg:
        name, *rest = _args.sdg.split("_eps")
        params = {"epsilon": float(rest[0])} if rest else {}
        sdg_list = [(name, params)]
    else:
        sdg_list = SDG_METHODS

    print(f"\nTabDDPMWithMLP sweep — adult size_{SAMPLE_SIZE} / sample_00")
    print(f"  retrain={_args.retrain}  retrain_mlp={_args.retrain_mlp}")
    print(f"  SDG methods : {[sdg_label(m, p) for m, p in sdg_list]}")
    print(f"  Attacks     : {[a for a, _ in ATTACKS]}")
    if not (_args.retrain or _args.retrain_mlp):
        print("\n  NOTE: pass --retrain to train diffusion + MLP stacker from scratch.")
        print("        pass --retrain-mlp to retrain only the MLP stacker.")

    summary = []
    for sdg_method, sdg_params in sdg_list:
        label = sdg_label(sdg_method, sdg_params)
        row = {"sdg": label}
        for attack_method, attack_params in ATTACKS:
            try:
                mean = run_one(sdg_method, sdg_params, attack_method, attack_params)
                row[attack_method] = mean
            except Exception as e:
                print(f"  ERROR ({label} / {attack_method}): {e}")
                traceback.print_exc()
                row[attack_method] = None
        summary.append(row)

    # Print compact summary table
    attack_names = [a for a, _ in ATTACKS]
    col_w = 18
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY — RA mean (rarity-weighted accuracy, higher = better)")
    print(f"{'='*60}")
    header = f"  {'SDG':<16}" + "".join(f"{a:>{col_w}}" for a in attack_names)
    print(header)
    print("  " + "-" * (14 + col_w * len(attack_names)))
    for row in summary:
        line = f"  {row['sdg']:<16}"
        for a in attack_names:
            v = row.get(a)
            line += f"{f'{v:.3f}' if v is not None else 'ERROR':>{col_w}}"
        print(line)
    print(f"{'='*60}\n")
