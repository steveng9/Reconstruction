#!/usr/bin/env python
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT, MIA_ON_DIFFUSION, RECON_SYNTH, REPO_ROOT

"""
MIA-strength rebuttal sweep, adult 1k variant.

Follow-up to run_mia_rebuttal_sweep.py (adult/cdc/nist_arizona at their
original sizes): the user's hypothesis is that membership signal should be
*more* detectable at smaller n, so this reruns the same 3 attacks
(SynthDistance, NNDR, RA-as-MIA) on adult at size_1000 instead of 10000,
across a wider generator set that includes non-DP baselines known to leak
(CellSuppression, RankSwap) alongside the DP generators and TabDDPM.

Generators:
  - DP, eps in {1, 1000}: PrivBayes, PrivSyn, MWEMPGM, MST, PrivateGSD
  - Non-DP, single variant (no epsilon): CellSuppression, RankSwap, Synthpop, TabDDPM

Run detached:
    nohup conda run -n recon_ python experiment_scripts/run_mia_rebuttal_sweep_adult1k.py \
          > outfiles/mia_rebuttal_sweep_adult1k.log 2>&1 &
"""

import sys
from pathlib import Path

sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(MIA_ON_DIFFUSION))
sys.path.append(str(MIA_ON_DIFFUSION / 'midst_models' / 'single_table_TabDDPM'))
sys.path.append(str(RECON_SYNTH))
sys.path.append(str(RECON_SYNTH / 'attacks'))
sys.path.append(str(RECON_SYNTH / 'attacks' / 'solvers'))

import pandas as pd
import wandb

from get_data import load_mia_data, QIs
from attacks import get_attack, ra_as_mia
from attacks.mia import synth_distance_mia, nndr_mia
from master_experiment_script import _prepare_config

DATA_ROOT = f"{DATA_ROOT}/adult/size_1000"
SAMPLE_DIR = f"{DATA_ROOT}/sample_00"
HOLDOUT_DIR = f"{DATA_ROOT}/sample_01"
DATASET = "adult"
SAMPLE_SIZE = 1_000
QI = "QI1"
DATA_TYPE = "categorical"
RA_METHOD = "RandomForest"
RA_PARAMS = {"max_depth": 15, "num_estimators": 25}
RA_HIDDEN_OVERRIDE = ["workclass", "occupation", "relationship", "income"]
N_TARGETS = None
SEED = 42

# (method, epsilon or None)
JOBS = [
    ("PrivBayes", 1.0), ("PrivBayes", 1000.0),
    ("PrivSyn", 1.0), ("PrivSyn", 1000.0),
    ("MWEMPGM", 1.0), ("MWEMPGM", 1000.0),
    ("MST", 1.0), ("MST", 1000.0),
    ("PrivateGSD", 1.0), ("PrivateGSD", 1000.0),
    ("CellSuppression", None),
    ("RankSwap", None),
    ("Synthpop", None),
    ("TabDDPM", None),
]


def sdg_dirname(method, eps):
    return f"{method}_eps{eps:g}" if eps is not None else method


def make_config(sdg_method, eps):
    return {
        "dataset": {"name": DATASET, "dir": SAMPLE_DIR, "size": SAMPLE_SIZE, "type": DATA_TYPE},
        "QI": QI,
        "data_type": DATA_TYPE,
        "attack_method": RA_METHOD,
        "sdg_method": sdg_method,
        "sdg_params": {"epsilon": eps} if eps is not None else None,
        "memorization_test": {"enabled": True, "holdout_dir": HOLDOUT_DIR},
        "mia_params": {"n_targets": N_TARGETS, "seed": SEED},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining": {"enabled": False},
            RA_METHOD: RA_PARAMS,
        },
    }


def run_one(sdg_method, eps):
    label = sdg_dirname(sdg_method, eps)
    synth_path = Path(SAMPLE_DIR) / label / "synth.csv"
    if not synth_path.exists():
        print(f"  [SKIP] {label}: synth.csv not found at {synth_path} (not generated yet)", flush=True)
        return None

    print(f"\n{'='*60}\n  adult (1k) / {label}\n{'='*60}", flush=True)

    cfg = make_config(sdg_method, eps)
    prepared_ra = _prepare_config(cfg)
    train, synth, holdout, meta = load_mia_data(cfg)
    qi = QIs[DATASET][QI]

    row = {"dataset": DATASET, "sdg_method": sdg_method, "epsilon": eps, "qi": QI}

    print("  [SynthDistance]", flush=True)
    sd_metrics, sd_scores, labels, all_targets = synth_distance_mia(
        cfg, synth, train, holdout, meta, return_raw=True
    )
    row["SynthDistance_auc"] = sd_metrics["MIA_auc"]
    row["SynthDistance_advantage"] = sd_metrics["MIA_advantage"]
    row["SynthDistance_tpr_at_fpr001"] = sd_metrics["MIA_tpr_at_fpr001"]

    print("  [NNDR]", flush=True)
    nn_metrics, nn_scores, _, _ = nndr_mia(cfg, synth, train, holdout, meta, return_raw=True)
    row["NNDR_auc"] = nn_metrics["MIA_auc"]
    row["NNDR_advantage"] = nn_metrics["MIA_advantage"]
    row["NNDR_tpr_at_fpr001"] = nn_metrics["MIA_tpr_at_fpr001"]

    print(f"  [RA-as-MIA / QI={QI}]", flush=True)
    attack_fn = get_attack(RA_METHOD, DATA_TYPE)
    ra_metrics, ra_scores, _, _ = ra_as_mia(
        attack_fn, prepared_ra, synth, train, holdout, qi, RA_HIDDEN_OVERRIDE,
        n_targets=N_TARGETS, seed=SEED,
    )
    row["RA_as_MIA_auc"] = ra_metrics["RA_as_MIA_auc"]
    row["RA_as_MIA_advantage"] = ra_metrics["RA_as_MIA_advantage"]
    row["RA_as_MIA_tpr_at_fpr001"] = ra_metrics["RA_as_MIA_tpr_at_fpr001"]

    print(f"  SynthDistance AUC={row['SynthDistance_auc']:.3f}  "
          f"NNDR AUC={row['NNDR_auc']:.3f}  RA-as-MIA AUC={row['RA_as_MIA_auc']:.3f}", flush=True)

    return row


def main():
    wandb.init(
        project="tabular-reconstruction-attacks",
        name="mia_rebuttal_sweep_adult1k",
        config={"jobs": [sdg_dirname(m, e) for m, e in JOBS], "ra_method": RA_METHOD,
                "n_targets": N_TARGETS, "seed": SEED},
        tags=["mia_vs_ra", "rebuttal", "adult_1k"],
        group="mia-rebuttal-sweep-adult1k-2026-07",
    )

    rows = []
    for sdg_method, eps in JOBS:
        try:
            row = run_one(sdg_method, eps)
            if row is not None:
                rows.append(row)
                label = sdg_dirname(sdg_method, eps)
                wandb.log({f"{label}/{k}": v for k, v in row.items()
                           if k not in ("dataset", "sdg_method", "epsilon", "qi")})
        except Exception as e:
            print(f"  ERROR for {sdg_dirname(sdg_method, eps)}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    wandb.finish()

    df = pd.DataFrame(rows)
    out_csv = "experiment_scripts/mia_rebuttal_sweep_adult1k_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")

    print(f"\n{'='*70}\n  SUMMARY (AUC) -- adult 1k\n{'='*70}")
    for attack_col in ["SynthDistance_auc", "NNDR_auc", "RA_as_MIA_auc"]:
        print(f"\n  {attack_col}:")
        print(df[["sdg_method", "epsilon", attack_col]].to_string(index=False))

    print("\nAll comparisons complete.")


if __name__ == "__main__":
    main()
