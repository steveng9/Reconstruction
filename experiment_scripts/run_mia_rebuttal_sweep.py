#!/usr/bin/env python
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT, MIA_ON_DIFFUSION, RECON_SYNTH, REPO_ROOT

"""
MIA-strength rebuttal sweep: SynthDistance, NNDR, RA-as-MIA across the 4
mechanistically-distinct DP generators (PrivBayes, PrivSyn, MWEMPGM,
PrivateGSD) at epsilon in {1, 1000}, on 3 datasets (adult 10k, cdc_diabetes
1k, nist_arizona 10k).

Addresses the POPETS reviewer critique: "the membership inference attacks
[...] have nearly the same performance [...] for both small and large
privacy budgets. That makes me wonder if the conclusion that DP protection
plateaus after privacy budget equals 1 is due to the fact that the
membership inference attacks are not strong enough." This extends the
existing Table `mia_comparison` (previously only MST eps=1/1000) to the
same 4 generators used in the epsilon-plateau sweep, to see whether the
plateau still shows up in MIA space with a wider set of generators, or
whether it's an artifact of testing only against MST.

Mirrors compare_mia_ra.py / run_mia_comparison_cdc.py / run_mia_comparison_arizona.py
(same attacks, same RF-based RA-as-MIA reduction) but generalizes across
dataset configs and generator/epsilon combos, and skips gracefully (with a
clear log line) if a synth.csv doesn't exist yet rather than crashing the
whole sweep.

Run detached:
    nohup conda run -n recon_ python experiment_scripts/run_mia_rebuttal_sweep.py \
          > outfiles/mia_rebuttal_sweep.log 2>&1 &
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(REPO_ROOT))
sys.path.append(str(MIA_ON_DIFFUSION))
sys.path.append(str(MIA_ON_DIFFUSION / 'midst_models' / 'single_table_TabDDPM'))
sys.path.append(str(RECON_SYNTH))
sys.path.append(str(RECON_SYNTH / 'attacks'))
sys.path.append(str(RECON_SYNTH / 'attacks' / 'solvers'))

import numpy as np
import pandas as pd
import wandb

from get_data import load_mia_data, QIs, minus_QIs
from attacks import get_attack, ra_as_mia
from attacks.mia import synth_distance_mia, nndr_mia
from master_experiment_script import _prepare_config

DATA_ROOT = str(DATA_ROOT)
RA_METHOD = "RandomForest"
RA_PARAMS = {"max_depth": 15, "num_estimators": 25}
EPSILONS = [1.0, 1000.0]
# MST is included here (it was not in the July run) because Table
# `mia_comparison`'s MST columns previously came from the older
# compare_mia_ra.py / run_mia_comparison_{cdc,arizona}.py scripts, which ran on
# a different train/holdout sample pairing AND on pre-repair synth. Routing MST
# through this sweep puts every DP column of that table on one code path, one
# sample pairing, and post-encoding-repair synth.
GENERATORS = ["MST", "PrivBayes", "PrivSyn", "MWEMPGM", "PrivateGSD"]

# Non-DP reference generators: not affected by the float-binning bug, but
# recomputed here anyway so the whole table is a single consistent measurement
# rather than half July-on-sample_01/02 and half August-on-sample_00/01. They
# take no epsilon, so they run exactly once each.
NONDP_GENERATORS = ["CellSuppression", "RankSwap", "TabDDPM", "Synthpop"]
N_TARGETS = None  # use full train + full holdout
SEED = 42

DATASET_CONFIGS = [
    {
        "dataset": "adult",
        "data_root": f"{DATA_ROOT}adult/size_10000",
        "sample_dir": f"{DATA_ROOT}adult/size_10000/sample_00",
        "holdout_dir": f"{DATA_ROOT}adult/size_10000/sample_01",
        "sample_size": 10_000,
        "qi": "QI1",
        "generators": GENERATORS,
        "ra_hidden_override": ["workclass", "occupation", "relationship", "income"],
    },
    {
        "dataset": "cdc_diabetes",
        "data_root": f"{DATA_ROOT}cdc_diabetes/size_1000",
        "sample_dir": f"{DATA_ROOT}cdc_diabetes/size_1000/sample_01",
        "holdout_dir": f"{DATA_ROOT}cdc_diabetes/size_1000/sample_02",
        "sample_size": 1_000,
        "qi": "QI1",
        "generators": GENERATORS,  # PrivateGSD synth now exists for cdc too
        "ra_hidden_override": [
            "Diabetes_binary", "Stroke", "HeartDiseaseorAttack", "CholCheck",
            "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
            "NoDocbcCost", "DiffWalk",
        ],
    },
    {
        "dataset": "nist_arizona_25feat",
        "data_root": f"{DATA_ROOT}nist_arizona_data/size_10000_25feat",
        "sample_dir": f"{DATA_ROOT}nist_arizona_data/size_10000_25feat/sample_01",
        "holdout_dir": f"{DATA_ROOT}nist_arizona_data/size_10000_25feat/sample_02",
        "sample_size": 10_000,
        "qi": "QI_medium",
        "generators": GENERATORS,
        "ra_hidden_override": [
            "CITIZEN", "EDUC", "FARM", "GQ", "MARST",
            "MIGRATE5", "NATIVITY", "OWNERSHP", "URBAN",
        ],
    },
]

DATA_TYPE = "categorical"


def sdg_dirname(method, eps):
    # eps is None for the non-DP reference generators, whose release directory
    # is just the bare method name with no _eps suffix.
    if eps is None:
        return method
    return f"{method}_eps{eps:g}"


def make_config(ds_cfg, sdg_method, eps):
    return {
        "dataset": {
            "name": ds_cfg["dataset"],
            "dir": ds_cfg["sample_dir"],
            "size": ds_cfg["sample_size"],
            "type": DATA_TYPE,
        },
        "QI": ds_cfg["qi"],
        "data_type": DATA_TYPE,
        "attack_method": RA_METHOD,
        "sdg_method": sdg_method,
        "sdg_params": ({} if eps is None else {"epsilon": eps}),
        "memorization_test": {
            "enabled": True,
            "holdout_dir": ds_cfg["holdout_dir"],
        },
        "mia_params": {"n_targets": N_TARGETS, "seed": SEED},
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining": {"enabled": False},
            RA_METHOD: RA_PARAMS,
        },
    }


def run_one(ds_cfg, sdg_method, eps):
    label = sdg_dirname(sdg_method, eps)
    synth_path = Path(ds_cfg["sample_dir"]) / label / "synth.csv"
    if not synth_path.exists():
        print(f"  [SKIP] {ds_cfg['dataset']} / {label}: synth.csv not found at {synth_path}"
              f" (not generated yet)", flush=True)
        return None

    print(f"\n{'='*60}\n  {ds_cfg['dataset']} / {label}\n{'='*60}", flush=True)

    cfg = make_config(ds_cfg, sdg_method, eps)
    prepared_ra = _prepare_config(cfg)
    train, synth, holdout, meta = load_mia_data(cfg)

    qi = QIs[ds_cfg["dataset"]][ds_cfg["qi"]]
    ra_hidden = ds_cfg["ra_hidden_override"]

    row = {"dataset": ds_cfg["dataset"], "sdg_method": sdg_method, "epsilon": eps, "qi": ds_cfg["qi"]}

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

    print(f"  [RA-as-MIA / QI={ds_cfg['qi']}]", flush=True)
    attack_fn = get_attack(RA_METHOD, DATA_TYPE)
    ra_metrics, ra_scores, _, _ = ra_as_mia(
        attack_fn, prepared_ra, synth, train, holdout, qi, ra_hidden,
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
        name="mia_rebuttal_sweep",
        config={
            "generators": GENERATORS + NONDP_GENERATORS,
            "epsilons": EPSILONS,
            "ra_method": RA_METHOD,
            "n_targets": N_TARGETS,
            "seed": SEED,
        },
        tags=["mia_vs_ra", "rebuttal", "epsilon_sweep"],
        group="mia-rebuttal-sweep-2026-08-rerun",
    )

    rows = []
    for ds_cfg in DATASET_CONFIGS:
        combos = [(g, e) for g in ds_cfg["generators"] for e in EPSILONS]
        combos += [(g, None) for g in NONDP_GENERATORS]
        for gen, eps in combos:
            try:
                row = run_one(ds_cfg, gen, eps)
                if row is not None:
                    rows.append(row)
                    # sdg_dirname (not an f-string on eps) because eps is None
                    # for the non-DP generators and would blow up on {eps:g}.
                    prefix = f"{ds_cfg['dataset']}/{sdg_dirname(gen, eps)}"
                    wandb.log({f"{prefix}/{k}": v for k, v in row.items()
                               if k not in ("dataset", "sdg_method", "epsilon", "qi")})
            except Exception as e:
                print(f"  ERROR for {ds_cfg['dataset']} / {gen} eps={eps}: {e}", flush=True)
                import traceback
                traceback.print_exc()

    wandb.finish()

    df = pd.DataFrame(rows)
    out_csv = "experiment_scripts/mia_rebuttal_sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {len(df)} rows to {out_csv}")

    print(f"\n{'='*70}\n  SUMMARY (AUC)\n{'='*70}")
    for dataset in df["dataset"].unique():
        sub = df[df["dataset"] == dataset]
        print(f"\n--- {dataset} ---")
        for attack_col in ["SynthDistance_auc", "NNDR_auc", "RA_as_MIA_auc"]:
            piv = sub.pivot_table(index="sdg_method", columns="epsilon", values=attack_col)
            print(f"\n  {attack_col}:")
            print(piv.round(3).to_string())

    print("\nAll comparisons complete.")


if __name__ == "__main__":
    main()
