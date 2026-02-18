#!/usr/bin/env python
"""
Comparison sweep: LinearReconstruction (SOTA) vs MLP, RandomForest, Random baseline.

Runs on a single binary hidden feature using all other features as QI:
  - adult   size_10k / sample_00  →  hidden: income
  - cdc_diabetes size_1k / sample_00  →  hidden: Diabetes_binary

All SDG methods, one sample each, with memorization test.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/run_linear_comparison.py adult
    python experiment_scripts/run_linear_comparison.py cdc

Reuses _run_attack, _score_reconstruction, _prepare_config from master_experiment_script.
"""

import sys
import argparse
import numpy as np

# Parse our dataset arg BEFORE importing master_experiment_script, which also
# runs argparse on import and would choke on unrecognised positional args.
_parser = argparse.ArgumentParser()
_parser.add_argument("dataset", choices=["adult", "cdc"],
                     help="Which dataset to run: 'adult' (10k) or 'cdc' (1k)")
_args, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining   # strip our arg before master import

# Set server paths before importing master_experiment_script (which runs argparse on import)
sys.path.insert(0, "/home/golobs/Reconstruction")
sys.path.append("/home/golobs/MIA_on_diffusion/")
sys.path.append("/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM")
sys.path.append("/home/golobs/recon-synth")
sys.path.append("/home/golobs/recon-synth/attacks")
sys.path.append("/home/golobs/recon-synth/attacks/solvers")

import pandas as pd
import wandb
from get_data import load_data
# Reuse core helpers from master script instead of reimplementing them
from master_experiment_script import _run_attack, _score_reconstruction, _prepare_config


def sdg_dirname(method, params=None):
    params = params or {}
    eps = params.get("epsilon") or params.get("eps")
    return f"{method}_eps{eps:g}" if eps is not None else method


# ── Adult QI discretization ───────────────────────────────────────────────────
# The linear attack generates k-way queries over Cartesian products of feature
# values.  With high-cardinality raw adult features (age: 73 vals, fnlwgt:
# thousands, capital-gain: hundreds, native-country: 41) the query matrix
# explodes.  Discretizing to ≤16 categories per feature keeps query count
# ~5k and the A matrix (10k × 5k) comfortably in memory.
#
# Discretization is fit on the training data (fnlwgt quantile edges) and
# applied identically to train, synth, and holdout so all three attacks
# operate on the same feature space.

_COUNTRY_US     = {'United-States'}
_COUNTRY_MEX_CA = {'Mexico','Cuba','Jamaica','Puerto-Rico','El-Salvador',
                   'Dominican-Republic','Guatemala','Columbia','Haiti',
                   'Nicaragua','Peru','Ecuador','Trinadad&Tobago','Honduras'}
_COUNTRY_EUROPE = {'England','Germany','Italy','Poland','Portugal','France',
                   'Yugoslavia','Greece','Ireland','Hungary','Scotland',
                   'Holand-Netherlands'}
_COUNTRY_ASIA   = {'China','India','Japan','Vietnam','Taiwan','Iran',
                   'Philippines','Cambodia','Thailand','Laos','Hong'}

def _group_country(c):
    c = str(c).strip()
    if c in _COUNTRY_US:     return 'US'
    if c in _COUNTRY_MEX_CA: return 'Mexico/CA'
    if c in _COUNTRY_EUROPE: return 'Europe'
    if c in _COUNTRY_ASIA:   return 'Asia'
    return 'Other'


def fit_adult_discretizer(train_df):
    """Fit discretization on training data; return a function that applies it."""
    # fnlwgt: fit quintile edges on train, apply fixed edges everywhere
    q = train_df['fnlwgt'].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
    q[0], q[-1] = -np.inf, np.inf

    def apply(df):
        df = df.copy()
        # age → 6 bins
        df['age'] = pd.cut(
            df['age'].astype(float),
            bins=[-np.inf, 25, 35, 45, 55, 65, np.inf],
            labels=['<=25','26-35','36-45','46-55','56-65','65+']
        ).astype(str)
        # fnlwgt → 5 quantile bins (edges from train)
        df['fnlwgt'] = pd.cut(
            df['fnlwgt'].astype(float), bins=q,
            labels=['Q1','Q2','Q3','Q4','Q5'], include_lowest=True
        ).astype(str)
        # capital-gain → 4 bins (91% zeros)
        df['capital-gain'] = pd.cut(
            df['capital-gain'].astype(float),
            bins=[-np.inf, 0, 3000, 10000, np.inf],
            labels=['0','low','med','high'], include_lowest=True
        ).astype(str)
        # capital-loss → 3 bins (95% zeros)
        df['capital-loss'] = pd.cut(
            df['capital-loss'].astype(float),
            bins=[-np.inf, 0, 2000, np.inf],
            labels=['0','low','high'], include_lowest=True
        ).astype(str)
        # hours-per-week → 4 bins
        df['hours-per-week'] = pd.cut(
            df['hours-per-week'].astype(float),
            bins=[-np.inf, 35, 40, 50, np.inf],
            labels=['part-time','full-time','overtime','heavy'], include_lowest=True
        ).astype(str)
        # native-country → 5 regions
        df['native-country'] = df['native-country'].apply(_group_country)
        return df

    return apply


# ── Dataset configs ───────────────────────────────────────────────────────────

DATA_ROOT = "/home/golobs/data/reconstruction_data"

DATASET_CONFIGS = [
    {
        "label":       "adult_10k",
        "dataset": {
            "name": "adult",
            "dir":  f"{DATA_ROOT}/adult/size_10000/sample_00",
            "size": 10000,
            "type": "categorical",
        },
        "holdout_dir": f"{DATA_ROOT}/adult/size_10000/sample_01",
        "QI":          "QI_linear",   # all features except income
        "hidden":      "income",
        "discretize":  True,          # high-cardinality features → binned
        "sdg_methods": [
            ("TabDDPM",       {}),
            ("TVAE",          {}),
            ("CTGAN",         {}),
            ("ARF",           {}),
            ("MST",           {"epsilon": 1.0}),
            ("MST",           {"epsilon": 10.0}),
            ("MST",           {"epsilon": 100.0}),
            ("MST",           {"epsilon": 1000.0}),
            ("AIM",           {"epsilon": 1.0}),
            ("AIM",           {"epsilon": 10.0}),
            ("Synthpop",      {}),
            ("RankSwap",      {}),
            ("CellSuppression", {}),
        ],
    },
    {
        "label":       "cdc_1k",
        "dataset": {
            "name": "cdc_diabetes",
            "dir":  f"{DATA_ROOT}/cdc_diabetes/size_1000/sample_00",
            "size": 1000,
            "type": "categorical",
        },
        "holdout_dir": f"{DATA_ROOT}/cdc_diabetes/size_1000/sample_01",
        "QI":          "QI_linear",   # all features except Diabetes_binary
        "hidden":      "Diabetes_binary",
        "discretize":  False,         # already binary / low-cardinality
        "sdg_methods": [
            ("TabDDPM",       {}),
            ("TVAE",          {}),
            ("CTGAN",         {}),
            ("ARF",           {}),
            ("MST",           {"epsilon": 1.0}),
            ("MST",           {"epsilon": 10.0}),
            ("MST",           {"epsilon": 100.0}),
            ("MST",           {"epsilon": 1000.0}),
            ("AIM",           {"epsilon": 1.0}),
            ("AIM",           {"epsilon": 10.0}),
            ("Synthpop",      {}),
            ("RankSwap",      {}),
            ("CellSuppression", {}),
        ],
    },
]

ATTACK_METHODS = ["LinearReconstruction", "MLP", "RandomForest", "Random"]

ATTACK_PARAMS = {
    "LinearReconstruction": {"k": 3, "n_procs": 4},
    "MLP": {
        "hidden_dims":   [128, 96, 64],
        "epochs":        250,
        "learning_rate": 0.0003,
        "batch_size":    264,
        "dropout_rate":  0.2,
        "patience":      50,
        "test_size":     0.2,
    },
    "RandomForest": {"max_depth": 25, "num_estimators": 25},
    "Random":       {},
}


# ── Config builder ────────────────────────────────────────────────────────────

def make_config(ds_cfg, sdg_method, sdg_params, attack_method) -> dict:
    return {
        "dataset":    ds_cfg["dataset"],
        "QI":         ds_cfg["QI"],
        "data_type":  ds_cfg["dataset"]["type"],
        "attack_method": attack_method,
        "sdg_method": sdg_method,
        "sdg_params": sdg_params or None,
        "memorization_test": {
            "enabled":     True,
            "holdout_dir": ds_cfg["holdout_dir"],
        },
        "attack_params": {
            "ensembling": {"enabled": False},
            "chaining":   {"enabled": False},
            **{k: v for k, v in ATTACK_PARAMS.items()},
        },
    }


# ── Experiment runner — uses master script helpers ────────────────────────────

def run_experiment(ds_cfg, sdg_method, sdg_params, attack_method):
    sdg_label = sdg_dirname(sdg_method, sdg_params)
    run_name  = f"{ds_cfg['label']}/{attack_method}/{sdg_label}"

    print(f"\n{'='*60}")
    print(f"  {run_name}")
    print(f"{'='*60}")

    cfg      = make_config(ds_cfg, sdg_method, sdg_params, attack_method)
    prepared = _prepare_config(cfg)

    wandb.init(
        project="tabular-reconstruction-attacks",
        name=run_name,
        config=cfg,
        tags=[ds_cfg["label"], "linear_comparison"],
        group=f"linear_comparison_{ds_cfg['label']}",
        reinit=True,
    )

    try:
        train, synth, qi, hidden_features, holdout = load_data(prepared)
        dataset_type = prepared["dataset"]["type"]

        if ds_cfg.get("discretize"):
            # Fit on train, apply identically to synth and holdout so all
            # three datasets share the same discretised feature space.
            discretize = fit_adult_discretizer(train)
            train   = discretize(train)
            synth   = discretize(synth)
            holdout = discretize(holdout)

        print("  Reconstructing training targets...")
        recon_train   = _run_attack(prepared, synth, train,   qi, hidden_features)
        train_scores  = _score_reconstruction(train,   recon_train,   hidden_features, dataset_type)

        print("  Reconstructing holdout (non-training) targets...")
        recon_holdout  = _run_attack(prepared, synth, holdout, qi, hidden_features)
        holdout_scores = _score_reconstruction(holdout, recon_holdout, hidden_features, dataset_type)

        # Single hidden feature → one score each
        train_score   = round(float(train_scores[0]),   2)
        holdout_score = round(float(holdout_scores[0]), 2)
        delta         = round(train_score - holdout_score, 2)

        print(f"\n  --- Results ({ds_cfg['hidden']}) ---")
        print(f"  train    = {train_score:.3f}")
        print(f"  nontrain = {holdout_score:.3f}")
        print(f"  delta    = {delta:+.3f}")

        results = {
            f"RA_train_{hidden_features[0]}":       train_score,
            f"RA_nontraining_{hidden_features[0]}": holdout_score,
            f"RA_delta_{hidden_features[0]}":       delta,
        }
        wandb.log(results)

        return {
            "sdg":      sdg_label,
            "attack":   attack_method,
            "train":    train_score,
            "nontrain": holdout_score,
            "delta":    delta,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        raise

    finally:
        wandb.finish()


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(ds_cfg, rows):
    if not rows:
        return

    attacks  = ATTACK_METHODS
    sdg_list = [sdg_dirname(m, p) for m, p in ds_cfg["sdg_methods"]]

    # Index results: (sdg, attack) → row
    idx = {(r["sdg"], r["attack"]): r for r in rows}

    w_sdg = max(len(s) for s in sdg_list)
    w_sdg = max(w_sdg, len("SDG method"))
    cw    = 6   # width per score cell

    def atk_header(a):
        short = {"LinearReconstruction": "Linear", "RandomForest": "RF",
                 "MLP": "MLP", "Random": "Rnd"}
        return short.get(a, a)

    # Header: SDG | [train nontrain delta] per attack
    atk_cols = "  ".join(
        f"{atk_header(a):^{cw*3+4}}" for a in attacks
    )
    sub_cols = "  ".join(
        f"{'train':>{cw}} {'nontrain':>{cw}} {'delta':>{cw}}" for _ in attacks
    )
    sep = "  " + "-" * (w_sdg + 4 + (cw * 3 + 6) * len(attacks))

    print(f"\n\n{'='*80}")
    print(f"  COMPARISON: {ds_cfg['label']}  |  hidden: {ds_cfg['hidden']}")
    print(f"  Metric: rarity-weighted accuracy (higher = better attack)")
    print(f"  delta = train - nontrain  (positive = memorization signal)")
    print(f"{'='*80}")
    print(f"  {'SDG method':<{w_sdg}}    {atk_cols}")
    print(f"  {'':<{w_sdg}}    {sub_cols}")
    print(sep)

    for sdg in sdg_list:
        cells = []
        for atk in attacks:
            r = idx.get((sdg, atk))
            if r:
                cells.append(
                    f"{r['train']:>{cw}.2f} {r['nontrain']:>{cw}.2f} {r['delta']:>+{cw}.2f}"
                )
            else:
                cells.append(f"{'N/A':>{cw}} {'':>{cw}} {'':>{cw}}")
        print(f"  {sdg:<{w_sdg}}    {'  '.join(cells)}")

    print(sep)

    # Average row
    avg_cells = []
    for atk in attacks:
        subset = [r for r in rows if r["attack"] == atk]
        if subset:
            avg_cells.append(
                f"{np.mean([r['train'] for r in subset]):>{cw}.2f} "
                f"{np.mean([r['nontrain'] for r in subset]):>{cw}.2f} "
                f"{np.mean([r['delta'] for r in subset]):>+{cw}.2f}"
            )
        else:
            avg_cells.append(f"{'N/A':>{cw}} {'':>{cw}} {'':>{cw}}")
    print(f"  {'AVERAGE':<{w_sdg}}    {'  '.join(avg_cells)}")
    print(f"{'='*80}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    key_map = {"adult": "adult_10k", "cdc": "cdc_1k"}
    selected = [ds for ds in DATASET_CONFIGS if ds["label"] == key_map[_args.dataset]]

    for ds_cfg in selected:
        print(f"\n\n{'#'*70}")
        print(f"#  Dataset: {ds_cfg['label']}  |  hidden: {ds_cfg['hidden']}")
        print(f"{'#'*70}")

        total = len(ds_cfg["sdg_methods"]) * len(ATTACK_METHODS)
        done  = 0
        ds_results = []

        for sdg_method, sdg_params in ds_cfg["sdg_methods"]:
            for attack_method in ATTACK_METHODS:
                done += 1
                print(f"\n[{done}/{total}]", end="")
                try:
                    summary = run_experiment(ds_cfg, sdg_method, sdg_params, attack_method)
                    if summary:
                        ds_results.append(summary)
                except Exception as e:
                    print(f"  SKIPPED ({sdg_dirname(sdg_method, sdg_params)} / {attack_method}): {e}")

        print_summary_table(ds_cfg, ds_results)

    print("All runs complete.")
