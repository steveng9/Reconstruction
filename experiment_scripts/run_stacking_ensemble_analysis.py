#!/usr/bin/env python3
"""
experiment_scripts/run_stacking_ensemble_analysis.py

Evaluates stacking (meta-learning) ensemble against individual attacks and
soft-voting ensembles, across multiple attack pairs and SDG methods.

Stacking trains a per-feature LogisticRegression meta-model on out-of-fold (OOF)
base-model predictions from synth, then applies it to aggregate predictions on
actual targets. Including QI features in the meta-model input lets it learn
which base model to trust for which sub-population.

Conditions evaluated
--------------------
  individual_{attack}  : baseline, no ensemble
  soft_{A}+{B}         : existing soft-voting pair
  stack_{A}+{B}        : stacking pair (meta-model)
  oracle_{A}+{B}       : per-record oracle for the pair (upper bound)

The oracle is computed as: if EITHER base model predicts correctly for a record,
credit that record. Shows theoretical headroom for the pair.

Attack pairs tested
-------------------
  Configure ATTACK_PAIRS below.

Parallelism
-----------
  Job unit: (sdg_method, sample_idx). Each worker handles all attacks + all pairs
  for that job, so individual attack results are computed once per job and reused
  across pairs. 3 SDG × 5 samples = 15 jobs.

  CLI:
    --workers N     number of parallel workers (default N_WORKERS)
    --serial        run in main process (Ctrl-C friendly, for debugging)
    --dry-run       print jobs without running
    --sdg NAME      restrict to one SDG method
    --sample N      restrict to one sample index

Output
------
  Console + CSV: experiment_scripts/stacking_analysis/{dataset}/results.csv
"""

# ── Path setup (main process) ─────────────────────────────────────────────────
import sys, os

RECON_ROOT = '/home/golobs/Reconstruction'

for _p in [
    '/home/golobs/MIA_on_diffusion/',
    '/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM',
    '/home/golobs/recon-synth',
    '/home/golobs/recon-synth/attacks',
    '/home/golobs/recon-synth/attacks/solvers',
]:
    if _p not in sys.path:
        sys.path.append(_p)

if RECON_ROOT in sys.path:
    sys.path.remove(RECON_ROOT)
sys.path.insert(0, RECON_ROOT)

from unittest.mock import MagicMock
sys.modules['wandb'] = MagicMock()

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_ROOT = '/home/golobs/data/reconstruction_data'

DATASET = {
    "name":       "adult",
    "root":       f"{DATA_ROOT}/adult/size_10000",
    "data_type":  "categorical",
    "qi_variant": "QI1",
}

SDG_METHODS = [
    "MST_eps10",
    "TabDDPM",
    "Synthpop",
]

ATTACK_PAIRS = [
    ("RandomForest", "MLP"),
    ("RandomForest", "LightGBM"),
    ("MLP",          "LightGBM"),
    ("KNN",          "RandomForest"),
]

ALL_INDIVIDUAL_ATTACKS = ["KNN", "RandomForest", "MLP", "LightGBM",
                           "NaiveBayes", "LogisticRegression"]

SAMPLES    = [0, 1, 2, 3, 4]
N_FOLDS    = 5
USE_PROBAS = True
N_WORKERS  = 4

OUT_DIR = os.path.join(RECON_ROOT, 'experiment_scripts', 'stacking_analysis')


# ── Worker initialiser (called once per spawned process) ──────────────────────

def _worker_setup():
    """Set up sys.path and mock wandb in each worker process."""
    import sys
    from unittest.mock import MagicMock

    sys.modules['wandb'] = MagicMock()

    # Append recon-synth paths without disturbing front of sys.path
    for _p in [
        '/home/golobs/MIA_on_diffusion/',
        '/home/golobs/MIA_on_diffusion/midst_models/single_table_TabDDPM',
        '/home/golobs/recon-synth',
        '/home/golobs/recon-synth/attacks',
        '/home/golobs/recon-synth/attacks/solvers',
    ]:
        if _p not in sys.path:
            sys.path.append(_p)

    # Force Reconstruction to index 0 so its attacks/ shadows recon-synth/attacks/
    if RECON_ROOT in sys.path:
        sys.path.remove(RECON_ROOT)
    sys.path.insert(0, RECON_ROOT)

    # master_experiment_script calls parse_args() at import time
    sys.argv = sys.argv[:1]


# ── Data / config helpers ─────────────────────────────────────────────────────

def _load_sample(sample_idx, sdg_method):
    import pandas as pd
    from pathlib import Path
    from get_data import QIs, minus_QIs
    sdir  = Path(DATASET["root"]) / f"sample_{sample_idx:02d}"
    train = pd.read_csv(sdir / "train.csv")
    synth = pd.read_csv(sdir / sdg_method / "synth.csv")
    ds, qi_v = DATASET["name"], DATASET["qi_variant"]
    return train, synth, QIs[ds][qi_v], minus_QIs[ds][qi_v]


def _make_cfg(attack_name, sample_idx, sdg_method):
    from attack_defaults import ATTACK_PARAM_DEFAULTS
    from pathlib import Path
    params = dict(ATTACK_PARAM_DEFAULTS.get(attack_name, {}))
    sample_dir = str(Path(DATASET["root"]) / f"sample_{sample_idx:02d}")
    return {
        "dataset":       {"name": DATASET["name"], "type": DATASET["data_type"],
                          "dir": sample_dir},
        "data_type":     DATASET["data_type"],
        "sdg_method":    sdg_method,
        "attack_method": attack_name,
        "attack_params": params,
    }


def _make_ensemble_cfg(method_names, sample_idx, sdg_method):
    from attack_defaults import ATTACK_PARAM_DEFAULTS
    from pathlib import Path
    sample_dir = str(Path(DATASET["root"]) / f"sample_{sample_idx:02d}")
    attack_params = {
        "ensembling": {
            "enabled":    True,
            "methods":    method_names,
            "aggregation": "stacking",
            "n_folds":    N_FOLDS,
            "use_probas": USE_PROBAS,
        }
    }
    for name in method_names:
        attack_params[name] = dict(ATTACK_PARAM_DEFAULTS.get(name, {}))
    return {
        "dataset":       {"name": DATASET["name"], "type": DATASET["data_type"],
                          "dir": sample_dir},
        "data_type":     DATASET["data_type"],
        "sdg_method":    sdg_method,
        "attack_method": method_names[0],
        "attack_params": attack_params,
    }


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(train, reconstructed, hidden_features):
    from scoring import calculate_reconstruction_score
    scores = calculate_reconstruction_score(train, reconstructed, hidden_features)
    return dict(zip(hidden_features, scores))


def _mean_ra(feat_scores):
    import numpy as np
    return float(np.mean(list(feat_scores.values()))) if feat_scores else float('nan')


# ── Per-job worker function ───────────────────────────────────────────────────

def run_one_job(job):
    """
    Process one (sdg_method, sample_idx) job.

    Runs all individual attacks + all pairs (soft + stacking + oracle) for this
    (sdg, sample) combo. Returns a list of result dicts.
    """
    sdg_method, sample_idx = job
    _worker_setup()

    import numpy as np
    from attacks import get_attack
    from enhancements.ensembling_wrapper import _soft_voting, _run_stacking_ensemble

    print(f"  [job] SDG={sdg_method}  sample={sample_idx:02d}", flush=True)
    rows = []

    try:
        train, synth, qi, hidden_features = _load_sample(sample_idx, sdg_method)
    except FileNotFoundError as e:
        print(f"    [SKIP] {e}")
        return rows

    # ── Individual attacks ────────────────────────────────────────────────────
    indiv_recon  = {}   # attack_name → reconstructed df
    indiv_probas = {}   # attack_name → probas
    indiv_classes = {}  # attack_name → classes

    for atk in ALL_INDIVIDUAL_ATTACKS:
        cfg = _make_cfg(atk, sample_idx, sdg_method)
        try:
            fn = get_attack(atk, DATASET["data_type"])
            recon, probas, classes = fn(cfg, synth, train, qi, hidden_features)
            indiv_recon[atk]   = recon
            indiv_probas[atk]  = probas
            indiv_classes[atk] = classes
            feat_scores = _score(train, recon, hidden_features)
            rows.append({
                "sdg": sdg_method, "sample": sample_idx,
                "condition": f"individual_{atk}",
                "attack_a": atk, "attack_b": "",
                "ra_mean": _mean_ra(feat_scores),
                **{f"ra_{f}": v for f, v in feat_scores.items()},
            })
        except Exception as e:
            print(f"    [WARN] individual {atk}: {e}")

    # ── Pairs: soft voting + stacking + oracle ────────────────────────────────
    for (atk_a, atk_b) in ATTACK_PAIRS:
        pair_label = f"{atk_a}+{atk_b}"

        # ── Soft voting (reuse cached individual runs) ────────────────────────
        if atk_a in indiv_recon and atk_b in indiv_recon:
            try:
                weights = np.array([0.5, 0.5])
                recon_soft = indiv_recon[atk_a].copy()
                for feat_idx, feat in enumerate(hidden_features):
                    feat_preds  = [indiv_recon[atk_a][feat], indiv_recon[atk_b][feat]]
                    all_probas  = [indiv_probas[atk_a],  indiv_probas[atk_b]]
                    all_classes = [indiv_classes[atk_a], indiv_classes[atk_b]]
                    recon_soft[feat] = _soft_voting(
                        feat_preds, all_probas, all_classes, feat_idx, weights)
                feat_scores = _score(train, recon_soft, hidden_features)
                rows.append({
                    "sdg": sdg_method, "sample": sample_idx,
                    "condition": f"soft_{pair_label}",
                    "attack_a": atk_a, "attack_b": atk_b,
                    "ra_mean": _mean_ra(feat_scores),
                    **{f"ra_{f}": v for f, v in feat_scores.items()},
                })
            except Exception as e:
                print(f"    [WARN] soft {pair_label}: {e}")

        # ── Stacking ──────────────────────────────────────────────────────────
        try:
            fn_a = get_attack(atk_a, DATASET["data_type"])
            fn_b = get_attack(atk_b, DATASET["data_type"])
            cfg  = _make_ensemble_cfg([atk_a, atk_b], sample_idx, sdg_method)
            recon_stack, _, _ = _run_stacking_ensemble(
                [fn_a, fn_b], [atk_a, atk_b],
                cfg, synth, train, qi, hidden_features,
                n_folds=N_FOLDS, use_probas=USE_PROBAS,
            )
            feat_scores = _score(train, recon_stack, hidden_features)
            rows.append({
                "sdg": sdg_method, "sample": sample_idx,
                "condition": f"stack_{pair_label}",
                "attack_a": atk_a, "attack_b": atk_b,
                "ra_mean": _mean_ra(feat_scores),
                **{f"ra_{f}": v for f, v in feat_scores.items()},
            })
        except Exception as e:
            print(f"    [WARN] stack {pair_label}: {e}")

        # ── Pair oracle ───────────────────────────────────────────────────────
        if atk_a in indiv_recon and atk_b in indiv_recon:
            try:
                recon_or = indiv_recon[atk_a].copy()
                for feat in hidden_features:
                    true_vals   = train[feat].astype(str).values
                    pred_a      = indiv_recon[atk_a][feat].astype(str).values
                    pred_b      = indiv_recon[atk_b][feat].astype(str).values
                    oracle_vals = np.where(pred_a == true_vals, pred_a, pred_b)
                    try:
                        recon_or[feat] = oracle_vals.astype(train[feat].dtype)
                    except (ValueError, TypeError):
                        recon_or[feat] = oracle_vals
                feat_scores = _score(train, recon_or, hidden_features)
                rows.append({
                    "sdg": sdg_method, "sample": sample_idx,
                    "condition": f"oracle_{pair_label}",
                    "attack_a": atk_a, "attack_b": atk_b,
                    "ra_mean": _mean_ra(feat_scores),
                    **{f"ra_{f}": v for f, v in feat_scores.items()},
                })
            except Exception as e:
                print(f"    [WARN] oracle {pair_label}: {e}")

    print(f"  [done] SDG={sdg_method}  sample={sample_idx:02d}  "
          f"({len(rows)} rows)", flush=True)
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import numpy as np
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",  type=int, default=N_WORKERS)
    parser.add_argument("--serial",   action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--sdg",      type=str, default=None)
    parser.add_argument("--sample",   type=int, nargs='+', default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, DATASET["name"])
    os.makedirs(out_path, exist_ok=True)

    sdg_methods = [args.sdg]  if args.sdg   else SDG_METHODS
    samples     = args.sample if args.sample is not None else SAMPLES

    jobs = [(sdg, s) for sdg in sdg_methods for s in samples]

    print(f"Stacking ensemble analysis")
    print(f"  Dataset:  {DATASET['name']}  size={DATASET['root'].split('size_')[-1]}")
    print(f"  SDGs:     {sdg_methods}")
    print(f"  Samples:  {samples}")
    print(f"  Pairs:    {ATTACK_PAIRS}")
    print(f"  Workers:  {'serial' if args.serial else args.workers}")
    print(f"  Jobs:     {len(jobs)}")

    if args.dry_run:
        for j in jobs:
            print(f"  [dry-run] {j}")
        return

    # ── Run jobs ──────────────────────────────────────────────────────────────
    all_rows = []

    if args.serial:
        for job in jobs:
            all_rows.extend(run_one_job(job))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            for rows in ex.map(run_one_job, jobs):
                all_rows.extend(rows)

    if not all_rows:
        print("No results — check for errors above.")
        return

    # ── Aggregate and print ───────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # Average over samples
    hidden_features = [c.replace("ra_", "") for c in df.columns
                       if c.startswith("ra_") and c != "ra_mean"]
    avg_cols = ["ra_mean"] + [f"ra_{f}" for f in hidden_features]
    summary  = (df.groupby(["sdg", "condition", "attack_a", "attack_b"])[avg_cols]
                  .mean()
                  .round(4)
                  .reset_index())

    # Save CSV
    csv_path = os.path.join(out_path, "results.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ── Print summary tables ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY  (mean RA %, averaged over {len(samples)} samples)")
    print(f"{'='*70}")

    for sdg in sdg_methods:
        sub = summary[summary["sdg"] == sdg].sort_values("condition")
        print(f"\n  SDG: {sdg}")
        print(f"  {'Condition':<35}  {'RA mean':>8}")
        print(f"  {'─'*46}")
        for _, row in sub.iterrows():
            print(f"  {row['condition']:<35}  {row['ra_mean']:>8.2f}")

        # For each pair: show best_indiv / soft / stack / oracle side by side
        print(f"\n  {'Pair':<25}  {'best indiv':>10}  {'soft':>8}  {'stack':>8}  {'oracle':>8}  {'stack gain':>10}")
        print(f"  {'─'*75}")
        for (atk_a, atk_b) in ATTACK_PAIRS:
            pair = f"{atk_a}+{atk_b}"
            def _get(cond):
                r = sub[sub["condition"] == cond]
                return float(r["ra_mean"].iloc[0]) if len(r) else float('nan')
            ra_a     = _get(f"individual_{atk_a}")
            ra_b     = _get(f"individual_{atk_b}")
            best     = max(ra_a, ra_b)
            soft     = _get(f"soft_{pair}")
            stack    = _get(f"stack_{pair}")
            oracle   = _get(f"oracle_{pair}")
            gain     = stack - best
            gain_str = f"{gain:+.2f}" if not np.isnan(gain) else "  n/a"
            print(f"  {pair:<25}  {best:>10.2f}  {soft:>8.2f}  {stack:>8.2f}  {oracle:>8.2f}  {gain_str:>10}")

    print(f"\nDone. Results in: {out_path}/")


if __name__ == "__main__":
    main()
