#!/usr/bin/env python
"""
Comprehensive synthetic data quality evaluation.

Computes fidelity and utility metrics for all synthetic datasets across the
five main experiment datasets (adult, arizona, sbo, cdc_diabetes, california),
averaging results across multiple training samples.

Metrics computed
----------------
Fidelity:
  mean_tvd           mean TVD across categorical columns           (↓ better)
  mean_jsd           mean Jensen-Shannon divergence (categorical)  (↓ better)
  pairwise_tvd       mean TVD on all 2-way joint marginals         (↓ better)
  cat_coverage       fraction of training categories in synth      (↑ better)
  mean_mean_err_pct  relative mean error for continuous columns    (↓ better)
  mean_std_err_pct   relative std error for continuous columns     (↓ better)
  mean_wasserstein   normalized Wasserstein distance (continuous)  (↓ better)
  corr_diff          mean absolute correlation matrix diff         (↓ better)
  sdv_col_shapes     SDV column shapes quality score              (↑ better)
  sdv_col_pairs      SDV column pair trends quality score         (↑ better)

Utility:
  prop_score         propensity AUC (distinguishability proxy)    (↓ better)
  tstr_score         TSTR f1_macro / R²: train-on-synth accuracy  (↑ better)
  trtr_score         TRTR baseline via 3-fold CV on real data      (↑ better)
  tstr_ratio         tstr_score / trtr_score (utility retention)  (↑ better)

A train-vs-train baseline row (one sample vs another) is also included.

Usage (from repo root):
    conda activate recon_
    python experiment_scripts/evaluate_synth_quality.py
    python experiment_scripts/evaluate_synth_quality.py --workers 16
    python experiment_scripts/evaluate_synth_quality.py --dry-run
    python experiment_scripts/evaluate_synth_quality.py --serial     # Ctrl-C killable
    python experiment_scripts/evaluate_synth_quality.py --dataset adult cdc_diabetes
    python experiment_scripts/evaluate_synth_quality.py --method MST_eps1 TabDDPM

Results written to: experiment_scripts/synth_quality_results_<timestamp>.csv
Progress log:        outfiles/eval_quality_progress.log   (tail -f to watch)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")


# ── Dataset configuration ──────────────────────────────────────────────────────

DATA_ROOT = Path("/home/golobs/data/reconstruction_data")

# Each entry: directory paths + ML utility target info.
# "size_dir" is relative to the dataset directory.
# One entry per (dataset, size) combination.  n_samples is an upper bound —
# discover_jobs silently skips sample directories that don't exist on disk.
DATASET_CONFIGS: list[dict] = [
    # ── Adult ──────────────────────────────────────────────────────────────────
    {"dataset": "adult", "size_dir": "size_1000",
     "target": "income", "task": "classification", "n_samples": 5},
    {"dataset": "adult", "size_dir": "size_10000",
     "target": "income", "task": "classification", "n_samples": 5},
    {"dataset": "adult", "size_dir": "size_20000",
     "target": "income", "task": "classification", "n_samples": 5},
    # ── Arizona (25-feature subset) ───────────────────────────────────────────
    {"dataset": "nist_arizona_data", "size_dir": "size_10000_25feat",
     "target": "EMPSTAT", "task": "classification", "n_samples": 5},
    # ── SBO ───────────────────────────────────────────────────────────────────
    {"dataset": "nist_sbo", "size_dir": "size_1000",
     "target": "SEX1", "task": "classification", "n_samples": 5},
    # ── CDC Diabetes ──────────────────────────────────────────────────────────
    {"dataset": "cdc_diabetes", "size_dir": "size_1000",
     "target": "Diabetes_binary", "task": "classification", "n_samples": 5},
    {"dataset": "cdc_diabetes", "size_dir": "size_100000",
     "target": "Diabetes_binary", "task": "classification", "n_samples": 5},
    # ── California ────────────────────────────────────────────────────────────
    {"dataset": "california", "size_dir": "size_1000",
     "target": "MedHouseVal", "task": "regression", "n_samples": 5},
]

# Skip pairwise TVD when a dataset has more than this many categorical columns
# (avoids O(n²) groupby chains on wide datasets like nist_sbo with 125 cat cols)
PAIRWISE_TVD_MAX_CAT_COLS = 30

# Cap rows used for propensity score and TRTR cross-validation
PROPENSITY_MAX_ROWS = 10_000
TRTR_CV_MAX_ROWS    = 10_000

N_WORKERS = 8


# ── Job discovery ──────────────────────────────────────────────────────────────

def discover_jobs(ds_filter: list[str] | None = None,
                  method_filter: list[str] | None = None) -> list[dict]:
    """Enumerate all (dataset, size, sample, method) evaluation jobs."""
    jobs: list[dict] = []

    for cfg in DATASET_CONFIGS:
        ds_name = cfg["dataset"]
        if ds_filter and ds_name not in ds_filter:
            continue

        ds_dir    = DATA_ROOT / ds_name
        size_dir  = ds_dir / cfg["size_dir"]
        meta_path = ds_dir / "meta.json"

        if not size_dir.exists():
            print(f"[SKIP] {ds_name}/{cfg['size_dir']}: directory not found")
            continue
        if not meta_path.exists():
            print(f"[SKIP] {ds_name}: no meta.json")
            continue

        for sample_idx in range(cfg["n_samples"]):
            sample_name = f"sample_{sample_idx:02d}"
            sample_dir  = size_dir / sample_name
            if not sample_dir.exists():
                continue
            train_path = sample_dir / "train.csv"
            if not train_path.exists():
                continue

            for method_dir in sorted(sample_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                synth_path = method_dir / "synth.csv"
                if not synth_path.exists():
                    continue
                if method_filter and method_dir.name not in method_filter:
                    continue

                jobs.append({
                    "dataset":     ds_name,
                    "size_dir":    cfg["size_dir"],
                    "sample":      sample_name,
                    "method":      method_dir.name,
                    "train_path":  str(train_path),
                    "synth_path":  str(synth_path),
                    "meta_path":   str(meta_path),
                    "target":      cfg["target"],
                    "task":        cfg["task"],
                    "is_baseline": False,
                })

    # Train-vs-train baseline: each sample paired with the next (round-robin)
    for cfg in DATASET_CONFIGS:
        ds_name = cfg["dataset"]
        if ds_filter and ds_name not in ds_filter:
            continue

        ds_dir    = DATA_ROOT / ds_name
        size_dir  = ds_dir / cfg["size_dir"]
        meta_path = ds_dir / "meta.json"

        if not size_dir.exists() or not meta_path.exists():
            continue

        sample_dirs = sorted([
            d for d in size_dir.iterdir()
            if d.is_dir() and d.name.startswith("sample_")
            and (d / "train.csv").exists()
        ])[:cfg["n_samples"]]

        if len(sample_dirs) < 2:
            continue

        for i, sample_dir in enumerate(sample_dirs):
            paired_dir = sample_dirs[(i + 1) % len(sample_dirs)]
            jobs.append({
                "dataset":     ds_name,
                "size_dir":    cfg["size_dir"],
                "sample":      sample_dir.name,
                "method":      "~train_baseline",
                "train_path":  str(sample_dir / "train.csv"),
                "synth_path":  str(paired_dir / "train.csv"),
                "meta_path":   str(meta_path),
                "target":      cfg["target"],
                "task":        cfg["task"],
                "is_baseline": True,
            })

    return jobs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_cat_dtypes(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Normalize categorical columns to consistent string representation.

    Handles int/float mismatches (e.g. "1" vs "1.0") that arise when different
    SDG methods output integer codes as floats.
    """
    if not cat_cols:
        return df
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace(r'^(-?\d+)\.0$', r'\1', regex=True))
    return df


def _filter_meta(meta: dict, columns) -> dict:
    """Return meta restricted to columns actually present in the dataframe."""
    cols = set(columns)
    return {
        "categorical": [c for c in meta.get("categorical", []) if c in cols],
        "continuous":  [c for c in meta.get("continuous",  []) if c in cols],
    }


# ── Fidelity metrics ───────────────────────────────────────────────────────────

def compute_fidelity(train_df: pd.DataFrame, synth_df: pd.DataFrame,
                     meta: dict) -> dict:
    """Compute all fidelity metrics comparing synth_df to train_df."""
    cat_cols = meta.get("categorical", [])
    num_cols = meta.get("continuous", [])

    train_df = _normalize_cat_dtypes(train_df, cat_cols)
    synth_df = _normalize_cat_dtypes(synth_df, cat_cols)

    results: dict = {}

    # ── Categorical ────────────────────────────────────────────────────────────
    if cat_cols:
        tvds, jsds, coverages = [], [], []
        for col in cat_cols:
            train_cats = set(train_df[col].dropna().unique())
            synth_cats = set(synth_df[col].dropna().unique())
            all_cats   = sorted(train_cats | synth_cats)

            tc = train_df[col].value_counts()
            sc = synth_df[col].value_counts()
            tp = np.array([tc.get(c, 0) for c in all_cats], dtype=float)
            sp = np.array([sc.get(c, 0) for c in all_cats], dtype=float)
            tp /= tp.sum() + 1e-12
            sp /= sp.sum() + 1e-12

            tvds.append(0.5 * np.abs(tp - sp).sum())
            jsds.append(float(jensenshannon(tp, sp)))
            coverages.append(
                len(synth_cats & train_cats) / max(len(train_cats), 1)
            )

        results["mean_tvd"]      = float(np.mean(tvds))
        results["mean_jsd"]      = float(np.mean(jsds))
        results["cat_coverage"]  = float(np.mean(coverages))

        # Pairwise TVD — skip for wide datasets (would be O(C²) with C=125 for SBO)
        if 2 <= len(cat_cols) <= PAIRWISE_TVD_MAX_CAT_COLS:
            pair_tvds = []
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    c1, c2 = cat_cols[i], cat_cols[j]
                    rj = (train_df[[c1, c2]].dropna()
                          .groupby([c1, c2]).size() / len(train_df))
                    sj = (synth_df[[c1, c2]].dropna()
                          .groupby([c1, c2]).size() / len(synth_df))
                    all_keys = rj.index.union(sj.index)
                    rv = rj.reindex(all_keys, fill_value=0.0).values
                    sv = sj.reindex(all_keys, fill_value=0.0).values
                    pair_tvds.append(0.5 * float(np.abs(rv - sv).sum()))
            results["pairwise_tvd"] = float(np.mean(pair_tvds))

    # ── Continuous ─────────────────────────────────────────────────────────────
    if num_cols:
        mean_errs, std_errs, wass_vals = [], [], []
        for col in num_cols:
            r = train_df[col].dropna().values.astype(float)
            s = synth_df[col].dropna().values.astype(float)
            if len(r) == 0 or len(s) == 0:
                continue

            r_mean, s_mean = r.mean(), s.mean()
            r_std,  s_std  = r.std(),  s.std()
            mean_errs.append(abs(r_mean - s_mean) / (abs(r_mean) + 1e-10))
            std_errs.append( abs(r_std  - s_std)  / (abs(r_std)  + 1e-10))

            r_range = r.max() - r.min()
            if r_range > 0:
                # Normalized Wasserstein: divide by range so values are in [0, 1]
                wass_vals.append(wasserstein_distance(r, s) / r_range)

        if mean_errs:
            results["mean_mean_err_pct"] = float(np.mean(mean_errs) * 100)
            results["mean_std_err_pct"]  = float(np.mean(std_errs)  * 100)
        if wass_vals:
            results["mean_wasserstein"] = float(np.mean(wass_vals))

    # ── Correlation (continuous pairs) ─────────────────────────────────────────
    if len(num_cols) >= 2:
        rc = train_df[num_cols].corr().values
        sc = synth_df[num_cols].corr().values
        idx = np.triu_indices(len(num_cols), k=1)
        results["corr_diff"] = float(np.abs(rc[idx] - sc[idx]).mean())

    # ── SDV quality ────────────────────────────────────────────────────────────
    try:
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.metadata import SingleTableMetadata

        sdv_meta = SingleTableMetadata()
        for col in cat_cols:
            sdv_meta.add_column(col, sdtype="categorical")
        for col in num_cols:
            sdv_meta.add_column(col, sdtype="numerical")

        quality = evaluate_quality(train_df, synth_df, sdv_meta, verbose=False)
        props = quality.get_properties()
        results["sdv_col_shapes"] = float(
            props.loc[props["Property"] == "Column Shapes", "Score"].values[0])
        results["sdv_col_pairs"] = float(
            props.loc[props["Property"] == "Column Pair Trends", "Score"].values[0])
    except Exception as e:
        results["sdv_error"] = repr(e)

    return results


# ── ML utility metrics ─────────────────────────────────────────────────────────

def _encode_features(train_df: pd.DataFrame, synth_df: pd.DataFrame,
                     cat_cols: list[str], num_cols: list[str],
                     target: str):
    """OrdinalEncode categorical features; return (X_real, X_synth, y_real, y_synth)."""
    from sklearn.preprocessing import OrdinalEncoder

    feature_cols = [c for c in (cat_cols + num_cols) if c != target
                    and c in train_df.columns and c in synth_df.columns]
    cat_feats    = [c for c in cat_cols if c in feature_cols]

    X_real  = train_df[feature_cols].copy()
    X_synth = synth_df[feature_cols].copy()
    y_real  = train_df[target].copy()
    y_synth = synth_df[target].copy()

    # Normalize dtype so OrdinalEncoder sees matching strings
    X_real  = _normalize_cat_dtypes(X_real,  cat_feats)
    X_synth = _normalize_cat_dtypes(X_synth, cat_feats)

    if cat_feats:
        combined = pd.concat([X_real[cat_feats], X_synth[cat_feats]], ignore_index=True)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(combined)
        X_real[cat_feats]  = enc.transform(X_real[cat_feats])
        X_synth[cat_feats] = enc.transform(X_synth[cat_feats])

    X_real  = X_real.fillna(-1).values.astype(float)
    X_synth = X_synth.fillna(-1).values.astype(float)

    return X_real, X_synth, y_real, y_synth


def compute_ml_utility(train_df: pd.DataFrame, synth_df: pd.DataFrame,
                       meta: dict, target: str, task: str) -> dict:
    """Train-on-Synthetic Test-on-Real (TSTR) + TRTR cross-val baseline.

    Returns tstr_score, trtr_score, tstr_ratio.
    For classification: F1 macro. For regression: R².
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import f1_score, r2_score
    from sklearn.model_selection import cross_val_score

    results: dict = {}

    if target not in train_df.columns or target not in synth_df.columns:
        return results

    cat_cols = meta.get("categorical", [])
    num_cols = meta.get("continuous", [])

    try:
        X_real, X_synth, y_real, y_synth = _encode_features(
            train_df, synth_df, cat_cols, num_cols, target
        )

        # Cap real data for cross-validation to keep runtime bounded
        n_cv = min(len(X_real), TRTR_CV_MAX_ROWS)
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(X_real), n_cv, replace=False)
        X_cv, y_cv = X_real[idx], y_real.iloc[idx]

        if task == "classification":
            y_real_str  = y_real.astype(str).values
            y_synth_str = y_synth.astype(str).values
            y_cv_str    = y_cv.astype(str).values

            rf_cls = dict(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
            # TSTR: train on synth, evaluate on full real
            clf = RandomForestClassifier(**rf_cls)
            clf.fit(X_synth, y_synth_str)
            tstr = f1_score(y_real_str, clf.predict(X_real),
                            average="macro", zero_division=0)

            # TRTR: 3-fold CV on (capped) real data
            scores = cross_val_score(
                RandomForestClassifier(**rf_cls), X_cv, y_cv_str,
                cv=3, scoring="f1_macro", error_score=0.0,
            )
            trtr = float(scores.mean())

        else:  # regression
            rf_reg = dict(n_estimators=50, max_depth=8, random_state=42, n_jobs=1)
            # TSTR
            reg = RandomForestRegressor(**rf_reg)
            reg.fit(X_synth, y_synth.values)
            tstr = float(r2_score(y_real.values, reg.predict(X_real)))

            # TRTR
            scores = cross_val_score(
                RandomForestRegressor(**rf_reg), X_cv, y_cv.values,
                cv=3, scoring="r2", error_score=0.0,
            )
            trtr = float(scores.mean())

        results["tstr_score"] = float(tstr)
        results["trtr_score"] = float(trtr)
        # Ratio clipped at 1.1 so pathological synth doesn't blow up the table
        results["tstr_ratio"] = float(np.clip(tstr / (abs(trtr) + 1e-10), 0.0, 1.1))

    except Exception as e:
        print(f"  [WARN] ML utility failed: {e}")

    return results


# ── Propensity score ───────────────────────────────────────────────────────────

def compute_propensity(train_df: pd.DataFrame, synth_df: pd.DataFrame,
                       meta: dict) -> dict:
    """Propensity score: AUC of a classifier trained to distinguish real vs synth.

    Score 0.5 = indistinguishable (good), 1.0 = perfectly distinguishable (bad).
    Uses logistic regression + 3-fold cross-validation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import OrdinalEncoder

    results: dict = {}

    cat_cols = meta.get("categorical", [])
    num_cols = meta.get("continuous", [])
    feature_cols = [c for c in (cat_cols + num_cols)
                    if c in train_df.columns and c in synth_df.columns]
    if not feature_cols:
        return results

    try:
        r = train_df[feature_cols].copy()
        s = synth_df[feature_cols].copy()

        # Cap to keep computation bounded
        if len(r) > PROPENSITY_MAX_ROWS:
            r = r.sample(PROPENSITY_MAX_ROWS, random_state=42)
        if len(s) > PROPENSITY_MAX_ROWS:
            s = s.sample(PROPENSITY_MAX_ROWS, random_state=42)

        cat_feats = [c for c in cat_cols if c in feature_cols]
        r = _normalize_cat_dtypes(r, cat_feats)
        s = _normalize_cat_dtypes(s, cat_feats)

        if cat_feats:
            combined = pd.concat([r[cat_feats], s[cat_feats]], ignore_index=True)
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            enc.fit(combined)
            r[cat_feats] = enc.transform(r[cat_feats])
            s[cat_feats] = enc.transform(s[cat_feats])

        r = r.fillna(-1)
        s = s.fillna(-1)

        X = np.vstack([r.values.astype(float), s.values.astype(float)])
        y = np.array([0] * len(r) + [1] * len(s))

        clf   = LogisticRegression(max_iter=500, random_state=42, n_jobs=1, C=1.0)
        aucs  = cross_val_score(clf, X, y, cv=3, scoring="roc_auc")
        results["prop_score"] = float(aucs.mean())

    except Exception as e:
        print(f"  [WARN] Propensity score failed: {e}")

    return results


# ── Per-job evaluation entry point ─────────────────────────────────────────────

def evaluate_job(job: dict) -> dict:
    """Evaluate one (dataset, sample, method) job; return flat results dict."""
    import warnings
    warnings.filterwarnings("ignore")

    with open(job["meta_path"]) as f:
        meta_raw = json.load(f)

    train_df = pd.read_csv(job["train_path"])
    synth_df = pd.read_csv(job["synth_path"])

    # Filter meta to columns present in this feature subset (e.g. 25-feat arizona)
    meta = _filter_meta(meta_raw, train_df.columns)

    result: dict = {
        "dataset":      job["dataset"],
        "size_dir":     job["size_dir"],
        "sample":       job["sample"],
        "method":       job["method"],
        "n_rows_train": len(train_df),
        "n_rows_synth": len(synth_df),
    }

    result.update(compute_fidelity(train_df, synth_df, meta))

    if not job["is_baseline"]:
        result.update(compute_ml_utility(
            train_df, synth_df, meta, job["target"], job["task"]
        ))
        result.update(compute_propensity(train_df, synth_df, meta))

    return result


# ── Multiprocessing worker ─────────────────────────────────────────────────────

def _worker(job: dict) -> tuple[str, dict | None, str | None]:
    """Worker entry point: run evaluation; return (key, result, error_or_None)."""
    import warnings
    warnings.filterwarnings("ignore")
    key = f"{job['dataset']}/{job['size_dir']}/{job['sample']}/{job['method']}"
    try:
        return key, evaluate_job(job), None
    except Exception:
        return key, None, traceback.format_exc()


# ── Progress logging ───────────────────────────────────────────────────────────

def _log(log_path: Path, msg: str) -> None:
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthetic data quality across all datasets and SDG methods."
    )
    parser.add_argument("--workers",      "-j", type=int, default=N_WORKERS,
                        help=f"Parallel workers (default: {N_WORKERS})")
    parser.add_argument("--dry-run",      action="store_true",
                        help="Print job list without running")
    parser.add_argument("--serial",       action="store_true",
                        help="Run sequentially in main process (Ctrl-C killable)")
    parser.add_argument("--progress-log", default="outfiles/eval_quality_progress.log",
                        help="Progress log file (tail -f to watch live)")
    parser.add_argument("--out",          default=None,
                        help="Output CSV path (default: experiment_scripts/synth_quality_results_<ts>.csv)")
    parser.add_argument("--dataset",      nargs="*",
                        help="Restrict to these dataset names")
    parser.add_argument("--method",       nargs="*",
                        help="Restrict to these SDG method names")
    args = parser.parse_args()

    log_path = Path(args.progress_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")   # clear on each run

    jobs = discover_jobs(ds_filter=args.dataset, method_filter=args.method)

    _log(log_path, f"Found {len(jobs)} evaluation jobs "
                   f"({sum(not j['is_baseline'] for j in jobs)} synth + "
                   f"{sum(j['is_baseline'] for j in jobs)} baselines).")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['dataset']}/{j['size_dir']}/{j['sample']}/{j['method']}")
        return

    out_path = (Path(args.out) if args.out else
                Path(__file__).parent /
                f"synth_quality_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    results: list[dict] = []
    n_done = n_fail = 0

    if args.serial:
        for job in jobs:
            key, res, err = _worker(job)
            n_done += 1
            if err:
                n_fail += 1
                _log(log_path,
                     f"FAIL [{n_done}/{len(jobs)}] {key}: {err.splitlines()[-1]}")
            else:
                results.append(res)
                _log(log_path, f"OK   [{n_done}/{len(jobs)}] {key}")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = {pool.submit(_worker, j): j for j in jobs}
            for future in as_completed(futures):
                key, res, err = future.result()
                n_done += 1
                if err:
                    n_fail += 1
                    _log(log_path,
                         f"FAIL [{n_done}/{len(jobs)}] {key}: {err.splitlines()[-1]}")
                else:
                    results.append(res)
                    _log(log_path, f"OK   [{n_done}/{len(jobs)}] {key}")

    _log(log_path,
         f"Complete: {n_done - n_fail}/{len(jobs)} succeeded, {n_fail} failed.")

    if not results:
        print("No results to write.")
        return

    df = pd.DataFrame(results)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _log(log_path, f"Results written to: {out_path}")
    print(f"\nResults written to: {out_path}")

    # Quick summary table printed to stdout
    metric_cols = [c for c in df.columns
                   if c not in {"dataset", "size_dir", "sample", "method",
                                "n_rows_train", "n_rows_synth"}]
    summary = (df.groupby(["dataset", "method"])[metric_cols]
                 .mean()
                 .round(4))
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)
    print("\n── Summary (averaged across samples) ──")
    print(summary.to_string())


if __name__ == "__main__":
    main()
