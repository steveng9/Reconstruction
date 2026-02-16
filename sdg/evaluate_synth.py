#!/usr/bin/env python
"""
Evaluate quality of generated synthetic data across SDG methods.

Usage (from Reconstruction/):
    python sdg/evaluate_synth.py                     # all datasets
    python sdg/evaluate_synth.py adult               # one dataset
    python sdg/evaluate_synth.py adult california     # multiple datasets
    python sdg/evaluate_synth.py --verbose adult      # per-column detail
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore")

DATA_ROOT = Path("/home/golobs/data/reconstruction_data")


# ============================================================
#  Discovery: find all (dataset, size, sample, method) combos
# ============================================================

def discover_synth_files(datasets=None):
    """Walk DATA_ROOT and yield entry dicts for each synth.csv found."""
    if datasets is None:
        datasets = sorted(d.name for d in DATA_ROOT.iterdir() if d.is_dir())

    for ds in datasets:
        ds_dir = DATA_ROOT / ds
        meta_path = ds_dir / "meta.json"
        if not meta_path.exists():
            print(f"[SKIP] {ds}: no meta.json")
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        for size_dir in sorted(ds_dir.glob("size_*")):
            size = size_dir.name
            for sample_dir in sorted(size_dir.glob("sample_*")):
                sample = sample_dir.name
                train_path = sample_dir / "train.csv"
                if not train_path.exists():
                    continue
                for method_dir in sorted(sample_dir.iterdir()):
                    if not method_dir.is_dir():
                        continue
                    synth_path = method_dir / "synth.csv"
                    if not synth_path.exists():
                        continue
                    yield {
                        "dataset": ds,
                        "size": size,
                        "sample": sample,
                        "method": method_dir.name,
                        "train_path": train_path,
                        "synth_path": synth_path,
                        "meta": meta,
                    }


# ============================================================
#  SDV metadata helper
# ============================================================

def build_sdv_metadata(meta, columns):
    """Build an SDV SingleTableMetadata object from our meta dict."""
    from sdv.metadata import SingleTableMetadata
    sdv_meta = SingleTableMetadata()
    for col in meta.get("categorical", []):
        if col in columns:
            sdv_meta.add_column(col, sdtype="categorical")
    for col in meta.get("continuous", []):
        if col in columns:
            sdv_meta.add_column(col, sdtype="numerical")
    return sdv_meta


# ============================================================
#  Metrics
# ============================================================

def evaluate_one(train_df, synth_df, meta):
    """Compute quality metrics comparing synth_df to train_df.

    Returns (results_dict, per_column_df).
    """
    cat_cols = [c for c in meta.get("categorical", []) if c in train_df.columns and c in synth_df.columns]
    num_cols = [c for c in meta.get("continuous", []) if c in train_df.columns and c in synth_df.columns]

    results = {}
    col_rows = []  # per-column detail rows

    # --- Categorical columns ---
    tvds, jsd_vals = [], []
    for col in cat_cols:
        all_cats = sorted(set(train_df[col].dropna().unique()) | set(synth_df[col].dropna().unique()))
        real_counts = train_df[col].value_counts()
        synth_counts = synth_df[col].value_counts()

        real_probs = np.array([real_counts.get(c, 0) for c in all_cats], dtype=float)
        synth_probs = np.array([synth_counts.get(c, 0) for c in all_cats], dtype=float)
        real_probs /= real_probs.sum() + 1e-12
        synth_probs /= synth_probs.sum() + 1e-12

        tvd = 0.5 * np.abs(real_probs - synth_probs).sum()
        jsd = jensenshannon(real_probs, synth_probs)
        tvds.append(tvd)
        jsd_vals.append(jsd)

        col_rows.append({
            "column": col, "type": "cat",
            "tvd": tvd, "jsd": jsd,
            "n_cats_real": len(real_counts), "n_cats_synth": len(synth_counts),
            "mean_err%": np.nan, "std_err%": np.nan,
        })

    if tvds:
        results["mean_tvd"] = np.mean(tvds)
        results["mean_jsd"] = np.mean(jsd_vals)

    # --- Continuous columns ---
    mean_errs, std_errs = [], []
    for col in num_cols:
        r = train_df[col].dropna()
        s = synth_df[col].dropna()
        if len(r) == 0 or len(s) == 0:
            continue

        r_mean, s_mean = r.mean(), s.mean()
        r_std, s_std = r.std(), s.std()
        mean_err = abs(r_mean - s_mean) / (abs(r_mean) + 1e-10)
        std_err = abs(r_std - s_std) / (abs(r_std) + 1e-10)
        mean_errs.append(mean_err)
        std_errs.append(std_err)

        col_rows.append({
            "column": col, "type": "num",
            "tvd": np.nan, "jsd": np.nan,
            "n_cats_real": np.nan, "n_cats_synth": np.nan,
            "mean_err%": mean_err * 100, "std_err%": std_err * 100,
        })

    if mean_errs:
        results["mean_mean_err%"] = np.mean(mean_errs) * 100
        results["mean_std_err%"] = np.mean(std_errs) * 100

    # --- Correlation matrix difference ---
    if len(num_cols) >= 2:
        real_corr = train_df[num_cols].corr()
        synth_corr = synth_df[num_cols].corr()
        corr_diff = (real_corr - synth_corr).abs()
        results["corr_diff"] = corr_diff.values[np.triu_indices_from(corr_diff, k=1)].mean()

    # --- SDV quality scores ---
    try:
        from sdv.evaluation.single_table import evaluate_quality
        sdv_meta = build_sdv_metadata(meta, train_df.columns)
        quality = evaluate_quality(train_df, synth_df, sdv_meta, verbose=False)
        props = quality.get_properties()
        results["sdv_col_shapes"] = props.loc[props["Property"] == "Column Shapes", "Score"].values[0]
        results["sdv_col_pairs"] = props.loc[props["Property"] == "Column Pair Trends", "Score"].values[0]
    except Exception as e:
        print(f"  [WARN] SDV quality failed: {e}")

    results["n_rows_synth"] = len(synth_df)

    col_df = pd.DataFrame(col_rows) if col_rows else pd.DataFrame()
    return results, col_df


def evaluate_baseline(train_df, meta):
    """Split train in half and evaluate one half against the other.

    This gives a "natural variation" baseline — the expected metric values
    when comparing two genuine samples from the same distribution.
    """
    n = len(train_df)
    half = n // 2
    shuffled = train_df.sample(frac=1, random_state=0).reset_index(drop=True)
    a = shuffled.iloc[:half]
    b = shuffled.iloc[half:2*half]
    results, _ = evaluate_one(a, b, meta)
    results["n_rows_synth"] = len(b)
    return results


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate synthetic data quality")
    parser.add_argument("datasets", nargs="*", help="Dataset names to evaluate (default: all)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-column details")
    args = parser.parse_args()

    datasets = args.datasets or None

    entries = list(discover_synth_files(datasets))
    if not entries:
        print("No synth.csv files found.")
        sys.exit(1)

    print(f"Found {len(entries)} synthetic datasets to evaluate.\n")

    rows = []
    col_detail_rows = []  # for verbose per-column output

    # Cache train data to avoid re-reading and to compute baseline once per sample
    train_cache = {}      # train_path -> train_df
    baseline_cache = {}   # (dataset, size, sample) -> baseline_results

    for entry in entries:
        tp = entry["train_path"]
        if tp not in train_cache:
            train_cache[tp] = pd.read_csv(tp)
        train_df = train_cache[tp]

        # Compute baseline once per sample
        key = (entry["dataset"], entry["size"], entry["sample"])
        if key not in baseline_cache:
            baseline_cache[key] = evaluate_baseline(train_df, entry["meta"])

        synth_df = pd.read_csv(entry["synth_path"])
        metrics, col_df = evaluate_one(train_df, synth_df, entry["meta"])

        row = {
            "dataset": entry["dataset"],
            "size": entry["size"],
            "sample": entry["sample"],
            "method": entry["method"],
        }
        row.update(metrics)
        rows.append(row)

        if args.verbose and len(col_df) > 0:
            col_df = col_df.copy()
            col_df["dataset"] = entry["dataset"]
            col_df["size"] = entry["size"]
            col_df["sample"] = entry["sample"]
            col_df["method"] = entry["method"]
            col_detail_rows.append(col_df)

    # Add baseline rows
    for (ds, sz, samp), bl in baseline_cache.items():
        row = {"dataset": ds, "size": sz, "sample": samp, "method": "~train_baseline"}
        row.update(bl)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Print summary grouped by dataset/size
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    metric_cols = [c for c in df.columns if c not in ["dataset", "size", "sample", "method"]]

    for (ds, sz), group in df.groupby(["dataset", "size"]):
        n_samples = group.loc[group["method"] != "~train_baseline", "sample"].nunique()
        print(f"\n{'='*90}")
        print(f"  {ds} / {sz}  ({n_samples} samples)")
        print(f"{'='*90}")

        summary = group.groupby("method")[metric_cols].mean()

        # Put baseline first, then sort the rest
        methods = sorted([m for m in summary.index if m != "~train_baseline"])
        summary = summary.loc[["~train_baseline"] + methods]

        show_cols = []
        if "mean_tvd" in summary.columns:
            show_cols += ["mean_tvd", "mean_jsd"]
        if "mean_mean_err%" in summary.columns:
            show_cols += ["mean_mean_err%", "mean_std_err%"]
        if "corr_diff" in summary.columns:
            show_cols += ["corr_diff"]
        if "sdv_col_shapes" in summary.columns:
            show_cols += ["sdv_col_shapes", "sdv_col_pairs"]
        show_cols += ["n_rows_synth"]

        print(summary[show_cols].to_string())
        print()

    # Verbose: compact per-column tables, one per (dataset, size, method)
    if args.verbose and col_detail_rows:
        all_cols = pd.concat(col_detail_rows, ignore_index=True)

        print(f"\n{'='*90}")
        print("  PER-COLUMN DETAIL  (averaged across samples)")
        print(f"{'='*90}")

        for (ds, sz), ds_group in all_cols.groupby(["dataset", "size"]):
            print(f"\n--- {ds} / {sz} ---\n")

            # Categorical columns table
            cat = ds_group[ds_group["type"] == "cat"]
            if len(cat) > 0:
                cat_pivot = cat.pivot_table(
                    index="column", columns="method",
                    values="tvd", aggfunc="mean",
                ).round(4)
                print("Categorical — TVD per column (lower=better):")
                print(cat_pivot.to_string())
                print()

            # Continuous columns table
            num = ds_group[ds_group["type"] == "num"]
            if len(num) > 0:
                num_pivot = num.pivot_table(
                    index="column", columns="method",
                    values="mean_err%", aggfunc="mean",
                ).round(2)
                print("Continuous — Mean error % per column (lower=better):")
                print(num_pivot.to_string())
                print()


if __name__ == "__main__":
    main()
