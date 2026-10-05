#!/usr/bin/env python
"""
compute_wasserstein_ohe.py — custom one-hot Wasserstein distance for all
synth.csv files on disk.

Metric (2026-09-11: now a faithful port of SyntheticData_MIA, the implementation
behind golob_privacy_2025, which the paper cites for this metric):
  * numeric columns (pandas numeric dtype) -> 20 equal-depth "baskets" built the
    SyntheticData_MIA way (encode_data.fit_discrete_features_evenly): sort the
    reference column, np.array_split it into 20 chunks, take each chunk's first
    value as a basket edge, and when a chunk starts on an already-used value
    (a spike, e.g. capital-gain's 92% zeros) bump the edge by +1 instead of
    dropping it. Values are assigned with np.digitize.
  * other columns -> one-hot over their categories.
  * distance = sum over all one-hot columns of |freq_train - freq_synth|
    (util.wasserstein_distance), i.e. the sum of per-feature TVDs x 2.

The previous version used pd.qcut(..., duplicates="drop"), which collapses a
spiky column to one or two bins (capital-gain -> 2 bins), so it could not see
that pre-binned DP releases decode capital-gain's zero bin to its midpoint
(2500). That hid ~1.8 per zero-inflated column. Kept as
`compute_wasserstein_ohe_legacy` for comparison only.

Deviation from SyntheticData_MIA, one only: its first basket edge is the
constant 0, which sends every negative value to the last basket (np.eye[-1]).
We use min(0, min(reference)) instead; this is identical for non-negative data
and only matters for California Housing's longitude.

The result is a separate join-key CSV: (dataset, size_dir, sample, method,
wasserstein_ohe).  Merge into any quality table on those four columns.

Usage:
    conda activate recon_
    python experiment_scripts/compute_wasserstein_ohe.py
    python experiment_scripts/compute_wasserstein_ohe.py --workers 8
    python experiment_scripts/compute_wasserstein_ohe.py --dry-run
    python experiment_scripts/compute_wasserstein_ohe.py --dataset adult cdc_diabetes
    python experiment_scripts/compute_wasserstein_ohe.py --out my_results.csv
"""

from __future__ import annotations
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT


import argparse
import csv
import json
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_ROOT   = Path(str(DATA_ROOT))

N_BINS      = 20   # equal-depth bins for continuous features


# ── Metric ────────────────────────────────────────────────────────────────────

def _col_freq_diffs(train_col: pd.Series, synth_col: pd.Series,
                    is_categorical: bool) -> float:
    """Sum of |freq_train(v) - freq_synth(v)| over one-hot columns for one feature."""
    if is_categorical:
        cats = sorted(set(train_col.astype(str).dropna()))
        t = train_col.astype(str)
        s = synth_col.astype(str)
        return sum(abs((t == c).mean() - (s == c).mean()) for c in cats)
    else:
        # Equal-depth bins fitted on train, applied to both
        try:
            _, edges = pd.qcut(train_col.dropna(), q=N_BINS, retbins=True,
                               duplicates="drop")
        except ValueError:
            # Fewer unique values than bins — fall back to unique-value encoding
            cats = sorted(set(train_col.dropna()))
            return sum(abs((train_col == c).mean() - (synth_col == c).mean())
                       for c in cats)
        edges[0]  = -np.inf
        edges[-1] =  np.inf
        t_bins = pd.cut(train_col, bins=edges, labels=False)
        s_bins = pd.cut(synth_col, bins=edges, labels=False)
        n_actual = len(edges) - 1
        return sum(abs((t_bins == b).mean() - (s_bins == b).mean())
                   for b in range(n_actual))


def compute_wasserstein_ohe_legacy(train_df: pd.DataFrame, synth_df: pd.DataFrame,
                             meta: dict) -> float:
    """
    Compute the custom one-hot Wasserstein distance between train and synth.

    Returns a non-negative float.  Lower is better (synth matches train).
    """
    cat_set  = set(meta.get("categorical", []) + meta.get("ordinal", []))
    cont_set = set(meta.get("continuous", []))
    common   = [c for c in train_df.columns
                if c in synth_df.columns and (c in cat_set or c in cont_set)]

    total = 0.0
    for col in common:
        total += _col_freq_diffs(train_df[col], synth_df[col],
                                 is_categorical=(col in cat_set))
    return total


def _mia_basket_edges(ref: pd.Series, n_bins: int = N_BINS) -> list:
    """SyntheticData_MIA encode_data.fit_discrete_features_evenly, numeric branch."""
    vals = np.sort(ref.dropna().values)
    splits = np.array_split(vals, n_bins)
    edges = [min(0, vals[0]) if len(vals) else 0]
    for i in range(1, n_bins):
        first = splits[i][0] if len(splits[i]) else edges[i - 1] + 1
        edges.append(first if first > edges[i - 1] else edges[i - 1] + 1)
    return edges


def _mia_freqs(col: pd.Series, edges: list, n_bins: int = N_BINS) -> np.ndarray:
    """np.eye(n_bins)[np.digitize(col, edges) - 1], summed and normalised."""
    idx = np.digitize(col.values, edges) - 1          # -1 wraps to the last basket, as in the original
    return np.bincount(idx % n_bins, minlength=n_bins) / len(col)


def compute_wasserstein_ohe_mia_port(train_df: pd.DataFrame, synth_df: pd.DataFrame,
                                    meta: dict | None = None) -> float:
    """
    One-hot Wasserstein distance, ported from SyntheticData_MIA
    (util.wasserstein_distance + encode_data.binarize_discrete_features_evenly).
    `meta` is accepted for call compatibility; as in the original, the numeric
    vs one-hot decision follows the column dtype.
    """
    from pandas.api.types import is_numeric_dtype
    total = 0.0
    for col in train_df.columns:
        if col not in synth_df.columns:
            continue
        t, s = train_df[col], synth_df[col]
        if is_numeric_dtype(t) and is_numeric_dtype(s):
            e = _mia_basket_edges(t)
            total += float(np.abs(_mia_freqs(t, e) - _mia_freqs(s, e)).sum())
        else:
            t, s = t.astype(str), s.astype(str)
            cats = sorted(set(t.unique()) | set(s.unique()))
            total += float(np.abs(t.value_counts(normalize=True).reindex(cats, fill_value=0).values
                                  - s.value_counts(normalize=True).reindex(cats, fill_value=0).values).sum())
    return total


def _freqs_over_union(t: pd.Series, s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Per-value frequencies of two columns over the union of their values.
    Numeric-looking columns are compared as floats so 1 and 1.0 are one value."""
    try:
        t, s = t.astype(float), s.astype(float)
    except (ValueError, TypeError):
        t, s = t.astype(str), s.astype(str)
    cats = sorted(set(t.dropna().unique()) | set(s.dropna().unique()))
    return (t.value_counts(normalize=True).reindex(cats, fill_value=0).values,
            s.value_counts(normalize=True).reindex(cats, fill_value=0).values)


def compute_wasserstein_ohe(train_df: pd.DataFrame, synth_df: pd.DataFrame,
                            meta: dict) -> float:
    """
    One-hot Wasserstein distance as reported in the paper (2026-09-11).

    Categorical / ordinal features (per meta) are one-hot encoded over their
    values; continuous features are cut into the same N_BINS equal-width bins
    over the real column's [min, max] that every pre-binned DP generator in
    sdg/ uses (np.linspace + np.searchsorted, synth values outside the range
    clipped to the end bins). The distance is the sum over all one-hot columns
    of |freq_train - freq_synth|, as in SyntheticData_MIA's
    util.wasserstein_distance.

    Why not SyntheticData_MIA's equal-depth baskets (compute_wasserstein_ohe_mia_port):
    those baskets are finest exactly where real values pile up (capital-gain's
    zeros), so a DP release that writes each bin back as its midpoint
    (0 -> 2500) is charged ~1.8 per such column at every epsilon, a fixed floor
    (~7.1 on Adult) that measures the decoding, not the generator.
    """
    cont = set(meta.get("continuous", []))
    feats = set(meta.get("categorical", [])) | set(meta.get("ordinal", [])) | cont
    total = 0.0
    for col in train_df.columns:
        if col not in synth_df.columns or col not in feats:
            continue
        t, s = train_df[col], synth_df[col]
        if col in cont:
            lo, hi = t.min(), t.max()
            if lo == hi:
                continue                      # one bin: both sides put all mass there
            inner = np.linspace(lo, hi, N_BINS + 1)[1:-1]
            ft = np.bincount(np.searchsorted(inner, t.astype(float).values), minlength=N_BINS) / len(t)
            fs = np.bincount(np.searchsorted(inner, s.astype(float).values), minlength=N_BINS) / len(s)
        else:
            ft, fs = _freqs_over_union(t, s)
        total += float(np.abs(ft - fs).sum())
    return total


# ── Job discovery ─────────────────────────────────────────────────────────────

def _discover_jobs(ds_filter: list[str] | None) -> list[dict]:
    """Find all (dataset, size_dir, sample, method) combos with a synth.csv."""
    jobs = []
    for ds_dir in sorted(DATA_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        if ds_filter and dataset not in ds_filter:
            continue
        meta_path = ds_dir / "meta.json"
        if not meta_path.exists():
            continue
        for size_dir in sorted(ds_dir.iterdir()):
            if not size_dir.is_dir() or not size_dir.name.startswith("size_"):
                continue
            for sample_dir in sorted(size_dir.iterdir()):
                if not sample_dir.is_dir() or not sample_dir.name.startswith("sample_"):
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
                    jobs.append({
                        "dataset":    dataset,
                        "size_dir":   size_dir.name,
                        "sample":     sample_dir.name,
                        "method":     method_dir.name,
                        "train_path": str(train_path),
                        "synth_path": str(synth_path),
                        "meta_path":  str(meta_path),
                    })
    return jobs


def _load_done_keys(out_path: Path) -> set[str]:
    """Read existing output CSV and return set of already-computed keys."""
    done = set()
    if not out_path.exists():
        return done
    try:
        df = pd.read_csv(out_path, usecols=["dataset", "size_dir", "sample", "method"])
        for _, row in df.iterrows():
            done.add(f"{row.dataset}|{row.size_dir}|{row.sample}|{row.method}")
    except Exception:
        pass
    return done


# ── Worker ────────────────────────────────────────────────────────────────────

def _worker(job: dict) -> tuple[dict | None, str | None]:
    import warnings
    warnings.filterwarnings("ignore")
    try:
        with open(job["meta_path"]) as f:
            meta_raw = json.load(f)

        train_df = pd.read_csv(job["train_path"])
        synth_df = pd.read_csv(job["synth_path"])

        # Filter meta to columns present in train (handles 25-feat subsets)
        cols = set(train_df.columns)
        meta = {
            "categorical": [c for c in meta_raw.get("categorical", []) if c in cols],
            "continuous":  [c for c in meta_raw.get("continuous",  []) if c in cols],
            "ordinal":     [c for c in meta_raw.get("ordinal",     []) if c in cols],
        }

        wd = compute_wasserstein_ohe(train_df, synth_df, meta)

        return {
            "dataset":         job["dataset"],
            "size_dir":        job["size_dir"],
            "sample":          job["sample"],
            "method":          job["method"],
            "wasserstein_ohe": wd,
        }, None

    except Exception:
        return None, traceback.format_exc()


# ── CSV writer ────────────────────────────────────────────────────────────────

_COLUMNS = ["dataset", "size_dir", "sample", "method", "wasserstein_ohe"]


def _write_row(path: Path, row: dict, wrote_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        if not wrote_header:
            w.writeheader()
        w.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute custom one-hot Wasserstein distance for all synth.csv files."
    )
    parser.add_argument("--workers",  "-j", type=int, default=8,
                        help="Parallel workers (default 8).")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print jobs and exit without running.")
    parser.add_argument("--dataset",  nargs="+", default=None,
                        help="Restrict to these dataset names.")
    parser.add_argument("--out",      default=None,
                        help="Output CSV path (default: experiment_scripts/wasserstein_ohe_<ts>.csv).")
    args = parser.parse_args()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else SCRIPTS_DIR / f"wasserstein_ohe_{ts}.csv"

    all_jobs  = _discover_jobs(args.dataset)
    done_keys = _load_done_keys(out_path)

    jobs = [j for j in all_jobs
            if f"{j['dataset']}|{j['size_dir']}|{j['sample']}|{j['method']}"
               not in done_keys]

    print(f"[{datetime.now():%H:%M:%S}] Found {len(all_jobs)} synth.csv files on disk.")
    print(f"[{datetime.now():%H:%M:%S}] {len(done_keys)} already computed, "
          f"{len(jobs)} remaining.")
    print(f"[{datetime.now():%H:%M:%S}] Output: {out_path}")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['dataset']}/{j['size_dir']}/{j['sample']}/{j['method']}")
        return

    if not jobs:
        print("Nothing to do.")
        return

    wrote_header = out_path.exists()
    ctx = mp.get_context("spawn")
    ok = fail = 0

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
        future_to_job = {pool.submit(_worker, j): j for j in jobs}
        for fut in as_completed(future_to_job):
            j = future_to_job[fut]
            label = f"{j['dataset']}/{j['size_dir']}/{j['sample']}/{j['method']}"
            try:
                result, err = fut.result()
            except Exception:
                err = traceback.format_exc()
                result = None

            if err or result is None:
                last_line = (err or "unknown error").strip().splitlines()[-1]
                print(f"[{datetime.now():%H:%M:%S}] FAIL  {label}: {last_line}")
                fail += 1
            else:
                _write_row(out_path, result, wrote_header)
                wrote_header = True
                print(f"[{datetime.now():%H:%M:%S}] OK    {label}  wd={result['wasserstein_ohe']:.4f}")
                ok += 1

    print(f"\n[{datetime.now():%H:%M:%S}] Done — ok={ok}  failed={fail}")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
