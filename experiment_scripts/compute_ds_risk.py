#!/usr/bin/env python
"""
Compute El Emam et al. (2020a) disclosure risk score ds for synthetic datasets.

    ds = (1/ns) × Σ_s  Es · Is · Rs

where for each target record s and hidden feature f:
  Es  = 1  if the number of real training records sharing s's QI profile ≤ EQ_THRESHOLD
  Is  = 1  if the naive QI-matching attack correctly infers the hidden attribute value
            (find synth records with identical QI values → take the mode → compare to true)
  Rs  = 1  if the true hidden attribute value is rare  (frequency ≤ RARE_THRESHOLD)

This is a *matching-based* attacker (no ML) — it captures risk from the simplest
possible inference: look up who in the synthetic data shares your background attributes.

Outputs a CSV to RESULTS_PATH with one row per (dataset, size, sample, sdg, qi, feature)
plus an "_overall" feature row aggregating across all hidden features.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/golobs/Reconstruction")

from get_data import QIs, minus_QIs  # QI and hidden-feature definitions


# ── Configuration ──────────────────────────────────────────────────────────────

DATASETS = [
    {"base": "adult", "name": "adult", "size": 1_000},
    {"base": "adult", "name": "adult", "size": 10_000},
]

DATA_ROOT_TPL = "/home/golobs/data/reconstruction_data/{base}/size_{size}"

SAMPLE_RANGE = range(5)  # sample_00 through sample_04

SDG_DIRS = [
    "AIM_eps1",
    "MST_eps1",
    "MST_eps10",
    "MST_eps100",
    "MST_eps1000",
]

QI_VARIANTS = ["QI1"]

EQ_THRESHOLD   = 5      # Es=1 if equivalence class size ≤ this
RARE_THRESHOLD = 0.10   # Rs=1 if value frequency ≤ this fraction of training records

RESULTS_PATH = Path(__file__).parent / "ds_risk_scores.csv"


# ── Core computation ──────────────────────────────────────────────────────────

def _equiv_class_sizes(train: pd.DataFrame, qi: list[str]) -> pd.Series:
    """Return a Series (same index as train) with the size of each record's QI group."""
    return train.groupby(qi, observed=True)[qi[0]].transform("count")


def _value_frequencies(train: pd.DataFrame, feature: str) -> dict:
    """Return {value: frequency_fraction} for a column."""
    counts = train[feature].value_counts()
    return (counts / len(train)).to_dict()


def compute_ds_for_sample(
    train: pd.DataFrame,
    synth: pd.DataFrame,
    qi: list[str],
    hidden_features: list[str],
    eq_threshold: int = EQ_THRESHOLD,
    rare_threshold: float = RARE_THRESHOLD,
) -> pd.DataFrame:
    """
    Compute per-feature ds scores for one (sample, SDG, QI) combination.

    Vectorized: pre-computes the modal hidden value per QI group in synth,
    then joins to training records — no per-record Python loops.

    Returns a DataFrame with columns:
        feature, n_targets, n_es, n_is, n_rs, n_eirs, ds
    """
    qi_available     = [c for c in qi if c in synth.columns and c in train.columns]
    hidden_available = [f for f in hidden_features if f in synth.columns and f in train.columns]

    if not qi_available or not hidden_available:
        return pd.DataFrame()

    n_targets = len(train)

    # Es: equivalence class size ≤ threshold
    eq_sizes = train.groupby(qi_available, observed=True)[qi_available[0]].transform("count")
    es_vec   = (eq_sizes.values <= eq_threshold)  # bool array (n_targets,)

    # Rs: per-record per-feature — true value is rare (freq ≤ rare_threshold)
    # Compute value frequencies once per feature
    rs_mat = np.zeros((n_targets, len(hidden_available)), dtype=bool)
    for j, feat in enumerate(hidden_available):
        freq = train[feat].map(train[feat].value_counts() / n_targets)
        rs_mat[:, j] = freq.values <= rare_threshold

    # Is: vectorized QI-matching inference via groupby + mode join
    # For each QI group in synth, compute modal value for each hidden feature.
    # Use first mode (pandas returns sorted modes; first = lowest value on tie — consistent).
    synth_modes = (
        synth.groupby(qi_available, observed=True)[hidden_available]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
        .reset_index()
        .rename(columns={f: f"__inferred_{f}" for f in hidden_available})
    )

    # Join inferred values onto training records by QI
    merged = train[qi_available + hidden_available].merge(
        synth_modes, on=qi_available, how="left"
    )

    is_mat = np.zeros((n_targets, len(hidden_available)), dtype=bool)
    for j, feat in enumerate(hidden_available):
        inferred = merged[f"__inferred_{feat}"]
        true_val = merged[feat]
        # Is=1: match found (not NaN) AND inferred == true
        is_mat[:, j] = inferred.notna().values & (inferred == true_val).values

    # Aggregate per feature
    rows = []
    for j, feat in enumerate(hidden_available):
        eirs = es_vec & is_mat[:, j] & rs_mat[:, j]
        rows.append({
            "feature":   feat,
            "n_targets": n_targets,
            "n_es":      int(es_vec.sum()),
            "n_is":      int(is_mat[:, j].sum()),
            "n_rs":      int(rs_mat[:, j].sum()),
            "n_eirs":    int(eirs.sum()),
            "ds":        round(float(eirs.sum()) / n_targets, 6),
        })

    df = pd.DataFrame(rows)

    # _overall: record contributes if Es=1 AND any feature has Is·Rs=1
    any_eirs = es_vec & (is_mat & rs_mat).any(axis=1)
    df = pd.concat([df, pd.DataFrame([{
        "feature":   "_overall",
        "n_targets": n_targets,
        "n_es":      int(es_vec.sum()),
        "n_is":      int(is_mat.any(axis=1).sum()),
        "n_rs":      int(rs_mat.any(axis=1).sum()),
        "n_eirs":    int(any_eirs.sum()),
        "ds":        round(float(any_eirs.sum()) / n_targets, 6),
    }])], ignore_index=True)

    return df


# ── Main sweep ────────────────────────────────────────────────────────────────

def main(args):
    all_rows = []

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d["name"] == args.dataset]
    sdg_dirs = SDG_DIRS
    if args.sdg:
        # --sdg accepts one or more names; search the data dirs directly (not filtered to SDG_DIRS)
        sdg_dirs = args.sdg

    for ds_cfg in datasets:
        base, name, size = ds_cfg["base"], ds_cfg["name"], ds_cfg["size"]
        data_root = Path(DATA_ROOT_TPL.format(base=base, size=size))

        for sample_idx in SAMPLE_RANGE:
            sample_dir = data_root / f"sample_{sample_idx:02d}"
            train_path = sample_dir / "train.csv"
            if not train_path.exists():
                print(f"  SKIP (no train.csv): {train_path}")
                continue
            train = pd.read_csv(train_path)

            for qi_variant in QI_VARIANTS:
                qi_list = QIs.get(name, {}).get(qi_variant)
                if qi_list is None:
                    print(f"  SKIP (no QI def): {name}/{qi_variant}")
                    continue
                hidden = minus_QIs.get(name, {}).get(qi_variant)
                if hidden is None:
                    # fall back: all non-QI columns
                    hidden = [c for c in train.columns if c not in qi_list]

                for sdg_dir in sdg_dirs:
                    synth_path = sample_dir / sdg_dir / "synth.csv"
                    if not synth_path.exists():
                        print(f"  SKIP (no synth.csv): {synth_path}")
                        continue

                    synth = pd.read_csv(synth_path)
                    print(f"  {name}/{size}/sample_{sample_idx:02d}/{sdg_dir}/{qi_variant} ...", end=" ", flush=True)

                    score_df = compute_ds_for_sample(
                        train, synth, qi_list, hidden,
                        eq_threshold=args.eq_threshold,
                        rare_threshold=args.rare_threshold,
                    )

                    if score_df.empty:
                        print("empty")
                        continue

                    for _, row in score_df.iterrows():
                        all_rows.append({
                            "dataset":        name,
                            "size":           size,
                            "sample":         sample_idx,
                            "sdg":            sdg_dir,
                            "qi":             qi_variant,
                            "eq_threshold":   args.eq_threshold,
                            "rare_threshold": args.rare_threshold,
                            **row.to_dict(),
                        })

                    overall = score_df[score_df["feature"] == "_overall"]
                    if not overall.empty:
                        print(f"ds_overall={overall.iloc[0]['ds']:.4f}")
                    else:
                        print("done")

    if not all_rows:
        print("No results computed.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(all_rows[0].keys())
    mode = "a" if args.append and out_path.exists() else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if mode == "w":
            writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n{'Appended' if mode == 'a' else 'Saved'} {len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        default=None,          help="Filter to one dataset name")
    parser.add_argument("--sdg",            nargs="+", default=None, help="One or more SDG dir names")
    parser.add_argument("--eq-threshold",   type=int,   default=EQ_THRESHOLD,   dest="eq_threshold")
    parser.add_argument("--rare-threshold", type=float, default=RARE_THRESHOLD, dest="rare_threshold")
    parser.add_argument("--out",    default=str(RESULTS_PATH))
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    main(parser.parse_args())
