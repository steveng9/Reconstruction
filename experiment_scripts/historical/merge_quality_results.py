#!/usr/bin/env python
"""
merge_quality_results.py — merge all quality result CSVs into one file.

Combines:
  - synth_quality_results_*.csv      (original evaluation runs)
  - fill_in_quality_results_*.csv    (fill-in evaluation runs)
  - wasserstein_ohe_*.csv            (joined on dataset/size_dir/sample/method)

Deduplication: when the same (dataset, size_dir, sample, method) key appears in
multiple files, the most recently dated file wins (files are sorted by name,
which encodes a timestamp).

Usage:
    python experiment_scripts/merge_quality_results.py
    python experiment_scripts/merge_quality_results.py --out my_merged.csv
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
KEY_COLS    = ["dataset", "size_dir", "sample", "method"]


def main():
    parser = argparse.ArgumentParser(description="Merge all synth quality CSVs.")
    parser.add_argument("--out", default=None,
                        help="Output path (default: experiment_scripts/quality_results_merged.csv)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else SCRIPTS_DIR / "quality_results_merged.csv"

    # ── Load and stack all quality metric CSVs (sorted by name = chronological) ─
    quality_files = sorted(
        glob.glob(str(SCRIPTS_DIR / "synth_quality_results_*.csv")) +
        glob.glob(str(SCRIPTS_DIR / "fill_in_quality_results_*.csv"))
    )
    if not quality_files:
        raise FileNotFoundError("No synth_quality_results_*.csv or fill_in_quality_results_*.csv found.")

    print(f"Loading {len(quality_files)} quality result file(s):")
    frames = []
    for f in quality_files:
        df = pd.read_csv(f)
        print(f"  {Path(f).name}: {len(df)} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(combined)} rows")

    # Deduplicate — later files (higher timestamp) win
    combined = combined.drop_duplicates(subset=KEY_COLS, keep="last")
    print(f"After dedup: {len(combined)} unique rows")

    # ── Load and stack all wasserstein CSVs ────────────────────────────────────
    wd_files = sorted(glob.glob(str(SCRIPTS_DIR / "wasserstein_ohe_*.csv")))
    if wd_files:
        print(f"\nLoading {len(wd_files)} wasserstein_ohe file(s):")
        wd_frames = []
        for f in wd_files:
            df = pd.read_csv(f)
            print(f"  {Path(f).name}: {len(df)} rows")
            wd_frames.append(df)
        wd = pd.concat(wd_frames, ignore_index=True)
        wd = wd.drop_duplicates(subset=KEY_COLS, keep="last")
        print(f"Wasserstein rows after dedup: {len(wd)}")

        combined = combined.merge(wd[KEY_COLS + ["wasserstein_ohe"]],
                                  on=KEY_COLS, how="left")
        n_matched = combined["wasserstein_ohe"].notna().sum()
        print(f"Wasserstein joined: {n_matched}/{len(combined)} rows have a value")
    else:
        print("\nNo wasserstein_ohe_*.csv found — column will be absent.")

    combined.to_csv(out_path, index=False)
    print(f"\nMerged CSV written to: {out_path}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
