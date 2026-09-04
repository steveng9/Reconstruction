#!/usr/bin/env python
import sys as _sys, pathlib as _pathlib
for _anc in _pathlib.Path(__file__).resolve().parents:
    if (_anc / "paths.py").exists():
        _sys.path.insert(0, str(_anc))
        break
from paths import DATA_ROOT

"""
fix_binned_synth_encoding.py

Repair synth.csv files whose integer-valued columns were written as float bin
midpoints (e.g. age = 22.475 instead of 22).

ROOT CAUSE
----------
sdg/smartnoise_methods.py inverse-transformed pre-binned continuous columns to
bin *midpoints* and then explicitly skipped those columns in the dtype-matching
cast (the `pre_binned` guard). So any generator run with
`bin_continuous_as_ordinal=True` emitted floats for columns that are integers in
the real data. Other generators (PrivSyn, MWEM-PGM) and the non-DP methods were
unaffected.

WHY IT MATTERS
--------------
Reconstruction scoring compares a predicted value against the true value by
equality. A float midpoint never equals a real integer, so every affected
*hidden* feature scores ~0 regardless of how well the generator actually
modelled it. Measured on Adult 10k / QI1 / RandomForest, `education-num` scored
0.0 on every float-encoded release and 31-60 on every integer-encoded one. That
artifact tracks the generator (MST / PrivBayes / PrivateGSD were float; PrivSyn /
MWEM-PGM were integer), so it masquerades as a privacy finding.

THE CORRECTION IS EXACT, NOT AN APPROXIMATION
---------------------------------------------
The stored value *is* `midpoints[bin_idx]` — a deterministic function of the bin
index the DP mechanism sampled. Applying np.rint recovers precisely what the
fixed generator would have written for that same draw. No DP noise is re-drawn
and no generator is re-run, so the corrected files are the same synthetic
records, decoded correctly. sdg/smartnoise_methods.py now applies the identical
np.rint at generation time; the two must stay in sync.

Only columns that are integral in the corresponding train.csv are touched, so
genuinely continuous data (California Housing) is left alone.

USAGE
-----
    python experiment_scripts/fix_binned_synth_encoding.py --dry-run
    python experiment_scripts/fix_binned_synth_encoding.py
    python experiment_scripts/fix_binned_synth_encoding.py --dataset adult --size 10000

Originals are copied to `synth.csv.floatbinned.bak` before rewriting (once —
a re-run will not clobber an existing backup). Idempotent: files already in
integer form are reported as clean and skipped.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(str(DATA_ROOT))
BACKUP_SUFFIX = ".floatbinned.bak"


def integral_columns(train_df):
    """Columns stored as integers in the real training data."""
    return {c for c in train_df.columns if train_df[c].dtype.kind in "iu"}


def needs_fix(series):
    """True if this float column carries non-integral values (bin midpoints).

    The comparison must be ABSOLUTE. np.allclose defaults to rtol=1e-5, a
    *relative* tolerance that scales with the magnitude of the values, so a
    column like fnlwgt (values ~1e5-1e6) tolerated an offset of up to ~14 and
    capital-gain (~1e5) up to ~0.97 -- larger than any bin-midpoint fraction
    can ever be. Both columns were therefore judged "already integral" and
    silently skipped, while small-magnitude columns in the same file (age,
    capital-loss) were correctly repaired. A bin midpoint is off by exactly
    half a bin width, which for an integral column is >= 1e-9 in absolute
    terms regardless of scale, so test that directly.
    """
    if series.dtype.kind != "f":
        return False
    vals = series.dropna().values
    if vals.size == 0:
        return False
    return bool(np.abs(vals - np.rint(vals)).max() > 1e-9)


def fix_one(synth_path, int_cols, dry_run):
    """Round midpoint columns back to integers. Returns list of fixed columns."""
    df = pd.read_csv(synth_path)
    targets = [c for c in df.columns if c in int_cols and needs_fix(df[c])]
    if not targets or dry_run:
        return targets

    backup = Path(str(synth_path) + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(synth_path, backup)

    for col in targets:
        df[col] = np.rint(df[col]).astype(np.int64)
    df.to_csv(synth_path, index=False)
    return targets


def iter_sample_dirs(dataset=None, size=None):
    """Yield (train.csv, [sdg dirs]) for every sample directory under DATA_ROOT."""
    for ds_dir in sorted(DATA_ROOT.iterdir()):
        if not ds_dir.is_dir() or (dataset and ds_dir.name != dataset):
            continue
        for size_dir in sorted(ds_dir.glob("size_*")):
            if size and size_dir.name != f"size_{size}":
                continue
            for sample_dir in sorted(size_dir.glob("sample_*")):
                train = sample_dir / "train.csv"
                if train.exists():
                    yield train, sample_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    ap.add_argument("--dataset", help="restrict to one dataset directory")
    ap.add_argument("--size", type=int, help="restrict to one sample size")
    args = ap.parse_args()

    n_files = n_fixed = 0
    by_generator = {}

    for train_path, sample_dir in iter_sample_dirs(args.dataset, args.size):
        int_cols = integral_columns(pd.read_csv(train_path, nrows=5000))

        for synth_path in sorted(sample_dir.glob("*/synth.csv")):
            n_files += 1
            gen = synth_path.parent.name
            try:
                fixed = fix_one(synth_path, int_cols, args.dry_run)
            except Exception as exc:
                print(f"  ERROR {synth_path}: {exc}", file=sys.stderr)
                continue
            if fixed:
                n_fixed += 1
                rel = synth_path.relative_to(DATA_ROOT)
                by_generator.setdefault(gen, []).append(str(rel))
                verb = "would fix" if args.dry_run else "fixed"
                print(f"  {verb}: {rel}  ({len(fixed)} cols: {', '.join(fixed[:4])}"
                      f"{'...' if len(fixed) > 4 else ''})")

    print(f"\n{'=' * 70}")
    print(f"scanned {n_files} synth.csv files; "
          f"{'would fix' if args.dry_run else 'fixed'} {n_fixed}")
    if by_generator:
        print("\naffected generators:")
        for gen in sorted(by_generator):
            print(f"  {gen:24} {len(by_generator[gen])} files")
    if args.dry_run:
        print("\n(dry run — nothing written)")
    else:
        print(f"\noriginals preserved as *{BACKUP_SUFFIX}")
        print("NOTE: every attack result computed against the old files is now "
              "stale.\n      Re-run the affected sweeps and mark the old DB rows "
              "superseded.")


if __name__ == "__main__":
    main()
