#!/usr/bin/env python
"""
One-off fix for experiment_scripts/new_dp_epsilon_sweep_*.csv: the original
_write_row() in run_new_dp_epsilon_sweep.py wrote a fresh DictWriter per row
using that row's own (varying) key set, so rows with different hidden-feature
counts (QI_large=5 feats vs QI_behavioral/QI1=9 feats) ended up with different
column counts under a single header taken from the first row. The first 12
columns are always positionally correct (fixed order in run_job's `result`
dict); only the trailing RA_<feat> columns need reassigning, using the known
hidden_features order per QI from get_data.py's minus_QIs dict.

Usage:
    python experiment_scripts/fix_new_dp_sweep_csv.py <bad.csv> <out_clean.csv>
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, "/home/golobs/Reconstruction")
from get_data import minus_QIs

BASE_FIELDS = ["dataset", "size", "sample", "sdg", "epsilon", "attack", "label",
               "qi", "ra_mean_train", "ra_mean_nontraining", "ra_mean_delta", "error"]


def main():
    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rows_out = []
    all_feat_cols = set()

    with open(in_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for raw in reader:
            base = dict(zip(BASE_FIELDS, raw[:12]))
            extra = raw[12:]
            qi = base["qi"]
            hidden = minus_QIs["adult"][qi]
            assert len(hidden) == len(extra), f"mismatch for qi={qi}: {len(hidden)} feats vs {len(extra)} values"
            feat_scores = {f"RA_{feat}": val for feat, val in zip(hidden, extra)}
            all_feat_cols.update(feat_scores.keys())
            rows_out.append({**base, **feat_scores})

    fieldnames = BASE_FIELDS + sorted(all_feat_cols)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows, {len(fieldnames)} columns -> {out_path}")


if __name__ == "__main__":
    main()
