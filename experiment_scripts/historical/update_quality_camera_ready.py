#!/usr/bin/env python
"""
update_quality_camera_ready.py — bring quality_results_merged.csv to the values
printed in the published paper.

Two things changed between the first artifact release and the published paper:

  1. Wass. OHE is computed on 20 equal-width bins over the real column's range
     (`compute_wasserstein_ohe.py`), which is the definition the paper's caption
     gives. The earlier equal-depth version is kept there as `..._legacy`.
  2. The Adult 10k AIM rows use the pre-binned AIM releases, at eps = 1, 10 and
     1000, matching the preprocessing of the other five DP generators.

This script records how the shipped CSV was produced. It reads per-run metric
files that are not part of the repository (they are intermediate outputs of
`evaluate_synth_quality.py` and `compute_wasserstein_ohe.py` on the full
synthetic-data tree), so it is not something a fresh clone can re-run; the
merged CSV it writes is what ships.

Usage:
    python experiment_scripts/historical/update_quality_camera_ready.py
"""
from pathlib import Path

import pandas as pd

E = Path(__file__).resolve().parent.parent
KEY = ["dataset", "size_dir", "sample", "method"]

# Later files win where a key repeats.
WASS_FILES = ["wasserstein_ohe_20260911_ew20.csv", "wasserstein_ohe_20260912_cdc.csv",
              "wasserstein_ohe_20260912_adult.csv",
              "wasserstein_ohe_20260912_adult_aim_highbudget.csv"]
# Pre-binned AIM on Adult 10k. The eps=1 release is stored as `AIM_eps1_pb`.
AIM_FILES = [("synth_quality_aim_eps1_pb_20260912.csv", {"AIM_eps1_pb": "AIM_eps1"}),
             ("synth_quality_aim_eps10_prebinned_20260912.csv", {}),
             ("synth_quality_aim_adult_eps1000_20260912.csv", {})]


def main():
    q = pd.read_csv(E / "quality_results_merged.csv")
    columns = list(q.columns)

    aim = []
    for name, rename in AIM_FILES:
        d = pd.read_csv(E / name)
        d = d[d.method != "~train_baseline"]
        d["method"] = d.method.replace(rename)
        aim.append(d)
    aim = pd.concat(aim)
    replaced = set(map(tuple, aim[KEY].values))
    q = q[[tuple(r) not in replaced for r in q[KEY].values]]
    q = pd.concat([q, aim], ignore_index=True)

    w = pd.concat([pd.read_csv(E / f) for f in WASS_FILES]).drop_duplicates(KEY, keep="last")
    adult10k = (w.dataset == "adult") & (w.size_dir == "size_10000")
    pb = w[adult10k & (w.method == "AIM_eps1_pb")].assign(method="AIM_eps1")
    w = pd.concat([w[~(adult10k & (w.method == "AIM_eps1"))], pb]).drop_duplicates(KEY, keep="last")

    q = q.drop(columns="wasserstein_ohe").merge(w[KEY + ["wasserstein_ohe"]], on=KEY, how="left")
    # Real-vs-real rows have no Wass. OHE; every synthetic release must have one.
    missing = q[q.wasserstein_ohe.isna() & (q.method != "~train_baseline")]
    if len(missing):
        raise SystemExit(f"{len(missing)} synthetic releases have no Wass. OHE:\n{missing[KEY]}")
    q = q.sort_values(KEY)[columns]
    q.to_csv(E / "quality_results_merged.csv", index=False)
    print(f"wrote {len(q)} rows to quality_results_merged.csv")


if __name__ == "__main__":
    main()
