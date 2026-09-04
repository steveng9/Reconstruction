# Historical / one-off scripts

These scripts were written for a specific run at a specific moment — filling a
gap in a sweep, repairing a data-integrity problem, archiving superseded rows,
re-running one dataset after a fix. They are kept because they document how the
results in `results.db` came to be, which matters for anyone auditing the
provenance of the paper's numbers.

**None of them is needed to build, run, or reproduce anything.** The scripts that
back the paper's experiments all live one directory up and are indexed in the
main `README.md` under "Reproducing Paper Results".

Most of these assume state that no longer exists (a partially-filled sweep, a
pre-repair database). Treat them as a record, not as tooling.

| Script | What it was for |
|---|---|
| `archive_superseded_runs.py` | Move superseded runs into `runs_superseded` after a re-run. |
| `supersede_floatbinned_runs.py` | Mark the runs invalidated by the float-binned encoding bug. |
| `fix_new_dp_sweep_csv.py` | Repair malformed rows in one DP-sweep output CSV. |
| `ingest_california_memorization.py` | One-time ingest of the California memorization results. |
| `merge_quality_results.py` | Merge two partial synthetic-quality result CSVs. |
| `run_fill_in_sweep.py`, `run_fill_in_marginalrf.py`, `run_marginalrf_combos_fill.py`, `run_new_attacks_fill_in_sweep.py` | Fill specific missing cells in the main grid. |
| `run_camera_fill_20260822.sh` | Fill the remaining camera-ready table cells. |
| `run_disparity_full_rerun_20260821.sh`, `run_disparity_mst_rerun_20260821.sh` | Re-run the disparate-impact analysis on post-repair synthetic data. |
| `run_mia_rerun_20260821.sh` | Re-run the MIA comparison on post-repair synthetic data. |
