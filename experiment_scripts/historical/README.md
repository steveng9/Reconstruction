# Historical / one-off scripts

These scripts were written for a specific run at a specific moment — filling a
gap in a sweep, repairing a data-integrity problem, archiving superseded rows,
re-running one dataset after a fix. They are kept because they document how the
results in `results.db` came to be, which matters for anyone auditing the
provenance of the paper's numbers.

**None of them is needed to build, run, or reproduce anything.** The scripts that
back the paper's experiments all live one directory up and are indexed in the
directory above and are indexed in [`../README.md`](../README.md).

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
| `migrate_to_db.py` | The 2026 WandB/CSV → `results.db` migration (Phase 4). Run once; the DB is now the source of truth. |
| `audit_and_verify.py` | Phases 1–2 of that migration: audit the result CSVs, then verify them against WandB. |
| `fix_binned_synth_encoding.py` | The float-binned encoding repair. Rewrote 327 `synth.csv` files whose binned columns had been written as floats. |
| `fill_in_quality.py` | Compute quality metrics for the synth files that the main evaluation had missed. |
| `regen_mst_adult10k.py`, `regen_mst_missing_eps.py` | Regenerate MST synthetic data for Adult 10k after the encoding fix. |
| `run_adult10k_mst_eps01_attacks.py` | Re-run attacks against the regenerated MST ε=0.1 data. |
| `run_marginalrf_qi_graph_eps01.py` | One CoBP-RA (QI-graph) variant at ε=0.1, to fill a single missing cell. |
| `run_mst_regen_pipeline.sh`, `run_mst_eps_fill_pipeline.sh` | The end-to-end drivers for the two MST regeneration passes above. |
| `run_private_gsd_adult10k_eps1000_only.sh` | Re-run only the PrivateGSD Adult-10k ε=1000 point after it failed to finish. |
