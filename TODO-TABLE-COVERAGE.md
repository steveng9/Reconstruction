# TODO: regenerate every paper table and figure from committed data

**Status:** 9 of the manuscript's 33 labelled tables and figures regenerate
today (plus one extra figure that has no label). This file tracks the other 24.

**Why it matters.** The PoPETs "Reproduced" badge asks for a clear mapping from
claims to experiments to results, with minimal manual effort. We meet that for
the five main claims — every table carrying one regenerates exactly from
`results.db` via `experiment_scripts/regen_camera_tables.py`. The remaining
objects are supporting material, and for most of them the *data* is already
committed; only the generator is missing. Finishing this turns "the numbers are
in a database" into "every number in the paper rebuilds with one command."

**Target.** `python experiment_scripts/regen_camera_tables.py` emits a `.tex`
for every table and figure in the paper, each byte-comparable against
`expected_output/tables/`.

---

## Done — regenerates from committed data (9 labelled + 1)

| Paper object | Output | Source |
|---|---|---|
| `tab:ra_mean_adult` | `table1_ra_mean_adult.tex` | `results.db` |
| `tab:quality_overview` | `table2_quality_overview.tex` | `quality_results_merged.csv` |
| `tab:eps_sweep` | `table4_eps_sweep_compact.tex` | `results.db` |
| `tab:eps_full` | `table4_eps_sweep_full.tex` | `results.db` |
| `tab:eps_stats` | `STATS_eps_curve.md` | `results.db` |
| `tab:mia_comparison` | `table6_mia_comparison.tex` | `mia_comparison_results.csv` |
| `tab:memorization_and_ds_risk` | `table7_memorization.tex` | `linear_sweep_adult_1000_*.csv`, `ds_risk_scores.csv` |
| `tab:disparate_impact` | `table9_disparate_impact.tex` | `per_attack_disparity_postrepair.csv` |
| `fig:eps_curves` | `fig_eps_curves.pdf` | `results.db` |
| (per-attack ε curves) | `fig_eps_curves_perattack.pdf` | `results.db` |

---

## Tier A — data already committed, generator missing (11 objects, 8 items)

Each of these needs only a new function in `regen_camera_tables.py`. No
experiments to re-run, no files to add. This is the bulk of the remaining work
and should be done first.

| # | Paper object | Data is in | Notes |
|---|---|---|---|
| A1 | `tab:ra_mean_cdc` | `results.db` (cdc_diabetes, 1k) | Same shape as Table 1; `table1()` can be parameterised by dataset instead of duplicated. |
| A2 | `tab:cdc_100k` | `results.db` (cdc_diabetes, 100k) | Same generator as A1, different size filter. |
| A3 | `tab:ra_mean_nist_sbo` | `results.db` (nist_sbo, 1k) | Same generator as A1. |
| A4 | `tab:feature_eps_breakdown` | `results.db` `feature_scores` | `wandb_to_latex_epsilon_sweep.py` already builds this from the DB — fold it in rather than rewriting. |
| A5 | `tab:memorization_california` | `results.db` (california, train/nontraining splits) | Continuous dataset: the metric is NRMSE, not $R_{adv}$, so it needs its own formatter. |
| A6 | `tab:quality_arizona`, `tab:quality_cdc`, `tab:quality_sbo`, `tab:quality_california` | `quality_results_merged.csv` (all five datasets present) | Four tables, one parameterised generator; `synth_quality_to_latex.py` has the column logic. |
| A7 | `tab:disparity_perfeature` | `per_attack_disparity_postrepair_perfeat.csv` | Now committed. |
| A8 | `tab:disparity_perattack` | `per_attack_disparity_postrepair.csv` | Ratio of outlier to non-outlier row-level $R_{adv}$; the CSV has both columns. |

**Check as you go:** each new table must match the printed one. Where it does
not, say so in the function docstring — `table7()` and `table9()` already set
that precedent, and it is how the stale MIA cells were caught.

## Tier B — data committed, formatter is WandB-only (3 objects, 2 items)

The data is in `results.db`, but the only existing formatter needs our Weights &
Biases credentials, so a reviewer cannot run it. Port the formatting logic to
read the DB, the way `wandb_to_latex.py` already does.

| # | Paper object | Blocked on |
|---|---|---|
| B1 | `tab:qi_analysis_adult`, `tab:qi_analysis_cdcdiabetes` | `qi_analysis/wandb_to_latex_qi.py` is WandB-only. DB has all the QI variants. |
| B2 | `tab:linear_sweep_summary` | `linear_sweep_to_latex.py` is WandB-only. DB has 914 LinearReconstruction runs. |

## Tier C — needs a result file that is not committed yet (1)

| # | Paper object | What is needed |
|---|---|---|
| C1 | `fig:heatmap_ensemble` | Identify which of the 17 `ensembling_heatmap_results_*.csv` files produced the printed figure, commit it, and wire `plot_ensembling_heatmap.py` into the regen script. The files are small (≤300 KB); the only real work is establishing which run is authoritative. |

## Tier D — not derived from this repository (9)

These are correctly out of scope. Record them here so the count reconciles and
nobody goes looking for a generator.

| Paper object | Why |
|---|---|
| `tab:nist_results` | Transcribed from NIST's official published CRC scoreboard. |
| `tab:datasets` | Hand-written description of the five datasets. |
| `tab:qi_def_adult`, `tab:qi_def_cdc`, `tab:qi_def_arizona`, `tab:qi_def_california`, `tab:qi_def_sbo` | Hand-written QI membership matrices. Worth cross-checking against the `QIs` dicts in `get_data.py`, but they are documentation, not results. |
| `fig:threat_model`, `fig:taxonomy` | Hand-drawn diagrams. |

---

## Known discrepancy to resolve

`tab:mia_comparison` as printed mixes pre- and post-repair runs: the NIST
Arizona block is current, but parts of the Adult and CDC blocks come from the
May 2026 runs, which predate the float-binned encoding fix. Seven cells differ
from the post-repair numbers, all by ≤0.01 AUC, and no claim changes.
`expected_output/tables/table6_mia_comparison.tex` is the post-repair version;
swapping it in resolves this. Bolding shifts on the Adult MST columns as a
consequence, because SynthDistance now edges NNDR there.

## Suggested order

1. **A1–A3** — one parameterised generator covers three tables and is the
   template for the rest.
2. **A6** — four tables from one generator.
3. **A4, A5, A7, A8** — one each.
4. **B1, B2** — port off WandB.
5. **C1** — resolve provenance, then wire up.

After each step, add the new file to the `for t in ...` list in `test.sh` so it
is checked on every run.
