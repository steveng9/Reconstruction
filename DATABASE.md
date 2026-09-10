# `results.db` — the results database as a standalone dataset

`experiment_scripts/results.db` holds every scored reconstruction run behind the
paper: 49,126 runs and 413,318 per-feature scores, spanning 184 attack labels,
59 generator configurations and five datasets.

It is plain SQLite with no extensions, so it is usable on its own — as a
secondary dataset for studying reconstruction risk — without installing this
repository or running anything in it. This file is what you need to do that.

If instead you want to *regenerate the paper's tables* from it, you do not need
any of this: run `python reproduce.py`, and see
[`ARTIFACT-APPENDIX.md`](ARTIFACT-APPENDIX.md).

```bash
sqlite3 experiment_scripts/results.db
```

```python
import sqlite3, pandas as pd
con = sqlite3.connect("experiment_scripts/results.db")
runs = pd.read_sql("SELECT * FROM runs WHERE split = 'standard'", con)
```

To run the worked queries below and see their output:

```bash
python examples/query_results_db.py
```

---

## Schema

The schema is self-documenting — every column carries a comment explaining its
domain:

```bash
sqlite3 experiment_scripts/results.db .schema
```

Two tables matter for analysis.

**`runs`** — one row per scored experiment, uniquely keyed by
`(dataset, dataset_size, sample, qi, sdg_method, attack_label, split)`.

| Column | Meaning |
|---|---|
| `dataset` | `adult`, `cdc_diabetes`, `california`, `nist_arizona`, `nist_sbo` |
| `dataset_size` | size of the training sample the generator saw (1000, 10000, …) |
| `sample` | which disjoint training sample (0–9) |
| `qi` | which quasi-identifier split — see below |
| `sdg_method` | generator configuration, e.g. `MST_eps10`, `RankSwap`, `CellSuppression` |
| `attack_label` | attack, ablation variant, chain or ensemble — see below |
| `split` | `standard`, or `train` / `nontraining` for the memorization test |
| `ra_mean` | **the score**: mean rarity-weighted reconstruction advantage $R_{adv}$, 0–100 |
| `attack_params_json`, `sdg_params_json` | full parameters, as JSON |
| `source_file`, `wandb_*`, `ingested_at` | provenance |
| `confidence` | `certain` for every row in the shipped database |

**`feature_scores`** — one row per (run, attribute), joined on `run_id`. This is
where per-attribute breakdowns come from; `runs.ra_mean` is the mean over these.

`runs_superseded` and `feature_scores_superseded` retain 17,108 runs invalidated
by the 2026-08 float-binned-encoding repair. They are kept for audit and are
**not** part of any result. Every analytical query should read `runs`, not these.

---

## Two conventions that will bite you first

### 1. Attack labels are internal names, not always the paper's names

Six attacks were renamed late in writing. The database stores the label that was
in use when the run was scored, so a query using the paper's name can silently
return fewer rows than exist — or none at all.

| Paper name | Also stored as | Rows under the old label |
|---|---|---|
| CoBP-RA | `MarginalRF` | 1,221 (plus 6,726 already under `CoBP-RA`) |
| ARFFormer | `Attention` | 317 (**all** of them) |
| MultiHeadMLP | `JointMLP` | 110 (**all** of them) |
| CondMST | `PartialMST` | 304 (**all** of them) |
| CondDDPM | `TabDDPM` | 516 (**all** of them) |
| CondRePaint | `ConditionedRePaint` | 339 (**all** of them) |

So `WHERE attack_label = 'CoBP-RA'` misses 1,221 runs, and
`WHERE attack_label = 'ARFFormer'` returns nothing whatsoever. Normalise first:

```sql
SELECT CASE attack_label
         WHEN 'MarginalRF'         THEN 'CoBP-RA'
         WHEN 'Attention'          THEN 'ARFFormer'
         WHEN 'JointMLP'           THEN 'MultiHeadMLP'
         WHEN 'PartialMST'         THEN 'CondMST'
         WHEN 'TabDDPM'            THEN 'CondDDPM'
         WHEN 'ConditionedRePaint' THEN 'CondRePaint'
         ELSE attack_label
       END AS attack,
       ROUND(AVG(ra_mean), 1) AS mean_radv
FROM runs WHERE split = 'standard' GROUP BY attack ORDER BY mean_radv DESC;
```

The authoritative mapping, including the ablation variants, is `LABEL_TO_METHOD`
in [`experiment_scripts/rerun_queue.py`](experiment_scripts/rerun_queue.py), and
`examples/query_results_db.py` applies it for you. Note that `TabDDPM` is
ambiguous across columns: as an `attack_label` it means the CondDDPM *attack*, but
as an `sdg_method` it means the TabDDPM *generator*. Both are correct; they are
different things in different columns.

### 2. `attack_label` also holds variants, chains and ensembles

The 184 labels are not 184 distinct attacks. They include:

- **ablation variants**, suffixed with an underscore — `MarginalRF_graphQI_entropyBP`,
  `MarginalRF_mst_local_100`;
- **ensembles**, joined with `+` — `NaiveBayes+RandomForest+MLP`;
- **oracles**, which are upper bounds and not attacks — `OracleEnsemble`,
  `FeatSelectorOracle`. These will top any ranking you do not exclude them from.

To compare the attacks as the paper's main tables do, restrict to labels with no
`_`, no `+` and no `Oracle`.

---

## Worked queries

Each of these runs as-is, and all are in `examples/query_results_db.py`.

**Which generator leaks the most?** (Main Result 1 and 2: the generator matters
more than the attack, and de-identification methods are the most exposed.)

```sql
SELECT sdg_method, ROUND(AVG(ra_mean), 1) AS mean_radv, COUNT(*) AS n
FROM runs
WHERE dataset = 'adult' AND dataset_size = 10000 AND split = 'standard'
GROUP BY sdg_method HAVING n >= 20
ORDER BY mean_radv DESC LIMIT 10;
```

**Which attributes of a person are recoverable?** — the `feature_scores` join.
Useful for disparate-impact analysis, and the reason the per-feature table exists.

```sql
SELECT f.feature, ROUND(AVG(f.ra_score), 1) AS mean_radv, COUNT(*) AS n
FROM feature_scores f JOIN runs r ON r.run_id = f.run_id
WHERE r.dataset = 'adult' AND r.dataset_size = 10000
  AND r.qi = 'QI1' AND r.split = 'standard'
GROUP BY f.feature ORDER BY mean_radv DESC;
```

On Adult this separates cleanly: `income` (63.5), `education-num` (54.4) and
`relationship` (51.1) are highly recoverable, while `capital-gain` (1.6) and
`fnlwgt` (0.8) are essentially not — high-cardinality near-continuous columns
resist reconstruction.

**Does more privacy budget mean more leakage?** (Main Result 4: risk rises with
ε up to about 10, then flattens.)

```sql
SELECT sdg_method, ROUND(AVG(ra_mean), 1) AS mean_radv
FROM runs
WHERE dataset = 'adult' AND dataset_size = 10000 AND qi = 'QI1'
  AND split = 'standard' AND sdg_method LIKE 'MST_eps%'
GROUP BY sdg_method
ORDER BY CAST(REPLACE(sdg_method, 'MST_eps', '') AS REAL);
```

**Is the attack memorizing, or learning the distribution?** (Main Result 5.)
The memorization test scores the same attack on records the generator saw
(`train`) and on held-out records (`nontraining`); a small gap means the
reconstruction is distributional.

```sql
SELECT attack_label,
       ROUND(AVG(CASE WHEN split = 'train'       THEN ra_mean END), 1) AS on_train,
       ROUND(AVG(CASE WHEN split = 'nontraining' THEN ra_mean END), 1) AS on_holdout
FROM runs
WHERE split IN ('train', 'nontraining') AND dataset = 'adult'
GROUP BY attack_label HAVING on_train IS NOT NULL AND on_holdout IS NOT NULL
ORDER BY on_train - on_holdout DESC;
```

---

## Caveats worth knowing before you draw conclusions

- **`split = 'unknown'`** marks 1,569 rows migrated from Weights & Biases logs
  that predate the split field. No table in the paper reads them; filter on
  `split = 'standard'` and they disappear.
- **Coverage is uneven.** The grid is not complete — some (dataset, generator,
  attack) cells were never run. Always select `COUNT(*)` alongside `AVG(ra_mean)`
  and check that the cells you are comparing rest on comparable numbers of runs.
- **A flat average over generators is not the paper's attack ranking.** The
  attacks sit within about five points of each other — that narrowness is Main
  Result 1 — so the ordering inside that band moves with the slice you pick.
  Regenerate Table 1 (Experiment 1) for the comparison the paper actually makes.
- **Averaging across QI splits mixes difficulties.** `QI1` and `QI_large` reveal
  different numbers of known attributes, so their scores are not comparable.
  Group by `qi`, or filter to one.
- **`ra_mean` is a mean over attributes**, so a dataset with many hard
  high-cardinality columns scores lower overall than one without. Compare within
  a dataset, not across datasets, unless that is the point of the comparison.
- **Three result families are deliberately outside this database** — membership
  inference, synthetic-data quality metrics, and the NIST CRC scoreboard. They
  are committed as CSVs; see the "What `results.db` does and does not contain"
  section of [`ARTIFACT-APPENDIX.md`](ARTIFACT-APPENDIX.md).

## Citation

If you use this database, please cite the paper — see
[`CITATION.cff`](CITATION.cff) or the BibTeX entry in
[`README.md`](README.md#citation).
