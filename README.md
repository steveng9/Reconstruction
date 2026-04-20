# Reconstruction Attack Experiment Suite

Survey-style framework for **reconstruction attacks** (attribute inference on synthetic tabular data) and **membership inference attacks (MIA)**. Given a synthetic dataset and partial knowledge of a target record (quasi-identifiers), attacks attempt to reconstruct unknown attribute values or determine membership.

## Quick Start

```bash
conda activate recon_

# Run reconstruction experiment
python master_experiment_script.py --n_runs 1 --on_server T

# Run MIA experiment
python master_experiment_script.py --mode mia --n_runs 1 --on_server T
```

## Generating Synthetic Data

Data generation is a two-step process managed by `sdg/generate_synth.py`. Set `DATASET`, `SAMPLE_SIZE`, and other options at the top of the script.

```bash
conda activate sdg

python sdg/generate_synth.py sample   # Step 1: create training samples
python sdg/generate_synth.py sdg      # Step 2: generate synthetic data (parallelized)
python sdg/generate_synth.py count    # Count generated synth.csv files
```

### Key Configuration Options

| Variable | Default | Description |
|---|---|---|
| `DATASET` | `"nist_sbo"` | Dataset name (must have entry in `_SDG_JOBS_BY_DATASET`) |
| `SAMPLE_SIZE` | `1000` | Rows per training sample |
| `NUM_SAMPLES` | `5` | Number of training samples to create |
| `DISJOINT` | `True` | Whether samples must be non-overlapping |
| `SAMPLES_TO_GENERATE` | `range(5)` | Which sample indices to run SDG for |
| `MAX_WORKERS` | `8` | Max parallel SDG jobs |

### Disjoint vs. Non-Disjoint Sampling

**`DISJOINT=True`** (default): Samples are non-overlapping slices of the shuffled dataset. Requires `NUM_SAMPLES × SAMPLE_SIZE ≤ total rows`. Safe to use as holdout sets in memorization tests.

**`DISJOINT=False`**: Each sample is drawn independently — samples may share rows. Use when the dataset isn't large enough for fully disjoint splits, or when you don't need memorization tests. A `NO_HOLDOUT` marker file is written into each sample directory; `get_data.py` will refuse to load these as holdout sets (both the train dir and holdout dir are checked).

### Per-Dataset SDG Job Configs

SDG jobs are defined in `_SDG_JOBS_BY_DATASET` — a dict keyed by dataset name. Setting `DATASET` at the top automatically selects the right job list with all dataset-specific parameters pre-configured. No more commenting/uncommenting.

To add a new dataset, add an entry to `_SDG_JOBS_BY_DATASET`. Missing entries raise a clear `ValueError`.

#### Dataset-specific notes

- **california**: No MST/AIM (all-continuous dataset, binning loses fidelity). No CellSuppression (no categorical QIs).
- **nist_sbo**: No AIM (130 cols, combinatorially infeasible). MST jobs use `bin_continuous_as_ordinal=True` (see below). TabDDPM uses wider architecture (`d_layers=[1024,2048,2048,2048,1024]`, `iterations=300000`) for the high-dimensional one-hot encoded input.

## SDG Methods

| Method | Type | Notes |
|---|---|---|
| MST | DP (marginal) | CPU. `epsilon` param. On wide datasets with skewed continuous cols, use `bin_continuous_as_ordinal=True` |
| AIM | DP (marginal) | CPU. Infeasible for 50+ columns |
| TVAE | Deep generative | GPU |
| CTGAN | Deep generative | GPU |
| ARF | Deep generative | GPU |
| TabDDPM | Diffusion | GPU. Supports `d_layers`, `iterations`, `num_timesteps` |
| Synthpop | R-based | CPU |
| RankSwap | De-identification | CPU. `swap_features` = continuous cols only |
| CellSuppression | De-identification | CPU. `key_vars` = categorical QI cols, must be non-empty |

### MST `bin_continuous_as_ordinal`

SmartNoise's private bound estimation (`approx_bounds`) fails when `preprocessor_eps` is too small for the data's range (e.g., `epsilon=0.1` → `preprocessor_eps=0.03` on financial columns spanning 0–140k). Setting `bin_continuous_as_ordinal=True` pre-bins continuous columns using the actual data range before fitting, bypassing `BinTransformer` entirely. MST bins continuous data internally anyway — this is equivalent and gives the full epsilon budget to synthesis.

## Evaluating Synthetic Data Quality

```bash
conda activate recon_

python sdg/evaluate_synth.py              # all datasets
python sdg/evaluate_synth.py nist_sbo    # one dataset
python sdg/evaluate_synth.py --verbose nist_sbo  # + per-column TVD/error tables
```

Metrics: `mean_tvd↓`, `mean_jsd↓`, `pairwise_tvd↓`, `mean_mean_err%↓`, `mean_std_err%↓`, `corr_diff↓`, `sdv_col_shapes↑`, `sdv_col_pairs↑`. A `~train_baseline` row (train vs. train round-robin) is shown when ≥2 samples exist.

## Data Layout

```
/home/golobs/data/reconstruction_data/
  {dataset}/
    meta.json
    full_data.csv
    size_{N}/
      sdg_log.txt
      sample_{XX}/
        train.csv
        [NO_HOLDOUT]          ← present if sample is non-disjoint
        {Method_params}/
          synth.csv
          sdg.log
```

## Memorization / Holdout Tests

Memorization tests compare attack performance on training members vs. held-out non-members. The holdout is a separate disjoint sample directory.

`get_data.py` enforces safety: if **either** the train dir or the holdout dir contains a `NO_HOLDOUT` marker, loading raises a `ValueError`. This prevents accidentally using overlapping samples as holdout.

## Row-Level RA Scores

Enable per-record reconstruction accuracy output for downstream subgroup, outlier, or demographic analysis:

```yaml
row_level_analysis:
  enabled: true
  compute_outliers: true        # flag QI-space outliers via IsolationForest
  outlier_method: "isolation_forest"   # or "gower_knn"
  outlier_percentile: 90
```

When enabled, a CSV is saved to `{data_dir}/{sdg_dir}/row_scores/{attack}__{QI}__{split}_run{N}.csv` containing QI values, true and reconstructed feature values, per-cell scores (`RA_row_{feat}`), a rarity-weighted mean (`RA_row_mean`), and optional outlier flags. WandB receives p10/p50/p90 percentiles and a score histogram.

## MIA and RA-as-MIA

**Standalone MIA mode** (SynthDistance or NNDR):
```bash
python master_experiment_script.py --mode mia --n_runs 1 --on_server T
```
Configure via `mia_method` and `mia_params` in the YAML. Set `n_targets: ~` (null) to use the full train + holdout rather than sampling.

**MIA vs RA-as-MIA comparison** — tests whether reconstruction accuracy is itself a membership signal and compares it against SynthDistance and NNDR:
```bash
python experiment_scripts/compare_mia_ra.py
```
Configure `SDG_METHODS`, `RA_METHOD`, `N_TARGETS` (None = full set), and `HOLDOUT_DIR` at the top of the script. Output includes per-method AUC/Advantage/TPR@FPR0.1%, quadrant analysis (where methods agree/disagree on training records), Spearman correlation between signal scores, and outlier intersection analysis (do high-scoring records cluster among QI-space outliers?).

## partialDiffusion Attacks

Three attacks based on training a diffusion model on the synthetic data, then using it to impute hidden features conditioned on a target's known QI values. They vary along two independent axes:

|  | Standard training | QI-conditioned training |
|---|---|---|
| **TabDDPM sampling** | *(not used)* | `TabDDPM` |
| **RePaint sampling** | `RePaint` | `ConditionedRePaint` |

- **Training axis** — whether QI values replace the noisy features during the forward diffusion process, directly conditioning the model on QI structure.
- **Sampling axis** — whether reconstruction uses straight diffusion sampling (TabDDPM) or RePaint's back-and-forth resampling, which iteratively re-noises the known QI values and denoises, improving coherence between QI and imputed hidden features.

Each attack saves model artifacts to its own subdirectory of the sample folder (`partial_tabddpm_artifacts/`, `repaint_artifacts/`, `conditioned_repaint_artifacts/`), so all three can coexist and be run with `--retrain` independently.

Run with `experiment_scripts/run_partial_diffusion_adult.py`:
```bash
# First run: train diffusion models
python experiment_scripts/run_partial_diffusion_adult.py --retrain

# Subsequent runs: tune reconstruction params without retraining
python experiment_scripts/run_partial_diffusion_adult.py
```

## MarginalRF Attack

`MarginalRF` combines Random Forest posterior probabilities with belief propagation over a pairwise marginal graph learned from the synthetic data.  It is registered under `data_type: "categorical"`.

### Algorithm

1. **RF phase** — train one RF per hidden feature on synth (QI → hidden) and collect per-target log-posteriors.
2. **Structure learning** — build a graph over hidden features weighted by pairwise mutual information from synth.
3. **Pairwise PMI tables** — for each graph edge, compute log-PMI from the K nearest synth neighbours of each target (local mode) or from all of synth (global mode).
4. **Row-level BP** — run sum-product belief propagation to let features inform each other within a row.
5. **Column correction** — adjust per-row beliefs so the aggregate prediction across all N targets matches the synth marginal for each feature (see below).

Steps 4–5 can be iterated (`col_correction_iters > 1`) for additional refinement.

### Graph types

| `graph_type` | Structure | BP | Notes |
|---|---|---|---|
| `"mst"` *(default)* | Maximum spanning tree | Exact, one pass | Guaranteed cycle-free |
| `"complete"` | Fully connected | Loopy (approximate) | Captures all pairwise interactions |
| `"topk"` | Top-k MI edges | Loopy if cycles exist | Middle ground; `top_k_edges` sets budget |

### Column Marginal Correction

After row-level BP, the individual row predictions are adjusted so their aggregate matches the synth marginal, capturing population-level information that the row-only BP cannot see.

| `col_correction_mode` | Correction target | When to use |
|---|---|---|
| `"global"` *(default)* | Global synth marginal `T_j(v)`, shared across all targets | Default; works well when targets are a representative sample |
| `"knn"` | Per-target local marginal from K nearest synth neighbours | Useful when QI is predictive and the local distribution differs substantially from the global |

`col_correction_alpha` (default `0.5`) controls the blend strength: `0.0` disables correction entirely, `1.0` fully enforces the target marginal.

`col_correction_iters` (default `1`) controls how many times the row-BP + column-correction cycle is repeated.  Higher values refine the beliefs but multiply compute cost proportionally.

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `num_estimators` | 25 | RF trees |
| `max_depth` | 25 | RF max depth |
| `max_pair_cardinality` | 50 | Features with more unique values skip the pairwise BP correction |
| `knn_k` | 100 | Synth neighbours for local PMI; `null` → global PMI |
| `graph_type` | `"mst"` | Graph structure (see above) |
| `lbp_max_iter` | 20 | Max loopy BP iterations (ignored for `"mst"`) |
| `lbp_damping` | 0.5 | Loopy BP step-size |
| `col_correction_alpha` | 0.5 | Column correction strength |
| `col_correction_mode` | `"global"` | `"global"` or `"knn"` |
| `col_correction_iters` | 1 | Outer (row-BP + correction) iterations |

## Architecture Overview

See `CLAUDE.md` for full architecture details (attack registry, config structure, scoring, external dependencies).
